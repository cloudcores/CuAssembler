# -*- coding: utf-8 -*-

import sympy
from sympy import Matrix # Needed by repr
from sympy.core.numbers import Rational
from io import StringIO, BytesIO
from .CuSMVersion import CuSMVersion
from .common import reprList
from CuAsm.CuAsmLogger import CuAsmLogger

class CuInsAssembler():
    '''CuInsAssembler is the assembler handles the values and weights of one type of instruction.'''

    def __init__(self, inskey, d=None, arch='sm_75'):
        ''' Initializer.

            inskey is mandatory, d is for initialization from saved repr.
        '''

        self.m_InsKey = inskey
        if d is not None:
            self.initFromDict(d)
        else:
            self.m_InsRepos = []
            self.m_InsModiSet = {}

            self.m_ValMatrix = None
            self.m_PSol = None
            self.m_PSolFac = None
            self.m_ValNullMat = []
            self.m_Rhs = None
            self.m_InsRecords = []
            self.m_Arch = CuSMVersion(arch)

    def initFromDict(self, d):
        self.m_InsKey     = d['InsKey']
        self.m_InsRepos   = d['InsRepos']
        self.m_InsModiSet = d['InsModiSet']

        self.m_ValMatrix  = d['ValMatrix']
        self.m_PSol       = d['PSol']
        self.m_PSolFac    = d['PSolFac']
        self.m_ValNullMat = d['ValNullMat']
        self.m_Rhs        = d['Rhs']
        self.m_InsRecords = d['InsRecords']
        self.m_Arch       = d['Arch']

    def expandModiSet(self, modi):
        ''' Push in new modifiers.
        
            NOTE: the order matters, since every modi has its own value.
        '''

        updated = False
        for m in modi:
            if m not in self.m_InsModiSet:
                self.m_InsModiSet[m] = len(self.m_InsModiSet)
                updated = True

        return updated

    def canAssemble(self, vals, modi):
        """ Check whether the input code can be assembled with current info.

        Args:
            vals ([int list]): value list in integers
            modi ([str list]): modifier list in strings

        Returns:
            (None , None) : input can be assembled
            (brief, info) : brief in ['NewModi', 'NewVals'], info gives the detailed info
        """

        if not all([m in self.m_InsModiSet for m in modi]):
            brief = 'NewModi'
            info = 'Unknown modifiers: (%s)' % (set(modi) - set(self.m_InsModiSet.keys()))
            return brief, info
        else:
            insval = vals.copy()
            insval.extend([1 if m in modi else 0 for m in self.m_InsModiSet])
            insvec = sympy.Matrix(insval)

            if self.m_ValNullMat is not None:
                insrhs = self.m_ValNullMat * insvec
                if not all([v==0 for v in insrhs]):
                    return 'NewVals', 'Insufficient basis, try gathering more instructions!'

        return None, None

    def push(self, vals, modi, code, ins_info):
        ''' Push in a new instruction.

            When its code can be assembled, verify the result,
            otherwise add new information to current assembler.
            @return:
                "NewInfo" for new information
                "Verified" for no new information, but the results is consistent
                False for inconsistent assembling result
        '''

        if not all([m in self.m_InsModiSet for m in modi]):
            # If new instruction contains unknown modifier,
            # it's never possible to be assembled by current assembler.
            CuAsmLogger.logProcedure('Pushing new modi (%s, %-20s): %s' % (self.m_Arch.formatCode(code), self.m_InsKey, ins_info))
            updated = self.expandModiSet(modi)
            self.m_InsRepos.append((vals, modi, code))
            self.buildMatrix()
            self.m_InsRecords.append(ins_info)
            return 'NewModi'
        else:
            # If the vals of new instruction lies in the null space of
            # current ValMatrix, it does not contain new information.
            insval = vals.copy()
            insval.extend([1 if m in modi else 0 for m in self.m_InsModiSet])
            insvec = sympy.Matrix(insval)

            if self.m_ValNullMat is None:
                doVerify = True
            else:
                insrhs = self.m_ValNullMat * insvec
                doVerify = all([v==0 for v in insrhs])

            if doVerify:
                # return 'Verified'
                inscode = self.m_PSol.dot(insvec) / self.m_PSolFac

                if inscode != code:
                    CuAsmLogger.logError("Error when verifying code!")
                    CuAsmLogger.logError("    InputCode : %s" % self.m_Arch.formatCode(code))
                    try:
                        CuAsmLogger.logError("    AsmCode   : %s" % self.m_Arch.formatCode(inscode))
                    except:
                        CuAsmLogger.logError("    AsmCode   : (%s)!" % str(inscode))

                    # print(self.__repr__())
                    # raise Exception("Inconsistent instruction code!")
                    return False
                else:
                    # print("Verified: 0x%032x" % code)
                    return 'Verified'

            else:
                CuAsmLogger.logProcedure('Pushing with new vals (%s, %-20s): %s' % (self.m_Arch.formatCode(code), self.m_InsKey, ins_info))
                self.m_InsRepos.append((vals, modi, code))
                self.m_InsRecords.append(ins_info)
                self.buildMatrix()
                return 'NewVals'

        # Never be here
        # return True

    def buildCode(self, vals, modi):
        '''Assemble with the input vals and modi.

        NOTE: This function didn't check the sufficiency of matrix.'''

        insval = vals.copy()
        insval.extend([1 if m in modi else 0 for m in self.m_InsModiSet])
        insvec = sympy.Matrix(insval)
        inscode = self.m_PSol.dot(insvec) / self.m_PSolFac

        return int(inscode)

    def buildMatrix(self):
        if len(self.m_InsRepos) == 0:
            return None, None

        M = []
        b = []
        zgen = range(len(self.m_InsModiSet))
        for vals, modis, code in self.m_InsRepos:
            l = [0 for x in zgen]
            for im in modis:
                l[self.m_InsModiSet[im]] = 1
            cval = vals.copy()
            cval.extend(l)
            M.append(cval)
            b.append(code)

        self.m_ValMatrix = sympy.Matrix(M)
        self.m_Rhs = sympy.Matrix(b)
        self.m_ValNullMat = self.getNullMatrix(self.m_ValMatrix)

        if self.m_ValNullMat is not None:
            M2 = self.m_ValMatrix.copy()
            b2 = self.m_Rhs.copy()
            for nn in range(self.m_ValNullMat.rows):
                M2 = M2.row_insert(0, self.m_ValNullMat.row(nn))
                b2 = b2.row_insert(0, sympy.Matrix([0]))
            self.m_PSol = M2.solve(b2)
        else:
            self.m_PSol = self.m_ValMatrix.solve(self.m_Rhs)

        self.m_PSol, self.m_PSolFac = self.getMatrixDenomLCM(self.m_PSol)
        return self.m_ValMatrix, self.m_Rhs

    def solve(self):
        ''' Try to solve every variable.

            This is possible only when ValNullMat is none.
        '''

        if self.m_ValNullMat is None:
            x = self.m_ValMatrix.solve(self.m_Rhs)
            print('Solution: ')
            for i,v in enumerate(x):
                print('%d : 0x%+033x' % (i,v))
            return x
        else:
            print('Not solvable!')
            return None

    def getNullMatrix(self, M):
        ''' Get the null space of current matrix M.

            And get the lcm for all fractional denominators.
            The null matrix is only for checking sufficiency of ValMatrix,
            thus it won't be affected by any non-zero common factor.
            Fractional seems much slower than integers.
        '''

        ns = M.nullspace()
        if len(ns)==0:
            return None
        else:
            nm = ns[0]
            for n in ns[1:]:
                nm = nm.row_join(n)

            # NullSpace won't be affected by a common factor.
            nmDenom, dm = self.getMatrixDenomLCM(nm.T)
            return nmDenom

    def getMatrixDenomLCM(self, M):
        ''' Get lcm of matrix denominator.

            In sympy, operation of fractionals seems much slower than integers.
            Thus we multiply a fraction matrix with the LCM of all denominators,
            then divide the result with the LCM.
        '''

        dm = 1
        for e in M:
            if isinstance(e, Rational):
                nom, denom = e.as_numer_denom()
                dm = sympy.lcm(denom, dm)
        return (M*dm, dm)

    def __repr__(self):
        ''' A string repr of current ins assembler.

            This will be used to dump it to text file and read back by setFromDict.
        '''
        sio = StringIO()

        sio.write('CuInsAssembler("", {"InsKey" : %s, \n' % repr(self.m_InsKey) )
        # sio.write('  "InsRepos" : %s, \n' % repr(self.m_InsRepos))
        sio.write('  "InsRepos" : ')
        reprList(sio, self.m_InsRepos)
        sio.write(', \n')

        sio.write('  "InsModiSet" : %s, \n' % repr(self.m_InsModiSet))
        sio.write('  "ValMatrix" : %s, \n' % repr(self.m_ValMatrix))
        sio.write('  "PSol" : %s, \n' % repr(self.m_PSol))
        sio.write('  "PSolFac" : %s, \n' % repr(self.m_PSolFac))
        sio.write('  "ValNullMat" : %s, \n' % repr(self.m_ValNullMat))
        #sio.write('  "InsRecords" : %s, \n' % repr(self.m_InsRecords))

        sio.write('  "InsRecords" : [')
        #reprList(sio, self.m_InsRecords)
        for addr, code, s in self.m_InsRecords:
            sio.write('(%#08x, %s, "%s"),\n'%(addr, self.m_Arch.formatCode(code), s))
        sio.write('], \n')

        sio.write('  "Rhs" : %s, \n' % repr(self.m_Rhs))
        sio.write('  "Arch" : %s })' % repr(self.m_Arch))

        return sio.getvalue()

    def __str__(self):

        return 'CuInsAssembler(%s)' % self.m_InsKey
