# -*- coding: utf-8 -*-

import re
import os
import time
import struct
import sympy
from sympy import Matrix # Needed by repr
from sympy.core.numbers import Rational
from io import StringIO, BytesIO

# NOTE: about constants translate dict
# 1) +/-QNAN is not recognized by python float(), use +/-NAN
#    +/-INF seems OK,
#    QNAN for FSEL may not work properly, needs special treatment
# 2) .reuse will be treated seperately for control codes, hence ignored here.
# 3) RZ may also appear in FADD/FMUL/FFMA.RZ ...
# 4) UPT is not found, may be just PT?
c_ConstTrDict = {r'(?<!\.)\bRZ\b' : 'R255', r'\bURZ\b' : 'UR63',
                 r'\bPT\b' : 'P7', r'\bQNAN\b' : 'NAN', r'\.reuse\b':''}

# Pattern that contains an instruction string (including address and code)
# NOTE: For maxwell/pascal, there may be braces "{}" for dual-issued instructions.
c_InsLinePattern = re.compile(r'^ *\/\*(\w+)\*\/ *(.*;) *\/\*(.*)\*\/')

# Pattern that matchs an instruction string
c_InsPattern = re.compile(r'(@!?U?P\d)? *(.*) *;')

# Pattern for branch type instructions
c_AddrInsPattern = re.compile(r'(@!?U?P\d)? *([\w.]+) +.*(0x[a-f0-9]+ *;)')

# Pattern for striping modifiers from an operand
# .*? for non-greedy match, needed for [R0.X4].A.B
c_ModifierPattern = re.compile(r'^([~\-\|!]*)(.*?)((\.\w+)*)$')

# Match Label+Index (including translated RZ/URZ/PT)
c_IndexedPattern = re.compile(r'\b(R|UR|P|UP|B|SB)(\d+)\b')

# Immediate floating point numbers, (NOTE: add 0f0000000 to skip conversion)
c_FIType = re.compile(r'^(((-?\d+)(\.\d*)?((e|E)[-+]?\d+)?)|([+-]?INF)|([+-]NAN)|(0[fF][0-9a-f]+))$')

# Pattern for constant memory, some instructions have a mysterious space between two square brackets...
c_ConstMemType = re.compile(r'c\[(0x\w+)\] *\[([+-?\w\.]+)\]')

# modifiers (1 char) that may appear before operands
c_OpPreModifierChar = {'!':'NOT', '-':'NEG', '|':'ABS', '~':'BITNOT'}

# Jump functions that may use the instruction address
c_AddrFuncs = {'BRA':0, 'BRX':1, 'BRXU':2, 'CALL':3, 'JMP':4,
              'JMX':5, 'JMXU':6, 'RET':7, 'BSSY':8, 'BSYNC':9}

# Functions that have position dependent modifiers, such as F2F.F16.F32 != F2F.F32.F16
c_PosDepFuncs = {'I2I':0, 'F2F':1, 'IDP':2, 'HMMA':3, 'IMMA':4}

# EIFMT/EIATTR for .nv.info in cubin, not complete yet
EIFMT = {1 : 'EIFMT_NVAL', 2 : 'EIFMT_BVAL', 3 : 'EIFMT_HVAL', 4 : 'EIFMT_SVAL'}
EIATTR = {0x0401 : 'EIATTR_CTAIDZ_USED',
          0x0504 : 'EIATTR_MAX_THREADS',
          0x0a04 : 'EIATTR_PARAM_CBANK',
          0x0f04 : 'EIATTR_EXTERNS',
          0x1004 : 'EIATTR_REQNTID',
          0x1104 : 'EIATTR_FRAME_SIZE',
          0x1204 : 'EIATTR_MIN_STACK_SIZE',
          0x1502 : 'EIATTR_BINDLESS_TEXTURE_BANK',
          0x1704 : 'EIATTR_KPARAM_INFO',
          0x1903 : 'EIATTR_CBANK_PARAM_SIZE',
          0x1b03 : 'EIATTR_MAXREG_COUNT',
          0x1c04 : 'EIATTR_EXIT_INSTR_OFFSETS',
          0x1d04 : 'EIATTR_S2RCTAID_INSTR_OFFSETS',
          0x1e04 : 'EIATTR_CRS_STACK_SIZE',
          0x2101 : 'EIATTR_EXPLICIT_CACHING',
          0x2304 : 'EIATTR_MAX_STACK_SIZE',
          0x2504 : 'EIATTR_LD_CACHEMOD_INSTR_OFFSETS',
          0x2804 : 'EIATTR_COOP_GROUP_INSTR_OFFSETS',
          0x2a01 : 'EIATTR_SW1850030_WAR',
          0x2b01 : 'EIATTR_WMMA_USED',
          0x2f04 : 'EIATTR_REGCOUNT',
          0x3001 : 'EIATTR_SW2393858_WAR',
          0x3104 : 'EIATTR_INT_WARP_WIDE_INSTR_OFFSETS'}

def intList2Str(vlist, s=None):
    if s:
        fmt = '0x%%0%dx' % s
    else:
        fmt = '0x%x'
    return '['+ (', '.join([fmt%v for v in vlist])) +']'

def binstr(v, l=128, sp=8):
    bv = bin(v)[2:]
    lb = len(bv)
    if lb<l:
        bv = '0' * (l-lb) + bv

    rs = ' '.join([bv[i:i+sp] for i in range(0,l,sp)])
    return rs

class CuInsAssembler():
    '''CuInsAssembler is the assembler handles the values and weights of one type of instruction.'''

    def __init__(self, inskey, d=None):
        '''Initializer.

        inskey is mandatory, d is for initialization from saved repr.'''

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

    def initFromDict(self, d):
        self.m_InsKey = d['InsKey']
        self.m_InsRepos = d['InsRepos']
        self.m_InsModiSet = d['InsModiSet']

        self.m_ValMatrix = d['ValMatrix']
        self.m_PSol = d['PSol']
        self.m_PSolFac = d['PSolFac']
        self.m_ValNullMat = d['ValNullMat']
        self.m_Rhs = d['Rhs']

    def expandModiSet(self, modi):
        ''' Push in new modifiers.'''

        updated = False
        for m in modi:
            if m not in self.m_InsModiSet:
                self.m_InsModiSet[m] = len(self.m_InsModiSet)
                updated = True

        return updated

    def push(self, vals, modi, code):
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
            print("Pushing with new modi (%s)..." % self.m_InsKey)
            updated = self.expandModiSet(modi)
            self.m_InsRepos.append((vals, modi, code))
            self.buildMatrix()
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
                    print("InputCode: 0x%032x" % code)
                    try:
                        print("AsmCode  : 0x%032x" % inscode)
                    except:
                        print("AsmCode  : (%s)!" % str(inscode))

                    # print(self.__repr__())
                    # raise Exception("Inconsistent instruction code!")
                    return False
                else:
                    # print("Verified: 0x%032x" % code)
                    return 'Verified'

            else:
                print("Pushing with new vals (%s)..." % self.m_InsKey)
                self.m_InsRepos.append((vals, modi, code))
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

        This is possible only when ValNullMat is none.'''

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
        '''Get the null space of current matrix M.

        And get the lcm for all fractional denominators.
        The null matrix is only for checking sufficiency of ValMatrix,
        thus it won't be affected by any non-zero common factor.
        Fractional seems much slower than integers.'''

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

        In sympy, operations of fractionals seem much slower than integers.
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

        sio.write('CuInsAssembler("", {"InsKey" : %s, ' % repr(self.m_InsKey) )
        sio.write('"InsRepos" : %s, ' % repr(self.m_InsRepos))
        sio.write('"InsModiSet" : %s, ' % repr(self.m_InsModiSet))

        sio.write('"ValMatrix" : %s, ' % repr(self.m_ValMatrix))
        sio.write('"PSol" : %s, ' % repr(self.m_PSol))
        sio.write('"PSolFac" : %s, ' % repr(self.m_PSolFac))
        sio.write('"ValNullMat" : %s, ' % repr(self.m_ValNullMat))
        sio.write('"Rhs" : %s }) ' % repr(self.m_Rhs))

        return sio.getvalue()

    def __str__(self):

        return 'CuInsAssembler(%s)' % self.m_InsKey

class CuInsFeeder():
    def __init__(self, fstream, insfilter=''):
        if isinstance(fstream, str):
            self.m_FName = fstream
            self.m_FStream = open(self.m_FName, 'r')
        else:
            self.m_FName = None
            self.m_FStream = fstream

        self.m_InsFilter = re.compile(insfilter)

    def __iter__(self):
        return self

    def __next__(self):
        res = self.fetchIns()
        if res is not None:
            return res
        else:
            raise StopIteration
            
    def fetchIns(self):
        while True:
            line = self.m_FStream.readline()
            if len(line) == 0:
                return None

            rf = self.m_InsFilter.search(line)
            r = c_InsLinePattern.search(line)

            if rf is not None and r is not None:
                addr = int(r.groups()[0].strip(), 16)
                code = int(r.groups()[2].strip(), 16)
                s = r.groups()[1].strip()

                l2 = self.m_FStream.readline()
                r2 = re.search(r'\/\*(.*)\*\/', l2)
                c2 = int(r2.groups()[0].strip(), 16)
                ctrlcodes = c2>>41
                c2 = c2 & 0x1ffffffffff # strip 23bit control codes
                code += (c2<<64)

                return (addr, code, s, ctrlcodes)

    def __del__(self):
        '''Close the stream if the handler is owned by this feeder.'''

        if self.m_FName is not None and not self.m_FStream.closed:
            self.m_FStream.close()

    def close(self):
        if not self.m_FStream.closed:
            self.m_FStream.close()
            return True
        else:
            return False

    def restart(self):
        if self.m_FStream.seekable:
            self.m_FStream.seek(0)
        else:
            raise Exception("This feeder cannot be restarted!")

    def tell(self):
        '''Report the progress.'''

        return self.m_FStream.tell()

class CuInsParser():
    '''CuInsParser will parse the instruction string to inskey, values, and modifiers.

    Which could be then assembled by CuInsAssembler.'''

    def __init__(self):
        self.m_InsAddr = 0             # ins address, needed by branch type of ins
        self.m_InsString = ''          # original asm string
        self.m_CTrString = ''          # constants translated asm string
        self.m_InsCode = None          # instruction code

        self.m_InsKey = ''             # key for current type of ins, eg: FFMA_R_R_R_R
        self.m_InsOp = ''              # function name, such as FFMA, MOV, ...
        self.m_InsOpFull = ''          # function name with modifiers
        self.m_InsPredVal = 0          # predicate value (0b****)
        self.m_InsPredStr = ''         # predicate string
        self.m_InsModifier = []        # modifier dict
        self.m_InsVals = []            # array of operand values (not include labels)

    def dumpInfo(self):
        print('#### CuInsParser @ 0x%016x ####' % id(self))
        print('InsString: ' + self.m_InsString)
        print('CTrString: ' + self.m_CTrString)
        print('InsAddr: 0x%x' % self.m_InsAddr)
        print('InsPred: %s (%s)' % (self.m_InsPredStr, bin(self.m_InsPredVal)) )
        print('InsCode: 0x%032x' % self.m_InsCode)
        print('InsKey: ' + self.m_InsKey)
        print('InsVals: ' + intList2Str(self.m_InsVals))
        print('InsModifier: ' + str(self.m_InsModifier))
        print('\n')

    def parse(self, s, addr=0, code=None):
        ''' Parse input string as instruction.'''

        self.m_InsString = s
        self.m_CTrString = self.__doConstTr(s)
        r = c_InsPattern.match(self.m_CTrString)
        if r is None:
            return None
            #raise ValueError("Unknown instruction: " + s)

        self.m_InsAddr = addr
        self.m_InsCode = code
        self.m_InsPredStr = r.groups()[0]

        # Currently pred is treated as known format operand
        # The value will be directly computed.
        self.m_InsPredVal = self.__parsePred(self.m_InsPredStr)

        ins_main = r.groups()[1]

        # TODO: use more robust tokenizer
        tokens = re.split(', ', ins_main)  # Splitting operands
                                          # usually ', ' will be sufficient to split the operands
                                          # ( ',' alone does not work for barset such as {3,4} )
                                          # ( space alone does not work for c[0x0] [0x0].F32 )
                                          # Exception: "RET.REL.NODEC R10 0x0 ;"
                                          # we will split it again, treat it as another separate operand
        ts = tokens[0].split(' ')
        ts.extend(tokens[1:])

        tokens = [t.strip() for t in ts]
        op_tokens = tokens[0].split('.') # Op and Op modifiers
        self.m_InsKey = op_tokens[0]
        self.m_InsOp = op_tokens[0]
        self.m_InsOpFull = tokens[0]

        self.m_InsVals = [self.m_InsPredVal]
        self.m_InsModifier = ['0_'+m for m in op_tokens] # TODO: May be we can treat pos dep modifiers here?

        for iop,op in enumerate(tokens[1:]):
            if len(op)==0: # ?
                continue

            optype, opval, opmodi = self.__parseOperand(op)
            self.m_InsKey += '_' + optype
            self.m_InsVals.extend(opval)
            self.m_InsModifier.extend([('%d_'%(iop+1))+m for m in opmodi])

        self.__specialTreatment() #

        return self.m_InsKey, self.m_InsVals, self.m_InsModifier

    def __doConstTr(self, s):
        '''Translate pre-defined constants (RZ/URZ/PT/...) to known or indexed values.'''

        for cm in c_ConstTrDict:
            s = re.sub(cm, c_ConstTrDict[cm], s)

        return s

    def __parseOperand(self, operand):
        '''Parse operand to (type, val, modi).

        Every operand should return with:
            type:str, val:list, modi:list'''

        #print('Parsing operand: ' + operand)

        # Every operand may have one or more modifiers
        op, modi = self.stripModifier(operand)

        if c_IndexedPattern.match(op) is not None:
            optype, opval, tmodi = self.__parseIndexedToken(op)
            opmodi = modi
        elif op[0] == '[': # address
            optype, opval, opmodi = self.__parseAddress(op)
        elif op[0] == '{': # BarSet such as {3,4}, only for DEPBAR (deprecated? could set in control codes)
            optype, opval, opmodi = self.__parseBarSet(op)
        elif op.startswith('c['):
            optype, opval, opmodi = self.__parseConstMemory(op)
            opmodi.extend(modi)
        elif op.startswith('0x'):
            optype = 'II'
            opval, opmodi = self.__parseIntImme(operand)
        elif c_FIType.match(operand) is not None:
            optype = 'FI'
            opval = [self.__parseFloatImme(operand)]
            opmodi = []
        else: # label type, keep as is
            optype = operand
            opval = [1]
            opmodi = []

        return optype, opval, opmodi

    def __parseIndexedToken(self, s):
        '''Parse index token such as R0, UR1, P2, UP3, B4, SB5, ...

         (RZ, URZ, PT should be translated In advance)'''

        tmain, modi = self.stripModifier(s)
        r = c_IndexedPattern.search(tmain)
        t = r.groups()[0]
        v = [int(r.groups()[1])]
        return t, v, modi

    def __parsePred(self, s):
        '''Parse predicates to values. '''

        if s is None or len(s)==0:
            return 7

        t, v, modi = self.__parseIndexedToken(s.lstrip('@'))
        if 'NOT' in modi:
            return v[0] + 8
        else:
            return v[0]

    def __parseFloatImme(self, s):
        '''Parse float point immediates to binary, according to the instruction precision.

        precision is the opcode precision, currently D/F/H for double/single(float)/half.
        NOTE: currently, +/-QNAN will be always translated to a UNIQUE binary,
              but sometimes nan could represent a set of values.
              But since it's not showed in the assembly string, there's no way to recover this value.
        '''

        # NEW feature: binary literal for float
        if s.startswith('0f') or s.startswith('0F'):
            v = int(s[2:], 16)
            return v

        f = float(s) # default to double

        if self.m_InsOp.startswith('F'):
            fbyte = struct.pack('f', f)
            v = struct.unpack('I', fbyte)[0]
        elif self.m_InsOp.startswith('D'):
            fbyte = struct.pack('d', f)
            v = struct.unpack('II', fbyte)[1] # for double immediates, only first 32bit is used.
                                              # TODO: handle endianness ???
        elif self.m_InsOp.startswith('H'):
            fbyte = struct.pack('e', f)
            v = struct.unpack('H', fbyte)[0]
        elif self.m_InsOp=="MUFU":
            # if '0_RCP64H' in self.m_InsModifier:
            #     fbyte = struct.pack('d', f)
            #     v = struct.unpack('II', fbyte)[1]
            # else:
            fbyte = struct.pack('f', f)
            v = struct.unpack('I', fbyte)[0]
        else:
            self.dumpInfo()
            raise ValueError('Unknown float precision (%s)!'% self.m_InsOp)

        return v

    def __parseIntImme(self, s):
        ''' Parse interger immediates.

        Positive int immediate are always kept as is,
        but negtive ints may depend on the type.
        When as arithmatic operand, it should be 32bit.
        When as address offset, it is 24bit.
        Currently we try to let the coefficient determined by the code, not predetermined.
        '''

        i = int(s, 16)

        if i>=0:
            return [i], []
        else:
            return [i], ['NegIntImme']

    def __parseConstMemory(self, s):
        opmain, opmodi = self.stripModifier(s)

        r = c_ConstMemType.match(opmain)
        if r is None:
            raise ValueError("Invalid constant memory operand: %s" %s)

        opval = [int(r.groups()[0], 16)]

        atype, aval, amodi = self.__parseAddress(r.groups()[1])

        optype = 'c' + atype
        opval.extend(aval)
        opmodi.extend(amodi)

        return optype, opval, opmodi

    def __parseBarSet(self, s):
        '''Parse operand type Bar, such as {3,4}.

        This instruction is deprecated, since now every instruction
        has control codes to set barriers.'''

        ss = s.strip('{}').split(',')
        v = 0
        for bs in ss:
            v += 1<<(int(bs))

        return 'BARSET', [v], []

    def __parseAddress(self, s):
        '''Parse operand type Address [R0.X8+UR4+-0x8]'''

        ss = s.strip('[]').split('+')

        optype = 'A'
        opval = []
        opmodi = []
        for ts in ss:
            if '0x' in ts:
                optype += 'I'
                i_opval, i_opmodi = self.__parseIntImme(ts)
                opval.extend(i_opval)
                opmodi.extend(i_opmodi)
            else:
                ttype, tval, tmodi = self.__parseIndexedToken(ts)
                optype += ttype
                opval.extend(tval)
                opmodi.extend(tmodi)

        return optype, opval, opmodi

    def __specialTreatment(self):
        ''' Special treatments after parsing.

        Handle exceptions that cannot processed with current approach.
        '''

        if self.m_InsOp == 'PLOP3': # immLut for PLOP3 is encoded with seperating 5+3 bits
                                      # e.g.: 0x2a = 0b00101010 => 00101 xxxxx 010
                                      # LOP3 seems fine
            v = self.m_InsVals[-2]
            self.m_InsVals[-2] = (v&7) + ((v&0xf8)<<5)
        elif self.m_InsOp in ['I2F','F2I','F2F']:
            # I2F/F2I/F2F has different OpCode for 32/64,
            # but 32bit modifier may not be displayed
            if '64' in self.m_InsOpFull:
                self.m_InsModifier.append('0_CVT64')
        elif self.m_InsOp in c_AddrFuncs: # Functions that use address of current instruction
            if self.m_InsKey.endswith('_II'):
                if self.m_InsOp in ['CALL', 'RET'] and 'ABS' in self.m_InsOpFull:
                    pass
                else:
                    # TODO: eliminate the hardcode for negative address offset
                    addr = self.m_InsVals[-1] - self.m_InsAddr - 0x10
                    if addr<0:
                        addr = 2**50 + addr
                    self.m_InsVals[-1] = addr

        if self.m_InsOp in c_PosDepFuncs:
            # the modifier of I2I/F2F is position dependent
            # eg: F2F.F32.F64 vs F2F.F64.F32
            # TODO: find all instructions with position dependent modifiers
            for i,m in enumerate(self.m_InsModifier):
                if m.startswith('0_'):
                    self.m_InsModifier[i] += '@%d'%i

    def stripModifier(self, s):
        '''Split the token to three parts

        preModifier([~-|!]), opmain, postModifier(.FTZ, .X16, ...) '''

        r = c_ModifierPattern.match(s)  # split token to three parts

        if r is None:
            raise ValueError("Unknown token %s" % s)
        else:
            pre = r.groups()[0]
            post = r.groups()[2]

            opmain = r.groups()[1]
            opmodi = []

            for c in pre:
                opmodi.append(c_OpPreModifierChar[c])

            for c in post.split('.'):
                if len(c)==0:
                    continue
                opmodi.append(c)

            return opmain, opmodi

    @staticmethod
    def transAddr2Label(s):
        pass

class CuInsAssemblerRepos():
    def __init__(self, InsAsmDict={}):
        self.reset(InsAsmDict)

    def reset(self, InsAsmDict={}):
        self.m_InsAsmDict = InsAsmDict

    def initFromFile(self, fname):
        with open(fname,'r') as fin:
            fconts = fin.read()
            asm_repos = eval(fconts)
            self.m_InsAsmDict = asm_repos.m_InsAsmDict

    def assemble(self, addr, s, ctrl):
        ins = CuInsParser()
        ins_key, ins_vals, ins_modi = ins.parse(s, addr, 0)
        if ins_key not in self.m_InsAsmDict:
            raise KeyError('Unknown ins_key(%s) in Repos!' % ins_key)

        insAsm = self.m_InsAsmDict[ins_key]
        code = insAsm.buildCode(ins_vals, ins_modi)

        if isinstance(ctrl, str):
            ctrlcodes = CuKernelAssembler.encodeCtrlCodes(ctrl)
        elif isinstance(ctrl, int):
            ctrlcodes = ctrl
        return code + (ctrlcodes << 105)

    def verify(self, feeder):
        res = True
        t0 = time.time()
        cnt = 0
        for addr, code, s, ctrlcodes in feeder:
            cnt += 1
            try:
                casm = self.assemble(addr, s, ctrlcodes)
                fcode = code + (ctrlcodes<<105)
                if fcode != casm:
                    print('Error when verifying :')
                    print('  ' + s)
                    print('  CodeOrg: %032x'%fcode)
                    print('  CodeAsm: %032x'%casm)
                    # raise Exception('Assembled code not match!')
            except:
                print('Error when assembling :')
                print('  ' + s)
                res = False
                
        t1 = time.time()
        
        if res:
            print("Verified %d ins in %8.3f secs." % (cnt, t1-t0))
            if (t0==t1):
                print("  ~Unknown ins per second." )
            else:
                print("  ~%8.2f ins per second." % (cnt/(t1-t0)))
        else:
            print("Verifying failed in %8.3f secs!!!" % (t1-t0))
            
        return res

    def update(self, feeder):
        ins = CuInsParser()
        t0 = time.time()
        cnt = 0
        for addr, code, s, ctrlcodes in feeder:
            cnt += 1
            ins_key, ins_vals, ins_modi = ins.parse(s, addr, code)

            if ins_key not in self.m_InsAsmDict:
                self.m_InsAsmDict[ins_key] = CuInsAssembler(ins_key)

            res = self.m_InsAsmDict[ins_key].push(ins_vals, ins_modi, code)

            if not res:
                print('  Str :' + s)
                print('  Addr:' + str(addr))
                raise Exception("Unmatched codes!")

        t1 = time.time()
        print("Updated %d ins in %8.3f secs." % (cnt, t1-t0))
        if (t0==t1):
            print("  ~Unknown ins per second." )
        else:
            print("  ~%8.2f ins per second." % (cnt/(t1-t0)))

    def save2file(self, fname):
        with open(fname,'w') as fout:
            fout.write(self.__repr__())

    def __repr__(self):
        sio = StringIO()

        sio.write('CuInsAssemblerRepos(')
        sio.write(repr(self.m_InsAsmDict))
        sio.write(')')

        return sio.getvalue()

    def __str__(self):
        return "CuInsAssemblerRepos(%d keys)" % len(self.m_InsAsmDict)

class CuKernelAssembler():
    def __init__(self, insasmrepos=None):
        if insasmrepos is None:
            self.m_InsAsmRepos = None
        elif isinstance(insasmrepos, str):
            self.initInsAsmRepos(insasmrepos)
        elif isinstance(insasmrepos, CuInsAssemblerRepos):
            self.m_InsAsmRepos = insasmrepos
        else:
            raise Exception("Unknown input for CuKernelAssembler!")

    def initInsAsmRepos(self, fname):
        with open(fname,'r') as fin:
            fconts = fin.read()
            self.m_InsAsmRepos = eval(fconts)

    def assembleCuAsm(self, cuasmcode):
        m_ins = re.compile(r'^\s*(L\w+\s*:)?\s*\[([R-]{4}:B[0-7-]{6}:R.:W.:.:S\d{2})\] +([^;]+;)')
        m_comment = re.compile(r'^\s*(#.*)?$') # comment or empty line

        if not isinstance(self.m_InsAsmRepos, CuInsAssemblerRepos):
            raise Exception("InsAsmRepos is not properly initialized!")

        labeldict = {}
        inslist = []

        counter = 0
        # first scan to gather instructions and labels
        for line in cuasmcode:
            # print('parsing '+line)
            if m_comment.match(line):
                continue
            else:
                res = m_ins.match(line)
                if res is None:
                    raise Exception("Unknown code line: %s" % line)

                label = res.groups()[0].strip(':')
                if label is not None:
                    if label in labeldict:
                        raise Exception("Redefined label in line:%s" % line)
                    labeldict[label] = counter

                inslist.append((counter, res.groups()[1], res.groups()[2]))
                counter += 1

        # second scan to encode the instructions
        # codes = []
        kcode = BytesIO()
        for idx, ctrlstr, ins in inslist:
            addr = idx * 0x10
            insstr = self.transLabel(ins, labeldict)
            code = self.m_InsAsmRepos.assemble(addr, insstr, ctrlstr)
            cbytes = struct.pack('QQ', code & ((1<<64)-1), code>>64)
            kcode.write(cbytes)

        return kcode.getvalue()

    #def __repr__(self):
    #    pass

    @staticmethod
    def transLabel(s, ldict):
        ''' Translate `Label` to the ins address.'''

        m = re.compile(r'`(L\w+)`')
        res = m.search(s)
        while res is not None:
            label = res.groups()[0]
            if label not in ldict:
                print(s)
                raise Exception('Unknown label %s' % label)

            addr = ldict[label] * 0x10

            s = s.replace(res.group(), '0x%x' % addr)

            res = m.search(s)

        return s

    @staticmethod
    def ParseNVInfo(binfo):
        #da = np.frombuffer(binfo, dtype='uint32')
        
        curr = 0

        attrlist = []
        
        bio = BytesIO(binfo)
        b = bio.read(4)
        while len(b)>0:
            v = struct.unpack('I', b)
            key = v & 0xffff
            t = v & 0xff
            l = v>>16

            curr += 1
            if t==1: # EIFMT_NVAL
                val = []
            elif t==2: # EIFMT_BVAL
                val = []
            elif t==3: # EIFMT_HVAL
                val = []
            elif t==4: # EIFMT_SVAL
                curr += l//4
            else:
                raise Exception("Unknwon EIFMT(%d) in nv.info!" % t)
            attrlist.append((key, val))
            
            b = bio.read(4)

    @staticmethod
    def decodeCtrlCodes(code):
        # c.f. : https://github.com/NervanaSystems/maxas/wiki/Control-Codes
        #      : https://arxiv.org/abs/1903.07486
        # reuse  waitbar  rbar  wbar  yield   stall
        #  0000   000000   000   000      0    0000
        #
        c_stall    = (code & 0x0000f) >> 0
        c_yield    = (code & 0x00010) >> 4
        c_writebar = (code & 0x000e0) >> 5  # write dependency barier
        c_readbar  = (code & 0x00700) >> 8  # read  dependency barier
        c_waitbar  = (code & 0x1f800) >> 11 # wait on dependency barier
        c_reuse    =  code >> 17

        s_yield = '-' if c_yield !=0 else 'Y'
        s_writebar = '-' if c_writebar == 7 else '%d'%c_writebar
        s_readbar = '-' if c_readbar == 7 else '%d'%c_readbar
        s_waitbar = ''.join(['-' if (c_waitbar & (2**i)) == 0 else ('%d'%i) for i in range(6)])
        s_stall = '%02d' % c_stall
        s_reuse = ''.join(['R' if (c_reuse&(2**i)) else '-' for i in range(4)])

        return '%s:B%s:R%s:W%s:%s:S%s' % (s_reuse, s_waitbar, s_readbar, s_writebar, s_yield, s_stall)

    @staticmethod
    def encodeCtrlCodes(s):
        s_reuse, s_waitbar, s_readbar, s_writebar, s_yield, s_stall = tuple(s.split(':'))

        reuse_tr = str.maketrans('R-','10')
        waitbar_tr = str.maketrans('012345-','1111110')

        c_reuse = int(s_reuse[::-1].translate(reuse_tr), 2)
        c_waitbar = int(s_waitbar[:0:-1].translate(waitbar_tr), 2)
        c_readbar = int(s_readbar[1].replace('-', '7'))
        c_writebar = int(s_writebar[1].replace('-','7'))
        c_yield = int(s_yield!='Y')
        c_stall = int(s_stall[1:])

        code = c_reuse<<17
        code += c_waitbar<<11
        code += c_readbar<<8
        code += c_writebar<<5
        code += c_yield<<4
        code += c_stall

        return code

if __name__ == '__main__':

    # kasm = CuKernelAssembler()
    # kasm.initInsAsmRepos('InsAsmRepos.txt')

    fname = 'G:\\Temp\\test.cuasm'
    with open(fname, 'r') as fin:
        kasm = CuKernelAssembler()
        kasm.initInsAsmRepos('InsAsmRepos.txt')

        codes = kasm.assembleCuAsm(fin)
        bio = BytesIO(codes)

        fin.seek(0)
        for line in fin:
            cb = bio.read(16)
            c0, c1 = struct.unpack('QQ', cb)
            print(line.rstrip())
            print('   0x%016x; 0x%016x' % (c0, c1))

