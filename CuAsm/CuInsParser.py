# -*- coding: utf-8 -*-

import re
import struct
from .common import *

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
