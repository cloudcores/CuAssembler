# -*- coding: utf-8 -*-

import re
import struct
from .common import *
from .CuSMVersion import CuSMVersion

# Pattern that matches an instruction string
p_InsPattern = re.compile(r'(@!?U?P\d|@!PT)?\s*(\w+.*)\s*;')

# Pattern that matches scoreboard sets, such as {1}, {4,2}
# Seems only appear after opcode DEPBAR
p_SBSet = re.compile(r'\{\s*(\d\s*,\s*)*\d\s*\}')

# NOTE: about constants translate dict
# 1) +/-QNAN is not recognized by python float(), use +/-NAN
#    +/-INF seems OK,
#    QNAN for FSEL may not work properly, needs special treatment
# 2) .reuse will be treated seperately for control codes, hence ignored here.
# 3) RZ may also appear in FADD/FMUL/FFMA.RZ ...
# 4) UPT is not found, may be just PT?
p_ConstTrDict = {r'(?<!\.)\bRZ\b' : 'R255', r'\bURZ\b' : 'UR63',
                r'\bPT\b' : 'P7', r'\bUPT\b' : 'UP7', r'\bQNAN\b' : 'NAN', r'\.reuse\b':''}

# Pattern for striping modifiers from an operand
# .*? for non-greedy match, needed for [R0.X4].A
p_ModifierPattern = re.compile(r'^([~\-\|!]*)(.*?)((\.\w+)*)$')

# Match Label+Index (including translated RZ/URZ/PT)
# SBSet is the score board set for DEPBAR, translated before parsing
p_IndexedPattern = re.compile(r'\b(R|UR|P|UP|B|SB|SBSET)(\d+)\b')

# Immediate floating point numbers, (NOTE: add 0f0000000 to skip conversion)
# Some instruction in 
p_FIType = re.compile(r'^(((-?\d+)(\.\d*)?((e|E)[-+]?\d+)?)(\.NEG)?|([+-]?INF)|([+-]NAN)|(0[fF][0-9a-f]+))$')

# Pattern for constant memory, some instructions have a mysterious space between two square brackets...
p_ConstMemType = re.compile(r'c\[(0x\w+)\] *\[([+-?\w\.]+)\]')

# Pattern for matching white spaces
p_WhiteSpace = re.compile(r'\s+')

# modifiers (1 char) that may appear before operands
c_OpPreModifierChar = {'!':'NOT', '-':'NEG', '|':'ABS', '~':'BITNOT'}

# Jump functions that may use the instruction address
# TODO: Some instruction already have neg sign before address, will it still work?
c_AddrFuncs = set(['BRA', 'BRX', 'BRXU', 'CALL', 'JMP',
                   'JMX', 'JMXU', 'RET', 'BSSY', 'BSYNC',
                   'SSY', 'CAL', 'PBK'])

# Functions that have position dependent modifiers, such as F2F.F16.F32 != F2F.F32.F16
c_PosDepFuncs = set(['I2I', 'F2F', 'IDP', 'HMMA', 'IMMA', 'XMAD', 'VADD'])

c_PosDepModis = set(['S8', 'S16', 'S32', 'S64', 'U8', 'U16', 'U32', 'U64', 'F16', 'F32', 'F64']) # TODO:

# I2F/F2I/F2F has different OpCode for 32/64,
# but 32bit modifier may not be displayed
# FRND may not need this
c_FloatCvtOpcodes = set(['I2F', 'I2I', 'F2I', 'F2F', 'FRND'])

class CuInsParser():
    ''' CuInsParser will parse the instruction string to inskey, values, and modifiers.

        Which could be then assembled by CuInsAssembler.

        Since the parser will consume considerable amount of memory, the "parse" should be
        called with limited instances, which will update the members accordingly.

        We don't make the "parse" a static function, since we frequently need to check some
        internal variables of parsing results, especially during debugging.
    '''

    # predicate value is the first element in value vector
    PRED_VAL_IDX = 0
    
    #
    OPERAND_VAL_IDX = 1

    def __init__(self, arch='sm_75'):
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

        self.m_SMVersion = CuSMVersion(arch)
        
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
        
        self.m_InsString = s.strip()
        self.m_CTrString = self.__preprocess(self.m_InsString)
        r = p_InsPattern.match(self.m_CTrString)
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
        tokens = re.split(',', ins_main)  # Splitting operands
                                          # usually ',' will be sufficient to split the operands
                                          # ( space does not work for c[0x0] [0x0].F32 )
                                          # And user may add extra spaces.
                                          #
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
        self.m_InsModifier = ['0_' + m for m in op_tokens] # TODO: May be we can treat pos dep modifiers here?

        for iop, op in enumerate(tokens[1:]):
            if len(op)==0: # ?
                continue

            optype, opval, opmodi = self.__parseOperand(op)
            self.m_InsKey += '_' + optype
            self.m_InsVals.extend(opval)
            self.m_InsModifier.extend([('%d_'%(iop+1))+m for m in opmodi])

        self.__specialTreatment() #

        return self.m_InsKey, self.m_InsVals, self.m_InsModifier

    def __preprocess(self, s):
        ''' Translate pre-defined constants (RZ/URZ/PT/...) to known or indexed values.

            Translate scoreboard sets {4,2} to SBSet
        
        '''
            
        for cm in p_ConstTrDict:
            s = re.sub(cm, p_ConstTrDict[cm], s)
        
        res = p_SBSet.search(s)
        if res is not None:
            SB_valstr = self.__transScoreboardSet(res.group())
            s = p_SBSet.sub(SB_valstr, s)
        
        return s

    def __parseOperand(self, operand):
        '''Parse operand to (type, val, modi).

        Every operand should return with:
            type:str, val:list, modi:list'''

        #print('Parsing operand: ' + operand)
        
        # all spaces inside the operand part of instruction are insignificant
        # subn returns (result, num_of_replacements), thus the trailing [0]
        operand = p_WhiteSpace.subn('', operand)[0]

        # Every operand may have one or more modifiers
        op, modi = self.stripModifier(operand)

        if p_IndexedPattern.match(op) is not None:
            optype, opval, tmodi = self.__parseIndexedToken(op)
            opmodi = modi
            opmodi.extend(tmodi)
        elif op[0] == '[': # address
            optype, opval, opmodi = self.__parseAddress(op)
        elif op[0] == '{': # BarSet such as {3,4}, only for DEPBAR (deprecated? could set in control codes)
                           # DEPBAR may wait a certain number of counts for one scoreboard, 
            optype, opval, opmodi = self.__parseBarSet(op)
        elif op.startswith('c['):
            optype, opval, opmodi = self.__parseConstMemory(op)
            opmodi.extend(modi)
        elif op.startswith('0x'):
            optype = 'II'
            opval, opmodi = self.__parseIntImme(operand)
        elif p_FIType.match(operand) is not None:
            optype = 'FI'
            opval, opmodi = self.__parseFloatImme(operand)
        else: # label type, keep as is
            optype = operand
            opval = [1]
            opmodi = []

        return optype, opval, opmodi

    def __parseIndexedToken(self, s):
        '''Parse index token such as R0, UR1, P2, UP3, B4, SB5, ...

         (RZ, URZ, PT should be translated In advance)'''

        tmain, modi = self.stripModifier(s)
        r = p_IndexedPattern.search(tmain)
        t = r.groups()[0]
        v = [int(r.groups()[1])]
        return t, v, modi

    def __parsePred(self, s):
        ''' Parse predicates (@!?U?P[\dT]) to values. 
        
        '''

        if s is None or len(s)==0:
            return 7

        t, v, modi = self.__parseIndexedToken(s.lstrip('@'))
        if 'NOT' in modi:
            return v[0] + 8
        else:
            return v[0]

    def __parseFloatImme(self, s):
        ''' Parse float point immediates to binary, according to the instruction precision.

            precision is the opcode precision, currently D/F/H for double/single(float)/half.
            NOTE: currently, +/-QNAN will be always translated to a UNIQUE binary,
              but sometimes nan could represent a set of values.
              But since it's not showed in the assembly string, there's no way to recover this value.

        '''

        p = self.m_InsOp[0]

        if p in set(['H', 'F', 'D']):
            prec = p
        elif self.m_InsOp=="MUFU": # It's rather wield that MUFU will have an imme input, any side effect?
            if '64' in self.m_InsOpFull:
                prec = 'D'
            else:
                prec = 'F'
        else:
            self.dumpInfo()
            raise ValueError('Unknown float precision (%s)!' % self.m_InsOp)
        
        if self.m_InsOp.endswith('32I'):
            nbits = 32
        else:
            nbits = -1

        v, modi = self.m_SMVersion.convertFloatImme(s, prec, nbits)
        return [v], modi

    def __parseIntImme(self, s):
        ''' Parse interger immediates.

            Positive int immediates are always kept as is,
            but negtive ints may depend on the type.
            Currently we try to let the coefficient determined by the code, not predetermined.

            TODO(Done): 
                Some ALU instructions such as IADD3 in sm5x/6x, the sign bit will be moved to the modifier.
                If the sign bit is explicitly show (such as -0x1), it can be handled by 'NegIntImme'.
                But if it's implicitly defined (such as 0xfffff, 20bit used, but int imme has only 19bit),
                we need to handle it seperately.
        '''

        i = int(s, 16)

        if i>=0:
            return self.m_SMVersion.splitIntImmeModifier(self, i)
        else:
            return [i], ['NegIntImme']

    def __parseConstMemory(self, s):
        opmain, opmodi = self.stripModifier(s)

        r = p_ConstMemType.match(opmain)
        if r is None:
            raise ValueError("Invalid constant memory operand: %s" %s)

        opval = [int(r.groups()[0], 16)]

        atype, aval, amodi = self.__parseAddress(r.groups()[1])

        optype = 'c' + atype
        opval.extend(aval)
        opmodi.extend(amodi)

        return optype, opval, opmodi

    def __transScoreboardSet(self, s):
        ''' Translate scoreboard set such as {3,4} to int values.

            This is done during preprocessing, since the comma(',') will be used to split the operands.
        '''

        ss = s.strip('{}').split(',')
        v = 0
        for bs in ss: # ???
            v += 1<<(int(bs))

        return 'SBSET%d'%v

    def __parseAddress(self, s):
        ''' Parse operand type Address [R0.X8+UR4+-0x8]

            Zero immediate will be appended if not present.
            It's harmless if there is no such field, since the value will always be 0.

            TODO(Done): what for [R0.U32+UR4.U64] ?? Could in another order?
                  May need extra tag in modifiers?
        '''

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

                # The modifier is prefixed by type
                # Thus     [R0.U32+UR4.U64] => ['R.U32', 'UR.U64']
                # (If any) [R0.U64+UR4.U32] => ['R.U64', 'UR.U32']
                opmodi.extend([ (ttype+'.'+m) for m in tmodi])

        # Pad with zero immediate if not present
        # Harmless even if it does not support immediates
        if not optype.endswith('I'):
            optype += 'I'
            opval.append(0)

        return optype, opval, opmodi

    def __specialTreatment(self):
        ''' Special treatments after parsing.

            Handle exceptions that cannot processed with current approach.

            TODO: Use dict mapping to subroutines, rather than if/else
                How??? F2F may need several special treatments...
        '''

        if self.m_InsOp == 'PLOP3': # immLut for PLOP3 is encoded with seperating 5+3 bits
                                      # e.g.: 0x2a = 0b00101010 => 00101 xxxxx 010
                                      # LOP3 seems fine
            v = self.m_InsVals[-2]
            self.m_InsVals[-2] = (v&7) + ((v&0xf8)<<5)

        elif self.m_InsOp in c_FloatCvtOpcodes:
            if '64' in self.m_InsOpFull:
                self.m_InsModifier.append('0_CVT64')

        elif self.m_InsOp in c_AddrFuncs: # Functions that use address of current instruction
            # CHECK: what if the address is not the last operand?
            if self.m_InsKey.endswith('_II'):
                if 'ABS' not in self.m_InsOpFull: # CHECK: Other absolute address?
                    addr = self.m_InsVals[-1] - self.m_InsAddr - self.m_SMVersion.getInstructionLength()
                    if addr<0:
                        self.m_InsModifier.append('0_NegAddrOffset')
                    
                    # The value length of same key should be kept the same
                    self.m_InsVals[-1] = addr

        if self.m_InsOp in c_PosDepFuncs:
            # the modifier of I2I/F2F is position dependent
            # eg: F2F.F32.F64 vs F2F.F64.F32
            # TODO: find all instructions with position dependent modifiers
            counter = 0
            for i,m in enumerate(self.m_InsModifier):
                if m.startswith('0_') and m[2:] in c_PosDepModis:
                    self.m_InsModifier[i] += '@%d'%counter
                    counter += 1

    def stripModifier(self, s):
        '''Split the token to three parts

        preModifier([~-|!]), opmain, postModifier(.FTZ, .X16, ...) '''

        r = p_ModifierPattern.match(s)  # split token to three parts

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
