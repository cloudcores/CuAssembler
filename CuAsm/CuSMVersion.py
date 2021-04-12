# -*- coding: utf-8 -*-
from io import BytesIO
import struct

class CuSMVersion(object):
    ''' CuSMVersion will handle most of sm version related features, thus it's used everywhere.

        Note the same version will share the same instance, since there is no private member needed.

        TODO: Use a better form of version related attributes, rather than defined seperately.
              A class with default values?
    '''

    __InstanceRepos = {}

    SMVersionDict = {'SM50':50, 'SM_50':50, '50':50, 50:50,
                    'SM52':52, 'SM_52':52, '52':52, 52:52,
                    'SM53':53, 'SM_53':53, '53':53, 53:53,
                    'SM60':60, 'SM_60':60, '60':60, 60:60,
                    'SM61':61, 'SM_61':61, '61':61, 61:61,
                    'SM62':62, 'SM_62':62, '62':62, 62:62,
                    'SM70':70, 'SM_70':70, '70':70, 70:70,
                    'SM72':72, 'SM_72':72, '72':72, 72:72,
                    'SM75':75, 'SM_75':75, '75':75, 75:75,
                    'SM80':80, 'SM_80':80, '80':80, 80:80,
                    'SM86':86, 'SM_86':86, '86':86, 86:86
                    }
    SMCodeNameDict = { 50:'Maxwell', 52:'Maxwell', 53:'Maxwell',
                       60:'Pascal',  61:'Pascal',  62:'Pascal',
                       70:'Volta',   72:'Turing',  75:'Turing',
                       80:'Ampere',  86:'Ampere'}

    PadBytes_5x_6x  = bytes.fromhex('e00700fc00801f00 000f07000000b050 000f07000000b050 000f07000000b050')
    Pad_CCode_5x_6x = 0x7e0               # [----:B------:R-:W-:Y:S00]
    Pad_ICode_5x_6x = 0x50b0000000070f00  # NOP

    PadBytes_7x_8x  = bytes.fromhex('1879000000000000 0000000000c00f00')
    Pad_CCode_7x_8x = 0x7e0               # [----:B------:R-:W-:Y:S00]
    Pad_ICode_7x_8x = 0x7918              # NOP

    PredCode_5x_6x =  0xf0000
    PredCode_7x_8x =  0xf000

    RelMaps_7x_8x = {'32@hi' : 'R_CUDA_ABS32_HI_32',
                     '32@lo' : 'R_CUDA_ABS32_LO_32',
                     'target': 'R_CUDA_ABS47_34'}
    
    RelMaps_5x_6x = {'32@hi' : 'R_CUDA_ABS32_HI_20',
                     '32@lo' : 'R_CUDA_ABS32_LO_20',
                     'target': 'R_CUDA_ABS32_20'}

    # keep 20bits, but the sign bit is moved to neg modifier
    FloatImmeFormat_5x_6x = {'H':('e','H', 16, 16), 'F':('f','I', 32, 20), 'D':('d','Q', 64, 20)}
    FloatImmeFormat_7x_8x = {'H':('e','H', 16, 16), 'F':('f','I', 32, 32), 'D':('d','Q', 64, 32)}

    # EIATTR_AutoGen is the EIATTR set can be handled by the assembler automatically
    # NOTE: REGCOUNT/PARAM* will be handled seperately

    EIATTR_AutoGen_7x_8x = set(['EIATTR_CTAIDZ_USED', 
                             'EIATTR_WMMA_USED',
                             'EIATTR_EXIT_INSTR_OFFSETS'])

    EIATTR_AutoGen_5x_6x = set(['EIATTR_CTAIDZ_USED', 
                             'EIATTR_WMMA_USED',
                             'EIATTR_EXIT_INSTR_OFFSETS',
                             'EIATTR_S2RCTAID_INSTR_OFFSETS'])

    def __init__(self, version):
        self.__mVersion = CuSMVersion.parseVersionNumber(version)
    
        self.__mMajor = self.__mVersion // 10
        self.__mMinor = self.__mVersion % 10
        if self.__mMajor<=6:
            self.__mFloatImmeFormat = self.FloatImmeFormat_5x_6x
        else:
            self.__mFloatImmeFormat = self.FloatImmeFormat_7x_8x

    def __new__(cls, version, *args, **kwargs):
        ''' Create new instance if the version is not in repos.

            Otherwise return current corresponding instance.
        '''
        vnum = CuSMVersion.parseVersionNumber(version)
        if vnum not in CuSMVersion.__InstanceRepos:
            instance = super().__new__(cls)
            CuSMVersion.__InstanceRepos[vnum] = instance
        else:
            instance = CuSMVersion.__InstanceRepos[vnum]
        
        return instance

    def getMajor(self):
        return self.__mMajor
    
    def getMinor(self):
        return self.__mMinor

    def getVersionNumber(self):
        return self.__mVersion

    def getVersionString(self):
        return 'SM_%d'%self.__mVersion

    def getNOP(self):
        ''' Get NOP instruction code (no control codes).'''
        if self.__mMajor<=6:
            return self.Pad_ICode_5x_6x
        else:
            return self.Pad_ICode_7x_8x
    
    def getPadBytes(self):
        ''' Get padding bytes.

            NOTE: For sm_5x/6x, the padding byte length is 32B (1+3 group);
                  For sm_7x/8x, the padding byte length is 16B.
        '''

        if self.__mMajor <= 6:
            return CuSMVersion.PadBytes_5x_6x
        else:
            return CuSMVersion.PadBytes_7x_8x

    def getInstructionLength(self):
        ''' (At least) Since Kepler, SASS becomes a constant length ISA.

            5.x 6.x :  64bit =  8 bytes (1 control codes + 3 normal instructions)
            7.x 8.x : 128bit = 16 bytes
        '''
        if self.__mMajor<=6:
            return 8
        else:
            return 16

    def mergeControlCodes(self, ins_code_list, ctrl_code_list, padding_align=None):
        ''' Merge control codes with instructions.

            padding_align = None means no padding
                          = N bytes means padding to N bytes boundary (usually 32/128)
        '''
        if padding_align is None:
            padding_align = self.getTextSectionSizeUnit()

        if self.__mMajor <= 6:
            return CuSMVersion.mergeControlCodes_5x_6x(ins_code_list, ctrl_code_list, padding_align)
        else:
            return CuSMVersion.mergeControlCodes_7x_8x(ins_code_list, ctrl_code_list, padding_align)

    def splitControlCodes(self, codebytes):
        if self.__mMajor<=6:
            return CuSMVersion.splitControlCodes_5x_6x(codebytes)
        else:
            return CuSMVersion.splitControlCodes_7x_8x(codebytes)

    def getInsOffsetFromIndex(self, idx):
        ''' Get instruction offset according to the instruction index.
        '''
        if self.__mMajor<=6:
            return (idx//3 + 1)*8 + idx*8
        else:
            return idx * 16

    def getInsIndexFromOffset(self, offset):
        ''' Get Instruction index according to the instruction offset.

            For SM_5x, SM_6x, offset should be multiple of  8
            For SM_7x, SM_8x, offset should be multiple of 16
        '''

        if self.__mMajor<=6:
            ridx = offset>>3
            if (ridx & 0x3) == 0: # Input is the control codes offset
                return -1
            v = (ridx>>2)*3 + (ridx & 0x3) - 1
            return v
        else:
            return offset >> 4

    def getInsRelocationType(self, key):
        ''' Get Instruction relocation type from keys.

            Available keys: ["32@hi", "32@lo", "target"]
        '''
        if self.__mMajor<=6:
            return self.RelMaps_5x_6x[key]
        else:
            return self.RelMaps_7x_8x[key]
 
    def getTextSectionSizeUnit(self):
        ''' The text section should be padded to integer multiple of this unit.

            NOTE: This is different from the section align, which is applied to offset, not size.
        '''
        if self.__mMajor <= 6:
            return 64
        else:
            return 128

    def setRegCountInNVInfo(self, nvinfo, reg_count_dict):
        ''' Update NVInfo for regcount, only for SM_70 and above.

            reg_count_dict = {kernelname_symidx:regnum, ...}
            Return: flag for whether found and updated.
        '''
        if self.__mMajor<=6: # No this nvinfo for SM<=6x
            return True
        else:
            return nvinfo.setRegCount(reg_count_dict)

    def extractFloatImme(self, bs):
        ''' Not implemented yet. '''
        pass

    def convertFloatImme(self, fval, prec, nbits=-1):
        ''' Convert float immediate to value (and modifiers if needed).

            Input:
                fval : float in string
                prec : string, 'H':half / 'F':float / 'D':double
                nbits: int, how many bits to keep, -1 means default values of given precision, 
                       only for opcodes end with "32I" in sm5x/sm6x
            Return:
                value, [modi]

        '''

        fval = fval.lower().strip() #

        if self.__mMajor<=6:
            if fval.startswith('-'):
                val = fval[1:]
                modi = ['FINeg']
            elif fval.endswith('.neg'): # Only for maxwell/pascal ?
                val = fval[:-4]
                modi = ['ExplicitFINeg']
            else:
                val = fval
                modi = []
        else:
            val = fval
            modi = []

        if val.startswith('0f'):
            v = int(val[2:], 16)
            return v, modi
        else:
            fv = float(val)
            ifmt, ofmt, fullbits, keepbits = self.__mFloatImmeFormat[prec]
            fb = struct.pack(ifmt, fv)
            ival = struct.unpack(ofmt, fb)[0]

            trunc_bits = fullbits - max(nbits, keepbits)
            if trunc_bits>0:
                ival = ival >> trunc_bits
            
            return ival, modi

    def splitIntImmeModifier(self, ins_parser, int_val):
        if self.__mMajor<=6 and (not ins_parser.m_InsOp.endswith('32I')) and ((int_val & 0x80000) != 0):
            new_val = int_val - (int_val & 0x80000)
            modi = ['ImplicitNegIntImme']
            return [new_val], modi
        else:
            return [int_val], []

    def formatCode(self, code):
        if self.__mMajor<=6:
            return '0x%016x'%code
        else:
            return '0x%027x'%code

    def genPredCode(self, ins_info):
        ''' Generate instruction string with modified predicates.

            If the instruction already has predicates, return None.
        '''

        addr, code, s = ins_info
        if s.startswith('@'):
            # print(s)
            return None
        
        # CHECK: currently seems all uniform path opcode with uniform predicate starts with U
        #
        if s.startswith('U'):
            if s.startswith('UNDEF'): # UNDEF is reserved for un-disassembled instructions
                return None
            else:
                s2 = '@UP0 ' + s
        else:
            s2 = '@P0 ' + s

        if self.__mMajor<=6:
            pred = CuSMVersion.PredCode_5x_6x
        else:
            pred = CuSMVersion.PredCode_7x_8x

        # @P0 will set the predicate bit to zero
        code2 = code - (code & pred)

        return addr, code2, s2

    def getNVInfoAttrAutoGenSet(self):
        ''' Get NVInfo attribute set can be automatically generated by kernel assembler.

            TODO: Current list is not complete, check the implementation in class CuNVInfo.
        '''
        if self.__mMajor <= 6:
            return CuSMVersion.EIATTR_AutoGen_5x_6x
        else:
            return CuSMVersion.EIATTR_AutoGen_7x_8x

    def __str__(self):
        return 'CuSMVersion(%d)'%self.__mVersion
    
    def __repr__(self):
        return 'CuSMVersion(%d)'%self.__mVersion

    @staticmethod
    def splitControlCodes_5x_6x(codebytes):
        ''' For 5.x~6.x, 1 64bit control codes + 3*64bit asm instructions

            NOTE: Storing too many big int in python may be very memory consuming.
                  So this may be called segment by segment.
        '''
        # 32B for 1+3 group
        assert (len(codebytes) & 0x1f) == 0

        ctrl_code_list = []
        ins_code_list = []
        bio = BytesIO(codebytes)
        bs = bio.read(32) # (1+3)*8
        while len(bs)==32:
            ccode = int.from_bytes(bs[0:8], 'little')

            ctrl_code_list.append((ccode & 0x00000000001fffff)      )
            ctrl_code_list.append((ccode & 0x000003ffffe00000) >> 21)
            ctrl_code_list.append((ccode & 0x7ffffc0000000000) >> 42)

            ins_code_list.append(int.from_bytes(bs[ 8:16], 'little'))
            ins_code_list.append(int.from_bytes(bs[16:24], 'little'))
            ins_code_list.append(int.from_bytes(bs[24:32], 'little'))

            bs = bio.read(32)

        return ctrl_code_list, ins_code_list

    @staticmethod
    def splitControlCodes_7x_8x(codebytes):
        ''' For 7.x~8.x, 1 128bit instructions, contains 

            NOTE: Storing too many bit int in python may be very memory consuming.
                  So this may be called segment by segment.
        '''

        # 16B for every instruction, should be aligned
        assert (len(codebytes) & 0xf) == 0

        ctrl_code_list = []
        ins_code_list = []
        
        bio = BytesIO(codebytes)
        bs = bio.read(16) # 128bit
        while len(bs)==16:
            code1 = int.from_bytes(bs[0:8], 'little')
            code2 = int.from_bytes(bs[8:16], 'little')

            ctrl_code_list.append( code2 >> 41)  # 23bit control codes
            ins_code_list.append( ((code2 & ((1<<41)-1))<<64) + code1)

            bs = bio.read(16)

        return ctrl_code_list, ins_code_list

    @staticmethod
    def mergeControlCodes_5x_6x(ins_code_list, ctrl_code_list, padding_align=None):
        n_ins = len(ins_code_list)
        if len(ctrl_code_list) != n_ins:
            raise Exception('Length of control codes(%d) != length of instruction(%d)!'
                            %(len(ctrl_code_list), n_ins))
        
        bio = BytesIO()
        nccode_intact =  n_ins // 3  # intact part of control code groups (1+3)
        for i in range(nccode_intact):
            ccode =  ctrl_code_list[3*i] + (ctrl_code_list[3*i+1]<<21) + (ctrl_code_list[3*i+2]<<42)
            bio.write(ccode.to_bytes(8, 'little'))
            bio.write(ins_code_list[3*i].to_bytes(8, 'little'))
            bio.write(ins_code_list[3*i+1].to_bytes(8, 'little'))
            bio.write(ins_code_list[3*i+2].to_bytes(8, 'little'))

        if nccode_intact * 3 != n_ins:
            
            ntail = n_ins - nccode_intact*3
            t_ctrl_code_list = ctrl_code_list[3*nccode_intact:]
            t_ins_code_list  = ins_code_list[3*nccode_intact:]
            
            npad = 3 - ntail
            for i in range(npad):
                t_ctrl_code_list.append(CuSMVersion.Pad_CCode_5x_6x)
                t_ins_code_list.append(CuSMVersion.Pad_ICode_5x_6x)
            
            ccode =  t_ctrl_code_list[0] + (t_ctrl_code_list[1]<<21) + (t_ctrl_code_list[2]<<42)
            bio.write(ccode.to_bytes(8, 'little'))
            bio.write(t_ins_code_list[0].to_bytes(8, 'little'))
            bio.write(t_ins_code_list[1].to_bytes(8, 'little'))
            bio.write(t_ins_code_list[2].to_bytes(8, 'little'))

        pos = bio.tell()
        padlen = padding_align * ((pos+padding_align-1) // padding_align) - pos
        npad = padlen // 32  # 32B = (1+3) instruction group
        bio.write(CuSMVersion.PadBytes_5x_6x * npad)
        
        return bio.getvalue()

    @staticmethod
    def mergeControlCodes_7x_8x(ins_code_list, ctrl_code_list, padding_align=None):
        n_ins = len(ins_code_list)
        if len(ctrl_code_list) != n_ins:
            raise Exception('Length of control codes(%d) != length of instruction(%d)!'
                            %(len(ctrl_code_list), n_ins))
        
        bio = BytesIO()
        for i in range(n_ins):
            code =  (ctrl_code_list[i]<<105) + ins_code_list[i]
            bio.write(code.to_bytes(16, 'little'))

        if padding_align is not None:
            pos = bio.tell()
            padlen = padding_align * ((pos+padding_align-1) // padding_align) - pos
            npad = padlen // 16      # 16B = 1 instruction
            bio.write(CuSMVersion.PadBytes_7x_8x * npad)
        return bio.getvalue()

    @staticmethod
    def parseVersionNumber(version):
        if isinstance(version, str):
            version = version.upper()
        
        if isinstance(version, CuSMVersion):
            version = version.__mVersion
        elif version in CuSMVersion.SMVersionDict:
            version = CuSMVersion.SMVersionDict[version]
        else:
            raise ValueError('Invalid SM version %s!!!' % version)
    
        return version

    
def testOffset():
    v5 = CuSMVersion(52)
    v7 = CuSMVersion(75)

    for i in range(32):
        v5_offset = v5.getInsOffsetFromIndex(i)
        v7_offset = v7.getInsOffsetFromIndex(i)
        print('%2d %04x %04x'%(i, v5_offset, v7_offset))
    
    for v in range(0, 32*8, 8):
        v5_idx = v5.getInsIndexFromOffset(v)
        v7_idx = v7.getInsIndexFromOffset(v)
        print('%04x %4d %4d'%(v, v5_idx, v7_idx))

def testInstance():

    v61_1 = CuSMVersion(61)
    v61_2 = CuSMVersion('61')
    v61_3 = CuSMVersion('sm_61')
    v61_4 = CuSMVersion('SM61')
    v61_5 = CuSMVersion(v61_1)

    v75_1 = CuSMVersion(75)
    v75_2 = CuSMVersion('75')
    v75_3 = CuSMVersion('sm_75')
    v75_4 = CuSMVersion('SM75')
    v75_5 = CuSMVersion(v75_2)

    print('%x'%id(v61_1))
    print('%x'%id(v61_2))
    print('%x'%id(v61_3))
    print('%x'%id(v61_4))
    print('%x'%id(v61_5))

    print('%x'%id(v75_1))
    print('%x'%id(v75_2))
    print('%x'%id(v75_3))
    print('%x'%id(v75_4))
    print('%x'%id(v75_5))

if __name__ == '__main__':

    testInstance()
