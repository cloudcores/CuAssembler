# -*- coding: utf-8 -*-

import re

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

# Pattern for instruction line of cuasm
c_CuAsmInsLine = re.compile(r'^\s*(L\w+\s*:)?\s*\[([R-]{4}:B[0-7-]{6}:R.:W.:.:S\d{2})\] +([^;]+;)')

# Pattern for skipped lines in cuasm
c_CuAsmSkipLine = re.compile(r'^\s*(#.*)?$') # comment or empty line

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