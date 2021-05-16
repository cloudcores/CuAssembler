# -*- coding: utf-8 -*-

from CuAsm.CubinFile import CubinFile
from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuAsmParser import CuAsmParser
from CuAsm.CuAsmLogger import CuAsmLogger

import os
import re
import subprocess
import math

# baseline result for:
# -  grid=1600, block=128, NIter=256, 192 instruction per iteration
# -  No bank conflict
# -  MemFreq = 2505 MHz, SMFreq = 954 MHz
BASELINE = 16.74

class CuAsmTemplate:
    def __init__(self, template_name):
        self.m_FileParts = []
        self.m_MarkerDict = {}

        p = re.compile(r'@CUASM_INSERT_MARKER_POS\.(\w+)')

        with open(template_name) as fin:
            # print(f'Loading template file {template_name}...')
            buf = ''
            iline = 0
            for line in fin:
                iline += 1
                res = p.search(line)
                if res is not None:
                    # push buf in and reset buf
                    self.m_FileParts.append(buf)
                    buf = ''

                    marker = res.groups()[0]
                    if marker in self.m_MarkerDict:
                        print(f'  Duplicate marker "{marker}" in line {iline:d}')
                    else:
                        # print(f'  New marker "{marker}" in line {iline:d}')
                        self.m_MarkerDict[marker] = None
                    
                    self.m_FileParts.append((marker, line))
                else:
                    buf += line

            self.m_FileParts.append(buf)

    def setMarker(self, marker, s):
        self.m_MarkerDict[marker] = s
    
    def resetAllMarkers(self):
        for k in self.m_MarkerDict:
            self.m_MarkerDict[k] = None

    def generate(self, outfile):
        with open(outfile, 'w') as fout:
            for p in self.m_FileParts:
                if isinstance(p, str):
                    fout.write(p)
                elif isinstance(p, tuple):
                    marker, orig_line = p

                    # original line is always written back
                    fout.write(orig_line)
                    if self.m_MarkerDict[marker] is not None:
                        fout.write(self.m_MarkerDict[marker])
                        fout.write('\n') # ensure a newline after the insertion

def cubin2cuasm(binname, asmname=None):
    cf = CubinFile(binname)
    
    if asmname is None:
        if binname.endswith('.cubin'):
            asmname = binname.replace('.cubin', '.cuasm')
        else:
            asmname = binname + '.cuasm'
            
    cf.saveAsCuAsm(asmname)

def cuasm2cubin(asmname, binname=None):
    cap = CuAsmParser()
    cap.parse(asmname)
    if binname is None:
        if asmname.endswith('.cuasm'):
            binname = asmname.replace('.cuasm', '.cubin')
        else:
            binname = asmname + '.cubin'

    cap.saveAsCubin(binname)

def build(use_driver_api=True):
    cuasm2cubin('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')
    if not use_driver_api:
        os.environ['PTXAS_HACK'] = "G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\ptxhack.map"
        subprocess.check_output('make clean')
        subprocess.check_output('make')

def template_test():
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    cat.setMarker('INIT', '      // hehe init here!')
    cat.setMarker('WORK_1', '      // work1 here!')

    cat.generate('test.cuasm')

def parseResult(res:str):

    nt = 0      
    tsum = 0.0
    tsq = 0.0
    tres = ''
    tlist = []

    for line in res.splitlines():
        r = re.search('Test(.*):(.*)ms', line)
        if r is not None:
            t = float(r.groups()[1])
            tlist.append(t)

            nt += 1
            tsum += t
            tsq += t*t
        
        r2 = re.search(r'res\[\s*0\]\s*:(.*)', line)
        if r2 is not None:
            tres = r2.groups()[0].strip()

    # assert(nt==5)
    
    tavg = tsum/nt
    tstd = math.sqrt(tsq/nt - tavg*tavg)

    # teq = tavg * 6 / 16.74 # ratio to base line per instruction
    sfull = '[' + (', '.join(['%8.3f'%t for t in tlist])) + ']'
    return tavg, tstd, tres, sfull

def run_exe():
    try:
        res = subprocess.check_output('regbank_test.exe')
        return res
    except subprocess.CalledProcessError as e:
        print(f'Error happened when running exe (return code={e.returncode})!')
        exit(e.returncode)

def test_NoBankConflict(stall1, stall2, yflag, regnum):
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    s_init  = '[----:B------:R-:W-:-:S01]  MOV R4, RZ ; \n'
    for r in range(8,17):
        s_init += f'[----:B------:R-:W-:-:S01]  MOV32I R{r:d}, 0x3f800000 ; \n'

    s_work1  = f'      [----:B------:R-:W-:{yflag}:S{stall1:02d}]  FFMA R4, R9, R10, R4; \n' # R4 += 1
    for i in range(5):
        s_work1 += f'      [----:B------:R-:W-:{yflag}:S{stall2:02d}]  FFMA R{i+12}, R9, R10, R11; \n' # R4 += 1
    s_work1 = s_work1 * 32

    s_work2 = s_work1

    cat.setMarker('INIT', s_init)
    cat.setMarker('WORK_1', s_work1)
    cat.setMarker('WORK_2', s_work2)

    s_regnum = f'  	.sectioninfo	@"SHI_REGISTERS={regnum}"'
    cat.setMarker('REGNUM', s_regnum)

    cat.generate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')

    build()

    res = run_exe()
    return parseResult(res.decode())

def doTest_NoBankConflict():
    with open('res_NoBankConflict.txt', 'w') as fout:
        regnum = 32
        # print(f'{"":8s} {"TAvg":>8s} ({"TRatio":>8s}) {"TStd":>8s}    {"Results":s}')
        for stall in range(1,16):
            for yflag in ['-', 'Y']:

                tavg, tstd, tres, sfull = test_NoBankConflict(stall, stall, yflag, regnum)
                tratio = tavg * 6 / BASELINE
                s = f'[{yflag}:S{stall:02d}]: {tavg:8.3f} ({tratio:8.4f}), {tstd:8.3f},  {tres:s}  {sfull}'
                print(s)
                fout.write(s+'\n')
        
        s = '\n######### S## + 5*S01 ############\n' # + f'{"":8s} {"TAvg":>8s}  {"TStd":>8s} ({"TRatio":>8s})   {"Results":s}'
        print(s)
        fout.write(s+'\n')
        for stall in range(1,16):
            for yflag in ['-', 'Y']:
                tavg, tstd, tres, sfull = test_NoBankConflict(stall, 1, yflag, 32)
                s = f'[{yflag}:S{stall:02d}]: {tavg:8.3f} ({tratio:8.4f}), {tstd:8.3f},  {tres:s}  {sfull}'
                print(s)
                fout.write(s+'\n')

def test_BankConflictComb(stall, ins_seq):
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')

    s_init  = '[----:B------:R-:W-:-:S01]  MOV32I R4, 0x7ff00000 ; \n' # R4 should not be modified in ins_seq
    s_work1 = ''
    for ins in ins_seq:
        s_work1 += f'      [----:B------:R-:W-:-:S{stall:02d}] {ins} \n' #

    s_work1 = s_work1 * 32
    s_work2 = s_work1

    cat.setMarker('INIT', s_init)
    cat.setMarker('WORK_1', s_work1)
    cat.setMarker('WORK_2', s_work2)
    cat.generate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')

    build()
    res = run_exe()
    
    return parseResult(res.decode()) 

def doTest_BankConflictComb():
    from test_bc_enum import buildCombDict, getCombStr, genBankConflictInsSeq

    comb_dict = buildCombDict()
    comb_keys = list(comb_dict.keys())
    comb_keys.sort(reverse=True)

    with open('res_BankConflictComb.txt', 'w') as fout:
        
        for i,k in enumerate(comb_keys):    
            i0, i1, i2 = comb_dict[k]
            ks = getCombStr(i0, i1, i2)
            ins_seq = genBankConflictInsSeq(i0,i1,i2)

            bsum = [i0[0]+i1[0]+i2[0], i0[1]+i1[1]+i2[1], i0[2]+i1[2]+i2[2], i0[3]+i1[3]+i2[3]]
            bs = ''.join(['%d'%x for x in bsum])
            besti = 2*max(bsum)

            tavg, tstd, tres, sfull = test_BankConflictComb(1, ins_seq)
            tratio = tavg * 6 / BASELINE
            s = f'Comb{i+1:4d} : [{ks}] ({bs}:{besti:2d}): {tavg:8.3f} ({tratio:8.4f}), {tstd:8.3f},  {tres:s}  {sfull}'
            print(s)
            fout.write(s+'\n')

def test_ReuseBankConflict(r1, r2, reuse1, reuse2):
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    s_init  = '[----:B------:R-:W-:-:S01]  MOV R4, RZ ; \n'
    for r in range(8,17):
        s_init += f'[----:B------:R-:W-:-:S01]  MOV32I R{r:d}, 0x3f800000 ; \n'

    reuse_s = reuse1 + reuse2 + '--'
    s_work1  = f'      [{reuse_s}:B------:R-:W-:-:S01]  FFMA R4, R{r1}, R{r2}, R4; \n' # R4 += 1
    for i in range(5):
        s_work1 += f'      [{reuse_s}:B------:R-:W-:-:S01]  FFMA R{i+20}, R{r1}, R{r2}, R16; \n' #
    s_work1 = s_work1 * 32

    s_work2 = s_work1

    cat.setMarker('INIT', s_init)
    cat.setMarker('WORK_1', s_work1)
    cat.setMarker('WORK_2', s_work2)
    # cat.setMarker('FINALIZE', s_final)

    cat.generate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')

    build()

    res = run_exe()
    
    return parseResult(res.decode()) 

def doTest_ReuseBankConflict():
    with open('res_ReuseBankConflict.txt', 'w') as fout:
        for r1 in range(8, 16):
            r1s = f'R{r1}'
            for r2 in range(8, 16):
                r2s = f'R{r2}'
                for reuse1 in ['-', 'R']:
                    for reuse2 in ['-', 'R']:
                        tavg, tstd, tres, sfull = test_ReuseBankConflict(r1, r2, reuse1, reuse2)
                        tratio = tavg * 6 / BASELINE
                        s = f'({r1s:3s}, {r2s:3s}, "{reuse1}{reuse2}"): {tavg:8.3f} ({tratio:8.4f}), {tstd:8.3f},  {tres:s}  {sfull}'
                        print(s)
                        fout.write(s+'\n')
                        

def test_ReuseStall(r1, r2, stall, reuse_s, regnum):
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    s_init  = '[----:B------:R-:W-:-:S01]  MOV R4, RZ ; \n'
    for r in range(8,17):
        s_init += f'[----:B------:R-:W-:-:S01]  MOV32I R{r:d}, 0x3f800000 ; \n'

    # r1 = 8
    # r2 = 12
    s_work1  = f'      [{reuse_s}:B------:R-:W-:-:S{stall:02d}]  FFMA R4, R{r1}, R{r2}, R4; \n' # R4 += 1
    for i in range(5):
        s_work1 += f'      [{reuse_s}:B------:R-:W-:-:S{stall:02d}]  FFMA R{i+20}, R{r1}, R{r2}, R16; \n' #
    s_work1 = s_work1 * 32

    s_work2 = s_work1

    cat.setMarker('INIT', s_init)
    cat.setMarker('WORK_1', s_work1)
    cat.setMarker('WORK_2', s_work2)

    s_regnum = f'  	.sectioninfo	@"SHI_REGISTERS={regnum}"'
    cat.setMarker('REGNUM', s_regnum)

    cat.generate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')

    build()

    res = run_exe()
    return parseResult(res.decode())

def doTest_ReuseStall():
    with open('res_ReuseStall.txt', 'w') as fout:
        
        s = '\n#### (R8, R12) ####\n'
        print(s)
        fout.write(s+'\n')

        for stall in range(1, 16):
            for reuse_s in ['----', 'RR--']:
                for regnum in [254, 160, 128, 96, 80, 64, 40, 32]: 
                    occu = min(16, 512//regnum)
                    tavg, tstd, tres, sfull = test_ReuseStall(8, 12, stall, reuse_s, regnum)
                    tratio = tavg * 6 / BASELINE
                    s = f'[{reuse_s}:S{stall:02d}] (RegNum={regnum:3d}, Occu={occu:2d}): {tavg:8.3f} ({tratio:8.4f}), {tstd:8.3f},  {tres:s}  {sfull}'
                    print(s)
                    fout.write(s+'\n')
        
        s = '\n#### (R8, R9) ####\n'
        print(s)
        fout.write(s+'\n')
        for stall in range(1, 16):
            for reuse_s in ['----', 'R---']:
                for regnum in [254, 160, 128, 96, 80, 64, 40, 32]:
                    occu = min(16, 512//regnum)
                    tavg, tstd, tres, sfull = test_ReuseStall(8, 9, stall, reuse_s, regnum)
                    tratio = tavg * 6 / BASELINE
                    s = f'[{reuse_s}:S{stall:02d}] (RegNum={regnum:3d}, Occu={occu:2d}): {tavg:8.3f} ({tratio:8.4f}), {tstd:8.3f},  {tres:s}  {sfull}'
                    print(s)
                    fout.write(s+'\n')
                    

def test_ReuseSwitch(stall, cycle, clip):
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    s_init  = '[----:B------:R-:W-:-:S01]  MOV R4, RZ ; \n'
    for r in range(8,17):
        s_init += f'[----:B------:R-:W-:-:S01]  MOV32I R{r:d}, 0x3f800000 ; \n'

    RList = [(8,12,16), (9,13,5), (10,14,6), (11,15,7)]
    if clip==0:
        s_work1  = f'      [----:B------:R-:W-:-:S{stall:02d}]  FFMA R4, R8, R12, R4; \n' # R4 += 1
    else:
        s_work1  = f'      [RR--:B------:R-:W-:-:S{stall:02d}]  FFMA R4, R8, R12, R4; \n' # R4 += 1
    for i in range(1,6):
        idx = i%cycle
        r1, r2, r3 = RList[idx]
        
        if idx>=clip:
            reuse_s = '----'
            # r1, r2, r3 = 10, 9, 8
        else:
            reuse_s = 'RR--'
            
        s_work1 += f'      [{reuse_s}:B------:R-:W-:-:S{stall:02d}]  FFMA R{i+17}, R{r1}, R{r2}, R{r3}; \n' # R4 += 1
    
    #print(s_work1)
    s1s = s_work1

    s_work1 = s_work1 * 32

    s_work2 = s_work1

    cat.setMarker('INIT', s_init)
    cat.setMarker('WORK_1', s_work1)
    cat.setMarker('WORK_2', s_work2)

    cat.generate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')

    build()

    res = run_exe()
    
    return parseResult(res.decode()), s1s

def doTest_ReuseSwitch():
    with open('res_ReuseSwitch.txt', 'w') as fout:
        for stall in range(1, 7):
            for cycle in [1, 2, 3]:
                for clip in range(0, cycle+1):
                    (tavg, tstd, tres, sfull), s1s = test_ReuseSwitch(stall, cycle, clip)
                    tratio = tavg * 6 / BASELINE
                    s = f'[S{stall:02d}, Cycle{cycle:d}, Clip{clip:d}]: {tavg:8.3f} ({tratio:8.4f}), {tstd:8.3f},  {tres:s}  {sfull}'
                    
                    print('---------------------------------\n')
                    print(s1s, end='')
                    print(s)

                    fout.write('---------------------------------\n')
                    fout.write(s1s)
                    fout.write(s+'\n')

def test_Simple():
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    s_init  = '[----:B------:R-:W-:-:S06]  MOV R4, RZ ; \n'
    for r in range(8,17):
        s_init += f'[----:B------:R-:W-:-:S06]  MOV32I R{r:d}, 0x3f800000 ; \n'

    stall = 1
    s_work1  = f'      [----:B------:R-:W-:Y:S{stall:02d}]  FFMA R4, R9, R10, R4; \n' # R4 += 1
    # for i in range(1,8):
    #    s_work1 += f'      [----:B------:R-:W-:-:S{stall:02d}]  FFMA R{24+i}, R{8+i}, R{9+i}, R{10+i}; \n' # R4 += 1

    s_work1 += f'      [RR--:B------:R-:W-:-:S{stall:02d}]  FFMA R18, R7, R11, R15;  \n' # R4 += 1
    s_work1 += f'      [RR--:B------:R-:W-:-:S{stall:02d}]  FFMA R19, R12, R8, R16;  \n' # R4 += 1
    s_work1 += f'      [RR--:B------:R-:W-:-:S{stall:02d}]  FFMA R20, R11, R7, R15; \n' # R4 += 1
    s_work1 += f'      [RR--:B------:R-:W-:-:S{stall:02d}]  FFMA R21, R8, R12, R16; \n' # R4 += 1
    s_work1 += f'      [RR--:B------:R-:W-:-:S{stall:02d}]  FFMA R22, R7, R11, R15;  \n' # R4 += 1
    s_work1 += f'      [RR--:B------:R-:W-:-:S{stall:02d}]  FFMA R23, R12, R8, R16;  \n' # R4 += 1
    s_work1 += f'      [RR--:B------:R-:W-:-:S{stall:02d}]  FFMA R24, R11, R7, R15;  \n' # R4 += 1

    s_work1 = s_work1 * 32
    s_work2 = s_work1

    # s_final  = '[----:B------:R-:W-:-:S06]  MOV32I R4, 0x3f800000 ; \n'

    cat.setMarker('INIT', s_init)
    cat.setMarker('WORK_1', s_work1)
    cat.setMarker('WORK_2', s_work2)
    # cat.setMarker('FINALIZE', s_final)

    cat.generate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')

    build()

    res = run_exe()
    print(res.decode())

def test_Simple2():
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    s_init  = '[----:B------:R-:W-:-:S06]  MOV R4, RZ ; \n'
    for r in range(8,17):
        s_init += f'[----:B------:R-:W-:-:S06]  MOV32I R{r:d}, 0x3f800000 ; \n'

    stall = 1
    s_work1  = f'      [----:B------:R-:W-:-:S03]  FFMA R4, R9, R10, R4; \n' # R4 += 1
    s_work1 += f'      [----:B------:R-:W-:-:S03]  FFMA R18, R9, R10, R11; \n'
    s_work1 += f'      [----:B------:R-:W-:-:S03]  FFMA R19, R9, R10, R11; \n'
    s_work1 += f'      [----:B------:R-:W-:-:S03]  FFMA R20, R9, R10, R11; \n'
    s_work1 += f'      [----:B------:R-:W-:-:S03]  FFMA R21, R9, R10, R11; \n'
    s_work1 += f'      [----:B------:R-:W-:-:S03]  FFMA R22, R9, R10, R11; \n'

    s_work1 = s_work1 * 32
    s_work2 = s_work1
    cat.setMarker('INIT', s_init)
    cat.setMarker('WORK_1', s_work1)
    cat.setMarker('WORK_2', s_work2)

    cat.generate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')

    build()

    res = run_exe()
    print(res.decode())

if __name__ == '__main__':
    
    # os.system('nvidia-smi -ac 2505,954')

    # cubin2cuasm('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.2.sm_50.cubin')
    # cubin2cuasm('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.uni.sm_50.cubin')
    # cubin2cuasm('G:\\Temp\CubinSample\\eigenvalues\\NVIDIA.4.sm_50.cubin')
    # build()
    CuAsmLogger.disable()
    # tmp_test()
    
    # doTest_NoBankConflict()
    doTest_BankConflictComb()
    # doTest_ReuseBankConflict()

    # doTest_ReuseSwitch()
    # doTest_ReuseStall()

    # test_Simple2()

    
