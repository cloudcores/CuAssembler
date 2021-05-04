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

def build():
    cuasm2cubin('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')
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

    for line in res.splitlines():
        r = re.search('Test(.*):(.*)ms', line)
        if r is not None:
            t = float(r.groups()[1])

            nt += 1
            tsum += t
            tsq += t*t
        
        r2 = re.search(r'res\[\s*0\]\s*:(.*)', line)
        if r2 is not None:
            tres = r2.groups()[0].strip()

    assert(nt==10)
    
    tavg = tsum/nt
    tstd = math.sqrt(tsq/nt - tavg*tavg)
    return tavg, tstd, tres

def test_NoBankConflict(stall=1, yflag='-'):
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    s_init  = '[----:B------:R-:W-:-:S06]  MOV R4, RZ ; \n'
    s_init += '[----:B------:R-:W-:-:S06]  MOV32I R9, 0x3f800000 ; \n'
    s_init += '[----:B------:R-:W-:-:S06]  MOV32I R10, 0x3f800000 ; \n'
    s_init += '[----:B------:R-:W-:-:S06]  MOV32I R11, 0x3f800000 ; \n'
    s_init += '[----:B------:R-:W-:-:S06]  MOV32I R12, 0x3f800000 ; \n'
    s_init += '[----:B------:R-:W-:-:S06]  MOV32I R13, 0x3f800000 ; \n'
    # s_init += '[----:B------:R-:W-:-:S06]  MOV R11, 0x3f8 ; \n'
    # s_init += '[----:B------:R-:W-:-:S06]  MOV R12, 0x3f8 ; \n'
    # s_init += '[----:B------:R-:W-:Y:S15]  MOV32I R10, 0x3f800000 ; \n'

    s_work1  = f'      [----:B------:R-:W-:{yflag}:S{stall:02d}]  FFMA R4, R9, R10, R4; \n' # R4 += 1
    for i in range(5):
        s_work1 += f'      [----:B------:R-:W-:{yflag}:S{stall:02d}]  FFMA R{i+12}, R9, R10, R11; \n' # R4 += 1
    s_work1 = s_work1 * 32

    s_work2 = s_work1

    # s_final  = '[----:B------:R-:W-:-:S06]  MOV32I R4, 0x3f800000 ; \n'

    cat.setMarker('INIT', s_init)
    cat.setMarker('WORK_1', s_work1)
    cat.setMarker('WORK_2', s_work2)
    # cat.setMarker('FINALIZE', s_final)

    cat.generate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')

    build()

    res = subprocess.check_output('regbank_test.exe')
    tavg, tstd, tres = parseResult(res.decode())
    return tavg, tstd, tres

def doTest_NoBankConflict():
    with open('res_NoBankConflict.txt', 'w') as fout:
        for stall in range(1,13):
            for yflag in ['-', 'Y']:
            
                tavg, tstd, tres = test_NoBankConflict(stall)
                s = f'[{yflag}:S{stall:02d}]: {tavg:8.3f}, {tstd:8.3f},  {tres:s}'
                print(s)
                fout.write(s+'\n')
                fout.flush()

def test_ReuseBankConflict(r1, r2, reuse1='-', reuse2='-'):
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    s_init  = '[----:B------:R-:W-:-:S06]  MOV R4, RZ ; \n'
    for r in range(8,17):
        s_init += f'[----:B------:R-:W-:-:S06]  MOV32I R{r:d}, 0x3f800000 ; \n'

    reuse_s = reuse1 + reuse2 + '--'
    s_work1  = f'      [{reuse_s}:B------:R-:W-:-:S01]  FFMA R4, R{r1}, R{r2}, R4; \n' # R4 += 1
    for i in range(5):
        s_work1 += f'      [{reuse_s}:B------:R-:W-:-:S01]  FFMA R{i+16}, R{r1}, R{r2}, R4; \n' # R4 += 1
    s_work1 = s_work1 * 32

    s_work2 = s_work1

    cat.setMarker('INIT', s_init)
    cat.setMarker('WORK_1', s_work1)
    cat.setMarker('WORK_2', s_work2)
    # cat.setMarker('FINALIZE', s_final)

    cat.generate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')

    build()

    res = subprocess.check_output('regbank_test.exe')
    tavg, tstd, tres = parseResult(res.decode())
    return tavg, tstd, tres

def doTest_ReuseBankConflict():
    with open('res_ReuseBankConflict.txt', 'w') as fout:
        for r1 in range(8, 16):
            r1s = f'R{r1}'
            for r2 in range(8, 16):
                r2s = f'R{r2}'
                for reuse1 in ['-', 'R']:
                    for reuse2 in ['-', 'R']:
                        tavg, tstd, tres = test_ReuseBankConflict(r1, r2, reuse1, reuse2)
                        s = f'({r1s:3s}, {r2s:3s}, "{reuse1}{reuse2}"): {tavg:8.3f}, {tstd:8.3f},  {tres:s}'
                        print(s)
                        fout.write(s+'\n')
                        fout.flush()

def test_ReuseStall(stall, reuse_s):
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    s_init  = '[----:B------:R-:W-:-:S06]  MOV R4, RZ ; \n'
    for r in range(8,17):
        s_init += f'[----:B------:R-:W-:-:S06]  MOV32I R{r:d}, 0x3f800000 ; \n'

    r1 = 8
    r2 = 12
    s_work1  = f'      [{reuse_s}:B------:R-:W-:-:S{stall:02d}]  FFMA R4, R{r1}, R{r2}, R4; \n' # R4 += 1
    for i in range(5):
        s_work1 += f'      [{reuse_s}:B------:R-:W-:-:S{stall:02d}]  FFMA R{i+16}, R{r1}, R{r2}, R4; \n' # R4 += 1
    s_work1 = s_work1 * 32

    s_work2 = s_work1

    cat.setMarker('INIT', s_init)
    cat.setMarker('WORK_1', s_work1)
    cat.setMarker('WORK_2', s_work2)

    cat.generate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')

    build()

    res = subprocess.check_output('regbank_test.exe')
    tavg, tstd, tres = parseResult(res.decode())
    return tavg, tstd, tres

def doTest_ReuseStall():
    with open('res_ReuseStall.txt', 'w') as fout:
        for stall in range(1, 13):
            for reuse_s in ['----', 'RR--']:
                tavg, tstd, tres = test_ReuseStall(stall, reuse_s)
                s = f'[{reuse_s}:S{stall:02d}]: {tavg:8.3f}, {tstd:8.3f},  {tres:s}'
                print(s)
                fout.write(s+'\n')
                fout.flush()

def test_ReuseSwitch(stall, cycle, clip):
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    s_init  = '[----:B------:R-:W-:-:S06]  MOV R4, RZ ; \n'
    for r in range(8,17):
        s_init += f'[----:B------:R-:W-:-:S06]  MOV32I R{r:d}, 0x3f800000 ; \n'

    RList = [(8,12,16), (9,13,5), (10,14,6), (11,15,7)]
    s_work1  = f'      [RR--:B------:R-:W-:-:S{stall:02d}]  FFMA R4, R8, R12, R4; \n' # R4 += 1
    for i in range(1,12):
        idx = i%cycle
        r1, r2, r3 = RList[idx]
        if idx>=clip:
            reuse_s = '----'
        else:
            reuse_s = 'RR--'
        s_work1 += f'      [{reuse_s}:B------:R-:W-:-:S{stall:02d}]  FFMA R{i+17}, R{r1}, R{r2}, R{r3}; \n' # R4 += 1
    
    s_work1 = s_work1 * 32

    s_work2 = s_work1

    cat.setMarker('INIT', s_init)
    cat.setMarker('WORK_1', s_work1)
    cat.setMarker('WORK_2', s_work2)

    cat.generate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')

    build()

    res = subprocess.check_output('regbank_test.exe')
    tavg, tstd, tres = parseResult(res.decode())
    return tavg, tstd, tres

def doTest_ReuseSwitch():
    with open('res_ReuseSwitch.txt', 'w') as fout:
        for stall in range(1, 8):
            for cycle in range(1, 5):
                for clip in range(0, cycle+1):
                    tavg, tstd, tres = test_ReuseSwitch(stall, cycle, clip)
                    s = f'[S{stall:02d}, Cycle{cycle:d}, Clip{clip:d}]: {tavg:8.3f}, {tstd:8.3f},  {tres:s}'
                    print(s)
                    fout.write(s+'\n')
                    fout.flush()

if __name__ == '__main__':
    # cubin2cuasm('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.2.sm_50.cubin')
    # cubin2cuasm('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.uni.sm_50.cubin')
    # build()
    CuAsmLogger.disable()
    # tmp_test()
    # doTest_ReuseSwitch()

    doTest_NoBankConflict()
    doTest_ReuseBankConflict()

    
