# -*- coding: utf-8 -*-

from CuAsm.CubinFile import CubinFile
from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuAsmParser import CuAsmParser

import os
import re

class CuAsmTemplate:
    def __init__(self, template_name):
        self.m_FileParts = []
        self.m_MarkerDict = {}

        p = re.compile(r'@CUASM_INSERT_MARKER_POS\.(\w+)')

        with open(template_name) as fin:
            print(f'Loading template file {template_name}...')
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
                        print(f'  New marker "{marker}" in line {iline:d}')
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
    os.system('make clean')
    os.system('make')

def template_test():
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    cat.setMarker('INIT', '      // hehe init here!')
    cat.setMarker('WORK_1', '      // work1 here!')

    cat.generate('test.cuasm')


def test_NoBankConflict():
    cat = CuAsmTemplate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.template.sm_50.cuasm')
    
    s_init  = '[----:B------:R-:W-:-:S06]  MOV R4, RZ ; \n'
    s_init += '[----:B------:R-:W-:-:S06]  MOV32I R9, 0x3f800000 ; \n'
    s_init += '[----:B------:R-:W-:-:S06]  MOV32I R10, 0x3f800000 ; \n'
    # s_init += '[----:B------:R-:W-:-:S06]  MOV R11, 0x3f8 ; \n'
    # s_init += '[----:B------:R-:W-:-:S06]  MOV R12, 0x3f8 ; \n'
    # s_init += '[----:B------:R-:W-:Y:S15]  MOV32I R10, 0x3f800000 ; \n'

    s_work1  = '[----:B------:R-:W-:Y:S15]  FFMA R4, R9, R10, R4; \n' # R4 += 1
    s_work1 = s_work1 * 128

    s_work2  = '[----:B------:R-:W-:-:S06]  FFMA R4, R9, R9, R4; \n' # R4 += 1
    s_work2 = s_work2 * 128

    s_final  = '[----:B------:R-:W-:-:S06]  MOV32I R4, 0x3f800000 ; \n'

    cat.setMarker('INIT', s_init)
    cat.setMarker('WORK_1', s_work1)
    cat.setMarker('WORK_2', s_work2)
    # cat.setMarker('FINALIZE', s_final)

    cat.generate('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.rep.sm_50.cuasm')

    build()

    os.system('regbank_test.exe')


if __name__ == '__main__':
    # cubin2cuasm('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.2.sm_50.cubin')
    # cubin2cuasm('G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\regbank_test.uni.sm_50.cubin')
    # build()

    # tmp_test()
    test_NoBankConflict()
