# -*- coding: utf-8 -*-

from CuAsm.CubinFile import CubinFile
from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuAsmParser import CuAsmParser
import os

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
    cuasm2cubin(r'G:\Work\CuAssembler\TestData\microbench\RegBank\regbank_test.rep.sm_50.cuasm')
    os.environ['PTXAS_HACK'] = "G:\\Work\\CuAssembler\\TestData\\microbench\\RegBank\\ptxhack.map"
    os.system('make')


if __name__ == '__main__':
    # cubin2cuasm(r'G:\Work\CuAssembler\TestData\microbench\RegBank\regbank_test.2.sm_50.cubin')
    build()
