# -*- coding: utf-8 -*-

from CuAsm.CubinFile import CubinFile
#from CuAsm.CuInsAssembler import 


def cubin2cuasm(binname):
    # print('Processing %s...' % binname)
    cf = CubinFile(binname)
    # cf.initCuKernelAsm('CuAsm/InsAsmRepos.txt')
    # cf.loadCubin(binname)

    asmname = binname.replace('.cubin', '.cuasm')
    # print('Saving to %s...' % asmname)
    cf.saveAsCuAsm(asmname)

if __name__ == '__main__':
    import glob
    fpattern = r'G:\Work\CuAssembler\TestData\CuTest\*.cubin'
    
    
    for fname in glob.glob(fpattern):
        cubin2cuasm(fname)
