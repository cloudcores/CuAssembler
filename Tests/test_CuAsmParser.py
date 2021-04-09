# -*- coding: utf-8 -*-
from CuAsm.CuAsmParser import CuAsmParser
import time

if __name__ == '__main__':

    # fprefix = 'G:\\Work\\CuAssembler\\TestData\\CuTest\\cudatest.7.sm_75'
    fprefix = 'G:\\Work\\CuAssembler\\TestData\\CuTest\\cudatest.6.sm_61'

    fname = fprefix + '.cuasm'
    bname = fprefix + '.cubin'
    sname = fprefix + '_new.cubin'

    cap = CuAsmParser()
    # cap.setInsAsmRepos('G:\\work\\tmp.sm_61.txt', arch='sm_61')
    cap.parse(fname)


    # cap.dispRelocationList()
    # cap.dispFileHeader()
    # cap.dispSymbolDict()
    # cap.dispSymtabDict()
    # cap.dispSegmentHeader()
    # cap.dispTables()
    
    #cap.dispFixupList()
    #cap.dispFixupList()
    #cap.dispRelocationList()
    #cap.dispLabelDict()

    cap.saveAsCubin(sname)
    cap.saveCubinCmp(bname, fprefix)

    cap.dispSectionList()