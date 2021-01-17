# -*- coding: utf-8 -*-
from CuAsm.CuAsmParser import CuAsmParser
import time

if __name__ == '__main__':

    fprefix = 'G:\\Work\\CuAssembler\\TestData\\CuTest\\cudatest.7.sm_75'

    fname = fprefix + '.cuasm'
    bname = fprefix + '.cubin'
    sname = fprefix + '_new.cubin'

    cap = CuAsmParser()
    cap.parse(fname)


    #cap.dispRelocationList()
    #cap.dispFileHeader()
    #cap.dispSectionList()
    #cap.dispSymbolDict()
    #cap.dispSymtabDict()
    #cap.dispSegmentHeader()
    #cap.dispTables()
    
    #cap.dispFixupList()
    #cap.dispFixupList()
    #cap.dispRelocationList()
    #cap.dispLabelDict()

    cap.saveAsCubin(sname)
    cap.saveCubinCmp(bname, fprefix)