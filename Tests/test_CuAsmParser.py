# -*- coding: utf-8 -*-
from CuAsm.CuAsmParser import CuAsmParser
import time

if __name__ == '__main__':
    fname = r'G:\Repos\Tests\Programs\cudatest.7.sm_75.cuasm'

    bname = r'G:\Repos\Tests\Programs\cudatest.7.sm_75.cubin'

    reposfile = r'G:\Repos\CuInsAsmRepos.sm_75.txt'
    cap = CuAsmParser()
    cap.setInsAsmRepos(reposfile, arch='sm_75')

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

    cap.saveAsCubin(r'G:\Repos\Tests\Programs\new_cudatest.7.sm_75.cubin')

    sav_prefix = r'G:\Repos\Tests\Programs\cudatest.7.sm_75'
    cap.saveCubinCmp(bname, sav_prefix)