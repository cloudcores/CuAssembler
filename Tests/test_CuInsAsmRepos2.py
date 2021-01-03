# -*- coding: utf-8 -*-

from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsFeeder import CuInsFeeder


if __name__ == '__main__':
    sassname = r'G:\Repos\Tests\Programs\cudatest.sm_75.sass'
    #sassname = r"G:\Temp\cudnn64_7.sm_50.sass"
    reposfile = r'G:\Repos\CuAsm\InsAsmRepos\CuInsAsmRepos.sm_75.txt'
    repos = CuInsAssemblerRepos(reposfile)


    reposfile2 = r'G:\Repos\CuInsAsmRepos.sm_75.txt'
    repos.merge(reposfile2)

    repos.completePredCodes()
    repos.save2file(r'G:\Repos\new.sm_75.txt')

