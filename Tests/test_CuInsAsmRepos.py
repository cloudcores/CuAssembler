# -*- coding: utf-8 -*-

from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsFeeder import CuInsFeeder


def constructReposFromFile(sassname, savname=None, arch='sm_75'):
    # initialize a feeder with sass
    feeder = CuInsFeeder(sassname, arch=arch)

    # initialize an empty repos
    repos = CuInsAssemblerRepos(arch=arch)#

    # Update the repos with instructions from feeder
    repos.update(feeder)

    # reset the feeder back to start
    feeder.restart()

    # verify the repos
    # actually the codes is already verifed during repos construction
    repos.verify(feeder)

    if savname is not None:
        repos.save2file(savname)

    return repos

def verifyReposFromFile(sassname, reposfile, arch='sm_75'):

    # initialize a feeder with sass
    feeder = CuInsFeeder(sassname, arch=arch)

    # initialize an empty repos
    repos = CuInsAssemblerRepos(reposfile, arch=arch)#

    # verify the repos
    repos.verify(feeder)

if __name__ == '__main__':
    sassname = r'G:\Repos\Tests\Programs\cudatest.sm_86.sass'
    #sassname = r"G:\Temp\cudnn64_7.sm_50.sass"
    reposfile = r'G:\Repos\CuInsAsmRepos.sm_86.txt'

    arch = 'sm_86'

    constructReposFromFile(sassname, reposfile, arch=arch)
    print('### Construction done!')
    
    verifyReposFromFile(sassname, reposfile, arch=arch)
    print('### Verification done!')
