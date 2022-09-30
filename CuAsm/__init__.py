name = 'CuAssembler'
version = '0.1'
author = 'cloudcores'
author_email='yangrc.chn@hotmail.com'
license='MIT'
description='An unofficial cuda assembler'

from CuAsm.CuSMVersion import CuSMVersion
from CuAsm.CuAsmLogger import CuAsmLogger

from CuAsm.CuInsParser import CuInsParser
from CuAsm.CuInsAssembler import CuInsAssembler
from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos

from CuAsm.CuAsmParser import CuAsmParser
from CuAsm.CuKernelAssembler import CuKernelAssembler

from CuAsm.CubinFile import CubinFile
from CuAsm.CuNVInfo import CuNVInfo
from CuAsm.CuInsFeeder import CuInsFeeder

from CuAsm.CuControlCode import CuControlCode
