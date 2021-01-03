# -*- coding: utf-8 -*-

from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.common import decodeCtrlCodes, binstr, hexstr

def doExtract(fname, insfilter='', arch='sm_75', maxcnt=100):
    feeder = CuInsFeeder(fname, instfilter=insfilter, arch=arch)
    
    print("# Searching %s with filter %s:"%(fname, insfilter))
    cnt = 0
    for addr, code, s, ctrl in feeder:
        #s = s.replace('.reuse','')
        ctrlstr = decodeCtrlCodes(ctrl)
        #print(' %s: 0x%06x  0x%032x     %s'%(ctrlstr, addr, code, s))
        print('0x%03x: [%s] %#016x  %s '%(addr, ctrlstr, code, s))
        print(binstr(code, 64))
        print(hexstr(code, 64))
        
        cnt += 1
        if cnt>=maxcnt:
            break
    
sassname1 = 'G:\\Temp\\cudnn64_7.sm_50.sass'
sassname2 = 'G:\\Temp\\cudnn64_7.sass'
#sassname1 = 'D:\\MyProjects\\Programs\\cudatest.sm_75.sass'
#sassname2 = 'D:\\MyProjects\\Programs\\cudatest.sm_52.sass'
#sassname = 'G:\\Temp\\cublas64_10.sm_75.sass'
# initialize a feeder with sass
#doExtract(sassname2, r'reuse', arch='sm_75', maxcnt=8)
#print(sassname1)
# doExtract(sassname1, '\.NEG', arch='sm_50', maxcnt=100)
# doExtract(sassname1, '-0\.5', arch='sm_50', maxcnt=100)
doExtract(sassname1, 'FADD.*0\.5', arch='sm_50', maxcnt=10)
#print(sassname2)
#doExtract(sassname2, '', arch='sm_52', maxcnt=888)
#doExtract(sassname1, 'PBK', arch='sm_50', maxcnt=8)
#doExtract(sassname1, 'BRX', arch='sm_50', maxcnt=8)
#
#doExtract(sassname1, 'BRA', arch='sm_50', maxcnt=8)
#doExtract(sassname1, 'CAL', arch='sm_50', maxcnt=8)
#doExtract(sassname1, 'BRK', arch='sm_50', maxcnt=8)
#doExtract(sassname1, 'RET', arch='sm_50', maxcnt=8)
#doExtract(sassname1, 'SSY', arch='sm_50', maxcnt=8)
#doExtract(sassname1, 'SYNC', arch='sm_50', maxcnt=8)
#
#doExtract(sassname2, 'BRA', arch='sm_75', maxcnt=8)
#doExtract(sassname2, 'CAL', arch='sm_75', maxcnt=8)
#doExtract(sassname2, 'BREAK', arch='sm_75', maxcnt=8)
#doExtract(sassname2, 'RET', arch='sm_75', maxcnt=8)
#doExtract(sassname2, 'BSSY', arch='sm_75', maxcnt=8)
#doExtract(sassname2, 'BSYNC', arch='sm_75', maxcnt=8)
