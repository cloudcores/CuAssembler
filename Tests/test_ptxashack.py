import os

def testEmptyArg():
    os.system('ptxas')

def testBuildPTX():
    os.system('ptxas test.ptx -o test.cubin -arch sm_50')

def testBuildPTXFail():
    os.system('ptxas test_notfound.ptx -o test.cubin -arch sm_50')

def testBuildPTXHack():
    os.environ['PTXAS_HACK'] = "G:\\Work\\CuAssembler\\Tools\\ptxhack.map"
    os.system('ptxas test.ptx -o test.cubin -arch sm_50')
    
def testBuildPTXHackSkip():
    os.environ['PTXAS_HACK'] = "G:\\Work\\CuAssembler\\Tools\\ptxhack.map"
    os.system('ptxas test_notfound.ptx -o test2.cubin -arch sm_50')


if __name__ == '__main__':
    # testEmptyArg()
    # testBuildPTX()
    # testBuildPTXFail()

    # testBuildPTXHack()
    testBuildPTXHackSkip()
