# -*- coding: utf-8 -*-

from CuAsm.CubinFile import CubinFile

binname = 'TestData/Mandelbrot.sm_75.cubin'

print('Processing %s...' % binname)
cf = CubinFile()
cf.initCuKernelAsm('CuAsm/InsAsmRepos.txt')
cf.loadCubin(binname)

asmname = binname.replace('.cubin', '.cuasm')
print('Saving to %s...' % asmname)
cf.saveAsCuAsm(asmname)

cf2 = CubinFile()
cf2.initCuKernelAsm('CuAsm/InsAsmRepos.txt')

print('Loading from %s...' % asmname)
cf2.loadFromCuAsm(asmname)

binname2 = asmname = binname.replace('.sm_75.cubin', '_2.sm_75.cubin')
print('Saving %s...' % binname2)
cf2.saveAsCubin(binname2)

fin1 = open(binname, 'rb')
fOrg = fin1.read()
fin1.close()

fin2 = open(binname2, 'rb')
fAsm = fin2.read()
fin2.close()

if len(fOrg) != len(fAsm):
    print("Error! The length does not match (%d vs %d)!" 
          %(len(fOrg), len(fAsm)))
else:
    passed = True
    for i in range(len(fOrg)):
        if fOrg[i] != fAsm[i]:
            print("Byte at %6d does not match (0x%02x vs 0x%02x)!" 
                   % (i, fOrg[i], fAsm[i]))
            passed = False
    
    if passed:
        print("The two cubins are exactly the same!")
    else:
        print("Error! The two cubins do not match!")