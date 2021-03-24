# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def loadTimeFile():
    t = np.fromfile('t.dat', dtype='float32')
    
    NBase = 64
    
    t2 = (t>1.2)
    t2 = t2.reshape((NBase,1024))
    
    ts = np.zeros((1024,), dtype='int32')
    chn_counter = 0
    for i in range(NBase):
        #mask = t2[i*1024:(i+1)*1024]
        mask = t2[i,:]
        v = ts[mask][0]
        
        if all(ts[mask]==v):
            if v==0:
                ts[mask] = chn_counter
                chn_counter += 1
        else:
            print("%d set not match!"%i)
            
    
#for i in range(NBase):
#    for j in range(NBase):
#        v = any(np.logical_xor(t2[i,:], t2[j,:]))
#        if v:
#            print('0  ',end='')
#        else:
#            print('1  ',end='')
#    
#    print()

ts = np.fromfile('group.dat', dtype='int32')

NSeg = 128
yoffset = 0.1/NSeg

tmat = ts.reshape((NSeg,-1))

plt.figure()
plt.clf()
#for i in range(16):
#    plt.plot(i*1.5+t2[i,:])
for i in range(NSeg):
    plt.plot(tmat[i,:]+i*yoffset, label='%d'%i)
    plt.text(0, tmat[i,0]+i*yoffset, '%d'%i)
#plt.legend()
plt.show()



