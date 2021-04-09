# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

def match_bit(x, v, vbit, xbit):
    rx = x ^ (1<<xbit)
    vs = (v & (1<<vbit)) > 0
    vc = vs ^ vs[rx]
    
    return vc.sum()

def binstr(b, l):
    bs = bin(b)[2:]
    nbs = len(bs)
    if nbs<l:
        bs = '0'*(l-nbs)+bs
    
    return bs

ts = np.fromfile('group_bak.dat', dtype='int32')-1

nts = len(ts)

N_xbit = nts.bit_length()-1
N_vbit = 3

NT = 2**N_xbit #len(ts)

ts = ts[0:NT]

x = np.r_[0:NT]

cs = np.zeros((N_xbit, N_vbit), dtype='double')

for vb in range(N_vbit):
    for xb in range(N_xbit):
        s = match_bit(x, ts, vb, xb)
        print(' %2d  %2d  %8d  %s'%(vb, xb, s, binstr(s, N_xbit+1)))
        
        cs[xb, vb] = s/NT
    
    print()

plt.figure()
plt.plot(cs)
plt.show()


