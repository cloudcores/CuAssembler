# -*- coding: utf-8 -*-
from collections import OrderedDict

# possible bank (read) distribution of ONE instruction
ins_bank_list = [
         (3,0,0,0), (0,3,0,0), (0,0,3,0), (0,0,0,3),
         (2,1,0,0), (2,0,1,0), (2,0,0,1), (1,2,0,0),
         (0,2,1,0), (0,2,0,1), (1,0,2,0), (0,1,2,0),
         (0,0,2,1), (1,0,0,2), (0,1,0,2), (0,0,1,2),
         (1,1,1,0), (1,1,0,1), (1,0,1,1), (0,1,1,1),
         (2,0,0,0), (0,2,0,0), (0,0,2,0), (0,0,0,2),
         (1,1,0,0), (1,0,1,0), (1,0,0,1), (0,1,1,0), (0,1,0,1), (0,0,1,1),
         (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)]


def getCombKey(ib0, ib1, ib2):
    ks = []
    ibs = ib0, ib1, ib2
    
    for i in range(3):
        sib0, sib1, sib2 = ibs[i], ibs[(i+1)%3], ibs[(i+2)%3]
        b0 = (sib0[0], sib1[0], sib2[0])
        b1 = (sib0[1], sib1[1], sib2[1])
        b2 = (sib0[2], sib1[2], sib2[2])
        b3 = (sib0[3], sib1[3], sib2[3])

        bs = [b0, b1, b2, b3]
        bs.sort(reverse=True)
        ks.append(tuple(bs))

    combkey = max(ks)
    return '%s'%str(combkey)

def getCombStr(ib0, ib1, ib2):
    ib0s = ''.join(['%d'%x for x in ib0])
    ib1s = ''.join(['%d'%x for x in ib1])
    ib2s = ''.join(['%d'%x for x in ib2])

    return ib0s +','+ib1s+','+ib2s

def writeComb2File(comb_dict, fname='comb.txt'):
    with open(fname, 'w') as fout:
        keys = list(comb_dict.keys())
        keys.sort(reverse=True)
        for i,k in enumerate(keys):
            i0,i1,i2 = comb_dict[k]
            s = getCombStr(i0, i1, i2)
            
            fout.write(f'Comb {i+1:4d}: {k}: {s}\n')

def buildCombDict():
    comb_dict = OrderedDict()
    n = len(ins_bank_list)

    for ib0 in ins_bank_list:
        for ib1 in ins_bank_list:
            for ib2 in ins_bank_list:
                comb = getCombKey(ib0, ib1, ib2)
                if comb not in comb_dict:
                    comb_dict[comb] = (ib0, ib1, ib2)
    return comb_dict

def genBankConflictInsSeq(i0, i1, i2):
    reglist = [[8,12,16],[9,13,17],[10,14,18],[11,15,19]]
    
    ilist = [i0, i1, i2, i0, i1, i2]
    regout = ['R%d'%x for x in range(20,26)]

    s = []
    for i_ins, ins_bank in enumerate(ilist):
        rs = []
        for ibank in range(4):
            rin = reglist[ibank]
            rs.extend(rin[0:ins_bank[ibank]])
        if len(rs)<3:
            rs.extend([rs[-1] for _ in range(3-len(rs))])

        regin = ', '.join(['R%d'%x for x in rs])
        s.append(f'FFMA {regout[i_ins]}, {regin};\n')
    
    return s

if __name__ == '__main__':
    comb_dict = buildCombDict()
    print('%d comb keys found!'%len(comb_dict))
    writeComb2File(comb_dict)

    keys = list(comb_dict.keys())
    keys.sort(reverse=True)
    for i,k in enumerate(keys):
        i0,i1,i2 = comb_dict[k]
        ks = getCombStr(i0, i1, i2)
        s = genBankConflictInsSeq(i0,i1,i2)
        print(f'#### {ks}')
        print(s)

        if i>100:
            break

            

