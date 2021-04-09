# -*- coding: utf-8 -*-

from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuInsParser import CuInsParser

def test_sm61():
    fname = r'G:\Work\CuAssembler\TestData\CuTest\cudatest.sm_61.sass'
    feeder = CuInsFeeder(fname, arch='sm_61')

    for  addr, code, s, ctrlcodes in feeder:
        print('0x%04x :   0x%06x   0x%016x   %s'% (addr, ctrlcodes, code, s))

def test_sm75():
    cnt = 0
    fname = r'G:\Work\CuAssembler\TestData\CuTest\cudatest.sm_75.sass'
    feeder = CuInsFeeder(fname, arch='sm_75')

    cip = CuInsParser(arch='sm_75')

    for  addr, code, s, ctrlcodes in feeder:
        print('0x%04x :   0x%06x   0x%028x   %s'% (addr, ctrlcodes, code, s))

        ins_key, ins_vals, ins_modi = cip.parse(s, addr, code)
        print('    Ins_Key = %s'%ins_key)
        print('    Ins_Vals = %s'%str(ins_vals))
        print('    Ins_Modi = %s'%str(ins_modi))

        cnt += 1
        if cnt>10:
            break




if __name__ == '__main__':
    test_sm75()
    # test_sm61()

