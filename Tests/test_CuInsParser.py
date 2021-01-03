# -*- coding: utf-8 -*-

from CuAsm.CuInsParser import CuInsParser

cip = CuInsParser(arch='sm_61')
res = cip.parse('FFMA R9, R3.reuse, -0.5, R2.reuse ;')

print(res)
print(cip)
