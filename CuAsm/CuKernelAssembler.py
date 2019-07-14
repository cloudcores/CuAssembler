# -*- coding: utf-8 -*-

import re
import struct
from .CuInsAssemblerRepos import CuInsAssemblerRepos

from .common import *
from io import BytesIO

class CuKernelAssembler():
    def __init__(self, insasmrepos=None):
        if insasmrepos is None:
            self.m_InsAsmRepos = None
        elif isinstance(insasmrepos, str):
            self.initInsAsmRepos(insasmrepos)
        elif isinstance(insasmrepos, CuInsAssemblerRepos):
            self.m_InsAsmRepos = insasmrepos
        else:
            raise Exception("Unknown input for CuKernelAssembler!")

    def initInsAsmRepos(self, fname):
        with open(fname,'r') as fin:
            fconts = fin.read()
            self.m_InsAsmRepos = eval(fconts)

    def assembleCuAsm(self, cuasmcode):
        m_ins = c_CuAsmInsLine
        m_comment = c_CuAsmSkipLine

        if not isinstance(self.m_InsAsmRepos, CuInsAssemblerRepos):
            raise Exception("InsAsmRepos is not properly initialized!")

        labeldict = {}
        inslist = []

        counter = 0
        # first scan to gather instructions and labels
        for line in cuasmcode:
            # print('parsing '+line)
            if m_comment.match(line):
                continue
            else:
                res = m_ins.match(line)
                if res is None:
                    raise Exception("Unknown code line: %s" % line)

                label = res.groups()[0].strip(':')
                if label is not None:
                    if label in labeldict:
                        raise Exception("Redefined label in line:%s" % line)
                    labeldict[label] = counter

                inslist.append((counter, res.groups()[1], res.groups()[2]))
                counter += 1

        # second scan to encode the instructions
        # codes = []
        kcode = BytesIO()
        for idx, ctrlstr, ins in inslist:
            addr = idx * 0x10
            insstr = self.transLabel(ins, labeldict)
            code = self.m_InsAsmRepos.assemble(addr, insstr, ctrlstr)
            cbytes = struct.pack('QQ', code & ((1<<64)-1), code>>64)
            kcode.write(cbytes)

        return kcode.getvalue()

    #def __repr__(self):
    #    pass

    @staticmethod
    def transLabel(s, ldict):
        ''' Translate `Label` to the ins address.'''

        m = re.compile(r'`(L\w+)`')
        res = m.search(s)
        while res is not None:
            label = res.groups()[0]
            if label not in ldict:
                print(s)
                raise Exception('Unknown label %s' % label)

            addr = ldict[label] * 0x10

            s = s.replace(res.group(), '0x%x' % addr)

            res = m.search(s)

        return s

    @staticmethod
    def ParseNVInfo(binfo):
        #da = np.frombuffer(binfo, dtype='uint32')

        curr = 0

        attrlist = []

        bio = BytesIO(binfo)
        b = bio.read(4)
        while len(b)>0:
            v = struct.unpack('I', b)
            key = v & 0xffff
            t = v & 0xff
            l = v>>16

            curr += 1
            if t==1: # EIFMT_NVAL
                val = []
            elif t==2: # EIFMT_BVAL
                val = []
            elif t==3: # EIFMT_HVAL
                val = []
            elif t==4: # EIFMT_SVAL
                curr += l//4
            else:
                raise Exception("Unknwon EIFMT(%d) in nv.info!" % t)
            attrlist.append((key, val))

            b = bio.read(4)

