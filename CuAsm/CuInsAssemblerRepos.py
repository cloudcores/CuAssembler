# -*- coding: utf-8 -*-

import re
import time
import struct
from .common import decodeCtrlCodes, encodeCtrlCodes
from .CuInsParser import CuInsParser
from .CuInsAssembler import CuInsAssembler

from io import StringIO

class CuInsAssemblerRepos():
    def __init__(self, InsAsmDict={}):
        self.reset(InsAsmDict)

    def reset(self, InsAsmDict={}):
        self.m_InsAsmDict = InsAsmDict

    def initFromFile(self, fname):
        with open(fname,'r') as fin:
            fconts = fin.read()
            asm_repos = eval(fconts)
            self.m_InsAsmDict = asm_repos.m_InsAsmDict

    def assemble(self, addr, s, ctrl):
        ins = CuInsParser()
        ins_key, ins_vals, ins_modi = ins.parse(s, addr, 0)
        if ins_key not in self.m_InsAsmDict:
            raise KeyError('Unknown ins_key(%s) in Repos!' % ins_key)

        insAsm = self.m_InsAsmDict[ins_key]
        code = insAsm.buildCode(ins_vals, ins_modi)

        if isinstance(ctrl, str):
            ctrlcodes = encodeCtrlCodes(ctrl)
        elif isinstance(ctrl, int):
            ctrlcodes = ctrl
        return code + (ctrlcodes << 105)

    def verify(self, feeder):
        res = True
        t0 = time.time()
        cnt = 0
        for addr, code, s, ctrlcodes in feeder:
            cnt += 1
            try:
                casm = self.assemble(addr, s, ctrlcodes)
                fcode = code + (ctrlcodes<<105)
                if fcode != casm:
                    print('Error when verifying :')
                    print('  ' + s)
                    print('  CodeOrg: %032x'%fcode)
                    print('  CodeAsm: %032x'%casm)
                    # raise Exception('Assembled code not match!')
            except:
                print('Error when assembling :')
                print('  ' + s)
                res = False

        t1 = time.time()

        if res:
            print("Verified %d ins in %8.3f secs." % (cnt, t1-t0))
            if (t0==t1):
                print("  ~Unknown ins per second." )
            else:
                print("  ~%8.2f ins per second." % (cnt/(t1-t0)))
        else:
            print("Verifying failed in %8.3f secs!!!" % (t1-t0))

        return res

    def update(self, feeder):
        ins = CuInsParser()
        t0 = time.time()
        cnt = 0
        for addr, code, s, ctrlcodes in feeder:
            cnt += 1
            ins_key, ins_vals, ins_modi = ins.parse(s, addr, code)

            if ins_key not in self.m_InsAsmDict:
                self.m_InsAsmDict[ins_key] = CuInsAssembler(ins_key)

            res = self.m_InsAsmDict[ins_key].push(ins_vals, ins_modi, code)

            if not res:
                print('  Str :' + s)
                print('  Addr:' + str(addr))
                raise Exception("Unmatched codes!")

        t1 = time.time()
        print("Updated %d ins in %8.3f secs." % (cnt, t1-t0))
        if (t0==t1):
            print("  ~Unknown ins per second." )
        else:
            print("  ~%8.2f ins per second." % (cnt/(t1-t0)))

    def save2file(self, fname):
        with open(fname,'w') as fout:
            fout.write(self.__repr__())

    def __repr__(self):
        sio = StringIO()

        sio.write('CuInsAssemblerRepos(')
        sio.write(repr(self.m_InsAsmDict))
        sio.write(')')

        return sio.getvalue()

    def __str__(self):
        return "CuInsAssemblerRepos(%d keys)" % len(self.m_InsAsmDict)
