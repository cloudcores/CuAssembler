# -*- coding: utf-8 -*-

import re
import time
import struct
from .common import decodeCtrlCodes, encodeCtrlCodes, reprDict, reprList

from .CuInsParser import CuInsParser
from .CuInsAssembler import CuInsAssembler
from .CuSMVersion import CuSMVersion
from .CuAsmLogger import CuAsmLogger
from .config import Config

from io import StringIO
from sympy import Matrix



class CuInsAssemblerRepos():
    ''' A repository consists of a set of instruction assemblers.

        TODO: Version control? Should work with CuInsParser/CuInsFeeder.
    '''

    def __init__(self, InsAsmDict={}, arch='sm_75'):
        
        self.arch = arch
        self.__mSMVersion = CuSMVersion(arch)
        self.m_InsParser = CuInsParser(arch)

        if isinstance(InsAsmDict, str):
            self.initFromFile(InsAsmDict)
        elif isinstance(InsAsmDict, dict):
            self.reset(InsAsmDict)
        else:
            raise ValueError('Unknown input type of InsAsmDict!')
            
    def setToDefaultInsAsmDict(self):
        fname = Config.getDefaultInsAsmReposFile(self.__mSMVersion.getVersionNumber())
        self.initFromFile(fname)
    
    def reset(self, InsAsmDict={}):
        self.m_InsAsmDict = InsAsmDict

    def initFromFile(self, fname):
        with open(fname,'r') as fin:
            fconts = fin.read()
            asm_repos = eval(fconts)
            self.m_InsAsmDict = asm_repos.m_InsAsmDict

    def assemble(self, addr, s, precheck=True):
        ins_key, ins_vals, ins_modi = self.m_InsParser.parse(s, addr, 0)
        if ins_key not in self.m_InsAsmDict:
            raise KeyError('Unknown ins_key(%s) in Repos!' % ins_key)

        insAsm = self.m_InsAsmDict[ins_key]
        if precheck:
            brief, info = insAsm.canAssemble(ins_vals, ins_modi)
            if brief is not None:
                raise ValueError('Assembling failed (%s): %s'%(brief, info))

        code = insAsm.buildCode(ins_vals, ins_modi)
        return code

    @CuAsmLogger.logTimeIt
    def verify(self, feeder):
        res = True
        t0 = time.time()
        cnt = 0
        for addr, code, s, ctrlcodes in feeder:
            cnt += 1
            try:
                casm = self.assemble(addr, s)
                if code != casm:
                    print('Error when verifying :')
                    print('  ' + s)
                    print('  CodeOrg: %032x'%code)
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

    @CuAsmLogger.logTimeIt
    def update(self, feeder, ins_asm_dict=None):
        ''' Update the input instruction assembler dict with input from feeder.

            For ins_asm_dict=None, use the internal self.m_InsAsmDict as dst.
        '''
        if ins_asm_dict is None:
            ins_asm_dict = self.m_InsAsmDict

        t0 = time.time()
        cnt = 0
        for addr, code, s, ctrlcodes in feeder:
            cnt += 1
            ins_key, ins_vals, ins_modi = self.m_InsParser.parse(s, addr, code)
            # print('%#6x : %s'%(addr, s))

            if ins_key not in ins_asm_dict:
                ins_asm_dict[ins_key] = CuInsAssembler(ins_key)

            ins_info = (addr, code, s)
            res = ins_asm_dict[ins_key].push(ins_vals, ins_modi, code, ins_info)

            if not res:
                print("ERROR!!! Unmatched codes!")
                print('  Str : ' + s)
                print('  Addr: %#6x'%addr)
                print(repr(ins_asm_dict[ins_key]))

        t1 = time.time()
        print("Updated %d ins in %8.3f secs." % (cnt, t1-t0))
        if (t0==t1):
            print("  ~Unknown ins per second." )
        else:
            print("  ~%8.2f ins per second." % (cnt/(t1-t0)))

    @CuAsmLogger.logTimeIt
    def save2file(self, fname):
        with open(fname, 'w') as fout:
            fout.write(self.__repr__())

    @CuAsmLogger.logTimeIt
    def rebuild(self):
        ''' When the CuInsParser is updated, the meaning of ins value/modifier may have changed.
        
            Thus CuInsAsmRepos should be rebuilt from original input (saved in ins records)
            TODO: We may store some redundant records?

        '''

        tmp_ins_asm_dict = {}
        feeder = self.iterRecords()

        self.update(feeder, tmp_ins_asm_dict)
        self.m_InsAsmDict = tmp_ins_asm_dict
    
    @CuAsmLogger.logTimeIt
    def merge(self, merge_source):
        ''' Merge instruction assembler from another source.

            TODO: Check version?
        '''
        if isinstance(merge_source, (str,dict)):
            repos = CuInsAssemblerRepos(merge_source)
        elif isinstance(merge_source, CuInsAssemblerRepos):
            repos = merge_source
        else:
            raise TypeError('Unknown merge source type!')
        
        feeder = repos.iterRecords()
        self.update(feeder)

    def iterRecords(self, ins_asm_dict=None):
        ''' A generator as internal instruction feeder, pulling from instruction records.
        
            The output should be with same format as CuInsFeeder.
        '''
        if ins_asm_dict is None:
            ins_asm_dict = self.m_InsAsmDict

        for ins_key, ins_asm in ins_asm_dict.items():
            for r in ins_asm.m_InsRecords:
                yield r[0], r[1], r[2], 0 # control codes in last pos

    def completePredCodes(self):
        ''' Some instructions seem very rarely appear with guard predicates.

            Thus when the instruction assemblers are gathered from ptxas output, 
            many of them will not able to encode predicates.

            This may give some useful infomation as performance guidelines.
            However, there will be certainly some occasions predicates will be needed.
        '''

        feeder = self.genPredRecords()
        self.update(feeder)

    def genPredRecords(self):
        ''' A generator that yields modified instruction info with predicates. '''
        for ins_key, ins_asm in self.m_InsAsmDict.items():
            ins_info = ins_asm.m_InsRecords[0]
            pred_ins_info = self.__mSMVersion.genPredCode(ins_info)
            if pred_ins_info is not None:
                yield pred_ins_info[0], pred_ins_info[1], pred_ins_info[2], 0

    def __repr__(self):
        sio = StringIO()

        sio.write('CuInsAssemblerRepos(')
        reprDict(sio, self.m_InsAsmDict)
        sio.write(', arch=%s)'%self.__mSMVersion)

        return sio.getvalue()

    def __str__(self):
        return "CuInsAssemblerRepos(%d keys)" % len(self.m_InsAsmDict)
