# -*- coding: utf-8 -*-

import re
from .CuSMVersion import CuSMVersion

# Pattern that contains an instruction string (including address and code)
# NOTE: For maxwell/pascal, there may be braces "{}" for dual-issued instructions.
p_InsLinePattern = re.compile(r'^ *\/\*(\w+)\*\/ *\{? *(.*;) *\}? *\/\*(.*)\*\/')
p_PureCodeLinePattern = re.compile(r'^\s*\/\* (0x[0-9a-f]{16}) \*\/')

class CuInsFeeder():
    def __init__(self, fstream, instfilter='', arch='sm_75'):
        if isinstance(fstream, str):
            self.__mFileName = fstream
            self.__mFStream = open(self.__mFileName, 'r')
        else:
            self.__mFileName = None
            self.__mFStream = fstream

        self.__mInstFilter = re.compile(instfilter)

        self.__mCtrlCodes = None
        self.__mGroupIdx = None
        self.__mLineNo = 0
        self.__mSMVersion = CuSMVersion(arch)

        if arch=='sm_75':
            self.fetchInst = self.fetchInst_sm75
        elif arch in ['sm_50', 'sm_52', 'sm_53', 'sm_60', 'sm_61', 'sm_62']:
            self.fetchInst = self.fetchInst_sm5x
        else:
            raise Exception("Unknown arch (%s)!" % arch)

    def __iter__(self):
        return self

    def __next__(self):
        res = self.fetchInst()
        if res is not None:
            return res
        else:
            raise StopIteration

    def fetchInst_sm75(self):
        while True:
            line = self.readline()
            if len(line) == 0:
                return None

            rf = self.__mInstFilter.search(line)
            r = p_InsLinePattern.search(line)

            if rf is not None and r is not None:
                addr = int(r.groups()[0].strip(), 16)
                code = int(r.groups()[2].strip(), 16)
                s = r.groups()[1].strip()

                l2 = self.readline()
                r2 = re.search(r'\/\*(.*)\*\/', l2)
                c2 = int(r2.groups()[0].strip(), 16)
                ctrlcodes = c2>>41
                # c2 = c2 & ((1<<41)-1) # strip 23bit control codes
                c2 = c2 & 0x1ffffffffff # strip 23bit control codes
                code += (c2<<64)

                return (addr, code, s, ctrlcodes)

    def fetchInst_sm5x(self):
        '''
                                                                             /* 0x00643c03fde01fef */
        /*0008*/                   MOV R1, c[0x0][0x20] ;                    /* 0x4c98078000870001 */
        /*0010*/                   IADD32I R1, R1, -0x170 ;                  /* 0x1c0fffffe9070101 */
        /*0018*/                   S2R R0, SR_LMEMHIOFF ;                    /* 0xf0c8000003770000 */'''

        while True:
            line = self.readline()
            if len(line) == 0:
                return None

            r1 = p_PureCodeLinePattern.match(line)
            if r1:
                ccode = int(r1.groups()[0], 16)

                self.__mCtrlCodes = [(ccode & 0x000000000001ffff),
                                     (ccode & 0x0000003fffe00000) >> 21,
                                     (ccode & 0x07fffc0000000000) >> 42]
                self.__mGroupIdx = 0
                continue

            # FIXME: handle splitted dual-issue lines: "{}"
            #        codes may appear in the second line after instruction text.
            r2 = p_InsLinePattern.search(line)
            if r2:
                addr = int(r2.groups()[0].strip(), 16)
                code = int(r2.groups()[2].strip(), 16)
                s = r2.groups()[1].strip()
                ccode = self.__mCtrlCodes[self.__mGroupIdx]
                self.__mGroupIdx += 1

                rf = self.__mInstFilter.search(line)
                if rf or ('NAN' in s):
                    return (addr, code, s, ccode)
                else:
                    continue

    def __del__(self):
        '''Close the stream if the handler is owned by this feeder.'''

        if self.__mFileName is not None and not self.__mFStream.closed:
            self.__mFStream.close()

    def close(self):
        if not self.__mFStream.closed:
            self.__mFStream.close()
            self.__mLineNo = 0
            return True
        else:
            return False

    def restart(self):
        if self.__mFStream.seekable:
            self.__mFStream.seek(0)
            self.__mLineNo = 0
        else:
            raise Exception("This feeder cannot be restarted!")

    def readline(self):
        self.__mLineNo += 1
        return self.__mFStream.readline()

    def tell(self):
        '''Report the progress.'''

        return self.__mFStream.tell()
    
    def tellLine(self):
        '''Report current line number.'''

        return self.__mLineNo
