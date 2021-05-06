# -*- coding: utf-8 -*-

import re
from .CuSMVersion import CuSMVersion

# Pattern that contains an instruction string (including address and code)
# NOTE: For maxwell/pascal, there may be braces "{}" for dual-issued instructions.
p_InsLinePattern = re.compile(r'^ *\/\*(\w+)\*\/ *\{? *(.*;) *\/\*(.*)\*\/')

# Pattern for the dual-issued instruction, the code is in next line.
p_InsLinePattern2 = re.compile(r'^ *\/\*(\w+)\*\/ *(.*\})')

#
p_PureCodeLinePattern = re.compile(r'^\s*\/\* (0x[0-9a-f]{16}) \*\/')

class CuInsFeeder():
    def __init__(self, fstream, arch='sm_75', linefilter=None):
        """ Construct a instruction feeder

        Args:
            fstream (str or file stream): file name or the file object
            arch (optional): SM version. Defaults to 'sm_75'.
            linefilter (optional): regex filter for lines. Usually used for feeding a perticular instruction.
                                Defaults to None.
        """

        if isinstance(fstream, str):
            self.__mFileName = fstream
            self.__mFStream = open(self.__mFileName, 'r')
                
        else:
            self.__mFileName = None
            self.__mFStream = fstream

        # compile line filter
        if linefilter is None or len(linefilter)==0:
            self.__mInsFilterFun = lambda x: True
        else:
            p = re.compile(linefilter)
            self.__mInsFilterFun = lambda x: p.search(x)

        self.__mCtrlCodes = None # buffered control codes, only for sm 5x/6x
        self.__mGroupIdx = 3 # 3 means all buffered control codes have consumed
        self.__mLineNo = 0
        self.__mSMVersion = CuSMVersion(arch)

        if self.__mSMVersion.getMajor() >= 7:
            self.fetchIns = self.fetchIns_sm7x_8x
        else:
            self.fetchIns = self.fetchIns_sm5x_6x

    def __iter__(self):
        return self

    def __next__(self):
        res = self.fetchIns()
        if res is not None:
            return res
        else:
            raise StopIteration

    def fetchIns_sm7x_8x(self):
        while True:
            line = self.readline()
            if len(line) == 0:
                return None

            r = p_InsLinePattern.search(line)
            if self.__mInsFilterFun(line) is not None and r is not None:
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

    def fetchIns_sm5x_6x(self):
        '''
                                                                             /* 0x00643c03fde01fef */
        /*0008*/                   MOV R1, c[0x0][0x20] ;                    /* 0x4c98078000870001 */
        /*0010*/                   IADD32I R1, R1, -0x170 ;                  /* 0x1c0fffffe9070101 */
        /*0018*/                   S2R R0, SR_LMEMHIOFF ;                    /* 0xf0c8000003770000 */'''

        while True:
            line = self.readline()
            if len(line) == 0:
                return None

            rc = p_PureCodeLinePattern.match(line)
            if rc:
                if self.__mGroupIdx != 3: #
                    raise Exception('Unexpected control code line encountered!')

                ccode = int(rc.groups()[0], 16)

                self.__mCtrlCodes = [(ccode & 0x00000000001fffff),
                                     (ccode & 0x000003ffffe00000) >> 21,
                                     (ccode & 0x7ffffc0000000000) >> 42]
                self.__mGroupIdx = 0
                continue

            # FIXME: handle splitted dual-issue lines: "{}"
            #        codes may appear in the second line after instruction text.
            r1 = p_InsLinePattern.match(line)
            r2 = p_InsLinePattern2.match(line)

            if r1:
                addr = int(r1.groups()[0].strip(), 16)
                code = int(r1.groups()[2].strip(), 16)
                s = r1.groups()[1].strip()
                ccode = self.__mCtrlCodes[self.__mGroupIdx]
                self.__mGroupIdx += 1

                if self.__mInsFilterFun(line) is not None: # or ('NAN' not in s):
                    return (addr, code, s, ccode)
                else:
                    continue
            elif r2:
                addr = int(r2.groups()[0].strip(), 16)
                s = r2.groups()[1].strip('{} ')
                ccode = self.__mCtrlCodes[self.__mGroupIdx]
                self.__mGroupIdx += 1

                line2 = self.readline()
                rc2 = p_PureCodeLinePattern.match(line2)
                if rc2:
                    code = int(rc2.groups()[0], 16)
                    return (addr, code, s, ccode)
                else:
                    raise Exception('The line following dual issue line is not a valid hex code line!')

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
        ''' A helper function for reading lines, with line number recorded.'''
        self.__mLineNo += 1
        return self.__mFStream.readline()

    def tell(self):
        '''Report the progress of file or stream.'''

        return self.__mFStream.tell()
    
    def tellLine(self):
        '''Report current line number.'''

        return self.__mLineNo
