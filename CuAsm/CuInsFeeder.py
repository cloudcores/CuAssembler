# -*- coding: utf-8 -*-

import re
from .common import c_InsLinePattern

class CuInsFeeder():
    def __init__(self, fstream, insfilter=''):
        if isinstance(fstream, str):
            self.m_FName = fstream
            self.m_FStream = open(self.m_FName, 'r')
        else:
            self.m_FName = None
            self.m_FStream = fstream

        self.m_InsFilter = re.compile(insfilter)

    def __iter__(self):
        return self

    def __next__(self):
        res = self.fetchIns()
        if res is not None:
            return res
        else:
            raise StopIteration

    def fetchIns(self):
        while True:
            line = self.m_FStream.readline()
            if len(line) == 0:
                return None

            rf = self.m_InsFilter.search(line)
            r = c_InsLinePattern.search(line)

            if rf is not None and r is not None:
                addr = int(r.groups()[0].strip(), 16)
                code = int(r.groups()[2].strip(), 16)
                s = r.groups()[1].strip()

                l2 = self.m_FStream.readline()
                r2 = re.search(r'\/\*(.*)\*\/', l2)
                c2 = int(r2.groups()[0].strip(), 16)
                ctrlcodes = c2>>41
                c2 = c2 & 0x1ffffffffff # strip 23bit control codes
                code += (c2<<64)

                return (addr, code, s, ctrlcodes)

    def __del__(self):
        '''Close the stream if the handler is owned by this feeder.'''

        if self.m_FName is not None and not self.m_FStream.closed:
            self.m_FStream.close()

    def close(self):
        if not self.m_FStream.closed:
            self.m_FStream.close()
            return True
        else:
            return False

    def restart(self):
        if self.m_FStream.seekable:
            self.m_FStream.seek(0)
        else:
            raise Exception("This feeder cannot be restarted!")

    def tell(self):
        '''Report the progress.'''

        return self.m_FStream.tell()
