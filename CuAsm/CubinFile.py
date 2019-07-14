# -*- coding: utf-8 -*-

from elftools.elf.elffile import ELFFile
from elftools.elf.structs import ELFStructs

from subprocess import check_output
from .Turing_Assembler import CuInsFeeder, CuInsParser, CuKernelAssembler
from io import StringIO
import re
from hashlib import blake2b

import logging
import sys

# logging.basicConfig(filename="CubinFile.log", filemode="w",
#     format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
#     datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

class CubinFile():
    def __init__(self):
        self.clear()
        self.m_CuKernelAsm = None

    def clear(self):
        self.m_ELFStructs = ELFStructs(little_endian=True, elfclass=64)
        self.m_ELFStructs.create_basic_structs()
        self.m_ELFStructs.create_advanced_structs()

        self.m_ELFFileHeader = None
        self.m_ELFSections = []
        self.m_ELFSegments = []
        self.m_CodeDict = {}

    def initCuKernelAsm(self, asmreposname="InsAsmRepos.txt"):
        self.m_CuKernelAsm = CuKernelAssembler(asmreposname)

    def loadCubin(self, binname):
        self.clear()
        with open(binname, 'rb') as fin:
            ef = ELFFile(fin)
            self.m_ELFStructs = ef.structs
            self.m_ELFFileHeader = ef.header

            for sec in ef.iter_sections():
                self.m_ELFSections.append((sec.name, sec.header, sec.data()))

            for seg in ef.iter_segments():
                self.m_ELFSegments.append(seg.header)

        self.__loadCubinCode(binname)

    def __loadCubinCode(self, binname):
        sass = check_output("cuobjdump -sass "+binname).decode()
        sasslines = sass.splitlines()

        kname_pattern = re.compile(r'^\s*Function : (\w+)\b')
        kstart_pattern = re.compile(r'^\s*\.headerflags\b')
        kend_pattern = re.compile(r'^\s*\.+\s*$')

        kdict = {}
        kname = ''
        kconts = ''
        doGather = False
        for line in sasslines:
            res = kname_pattern.match(line)
            if res is not None:
                kname = res.groups()[0]
                # print('NewKernel: ' + kname)
                continue
            elif kstart_pattern.match(line):
                doGather = True
            elif kend_pattern.match(line):
                doGather = False
                kdict[kname] = kconts
                kconts = ''

            if doGather:
                kconts += line + '\n'

        self.m_CodeDict = self.transKernelCode(kdict)

    @staticmethod
    def transKernelCode(kdict):
        tdict = {}
        for k in kdict:
            codes = kdict[k] #.splitlines()
            sio = StringIO(codes)
            insfeeder = CuInsFeeder(sio)
            kins = [ins for ins in insfeeder]
            tdict[k] = kins

        return tdict

    @staticmethod
    def genKernelText(tdict):
        sio = StringIO()
        #sio.write('\n')
        spattern = re.compile('^\s*(@\S+)?\s*(.*;)\s*$')
        for addr, code, s, ctrlcodes in tdict:
            fullcode = code + (ctrlcodes<<105)
            c0 = fullcode & ((1<<64) - 1)
            c1 = fullcode >> 64
            cstr = CuKernelAssembler.decodeCtrlCodes(ctrlcodes)

            # align predicate and main part of instructions
            res = spattern.match(s)
            spred = res.groups()[0]
            if spred is None:
                spred = ''
            smain = res.groups()[1]

            sio.write('L%05x: [%s]  %6s %-72s /* 0x%016x; 0x%016x */\n'
                      % (addr, cstr, spred, smain, c0, c1))
        return sio.getvalue()

    @staticmethod
    def bytes2hex(bs, linewidth=32):
        length = len(bs)
        if length == 0:
            return ''

        sio = StringIO()
        for i in range(0, length, linewidth):
            sio.write(bs[i:min(i+linewidth, length)].hex() + '\n')

        return sio.getvalue()

    def __hexsign(self, bytes):
        h = blake2b(digest_size=16, key=b'Cubin ELF Secret Key')
        h.update(bytes)
        return h.hexdigest()

    @staticmethod
    def dict2comment(d):
        sio = StringIO()
        for k in d:
            sio.write('# %16s : %s\n' %(k, d[k]))
        return sio.getvalue()

    def saveAsCuAsm(self, asmname):
        with open(asmname, 'w+') as fout:
            # output file header
            fout.write('# file header \n')
            h2 = self.m_ELFFileHeader.copy()
            fhcont = self.m_ELFStructs.Elf_Ehdr.build(h2)
            fhcont_hex = self.bytes2hex(fhcont)
            fhsign = self.__hexsign(fhcont)

            fout.write('.FileHeader {"Signature":"%s", "Type":"Data"}\n' % fhsign)
            fout.write(self.dict2comment(self.m_ELFFileHeader))
            fout.write(fhcont_hex + '\n')

            # output sections
            for name, header, data in self.m_ELFSections:
                shcont = self.m_ELFStructs.Elf_Shdr.build(header)
                shcont_hex = self.bytes2hex(shcont)
                shcont_sign = self.__hexsign(shcont)

                fout.write('# section %s\n' % name)
                fout.write('.SectionHeader {"Signature":"%s", "Type":"Data", "Name":"%s"}\n'
                           % (shcont_sign, name) )
                fout.write(self.dict2comment(header))
                fout.write(shcont_hex + '\n')

                if name.startswith('.text.'):
                    kname = name.replace('.text.','')
                    codes = self.m_CodeDict[kname]
                    shdata_hex = self.genKernelText(codes)
                    dtype="Code"
                else:
                    shdata_hex = self.bytes2hex(data)
                    dtype="Data"

                shdata_sign = self.__hexsign(data)
                fout.write('.SectionData {"Signature":"%s", "Type":"%s"}\n' % (shdata_sign, dtype))
                fout.write(shdata_hex + '\n')

            # output segments
            for segheader in self.m_ELFSegments:
                seghcont = self.m_ELFStructs.Elf_Phdr.build(segheader)
                seghcont_hex = self.bytes2hex(seghcont)

                seghcont_sign = self.__hexsign(seghcont)
                fout.write('# segment \n')
                fout.write('.SegmentHeader {"Signature":"%s", "Type":"Data"}\n' % seghcont_sign)
                fout.write(self.dict2comment(segheader))
                fout.write(seghcont_hex + '\n')

    def loadFromCuAsm(self, asmname):
        self.clear()

        if self.m_CuKernelAsm is None:
            raise Exception("No kernel assembler is available!")

        secheader_list = []
        secname_list = [] #
        secdata_list = []

        segheader_list = []

        m_comment = re.compile(r'^\s*(#.*)?$') # comment or empty line
        m_directive = re.compile(r'^\s*(\.\w+)\s+(\{.*\})') # directive (startwith ".")

        skipfun = lambda s: m_comment.match(s) is not None
        stopfun = lambda s: m_directive.match(s) is not None

        with open(asmname, 'r') as fin:
            lines = fin.readlines()

        idx = 0
        while idx<len(lines):
            line = lines[idx]
            if m_comment.match(line) is not None: # skip comment
                idx += 1
                continue
            else:
                res = m_directive.match(line)
                if res is not None:
                    directive = res.groups()[0]
                    attrs = eval(res.groups()[1])
                    if attrs['Type'] == 'Data':
                        data, idx = self.gatherData(lines, idx+1, skipfun, stopfun)
                        if self.__hexsign(data) != attrs['Signature']:
                            # print(self.__hexsign(data))
                            # print(attrs['Signature'])
                            raise Exception('%s:%d Data signature not match!' % (asmname, idx))
                    elif attrs['Type'] == 'Code':
                        codes, idx = self.gatherCode(lines, idx+1, skipfun, stopfun)
                        data = self.m_CuKernelAsm.assembleCuAsm(codes)
                    else:
                        raise Exception('%s:%d Unknown directive with attribute Type="%s"!' % (asmname, idx, attrs['Type']))

                    if directive == '.FileHeader':
                        logging.info("%s:%d Loading file header..." % (asmname, idx))
                        self.m_ELFFileHeader = self.m_ELFStructs.Elf_Ehdr.parse(data)
                    elif directive == '.SectionHeader':
                        logging.info("%s:%d Loading section header..." % (asmname, idx))
                        if len(secheader_list) != len(secdata_list):
                            raise Exception('%s:%d Unmatched section header!' %(asmname, idx))
                        secheader_list.append(data)
                        secname_list.append(attrs['Name'])
                    elif directive == '.SectionData':
                        logging.info("%s:%d Loading section data..." % (asmname, idx))
                        if len(secheader_list)-len(secdata_list) != 1:
                            raise Exception('%s:%d Unmatched section data!' %(asmname, idx))
                        secdata_list.append(data)
                    elif directive == '.SegmentHeader':
                        logging.info("%s:%d Loading segment section..." % (asmname, idx))
                        if len(segheader_list)>3:
                            raise Exception('%s:%d Extra segment header!' %(asmname, idx))
                        segheader_list.append(data)
                    else:
                        raise Exception('Unknown directive %s' % directive)
                else: # something wrong here, gatherCode/gatherData should stop at directives
                    raise Exception("Directive not found!")

        # gather section headers and data
        self.m_ELFSections = [(sname, self.m_ELFStructs.Elf_Shdr.parse(sh), sd)
                                for sname, sh, sd in zip(secname_list, secheader_list, secdata_list)]
        self.m_ELFSegments = [self.m_ELFStructs.Elf_Phdr.parse(sh) for sh in segheader_list ]

        # do layouts

    @staticmethod
    def gatherData(lines, idx, skipfun, stopfun):
        data = b''
        for i in range(idx, len(lines)):
            if skipfun(lines[i]):
                continue
            elif stopfun(lines[i]):
                break
            else:
                data += bytearray.fromhex(lines[i])

        return data, i

    @staticmethod
    def gatherCode(lines, idx, skipfun, stopfun):
        codes = []
        for i in range(idx, len(lines)):
            if skipfun(lines[i]):
                continue
            elif stopfun(lines[i]):
                break
            else:
                codes.append(lines[i])
        return codes, i

    def saveAsCubin(self, binname):
        with open(binname,'wb') as fout:
            # write file identifier and elf header
            fheader = self.m_ELFStructs.Elf_Ehdr.build(self.m_ELFFileHeader)
            logging.info("Writing file header (%8d @ %8d)."%(len(fheader), fout.tell()))
            fout.write(fheader)

            # write datas
            for name, header, data in self.m_ELFSections:
                align = header['sh_addralign']
                curpos = fout.tell()
                if align>0 and curpos % align != 0:
                    padlen = ( (curpos+align-1) // align)*align
                    npad = padlen - curpos
                    logging.info("Writing section pad data (%8d @ %8d)."%(npad, fout.tell()))
                    fout.write(b'\x00' * npad)

                logging.info("Writing section data (%8d @ %8d)."%(len(data),fout.tell()))
                logging.info("   HeaderPos: %8d." % header['sh_offset'])
                fout.write(data)

            # write section header
            for name,header,data in self.m_ELFSections:
                fdata = self.m_ELFStructs.Elf_Shdr.build(header)
                logging.info("Writing section header (%8d @ %8d)."%(len(fdata),fout.tell()))
                fout.write(fdata)

            # write segment header
            for header in self.m_ELFSegments:
                fdata = self.m_ELFStructs.Elf_Phdr.build(header)
                logging.info("Writing program header (%8d @ %8d)."%(len(fdata),fout.tell()))
                fout.write(fdata)

            fout.close()


if __name__ == '__main__':

    fdir = 'G:\\Temp\\CubinTest\\'
    binname = r'D:\Programs\VisualStudio\CubinProbe\CubinProbe\x64\Release\kernel.compute_75.sm_75.cubin'

    from glob import iglob
    for binname in iglob(fdir + "*.sm_75.cubin"):
        print('Processing %s...' % binname)
        cf = CubinFile()
        cf.initCuKernelAsm()
        cf.loadCubin(binname)

        asmname = binname.replace('.cubin', '.cuasm')
        print('Saving to %s...' % asmname)
        cf.saveAsCuAsm(asmname)

        cf2 = CubinFile()
        cf2.initCuKernelAsm()

        print('Loading from %s...' % asmname)
        cf2.loadFromCuAsm(asmname)

        binname2 = binname + '2'
        print('Saving %s...' % binname2)
        cf2.saveAsCubin(binname2)

