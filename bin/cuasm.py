#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from CuAsm.CubinFile import CubinFile
from CuAsm.CuAsmParser import CuAsmParser
from CuAsm.CuAsmLogger import CuAsmLogger
import argparse

desc_msg = '''
    Convert cubin from/to cuasm files.

    NOTE 1: if the output file already exist, the original file will be renamed to "outfile~". 
    NOTE 2: if the logfile already exist, original logs will be rolled to logname.1, logname.2, until logname.3.
''' 

epilog_msg = '''
Examples:
    $ cuasm a.cubin
        disassemble a.cubin => a.cuasm, text mostly inherited from nvdisasm. If output file name is not given,
        the default name is replacing the ext to .cuasm

    $ cuasm a.cuasm 
        assemble a.cuasm => a.cubin. If output file name is not given, default to replace the ext to .cubin

    $ cuasm a.cubin -o x.cuasm
        disassemble a.cubin => x.cuasm, specify the output file explicitly
    
    $ cuasm a.cubin x.cuasm
        same as `cuasm a.cubin -o x.cuasm`

    $ cuasm a.o --bin2asm
        disassemble a.o => a.cuasm, file type with extension ".o" is not recognized.
        Thus conversion direction should be specified explicitly by "--bin2asm/--asm2bin".

    $ cuasm a.cubin -f abc -v 
        disassemble a.cubin => a.cuasm, save log to abc.log, and verbose mode
'''

def checkInFileExistence(fname):
    if os.path.isfile(fname):
        return True
    else:
        print(f'IOError! Input file "{fname}" not found!')
        exit(-1)

def checkOutFileBackup(fname, doBackup=True):
    if os.path.exists(fname):
        if os.path.isdir(fname):
            print(f'IOError!!! Output file "{fname}" is an existing directory!')
            exit(-1)
        else: # isfile
            if doBackup:
                bname = fname + '~'
                CuAsmLogger.logWarning(f'Backup existing file {fname} to {bname}...')
                shutil.move(fname, bname)
            else:
                pass

def cubin2cuasm(binname, asmname=None):
    if asmname is None:
        fbase, fext = os.path.splitext(binname)
        asmname = fbase + '.cuasm'

    cf = CubinFile(binname)
    checkOutFileBackup(asmname)
    cf.saveAsCuAsm(asmname)

def cuasm2cubin(asmname, binname=None):
    cap = CuAsmParser()
    cap.parse(asmname)

    if binname is None:
        fbase, fext = os.path.splitext(asmname)
        binname = fbase + '.cubin'
    
    checkOutFileBackup(binname)
    cap.saveAsCubin(binname)

def doProcess(src:str, dst:str, direction = 'auto'):
    ''' Do process from src to dst.
    
        direction:
            auto     : determined from src file extension (default)
            bin2asm  : cubin -> cuasm
            asm2bin  : cuasm -> cubin
    '''
    _, fext = os.path.splitext(src)
    
    if direction == 'bin2asm' or fext in {'.cubin', '.bin'}:
        cubin2cuasm(src, dst)
    elif direction == 'asm2bin' or fext in {'.cuasm', '.asm'}:
        cuasm2cubin(src, dst)
    else:
        print('The first infile should be with ext ".cubin" or ".cuasm", otherwise specify direction by option --bin2asm or --asm2bin!')
        exit(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='cuasm', formatter_class=argparse.RawDescriptionHelpFormatter, description=desc_msg, epilog= epilog_msg)
    parser.add_argument('infile', type=str, nargs='+', help='Input filename. If not with extension .cubin/.bin/.cuasm/.asm, direction option --bin2asm or --asm2bin should be specified.')
    parser.add_argument('-o', '--output', dest='outfile', help='Output filename, inferred from input filename if not given.')
    parser.add_argument('-f', '--logfile', dest='logfile', help='File name for saving the log, default to none.')

    group_loglevel = parser.add_mutually_exclusive_group()
    group_loglevel.add_argument("-v", "--verbose", action="store_true", help='Verbose mode, showing almost every log.')
    group_loglevel.add_argument("-q", "--quiet", action="store_true", help='Quiet mode, no log unless errores found.')

    group_direction = parser.add_mutually_exclusive_group()
    group_direction.add_argument("--bin2asm", action="store_true", help='Convert from cubin to cuasm.')
    group_direction.add_argument("--asm2bin", action="store_true", help='Convert from cuasm to cubin.')

    args = parser.parse_args()

    # determine convert direction
    if args.bin2asm:
        direction = 'bin2asm'
    elif args.asm2bin:
        direction = 'asm2bin'
    else:
        direction = 'auto'

    # parse src and dst filenames
    if len(args.infile) == 1:  # args.infile is always a list 
        infile = args.infile[0]
        outfile = args.outfile  # None if not set
    elif len(args.infile) == 2:
        infile = args.infile[0]
        outfile = args.infile[1]
    else:
        print('The infile should be of length 1 or 2 (second as output) !!!')
        print(f'    Input infile = {args.infile} with length {len(args.infile)} !!!')
        exit(-1)
    
    checkInFileExistence(infile)

    # parse stdout log level
    if args.verbose:
        stdout_level = 0
        file_level = 0
    elif args.quiet:
        stdout_level = 40 # ERROR
        file_level = 25
    else:
        stdout_level = 25 # PROCEDURE
        file_level = 15 # INFO

    if args.logfile is not None:
        CuAsmLogger.initLogger(args.logfile, file_level=file_level, stdout_level=stdout_level)
    else:
        CuAsmLogger.initLogger(log_file=None, stdout_level=stdout_level)

    doProcess(infile, outfile, direction)
