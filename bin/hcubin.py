#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import shutil

from CuAsm.utils.CubinUtils import fixCubinDesc
from CuAsm.CuAsmLogger import CuAsmLogger

desc_msg = '''
    Hack the sm8x cubin with valid cache-policy desc bit set.
    
    Currently the disassembly of nvdisasm will not show default cache-policy UR:

    /*00b0*/                   LDG.E R8, [R2.64] ;                      /* 0x0000000402087981 */
                                                                        /* 0x000ea8000c1e1900 */
    /*00c0*/                   LDG.E R9, desc[UR6][R2.64+0x400] ;       /* 0x0004000602097981 */
                                                                        /* 0x000ea8200c1e1900 */
    
    The first disassembly line should be `LDG.E R8, desc[UR4][R2.64] ;`,
    in which UR[4:5] is the default cache-policy UR and not showed, which may cause assembly confusion. 

    But if the 102th bit(the "2" in last line 0x000ea8200c1e1900) is set, 
    all cache-policy UR will be showed, that will complete the assembly input for the encoding. 
    
    This script will set that bit for every instruction that needs desc shown.
''' 

epilog_msg = '''
Examples:
    $ hcubin a.cubin
        hack a.cubin into a.hcubin, default output name is replacing the ext to .hcubin
    
    $ hcubin a.cubin -o x.bin
        hack a.cubin into x.bin

    $ hcubin a.cubin x.bin
        same as `hcubin a.cubin -o x.bin`
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

def hcubin(fin, fout=None):
    fbase, fext = os.path.splitext(fin)
    if fout is None:
        fout = fbase + '.hcubin'

    checkInFileExistence(fin)
    checkOutFileBackup(fout)

    return fixCubinDesc(fin, fout)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog='hcubin', formatter_class=argparse.RawDescriptionHelpFormatter, description=desc_msg, epilog= epilog_msg)
    parser.add_argument('infile', type=str, nargs='+', help='Input filename, should be a valid cubin file.')
    parser.add_argument('-o', '--output', dest='outfile', help='Output filename, infered from input filename if not given.')
    parser.add_argument('-f', '--logfile', dest='logfile', help='File name for saving the logs, default to none.')

    group_loglevel = parser.add_mutually_exclusive_group()
    group_loglevel.add_argument("-v", "--verbose", action="store_true", help='Verbose mode, showing almost every log.')
    group_loglevel.add_argument("-q", "--quiet", action="store_true", help='Quiet mode, no log unless errores found.')

    args = parser.parse_args()

    # parse src and dst filenames
    if len(args.infile) == 1:  # args.infile is always a list 
        infile = args.infile[0]
        outfile = args.outfile  # None if not set
    elif len(args.infile) == 2:
        infile = args.infile[0]
        outfile = args.infile[1]
    else:
        print('The infile should be of length 1 (second infered by replacing file extension) or 2 !!!')
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

    if not hcubin(infile, outfile):
        # not hacked
        exit(-1)
