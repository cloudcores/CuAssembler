#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from subprocess import CalledProcessError, check_output
import shutil
from io import StringIO
from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuAsmLogger import CuAsmLogger
from CuAsm.utils.CubinUtils import fixCubinDesc
from CuAsm.common import getTempFileName
import argparse

desc_msg = '''
    Format sass with control codes from input sass/cubin/exe/...

    The original dumped sass by `cuobjdump -sass *.exe` will not show scoreboard control codes, 
    which make it obscure to inspect the dependencies of instructions. 
    This script will extract the scoreboard info and show them with original disassembly. 

    CAUTION: the sass input should with exactly same format of `cuobjdump -sass`, otherwise
             the parser may not work correctly.

    NOTE 1: For cubins of sm8x, the cache-policy desc bit of some instruction will be set to 1
            to show desc[UR#] explicitly, other type of inputs(sass/exe/...) won't do the hack,
            which means some instructions may not be assembled normally as in cuasm files.
            This also implies for desc hacked sass, code of instructions may be not consistent either.

    NOTE 2: if the output file already exist, the original file will be renamed to "outfile~". 
    NOTE 3: if the logfile already exist, original logs will be rolled to log.1, log.2, until log.3.
''' 

epilog_msg = '''
Examples:
    $ dsass a.cubin
        dump sass from a.cubin, and write the result with control code to a.dsass

    $ dsass a.exe -o a.txt
        dump sass from a.cubin, and write the result with control code to a.txt

    $ dsass a.sass
        translate the cuobjdumped sass into a.dsass

    $ dsass a.cubin -f abc -v 
        convert a.cubin => a.dsass, save log to abc.log, and verbose mode

    $ dsass a.cubin -k
        usually lines with only codes in source sass will be ignored for compact output.
        use option -k/--keepcode to keep those lines. 
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

def dsass(fin, fout=None, keepcode=False, no_desc_hack=False):
    fbase, fext = os.path.splitext(fin)
    fext = fext.lower()
    if fout is None:
        fout = fbase + '.dsass'
    checkOutFileBackup(fout)

    if keepcode:
        codeonly_line_mode='keep'
    else:
        codeonly_line_mode='none'

    if fext == '.sass':
        feeder = CuInsFeeder(fin)
        CuAsmLogger.logEntry(f'Translating to {fout}...')
        feeder.trans(fout, codeonly_line_mode=codeonly_line_mode)

    elif fext == '.dsass':
        CuAsmLogger.logError(f'Input file "{fin}" is already a dsass file!!! Skipping...')
        exit(-1)
    elif fext == '.cubin':
        try:
            if no_desc_hack:
                doDescHack = False
                binname = fin
            else:
                tmpname = getTempFileName(suffix='cubin')
                doDescHack = fixCubinDesc(fin, tmpname) # , always_output=False
                if doDescHack:
                    binname = tmpname
                    CuAsmLogger.logWarning(f'Cubin {fin} needs desc hack!')
                else:
                    binname = fin
                
            CuAsmLogger.logEntry(f'Dumping sass from {binname}...')
            sass_b = check_output(['cuobjdump', '-sass', binname])
            sass = sass_b.decode()
            if doDescHack:
                os.remove(tmpname)
            
        except CalledProcessError as cpe:
            CuAsmLogger.logError('Error when running cuobjdump!' + cpe.output.decode())
            exit(-1)
        except Exception as e:
            CuAsmLogger.logError('DumpSass Error!' + str(e))
            exit(-1)
        
        sio = StringIO(sass)
        feeder = CuInsFeeder(sio)
        CuAsmLogger.logEntry(f'Translating to {fout} ...')
        feeder.trans(fout, codeonly_line_mode=codeonly_line_mode)        
    else: # default treat as bin
        try:
            CuAsmLogger.logEntry(f'Dumping sass from {fin}...')
            sass_b = check_output(['cuobjdump', '-sass', fin])
            sass = sass_b.decode()
        except CalledProcessError as cpe:
            CuAsmLogger.logError('Error when running cuobjdump!' + cpe.output.decode())
            exit(-1)
        except Exception as e:
            CuAsmLogger.logError('DumpSass Error!' + str(e))
            exit(-1)
        
        sio = StringIO(sass)
        feeder = CuInsFeeder(sio)
        CuAsmLogger.logEntry(f'Translating to {fout} ...')
        feeder.trans(fout, codeonly_line_mode=codeonly_line_mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='dsass', formatter_class=argparse.RawDescriptionHelpFormatter, description=desc_msg, epilog= epilog_msg)
    parser.add_argument('infile', type=str, nargs='+', help='Input filename, can be dumped sass, cubin, or binary contains cubin.')
    parser.add_argument('-o', '--output', dest='outfile', help='Output filename, infered from input filename if not given.')
    parser.add_argument('-k', '--keepcode', action="store_true", help='Keep code-only lines in input sass, default to strip.')
    parser.add_argument('-n', '--nodeschack', action="store_true", help='Do not hack desc bit, no matter SM version it is.')
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
        print('The infile should be of length 1 or 2 (second as output)!!!')
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

    dsass(infile, outfile, keepcode=args.keepcode, no_desc_hack=args.nodeschack)
