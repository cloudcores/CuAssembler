#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import re
import shlex, shutil
from subprocess import CalledProcessError, check_output, DEVNULL, PIPE, STDOUT

usage_msg = '''
Usage: hnvcc args...

    hnvcc is the hacked wrapper of nvcc.
    The operation depends on the environment variable 'HNVCC_OP':
        Not-set or 'none' : call original nvcc
        'dump' : dump cubins to hack.fname.sm_#.cubin, backup existing files.
        'hack' : hack cubins with hack.fname.sm_#.cubin, skip if not exist 
        Others : error

    CAUTION:
        hnvcc hack/dump need to append options "-keep"/"-keep-dir" to nvcc.
        If these options are already in option list, hnvcc may not work right.

    Examples:
        $ hnvcc test.cu -arch=sm_75 -o test               
            call original nvcc

        $ HNVCC_OP=dump test.cu -arch=sm_75 -o test       
            dump test.sm_#.cubin to hack.test.sm_#.cubin

        $ HNVCC_OP=hack test.cu -arch=sm_75 -o test       
            hack test.sm_#.cubin with hack.test.sm_#.cubin
'''

# environment var for specify the op
HNVCC_OP = 'HNVCC_OP'

# temporary dir for intermediate files, will be deleted after dump/hack
KEEP_DIR = 'hnvcc_keep_dir'

# prefix for hacked cubin
HACK_PREFIX = 'hack'

# prefix for hacked cubin
DUMP_PREFIX = 'dump'

# pattern matching ENV var define line
p_VarLine = re.compile(r'#\$ \w+=')

def printUsage():
    print(usage_msg)


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
                print(f'Backup existing file {fname} to {bname}...')
                shutil.move(fname, bname)
            else:
                pass

def run(args, doCheck=True, outputToNull=True):
    try:
        out_b = check_output(args, stderr=STDOUT)
        out_s = out_b.decode().strip()
        res = 0, out_s
    except CalledProcessError as cpe:
        print(f'Error when running {args}')
        print(cpe.output.decode())
        res = -1, ''
    except Exception as e:
        print(f'Error when calling run(args) with args={args}!')
        print(str(e))
        res = -2, ''

    if doCheck and res[0] != 0 :
        raise Exception('Check failed! Abort...')
    
    # print(res)
    return res

def getCubinArg(cmd):
    '''Get cubin filename in ptxas args. '''
    for s in cmd[::-1]:
        # usually cubin is the last arg
        if s.endswith('.cubin'):
            return s
    
    return None

def doHackOrDump(args, op):
    # make args for dryrun, get command list
    args_dryrun = args.copy()
    args_dryrun[0] = 'nvcc'
    args_dryrun.extend(['-keep', f'-keep-dir={KEEP_DIR}', '-dryrun'])
    
    print('#### Getting command list...')
    _, out_s = run(args_dryrun) # get command list of nvcc
    if len(out_s)==0:
        raise Exception('Empty command list!!!')

    cmds = []

    for line in out_s.splitlines():
        sline = line.strip()
        if sline == '' or not sline.startswith('#$ ') or p_VarLine.match(sline):
            continue
        
        if (sline.startswith('#$ rm') or sline.startswith('#$ erase')) and sline.endswith('_dlink.reg.c'):
            continue

        cmds.append(shlex.split(sline[3:]))
    
    # make temp dir if not exist
    if os.path.isdir(KEEP_DIR):
        doDeleteKeepDir = False
    else:
        print(f'#### Creating keep dir {KEEP_DIR}')
        doDeleteKeepDir = True
        os.mkdir(KEEP_DIR)
    
    if op == 'hack':
        for cmd in cmds:
            if cmd[0] == 'ptxas':
                fullname = getCubinArg(cmd)
                cubinname = os.path.basename(fullname)
                hackname = HACK_PREFIX + '.' + cubinname
                if os.path.isfile(hackname):
                    print(f'#### Hacking with {hackname}...')
                    shutil.copy(hackname, fullname)
                else:
                    # run original ptxas if hack file does not exist
                    print(f'#### Keeping {cubinname}...')
                    run(cmd)
            else:
                run(cmd)
            
    elif op == 'dump':
        # strip useless commands after last ptxas
        # print(cmds)
        cmds_rev = cmds[::-1].copy()
        for cmd in cmds_rev:
            if cmd[0] != 'ptxas':
                cmds.pop()
            else:
                break
        
        # print(cmds)

        for cmd in cmds:
            if cmd[0] == 'ptxas':
                fullname = getCubinArg(cmd)
                cubinname = os.path.basename(fullname)
                dumpname = DUMP_PREFIX + '.' + cubinname
                if os.path.isfile(dumpname): # backup
                    shutil.copy(dumpname, dumpname+'~')
                
                run(cmd)
                print(f'#### Dumping {dumpname}...')
                shutil.copy(fullname, dumpname)
            else:
                run(cmd)

    else:
        raise Exception(f'Unknown op "{op}"')

    if doDeleteKeepDir:
        print(f'#### Removing keep dir {KEEP_DIR}...')
        shutil.rmtree(KEEP_DIR)

def hnvcc(args):
    # get op
    if HNVCC_OP not in os.environ:
        op = 'none'
    else:
        op = os.environ[HNVCC_OP].lower().strip()
        if op not in {'hack', 'dump', 'none'}:
            raise ValueError(f'Unknown HNVCC_OP "{op}"')

    if op == 'none':
        args[0] = 'nvcc'
        retcode, res = run(args)
    elif op in {'hack', 'dump'}:
        doHackOrDump(args, op)
    else:
        raise Exception('Never be here!')

if __name__ == '__main__':
    if os.name == 'nt':
        print('Sorry! This script (hnvcc) does not work right for windows...')
        exit(-1)
    
    if len(sys.argv) == 1:
        printUsage()
    elif len(sys.argv) == 2 and sys.argv[1]=='-h':
        printUsage()
    else:
        hnvcc(sys.argv)