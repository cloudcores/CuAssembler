import os
import sys
import re
from enum import IntEnum, auto, unique

# name of env var for enable ptxas hack
# the contents of the var is the file path to the hack mapping file
PTXAS_HACK = 'PTXAS_HACK'
PTXAS_PATH = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\bin\\ptxas_orig.exe'

@unique
class HackErrorCode(IntEnum):
    # Skipable errors
    HackSuccess = 0
    HackNotEnabled = auto()
    HackMapIsEmpty = auto()
    HackPTXNotMatch = auto()

    # errors before seperator can be safely skipped, original ptxas will be called;
    # errors after should be treated as fatal error.
    HackErrorSeperator = auto()

    #
    UnknownPTXFileName = auto()
    UnknownCubinFileName = auto()
    InputPTXFileNotFound = auto()
    HackMapFileNotFound = auto()
    ReplaceFileNotFound = auto()
    UnknownRuntimeError = auto()

def parseArgs(args):

    arch = None
    ptxpath = None
    binpath = None

    for iarg, arg in enumerate(args):
        larg = arg.lower()
        if larg.endswith('.ptx'):
            ptxpath = arg
        elif larg == '-o':
            binpath = args[iarg+1]
        elif larg.startswith('-arch='):
            arch = larg.split('=')[1]
        elif larg.startswith('-arch'):
            arch = args[iarg+1].lower()
    
    return arch, ptxpath, binpath

def checkPTXFile(arch:str, ptxpath:str, pf_list:list):
    '''
        ptxpath: input ptx file path
        pf_list: list of 
    '''

    if pf_list is None or len(pf_list)==0:
        return None
    
    ptxdir, ptxname = os.path.split(ptxpath)

    with open(ptxpath, 'r') as fin:
        ptxlines = fin.read()

    for rule in pf_list:
        # once mismatched, go to next rule
        if 'arch' in rule:
            if rule['arch'] != arch:
                continue

        if 'ptxname' in rule:
            p = re.compile(rule['ptxname'])
            if p.search(ptxname) is None:
                continue

        if 'contents' in rule:
            p = re.compile(rule['contents'])
            if p.search(ptxlines) is None:
                continue
        
        return rule['replacement']

    return None

def buildPFList(pf_file):
    with open(pf_file, 'r') as fin:
        import ast
        lines = fin.read()
        pf_list = ast.literal_eval(lines)

    validkeyset = set(['arch', 'ptxname', 'contents', 'replacement'])

    for i, rule in enumerate(pf_list):
        if 'replacement' not in rule:
            raise KeyError('No replacement file given for rule %d (%s)'%(i+1, str(rule)) )
        if len(rule)==1:
            raise KeyError('No match rule defined for rule %d (%s)'%(i+1, str(rule)))

        rulekeyset = set(rule.keys())
        if not validkeyset.issuperset(rulekeyset):
            raise KeyError('Invalid key (%s) for rule %d (%s)'%
                (str(rulekeyset-validkeyset), i+1, str(rule)))

    # print(pf_list)
    return pf_list

def doFileCopy(src, dst):
    from shutil import copyfile
    copyfile(src, dst)

def runPTXAS(args):
    # print('Args = %s'%str(args))
    from subprocess import run
    if args is not None and len(args)>0:
        fargs = [PTXAS_PATH]
        fargs.extend(args)
        res = run(fargs)
    else:
        res = run(PTXAS_PATH)
    return res.returncode

def main():
    if PTXAS_HACK not in os.environ or len(os.environ[PTXAS_HACK].strip(' '))==0:
        return HackErrorCode.HackNotEnabled

    arch, ptxname, binname = parseArgs(sys.argv[1:])
    if ptxname is None:
        return HackErrorCode.UnknownPTXFileName
    elif binname is None:
        return HackErrorCode.UnknownCubinFileName
    
    ptx_hackmap_file = os.environ[PTXAS_HACK].strip(' ')
    if not os.path.isfile(ptx_hackmap_file):
        print('Hack map file "%s" is not found!'%ptx_hackmap_file)
        return HackErrorCode.HackMapFileNotFound
        
    pf_list = buildPFList(ptx_hackmap_file)
    if len(pf_list) == 0:
        return HackErrorCode.HackMapIsEmpty

    if not os.path.isfile(ptxname):
        print('Input ptx file "%s" is not found!'%ptxname)
        return HackErrorCode.InputPTXFileNotFound

    repname = checkPTXFile(arch, ptxname, pf_list)

    if repname is None:
        return HackErrorCode.HackPTXNotMatch
    
    if not os.path.isfile(repname):
        print('The replacement cubin "%s" for input ptx "%s" is not found!'%(repname, ptxname))
        return HackErrorCode.ReplaceFileNotFound

    print('Hacking "%s" with "%s"...'%(binname, repname))
    doFileCopy(repname, binname)
    return HackErrorCode.HackSuccess

if __name__ == '__main__':
    try:
        code = main()

        if code > HackErrorCode.HackErrorSeperator:
            # Hack enabled, but something wrong
            print("### PTXAS hack failed due to %s!"%code.name)
            exit(-1)
            
        elif code>0:
            # Hack disabled, or skipped safely (such as unmatched ptx)
            returncode = runPTXAS(sys.argv[1:])
            exit(returncode)

        else:
            # Hack success
            exit(0)

    except Exception as e:
        print("### PTXAS hack runtime error! %s"%e.args)
        raise e
        exit(-1)
