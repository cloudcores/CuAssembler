# CuAssembler: An unofficial CUDA assembler

## What is CuAssembler

**CuAssembler** is an unofficial assembler for nvidia CUDA. It's an assembler, reads assemblies(sass) and writes machine codes(in cubin). It's not another compiler just like officially provided by nvidia such as `nvcc` for CUDA C, and `ptxas` for ptx.

The aim of **CuAssembler** is to bridge the gap between `ptx`(the lowest level officially supported and documented by nvidia) and the machine code. Some similar packages include `asfermi` and `maxas`, which can only handle some of CUDA instruction sets. CuAssembler currently supports `Pascal/Volta/Turing/Ampere` instruction set(`SM60/61/70/75/80/86/...`), but the mechanism could be easily extended to older and possibly future CUDA instruction sets, since most of the instruction sets could be probed automatically.

**NOTE**: This library is still in its infancy, there are still a lot of works to be done. Interfaces and architectures are subject to change, use it at your own risk.

## When and how should CuAssembler be used

Many CUDA users will inspect the generated sass code by `cuobjdump` after doing optimization of CUDA c code. The easiest way to tune the sass code is to modify CUDA c code itself, and then recheck the generated sass code again. For many cases, this will lead you to good enough codes (If you are really good at this :) ). However, for those ninja programers that want to optimize the codes down to every instruction, it would be rather upset when they cannot command the compiler to generate the code they want. An alternative tuning method is to modify the intermediate ptx code, which is full of vague variables particularly tedious and difficult to follow, and the generated machine codes are still not always satisfying. CuAssembler allows the user to tune the generated sass code directly.

It should be emphasized that, for most CUDA programmers, CUDA C (sometimes ptx) is always the first choice. It is full featured with great deal of compiling optimizations, officially supported and well documented by nvidia. They know best of their hardware, hence the compiler is also capable of doing some architecture specific optimizations. When the generated sass code is far from expected, you are very likely to have plenty of space for high level languages to play with. There are also large amount of communities and forums which could turn to for help. Playing with assemblies is pretty pain-staking comparing with high level languages, you need to worry about everything that could be done automatically by the compiler. It becomes an eligible option only when you are already quite familiar with CUDA c and ptx, and have played all the tricks you know to optimize, but still find the generated codes are not satisfying. Even in this case, it's still much more convenient to start with CUDA c or ptx, and then do some minor amendments based on the generated sass. This is the main usage CuAssembler is designed for: providing an option for minor adjustment of the generated machine codes, which is not possible by official tools.

Another important usage of CuAssembler is for micro-benchmarking, i.e., probing some specific details of micro-architecture by some specifically designed small programs. Good code optimization usually needs quite deep understanding of the hardware, especially performance related figures, such as the latency and throughput of different instructions, the cache hierarchy, the latency and throughput of every level of caches, cache replacement policies, etc. Many micro-benchmarking could be done using CUDA c, but its more straightforward and flexible when using assemblies, since you can not only arrange the instructions in any order you want, but also set the control codes directly, which is no way to be done in CUDA c or ptx.

As an assembler, CuAssembler simply translates the assemblies to machine codes literally, and then embeds them to cubin so it can be loaded and executed. It's programers' responsibility to guarantee the correctness of the code semantically, such as explicit register allocation, proper arrangement of instructions, and correct usage of registers (e.g., register pair for 64bit variables always starts from even). So you should get familiar with those conventions first, otherwise it's not possible to write legal assembly codes, and this kind of error will be far from conspicuous to catch. Nevertheless, legal assemblies does not imply legal program. There are many kinds of resources involved in CUDA program, such as general purpose registers, predicates, shared memories, and many others. They should match with the hardware configurations, and should be eligible for launching the specified dimension of blocks. Checking rigorous correctness of the whole program needs comprehensive understanding of the launch model and instruction set, both grammatically and semantically, far from possible without official support. So, it is left to the user to guarantee the correctness of the program, with very limited help from the assembler.

## A short HOWTO

CuAssembler is not designed for creating CUDA program from scratch, it has to work with other CUDA toolkits. A good start of cubin is needed, maybe generated by `nvcc` from CUDA C using option `-cubin` or `--keep`, or `ptxas` from hand-written or tuned PTX codes. Currently `nvcc` doesn't support resuming the linking with modified cubin directly(not likely even in the future, due to its vulnerability). Thus the generated cubin usually need to be loaded in driver api. However, `nvcc` has a `--dryrun` option that can list all the commands that really builds up the compiling steps, we may hack this script(actually, just the `ptxas` step for generating cubin from ptx). Then we can run this program just using runtime api, which is much simpler. However, this also implies a limitation of our approach, all the sections, symbols, global variables in cubin should kept as is, otherwise the hacking may not work properly.

Remember to keep other optimization works done before coding with CuAssembler, since any modification of the input cubin may invalidate the modification done in CuAssembler, then you may need to redo all the work again.

See the [User Guide](UserGuide.md) and [Tutorial](Tutorial.md) for basic tutorial and introduction of input formats.

### Prerequisites

* **CUDA toolkit 10+**: Version 10+ is needed to support `sm_75` (turing instruction sets), and version 11+ for ampere. Actually only the stand-alone program `nvdisasm` will be used by CuAssembler when saving `cubin` as `cuasm`, and `cuobjdump` may be used to dump sass. If you start from `cuasm`, no CUDA toolkit will be required. **NOTE**: it is known that some instructions or modifiers may not show up in the disassembly text, at least in some versions. Thus you may need to check some new version, if they have been fixed. Since `nvdisasm` and `cuobjdump` is stand-alone, you don't need to fetch full toolkit, just these two programs will do the job. 
* **Python 3.8+**: Previous python versions may also supported, but not tested yet.
* **Sympy 1.4+**: Integeral (or rational) matrices with arbitrary precision needed by solving the LAE, and carrying out the null space of `V`. **NOTE**: before 1.4, sympy seems to cache all big integers, which may work like memory leak when you've assembled many instructions.
* **pyelftools**: elf toolkit for handling cubin files.

`sympy` and `pyelftools` can be obtained with `pip install sympy pyelftools`.

### Settings and Simple Usage

**PATH** and **PYTHONPATH**: you may need to add CuAssembler bin path (`CuAssembler/bin`) to system `PATH` for scripts to work, and including the root dir in `PYTHONPATH` is required for `import CuAsm`. Thus you may add these lines to your `.bashrc`(change the path accordingly):

```
  export PATH=${PATH}:~/works/CuAssembler/bin
  export PYTHONPATH=${PYTHOPATH}:~/works/CuAssembler/
```

In the dir `bin`, CuAssembler offers several python scripts(`cuasm/hnvcc/hcubin/dsass/...`) to accelerate the development procedure. Running with `python cuasm.py` or simply `cuasm.py` is not simple enough, thus a simbol link can be created: 

```
ln -s cuasm.py cuasm
chmod a+x cuasm
```

You may just put this symbol link under your current `PATH` instead of adding `CuAssembler/bin` to your system `PATH`.

NOTE: most scripts(except `hnvcc`) also work under windows, the `*.bat` files under `bin` is the command-line wrapper.

#### cuasm

```
usage: cuasm [-h] [-o OUTFILE] [-f LOGFILE] [-v | -q] [--bin2asm | --asm2bin] infile [infile ...]

    Convert cubin from/to cuasm files.

    NOTE 1: if the output file already exist, the original file will be renamed to "outfile~".
    NOTE 2: if the logfile already exist, original logs will be rolled to logname.1, logname.2, until logname.3.

positional arguments:
  infile                Input filename. If not with extension .cubin/.bin/.cuasm/.asm, direction option --bin2asm or --asm2bin should be specified.

options:
  -h, --help            show this help message and exit
  -o OUTFILE, --output OUTFILE
                        Output filename, inferred from input filename if not given.
  -f LOGFILE, --logfile LOGFILE
                        File name for saving the log, default to none.
  -v, --verbose         Verbose mode, showing almost every log.
  -q, --quiet           Quiet mode, no log unless errores found.
  --bin2asm             Convert from cubin to cuasm.
  --asm2bin             Convert from cuasm to cubin.

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
```

#### dsass

```
usage: dsass [-h] [-o OUTFILE] [-k] [-n] [-f LOGFILE] [-v | -q] infile [infile ...]

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

positional arguments:
  infile                Input filename, can be dumped sass, cubin, or binary contains cubin.

options:
  -h, --help            show this help message and exit
  -o OUTFILE, --output OUTFILE
                        Output filename, infered from input filename if not given.
  -k, --keepcode        Keep code-only lines in input sass, default to strip.
  -n, --nodeschack      Do not hack desc bit, no matter SM version it is.
  -f LOGFILE, --logfile LOGFILE
                        File name for saving the logs, default to none.
  -v, --verbose         Verbose mode, showing almost every log.
  -q, --quiet           Quiet mode, no log unless errores found.

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
```

#### hnvcc

**NOTE**: hnvcc only works under linux.

```
Usage: hnvcc nvcc_args...

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
```

#### hcubin

```
usage: hcubin [-h] [-o OUTFILE] [-f LOGFILE] [-v | -q] infile [infile ...]

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

positional arguments:
  infile                Input filename, should be a valid cubin file.

options:
  -h, --help            show this help message and exit
  -o OUTFILE, --output OUTFILE
                        Output filename, infered from input filename if not given.
  -f LOGFILE, --logfile LOGFILE
                        File name for saving the logs, default to none.
  -v, --verbose         Verbose mode, showing almost every log.
  -q, --quiet           Quiet mode, no log unless errores found.

Examples:
    $ hcubin a.cubin
        hack a.cubin into a.hcubin, default output name is replacing the ext to .hcubin

    $ hcubin a.cubin -o x.bin
        hack a.cubin into x.bin

    $ hcubin a.cubin x.bin
        same as `hcubin a.cubin -o x.bin`
```

### Classes

* **CuAsmLogger**: A logger class utilizing python logging module. Note all the logging is done by a private logger, thus other loggers are not likely to be affected, if they use their own logger.
* **CuAsmParser**: A parser class that can parse the user modified `.cuasm` text file, and save the result as `.cubin`. 
* **CubinFile**: It can read in a `.cubin` file, rewrite it into an editable `.cuasm` text file.
* **CuInsAssembler**: The class that handles the value matrix `V` and solution of `w` for a special instruction *key*, such as `FFMA_R_R_R_R`.
* **CuInsAssemblerRepos**: Repository of `CuInsAssembler` for all known *keys*. Constructing a workable repos from scratch is very time consuming, and it requires quite wide range of inputs that cover all frequently used instructions. Thus a pre-gathered repos is available in `DefaultInsAsmRepos.${arch}.txt`. **Note**: the repository may be incomplete, but user can easily update it.
* **CuInsParser**: The class to parse the instruction string to *keys*, *values* and *modifiers*.
* **CuInsFeeder**: A simple instruction feeder reading instructions from sass dumped by `cuobjdump`.
* **CuKernelAssembler**: Assembler for a kernel, which should handle all kernel wide parameters, mostly nvinfo attributes.
* **CuNVInfo**: A simple class that handles `NVInfo` section of cubin. This class is far from complete and robust, thus some `NVInfo` attributes have very limited support in CuAssembler.
* **CuSMVersion**：A class that provides a uniform interface of all SM versions. All other classes are not recommended to contain architecture dependent treatments (well, at least hopefully...). Thus for future architectures, most of the work should be in this class.  

## Future plan

Likely to support:

* Better coverage of intructions, bugfixes for officially unsupported instructions.
* Extend to more compute capabilities, `sm_60/61/70/75/80/86` will be mostly concerned. 
* More robust correctness check with aid of `nvdisasm`.
* Automatically set control codes. 
* Alias and variable support, for easier programming, may be achieved by preprocessing?

Less likely to support, but still on the plan:
* Register counting, and possibly register allocation
* More robust parsing and user friendly error reporting.
* Control flow support? May also be achieved by preprocessing in python?
* And others...
