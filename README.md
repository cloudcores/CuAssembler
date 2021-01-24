# CuAssembler: An unofficial CUDA assembler

## What is CuAssembler

**CuAssembler** is an unofficial assembler for nvidia cuda. It's an assembler, reads assemblies(sass) and writes machine codes(in cubin). It's not another compiler just like officially provided by nvidia such as `nvcc` for cuda c, and `ptxas` for ptx.

The aim of **CuAssembler** is to bridge the gap between `ptx`(the lowest level officially supported and documented by nvidia) and the machine code. Some similar packages include `asfermi` and `maxas`, which can only handle some of cuda instruction sets. CuAssembler currently only supports `Turing` instruction set, but the mechanism could be easily extended to older and possibly future cuda instruction sets, since most of the instruction sets could be probed automatically.

**NOTE**: This library is still in its infancy, there are still a lot of work to be done. Interfaces and architectures are subject to change, use it at your own risk.

## When and how should CuAssembler be used

Many cuda users will inspect the generated sass code by `cuobjdump` after doing optimization of cuda c code. The easiest way to tune the sass code is to modify cuda c code itself, and then recheck the generated sass code again. For many cases, this will lead you to good enough codes (If you are really good at this :) ). However, for those ninja programers that want to optimize the codes down to every instruction, it would be rather upset when they cannot command the compiler to generate the code they want. An alternative tuning method is to modify the intermediate ptx code, which is full of vague variables particularly tedious and difficult to follow, and the generated machine codes are still not always satisfying. CuAssembler allows the user to tune the generated sass code directly.

It should be emphasized that, for most cuda programmers, cuda c (sometimes ptx) is always the first choice. It is full featured with great deal of compiling optimizations, officially supported and well documented by nvidia. They know best of their hardware, hence the compiler is also capable of doing some architecture specific optimizations. When the generated sass code is far from expected, you are very likely to have plenty of space for high level languages to play with. There are also large amount of communities and forums which could turn to for help. Playing with assemblies is pretty pain-staking comparing with high level languages, you need to worry about everything that could be done automatically by the compiler. It becomes an eligible option only when you are already quite familiar with cuda c and ptx, and have played all the tricks you know to optimize, but still find the generated codes are not satisfying. Even in this case, it's still much more convenient to start with cuda c or ptx, and then do some minor amendments based on the generated sass. This is the main usage CuAssembler is designed for: providing an option for minor adjustment of the generated machine codes, which is not possible by official tools.

Another important usage of CuAssembler is for micro-benchmarking, i.e., probing some specific details of micro-architecture by some specifically designed small programs. Good code optimization usually needs quite deep understanding of the hardware, especially performance related figures, such as the latency and throughput of different instructions, the cache hierarchy, the latency and throughput of every level of caches, cache replacement policies, etc. Many micro-benchmarking could be done using cuda c, but its more straightforward and flexible when using assemblies, since you can not only arrange the instructions in any order you want, but also set the control codes directly, which is no way to be done in cuda c or ptx.

As an assembler, CuAssembler simply translates the assemblies to machine codes literally, and then embeds them to cubin so it can be loaded and executed. It's programers' responsibility to guarantee the correctness of the code semantically, such as explicit register allocation, proper arrangement of instructions, and correct usage of registers (e.g., register pair for 64bit variables always starts from even). So you should get familiar with those conventions first, otherwise it's not possible to write legal assembly codes, and this kind of error will be far from conspicuous to catch. Nevertheless, legal assemblies does not imply legal program. There are many kinds of resources involved in cuda program, such as general purpose registers, predicates, shared memories, and many others. They should match with the hardware configurations, and should be eligible for launching the specified dimension of blocks. Checking rigorous correctness of the whole program needs comprehensive understanding of the launch model and instruction set, both grammatically and semantically, far from possible without official support. So, it is left to the user to guarantee the correctness of the program, with very limited help from the assembler.

## A short HOWTO

CuAssembler is not designed for creating CUDA program from scratch, it has to work with other CUDA toolkits. A good start of cubin is needed, maybe generated by `nvcc` from CUDA C using option `--keep`, or `ptxas` from hand-written or tuned PTX codes. Currently `nvcc` doesn't support resuming the linking with modified cubin directly(not likely even in the future, due to its vulnerability). Thus the generated cubin usually need to be loaded in driver api. However, `nvcc` has a `--dryrun` option that can list all the commands that really builds up the compiling steps, we may hack this script(actually, just the `ptxas` step for generating cubin from ptx). Then we can run this program just using runtime api, which is much simpler. However, this also implies a limitation of our approach, all the sections, symbols, global variables in cubin should kept as is, otherwise the hacking may not work properly.

Remember to keep other optimization works done before coding with CuAssembler, since any modification of the input cubin may invalidate the modification done in CuAssembler, then you may need to redo all the work again.

See the long HOWTO and tutorial for more information.

### Prerequisites

* **CUDA toolkit 10+**: Version 10+ is needed to support `sm_75` (turing instruction sets), and version 11+ for ampere. Actually only the stand-alone program `nvdisasm` will be used by CuAssembler when saving `cubin` as `cuasm`, and `cuobjdump` may be used to dump sass. If you start from `cuasm`, no cuda toolkit will be required. **NOTE**: it is known that some instructions or modifiers may not show up in the disassembly text, at least in some versions. Thus you may need to check some new version, if they have been fixed. Since `nvdisasm` and `cuobjdump` is stand-alone, you don't need to fetch full toolkits, just these two program will do the job. 
* **Python 3.7+**: Previous python versions may also supported, but not tested yet.
* **Sympy 1.4+**: Integeral (or rational) matrices with arbitrary precision needed by solving the LAE, and carrying out the null space of `V`. **NOTE**: before 1.4, sympy seems to cache all big integers, which may work like memory leak when you've assembled many instructions.
* **pyelftools**: elf toolkit for handling cubin files.

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
* Extend to more compute capabilities, `sm_61`, `sm_75`, `sm_86` will be mostly concerned. 
* More robust correctness check with aid of `nvdisasm`.
* Automatically set control codes. 
* Alias and variable support, for easier programming, may be achieved by preprocessing?

Less likely to support, but still on the plan:
* Register counting, and possibly register allocation
* More robust parsing and user friendly error reporting.
* Control flow support? May also be achieved by preprocessing in python?
* And others...