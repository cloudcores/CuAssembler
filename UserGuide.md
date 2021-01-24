CuAssembler User Guide
- [A simple tutorial for using CuAssembler with CUDA runtime API](#a-simple-tutorial-for-using-cuassembler-with-cuda-runtime-api)
  - [Start from a CUDA C example](#start-from-a-cuda-c-example)
  - [Build CUDA C into Cubin](#build-cuda-c-into-cubin)
  - [Disassemble Cubin to CuAsm](#disassemble-cubin-to-cuasm)
  - [Adjust the assembly code in cuasm](#adjust-the-assembly-code-in-cuasm)
  - [Assemble cuasm into cubin](#assemble-cuasm-into-cubin)
  - [Hack the original executable](#hack-the-original-executable)
  - [Run or debug the executable](#run-or-debug-the-executable)
- [Format of cubin and cuasm](#format-of-cubin-and-cuasm)
  - [File Structure](#file-structure)
  - [Sections and Segments](#sections-and-segments)
  - [Labels and Symbols](#labels-and-symbols)
  - [Traps and Pitfalls](#traps-and-pitfalls)
- [How CuAssembler works](#how-cuassembler-works)
  - [Automatic Instruction Encoding](#automatic-instruction-encoding)
  - [Special Treatments of Encoding](#special-treatments-of-encoding)
  - [Instruction Assembler Repository](#instruction-assembler-repository)

# A simple tutorial for using CuAssembler with CUDA runtime API

We will show the basic usage of CuAssembler, by a simple `cudatest` case. CuAssembler is just an assembler, its main purpose is to generate the cubin file according to user input assembly. All device initialization, data preparation and kernel launch should be done by the user, possibly using CUDA driver API. However, it's usually more convenient to start from runtime API. Here we will demonstrate the general workflow for using CUDA runtime API with CuAssembler.

The tutorial is far from complete, many basic knowledge of CUDA is needed for this trivial task. The code is not fully shown, and some common building steps are ignored, but I think you should get the idea... If not, you are probably too early to be here, please get familiar with basic CUDA usage first~

Some useful references of prerequisites knowledge:
* Basic knowledge of [CUDA](https://docs.nvidia.com/cuda/index.html). 
* [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) and [CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html): many users just utilize those tools via IDE, but here, you will have to play with them in command line from time to time.
* ELF Format: There are many references on the format of ELF, both generic and architecture dependent, for example, [this one](http://downloads.openwatcom.org/ftp/devel/docs/elf-64-gen.pdf). Currently only **64bit** version of ELF (**little endian**) is supported by CuAssembler. 
* Common assembly directives: `nvdisasm` seems to reuse many conventions of gnu assembler. Since no doc is provided on the grammar of `nvdisasm` disassemblies, get familiar with [Gnu Assembler directives](https://ftp.gnu.org/old-gnu/Manuals/gas-2.9.1/html_chapter/as_7.html) would be helpful.
* CUDA PTX and SASS instructions: Before you can write any assemblies, you need to know the language first. Currently no official (at least no comprehensive) doc is provided on SASS, just [simple opcodes list](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-ref). Get familiar with [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) and its documentation will be greatly helpful to get the semantics of SASS assemblies.

## Start from a CUDA C example

First we need to create a `cudatest.cu` file with enough information of kernels. You may start from any other CUDA samples with explicit kernel definitions. An example may look like this:

```c++
// ... includes ignored
__global__ void vectorAdd(const float* a, const float* b, float* c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}

int main()
{
    // Other runtime API codes are ignored, it's irrelevant of our procedure.
    return 0;
}
```

Currently CuAssembler does not fully support modification of kernel args, globals (constants, texture/surface references), thus all these information(size, name, etc.) should be defined in CUDA C, and inherited from cubin. Best practice here is to make a naive working version of the kernel, with all required resources prepared. Then in assembly, only the instructions need to be modified, that's the most robust way CuAssembler can be used. 

**NOTE**: when you get into the stage of final assembly tuning, modifying the original CUDA C would be very unreliable, and usually rather error-prone, thus it's strongly recommended to keep all the staff unchanged in CUDA C. If you really need this, you probably have to make a big restructuring of the generated assembly. Making version control of the generated `cuasm` file may help you get through this more easily, and hopefully less painfully.

## Build CUDA C into Cubin

`nvcc` is the canonical way to build a `.cu` file into executable, such as `nvcc -o cudatest cudatest.cu`. However, we need the intermediate `cubin` file to start with. thus we will use the `--keep` option of `nvcc`, which will keep all intermediate files (such as ptx, cubin, etc.). By default, only the lowest supported SM version(by current version of NVCC) of ptx and cubin will be generated, if you need a specific SM version of cubin, you need to specify the `-gencode` option, such as `-gencode=arch=compute_75,code=\"sm_75,compute_75\"` for turing (`sm_75`). The full command may look like:

```
    nvcc -o cudatest cudatest.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --keep
```

Then you will get cubins such as `cudatest.1.sm_75.cubin`, under the intermediate files directory (maybe just current directory). Then we get a cubin to start with.

**NOTE**: Sometimes `nvcc` may generate several `cubin` of different versions, and possibly an extra empty cubin of every SM version. You can check the contents by `nvdisasm`, or just judging by the file size.

Another important information from `nvcc` is that we need full building steps. Thus we use the `--dryrun` option to list all the steps invoked by `nvcc`.

```
    nvcc -o cudatest cudatest.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --dryrun
```

You may get something like this (some lines are ignored, you may have different output):

```bat
    ...
#$ cl.exe > "cudatest.cpp1.ii" -D__CUDA_ARCH__=750 -nologo -E -TP  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin/../include"    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=74 -D__CUDA_API_VER_MAJOR__=11 -D__CUDA_API_VER_MINOR__=1 -FI "cuda_runtime.h" -EHsc "cudatest.cu"
#$ cicc --microsoft_version=1925 --msvc_target_version=1925 --compiler_bindir "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.25.28610/bin/Hostx64/x64/../../../../../../.." --orig_src_file_name "cudatest.cu" --allow_managed  -arch compute_75 -m64 -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "cudatest.fatbin.c" -tused -nvvmir-library "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin/../nvvm/libdevice/libdevice.10.bc" --gen_module_id_file --module_id_file_name "cudatest.module_id" --gen_c_file_name "cudatest.cudafe1.c" --stub_file_name "cudatest.cudafe1.stub.c" --gen_device_file_name "cudatest.cudafe1.gpu"  "cudatest.cpp1.ii" -o "cudatest.ptx"
#$ ptxas -arch=sm_75 -m64 "cudatest.ptx"  -o "cudatest.sm_75.cubin"
#$ fatbinary --create="cudatest.fatbin" -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=elf,sm=75,file=cudatest.sm_75.cubin" "--image3=kind=ptx,sm=75,file=cudatest.ptx" --embedded-fatbin="cudatest.fatbin.c"
#$ cl.exe > "cudatest.cpp4.ii" -nologo -E -TP -D__CUDACC__ -D__NVCC__  "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin/../include"    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=74 -D__CUDA_API_VER_MAJOR__=11 -D__CUDA_API_VER_MINOR__=1 -FI "cuda_runtime.h" -EHsc "cudatest.cu"
#$ cudafe++ --microsoft_version=1925 --msvc_target_version=1925 --compiler_bindir "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.25.28610/bin/Hostx64/x64/../../../../../../.." --orig_src_file_name "cudatest.cu" --allow_managed --m64 --parse_templates --gen_c_file_name "cudatest.cudafe1.cpp" --stub_file_name "cudatest.cudafe1.stub.c" --module_id_file_name "cudatest.module_id" "cudatest.cpp4.ii"
#$ cl.exe -Fo"cudatest.obj" -D__CUDA_ARCH__=750 -nologo -c -TP  -DCUDA_DOUBLE_MATH_FUNCTIONS "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin/../include"   -EHsc "cudatest.cudafe1.cpp"
#$ nvlink -optf "cudatest_dlink.sm_75.cubin.optf"
#$ fatbinary --create="cudatest_dlink.fatbin" -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " -link "--image3=kind=elf,sm=75,file=cudatest_dlink.sm_75.cubin" --embedded-fatbin="cudatest_dlink.fatbin.c"
#$ cl.exe -Fo"cudatest_dlink.obj" -nologo -c -TP -DFATBINFILE="\"cudatest_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"cudatest_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin/../include"    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=74 -D__CUDA_API_VER_MAJOR__=11 -D__CUDA_API_VER_MINOR__=1 -EHsc "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin\crt\link.stub"
#$ cl.exe -Fe"cudatest.exe" -nologo "cudatest_dlink.obj" "cudatest.obj" -link -INCREMENTAL:NO   "/LIBPATH:C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin/../lib/x64"  cudadevrt.lib  cudart_static.lib
```

Saving those commands to a script file(e.g., `*.bat` for windows, `*.sh` for linux). Since we will need them when we want to embed the hacked cubin back to the executable, and run it as if the hacking does not happen at all.

## Disassemble Cubin to CuAsm

Then we can create a python script of CuAssembler to disassemble the `cubin` into `cuasm`:

```python
from CuAsm.CubinFile import CubinFile

binname = 'cudatest.2.sm_75.cubin'
cf = CubinFile(binname)
asmname = binname.replace('.cubin', '.cuasm')
cf.saveAsCuAsm(asmname)
```

**NOTES**: CuAssembler is a python package, with default package name as directory name `CuAsm`. To make the package visible to python importing, you may need to append its parent dir to environment variable `PYTHONPATH`, or just copy the dir to any current `PYTHONPATH`. If you just want to make it temporally importable, you can append it to `sys.path` in the python script.

## Adjust the assembly code in cuasm

Most contents of `cuasm` file is copied from `nvdisasm` result of the cubin, with some supplementary ELF information explicitly recorded in text format , such as section header attributes, implicit sections(such as `.strtab/.shstrtab/.symtab`) not shown in disassembly. All the information inherited directly from the cubin should not be modified (unless have to, such the offset and size of sections). This does not mean these information cannot be automatically generated, but since NVIDIA provides no information about their conventions, probing them all would be rather pain-staking. Thus it's much safer and easier to keep them as is. Actually, most adjustment of those information (such as add a kernel, global, etc.) can be achieved by modifying the original CUDA C code, which is officially supported and much more reliable.

See an [example cuasm](TestData/CuTest/cudatest.7.sm_75.cuasm) in `TestData` for more information.

## Assemble cuasm into cubin

Assembling cuasm into cubin is also trivial:

```python
from CuAsm.CuAsmParser import CuAsmParser

asmname = 'cudatest.7.sm_75.cuasm'
binname = 'new_cudatest.7.sm_75.cubin'
cap = CuAsmParser()
cap.parse(asmname)
cap.saveAsCubin(binname)
```

I prefer to rename the generated `cubin` to a different name with respect to the original cubin. Since we may need to check or compare some information of those two cubins simultaneously.

## Hack the original executable 

As soon as you get a hacked cubin, the easiest way to put it back to the executable is to mimic the behavior of the original building steps. Take a look at the output of `nvcc` with `--dryrun` option, there will be a step which looks like:

```bat
ptxas -arch=sm_75 -m64 "cudatest.ptx"  -o "cudatest.sm_75.cubin"
```

You can delete all the steps before this one (include this ptxas step), rename your hacked cubin to `cudatest.sm_75.cubin`, and run the rest of those building steps. That will give you an executable just like run `nvcc` directly (at least hopefully, if everything goes well).

Sometimes you may not need to hack all of the cubins, you can freely hack one or more ptxas steps, since ptxas just accepts one file at a time. For more convenient usage, you may also copy those steps into a makefile, and run the rebuild steps if any dependent file is modified. You can even make a script or environment variable to switch between the hacked version and original version.

## Run or debug the executable

If everything goes right, the hacked cubin should work as good as the original one. However, if some mismatches exist with respect to the original CUDA C file(such as kernel names, kernel arg arrangements, global contants, and global texture/surface references), the executable may not work right. That's why we should always get those information ready before hacking the cubin. Another issue is, some symbol information will be used for proper debugging. Thus you should not modify them as well (symbol offsets and sizes will be automatically updated by assembler). 

See next section for more info on the contents of cubin and cuasm.

# Format of cubin and cuasm

**Cubin** is an ELF format binary, thus most of its file structure will follow the convension of ELF. However, there are also many CUDA specific features involved. **Cuasm** is just a text form of cubin, with most of the cubin information explicitly described with assembly directives. Most of the assembly directives will follow the same semantics of `nvdisasm` (actually most of the cuasm contents are copied from `nvdisasm` output), but there are also some new directives, helping make some information clear and explicit.

## File Structure

An ELF file contains a file header, several sections, none or several program segments. Usually the cubin files are arranged like this:

* **File Header** : The ELF file header will specify some common information of every ELF, such as identifier magic number, 32bit or 64bit, section header offset, program header offset, etc. For cubin, it will also specify the version of current cubin: virtual architecture version and SM version, toolkit version etc. 
* **Section Data** : The data of every section.
* **Section Header** : Header information of every section. Defines the section name string offset, section offset, size, flags, type, and extra info, linkage with respect to other sections, etc.
* **Segment Header** : Header information of every segment. Defines how the sections will be loaded. **NOTE**: for some `ET_REL` type of ELF, there could be no segment. They are likely to be linked to another cubin for final executable. Currently, only `ET_EXEC` type of ELF is tested, it usually contains 3 segments.

## Sections and Segments

This is a sample layout of sections:

```
Index Offset   Size ES Align        Type        Flags Link     Info Name
    1     40    418  0  1            STRTAB       0    0        0 .shstrtab
    2    458    783  0  1            STRTAB       0    0        0 .strtab
    3    be0    450 18  8            SYMTAB       0    2       22 .symtab
    4   1030    5c8  0  1          PROGBITS       0    0        0 .debug_frame
    5   15f8    21c  0  4         CUDA_INFO       0    3        0 .nv.info
    6   1814     78  0  4         CUDA_INFO       0    3       1a .nv.info._Z7argtestPiS_S_
    7   188c     4c  0  4         CUDA_INFO       0    3       1b .nv.info._Z11shared_testfPf
    8   18d8     60  0  4         CUDA_INFO       0    3       1c .nv.info._Z11nvinfo_testiiPi
    9   1938     4c  0  4         CUDA_INFO       0    3       1d .nv.info._Z5childPii
    a   1984     5c  0  4         CUDA_INFO       0    3       1e .nv.info._Z10local_testiiPi
    b   19e0     4c  0  4         CUDA_INFO       0    3       1f .nv.info._Z4test6float4PS_
    c   1a30     d0  8  8    CUDA_RELOCINFO       0    0        0 .nv.rel.action
    d   1b00     70 10  8               REL       0    3       1a .rel.text._Z7argtestPiS_S_
    e   1b70     30 18  8              RELA       0    3       1a .rela.text._Z7argtestPiS_S_
    f   1ba0     40 10  8               REL       0    3       1d .rel.text._Z5childPii
   10   1be0     40 10  8               REL       0    3       14 .rel.nv.constant0._Z7argtestPiS_S_
   11   1c20     d0 10  8               REL       0    3        4 .rel.debug_frame
   12   1cf0    141  0  4          PROGBITS       2    0        0 .nv.constant3
   13   1e34     48  0  4          PROGBITS       2    0       1a .nv.constant2._Z7argtestPiS_S_
   14   1e7c    188  0  4          PROGBITS       2    0       1a .nv.constant0._Z7argtestPiS_S_
   15   2004    170  0  4          PROGBITS       2    0       1b .nv.constant0._Z11shared_testfPf
   16   2174    170  0  4          PROGBITS       2    0       1c .nv.constant0._Z11nvinfo_testiiPi
   17   22e4    16c  0  4          PROGBITS       2    0       1d .nv.constant0._Z5childPii
   18   2450    170  0  4          PROGBITS       2    0       1e .nv.constant0._Z10local_testiiPi
   19   25c0    178  0  4          PROGBITS       2    0       1f .nv.constant0._Z4test6float4PS_
   1a   2780    d80  0 80          PROGBITS       6    3 18000023 .text._Z7argtestPiS_S_
   1b   3500    200  0 80          PROGBITS  100006    3  c000029 .text._Z11shared_testfPf
   1c   3700    100  0 80          PROGBITS       6    3  a00002a .text._Z11nvinfo_testiiPi
   1d   3800    280  0 80          PROGBITS       6    3  e00002b .text._Z5childPii
   1e   3a80    180  0 80          PROGBITS       6    3  d00002c .text._Z10local_testiiPi
   1f   3c00    480  0 80          PROGBITS       6    3  a00002d .text._Z4test6float4PS_
   20   4080     5c  0  8          PROGBITS       3    0        0 .nv.global.init
   21   40e0      0  0 10            NOBITS       3    0       1a .nv.shared._Z7argtestPiS_S_
   22   40e0     a0  0  4            NOBITS       3    0        0 .nv.global
   23   40e0   1010  0 10            NOBITS       3    0       1b .nv.shared._Z11shared_testfPf
   24   40e0      0  0 10            NOBITS       3    0       1c .nv.shared._Z11nvinfo_testiiPi
   25   40e0      0  0 10            NOBITS       3    0       1d .nv.shared._Z5childPii
   26   40e0      0  0 10            NOBITS       3    0       1e .nv.shared._Z10local_testiiPi
   27   40e0      0  0 10            NOBITS       3    0       1f .nv.shared._Z4test6float4PS_
```

* `.shstrtab/.strtab/.symtab` : tables for section string, symbol string, and symbol entries. Currently all off them are copied from the original cubin.
* `.nv.info.*` : Some attributes associated with kernels. `cuobjdump -elf *.cubin` can show those information in human readable format. Some of those attributes should be modified when kernel text changed, some can be done by CuAssembler, but there are more that cannot. 
* `.rel.*` : Relocations. Relocation section should work with the associated section, such as `.rel.abc` vs `.abc`. Relocation is a special mechanism which allows runtime initialization of some symbols unknown during compile-time, such as some global constants and function entries. 
* `.nv.constant#.*` :
* `.text.*` :
* `.nv.shared.*` :

## Labels and Symbols


## Traps and Pitfalls


# How CuAssembler works

## Automatic Instruction Encoding

Most work of assembler is to encode the instruction. For turing, every instruction is 128bit, split into two 64bit lines in `cuobjdump` dumped sass. For example, the instruction:

```
    /*1190*/    @P0 FADD.FTZ R13, -R14, -RZ ;    /* 0x800000ff0e0d0221 */
                                                 /* 0x000fc80000010100 */
```

Here is the nomenclature of CuAssembler on how these fields will be called: `/*1190*/` is the instruction *address* (in hex). `@P0` is the *predicate*, or more specifically the guard predicate, as stated in ptx documentation. `FADD` is the type of operation (referred as *opcode*, actually the opcode is usually the code field used to encode the operation, and `FADD` is the mnemonics of the opcode, which may used interchangably): single precision float addition. `.FTZ` is a *modifier* of `FADD`, means flush-to-zero when any of the inputs or outputs is denormal. `R13`, `-R14`, `-RZ` are the *operands* of `FADD`, means `R13 = (-R14) + (-RZ)`. `RZ` is an register that always yields 0. *Modifier* is not only for the opcode, it also includes anything that can modify the original semantics of the operands, such as the minus "-" or absolute "|*|".

Every field will encode some bits of the instruction. Three operands (`R13`, `-R14`, `-RZ`) are all of type register, so those fields will be not only depend on the content, but also position dependent. Thus the final code can be written as sum of the encoding of every field:

>`c = c("@P0") + c("FADD") + c("FTZ") + c("0_R13") + c("1_-R14") + c("2_-RZ")`.

The minus in operand `-R14`, `-RZ` can also be considered as negative modifier "Neg", and in Turing, `RZ` is always an alias of `R255`. Any other modifier for operands (currently known: "!" for predicate not, "-" for numerical negative, "|" for numerical abs, "~" for bitnot, and some bit field or type specifiers ".H0_H0", ".F32", etc.) will also be striped as separate fields. Hence the code becomes:

>`c = c("@P0") + c("FADD") + c("FTZ") + c("0_R13") + c("1_Neg") + c("1_R14") + c("2_Neg") + c("2_R255")`.

Now the problem becomes how to encode those elemental fields. We separate the encoding of every field into two parts, `Code = Value*Weight`, in which `Value` only depends on the content, `Weight` only depends on the position where the element appears (both for operands and their modifiers).

For turing architecture, we have these elemental operands, each with some values defined, and with a *label* to identify the operand type:

* **Indexed type**: Indexed type is a type prefix followed by a positive integer index, such as registers `R#`, predicates `P#`, uniform registers and predicates `UR#` and `UP#`, convergence barriers `B#`, and scoreboards `SB#`. The value of indexed type is just the index. The label is just the prefix.
* **Address**: Memory address in a square bracket `[0x####]`. Inside the bracket, there could also be an offset specified by register: `[R#+0x####]`, or even more complicated: `[UR#+R#.X16+0x####]`. The value of address could be a list, including the value of the register and the offset. The label is `A` followed by labels inside the bracket, such as `R` for register only, `RI` for register+immediate offset, `I` for immediate only. E.g., `[UR5+R8.X16+0x10]` will have value list `[5, 8, 16]`, and label `AURRI`, `.X16` will be striped into modifiers. Currently, all addresses will be padded with implicit immediate offset `0`, if not present. It's harmless even if it doesnot support this, since the value is zero, the encoding contribution should vanish. Another note is that some modifiers may working together with values, such as `[UR5.U64+R8.U32]`, thus we annotate the modifiers with their associating types, as `UR.U64` and `R.U32`, avoiding any possibility of ambiguity, if any.
* **Constant memory**: Constant memory `c[0x##][0x####]`, first bracket for constant memory bank, second for memory address. The value of constant memory is the list of the constant bank, and the value of the memory address. The label is `cA` followed by the label of the memory address of second bracket.
* **Integer immediate**: Integer immediate such as `0x0` (**NOTE**: integer immediate should always be in hex, raw `1` or `0` will be treated as float immediates). The value is just the bit representation of the integer. NOTE: the negative sign should be treated as an modifier, since we don't know how many bits will the value takes. The label is `II`.
* **Float immediate**: Float immediate such as `5`, `2.0`, `-2.34e-3`. The value of float immediate is just its binary representation, depend on the precision(32bit or 16bit, 64bit not found yet). Denormals such as `+Inf`, `-Inf` and `QNAN` are also possible. The label is `FI`.
* **Scoreboard Set**: This type is only for instruction `DEPBAR` for setting the set of scoreboards to be waited, such as `{1,3}`. Currently there are 6 barriers, with each value corresponding to 1bit. Scoreboards count waited in control codes should be 0, yet `DEPBAR` is capable of waiting a scoreboard with non-zero number. For example, 8 requests is sent to scoreboard 5, every request will increment SB5, and every complete request will decrement SB5. If we only need 3 of them to be back, we can just wait the scoreboard dropping to `8-3=5`, which can be achieved by `DEPBAR.LE SB5, 0x5 ;`. **NOTE**: the comma inside the brakets will affect the splitting of operands, thus the scoreboard sets will be translated to an indexed type `SBSET#` during parsing. The label is `SBSET`.
* **Label**: Any other type not included above. Usually a string, such as `SR_TID.X`, `SR_LEMASK`, `3D`, `ARRAY_2D`, etc. The value of label is quite like the modifier, its value will depend on the context. Usually we set the value to **1**, and let the weight be the real encoding. It's label is just itself.

Then we can obtain the value list of the example instruction `@P0 FADD.FTZ R13, -R14, -RZ`:

>`V = [0, 1, 1, 13, 1, 14, 1, 255]`,

the weight list:

>`w = [w("@P0"), w("FADD"), w("FTZ"), w("0_R13"), w("1_Neg"), w("1_R14"), w("2_Neg"), w("2_R255")]`

 is to be determined. The interesting part is, if we dump instructions with `cuobjdump`, the value lists of every instruction would be readily available and the answer `c = v.*w` is already known! Providing we can gather enough instructions of the same type, we will be able to solve the `w` with linear algebra equations `c = V*w` !

Then what kind of instructions are of the same type? Theoretically, you can always keep value as singleton `1`, and merge all modifiers into one, then let weight be the code! In this case, only instructions in your dictionary can be assembled! It definitely requires too much space, and has too many drawbacks. We should search for some patterns that maximize versatility, yet still minimize the requirement of known instructions input.

For every instruction, minimum length of values is two (one for predicate, one for opcode such as `FADD, IMAD, BRA`) plus number of non-fixed valued operands (i.e., not labels). Therefor we put instructions of the same operation with same number and type of operands into the same categories, and label it by connecting them with underline, such as `FFMA_R_R_R_R`, `IMAD_R_R_II_R`, `FADD_R_R_cAI`, which is called **Key** of this type of instructions. Then we can gather all possible known instruction encodings to solve the unknown weights corresponding to the **Key** of the instruction type.

Solving the weights is usually trivial, as long as you can collect all the instructions you want. Sadly enough, it's not always possible. In case we didn't gather enough instructions, `V` will be rectangular, and solving each element of `w` is impossible. But luckily, it's not always necessary to have all the weights known! We only need to make sure the value list of the instruction `v` to be assembled should be a linear combination of rows of `V`. This is equivalent to check whether `v` lies in the null space of `V`. The interesting thing is, although in this case there are infinite solutions for `V*w = c`, but any of it will give the same result for the code to be assembled: `v.*w`!

This also provides a hint on how `V` could be constructed. Due to unlimited modifiers could be applied to any key, the length of values is generally unknown at first, thus the size of value matrix `V` may be updated incrementally. When new instruction is pushed in, check there is any new modifier first, if not, check whether its value lies in the null space of `V`, if also not, update `V` correspondingly.

## Special Treatments of Encoding

The framework described above tries to maximize the versatility, yet still keep the work needed at minimum level. Currently, we found it works fine for turing, and it is believed it should also work for any previous and possibly future cuda instruction sets.

However, although CuAssembler tries to coordinate with any convention of assembly of `cuobjdump`, this complicated language is defined by nvidia, not us, hence there are inevitably some exceptions that cannot fit into our simple framework:

* **PLOP3**: the bits of `immLut` in `PLOP3` does not put together. For example, in `PLOP3.LUT P2, PT, P3, P2, PT, 0x2a, 0x0 ;`, the immLut `0x2a = 0b00101010`, is encoded like `0b 00101 xxxxx 010`, with other 5 bits in between. So this operand will be treated specifically with splitting the bit in advance, `LOP3` seems fine.
* **I2F, F2I, I2I, F2F for 64bit types**: 64bit datatype conversions have different opcode with respect to 32bit. But the modifier for 32bit is not explicitly displayed, then modifier such as `F64` cannot handle both the difference between `F32` and `F64` as well as the opcode change. For this case, we just appended a new modifier `CVT64` to let it work with `F64` together.
* **BRA, BRX, JMP, CALL, RET, etc.**: All the branch or jump type instructions have an operand for the target address to jump to. However, in real encoding, they all need to know the address of current instruction, and it is actually the *relative offset* to be the operand. The problem here is that the relative offset could be negative, which needs another modifier to probe the number of bits to be used. Currently, we simply modified the target address operand, and added the negative modifier if needed.
* **I2I, F2F, IDP, HMMA, etc.**: Instructions with some position dependent modifiers, e.g., `F2F.F32.F64` is not same as `F2F.F64.F32`. We just appended an extra postfix `@idx` after each modifier, such that they could be discriminated. This only works for instructions with constant number of those type modifiers(not including operand modifiers). It seems there are some instructions with variable number of modifiers, containing some position dependent modifiers, such as `HMMA` and `IDP`. Still working on it~

Thanks to those special treatments, CuAssembler is supposed to be able to re-assemble all instructions from sass dumped by cuobjdump. But there are always exceptions, well, at least there is. Currently, the only type of instruction cannot be re-assembled from `cuobjdump` is:

>`FSEL R5, R5, +QNAN , P0 ;`

In our treatment, `+QNAN` is float immediate, but its bit representation is not *UNIQUE*, there are a class of `+QNAN` defined in IEEE 754, with same exponent but arbitrary non-zero significand. Here `FSEL` seems setting the register to one special binary rather than plain `+QNAN`. But since the information is not included in the instruction itself, there is no way to recover it. For this case, we add another way of representing float immediates with every bit explicitly set, e.g., `0F3f800000`, just like the way float literals used in ptx.

According to our tests, all other type of instruction can be re-assembled with exactly the same code, just from dumped sass without any modification.

**NOTE**: Well... After more exhaustive tests, there are some other mysterious instruction cannot be recovered. such as `B2R` for turing, and `LDG/STG` for ampere. It seems some modifiers are not shown in the assembly text. And some intructions even don't show up in SASS assembly... Any instructions that cannot be fully recovered from assembly text(if any...) are not likely to work in CuAssembler, unless we will do the disassembly. Bug reports are sent to nvidia, hopefully they could be fixed...

## Instruction Assembler Repository

