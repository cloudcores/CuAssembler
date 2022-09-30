- [Tutorial for using CuAssembler with CUDA runtime API](#tutorial-for-using-cuassembler-with-cuda-runtime-api)
  - [Start from a CUDA C example](#start-from-a-cuda-c-example)
  - [Blitz Version](#blitz-version)
  - [Long March Version](#long-march-version)
    - [Build CUDA C into Cubin](#build-cuda-c-into-cubin)
    - [Disassemble Cubin to Cuasm](#disassemble-cubin-to-cuasm)
    - [Adjust the assembly code in cuasm](#adjust-the-assembly-code-in-cuasm)
    - [Assemble cuasm into cubin](#assemble-cuasm-into-cubin)
    - [Hack the original executable](#hack-the-original-executable)
    - [Run or debug the executable](#run-or-debug-the-executable)

We will show the basic usage of CuAssembler, by a simple `cudatest` case. **CuAssembler** is just an assembler, its main purpose is to generate the cubin file according to user input assembly. All device initialization, data preparation and kernel launch should be done by the user, possibly using CUDA driver API. However, it's usually more convenient to start from runtime API. Here we will demonstrate the general workflow for using CUDA runtime API with CuAssembler.

This tutorial is far from complete, many basic knowledge of CUDA is needed for this trivial task. The code is not fully shown, and some common building steps are ignored, but I think you can get the idea... If not, you are probably too early to be here, please get familiar with basic CUDA usage first~

Some useful references of prerequisite knowledge:
* Basic knowledge of [CUDA](https://docs.nvidia.com/cuda/index.html), at least the CUDA C programming guide. 
* [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) and [CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html): many users just utilize those tools via IDE, but here, you will have to play with them in command line from time to time.
* ELF Format: There are many references on the format of ELF, both generic and architecture dependent, for example, [this one](http://downloads.openwatcom.org/ftp/devel/docs/elf-64-gen.pdf). Currently only **64bit** version of ELF (**little endian**) is supported by CuAssembler. 
* Common assembly directives: `nvdisasm` seems to resemble many conventions of gnu assembler. Since no doc is provided on the grammar of `nvdisasm` disassemblies, get familiar with [Gnu Assembler directives](https://ftp.gnu.org/old-gnu/Manuals/gas-2.9.1/html_chapter/as_7.html) would be helpful. Actually only very few directives are used in cuasm, look it up in this manual if you need more information. **NOTE**: some directives may be architecture dependent, you may need to discriminate them by yourself.
* CUDA PTX and SASS instructions: Before you can write any assemblies, you need to know the language first. Currently no official (at least no comprehensive) doc is provided on SASS, just [simple opcodes list](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-ref). Get familiar with [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) and its documentation will be greatly helpful to understand the semantics of SASS assemblies. 

# Tutorial for using CuAssembler with CUDA runtime API

As stated in [ReadMe](README.md), the most common usage of CuAssembler is to hack cubin with user modified assembly, which is not supported officially. However, for many cases, we still want to reuse most of the compiling steps of nvcc, and only make slight modification to the final cubin. Thus it will be very much more convenient if we can still use the hacked cubin with CUDA runtime API, rather than loading it in driver API with a new program.

Here we will show an example for hacking the cubin inside the nvcc building steps.

## Start from a CUDA C example

First we need to create a `cudatest.cu` file with enough information of kernels. You may start from any other CUDA samples with explicit kernel definitions. Some CUDA programs do not have explicit kernels written by user, instead, they may invoke some kernels pre-compiled in libraries. In this case you cannot hack the cubin by runtime API, you need to hack the library! That would be totally a different story, currently we just focus on the *user kernels*, rather than *library kernels*. An example of kernel may look like this (other lines are ignored):

```c++
__global__ void vectorAdd(const float* a, const float* b, float* c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}
```

Currently CuAssembler does not fully support modification of kernel args, globals (constants, texture/surface references), thus all these information(size, name, etc.) should be defined in CUDA C, and inherited from cubin into CuAssembler. Best practice here is to make a naive working version of the kernel, with all required resources prepared. Then in assembly, only the instructions need to be modified, that's the most robust way CuAssembler can be used. 

**NOTE**: when you get into the stage of final assembly tuning, modifying the original CUDA C would be very unreliable, and usually rather error-prone, thus it's strongly recommended to keep all the staff unchanged in CUDA C. If you really need this, you probably have to make a big restructuring of the generated assembly. Making version control of the generated `*.cuasm` file may help you get through this more easily, and hopefully less painfully.

## Blitz Version
CuAssembler offers a set of user tools to accelerate basic development steps for hacking the cubins, see section "Settings and Simple Usage" in [ReadMe](README.md) for more details on the basic usage of those scripts. With aid of those scripts, the hacking and resuming can be done very quickly.

* Step 1: copy the makefile `CuAssembler/bin/makefile` to the same dir as `cudatest.cu`. Set `BASENAME` in makefile to `BASENAME=cudatest`. Set the `ARCH` to your SM version. Add `$INCLUDE` or `$LINK` if you need other includes or links.
* Step 2: run `make d2h`. You will get 3 new files:
  * `dump.cudatest.sm_75.cubin` : the original cubin compiled from the `cudatest.cu`.
  * `dump.cudatest.sm_75.cuasm` : the disassembly of original cubin.
  * `hack.cudatest.sm_75.cuasm` : a copy of `dump.cudatest.sm_75.cuasm`, which can be modified by user.
* Step 3: modify `hack.cudatest.sm_75.cuasm`.
* Step 4: run `make hack`, this will assemble `hack.cudatest.sm_75.cuasm` to `hack.cudatest.sm_75.cubin`, and replace the original cubin with this hacked version, and then resume the building step and generate the final executable `cudatest`.
* Step 5: run `cudatest` to check your result! 


## Long March Version

### Build CUDA C into Cubin

`nvcc` is the canonical way to build a `.cu` file into executable, such as `nvcc -o cudatest cudatest.cu`. However, we need the intermediate `cubin` file to start with. Thus we will use the `--keep` option of `nvcc`, which will keep all intermediate files (such as ptx, cubin, etc.). By default, only the lowest supported SM version of ptx and cubin will be generated, if you need a specific SM version of cubin, you need to specify the `-gencode` option, such as `-gencode=arch=compute_75,code=\"sm_75,compute_75\"` for turing (`sm_75`). The full command may look like:

```
    nvcc -o cudatest cudatest.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --keep
```

Then you will get cubins such as `cudatest.1.sm_75.cubin` (probably different number), under the intermediate files directory (maybe just current directory). Then we get a cubin to start with.

**NOTE**: Sometimes `nvcc` may generate several `cubin` of different versions, and possibly an extra empty cubin of every SM version. You can check the contents by `nvdisasm`, or just judging by the file size.

Another important information from `nvcc` is that we need full building steps. Thus we use the `--dryrun` option to list all the steps invoked by `nvcc`.

```
    nvcc -o cudatest cudatest.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --dryrun
```

You may get something like this (some lines are ignored, you may have different output):

```sh
$ nvcc -o cudatest cudatest.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --dryrun
...
#$ gcc -D__CUDA_ARCH__=750 -D__CUDA_ARCH_LIST__=750 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda-11.6/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=6 -D__CUDACC_VER_BUILD__=55 -D__CUDA_API_VER_MAJOR__=11 -D__CUDA_API_VER_MINOR__=6 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "cudatest.cu" -o "/tmp/tmpxft_0000016a_00000000-9_cudatest.cpp1.ii" 
#$ cicc --c++14 --gnu_version=70500 --display_error_number --orig_src_file_name "cudatest.cu" --orig_src_path_name "temp/cudatest.cu" --allow_managed   -arch compute_75 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "tmpxft_0000016a_00000000-3_cudatest.fatbin.c" -tused --gen_module_id_file --module_id_file_name "/tmp/tmpxft_0000016a_00000000-4_cudatest.module_id" --gen_c_file_name "/tmp/tmpxft_0000016a_00000000-6_cudatest.cudafe1.c" --stub_file_name "/tmp/tmpxft_0000016a_00000000-6_cudatest.cudafe1.stub.c" --gen_device_file_name "/tmp/tmpxft_0000016a_00000000-6_cudatest.cudafe1.gpu"  "/tmp/tmpxft_0000016a_00000000-9_cudatest.cpp1.ii" -o "/tmp/tmpxft_0000016a_00000000-6_cudatest.ptx"
#$ ptxas -arch=sm_75 -m64  "/tmp/tmpxft_0000016a_00000000-6_cudatest.ptx"  -o "/tmp/tmpxft_0000016a_00000000-10_cudatest.sm_75.cubin" 
#$ fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=elf,sm=75,file=/tmp/tmpxft_0000016a_00000000-10_cudatest.sm_75.cubin" "--image3=kind=ptx,sm=75,file=/tmp/tmpxft_0000016a_00000000-6_cudatest.ptx" --embedded-fatbin="/tmp/tmpxft_0000016a_00000000-3_cudatest.fatbin.c" 
#$ rm /tmp/tmpxft_0000016a_00000000-3_cudatest.fatbin
#$ gcc -D__CUDA_ARCH_LIST__=750 -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda-11.6/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=6 -D__CUDACC_VER_BUILD__=55 -D__CUDA_API_VER_MAJOR__=11 -D__CUDA_API_VER_MINOR__=6 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "cudatest.cu" -o "/tmp/tmpxft_0000016a_00000000-5_cudatest.cpp4.ii" 
#$ cudafe++ --c++14 --gnu_version=70500 --display_error_number --orig_src_file_name "cudatest.cu" --orig_src_path_name "temp/cudatest.cu" --allow_managed  --m64 --parse_templates --gen_c_file_name "/tmp/tmpxft_0000016a_00000000-6_cudatest.cudafe1.cpp" --stub_file_name "tmpxft_0000016a_00000000-6_cudatest.cudafe1.stub.c" --module_id_file_name "/tmp/tmpxft_0000016a_00000000-4_cudatest.module_id" "/tmp/tmpxft_0000016a_00000000-5_cudatest.cpp4.ii" 
#$ gcc -D__CUDA_ARCH__=750 -D__CUDA_ARCH_LIST__=750 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS "-I/usr/local/cuda-11.6/bin/../targets/x86_64-linux/include"   -m64 "/tmp/tmpxft_0000016a_00000000-6_cudatest.cudafe1.cpp" -o "/tmp/tmpxft_0000016a_00000000-11_cudatest.o" 
#$ nvlink -m64 --arch=sm_75 --register-link-binaries="/tmp/tmpxft_0000016a_00000000-7_cudatest_dlink.reg.c"    "-L/usr/local/cuda-11.6/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-11.6/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "/tmp/tmpxft_0000016a_00000000-11_cudatest.o"  -lcudadevrt  -o "/tmp/tmpxft_0000016a_00000000-12_cudatest_dlink.sm_75.cubin"
#$ fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " -link "--image3=kind=elf,sm=75,file=/tmp/tmpxft_0000016a_00000000-12_cudatest_dlink.sm_75.cubin" --embedded-fatbin="/tmp/tmpxft_0000016a_00000000-8_cudatest_dlink.fatbin.c" 
#$ rm /tmp/tmpxft_0000016a_00000000-8_cudatest_dlink.fatbin
#$ gcc -D__CUDA_ARCH_LIST__=750 -c -x c++ -DFATBINFILE="\"/tmp/tmpxft_0000016a_00000000-8_cudatest_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_0000016a_00000000-7_cudatest_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  "-I/usr/local/cuda-11.6/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=6 -D__CUDACC_VER_BUILD__=55 -D__CUDA_API_VER_MAJOR__=11 -D__CUDA_API_VER_MINOR__=6 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -m64 "/usr/local/cuda-11.6/bin/crt/link.stub" -o "/tmp/tmpxft_0000016a_00000000-13_cudatest_dlink.o" 
#$ g++ -D__CUDA_ARCH_LIST__=750 -m64 -Wl,--start-group "/tmp/tmpxft_0000016a_00000000-13_cudatest_dlink.o" "/tmp/tmpxft_0000016a_00000000-11_cudatest.o"   "-L/usr/local/cuda-11.6/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-11.6/bin/../targets/x86_64-linux/lib"  -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group -o "cudatest" 
```

Saving those commands to a script file(e.g., `*.sh` for linux, `*.bat` for windows, remember to **uncomment** it first). We will need them when we want to embed the hacked cubin back to the executable, and run it as if the hacking does not happen at all.

### Disassemble Cubin to Cuasm

Cubin is binary, it cannot be modified directly by the user. Thus we need to disassemble it first.

**Command-Line Approach**:

The script `CuAssembler/bin/cuasm.py` provides a handy way to disassembly cubin into cuasm text form. Run `cuasm -h` for more information.

```
cuasm cudatest.sm_75.cubin
```

This will generate the disassembly file `cudatest.sm_75.cuasm`, easier to understand and edit. NOTE: the disassembly is mostly inherited from `nvdisasm`, but with some new directives for CuAssembler to assemble it back. The original disassembly of `nvdisasm` will not be recognized by CuAssembler.

**Programming Approach**:
Since CuAssembler is a python package, most of the functionalities can be reached with programming. We can create a python script of CuAssembler to disassemble the `cubin` into `cuasm`. 

```python
from CuAsm.CubinFile import CubinFile

binname = 'cudatest.sm_75.cubin'
cf = CubinFile(binname)
asmname = binname.replace('.cubin', '.cuasm')
cf.saveAsCuAsm(asmname)
```

For most cases, the command-line approach is more handy, yet the programming approach is more flexible and can support much more complex pre-processing or post-processing.

### Adjust the assembly code in cuasm

Most contents of `cuasm` file is copied from `nvdisasm` result of the cubin, with some supplementary ELF information explicitly recorded in text format , such as file header attributes, section header attributes, implicit sections(such as `.strtab/.shstrtab/.symtab`) not shown in disassembly. All these information inherited directly from the cubin should not be modified (unless have to, such the offset and size of sections, which will be done by the assembler automatically). This does not mean these information cannot be automatically generated, but since NVIDIA provides no information about their conventions, probing them all would be rather pain-staking. Thus it's much safer and easier to keep them as is. Actually, most adjustment of those information (such as add a kernel, global, etc.) can be achieved by modifying the original CUDA C code, which is officially supported and much more reliable.

See an [example cuasm](TestData/CuTest/cudatest.7.sm_75.cuasm) in `TestData` for more information. 

### Assemble cuasm into cubin

`*.cuasm` files cannot be recognized by CUDA, thus it should be assembled back to `*.cubin` and then can be used.

**Command-Line Approach**:

Assembling cuasm into cubin is also trivial:

```
cuasm cudatest.sm_75.cuasm -o new_cudatest.sm_75.cubin
```

CAUTION: the default output of `cuasm cudatest.sm_75.cuasm` may override original `cudatest.sm_75.cubin`, thus it is recommended to use a new name. To avoid un-intended overwrite, `cuasm` will create a backup `.cubin~` if necessary.

**Programming Approach**:

```python
from CuAsm.CuAsmParser import CuAsmParser

asmname = 'cudatest.7.sm_75.cuasm'
binname = 'new_cudatest.7.sm_75.cubin'
cap = CuAsmParser()
cap.parse(asmname)
cap.saveAsCubin(binname)
```

### Hack the original executable 

As soon as you get a hacked cubin, the easiest way to put it back to the executable is to mimic the behavior of the original building steps. Take a look at the output of `nvcc` with `--dryrun` option, there will be a step which looks like:

```bat
ptxas -arch=sm_75 -m64 "cudatest.ptx"  -o "cudatest.sm_75.cubin"
```

You can delete all the steps before this one (include this `ptxas` step), rename your hacked cubin to `cudatest.sm_75.cubin`, and run the rest of those building steps. That will give you an executable just like run `nvcc` directly.

Sometimes you may not need to hack all of the cubins, you can freely hack one or more `ptxas` steps, since `ptxas` just accepts one file at a time. For more convenient usage, you may also copy those steps into a makefile, and run the rebuild steps if any dependent file is modified. You can even make a script or set an environment variable to switch between the hacked version and original version.

### Run or debug the executable

If everything goes right, the hacked cubin and final executable should work as good as the original one. However, if some mismatches exist with respect to the original CUDA C file(such as kernel names, kernel arg arrangements, global contants, and global texture/surface references), the executable may not work right. That's why we should always get those information ready before hacking the cubin. Another issue is, some symbol information will be used for proper debugging. Thus you should not modify them as well (symbol offsets and sizes will be automatically updated by assembler). 

**NOTE**: debug version of cubin contains far too much information(DWARF for source line correlations...and many more), which is very difficult to process in assembler. Thus you should not use CuAssembler with debug version of cubin. That's another reason why it's recommended to work on a naive but correct version of CUDA C first. NVIDIA provides tools for final SASS level debugging (such as NSight VS version and `cuda-gdb`), there are no source code correlation in this level.
