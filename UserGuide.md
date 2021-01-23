CuAssembler User Guide
- [A simple tutorial for using CuAssembler with CUDA runtime API](#a-simple-tutorial-for-using-cuassembler-with-cuda-runtime-api)
  - [Start from a CUDA C example](#start-from-a-cuda-c-example)
  - [Build CUDA C into Cubin](#build-cuda-c-into-cubin)
  - [Disassemble Cubin to CuAsm](#disassemble-cubin-to-cuasm)
  - [Adjust the assembly code](#adjust-the-assembly-code)
  - [Assemble cuasm into cubin](#assemble-cuasm-into-cubin)
  - [Hack the orginal executable](#hack-the-orginal-executable)

# A simple tutorial for using CuAssembler with CUDA runtime API

We will show the basic usage of CuAssembler, by a simple `cudatest` case. CuAssembler is just an assembler, its main purpose is to generate the cubin file according to user input assembly. All device initialization, data preparation and kernel launch should be done by the user, possibly using CUDA driver API. However, it's usually more convinent to start from runtime API. Here we will demostrate the general workflow for using CUDA runtime API with CuAssembler.

The tutorial is far from complete, many basic knowledges of CUDA is needed for this trivial task. The code is not fully shown, and some commen building steps are ignored, but I think you should get the idea... If not, you are probably too early to be here, please get familiar with basic CUDA usage first~

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

Currently CuAssembler does not fully support modification of kernel args, globals (contants, texture/surface references), thus all these information(size, name, etc.) should be defined in CUDA C, and inherited from cubin. Best practice here is to make a naive working version of the kernel, with all required resources prepared. Then in assembly, only the instructions need to be modified, that's the most robust way CuAssembler can be used. 

**NOTE**: when you get into the stage of final assembly tuning, modifying the orignal CUDA C would be very unreliable, and usually rather error-prone, thus it's strongly recommended to keep all the staff unchanged in CUDA C. If you really need this, you probably have to make a big restructuring of the generated assembly. Making version control of the generated `cuasm` file may help you get through this more easily, and hopefully less painfully.

## Build CUDA C into Cubin

`nvcc` is the canonical way to build a `.cu` file into executable, such as `nvcc -o cudatest cudatest.cu`. However, we need the intermediate `cubin` file to start with. thus we will use the `--keep` option of `nvcc`, which will keep all intermediate files (such as ptx, cubin, etc.). By default, only the lowest supported SM version(by current version of NVCC) of ptx and cubin will be generated, if you need a specific SM version of cubin, you need to specify the `-gencode` option, such as `-gencode=arch=compute_75,code=\"sm_75,compute_75\"` for turing (`sm_75`). The full command may look like:

```
    nvcc -o cudatest cudatest.cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" --keep
```

Then you will get cubins such as `cudatest.1.sm_75.cubin`, under the intemediate files directory (maybe just current directory). Then we get a cubin to start with.

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

Saving those commands to a script file(e.g., bat for windows, sh for linux). Since we will need them when we want to embed the hacked cubin back to the executable, and run it as if the hacking does not happen at all.

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

## Adjust the assembly code

Most contents of `cuasm` file is copied from `nvdisasm` result of the cubin, with some supplementary ELF information explicitly recorded in text format , such as section header attributes, implicit sections(such as `.strtab/.shstrtab/.symtab`) not shown in disassembly. All the information inherited directly from the cubin should not be modified (unless have to, such the offset and size of sections). This does not mean these information cannot be automatically generated, but since NVIDIA provides no information about their conventions, probing them all would be rather pain-staking. Thus it's much safer and easier to keep them as is. Actually, most adjustment of those information (such as add a kernel, global, etc.) can be achieved by modifying the original CUDA C code, which is officially supported and much more reliable.


See an [example cuasm](TestData/CuTest/cudatest.7.sm_75.cuasm) in `TestData` for more informations.

## Assemble cuasm into cubin

Assembling cuasm into cubin is also trivial. 

```python
from CuAsm.CuAsmParser import CuAsmParser

asmname = 'cudatest.7.sm_75.cuasm'
binname = 'new_cudatest.7.sm_75.cubin'
cap = CuAsmParser()
cap.parse(asmname)
cap.saveAsCubin(binname)
```


## Hack the orginal executable 

