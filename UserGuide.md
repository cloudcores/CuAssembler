CuAssembler User Guide
- [A brief instruction on format of cubin and cuasm](#a-brief-instruction-on-format-of-cubin-and-cuasm)
  - [File Structure](#file-structure)
  - [Sections and Segments](#sections-and-segments)
  - [Basic syntax of cuasm](#basic-syntax-of-cuasm)
  - [Kernel text sections](#kernel-text-sections)
  - [Limitations, Traps and Pitfalls](#limitations-traps-and-pitfalls)
- [How CuAssembler works](#how-cuassembler-works)
  - [Automatic Instruction Encoding](#automatic-instruction-encoding)
  - [Special Treatments of Encoding](#special-treatments-of-encoding)
  - [Instruction Assembler Repository](#instruction-assembler-repository)

# A brief instruction on format of cubin and cuasm

**Cubin** is an ELF format binary, thus most of its file structure will follow the generic conventions of ELF. However, there are also many CUDA specific features involved. **Cuasm** is just a text form of cubin, with most of the cubin information explicitly described with assembly directives. Most of the assembly directives will follow the same semantics of `nvdisasm` (actually most of the cuasm contents are copied from `nvdisasm` output), but there are also some new directives, helping make some information clear and explicit.

## File Structure

An ELF file contains a file header, several sections, none or several program segments. Usually the cubin files are arranged like this:

* **File Header** : The ELF file header will specify some common information, such as identifier magic number, 32bit or 64bit, section header offset, program header offset, etc. For cubin, it will also specify the version of current cubin: virtual architecture version and SM version, toolkit version etc. 
* **Section Data** : The data of every section.
* **Section Header** : Header information of every section. Defines the section name, section offset, size, flags, type, extra info, and linkage with respect to other sections, etc.
* **Segment Header** : Header information of every segment. Defines how the sections will be loaded. **NOTE**: for some `ET_REL` type of ELF, there may be no segment. They are likely to be linked to another cubin for final executable. Currently, only `ET_EXEC` type of ELF is tested, it usually contains 3 segments.

## Sections and Segments

This is a sample layout of cubin sections (debug version of cubin will have much more sections, which are not concerned here):

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

* `.shstrtab/.strtab/.symtab` : tables for section string, symbol string, and symbol entries. Currently all of them are copied from the original cubin.
* `.nv.info.*` : Some attributes associated with kernels. `cuobjdump -elf *.cubin` can show those information in human readable format. Some of those attributes should be modified when kernel text changed, some can be done by CuAssembler(such as `EIATTR_EXIT_INSTR_OFFSETS`, `EIATTR_CTAIDZ_USED`, etc.), but there are more that cannot. Some attributes are strongly correlated to the offset of some instructions, CuAssembler utilizes a special form of label to handle this kind of attributes, which is necessary to make them work when the instruction sequence is changed.
* `.rel.*` : Relocations. Relocation section should work with the associated section, such as `.rel.abc` to `.abc`. Relocation is a special mechanism which allows runtime initialization of some symbols unknown during compile-time, such as some global constants and function entries. 
* `.nv.constant#.*` : constant memory contents for global constants or kernel dependent constants. The actual arrangement of contant memories may vary with respect to SM version(or even toolkit version?), thus you'd better check it in original SASS code generated with CUDA C. In the example above, constant bank 3 `.nv.constant3` is for global, referred by `c[0x3][###]`. Bank 2 is for compiler generated constants, and bank 0 for kernel arguments and grid/block constants. Both of them are kernel dependent.
* `.text.*` : Kernel instruction sections. Most of the modification should be done to these sections.
* `.nv.shared.*` : Nobits sections. I don't find the shared memory is runtime initializable, thus seems they are only for space allocation.

## Basic syntax of cuasm
Most of the syntax of cuasm will follow the convention of `nvdisasm`, but since `nvdisasm` does not show all the information of cubin. We need more syntax to describe the file more specifically.

**Comments**:

C style comments `/* ... */` and cpp style comments `// ...` are supported. A special form of branch target annotation `(* ... *)` will also be treated as comments. They will all be replaced to spaces. **NOTE**: Currently no cross line comments are allowed, all comments should be in the same line.

**Directives**:
A directive is a predefined keyword, usually starts with a dot `.`. Current list of supported directives defined by `nvdisasm`:

| Directive          | Notes          |
|--------------------|----------------|
| `.headerflags`*     | set ELF header |
| `.elftype`*         | set ELF type |
| `.section`*         | declare a section |
| `.sectioninfo`*     | set section info |
| `.sectionflags`*    | set section flags |
| `.sectionentsize`*  | set section entsize |
| `.align`           | set alignment |
| `.byte`            | emit bytes |
| `.short`           | emit shorts |
| `.word`            | emit word (4B?) |
| `.dword`           | emit dword (8B?) |
| `.type`*           | set symbol type |
| `.size`*           | set symbol size |
| `.global`*          | declare a global symbol |
| `.weak`*            | declare a weak symbol |
| `.zero`            | emit zero bytes |
| `.other`*           | set symbol other  |

Directives annotated with an asterisk are currently not really functional, since the contents are actually copied from original cubin. CuAssembler defined some new internal directives(prefixed with `.__`) that keeps those information unchanged. 

|  |                                |
|--|--------------------------------|
| ELF header | |
| | .__elf_ident_osabi |
| | .__elf_ident_abiversion |
| | .__elf_type |
| | .__elf_machine |
| | .__elf_version |
| | .__elf_entry |
| | .__elf_phoff |
| | .__elf_shoff |
| | .__elf_flags |
| | .__elf_ehsize |
| | .__elf_phentsize |
| | .__elf_phnum |
| | .__elf_shentsize |
| | .__elf_shnum |
| | .__elf_shstrndx |
| Section header | |
| | .__section_name |
| | .__section_type |
| | .__section_flags |
| | .__section_addr |
| | .__section_offset |
| | .__section_size |
| | .__section_link |
| | .__section_info |
| | .__section_entsize |
| Segment header | |
| | .__segment |
| | .__segment_offset |
| | .__segment_vaddr |
| | .__segment_paddr |
| | .__segment_filesz |
| | .__segment_memsz |
| | .__segment_align |
| | .__segment_startsection |
| | .__segment_endsection |

**Labels and Symbols**:

A **label** is just an identifier(may include `.`, `$`, and any word character) followed by a colon `label:`, such as:

```
  _Z10local_testiiPi:
  .L_203:
  __cudart_i2opi_f:
  $str:
  $_Z7argtestPiS_S_$_Z2f1ii:
```

Labels can be used for reference when the real offset should be filled.

A **symbol** is a special label that may be visible externally, i.e., give current address when the module is loaded. A symbol can be defined as:

```asm
.global         _Z10local_testiiPi
.type           _Z10local_testiiPi,@function
.size           _Z10local_testiiPi,(.L_203 - _Z10local_testiiPi)
.other          _Z10local_testiiPi,@"STO_CUDA_ENTRY STV_DEFAULT"
_Z10local_testiiPi:
```

The last line is actually a label(with **same identifier**) that tells the location of the symbol. Symbols without corresponding label are usually defined externally, such as `vprintf` and some other internal non-inline device functions. Every symbol has an entry in `.symtab` section. `cuobjdump -elf *.cubin` can show those entries in human readable form.

**CAUTION**: 
1. There are far too many treatments needed for different types of symbols. I don't want to follow those tedious or even troublesome treatments(and probably hidden convention privately defined by NVIDIA). Since most of those symbols can be prepared by CUDA C, I just copied them from original cubin, but still keeping those statements legal yet nonfunctional.
2. Cubin of ELF type `ET_REL` may have more types of symbol, probably for later linking? It's quite difficult to support them all, thus `ET_REL` type of ELF will not be supported.

## Kernel text sections

Kernel text sections are the most frequently part to be modified for CuAssembler. Here we use a simple kernel to show some basic conventions of `cuasm`.

```c++
__constant__ int C1[11];       // C1 will be stored in constant memory
__device__ int GlobalC1[7];    // GlobalC1 will be stored in device memory (RW), loaded with relocated address
__global__ void simpletest(const int4 VAL, int* v) // contents of VAL and address of v will be stored in constant memory
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int a = v[idx]*VAL.x + GlobalC1[idx%16];

    // SHFL is an instruction needs an associated .nv.info attribute.
    a = __shfl_up_sync(0xffffffff, a, 1);  

    if (VAL.z > 0) // predicated statement
        a += C1[VAL.y]; 
    v[idx] = a;
}
```

The corresponding `cuasm` region of codes will be as follows(generated by CUDA toolkit 11.1, SM_75):

```
// --------------------- .text._Z10simpletest4int4Pi      --------------------------
.section	.text._Z10simpletest4int4Pi,"ax",@progbits
.__section_name         0x35b 	// offset in .shstrtab
.__section_type         SHT_PROGBITS
.__section_flags        0x6
.__section_addr         0x0
.__section_offset       0x4500 	// maybe updated by assembler
.__section_size         0x200 	// maybe updated by assembler
.__section_link         3
.__section_info         0xc000030
.__section_entsize      0
.align                128 	// equivalent to set sh_addralign
  .sectioninfo	@"SHI_REGISTERS=12"
  .align	128
        .global         _Z10simpletest4int4Pi
        .type           _Z10simpletest4int4Pi,@function
        .size           _Z10simpletest4int4Pi,(.L_228 - _Z10simpletest4int4Pi)
        .other          _Z10simpletest4int4Pi,@"STO_CUDA_ENTRY STV_DEFAULT"
_Z10simpletest4int4Pi:
.text._Z10simpletest4int4Pi:
    [B------:R-:W-:Y:S08]         /*0000*/                   MOV R1, c[0x0][0x28] ;
    [B------:R-:W0:-:S01]         /*0010*/                   S2R R2, SR_CTAID.X ;
    [B------:R-:W-:-:S01]         /*0020*/                   UMOV UR4, 32@lo(GlobalC1) ;
    [B------:R-:W-:-:S01]         /*0030*/                   MOV R9, 0x4 ;
    [B------:R-:W-:-:S01]         /*0040*/                   UMOV UR5, 32@hi(GlobalC1) ;
    [B------:R-:W0:-:S01]         /*0050*/                   S2R R3, SR_TID.X ;
    [B------:R-:W-:-:S01]         /*0060*/                   MOV R4, UR4 ;
    [B------:R-:W-:-:S02]         /*0070*/                   IMAD.U32 R5, RZ, RZ, UR5 ;
    [B0-----:R-:W-:Y:S05]         /*0080*/                   IMAD R2, R2, c[0x0][0x0], R3 ;
    [B------:R-:W-:Y:S04]         /*0090*/                   SHF.R.S32.HI R3, RZ, 0x1f, R2 ;
    [B------:R-:W-:Y:S04]         /*00a0*/                   LEA.HI R3, R3, R2, RZ, 0x4 ;
    [B------:R-:W-:Y:S05]         /*00b0*/                   LOP3.LUT R3, R3, 0xfffffff0, RZ, 0xc0, !PT ;
    [B------:R-:W-:-:S02]         /*00c0*/                   IMAD.IADD R7, R2.reuse, 0x1, -R3 ;
    [B------:R-:W-:Y:S04]         /*00d0*/                   IMAD.WIDE R2, R2, R9, c[0x0][0x170] ;
    [B------:R-:W-:Y:S04]         /*00e0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;
    [B------:R-:W2:-:S04]         /*00f0*/                   LDG.E.SYS R0, [R2] ;
    [B------:R-:W2:-:S01]         /*0100*/                   LDG.E.SYS R5, [R4] ;
    [B------:R-:W-:Y:S04]         /*0110*/                   MOV R6, c[0x0][0x168] ;
    [B------:R-:W-:Y:S12]         /*0120*/                   ISETP.GE.AND P0, PT, R6, 0x1, PT ;
    [B------:R-:W-:Y:S06]         /*0130*/               @P0 IMAD R6, R9, c[0x0][0x164], RZ ;
    [B------:R-:W0:-:S01]         /*0140*/               @P0 LDC R6, c[0x3][R6] ;
    [B--2---:R-:W-:Y:S08]         /*0150*/                   IMAD R0, R0, c[0x0][0x160], R5 ;
.CUASM_OFFSET_LABEL._Z10simpletest4int4Pi.EIATTR_COOP_GROUP_INSTR_OFFSETS.#:
    [B------:R-:W0:-:S02]         /*0160*/                   SHFL.UP PT, R7, R0, 0x1, RZ ;
    [B0-----:R-:W-:Y:S08]         /*0170*/               @P0 IMAD.IADD R7, R7, 0x1, R6 ;
    [B------:R-:W-:-:S01]         /*0180*/                   STG.E.SYS [R2], R7 ;
    [B------:R-:W-:-:S05]         /*0190*/                   EXIT ;
.L_20:
    [B------:R-:W-:Y:S00]         /*01a0*/                   BRA `(.L_20);
.L_228:
```

Here are some explanations:

* `.section	.text._Z10simpletest4int4Pi,"ax",@progbits` declares a section with specified name, flags, type. `_Z10simpletest4int4Pi` is a **mangled** name of `void simpletest(const int4 VAL, int* v)`, you can use `c++filt` or `cu++filt` to demangle it to the original form. If you don't want the mangle treatment, use `extern "C"` to embrace the declaration (actually, strongly not recommended).
* `.__section_*` directives: internal directives to define the section header attributes. NVIDIA seems have some of their own internal flags. Users are not likely to care about these, hence they are usually kept as is.
* `.align 128` set current section to 128B alignment. Which means last section may need some padding if the offset of this section is not a multiple of 128B.
* `.sectioninfo	@"SHI_REGISTERS=12"`: set register numbers used in current kernel. **NOTE**: For Turing and Ampere, 2 extra GPRs are occupied for some unknown reasons. Thus if the largest GPR number you used in your kernel is `R20`, you need to set `@"SHI_REGISTERS=23"` (GPR numbers from 0, `R20` means 21 used, plus 2 extra, that's 23). The maximum GPR number is 255 (the encoding of `R255` is occupied by `RZ`), which means `R252` is the largest GPR number used in kernel text.
* For kernels utilize block-wide barriers(such as `__syncthreads()`), there may be another attribute specifying number of barriers required, such as `.sectionflags @"SHF_BARRIERS=1"`. Currently a kernel can use up to 16 barriers.
* `.global _Z10simpletest4int4Pi`: a symbol is defined for current kernel. It will be used for both function exporting (global symbol is visible externally, which means it's accessible via driver API `cuModuleGetFunction`), and possibly debugging (see `.debug_frame` section). As stated before, symbols are all kept as is. 
* `[B------:R-:W-:Y:S08]         /*0000*/                   MOV R1, c[0x0][0x28] ;`: this is the canonical form of instruction line. A control code, a commented address in hex, and then the instruction assembly string. The text form of control codes is slightly different from the one used in [maxas by Scott Gray](https://github.com/NervanaSystems/maxas/wiki/Control-Codes). Here, the control codes are splitted into 6 fields seperated with colon `:`:
    - **Reuse flags**(Deprecated!!! reuse will be set according to `.reuse` modifier of the assembly text.): the 4bit reuse flags indicate the value of current slot of GPR will be re-read by later instructions. There are at least 3 slots (possibly 4? Never see the forth bit set...) of reuse caches, with each bit set to `R` for reuse, and `-` for none. It seems reuse caches only work for ALU instructions, with each slot corresponding to an operand position, which will be suffixed by `.reuse`(**NOTE**: CuAssembler will not care about the `.reuse` suffix in instruction string, only sequences in control codes part matter). But some inconsistency can also be found, such as:
  
      >  [-R--:B------:R-:W-:-:S02]         /*09c0*/                   IABS R7, R5.reuse ;
      
    Reuse of GPR will not only help mitigating the register bank conflict, and may also reduce some power consumption.  

    - **Barrier on scoreboard**:There are 6 scoreboards(called *dependency barrier* in maxas), numbered from 0 to 5 (1-6 in maxas), which are also consistent with `DEPBAR` scoreboard operands, such as `SB0` and `SB5`. This barrier field has 6 bits, one for each scoreboard. **NOTE**: instead of showing the aggregate number as in maxas, here all bits are unpacked, each bit will show wait(the scoreboard number) or no-wait(`-`), for better visual inspectation with respect to the instruction setting corresponding scoreboards. There may be multiple scoreboards to barrier, such as `B01--4-` means wait until scoreboards `0,1,4` are all cleared. 
    - **Set scoreboard for reading**: `R#`, set a scoreboard (in number) to hold contents of some source GPR operands. Usually for memory instructions.
    - **Set scoreboard for writing**: `W#`, set a scoreboard (in number) to prevent reading the destination GPR until it's ready. This is used for variable latency instructions, such as memory load, double precision instruction, transendental function instruction, S2R instructions, etc. The scoreboard dependency can be resolved by not only the barrier field of control codes, but also by `DEPBAR` instruction.
    - **Yield**: Whether try to yield to another warp. This bit may have different meaning for different instruction context. `Y` for try to yield to another eligible warp, `-` for no yield.
    - **Stall count**: Stall the instruction issue for a certain number of clock cycles(in digital number, 0~15). The yield field and stall count field may have different meaning for different instruction, which is not quite clear since no official information is disclosed.
  
* A special label `.CUASM_OFFSET_LABEL._Z10simpletest4int4Pi.EIATTR_COOP_GROUP_INSTR_OFFSETS.#`: for every kernel, there will be some associated NVINFO attributes. `OFFSETS` type of attributes will be generated for some special kind of instructions. Since rules of such instructions are far from complete yet, some of them cannot be handled by CuAssembler now. Thus we just add a special offset label in the form of `.CUASM_OFFSET_LABEL.{KernelName}.{AttrName}.#`, then this offset will be appended to corresponding NVINFO attributes list. For cuasm generated from cubin, those labels will be appended automatically. This treatment will help mitigate some manual works when NVINFO support is not complete, but still do not want to edit the NVINFO section by hand.

Some maybe useful conventions of CUDA SASS:
* Integer immediate always in hex. That is, `0x1` means integer 1, `1` means float 1 (precision will depend on the opcode).  
* Local labels will be `.L_###`, global labels (symbols) will usually have a name in `.symtab`. The local label address can be obtained and filled by assembler directly, such as `BRA ``.L0` for `.L0=0x1000` will be translated to `BRA 0x1000`. But an address of symbol may need an relocation, which may be set by the program loader. Such as `MOV R2, 32@lo(flist) ;` or `CALL.REL.NOINC R6` ``(_Z7argtestPiS_S_) ;`` will be filled with zero by assembler, but also generate a relocation entries in corresponding section. The address will be filled by the program loader.
* Constant memory as ALU operands does not support GPR indexing, it is only allowed for `LDC` instruction.
* And more...

## Limitations, Traps and Pitfalls

* Parser for instructions are not quite robust, some syntax errors cannot be identified.
* No range check for all types, such as GPR index, barrier index, scoreboard index, float immediates, especially for integer immediates (as ALU operands, memory offsets, etc.).
* Some instructions may have some restrictions of operands. Such as 64bit GPR address should start from even GPR index, address of some types should be aligned, etc. Currently it's user's obligation to guarantee the correctness.
* Some hidden rules may exist for the combination of modifiers, which means the modifiers may not all work independently. However, we donot have a list of them, thus, we leave this work to user.
* Section info and symbols are not modifiable (In the future, appending may be supported...). The reason have been stated several times: just keep all the hidden conventions as is, use CUDA C to generate those information.

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
* **Scoreboard Set**: This type is only for instruction `DEPBAR` for setting the set of scoreboards to be waited, such as `{1,3}`. Currently there are 6 scoreboards, with each value corresponding to 1bit. Scoreboards count waited in control codes should be 0, yet `DEPBAR` is capable of waiting a scoreboard with non-zero number. For example, 8 requests is sent to scoreboard 5, every request will increment SB5, and every complete request will decrement SB5. If we only need 3 of them to be back, we can just wait the scoreboard dropping to `8-3=5`, which can be achieved by `DEPBAR.LE SB5, 0x5 ;`. **NOTE**: the comma inside the brakets will affect the splitting of operands, thus the scoreboard sets will be translated to an indexed type `SBSET#` during parsing. The label is `SBSET`.
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

CuAssembler needs abundant inputs(with enough diversity) to build all the matrices for instruction encoding. Currently, with every release of CUDA toolkit, a bunch of libraries with buildin kernels are provided (usually in `bin` directory of CUDA installation path, suffixed `.dll` for windows, and `.so` for linux). The user can dump the sass to file of specific version as follows:

```
  cuobjdump -sass -arch sm_75 cublas64_11.dll > cublas64_11.sm_75.sass
```

**NOTE**: `cuobjdump` doesnot have an option to save the result to file, it always dumps to `stdout`, thus you need to redirect result to file if you want to save it.


In the dumped SASS file, You will have a long list of kernel codes, such as:

```
Fatbin elf code:
================
arch = sm_75
code version = [1,7]
producer = <unknown>
host = windows
compile_size = 64bit

	code for sm_75
		Function : _Z7argtestPiS_S_
	.headerflags    @"EF_CUDA_SM75 EF_CUDA_PTX_SM(EF_CUDA_SM75)"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;               /* 0x00000a00ff017624 */
                                                                                         /* 0x000fd000078e00ff */
        /*0010*/                   ULDC.64 UR36, c[0x0][0x160] ;                         /* 0x0000580000247ab9 */
                                                                                         /* 0x000fe20000000a00 */
        /*0020*/                   IADD3 R1, R1, -0x28, RZ ;                             /* 0xffffffd801017810 */
                                                                                         /* 0x000fe20007ffe0ff */
  ...
```


In CuAssembler, `CuInsFeeder` class can read this SASS file and iteratively yields instructions, including the address, instruction code, instruction string, control codes. `cuobjdump` utilizes almost the same syntax as `nvdisasm`, but no explicit labels or symbols are displayed, thus this format can not only work for instruction gathering, but also for assembling from `nvdisasm` assemblies. 

`CuInsParser` will read in the instruction string and address, and then parse it into intruction value vector and modifier set. A sample code snippet:

```python
fname = r'TestData\CuTest\cudatest.sm_75.sass'
feeder = CuInsFeeder(fname)

cip = CuInsParser(arch='sm_75')

for  addr, code, asm, ctrl in feeder:
    print('0x%04x :   0x%06x   0x%028x   %s'% (addr, ctrl, code, asm))

    ins_key, ins_vals, ins_modi = cip.parse(asm, addr, code)
    print('    Ins_Key = %s'%ins_key)
    print('    Ins_Vals = %s'%str(ins_vals))
    print('    Ins_Modi = %s'%str(ins_modi))
```

This may yield a list such as:

```
0x0000 :   0x0007e8   0x0000078e00ff00000a00ff017624   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
    Ins_Key = IMAD_R_R_R_cAI
    Ins_Vals = [7, 1, 255, 255, 0, 40]
    Ins_Modi = ['0_IMAD', '0_MOV', '0_U32']
0x0010 :   0x0007f1   0x000000000a000000580000247ab9   ULDC.64 UR36, c[0x0][0x160] ;
    Ins_Key = ULDC_UR_cAI
    Ins_Vals = [7, 36, 0, 352]
    Ins_Modi = ['0_ULDC', '0_64']
0x0020 :   0x0007f1   0x000007ffe0ffffffffd801017810   IADD3 R1, R1, -0x28, RZ ;
    Ins_Key = IADD3_R_R_II_R
    Ins_Vals = [7, 1, 1, -40, 255]
    Ins_Modi = ['0_IADD3', '3_NegIntImme']
0x0030 :   0x000751   0x00000c1ee90000000024ff057981   LDG.E.SYS R5, [UR36] ;
    Ins_Key = LDG_R_AURI
    Ins_Vals = [7, 5, 36, 0]
    Ins_Modi = ['0_LDG', '0_E', '0_SYS']
```

The CuAssembler class `CuInsAssembler` is obligated to encode the instruction according to the value vector and modifier set. Since every instruction key has different value meaning and different modifier set, an instance of `CuInsAssembler` will only handle one key. An instance of `CuInsAssemblerRepos` class holds a repository for all known instruction keys. Given an SASS file source, `CuInsAssemblerRepos` can build the repository with instructions therein, and save the result to a file for later use:

```python
sassname = 'cublas64_11.sm_75.sass'
arch = 'sm_75'
feeder = CuInsFeeder(sassname, arch=arch)   # initialize a feeder
repos = CuInsAssemblerRepos(arch=arch)      # initialize an empty repos
repos.update(feeder)                        # Update the repos with instructions from feeder
repos.save2file('Repos.'+arch+'.txt')       # Save the repos to file, may be loaded back later
```

Building repository is usually rather time-consuming, thus a prebuilt repository is available in the `InsAsmRepos` directory (The coverage of `sm_75` is good, but poor for `sm_61` and `sm_86`, and probably wrong treatments). `CuInsAssemblerRepos` also provides subroutines to update, verify, and merge the repository from a new SASS file, or even from another repository. 

**NOTE**: Since there are quite a lot architecture dependent treatments needed, `CuInsFeeder`, `CuInsParser`, `CuInsAssembler`, `CuInAssemblerRepos` are all architecture dependent. You should not mix different SM versions of them. Some SM versions are quite close(such as maxwell and pascal), but it's still recommended to make seperate instances for them.