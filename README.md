# CuAssembler: An unofficial CUDA assembler

## What is CuAssembler

**CuAssembler** is an unofficial assembler for nvidia cuda. It's an assembler, reads assemblies(sass) and writes machine codes(in cubin). It's not another compiler just like officially provided by nvidia such as `nvcc` for cuda c, and `ptxas` for ptx.

The aim of CuAssembler is to bridge the gap between `ptx`(the lowest level officially supported and documented by nvidia) and the machine code. Some similar packages include `asfermi` and `maxas`, which can only handle some of cuda instruction sets. CuAssembler currently only supports `Turing` instruction set, but the mechanism could be easily extended to older and possibly future cuda instruction sets, since most of the instruction sets could be probed automatically.

**NOTE**: This library is still in its infancy, there are still a lot of work to be done. Interfaces and architectures are subject to change, use it at your own risk.

## When and how should CuAssembler be used

Many cuda users will inspect the generated sass code by `cuobjdump` after doing optimization of cuda c code. The easiest way to tune the sass code is to modify cuda c code itself, and then recheck the generated sass code again. For many cases, this will lead you to good enough codes. However, for those ninja programers that want to optimize the codes down to every instruction, it would be rather upset when they cannot command the compiler to generate the code they want. An alternative tuning method is to modify the intermediate ptx code, which is full of vague variables particularly tedious and difficult to follow, and the generated machine codes are still not always satisfying. CuAssembler allows the user to tune the generated sass code directly.

It should be emphasized that, for most cuda programmers, cuda c (sometimes ptx) is always the first choice. It is full featured with great deal of compiling optimizations, officially supported and well documented by nvidia. They know best of their hardware, hence the compiler is also capable of doing some architecture specific optimizations. When the generated sass code is far from expected, you are very likely to have plenty of space for high level languages to play with. There are also large amount of communities and forums which could turn to for help. Playing with assemblies is pretty pain-staking comparing with high level languages, you need to worry about everything that could be done automatically by the compiler. It becomes an eligible option only when you are already quite familiar with cuda c and ptx, and have played all the tricks you know to optimize, but still find the generated codes are not satisfying. Even in this case, it's still much more convenient to start with cuda c or ptx, and then do some minor amendments based on the generated sass. This is the main usage CuAssembler is designed for: providing an option for minor adjustment of the generated machine codes, which is not possible by official tools.

Another important usage of CuAssembler is for micro-benchmarking, i.e., probing some specific details of micro-architecture by some specifically designed small programs. Good code optimization usually needs quite deep understanding of the hardware, especially performance related figures, such as the latency and throughput of different instructions, the cache hierarchy, the latency and throughput of every level of caches, cache replacement policies, etc. Many micro-benchmarking could be done using cuda c, but its more straightforward and flexible when using assemblies, since you can not only arrange the instructions in any order you want, but also set the control codes directly, which is no way to be done in cuda c or ptx.

As an assembler, CuAssembler simply translates the assemblies to machine codes literally, and then embeds them to cubin so it can be loaded and executed. It's programers' responsibility to guarantee the correctness of the code semantically, such as explicit register allocation, proper arrangement of instructions, and correct usage of registers (e.g., register pair for 64bit variables always starts from even). So get familiar with those conventions first, otherwise it's not possible to write legal assembly codes, and this kind of error will be far from conspicuous to catch. Nevertheless, legal assemblies does not imply legal program. There are many kinds of resources involved in cuda program, such as general purpose registers, predicates, shared memories, and many others. They should match with the hardware configurations, and should be eligible for launching the specified dimension of blocks. Checking rigorous correctness of the whole program needs comprehensive understanding of the launch model and instruction set, both grammatically and semantically, far from possible without official support. So, it is left to the user to guarantee the correctness of the program, with very limited help from the assembler.

## How does CuAssembler work

Most work of assembler is to encode the instruction. For turing, every instruction is 128bit, split into two 64bit lines in `cuobjdump`. For example, the instruction:

```
    /*1190*/    @P0 FADD.FTZ R13, -R14, -RZ ;    /* 0x800000ff0e0d0221 */
                                                 /* 0x000fc80000010100 */
```

Here is the nomenclature of CuAssembler on how these fields will be called: `/*1190*/` is the instruction *address*. `@P0` is the *predicate*. `FADD` is the type of operation (referred as *op*): single precision float addition. `.FTZ` is a *modifier* of `FADD`, means flush-to-zero when any of the inputs is denormal (inf or nan). `R13`, `-R14`, `-RZ` are the *operands* of `FADD`, means `R13 = (-R14) + (-RZ)`. `RZ` is an register that always yields 0. *Modifier* is not only for the op, also includes anything that can modify the original semantics of the operands, such as the minus "-" or absolute "|*|".

Every field will encode some bits of the instruction. Three operands (`R13`, `-R14`, `-RZ`) are all of type register, so those fields will be not only depend on the content, but also position dependent. Thus the final code can be written as sum of the encoding of every field:

>`c = c("@P0") + c("FADD") + c("FTZ") + c("0_R13") + c("1_-R14") + c("2_-RZ")`.

The minus in operand `-R14`, `-RZ` can also be considered as negative modifier "Neg", and in Turing, `RZ` is always an alias of `R255`. Any other modifier for operands (currently known: "!" for predicate not, "-" for numerical negative, "|" for numerical abs, "~" for bitnot, and some bit field or type specifiers ".H0_H0", ".F32", etc.) will also be striped as separate fields. Hence the code becomes:

>`c = c("@P0") + c("FADD") + c("FTZ") + c("0_R13") + c("1_Neg") + c("1_R14") + c("2_Neg") + c("2_R255")`.

Now the problem becomes how to encode those elemental fields. We separate the encoding of every field into two parts, `Code = Value*Weight`, in which `Value` only depends on the content, `Weight` only depends on the position where the element appears.

For turing architecture, we have these elemental operands, each with some values defined, and with a *label* to identify the operand type:

* **Indexed type**: Indexed type is a type prefix followed by a positive integer index, such as registers `R#`, predicates `P#`, uniform registers and predicates `UR#` and `UP#`, barriers `B#`, and synchronizing barriers `SB#`. The value of indexed type is just the index. The label is just the prefix.
* **Address**: Memory address in a square bracket `[0x####]`. Inside the bracket, there could also be an offset specified by register: `[R#+0x####]`, or even more complicated: `[UR#+R#.X16+0x####]`. The value of address could be a list, including the value of the register and the offset. The label is `A` followed by labels inside the bracket, such as `R` for register only, `RI` for register+immediate offset, `I` for immediate only. E.g., `[UR5+R8.X16+0x10]` will have value list `[5, 8, 16]`, and label `AURRI`, `.X16` will be striped into modifiers.
* **Constant memory**: Constant memory `c[0x##][0x####]`, first bracket for constant memory bank, second for memory address. The value of constant memory is the list of the constant bank, and the value of the memory address. The label is `cA` followed by the label of the memory address of second bracket.
* **Integer immediate**: Integer immediate such as `0x0`. The value is just the bit representation of the integer. NOTE: the negative sign should be treated as an modifier, since we don't know how many bits will the value takes. The label is `II`.
* **Float immediate**: Float immediate such as `5`, `2.0`, `-2.34e-3`. The value of float immediate is just its binary representation, depend on the precision(32bit or 16bit, 64bit not found yet). Denormals such as `+Inf`, `-Inf` and `QNAN` are also possible. The label is `FI`.
* **Barset**: Deprecated instruction `DEPBAR` for setting the set of barrier to be waited, such as `{1,3}`. Currently there are 6 barriers, with each value corresponding to 1bit. The label is `BARSET`.
* **Label**: Any other type not included above. Usually a string, such as `SR_TID.X`, `SR_LEMASK`, `3D`, `ARRAY_2D`, etc. The value of label is quite like the modifier, its value will depend on the context. Usually we set the value to **1**, and let the weight be the real encoding. It's label is just itself.

Then we can obtain the value list of the example instruction `@P0 FADD.FTZ R13, -R14, -RZ`:

>`V = [0, 1, 1, 13, 1, 14, 1, 255]`,

the weight list:

>`w = [w("@P0"), w("FADD"), w("FTZ"), w("0_R13"), w("1_Neg"), w("1_R14"), w("2_Neg"), w("2_R255")]`

 is to be determined. The interesting part is, if we dump instructions with `cuobjdump`, the value lists of every instruction would be readily available and the answer `c = v.*w` is already known! Providing we can gather enough instructions of the same type, we will be able to solve the `w` with linear algebra equations `c = V*w` !

Then what kind of instructions are of the same type? Theoretically, you can always keep value as singleton `1`, and merge all modifiers into one, then let weight be the code! In this case, only instructions in your dictionary can be assembled! It definitely requires too much space, and has too many drawbacks. We should search for some patterns that maximize versatility, yet still minimize the requirement of known instructions input.

For every instruction, minimum length of values is two (one for predicate, one for op such as `FADD, IMAD, BRA`) plus number of non-fixed valued operands (i.e., not labels). Therefor we put instructions of the same operation with same number and type of operands into the same categories, and label it by connecting them with underline, such as `FFMA_R_R_R_R`, `IMAD_R_R_II_R`, `FADD_R_R_cAI`, which is called **Key** of this type of instructions. Then we can gather all possible known instruction encodings to solve the unknown weights corresponding to the **Key** of the instruction type.

Solving the weights is usually trivial, as long as you can collect all the instructions you want. Sadly enough, it's not always possible. In case we didn't gather enough instructions, `V` will be rectangular, and solving each element of `w` is impossible. But luckily, it's not always necessary to have all the weights known! We only need to make sure the value list of the instruction `v` to be assembled should be a linear combination of rows of `V`. This is equivalent to check whether `v` lies in the null space of `V`. The interesting thing is, although in this case there are infinite solutions for `V*w = c`, but any of it will give the same result for the code to be assembled: `v.*w`!

This also provides a hint on how `V` could be constructed. Due to unlimited modifiers could be applied to any key, the length of values is generally unknown at first, thus the size of value matrix `V` may be updated incrementally. When new instruction is pushed in, check there is any new modifier first, if not, check whether its value lies in the null space of `V`, if also not, update `V` correspondingly.

## Special treatments

The framework described above tries to maximize the versatility, yet still keep the work needed at minimum level. Currently, we found it works fine for turing, and it is believed it should also work for any previous and possibly future cuda instruction sets.

However, although CuAssembler tries to coordinate with any convention of assembly of `cuobjdump`, this complicated language is defined by nvidia, not us, hence there are inevitably some exceptions that cannot fit into our simple framework:

* **PLOP3**: the bits of `immLut` in `PLOP3` does not put together. For example, in `PLOP3.LUT P2, PT, P3, P2, PT, 0x2a, 0x0 ;`, the immLut `0x2a = 0b00101010`, is encoded like `0b 00101 xxxxx 010`, with other 5 bits in between. So this operand will be treated specifically with splitting the bit in advance, `LOP3` seems fine.
* **I2F, I2I, F2F for 64bit types**: 64bit datatype conversions have different opcode with respect to 32bit. But the modifier for 32bit is not explicitly displayed, then modifier such as `F64` cannot handle both the difference between `F32` and `F64` as well as the opcode change. For this case, we just appended a new modifier `CVT64` to let it work with `F64` together.
* **BRA, BRX, JMP, CALL, RET, etc.**: All the branch or jump type instructions have an operand for the target address to jump to. However, in real encoding, they all need to know the address of current instruction, and it is actually the *relative offset* to be the operand. The problem here is that the relative offset could be negative, which needs another modifier to probe the number of bits to be used. Currently, we simply modified the target address operand, and added the negative offset if necessary.
* **I2I, F2F, IDP, HMMA, etc.**: Instructions with some position dependent modifiers, e.g., `F2F.F32.F64` is not same as `F2F.F64.F32`. We just appended an extra postfix `@idx` after each modifier, such that they could be discriminated. This only works for instructions with constant number of op modifiers(not including operand modifiers). It seems there are some instructions with variable number of modifiers, containing some position dependent modifiers, such as `HMMA` and `IDP`. Still working on it~

Thanks to those special treatments, CuAssembler is supposed to be able to re-assemble all instructions from sass dumped by cuobjdump. But there are always exceptions, well, at least there is. Currently, the only type of instruction cannot be re-assembled from cuobjdump is:

>`FSEL R5, R5, +QNAN , P0 ;`

In our treatment, `+QNAN` is float immediate, but its bit representation is not *UNIQUE*, there are a class of `+QNAN` defined in IEEE 754, with same exponent but arbitrary non-zero significand, which could be used to identify the source of nan. Here `FSEL` seems setting the register to one special binary rather than plain `+QNAN`. But since the information is not included in the instruction itself, there is no way to recover it. For this case, we add another way of representing float immediates with every bit explicitly set, e.g., `0F3f800000`, just like the way float literals used in ptx.

According to our tests, all other type of instruction can be re-assembled with exactly the same code, just from dumped sass without any modification.

## A short HOWTO

CuAssembler is not designed for creating CUDA program from scratch, it has to work with other CUDA toolkits. A good start point of cubin is needed, maybe generated by `nvcc` from CUDA C using option `--keep`, or `ptxas` from hand-written or tuned PTX codes. Currently `nvcc` doesn't support resuming the linking with modified cubin (not likely even in the future, due to its vulnerability), therefore the cubin can only be loaded and launched with driver api. Remember to keep other optimization works done before coding with CuAssembler, since any modification of the input cubin may invalidate the modification done in CuAssembler, then you may need to redo all the work again.

For small test programs, it is *planed* to support starting with cuda c in python (like `pycuda`), modifying the cubin, and then launching from python with driver api inputs automatically prepared from cuda c codes.

### Prerequisites

* **CUDA toolkit 10+**: Version 10+ is needed to support `sm_75` (turing instruction sets).
* **Python 3.7+**: Previous python versions may also supported, but not tested yet.
* **Sympy 1.4+**: Integeral (or rational) matrices with arbitrary precision needed by solving the LAE, and carrying out the null space of `V`. **NOTE**: before 1.4, sympy seems caching all big integers, which may work like memory leak when you've assembled many instructions.
* **pyelftools**: elf toolkit for handling cubin files.

### Classes

* **CuInsParser**: The class to parse the instruction string to *keys*, *values* and *modifiers*.
* **CuInsAssembler**: The class that handles the value matrix `V` and solution of `w` for a special *key*.
* **CuInsFeeder**: A simple instruction feeder reading instructions from sass dumped by `cuobjdump`.
* **CuInsAssemblerRepos**: Repository of `CuInsAssembler` for all known *keys*. Constructing a workable repos from scratch is very time consuming, and it requires quite wide range of inputs that cover all frequently used instructions. Thus a pre-gathered repos is available in `CuInsAsmRepos.txt`.
* **CuKernelAssembler** (TBD): Assembler for a kernel, which should handle all kernel wide parameters.
* **CubinFile** (TBD): It can read in a cubin file, save as an editable *cuasm* file, and then assemble the modified cuasm back to cubin.
* **CuProgram** (TBD)：Still in progress...

### An example

Here is a snippet from a cuasm file:

```
# Comments
L004b0: [R---:B------:R-:W-:-:S01]         IMAD.SHL.U32 R8, R4.reuse, 0x4, RZ ;  /* 0x0000000404087824; 0x040fe200078e00ff */
L004c0: [----:B------:R-:W-:Y:S07]         IADD3 R4, R4, 0x20, RZ ;              /* 0x0000002004047810; 0x000fce0007ffe0ff */
L004d0: [----:B------:R-:W1:-:S01]         LDS.U R8, [R8+0x10] ;                 /* 0x0000100008087984; 0x000e620000001800 */
L004e0: [----:B0-----:R-:W-:Y:S04]     @P0 FADD R5, RZ, R7 ;                     /* 0x00000007ff050221; 0x001fc80000000000 */
L004f0: [----:B-1----:R-:W-:Y:S08]         FADD R5, R8, R5 ;                     /* 0x0000000508057221; 0x002fd00000000000 */
L00500: [R---:B------:R-:W-:-:S02]         SHF.L.U32 R8, R4.reuse, 0x2, RZ ;     /* 0x0000000204087819; 0x040fe400000006ff */
```

Comment lines start with `#`. Every instruction line starts with an optional label such as `L004b0`, `Labc`, `LSet_0` (all must start with **L**), which could be referenced as address. Then comes the control codes part, it uses similar visual display as ``maxas``, but with more specific fields, which makes the correlation much easier to catch when inspecting the codes. Then follows the instruction contents, which end with ";". All contents after ";" are ignored. The instruction codes therein are inherited from original code, it becomes useless when you modified the assembly.

**NOTES**:
* Labels are optional, if exist, always start with "L", as a valid identifier.
* Every instruction should consume exactly one line.
* The reuse flag *.reuse* for operands is ignored, it will be determined by the reuse part of the control codes.

A *cuasm* file contains all information in a cubin, so there are also plenty of codes for ELF objects descriptions. Currently only code assemblies are modifiable.

## Future plan

* Extend to more instruction sets.
* More robust parsing and user friendly error reporting.
* Launch cubin in python, support more flexible configurations.
* A mechanism to probe most instructions with automated ptx bombardier.
* And many more...
