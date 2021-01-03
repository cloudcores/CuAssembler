# -*- coding: utf-8 -*-

from elftools.elf.elffile import ELFFile
from elftools.elf.structs import ELFStructs

def buildStringDict(bytelist):
    p = 0
    counter = 0
    
    sdict = {}
    while True:
        counter += 1
        pnext = bytelist.find(b'\x00', p)
        if pnext<0:
            break
        
        s = bytelist[p:pnext+1]
        if s not in sdict:
            sdict[s] = (p,pnext+1)
        p = pnext+1

    return sdict

def bytes2text(bs, label=''):
    p = 0
    counter = 0
    
    while True:
        counter += 1
        pnext = bs.find(b'\x00', p)
        if pnext<0:
            break
        
        s = bs[p:pnext+1]
        print('    // %4d: %s[%d] = %s '%(p, label, counter, repr(s)))
        
        p0 = p
        while p0<pnext+1:
            print('    /*%04x*/ .byte '%p0 + ', '.join(['0x%02x'%b for b in s[p0-p:p0-p+8]]))
            p0 += 8
        
        print()
        p = pnext+1

elf_s = ELFStructs(little_endian=True, elfclass=64)
elf_s.create_basic_structs()
elf_s.create_advanced_structs()

#binname = 'D:\MyProjects\Programs\cudatest.7.sm_75.cubin'
#binname = 'D:\MyProjects\Programs\cudatest.1.sm_52.cubin'
#binname = r'G:\Repos\Tests\Programs\cudatest.5.sm_52.cubin'
#binname = r'G:\Repos\Tests\Programs\cudatest.5.sm_52.cubin'
binname = r'G:\Repos\Tests\Programs\cudatest.7.sm_75.cubin'
#fname = r'D:\MyProjects\Programs\cudatest.2.sm_75.asm'
#fname = r'D:\MyProjects\CuAssembler\TestData\Mandelbrot.sm_75.asm'
#fname = r'G:\Repos\Tests\Programs\cudatest.7.sm_75.cuasm'
    
fin =  open(binname, 'rb')
ef = ELFFile(fin)

#.strtab
#.shstrtab
#.symtab

s1 = ef.get_section_by_name('.strtab')
s2 = ef.get_section_by_name('.shstrtab')
s3 = ef.get_section_by_name('.symtab')
s4 = ef.get_section_by_name('.nv.info')

#sstring = s1.data().decode()
#slist = sstring.split('\x00')
#
#print('\n.strtab')
#for i,s in enumerate(slist):
#    print('%4d : %s'%(i,s))
#
#shstring = s2.data().decode()
#shtabset = set([])
#sh_slist = shstring.split('\x00')
#print('\n.shstrtab')
#for i,s in enumerate(sh_slist):
#    print('%4x : %s'%(i,s))
#    shtabset.add(s)
#
##print('\nSymbols:')
#symtab = []
#for s in s3.iter_symbols():
#    print('\n'+s.name)
#    print(s.entry)
#    symtab.append(s.name)

#secset = set([])
#print('\nSections:')
#for i,s in enumerate(ef.iter_sections()):
#    secset.add(s.name)
#
#
#print('\nSections:')
#for i,s in enumerate(ef.iter_sections()):
#    
#    p = s.header.sh_name
#    pend = shstring.find('\x00', p)
#    pstr = shstring[p:pend]
#    
#    print('%2d: '%i+s.name)
#    print('    '+pstr)
#    
#
#for shs in sh_slist:
#    if shs not in secset:
#        print(shs)


#for i, symbol in enumerate(s3.iter_symbols()):
#    print('%4d: '%i+symbol.name)
#    
#    p = symbol.entry.st_name
#    pend = sstring.find('\x00', p)
#    pstr = sstring[p:pend]
#    print('      '+pstr)
#
#with open(r'D:\MyProjects\Programs\relocation_table.txt','r') as fin:
#    
#    symtype = [s.strip() for s in fin.readlines()]
#    
#    
#for s in ef.iter_sections():
#    if s.name.startswith('.rel'):
#        for rel in s.iter_relocations():
#            
#            sym = rel.entry.r_info_sym
#            t   = rel.entry.r_info_type
#            print('Relocation of symbol %16s (%2d) with type %-28s (%2d)'% (
#                    symtab[sym], sym, symtype[t], t))
print(ef.header)
print()

for sec in ef.iter_sections():
    s0 = sec.header['sh_offset']
    s1 = s0 + sec.header['sh_size']
    print('%-32s  %8d   %8d   %8d   %8d'%(sec.name, s0, sec.header['sh_size'],
                                                s1, sec.header['sh_addralign']))

print()
for seg in ef.iter_segments():
    print(seg.header)



    