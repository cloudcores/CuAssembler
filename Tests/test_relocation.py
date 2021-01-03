# -*- coding: utf-8 -*-

from elftools.elf.elffile import ELFFile
from subprocess import check_output

import glob
import os
import re


def getRelSectionInfo(binname):
    
    m = re.compile('\.section\s+([.\w]+)\s*REL')
    lines = check_output(['cuobjdump','-elf',binname]).decode().splitlines()
    
    RelSecDict = {}
    sname = ''
    rellines = []
    doGather = False
    for line in lines:
        line = line.strip()
        if doGather:
            if len(line) == 0:
                RelSecDict[sname] = rellines
                doGather = False
            else:
                rellines.append(line)
            
            continue
        
        res = m.match(line)
        if res:
            sname = res.groups()[0]
            rellines = []
            doGather = True
    
    if doGather:
        if sname not in RelSecDict:
            RelSecDict[sname] = rellines
            
    return RelSecDict

def getELFRelInfo(binname):
    RelSecDict = {}
    
    with open(binname, 'rb') as fin:
        ef = ELFFile(fin)
        sym_section = ef.get_section_by_name('.symtab')

        for sec in ef.iter_sections():
            if not sec.name.startswith('.rel'):
                continue
            
            rels = []
            for rel in sec.iter_relocations():
                symname = sym_section.get_symbol(rel.entry['r_info_sym']).name
                
                offset = rel['r_offset']
                #secmain = ef.get_section_by_name(sec.name[4:])
                mname = re.sub('^\.rela?', '', sec.name)
                secmain = ef.get_section_by_name(mname)
                
                if secmain is None:
                    print('Unmatched section %s'%sec.name)
                    continue
                
                val = secmain.data()[offset:offset+8]
                
                rels.append((symname, rel, val))
        
            RelSecDict[sec.name] = rels

    return RelSecDict

fdir = r'G:\Repos\Tests\Programs\\'
#fdir = r'G:\Temp\CubinFull\cublas64_10\\'

for fname in glob.glob(fdir + '*.cubin'):
    print('#### %s'%fname)
    d1 = getELFRelInfo(fname)
    #print(d1)
    
    d2 = getRelSectionInfo(fname)
    #print(d2)
    #break
    
    for k in d1:
        print('Section %s :'%k)
        v1 = d1[k]
        v2 = d2[k]
        for s1, s2 in zip(d1[k], d2[k]):
            print('  %-6d %s    type=%2d, val=%s'%(s1[1]['r_offset'],
                                s1[0], s1[1]['r_info_type'], s1[2].hex()))
            print('  '+s2)
            print()