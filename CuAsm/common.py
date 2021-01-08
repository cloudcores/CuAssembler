# -*- coding: utf-8 -*-

import re
from io import StringIO

# TODO: A class for control codes?
# Pattern for control codes string
c_ControlCodesPattern = re.compile(r'(R|-){4}:B(0|-)(1|-)(2|-)(3|-)(4|-)(5|-):R[0-5\-]:W[0-5\-]:(Y|-):S\d{2}')


def alignTo(pos, align):
    ''' Padding current position to given alignment.
    
        Return: tuple(newpos, padsize)
    '''

    if align==0 or align==1:
        return pos, 0

    npos = ((pos + align -1 ) // align) * align
    return npos, npos-pos

def intList2Str(vlist, s=None):
    if s:
        fmt = '0x%%0%dx' % s
    else:
        fmt = '0x%x'
    return '['+ (', '.join([fmt%v for v in vlist])) +']'

def binstr(v, l=128, w=4, sp=' '):
    bv = bin(v)[2:]
    lb = len(bv)
    if lb<l:
        bv = '0' * (l-lb) + bv

    return sp.join([bv[i:i+w] for i in range(0, l, w)])

def hexstr(v, l=128, w=1, sp=' '*4):
    hv = '%x'%v
    lhex = l//4
    lb = len(hv)

    if lb<lhex:
        hv = '0' * (lhex-lb) + hv

    return sp.join([hv[i:i+w] for i in range(0, lhex, w)])

def decodeCtrlCodes(code):
    # c.f. : https://github.com/NervanaSystems/maxas/wiki/Control-Codes
    #      : https://arxiv.org/abs/1903.07486
    # reuse  waitbar  rbar  wbar  yield   stall
    #  0000   000000   000   000      0    0000
    #
    c_stall    = (code & 0x0000f) >> 0
    c_yield    = (code & 0x00010) >> 4
    c_writebar = (code & 0x000e0) >> 5  # write dependency barrier
    c_readbar  = (code & 0x00700) >> 8  # read  dependency barrier
    c_waitbar  = (code & 0x1f800) >> 11 # wait on dependency barrier
    c_reuse    =  code >> 17

    s_yield = '-' if c_yield !=0 else 'Y'
    s_writebar = '-' if c_writebar == 7 else '%d'%c_writebar
    s_readbar = '-' if c_readbar == 7 else '%d'%c_readbar
    s_waitbar = ''.join(['-' if (c_waitbar & (2**i)) == 0 else '%d'%i for i in range(6)])
    s_stall = '%02d' % c_stall
    s_reuse = ''.join(['R' if (c_reuse&(2**i)) else '-' for i in range(4)])

    return '%s:B%s:R%s:W%s:%s:S%s' % (s_reuse, s_waitbar, s_readbar, s_writebar, s_yield, s_stall)

def encodeCtrlCodes(s):
    if not c_ControlCodesPattern.match(s):
        raise ValueError('Invalid control code strings: %s !!!'%s)

    s_reuse, s_waitbar, s_readbar, s_writebar, s_yield, s_stall = tuple(s.split(':'))

    reuse_tr = str.maketrans('R-','10')
    waitbar_tr = str.maketrans('012345-','1111110')

    c_reuse = int(s_reuse[::-1].translate(reuse_tr), 2)
    c_waitbar = int(s_waitbar[:0:-1].translate(waitbar_tr), 2)
    c_readbar = int(s_readbar[1].replace('-', '7'))
    c_writebar = int(s_writebar[1].replace('-','7'))
    c_yield = int(s_yield!='Y')
    c_stall = int(s_stall[1:])

    code = c_reuse<<17
    code += c_waitbar<<11
    code += c_readbar<<8
    code += c_writebar<<5
    code += c_yield<<4
    code += c_stall

    return code

def splitAsmSection(lines):
    ''' Split assembly text line list into a set of sections.
        NOTE: the split is done according to ".section" directive

        Return: section_markers, a dict of section markers (a tuple of start line and end line).
                The header markers is stored in entry "$FileHeader".
        An example: section_markers = {'$FileHeader':(0,4), '.shstrtab':(4,82), '.strtab':(82,140),...}
    '''
    m_secdirective = re.compile(r'^\s*\.section\s+([\.\w]+),')

    secnames = []
    markers = [0]

    for iline, line in enumerate(lines):
        res = m_secdirective.match(line)
        if  res is not None:
            secname = res.groups()[0]
            # print("Line%4d (%s): %s"%(iline, secname, line.strip()))
            secnames.append(secname)

            # usually the previous line of .section will be a comment line
            # when splitting sections, we may want to skip this line
            has_prev_comment = False
            if iline>0:
                prev_line = lines[iline-1].strip()
                if prev_line.startswith('//') and secname in prev_line:
                    has_prev_comment = True
            
            if has_prev_comment:
                markers.append(iline-1)
            else:
                markers.append(iline)  
            markers.append(iline)

    # 
    markers.append(len(lines))

    section_markers = {}
    section_markers['$FileHeader'] = (markers[0], markers[1])  # File header parts

    for isec, secname in enumerate(secnames):
        section_markers[secname] = (markers[2*isec+2], markers[2*isec+3])
        
    return section_markers

def stringBytes2Asm(ss, label='', width=8):
    ''' Convert b'\x00' seperated string bytes into assembly bytes.
        label is the name of this string list, only for displaying the entry lists in comments.
        width is number of bytes to display per line.
    '''
    p = 0
    counter = 0
    
    sio = StringIO()
    while True:
        pnext = ss.find(b'\x00', p)
        if pnext<0:
            break
        
        s = ss[p:pnext+1]
        sio.write('    // %s[%d] = %s \n'%(label, counter, repr(s)))
        
        p0 = p
        while p0<pnext+1:
            sio.write('    /*%04x*/ .byte '%p0 + ', '.join(['0x%02x'%b for b in s[p0-p:p0-p+width]]))
            sio.write('\n')
            p0 += width
        
        sio.write('\n')
        p = pnext+1

        counter += 1

    return sio.getvalue()

def bytes2Asm(bs, width=8, addr_offset=0, ident='    '):
    ''' Convert bytes into assembly bytes.
        width is the max display length of one line.
    '''
    sio = StringIO()

    p = 0
    while p<len(bs):
        blist = ', '.join(['0x%02x'%b for b in bs[p:p+width]])
        sio.write('%s/*%04x*/ .byte '%(ident, p+addr_offset) + blist)
        sio.write('\n')
        p += width
        
    return sio.getvalue()

def bytesdump(inname, outname):
    with open(inname, 'rb') as fin:
        bs = fin.read()

    bstr = bytes2Asm(bs)

    with open(outname, 'w') as fout:
        fout.write(bstr)

def reprDict(sio, d):
    sio.write('{')
    n = len(d)
    cnt = 0
    for k, v in d.items():
        sio.write(repr(k) + ':' + repr(v))
        if cnt<n-1:
            sio.write(',\n')
        cnt += 1
    sio.write('}')

def reprList(sio, l):
    sio.write('[')
    n = len(l)
    cnt = 0
    for v in l:
        sio.write(repr(v))
        if cnt< n-1:
            sio.write(',\n')
        cnt += 1
    sio.write(']')


if __name__ == '__main__':
    cs = ['----:B--2---:R0:W1:-:S07',
            '----:B01--4-:R-:W-:-:S05',
            '----:B------:R-:W0:-:S01',
            '-R--:B------:R-:W-:-:S01',
            '----:B------:R2:W1:-:S01',
            '----:B0-----:R-:W-:Y:S04',
            'R-R-:B------:R-:W-:-:S01',
            '----:B0----5:R0:W5:Y:S05',
            '----:B------:R-:W0:-:S02',
            '----:B0-----:R-:W0:-:S02',
            '----:B0-----:R-:W-:Y:S04']
    for s in cs:
        c = encodeCtrlCodes(s)
        s2 = decodeCtrlCodes(c)

        print('0x%06x:'%c)
        print('    %s'%s)
        print('    %s'%s2)
        if s != s2:
            print('!!! Unmatched !')