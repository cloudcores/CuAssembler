# -*- coding: utf-8 -*-

from CuAsm.CuNVInfo import CuNVInfo 
import networkx as nx

from elftools.elf.elffile import ELFFile

import glob
import matplotlib.pyplot as plt

ROOT_NODE = 'Root'

def updateGraphByCubin(g, cubinname):
    
    with open(cubinname, 'rb') as fin:
        ef = ELFFile(fin)

        for sec in ef.iter_sections():
            if sec.name.startswith('.nv.info.'):
                nvinfo = CuNVInfo(sec.data())
                prev = ROOT_NODE

                for i, (attr, length, val) in enumerate(nvinfo.m_AttrList):
                    if attr.startswith('EIATTR_UNKNOWN_'):
                        print(cubinname)
                    if not g.has_edge(prev, attr):
                        g.add_edge(prev, attr)
                        print('Add edge %s to %s'%(prev, attr))

                    prev = attr

def build(g, fpattern):
    for fname in glob.glob(fpattern):
        updateGraphByCubin(g, fname)

def getGraph():
    G = nx.DiGraph()
    # G.add_node(ROOT_NODE)
    # build(G, r'G:\Temp\CubinSample\*\*.cubin')
    # build(G, r'G:\Temp\CubinFull\*\*.cubin')
    
    return G

if __name__ == '__main__':
    G = getGraph()
    
    # G.layout('dot')
    # G.draw('a.png')
    # nx.drawing.nx_agraph.write_dot(G, 'a.dot')

    # pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    # nx.draw(G, pos, with_labels=False, arrows=True)
    # plt.savefig('nx_test.png')
    plt.figure(1)
    nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G), with_labels=False)
    plt.show()
    