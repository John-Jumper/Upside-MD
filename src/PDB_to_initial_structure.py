#!/usr/bin/env python

import prody
import numpy as np
import sys
import cPickle
import argparse

three_letter_aa = dict(
        A='ALA', C='CYS', D='ASP', E='GLU',
        F='PHE', G='GLY', H='HIS', I='ILE',
        K='LYS', L='LEU', M='MET', N='ASN',
        P='PRO', Q='GLN', R='ARG', S='SER',
        T='THR', V='VAL', W='TRP', Y='TYR')

aa_num = dict([(k,i) for i,k in enumerate(sorted(three_letter_aa.values()))])

one_letter_aa = dict([(v,k) for k,v in three_letter_aa.items()])

parser = argparse.ArgumentParser()
parser.add_argument('pdb', help='input .pdb file')
parser.add_argument('output', help='output .pkl file')
parser.add_argument('--chain', default=None, help='Choose only a specific chain')
parser.add_argument('--output-fasta', default=None, help='Output a fasta file as well')
args = parser.parse_args()

structure = prody.parsePDB(args.pdb)
sel_str = 'name N CA C'
if args.chain is not None: sel_str += ' and chain %s' % args.chain
coords = structure.select(sel_str).getCoords()
print len(coords)/3., 'residues found.'
bond_lengths = np.sqrt(np.sum(np.diff(coords,axis=0)**2,axis=-1))

if bond_lengths.max() > 2.:
    print "WARNING: %i separate chains found" % (bond_lengths>2.).sum()

f=open(args.output,'wb')
cPickle.dump(coords[...,None], f, -1)
f.close()

if args.output_fasta is not None:
    f=open(args.output_fasta,'w')
    print >>f, '> Created from %s' % args.pdb
    sel_str = 'name CA'
    if args.chain is not None: sel_str += ' and chain %s' % args.chain
    coords = structure.select(sel_str)
    seq = ''.join([one_letter_aa[s] for s in coords.getResnames()])
    nlines = int(np.ceil(len(seq)/80.))
    for nl in range(nlines):
        print >>f,  seq[80*nl:80*(nl+1)]

