#!/usr/bin/env python

import prody
import numpy as np
import sys
import cPickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('pdb', help='input .pdb file')
parser.add_argument('output', help='output .pkl file')
parser.add_argument('--chain', default=None, help='Choose only a specific chain')
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
