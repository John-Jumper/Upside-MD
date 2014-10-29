#!/usr/bin/env python

import prody
import numpy as np
import sys
import cPickle

pdb, chain, output_file = sys.argv[1:]
structure = prody.parsePDB(pdb)
coords = structure.select('name N CA C and chain %s' % chain).getCoords()
print len(coords)/3., 'residues found.'
bond_lengths = np.sqrt(np.sum(np.diff(coords,axis=0)**2,axis=-1))
assert bond_lengths.max() < 2.   # check for chain continuity

f=open(output_file,'wb')
cPickle.dump(coords[...,None], f, -1)
f.close()
