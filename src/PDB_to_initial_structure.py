#!/usr/bin/env python

import prody
import numpy as np
import sys
import cPickle
import argparse

def rmsd_transform(target, model):
    assert target.shape == model.shape == (model.shape[0],3)
    base_shift_target = target.mean(axis=0)
    base_shift_model  = model .mean(axis=0)
    
    target = target - target.mean(axis=0)
    model = model   - model .mean(axis=0)

    R = np.dot(target.T, model)
    U,S,Vt = np.linalg.svd(R)
    if np.linalg.det(np.dot(U,Vt))<0.:
        Vt[:,-1] *= -1.  # fix improper rotation
    rot = np.dot(U,Vt)
    shift = base_shift_target - np.dot(rot, base_shift_model)
    return rot, shift


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
parser.add_argument('--model', default=None, help='Choose only a specific model in the .pdb')
parser.add_argument('--output-fasta', default=None, help='Output a fasta file as well')
parser.add_argument('--output-sidechain-com', default=None, 
        help='Output a file containing the sidechain center-of-mass positions ' +
        '(based on alignment of backbone atoms) as well.  Requires --sidechain-com-reference')
parser.add_argument('--sidechain-com-reference', default=None,
        help='Reference file for model geometry and sidechain center-of-mass positions')
args = parser.parse_args()

if args.output_sidechain_com and not args.sidechain_com_reference:
    parser.error('--output-sidechain-com requires --sidechain-com-reference')

structure = prody.parsePDB(args.pdb, model=args.model)
sel_str = 'name N CA C'
if args.chain is not None: sel_str += ' and chain %s' % args.chain
coords = structure.select(sel_str).getCoords()
print len(coords)/3., 'residues found.'
bond_lengths = np.sqrt(np.sum(np.diff(coords,axis=0)**2,axis=-1))

if bond_lengths.max() > 2.:
    print "WARNING: %i separate chains found" % (1+(bond_lengths>2.).sum())

f=open(args.output,'wb')
cPickle.dump(coords[...,None], f, -1)
f.close()

sel_str = 'name CA'
if args.chain is not None: sel_str += ' and chain %s' % args.chain
coords_CA = structure.select(sel_str)
seq = ''.join([one_letter_aa[s] for s in coords_CA.getResnames()])
    
if args.output_fasta is not None:
    print coords.shape
    f=open(args.output_fasta,'w')
    print >>f, '> Created from %s' % args.pdb
    nlines = int(np.ceil(len(seq)/80.))
    for nl in range(nlines):
        print >>f,  seq[80*nl:80*(nl+1)]

def vmag(x):
    assert x.shape[-1] == 3
    return np.sqrt(x[...,0]**2+x[...,1]**2+x[...,2]**2)

np.set_printoptions(precision=2,suppress=True)
if args.output_sidechain_com:
    n_res = len(seq)
    assert len(coords) == 3*n_res
    import tables

    sc_lib = tables.open_file(args.sidechain_com_reference)
    sc_dict = dict(zip(
        map(str, sc_lib.root.names[:]),
        sc_lib.root.com_pos[:]))
    model_geom = sc_lib.root.backbone_geom[:3]
    sc_lib.close()

    com = np.zeros((n_res,3))
    for nr in range(n_res):
        z = coords[3*nr+0:3*nr+3]
        rot,trans = rmsd_transform(z, model_geom)
        com[nr] = np.dot(sc_dict[three_letter_aa[seq[nr]]], rot.T) + trans

    f = open(args.output_sidechain_com,'w')
    for nr in range(n_res):
        print >>f, '% 3i %3s % 8.3f % 8.3f % 8.3f' % (nr, three_letter_aa[seq[nr]], com[nr,0], com[nr,1], com[nr,2])
    f.close()
