#!/usr/bin/env python

import prody
import numpy as np
import sys
import cPickle
import argparse
import tables

model_geom = np.zeros((3,3))
model_geom[:,0] = (-1.19280531, -0.83127186, 0.)  # N
model_geom[:,1] = ( 0.,          0.,         0.)  # CA
model_geom[:,2] = ( 1.25222632, -0.87268266, 0.)  # C
model_geom -= model_geom.mean(axis=0)

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


def dihedral(x1,x2,x3,x4):
    '''four atom dihedral angle in radians'''
    b1 = x2-x1
    b2 = x3-x2
    b3 = x4-x3
    b2b3 = np.cross(b2,b3)
    b2mag = np.sqrt(np.sum(b2**2, axis=-1))
    return np.arctan2(
            b2mag * (b1*b2b3).sum(axis=-1),
            (np.cross(b1,b2) * b2b3).sum(axis=-1))

def initial_rama(pos):
    n_res = len(pos)/3
    assert pos.shape == (3*n_res,3)

    pos = pos.reshape((-1,3,3))

    N  = 0
    CA = 1
    C  = 2

    rama = np.zeros((n_res,2)) - 1.3963
    rama[1:,0] = dihedral(
            pos[0:-1, C],
            pos[1:,   N],
            pos[1:,   CA],
            pos[1:,   C])

    rama[:-1,1] = dihedral(
            pos[0:-1,  N],
            pos[0:-1,  CA],
            pos[0:-1,  C],
            pos[1:,    N])

    return rama


def parse_dynamic_backbone_point(fname):
    import scipy.interpolate as interp

    lib = tables.open_file(fname)
    ret = dict()
    for resname,g in lib.root._v_children.items():
        # construct splines for each dimension
        splines = [interp.RectBivariateSpline(
            g.bin_values[:],
            g.bin_values[:],
            g.center[:,:,d]) for d in range(3)]

        ret[resname] = lambda phi,psi,splines=splines: np.concatenate(
                [s(phi,psi,grid=False)[...,None] for s in splines],axis=-1)
    lib.close()
    return ret


def parse_static_backbone_point(fname):
    import scipy.interpolate as interp

    sc_lib = tables.open_file(fname)
    sc_dict = dict(zip(
        map(str, sc_lib.root.names[:]),
        sc_lib.root.com_pos[:]))
    sc_lib.close()

    return dict([(resname, lambda phi,psi,pt=pt: pt) for resname,pt in sc_dict.items()])


three_letter_aa = dict(
        A='ALA', C='CYS', D='ASP', E='GLU',
        F='PHE', G='GLY', H='HIS', I='ILE',
        K='LYS', L='LEU', M='MET', N='ASN',
        P='PRO', Q='GLN', R='ARG', S='SER',
        T='THR', V='VAL', W='TRP', Y='TYR')

aa_num = dict([(k,i) for i,k in enumerate(sorted(three_letter_aa.values()))])

one_letter_aa = dict([(v,k) for k,v in three_letter_aa.items()])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb', help='input .pdb file')
    parser.add_argument('output', help='output .pkl file')
    parser.add_argument('--chain', default=None, help='Choose only a specific chain')
    parser.add_argument('--model', default=None, help='Choose only a specific model in the .pdb')
    parser.add_argument('--output-fasta', default=None, help='Output a fasta file as well')
    parser.add_argument('--output-rama', default='', help='Output rama angles')
    parser.add_argument('--output-sidechain-com', default=None, 
            help='Output a file containing the sidechain center-of-mass positions ' +
            '(based on alignment of backbone atoms) as well.  Requires --sidechain-com-reference')
    parser.add_argument('--static-sidechain-com-reference', default=None,
            help='Reference file for sidechain center-of-mass positions')
    parser.add_argument('--dynamic-sidechain-com-reference', default=None,
            help='Reference file for backbone-dependent sidechain center-of-mass positions')
    args = parser.parse_args()
    
    if args.output_sidechain_com and not (args.static_sidechain_com_reference or args.dynamic_sidechain_com_reference):
        parser.error('--output-sidechain-com requires --static-sidechain-com-reference or --dynamic-sidechain-com-reference')
    if args.static_sidechain_com_reference and args.dynamic_sidechain_com_reference:
        parser.error('--static-sidechain-com-reference is incompatible with --dynamic-sidechain-com-reference')

    
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

    n_res = len(seq)
    rama = initial_rama(coords)
    if args.output_rama:
        deg = np.pi/180.
        f = open(args.output_rama,'w')
        for nr in range(n_res):
            print >>f, '% 3i %3s % 8.3f % 8.3f' % (nr, three_letter_aa[seq[nr]], rama[nr,0]/deg, rama[nr,1]/deg)
        f.close()

    np.set_printoptions(precision=2,suppress=True)
    if args.output_sidechain_com:
        assert len(coords) == 3*n_res

        if args.dynamic_sidechain_com_reference:
            sc_dict = parse_dynamic_backbone_point(args.dynamic_sidechain_com_reference) 
        else:
            sc_dict = parse_static_backbone_point (args.static_sidechain_com_reference) 
    
        com = np.zeros((n_res,3))
        for nr in range(n_res):
            z = coords[3*nr+0:3*nr+3]
            rot,trans = rmsd_transform(z, model_geom)
            com[nr] = np.dot(sc_dict[three_letter_aa[seq[nr]]](rama[nr,0],rama[nr,1]), rot.T) + trans
    
        f = open(args.output_sidechain_com,'w')
        for nr in range(n_res):
            print >>f, '% 3i %3s % 8.3f % 8.3f % 8.3f' % (nr, three_letter_aa[seq[nr]], com[nr,0], com[nr,1], com[nr,2])
        f.close()

if __name__ == '__main__':
    main()
