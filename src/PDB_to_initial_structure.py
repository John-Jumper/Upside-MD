#!/usr/bin/env python

import prody
import numpy as np
import sys
import cPickle
import argparse
import tables
import collections
import re

np.set_printoptions(precision=2,suppress=True)
deg = np.pi/180.

three_letter_aa = dict(
        A='ALA', C='CYS', D='ASP', E='GLU',
        F='PHE', G='GLY', H='HIS', I='ILE',
        K='LYS', L='LEU', M='MET', N='ASN',
        P='PRO', Q='GLN', R='ARG', S='SER',
        T='THR', V='VAL', W='TRP', Y='TYR')
three_letter_aa["*P"] = "CPR"

aa_num = dict([(k,i) for i,k in enumerate(sorted(three_letter_aa.values()))])

one_letter_aa = dict([(v,k) for k,v in three_letter_aa.items()])

def vmag(x):
    assert x.shape[-1] == 3
    return np.sqrt(np.sum(x**2,axis=-1))

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

Residue = collections.namedtuple('Residue', 'resnum restype phi psi omega N CA C CB CG chi1')

def read_residues(chain):
    ignored_restypes = dict()

    residues = []
    for res in chain.iterResidues():
        restype = res.getResname()
        if restype not in one_letter_aa: 
            ignored_restypes[restype] = ignored_restypes.get(restype,0) + 1
            continue  # let's hope it is HOH or something

        try:
            phi = prody.calcPhi(res)*deg
        except ValueError:
            phi = np.nan

        try:
            psi = prody.calcPsi(res)*deg
        except ValueError:
            psi = np.nan

        try:
            omega = prody.calcOmega(res)*deg
        except ValueError:
            omega = np.nan

        adict   = dict((a.getName(),a.getCoords()) for a in res.iterAtoms())
        cg_list = [v for k,v in adict.items() if re.match(".G1?$",k)]
        assert len(cg_list) in (0,1)

        r = Residue(
            res.getResnum(),
            restype if not (restype=='PRO' and np.abs(omega)<90.*deg) else 'CPR',
            phi,psi,omega,
            adict.get('N',  np.nan*np.ones(3)),
            adict.get('CA', np.nan*np.ones(3)),
            adict.get('C',  np.nan*np.ones(3)),
            adict.get('CB', np.nan*np.ones(3)),
            cg_list[0] if cg_list else np.nan*np.ones(3),
            np.nan)
        r = r._replace(chi1=dihedral(r.N,r.CA,r.CB,r.CG))
        residues.append(r)

    return residues, ignored_restypes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb', help='input .pdb file')
    parser.add_argument('basename', help='output basename')
    parser.add_argument('--model', default=None, help='Choose only a specific model in the .pdb')
    parser.add_argument('--chains', default='', 
            help='Comma-separated list of chains to parse (e.g. --chains=A,C,E). Default is all chains.')
    parser.add_argument('--allow-unexpected-chain-breaks', default=False, action='store_true', 
            help='Do not fail on unexpected chain breaks (dangerous)')
    args = parser.parse_args()
    
    structure = prody.parsePDB(args.pdb, model=args.model)
    print

    chain_ids = set(x for x in args.chains.split(',') if x)
    chains = [(ch.getChid(),read_residues(ch)) for ch in structure.iterChains() 
        if not chain_ids or ch.getChid() in chain_ids]

    missing_chains = chain_ids.difference(x[0] for x in chains)
    if missing_chains:
        print >>sys.stderr, "ERROR: missing chain %s" % ",".join(sorted(missing_chains))
        sys.exit(1)

    coords = []
    sequence = []
    chi = []
    unexpected_chain_breaks = False

    for chain_id, (res,ignored_restype) in chains:
        print "chain %s:" % chain_id

        # only look at residues with complete backbones
        res = [r for r in res if np.all(np.isfinite(np.array((r.N,r.CA,r.C))))]

        print '    found %i usable residues' % len(res)
        print

        for k,v in sorted(ignored_restype.items()):
            print '    ignored restype %-3s (%i residues)'%(k,v)
        print

        for i,r in enumerate(res):
            if coords: # residues left by a previous chain
                dist = vmag(r.N-coords[-1])
                if dist > 2.:
                    print '    %s chain break at residue %i (%4.1f A)' % (('UNEXPECTED' if i else 'expected  '),
                         len(coords)/3, dist)
                    if i: unexpected_chain_breaks = True

            coords.extend([r.N,r.CA,r.C])
            sequence.append(r.restype)
            chi.append(r.chi1)
        print
    coords = np.array(coords)

    if unexpected_chain_breaks and not args.allow_unexpected_chain_breaks:
        print >>sys.stderr, ('ERROR: see above for unexpected chain breaks, probably missing residues in crystal '+
                'structure (--allow-unexpected-chain-breaks to suppress this error at your own risk)')
        sys.exit(1)

    fasta_seq = ''.join(one_letter_aa[s] for s in sequence)

    with open(args.basename + '.initial.pkl','wb') as f:
        cPickle.dump(coords[...,None], f, -1)
        f.close()

    with open(args.basename+'.fasta','w') as f:
        print >>f, '> Created from %s' % args.pdb
        nlines = int(np.ceil(len(fasta_seq)/80.))
        for nl in range(nlines):
            print >>f,  fasta_seq[80*nl:80*(nl+1)]
    
    with open(args.basename+'.chi','w') as f:
        print >>f, 'residue restype rotamer chi1'
        for nr,restype in enumerate(sequence):
            if np.isnan(chi[nr]): continue  # no chi1 data to write

            # PRO is a special case since it only has two states
            # the PRO state 1 is the state from -120 to 0
            chi1_state = 1
            if    0.*deg <= chi[nr] < 120.*deg: chi1_state = 0
            if -120.*deg <= chi[nr] <   0.*deg and restype not in ('PRO','CPR'): chi1_state = 2

            print >>f, '% 3i %3s %i % 8.3f' % (nr, restype, chi1_state, chi[nr]/deg)


if __name__ == '__main__':
    main()
