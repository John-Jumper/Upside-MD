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

nonstandard_restype_conversion = dict(
        MSE='MET',  # selenomethionine
        )

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

Residue = collections.namedtuple('Residue', 'resnum chain restype phi psi omega N CA C CB CG CD chi1 chi2')

def read_residues(chain):
    ignored_restypes = dict()

    residues = []
    for res in chain.iterResidues():
        restype = res.getResname()

        if restype in nonstandard_restype_conversion:
            restype = nonstandard_restype_conversion[restype]

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
        cg_list = [v for k,v in adict.items() if re.match("[^H]G1?$",k)]
        cd_list = [v for k,v in adict.items() if re.match("[^H]D1?$",k)]
        if len(cg_list) not in (0,1):
            raise RuntimeError('Residue %i CG-list %s has too many items'%(res.getResindex(), [k for k,v in adict.items() if re.match("[^H]G1?$",k)],))
        if len(cd_list) not in (0,1):
            raise RuntimeError('Resdiue %i CD-list %s has too many items'%(res.getResindex(), [k for k,v in adict.items() if re.match("[^H]D1?$",k)],))

        # note that you need to check the residue *before* to see if a proline is cis
        r = Residue(
            res.getResnum(),
            res.getChain(),
            restype if not (restype=='PRO' and residues and np.abs(residues[-1].omega)<90.*deg) else 'CPR',
            phi,psi,omega,
            adict.get('N',  np.nan*np.ones(3)),
            adict.get('CA', np.nan*np.ones(3)),
            adict.get('C',  np.nan*np.ones(3)),
            adict.get('CB', np.nan*np.ones(3)),
            cg_list[0] if cg_list else np.nan*np.ones(3),
            cd_list[0] if cd_list else np.nan*np.ones(3),
            np.nan, np.nan)
        r = r._replace(chi1=dihedral(r.N,r.CA,r.CB,r.CG), chi2=dihedral(r.CA,r.CB,r.CG,r.CD))
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
    parser.add_argument('--record-chain-breaks', action='store_true', help='Record index of chain first residues to help automate generation of system files for multiple chains')
    parser.add_argument('--disable-recentering', action='store_true',
            help='If turned on, disable recentering of the structure.')
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
    chain_resnum = []
    chain_first_residue = []

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
                    if not i and args.record_chain_breaks: chain_first_residue.append(len(coords)/3)

            coords.extend([r.N,r.CA,r.C])
            sequence.append(r.restype)
            chi.append((r.chi1,r.chi2))
            chain_resnum.append((str(chain_id),r.resnum))
        print
    coords = np.array(coords)
    if not args.disable_recentering:
        coords -= coords.mean(axis=0) # move com to origin
    chi = np.array(chi)

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
        print >>f, 'residue restype  chain  resnum      chi1     chi2'
        for nr,restype in enumerate(sequence):
            # if np.isnan(chi[nr]): continue  # no chi1 data to write

            # # PRO is a special case since it only has two states
            # # the PRO state 1 is the state from -120 to 0
            # chi1_state = 1
            # if    0.*deg <= chi[nr] < 120.*deg: chi1_state = 0
            # if -120.*deg <= chi[nr] <   0.*deg and restype not in ('PRO','CPR'): chi1_state = 2

            print >>f, '% 7i %7s %5s   %6s  % 8.3f % 8.3f' % (
                    nr, restype, chain_resnum[nr][0], chain_resnum[nr][1], chi[nr,0]/deg, chi[nr,1]/deg)

    if chain_first_residue:
        with open(args.basename+'.chain_breaks','w') as f:
            print >>f, ' '.join([str(i) for i in chain_first_residue])

if __name__ == '__main__':
    main()
