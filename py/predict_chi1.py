#!/usr/bin/env python

import sys
import os
import tempfile
import time
import numpy as np
import tables as tb
import shutil
import subprocess as sp

deg = np.pi/180.

compute_chi1_state = lambda chi1: (((chi1/deg)%360.)/120.).astype('i')


class Chi1Predict(object):
    def __init__(self, sidechain_file):
        with tb.open_file(sidechain_file) as t:
            self.restype_dict = dict((x,i) for i,x in enumerate(t.root.restype_order[:]))
            self.n_restype = len(self.restype_dict)
            self.restype_dict['CPR'] = self.restype_dict['PRO']
            self.restype_and_chi_and_state = t.root.restype_and_chi_and_state[:]
            rotamer_start_stop_bead = t.root.rotamer_start_stop_bead

        x = self.restype_and_chi_and_state
        chi1_state_ref = compute_chi1_state(x[:,1])
        self.chi1_partition = dict([
                (aa,[np.array(sorted(set(x[(x[:,0]==self.restype_dict[aa])&(chi1_state_ref==j),-1].astype('i')))) 
                     for j in range(3)])
          for aa in sorted(self.restype_dict)])
        self.chi1_partition['CPR'] = self.chi1_partition['PRO']


    def predict_chi1(self, seq, residue, rotamer_posterior_prob):
        assert len(residue) == len(rotamer_posterior_prob)

        chi1_prob_array = []
        for resnum,aa in enumerate(seq):
            if aa=='ALA' or aa=='GLY':
                chi1_prob_array.append(np.array([1.,0.,0.]))
            else:
                correct_residue = residue==resnum
                admissible_restype = self.restype_and_chi_and_state[:,0] == self.restype_dict[aa]
                probs = rotamer_posterior_prob[correct_residue]  # find all probs for correct residue
                chi1_probs = np.array([probs[s].sum() for s in self.chi1_partition[aa]])
                chi1_prob_array.append(chi1_probs)
        return np.array(chi1_prob_array, dtype='f4')

    def compute_zero_one_stats(self, seq, chi1_prob, chi1_states):
        results = np.zeros((self.n_restype,2), dtype='i8')  # correct,total chi1
        assert len(seq) == len(chi1_prob) == len(chi1_states)
        for aa,p,state in zip(seq,chi1_prob,chi1_states):
           results[self.restype_dict[aa],0] += np.argmax(p) == state
           results[self.restype_dict[aa],1] += 1
        return results


def main():
    import pandas as pd
    src = os.path.expanduser('~/upside/src')
    sys.path.append(src)
    import upside_engine as ue
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sidechain-param', help='Parameter file for side chains', required=True)
    parser.add_argument('--chain', default=None, help='Only load specific chain')
    parser.add_argument('pdb_input',  help='Input pdb file')
    parser.add_argument('chi_output', help='Output chi file')
    args = parser.parse_args()

    direc = tempfile.mkdtemp()

    predictor = Chi1Predict(args.sidechain_param)

    try:
        base_initial = os.path.join(direc, 'initial_test')
        sp.check_call([os.path.join(src, 'PDB_to_initial_structure.py')] +
                (['--chain=%s'%args.chain] if args.chain is not None else []) +
                ['--allow-unexpected-chain-breaks', args.pdb_input, base_initial])

        config = os.path.join(direc, 'config.h5')
        sp.check_call([
            os.path.join(src,'upside_config.py'),
                '--fasta=%s.fasta'%base_initial,
                '--initial-structures=%s.initial.pkl'%base_initial,
                '--loose-hbond-criteria',  # handle poor hbond geometry in some crystal structures
                '--dynamic-rotamer-1body',
                '--rotamer-placement=%s'  %args.sidechain_param,
                '--rotamer-interaction=%s'%args.sidechain_param,
                '--debugging-only-disable-basic-springs',
                '--no-backbone',
                '--hbond-energy=-1e-5',
                '--output=%s'%config])

        with tb.open_file(config) as t:
            pos = t.root.input.pos[:,:,0]
            seq = t.root.input.sequence[:]
            residue = t.root.input.potential.placement_fixed_point_vector_only.affine_residue[:]

        engine = ue.Upside(config)
        
        t0 = time.time()
        engine.energy(pos)
        sens = engine.get_sens('hbond_coverage')[:,0]
        t2 = time.time()

        chi1_true = pd.read_csv(base_initial+'.chi', delim_whitespace=1)
    finally:
        shutil.rmtree(direc)

    print
    print 'Time to compute %.5f seconds for %i residues' % (t2-t0, len(pos)/3)

    chi1_prob_array = predictor.predict_chi1(seq, residue, sens)

    assert len(chi1_true) == len(seq)
    with open(args.chi_output,'wt') as f:
        print >>f, 'residue restype chain resnum chi1_prob0 chi1_prob1 chi1_prob2 chi1_from_input_file'
        for resnum in range(len(seq)):
            print >>f, '%i %s %s %s %.4f %.4f %.4f %.1f' % (
                    resnum,
                    (seq[resnum] if seq[resnum]!='CPR' else 'PRO'),
                    chi1_true.chain[resnum],
                    chi1_true.resnum[resnum],
                    chi1_prob_array[resnum,0],
                    chi1_prob_array[resnum,1],
                    chi1_prob_array[resnum,2],
                    chi1_true.chi1[resnum],
                    )

if __name__ == '__main__':
    main()
