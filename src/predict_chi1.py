#!/usr/bin/env python

import sys
import os
import tempfile
import time
import pandas as pd
import numpy as np
import tables as tb
import shutil
import subprocess as sp

src = os.path.expanduser('~/upside/src')
sys.path.append(src)
import upside_engine as ue


def main():
    sidechain_file, pdb_file, chain, output_file = sys.argv[1:]
    direc = tempfile.mkdtemp()

    deg = np.pi/180.
    with tb.open_file('/home/jumper/upside-parameters/sidechain.h5') as t:
        restype_dict = dict((x,i) for i,x in enumerate(t.root.restype_order[:]))
        restype_dict['CPR'] = restype_dict['PRO']
        restype_and_chi_and_state = t.root.restype_and_chi_and_state[:]
        rotamer_start_stop_bead = t.root.rotamer_start_stop_bead
    compute_chi1_state = lambda chi1: (((chi1/deg)%360.)/120.).astype('i')
    chi1_state_ref = compute_chi1_state(restype_and_chi_and_state[:,1])

    x = restype_and_chi_and_state
    chi1_partition = dict([
            (aa,[np.array(sorted(set(x[(x[:,0]==restype_dict[aa])&(chi1_state_ref==j),-1].astype('i')))) 
                 for j in range(3)])
      for aa in sorted(restype_dict)])
    chi1_partition['CPR'] = chi1_partition['PRO']

    try:
        base_initial = os.path.join(direc, 'initial_test')
        sp.check_call([
            os.path.join(src, 'PDB_to_initial_structure.py'),
            '--chains', chain,
            '--allow-unexpected-chain-breaks',
            pdb_file, base_initial])

        config = os.path.join(direc, 'config.h5')
        sp.check_call([
            os.path.join(src,'upside_config.py'),
                '--fasta=%s.fasta'%base_initial,
                '--initial-structures=%s.initial.pkl'%base_initial,
                '--loose-hbond-criteria',  # handle poor hbond geometry in some crystal structures
                '--dynamic-rotamer-1body',
                '--rotamer-placement=%s'%sidechain_file, 
                '--rotamer-interaction=%s'%sidechain_file, 
                '--debugging-only-disable-basic-springs',
                '--no-backbone',
                '--hbond-energy=-1e-5',
                '--output=%s'%config])

        with tb.open_file(config) as t:
            pos = t.root.input.pos[:,:,0]
            seq = t.root.input.sequence[:]
            residue = t.root.input.potential.placement_fixed_point_vector_only.affine_residue[:]

        engine = ue.Upside(pos.shape[0], config)
        
        t0 = time.time()
        engine.energy(pos)
        sens = engine.get_sens('hbond_coverage')[:,0]
        t2 = time.time()

        chi1_true = pd.read_csv(base_initial+'.chi', delim_whitespace=1)
    finally:
        shutil.rmtree(direc)

    print
    print 'Time to compute %.4f seconds' % (t2-t0)

    chi1_prob_array = []
    for resnum,aa in enumerate(seq):
        if aa=='ALA' or aa=='GLY':
            chi1_prob_array.append(np.array([1.,0.,0.]))
            continue
        correct_residue = residue==resnum
        admissible_restype = restype_and_chi_and_state[:,0] == restype_dict[aa]
        probs = sens[correct_residue]
        chi1_probs = np.array([probs[s].sum() for s in chi1_partition[aa]])
        chi1_prob_array.append(chi1_probs)
    chi1_prob_array = np.array(chi1_prob_array, dtype='f8')

    assert len(chi1_true) == len(seq)
    with open(output_file,'wt') as f:
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
