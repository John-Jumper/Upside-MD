#!/usr/bin/env python
import tables as tb
import numpy as np
import argparse
import sys

def find_multichain_terms(ids, chain_starts):
    assert len(ids.shape) == 2
    chain_starts = np.array(chain_starts, dtype='i')
    assert len(chain_starts.shape) == 1
    chain_num = (ids[:][:,:,None] >= chain_starts[None,None,:]).sum(axis=-1)
    multichain = np.array([len(set(x)) > 1 for x in chain_num])
    return multichain


def cut_out_rows(hdf_array, remove_row):
    assert remove_row.shape ==(hdf_array.shape[0],)
    assert remove_row.dtype == np.bool
    keep_row = -remove_row
    n_keep   = keep_row.sum()

    hdf_array[:n_keep] = hdf_array[:][keep_row]
    hdf_array.truncate(n_keep)


def multicut(chain_starts, pot, grp_name, id_name, other_names, consider_row = None):
    if grp_name not in pot: return
    grp = pot._v_children[grp_name]

    multichain = find_multichain_terms(grp._v_children[id_name], chain_starts)
    if consider_row is not None: multichain = multichain & consider_row
    print 'Cutting %s (%i rows)' % (grp, multichain.sum())
    for nm in [id_name] + list(other_names):
        print '    %s' % nm
        cut_out_rows(grp._v_children[nm], multichain)
    print

def unsupported(pot, grp_name):
    if grp_name in pot: 
        print '!!!!!!!!   Error: %s is not supported   !!!!!!!!' % grp_name
        print


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config to modify')
    parser.add_argument('--chain-first-residue', default=[0], type=int, action='append', help=
            'First residue index of a new chain.  May be specified multiple times.  --chain-first-residue=0 is assumed '+
            'and need not be specified.')
    parser.add_argument('--jump-length-scale', type=float, default=5., help='Translational gaussian width in angstroms for Monte Carlo JumpSampler. Default: 5 angstroms')
    parser.add_argument('--jump-rotation-scale', type=float, default=30., help='Rotational gaussian width in degrees for Monte Carlo JumpSampler. Default: 30 degrees')
    parser.add_argument('--remove-pivot', action='store_true', help='Whether to remove the MC PivotSampler param group to isolate JumpSampler for testing')
    args = parser.parse_args()

    print 'This program is an ugly hack, and your simulation may give very bad results.'
    print 'If you are lucky, the results will be only a little bad.'
    print
    print 'Breaking chain at residues %s' % args.chain_first_residue
    print

    t = tb.open_file(args.config, 'a')
    pot = t.root.input.potential
    chain_starts = np.array(args.chain_first_residue)*3
    n_chains = len(chain_starts)

    # Setting Jump Sampler params
    if args.remove_pivot:
        t.remove_node("/input", "pivot_moves", recursive=True)

    # Need to add one atom past the last atom so that the last chain is processed
    chain_starts_plus = np.append(chain_starts, len(t.root.input.sequence)*3)

    jump_atom_range = np.array([[chain_starts_plus[i], chain_starts_plus[i+1]] for i in xrange(n_chains)], dtype='int32')
    jump_sigma_trans = np.array([args.jump_length_scale]*n_chains, dtype='float32')
    jump_sigma_rot = np.array([args.jump_rotation_scale*np.pi/180.]*n_chains, dtype='float32') # Converts to radians

    print "jump atom_range:\n{}\nsigma_trans:\n{}\nsigma_rot:\n{}\n".format(jump_atom_range, jump_sigma_trans, jump_sigma_rot)

    jump_grp = t.create_group("/input","jump_moves","JumpSampler Params")
    t.create_array(jump_grp, "atom_range", jump_atom_range, "First, last atom num demarking each chain")
    t.create_array(jump_grp, "sigma_trans", jump_sigma_trans, "Translational gaussian width")
    t.create_array(jump_grp, "sigma_rot", jump_sigma_rot, "Rotational gaussian width")

    # FIXME don't attempt pivots across chain breaks

    # Breaking interactions
    multicut(chain_starts, pot, 'angle_spring', 'id', 'equil_dist spring_const'.split())
    multicut(chain_starts, pot, 'dihedral_spring', 'id', 'equil_dist spring_const'.split())
    multicut(chain_starts, pot, 'dist_spring', 'id', 'equil_dist spring_const bonded_atoms'.split(), pot.dist_spring.bonded_atoms[:].astype('bool'))

    if 'infer_H_O' in pot: 
        print 'Checking %s' % pot.infer_H_O
        print
        is_bad =(any(find_multichain_terms(pot.infer_H_O.donors   .id,chain_starts)) or
                 any(find_multichain_terms(pot.infer_H_O.acceptors.id,chain_starts)))

        if is_bad:
            print '!!! Error: You must use --hbond-excluded-residues in upside_config to produce chain breaks for the hbonds !!!'
            print

    unsupported(pot, 'basin_correlation_pot')

    if 'rama_coord' in pot:
        tbl = pot.rama_coord.id
        multichain_locs = find_multichain_terms(tbl[:], chain_starts).nonzero()[0]
        print 'Editing %s (%i rows)' %(pot.rama_coord, len(multichain_locs))
        for loc in multichain_locs:
            ids = tbl[loc]
            assert ids.shape == (5,)
            chain_num = (ids[:,None]>=chain_starts).sum(axis=-1)
            if not (chain_num[1] == chain_num[2] == chain_num[3] and (chain_num[0]==chain_num[1] or chain_num[3]==chain_num[4])): 
                    raise ValueError("Weird rama_coord %i, unable to proceed" % loc)

            if chain_num[0]==chain_num[1]: 
                tbl[loc,4] = -1  # cut psi
            else: 
                tbl[loc,0] = -1  # cut phi
    t.close()

if __name__ == '__main__':
    main()
