#!/usr/bin/env python

import numpy as np
import tables as tb
import sys,os
import cPickle


three_letter_aa = dict(
        A='ALA', C='CYS', D='ASP', E='GLU',
        F='PHE', G='GLY', H='HIS', I='ILE',
        K='LYS', L='LEU', M='MET', N='ASN',
        P='PRO', Q='GLN', R='ARG', S='SER',
        T='THR', V='VAL', W='TRP', Y='TYR')

aa_num = dict([(k,i) for i,k in enumerate(sorted(three_letter_aa.values()))])

one_letter_aa = dict([(v,k) for k,v in three_letter_aa.items()])

deg=np.deg2rad(1)

default_filter = tb.Filters(complib='zlib', complevel=5, fletcher32=True)
n_bit_rotamer = 4

def vmag(x):
    assert x.shape[-1]
    return np.sqrt(x[:,0]**2+x[:,1]**2+x[:,2]**2)

def create_array(grp, nm, obj=None):
    return t.create_earray(grp, nm, obj=obj, filters=default_filter)


def write_cavity_radial(cavity_radius):
    g = t.create_group(t.root.input.potential, 'cavity_radial')
    g._v_attrs.arguments = np.array(['pos'])

    create_array(g, 'id',              np.arange(n_atom))
    create_array(g, 'radius',          np.ones(n_atom)*cavity_radius)
    create_array(g, 'spring_constant', np.ones(n_atom)*5.)


def write_z_flat_bottom(parser, fasta, z_spring_table):
    fields = [ln.split() for ln in open(z_spring_table,'U')]
    header = 'residue z0 radius spring_constant'
    actual_header = [x.lower() for x in fields[0]]
    if actual_header != header.split():
        parser.error('First line of z-flat-bottom table must be "%s" but is "%s"'
                %(header," ".join(actual_header)))
    if not all(len(f)==len(fields[0]) for f in fields):
        parser.error('Invalid format for z-flat-bottom file')
    fields = fields[1:]
    n_spring = len(fields)

    g = t.create_group(t.root.input.potential, 'z_flat_bottom')
    g._v_attrs.arguments = np.array(['pos'])

    atom            = np.zeros((n_spring), dtype='i')
    z0              = np.zeros((n_spring,))
    radius          = np.zeros((n_spring,))
    spring_constant = np.zeros((n_spring,))

    for i,f in enumerate(fields):
        res = int(f[0])
        msg = 'Z_flat energy specified for residue %i (zero is first residue) but there are only %i residues in the FASTA'
        if not (0 <= res < len(fasta)): raise ValueError(msg % (res, len(fasta)))
        atom[i] = int(f[0])*3 + 1  # restrain the CA atom in each residue

        z0[i]              = float(f[1])
        radius[i]          = float(f[2])
        spring_constant[i] = float(f[3])

    create_array(g, 'atom',            obj=atom)
    create_array(g, 'z0',              obj=z0)
    create_array(g, 'radius',          obj=radius)
    create_array(g, 'spring_constant', obj=spring_constant)


def write_tension(parser, fasta, tension_table):
    fields = [ln.split() for ln in open(tension_table,'U')]
    header = 'residue tension_x tension_y tension_z'
    actual_header = [x.lower() for x in fields[0]]
    if actual_header != header.split():
        parser.error('First line of tension table must be "%s" but is "%s"'
                %(header," ".join(actual_header)))
    if not all(len(f)==len(fields[0]) for f in fields):
        parser.error('Invalid format for tension file')
    fields = fields[1:]
    n_spring = len(fields)

    g = t.create_group(t.root.input.potential, 'tension')
    g._v_attrs.arguments = np.array(['pos'])

    atom    = np.zeros((n_spring,), dtype='i')
    tension = np.zeros((n_spring,3))

    for i,f in enumerate(fields):
        res = int(f[0])
        msg = 'tension energy specified for residue %i (zero is first residue) but there are only %i residues in the FASTA'
        if not (0 <= res < len(fasta)): raise ValueError(msg % (res, len(fasta)))
        atom[i] = int(f[0])*3 + 1  # restrain the CA atom in each residue
        tension[i] = [float(x) for x in (f[1],f[2],f[3])]

    create_array(g, 'atom',    obj=atom)
    create_array(g, 'tension_coeff', obj=tension)


def write_backbone_pair(fasta):
    n_res = len(fasta)
    grp = t.create_group(potential, 'backbone_pairs')
    grp._v_attrs.arguments = np.array(['affine_alignment'])

    ref_pos = np.zeros((n_res,4,3))
    ref_pos[:,0] = (-1.19280531, -0.83127186,  0.)        # N
    ref_pos[:,1] = ( 0.,          0.,          0.)        # CA
    ref_pos[:,2] = ( 1.25222632, -0.87268266,  0.)        # C
    ref_pos[:,3] = ( 0.,          0.94375626,  1.2068012) # CB
    ref_pos[fasta=='GLY',3] = np.nan

    ref_pos -= ref_pos[:,:3].mean(axis=1)[:,None]

    create_array(grp, 'id', obj=np.arange(n_res))
    create_array(grp, 'ref_pos', obj=ref_pos)
    create_array(grp, 'n_atom',  obj=np.isfinite(grp.ref_pos[:].sum(axis=-1)).sum(axis=-1))


def write_affine_alignment(n_res):
    grp = t.create_group(potential, 'affine_alignment')
    grp._v_attrs.arguments = np.array(['pos'])

    ref_geom = np.zeros((n_res,3,3))
    ref_geom[:,0] = (-1.19280531, -0.83127186, 0.)  # N
    ref_geom[:,1] = ( 0.,          0.,         0.)  # CA
    ref_geom[:,2] = ( 1.25222632, -0.87268266, 0.)  # C
    ref_geom -= ref_geom.mean(axis=1)[:,None]

    N  = np.arange(n_res)*3 + 0
    CA = np.arange(n_res)*3 + 1
    C  = np.arange(n_res)*3 + 2

    atoms = np.column_stack((N,CA,C))
    create_array(grp, 'atoms', obj=atoms)
    create_array(grp, 'ref_geom', obj=ref_geom)


def write_infer_H_O(fasta, excluded_residues):
    n_res = len(fasta)
    # note that proline is not an hbond donor since it has no NH
    donor_residues    = np.array([i for i in range(n_res) if i>0       and i not in excluded_residues and fasta[i]!='PRO'])
    acceptor_residues = np.array([i for i in range(n_res) if i<n_res-1 and i not in excluded_residues])

    print
    print 'hbond, %i donors, %i acceptors in sequence' % (len(donor_residues), len(acceptor_residues))

    H_bond_length = 0.88
    O_bond_length = 1.24

    grp = t.create_group(potential, 'infer_H_O')
    grp._v_attrs.arguments = np.array(['pos'])

    donors    = t.create_group(grp, 'donors')
    acceptors = t.create_group(grp, 'acceptors')

    create_array(donors,    'residue', obj=donor_residues)
    create_array(acceptors, 'residue', obj=acceptor_residues)

    create_array(donors,    'bond_length', obj=H_bond_length*np.ones(len(   donor_residues)))
    create_array(acceptors, 'bond_length', obj=O_bond_length*np.ones(len(acceptor_residues)))

    create_array(donors,    'id', obj=np.array((-1,0,1))[None,:] + 3*donor_residues   [:,None])
    create_array(acceptors, 'id', obj=np.array(( 1,2,3))[None,:] + 3*acceptor_residues[:,None])


def write_environment(fasta, environment_library):
    cgrp = t.create_group(potential, 'environment_vector')
    cgrp._v_attrs.arguments = np.array(['placement_cb_only','placement4_weighted'])
    # cgrp._v_attrs.arguments = np.array(['placement_point_vector','placement4_weighted'])

    with tb.open_file(environment_library) as data:
         restype_order = dict([(str(x),i) for i,x in enumerate(data.root.restype_order[:])])
         create_array(cgrp, 'interaction_param', data.root.coverage_interaction[:])

    grp = t.create_group(potential, 'placement_cb_only')
    grp._v_attrs.arguments = np.array(['affine_alignment','rama_coord',])
    create_array(grp, 'rama_residue',    np.arange(len(fasta)))
    create_array(grp, 'affine_residue',  np.arange(len(fasta)))
    create_array(grp, 'layer_index',     np.zeros(len(fasta),dtype='i'))

    CB_placement_pos_and_dir = np.zeros((1,36,36,6), dtype='f4')
    CB_placement_pos_and_dir[:] = [-0.02366877,  1.51042092,  1.20528042, 
                                   -0.00227792,  0.61587566, 0.78784013]

    create_array(grp, 'placement_data',  CB_placement_pos_and_dir)

    # group1 is the source sidechain rotamer
    create_array(cgrp, 'index1', np.arange(len(fasta)))
    create_array(cgrp, 'type1',  np.array([restype_order[s] for s in fasta]))
    create_array(cgrp, 'id1',    np.arange(len(fasta)))

    # # group1 is the source sidechain rotamer
    # rot_grp = t.root.input.potential.placement_point_vector
    # create_array(cgrp, 'index1', np.arange(len(rot_grp.beadtype_seq[:])))
    # create_array(cgrp, 'type1',  np.array([restype_order[s] for s in rot_grp.beadtype_seq[:]]))
    # create_array(cgrp, 'id1',    rot_grp.affine_residue[:])

    # group 2 is the weighted points to interact with
    w_grp = t.root.input.potential.placement4_weighted
    create_array(cgrp, 'index2', np.arange(len(w_grp.beadtype_seq[:])))
    create_array(cgrp, 'type2',  np.array([restype_order[s] for s in w_grp.beadtype_seq[:]]))
    create_array(cgrp, 'id2',    w_grp.affine_residue[:])

    # egrp = t.create_group(potential, 'environment_energy')
    # egrp._v_attrs.arguments = np.array(['environment_vector'])

    # with tb.open_file(environment_library) as data:
    #      restype_order = dict([(str(x),i) for i,x in enumerate(data.root.restype_order[:])])
    #      create_array(egrp, "linear_weight0", obj=data.root.linear_weight0[:])
    #      create_array(egrp, "linear_shift0",  obj=data.root.linear_shift0 [:])
    #      create_array(egrp, "linear_weight1", obj=data.root.linear_weight1[:])
    #      create_array(egrp, "linear_shift1",  obj=data.root.linear_shift1 [:])
    # create_array(egrp, 'output_restype',  np.array([restype_order[s] for s in rot_grp.beadtype_seq[:]]))

    egrp = t.create_group(potential, 'simple_environment')
    egrp._v_attrs.arguments = np.array(['environment_vector'])

    with tb.open_file(environment_library) as data:
         restype_order = dict([(str(x),i) for i,x in enumerate(data.root.restype_order[:])])
         restype_coeff = data.root.restype_coeff[:]

    coeff = np.array([restype_coeff[restype_order[s]] for s in fasta])
    create_array(egrp, 'coefficients', obj=coeff);


def write_count_hbond(fasta, hbond_energy, coverage_library):
    n_res = len(fasta)

    infer_group = t.get_node('/input/potential/infer_H_O')

    n_donor    = infer_group.donors   .id.shape[0]
    n_acceptor = infer_group.acceptors.id.shape[0]

    igrp = t.create_group(potential, 'protein_hbond')
    igrp._v_attrs.arguments = np.array(['infer_H_O'])

    # group1 is the HBond donors
    create_array(igrp, 'index1', np.arange(0,n_donor))
    create_array(igrp, 'type1',  np.zeros(n_donor, dtype='i'))
    create_array(igrp, 'id1',    infer_group.donors.residue[:])

    # group 2 is the HBond acceptors
    create_array(igrp, 'index2', np.arange(n_donor,n_donor+n_acceptor))
    create_array(igrp, 'type2',  np.zeros(n_acceptor, dtype='i'))
    create_array(igrp, 'id2',    infer_group.acceptors.residue[:])

    # parameters are inner_barrier, inner_scale, outer_barrier, outer_scale, wall_dp, inv_dp_width
    # FIXME currently changing these has no effect
    create_array(igrp, 'interaction_param', np.array([[
        [1.4,   1./0.10,
         2.5,   1./0.125,
         0.682, 1./0.05]]]))

    cgrp = t.create_group(potential, 'hbond_coverage')
    cgrp._v_attrs.arguments = np.array(['protein_hbond','placement_point_vector'])

    with tb.open_file(coverage_library) as data:
         create_array(cgrp, 'interaction_param', data.root.coverage_interaction[:])
         bead_num = dict((k,i) for i,k in enumerate(data.root.bead_order[:]))
         hydrophobe_placement = data.root.hydrophobe_placement[:]
         hydrophobe_interaction = data.root.hydrophobe_interaction[:]

    # group1 is the HBond partners
    create_array(cgrp, 'index1', np.arange(n_donor+n_acceptor))
    create_array(cgrp, 'type1',  1*(np.arange(n_donor+n_acceptor) >= n_donor))  # donor is 0, acceptor is 1
    create_array(cgrp, 'id1',    np.concatenate([infer_group.donors   .residue[:],
                                                 infer_group.acceptors.residue[:]]))

    # group 2 is the sidechains
    rseq = t.root.input.potential.placement_point_vector.beadtype_seq[:]
    sc_resnum = t.root.input.potential.placement_point_vector.affine_residue[:]
    create_array(cgrp, 'index2', np.arange(len(rseq)))
    create_array(cgrp, 'type2',  np.array([bead_num[s] for s in rseq]))
    create_array(cgrp, 'id2',    sc_resnum)

    grp = t.create_group(potential, 'placement_fixed_point_vector_scalar')
    grp._v_attrs.arguments = np.array(['affine_alignment'])
    create_array(grp, 'affine_residue',  np.arange(3*n_res)/3)
    create_array(grp, 'layer_index',     np.arange(3*n_res)%3)
    create_array(grp, 'placement_data',  hydrophobe_placement)

    cgrp = t.create_group(potential, 'hbond_coverage_hydrophobe')
    cgrp._v_attrs.arguments = np.array(['placement_fixed_point_vector_scalar','placement_point_vector'])

    with tb.open_file(coverage_library) as data:
         create_array(cgrp, 'interaction_param', data.root.hydrophobe_interaction[:])
         bead_num = dict((k,i) for i,k in enumerate(data.root.bead_order[:]))

    # group1 is the hydrophobes
    # create_array(cgrp, 'index1', np.arange(n_res))
    # create_array(cgrp, 'type1',  0*np.arange(n_res))
    # create_array(cgrp, 'id1',    np.arange(n_res))
    create_array(cgrp, 'index1', np.arange(3*n_res))
    create_array(cgrp, 'type1',  np.arange(3*n_res)%3)
    create_array(cgrp, 'id1',    np.arange(3*n_res)/3)

    # group 2 is the sidechains
    rseq = t.root.input.potential.placement_point_vector.beadtype_seq[:]
    create_array(cgrp, 'index2', np.arange(len(rseq)))
    create_array(cgrp, 'type2',  np.array([bead_num[s] for s in rseq]))
    create_array(cgrp, 'id2',    sc_resnum)

    if hbond_energy > 0.:
        print '\n**** WARNING ****  hydrogen bond formation energy set to repulsive value\n'

    grp = t.create_group(potential, 'hbond_energy')
    grp._v_attrs.arguments = np.array(['protein_hbond'])
    grp._v_attrs.protein_hbond_energy = hbond_energy



def make_restraint_group(group_num, residues, initial_pos, strength):
    np.random.seed(314159)  # make groups deterministic

    grp = t.root.input.potential.dist_spring

    id = grp.id[:]
    equil_dist = grp.equil_dist[:]
    spring_const = grp.spring_const[:]
    bonded_atoms = grp.bonded_atoms[:]
    n_orig = id.shape[0]

    r_atoms = np.array([(3*i+0,3*i+1,3*i+2) for i in sorted(residues)]).reshape((-1,))
    random_pairing = lambda: np.column_stack((r_atoms, np.random.permutation(r_atoms)))

    pairs = np.concatenate([random_pairing() for i in range(2)], axis=0)
    pairs = [((x,y) if x<y else (y,x)) for x,y in pairs if x/3!=y/3]   # avoid same-residue restraints
    pairs = np.array(sorted(set(pairs)))

    pair_dists = vmag(initial_pos[pairs[:,0]]-initial_pos[pairs[:,1]])

    grp.id._f_remove()
    grp.equil_dist._f_remove()
    grp.spring_const._f_remove()
    grp.bonded_atoms._f_remove()

    create_array(grp, 'id',           obj=np.concatenate((id,          pairs),      axis=0))
    create_array(grp, 'equil_dist',   obj=np.concatenate((equil_dist,  pair_dists), axis=0))
    create_array(grp, 'spring_const', obj=np.concatenate((spring_const,strength*np.ones(len(pairs))),axis=0))
    create_array(grp, 'bonded_atoms', obj=np.concatenate((bonded_atoms,np.zeros(len(pairs),dtype='int')),axis=0))


def make_tab_matrices(phi, theta, bond_length):
    '''TAB matrices are torsion-angle-bond affine transformation matrices'''
    phi         = np.asarray(phi)
    theta       = np.asarray(theta)
    bond_length = np.asarray(bond_length)

    assert phi.shape == theta.shape == bond_length.shape
    r = np.zeros(phi.shape + (4,4), dtype=(phi+theta+bond_length).dtype)
    
    cp = np.cos(phi  ); sp = np.sin(phi  )
    ct = np.cos(theta); st = np.sin(theta)
    l  = bond_length

    r[...,0,0]=   -ct; r[...,0,1]=    -st; r[...,0,2]=   0; r[...,0,3]=   -l*ct;
    r[...,1,0]= cp*st; r[...,1,1]= -cp*ct; r[...,1,2]= -sp; r[...,1,3]= l*cp*st;
    r[...,2,0]= sp*st; r[...,2,1]= -sp*ct; r[...,2,2]=  cp; r[...,2,3]= l*sp*st;
    r[...,3,0]=     0; r[...,3,1]=      0; r[...,3,2]=   0; r[...,3,3]=       1;

    return r


def construct_equilibrium_structure(rama, angles, bond_lengths):
    assert rama.shape == angles.shape == bond_lengths.shape
    n_res = rama.shape[0]
    n_atom = 3*n_res
    assert rama.shape == (n_res,3)

    t = np.zeros(n_atom)
    a = angles.ravel()
    b = bond_lengths.ravel()

    t[3::3] = rama[:-1,1]
    t[4::3] = rama[:-1,2]
    t[5::3] = rama[1: ,0]

    transforms = make_tab_matrices(t,a,b)
    curr_affine = np.eye(4)
    pos = np.zeros((3*n_res,3))

    # right apply all transformations

    for i,mat in enumerate(transforms):
        curr_affine = np.dot(curr_affine, mat)
        pos[i] = curr_affine[:3,3]
    return pos


def random_initial_config(n_res):
    # a reasonable model where the chain grows obeying sensible angles and omegas
    rama    = np.random.random((n_res,3))*2*np.pi - np.pi
    angles  = np.zeros_like(rama)
    lengths = np.zeros_like(rama)

    rama[:,2] = np.pi   # all trans omega's

    angles[:,0] = 120.0*deg  # CA->C->N angle
    angles[:,1] = 120.0*deg  # C->N->CA angle
    angles[:,2] = 109.5*deg  # N->CA->C angle

    lengths[:,0] = 1.453
    lengths[:,1] = 1.526
    lengths[:,2] = 1.300
    return construct_equilibrium_structure(rama, angles, lengths)


# write dist_spring potential
def write_dist_spring(args):
    # create a linear chain
    grp = t.create_group(potential, 'dist_spring')
    grp._v_attrs.arguments = np.array(['pos'])
    id = np.arange(n_atom-1)
    id = np.column_stack((id,id+1))

    equil_dist = np.zeros(id.shape[0])
    equil_dist[0::3] = 1.453
    equil_dist[1::3] = 1.526
    equil_dist[2::3] = 1.300

    spring_const = args.bond_stiffness*np.ones(id.shape[0])
    bonded_atoms = np.ones(id.shape[0], dtype='int')

    create_array(grp, 'id', obj=id)
    create_array(grp, 'equil_dist',   obj=equil_dist)
    create_array(grp, 'spring_const', obj=spring_const)
    create_array(grp, 'bonded_atoms', obj=bonded_atoms)

def write_angle_spring(args):
    grp = t.create_group(potential, 'angle_spring')
    grp._v_attrs.arguments = np.array(['pos'])
    id = np.arange(n_atom-2)
    id = np.column_stack((id,id+2,id+1))
    equil_angles = np.zeros(id.shape[0])
    equil_angles[0::3] = np.cos(109.5*deg)  # N->CA->C angle
    equil_angles[1::3] = np.cos(120.0*deg)  # CA->C->N angle
    equil_angles[2::3] = np.cos(120.0*deg)  # C->N->CA angle

    create_array(grp, 'id', obj=id)
    create_array(grp, 'equil_dist',   obj=equil_angles)
    create_array(grp, 'spring_const', obj=args.angle_stiffness*np.ones(id.shape[0]))

def write_dihedral_spring(fasta_seq):
    # this is primarily used for omega bonds
    grp = t.create_group(potential, 'dihedral_spring')
    grp._v_attrs.arguments = np.array(['pos'])
    id = np.arange(1,n_atom-3,3)  # start at CA atom
    id = np.column_stack((id,id+1,id+2,id+3))

    target_angle = np.where((fasta_seq[1:]=='CPR'), 0.*deg, 180.*deg)

    create_array(grp, 'id', obj=id)
    create_array(grp, 'equil_dist',   obj=target_angle)
    create_array(grp, 'spring_const', obj=30.0*np.ones(id.shape[0]))


def basin_cond_prob_fcns(a_phi, a_psi):
    def basin_box(phi0,phi1, psi0,psi1):
        if phi0 > phi1: phi1 += 2*np.pi
        if psi0 > psi1: psi1 += 2*np.pi
        assert phi0 < phi1
        assert psi0 < psi1

        phi_mid  = 0.5*(phi1 + phi0)
        psi_mid  = 0.5*(psi1 + psi0)

        phi_switch = np.cos(phi1 - phi_mid)
        psi_switch = np.cos(psi1 - psi_mid)

        def f(phi,psi, phi_mid=phi_mid, psi_mid=psi_mid, phi_switch=phi_switch, psi_switch=psi_switch,
                a_phi=a_phi, a_psi=a_psi):
            dphi = np.cos(phi - phi_mid)  # cos in the loc function ensures continuous, periodic function
            dpsi = np.cos(psi - psi_mid)

            return 1./(
                 (1.+np.exp(-a_phi*(dphi-phi_switch))) *
                 (1.+np.exp(-a_psi*(dpsi-psi_switch))) )
        return f

    bb = lambda phi0, phi1, psi0,psi1: basin_box(phi0*deg, phi1*deg, psi0*deg, psi1*deg)

    basin_fcns = [
            bb(-180.,   0., -100.,  50.),   # alpha_R
            bb(-180.,-100.,   50.,-100.),   # beta
            bb(-100.,   0.,   50.,-100.),   # PPII
            bb(   0., 180.,  -50., 100.),   # alpha_L
            bb(   0., 180.,  100., -50.)]   # gamma

    basin_cond_prob = [
        (lambda phi,psi, bf=bf: bf(phi,psi)/sum(bf2(phi,psi) for bf2 in basin_fcns))
        for bf in basin_fcns]

    return basin_cond_prob


def mixture_potential(weights, potentials):
    ''' potentials must be normalized to the same value, preferably 1 '''
    potentials = np.array(potentials)
    assert len(weights) == len(potentials)
    weights = np.array(weights)
    weights = weights / weights.sum(axis=0)

    # ensure that we broadcast correctly against the potential
    weight_broadcast_shape = weights.shape + (1,)*(len(potentials.shape)-len(weights.shape))
    weights = weights.reshape(weight_broadcast_shape)

    potentials = potentials - np.log(weights)

    min_pot = potentials.min(axis=0)
    return min_pot - np.log(np.exp(min_pot-potentials).sum(axis=0))


def read_rama_maps_and_weights(seq, rama_group, mode='mixture'):
    assert mode in ['mixture', 'product']
    restype = rama_group._v_attrs.restype
    dirtype = rama_group._v_attrs.dir
    ridx = dict([(x,i) for i,x in enumerate(restype)])
    didx = dict([(x,i) for i,x in enumerate(dirtype)])

    dimer_pot    = rama_group.dimer_pot[:]
    dimer_weight = rama_group.dimer_weight[:]

    assert len(seq) >= 3   # avoid bugs

    # cis-proline is only CPR when it is the central residue, otherwise just use PRO
    V = lambda r,d,n: dimer_pot   [ridx[r], didx[d], (ridx[n] if n!='CPR' else ridx['PRO'])]
    W = lambda r,d,n: dimer_weight[ridx[r], didx[d], (ridx[n] if n!='CPR' else ridx['PRO'])]

    pots    = np.zeros((len(seq), dimer_pot.shape[-2], dimer_pot.shape[-1]), dtype='f4')
    weights = np.zeros((len(seq),),dtype='f4')

    pots   [0] = V(seq[0], 'right', seq[1])
    weights[0] = W(seq[0], 'right', seq[1])
        
    for i,l,c,r in zip(range(1,len(seq)-1), seq[:-2], seq[1:-1], seq[2:]):
        if   mode == 'product':
            pots[i]    = V(c,'left',l) + V(c,'right',r) - V(c,'right','ALL') 
            weights[i] = 0.5*(W(c,'left',l) + W(c,'right',r))  # always just average weights
        elif mode == 'mixture':
            # it's a little sticky to figure out what the mixing proportions should be
            # there is basically a one-sided vs two-sided problem (what if we terminate a sheet?)
            # I am going with one interpretation that may not be right
            pots[i]    = mixture_potential([W(c,'left',l), W(c,'right',r)], [V(c,'left',l), V(c,'right',r)])
            weights[i] = 0.5*(W(c,'left',l) + W(c,'right',r))
        else:
            raise RuntimeError('impossible')
        
    pots   [-1] = V(seq[-1], 'left', seq[-2])
    weights[-1] = W(seq[-1], 'left', seq[-2])

    # Ensure normalization
    pots -= -np.log(np.exp(-1.0*pots).sum(axis=(-2,-1), keepdims=1))

    return pots, weights


def read_weighted_maps(seq, rama_library_h5, sheet_mixing=None):
    with tb.open_file(rama_library_h5) as tr:
        coil_pots, coil_weights = read_rama_maps_and_weights(seq, tr.root.coil, mode='mixture')

        if sheet_mixing is None:
            return coil_pots
        else:
            sheet_pots, sheet_weights = read_rama_maps_and_weights(seq, tr.root.sheet)
            return mixture_potential([coil_weights, sheet_weights*np.exp(-sheet_mixing)], 
                                     [coil_pots,    sheet_pots])


def write_torus_dbn(seq, torus_dbn_library):
    # FIXME use omega emission to handle CPR code
    with tb.open_file(torus_dbn_library) as data:
        dbn_aa_num = dict((x,i) for i,x in enumerate(data.root.restype_order[:]))

        log_normalization = data.root.TORUS_LOGNORMCONST[:]
        kappa = data.root.TORUS_KAPPA[:]
        mu = data.root.TORUS_MU[:]
        aa_emission_energy = -np.log(data.root.AA_EMISSION[:])

        transition_matrix = data.root.HIDDEN_TRANSITION[:]

        n_state = transition_matrix.shape[0]

    rtype = np.array([dbn_aa_num[s] for s in seq])  # FIXME handle cis-proline, aka CPR
    prior_offset = np.zeros((len(rtype),n_state),'f4')
    
    for i,r in enumerate(seq):
        prior_offset[i,:] = aa_emission_energy[:,dbn_aa_num[r]]

    basin_param = np.zeros((n_state,6),'f4')
    basin_param[:,0] = log_normalization.ravel()
    basin_param[:,1] = kappa[:,0]
    basin_param[:,2] = mu   [:,0]
    basin_param[:,3] = kappa[:,1]
    basin_param[:,4] = mu   [:,1]
    basin_param[:,5] = kappa[:,2]

    egrp = t.create_group(potential, 'torus_dbn')
    egrp._v_attrs.arguments = np.array(['rama_coord'])

    # since Rama angles are not valid for the first and last angles,
    # don't confuse the HMM by including them
    create_array(egrp, 'id', np.arange(1,len(seq)-1))
    create_array(egrp, 'prior_offset', prior_offset[1:-1])
    create_array(egrp, 'basin_param',  basin_param)

    hgrp = t.create_group(potential, 'fixed_hmm')
    hgrp._v_attrs.arguments = np.array(['torus_dbn'])

    create_array(hgrp, 'index', np.arange(egrp.id.shape[0]))
    create_array(hgrp, 'transition_matrix', transition_matrix)


def write_rama_map_pot(seq, rama_library_h5, sheet_mixing_energy=None, helical_energy_shift=None):
    grp = t.create_group(potential, 'rama_map_pot')
    grp._v_attrs.arguments = np.array(['rama_coord'])

    rama_pot = read_weighted_maps(seq, rama_library_h5, sheet_mixing_energy)

    if sheet_mixing_energy is not None:
        # support finite differencing for potential derivative
        eps = 1e-2
        grp._v_attrs.sheet_eps = eps
        #create_array(grp, 'more_sheet_rama_pot', read_weighted_maps(seq, rama_library_h5, sheet_mixing_energy+eps))
        #create_array(grp, 'less_sheet_rama_pot', read_weighted_maps(seq, rama_library_h5, sheet_mixing_energy-eps))

    if helical_energy_shift is not None:
        assert len(rama_pot.shape) == 3
        phi = np.linspace(-np.pi,np.pi,rama_pot.shape[1],endpoint=False)[:,None]
        psi = np.linspace(-np.pi,np.pi,rama_pot.shape[2],endpoint=False)[None,:]
        sigmoid_lessthan = lambda a,b: 1./(1.+np.exp(-(b-a)/(10.*deg)))
        helical_basin = sigmoid_lessthan(phi,0.*deg) * sigmoid_lessthan(-100.*deg,psi) * sigmoid_lessthan(psi,50.*deg)
        rama_pot += (helical_energy_shift * helical_basin)[None,:,:]

    # let's remove the average energy from each Rama map 
    # so that the Rama potential emphasizes its variation

    rama_pot -= (rama_pot*np.exp(-rama_pot)).sum(axis=(-2,-1),keepdims=1)

    create_array(grp, 'residue_id',   obj=np.arange(len(seq)))
    create_array(grp, 'rama_map_id',  obj=np.arange(rama_pot.shape[0]))
    create_array(grp, 'rama_pot',     obj=rama_pot)


def compact_sigmoid(x, sharpness):
    y = x*sharpness;
    result = 0.25 * (y+2) * (y-1)**2
    result = np.where((y< 1), result, np.zeros_like(result))
    result = np.where((y>-1), result, np.ones_like (result))
    return result


def double_compact_sigmoid(x, half_width, sharpness):
    return compact_sigmoid(x-half_width, sharpness) * compact_sigmoid(-x-half_width, sharpness)


def angular_compact_double_sigmoid(theta, center, half_width, sharpness):
    dev = theta-center
    dev = np.where((dev< np.pi), dev, dev-2*np.pi)
    dev = np.where((dev>-np.pi), dev, dev+2*np.pi)
    return double_compact_sigmoid(dev, half_width, sharpness)


def rama_box(rama, center, half_width, sharpness):
    # print 'center', center
    # print 'half_width', half_width
    # print 'sharpness', sharpness, 1/sharpness
    assert rama.shape[-1] == center.shape[-1] == half_width.shape[-1] == 2

    s = center.shape[:-1]
    if not s:
        return (angular_compact_double_sigmoid(rama[...,0], rama, center, half_width, sharpness)*
                angular_compact_double_sigmoid(rama[...,1], rama, center, half_width, sharpness))
    else:
        result = np.zeros(rama.shape[:-1] + center.shape[:-1]) 
        for inds in np.indices(s).reshape((len(s),-1)).T:
            inds = tuple(inds)
            if len(inds) == 1: inds = inds[0]
            value = (
                    angular_compact_double_sigmoid(rama[...,0], center[inds,0], half_width[inds,0], sharpness)*
                    angular_compact_double_sigmoid(rama[...,1], center[inds,1], half_width[inds,1], sharpness))
            result[...,inds] = value
        return result


def read_fasta(file_obj):
    lines = list(file_obj)
    assert lines[0][0] == '>'
    one_letter_seq = ''.join(x.strip().replace('\r','') for x in lines[1:])
    seq = []
    cis_state = False
    for a in one_letter_seq:
        if cis_state:
            assert a == 'P'  # proline must follow start
            seq.append('CPR')
            cis_state = False
        elif a == "*":
            cis_state = True
        else:
            seq.append(three_letter_aa[a])
    return np.array(seq)


def write_contact_energies(parser, fasta, contact_table):
    fields = [ln.split() for ln in open(contact_table,'U')]
    if [x.lower() for x in fields[0]] != 'residue1 residue2 r0 width energy'.split():
        parser.error('First line of contact energy table must be "residue1 residue2 r0 width energy"')
    if not all(len(f)==5 for f in fields):
        parser.error('Invalid format for contact file')
    fields = fields[1:]
    n_contact = len(fields)

    g = t.create_group(t.root.input.potential, 'contact')
    g._v_attrs.arguments = np.array(['placement_point_only_backbone_dependent_point'])

    id         = np.zeros((n_contact,2), dtype='i')
    r0         = np.zeros((n_contact,))
    scale      = np.zeros((n_contact,))
    energy     = np.zeros((n_contact,))

    for i,f in enumerate(fields):
        id[i] = (int(f[0]), int(f[1]))
        msg = 'Contact energy specified for residue %i (zero is first residue) but there are only %i residues in the FASTA'
        if not (0 <= id[i,0] < len(fasta)): raise ValueError(msg % (id[i,0], len(fasta)))
        if not (0 <= id[i,1] < len(fasta)): raise ValueError(msg % (id[i,1], len(fasta)))

        r0[i]     =     float(f[2])
        scale[i]  = 0.5/float(f[3])  # compact_sigmoid cuts off at +/- 1/scale
        energy[i] =     float(f[4])

    if energy.max() > 0.:
        print ('\nWARNING: Some contact energies are positive (repulsive).\n'+
                 '         Please ignore this warning if you intendent to have repulsive contacts.')

    create_array(g, 'id',         obj=id)
    create_array(g, 'r0',         obj=r0)
    create_array(g, 'scale',      obj=scale)
    create_array(g, 'energy',     obj=energy)

def write_rama_coord():
    grp = t.create_group(potential, 'rama_coord')
    grp._v_attrs.arguments = np.array(['pos'])
    n_res = n_atom/3
    N_id = 3*np.arange(n_res)
    id = np.column_stack((N_id-1,N_id,N_id+1,N_id+2,N_id+3))
    id[id>=n_atom] = -1  # last atom is non-existent (as is first)
                         #   and non-existence is indicated by -1
    create_array(grp, 'id', id)


def write_backbone_dependent_point(fasta, library):
    grp = t.create_group(potential, 'placement_point_only_backbone_dependent_point')
    grp._v_attrs.arguments = np.array(['affine_alignment','rama_coord'])

    with tb.open_file(library) as data:
        n_restype = len(aa_num)
        n_bin = data.get_node('/ALA/center').shape[0]-1  # ignore periodic repeat of last bin
        point_map = np.zeros((n_restype,n_bin,n_bin, 3),dtype='f4')  

        for rname,idx in sorted(aa_num.items()):
            point_map[idx] = data.get_node('/%s/center'%rname)[:-1,:-1]

    create_array(grp, 'rama_residue',    np.arange(len(fasta)))
    create_array(grp, 'affine_residue',  np.arange(len(fasta)))
    create_array(grp, 'layer_index',     np.array([aa_num[s] for s in fasta]))
    create_array(grp, 'placement_data',  point_map)


def write_sidechain_radial(fasta, library, excluded_residues, suffix=''):
    g = t.create_group(t.root.input.potential, 'radial'+suffix)
    g._v_attrs.arguments = np.array(['placement_point_only_backbone_dependent_point'])
    for res_num in excluded_residues:
        if not (0<=res_num<len(fasta)):
            raise ValueError('Residue number %i is invalid'%res_num)

    residues = sorted(set(np.arange(len(fasta))).difference(excluded_residues))

    with tb.open_file(library) as params:
        resname2restype = dict((x,i) for i,x in enumerate(params.root.names[:]))
        n_type = len(resname2restype)

        create_array(g, 'index', obj=np.array(residues))
        create_array(g, 'type',  obj=np.array([resname2restype[x] for x in fasta[residues]]))
        create_array(g, 'id',    obj=np.array(residues))  # FIXME update for chain breaks
        create_array(g, 'interaction_param', obj=params.root.interaction_param[:])


def write_weighted_placement(fasta, placement_library):
    assert not 'Weighted placement is currently broken because it does not handle bead types'
    with tb.open_file(placement_library) as data:
        restype_num = dict((aa,i) for i,aa in enumerate(data.root.restype_order[:]))
        placement_pos  = data.root.rotamer_center[:].transpose((2,0,1,3)) # must put layer index first
        placement_prob = data.root.rotamer_prob  [:].transpose((2,0,1))[...,None]
        start_stop = data.root.rotamer_start_stop[:]

    # truncate to only take the first point from the pos
    placement_data = np.concatenate((placement_pos[...,:3], placement_prob), axis=-1)

    rama_residue = []
    affine_residue = []
    layer_index = []
    beadtype_seq = []

    for rnum,aa in enumerate(fasta):
        restype = restype_num[aa]
        start,stop = start_stop[restype]
        n_rot = stop-start

        rama_residue  .extend([rnum]*n_rot)
        affine_residue.extend([rnum]*n_rot)
        layer_index   .extend(np.arange(start,stop))
        beadtype_seq   .extend([aa]*n_rot)

    grp = t.create_group(potential, 'placement4_weighted')
    grp._v_attrs.arguments = np.array(['affine_alignment','rama_coord'])
    create_array(grp, 'rama_residue',    rama_residue)
    create_array(grp, 'affine_residue',  affine_residue)
    create_array(grp, 'layer_index',     layer_index)
    create_array(grp, 'placement_data',  placement_data)
    create_array(grp, 'beadtype_seq',     beadtype_seq)


def write_rotamer_placement(fasta, placement_library, fix_rotamer):
    def compute_chi1_state(angles):
        chi1_state = np.ones(angles.shape, dtype='i')
        chi1_state[(   0.*deg<=angles)&(angles<120.*deg)] = 0
        chi1_state[(-120.*deg<=angles)&(angles<  0.*deg)] = 2
        return chi1_state

    with tb.open_file(placement_library) as data:
        restype_num = dict((aa,i) for i,aa in enumerate(data.root.restype_order[:]))
        placement_pos = data.root.rotamer_center[:].transpose((2,0,1,3)) # must put layer index first
        placement_energy = -np.log(data.root.rotamer_prob[:].transpose((2,0,1)))[...,None]
        start_stop = data.root.rotamer_start_stop_bead[:]
        find_restype =                       data.root.restype_and_chi_and_state[:,0].astype('i')
        find_chi1 =                          data.root.restype_and_chi_and_state[:,1]
        find_chi1_state = compute_chi1_state(data.root.restype_and_chi_and_state[:,1])
        find_chi2 =                          data.root.restype_and_chi_and_state[:,2]
        find_state =                         data.root.restype_and_chi_and_state[:,3].astype('i')

    fix = dict()
    if fix_rotamer:
        fields = [x.split()[:4] for x in list(open(fix_rotamer))]  # only consider first 4 column

        header = 'residue restype chi1 chi2'
        actual_header = [x.lower() for x in fields[0]]
        if actual_header != header.split():
            parser.error('First line of fix-rotamer table must be "%s" but is "%s"'
                    %(header," ".join(actual_header)))

        for residue, restype, chi1, chi2 in fields[1:]:
            if fasta[int(residue)] != (restype if restype != 'CPR' else 'PRO'): 
                raise RuntimeError("fix-rotamer file does not match FASTA"
                    + ", residue %i should be %s but fix-rotamer file has %s"%(
                        int(residue), fasta[int(residue)], restype))
            chi1 = float(chi1)*deg  # convert to radians internally
            chi2 = float(chi2)*deg

            if restype == 'GLY' or restype == 'ALA':
                fix_state = 0
            else:
                # determine states that have the right restype and compatible chi1
                chi1_state = compute_chi1_state(np.array([chi1]))[0]
                restype_admissible = find_restype == restype_num[fasta[int(residue)]]
                chi1_admissible = find_chi1_state == chi1_state
                admissible = restype_admissible&chi1_admissible
                admissible_chi2 = find_chi2[admissible]
                admissible_state = find_state[admissible]
                if len(admissible_state)==1:  # handle short residues (like VAL)
                    fix_state = admissible_state[0]
                else:
                    # now find the closest chi2 among those states and read off the state index
                    chi2_dist = (admissible_chi2-chi2)%(2*np.pi)
                    chi2_dist[chi2_dist>np.pi] -= 2*np.pi  # find closest periodic image
                    fix_state = admissible_state[np.argmin(chi2_dist)]

            fix[int(residue)] = fix_state

    rama_residue = []
    affine_residue = []
    layer_index = []
    beadtype_seq = []
    id_seq = []

    count_by_n_rot = dict()

    for rnum,aa in enumerate(fasta):
        restype = restype_num[aa]
        start,stop,n_bead = start_stop[restype]
        assert (stop-start)%n_bead == 0
        n_rot = (stop-start)//n_bead

        # if it should be fixed, then we must modify these answers to get a single rotamer
        if rnum in fix:
            if not (0 <= fix[rnum] < n_rot): raise ValueError('invalid fix rotamer state')
            start,stop = start+n_bead*fix[rnum], start+n_bead*(fix[rnum]+1)
            n_rot = 1

        if n_rot not in count_by_n_rot: 
            count_by_n_rot[n_rot] = 0;

        base_id = (count_by_n_rot[n_rot]<<n_bit_rotamer) + n_rot
        count_by_n_rot[n_rot] += 1

        rama_residue  .extend([rnum]*(stop-start))
        affine_residue.extend([rnum]*(stop-start))
        layer_index   .extend(np.arange(start,stop))
        beadtype_seq  .extend(['%s_%i'%(aa,i) for i in range(n_bead)]*n_rot)
        id_seq        .extend(np.arange(stop-start)//n_bead + (base_id<<n_bit_rotamer))

    grp = t.create_group(potential, 'placement_point_vector')
    grp._v_attrs.arguments = np.array(['affine_alignment','rama_coord'])
    create_array(grp, 'rama_residue',    rama_residue)
    create_array(grp, 'affine_residue',  affine_residue)
    create_array(grp, 'layer_index',     layer_index)
    create_array(grp, 'placement_data',  placement_pos[...,:6])
    create_array(grp, 'beadtype_seq',    beadtype_seq)
    create_array(grp, 'id_seq',          np.array(id_seq))

    grp = t.create_group(potential, 'placement_scalar')
    grp._v_attrs.arguments = np.array(['affine_alignment','rama_coord'])
    create_array(grp, 'rama_residue',    rama_residue)
    create_array(grp, 'affine_residue',  affine_residue)
    create_array(grp, 'layer_index',     layer_index)
    create_array(grp, 'placement_data',  placement_energy)


def write_rotamer(fasta, interaction_library, damping):
    g = t.create_group(t.root.input.potential, 'rotamer')
    args = ['placement_point_vector','placement_scalar']
    def arg_maybe(nm):
        if nm in t.root.input.potential: args.append(nm)
    arg_maybe('hbond_coverage')
    arg_maybe('hbond_coverage_hydrophobe')

    g._v_attrs.arguments = np.array(args)
    g._v_attrs.max_iter = 1000
    g._v_attrs.tol      = 1e-3
    g._v_attrs.damping  = damping
    g._v_attrs.iteration_chunk_size = 2

    pg = t.create_group(g, "pair_interaction")

    with tb.open_file(interaction_library) as data:
         create_array(pg, 'interaction_param', data.root.pair_interaction[:])
         bead_num = dict((k,i) for i,k in enumerate(data.root.bead_order[:]))
         # pg._v_attrs.energy_cap = data.root._v_attrs.energy_cap_1body
         # pg._v_attrs.energy_cap_width = data.root._v_attrs.energy_cap_width_1body

    rseq = t.root.input.potential.placement_point_vector.beadtype_seq[:]
    create_array(pg, 'index', np.arange(len(rseq)))
    create_array(pg, 'type',  np.array([bead_num[s] for s in rseq]))
    create_array(pg, 'id',    t.root.input.potential.placement_point_vector.id_seq[:])


def write_membrane_potential(sequence, potential_library_path, scale, membrane_thickness,
		             excluded_residues, UHB_residues_type1, UHB_residues_type2):
    grp = t.create_group(t.root.input.potential, 'membrane_potential')
    grp._v_attrs.arguments = np.array(['placement_point_only_backbone_dependent_point'])

    potential_library = tb.open_file(potential_library_path)
    resnames  = potential_library.root.names[:]
    z_energy  = potential_library.root.z_energy[:]
    z_lib_min = potential_library.root.z_energy._v_attrs.z_min
    z_lib_max = potential_library.root.z_energy._v_attrs.z_max
    potential_library.close()

    for res_num in list(excluded_residues): 
        if not (0<=res_num<len(sequence)):
            raise ValueError('Residue number %i is invalid in excluded_residues'%res_num)
    for res_num in list(UHB_residues_type1):
        if not (0<=res_num<len(sequence)):
            raise ValueError('Residue number %i is invalid in UHB_residues_type1'%res_num)
    for res_num in list(UHB_residues_type2):
        if not (0<=res_num<len(sequence)):
            raise ValueError('Residue number %i is invalid in UHB_residues_type2'%res_num)
					     
    UHB_residues_type1_included = sorted(set(UHB_residues_type1).difference(excluded_residues))
    UHB_residues_type2_included = sorted(set(UHB_residues_type2).difference(excluded_residues))

    if set(UHB_residues_type1_included).intersection(UHB_residues_type2_included) != None:
	for res_num in set(UHB_residues_type1_included).intersection(UHB_residues_type2_included):
	    raise ValueError('Residue number %i is in both UHB_type1 and UHB_type2 lists'%res_num)

    fasta_one_letter = [one_letter_aa[x] for x in sequence]
    print 
    print 'membrane_potential_residues_excluded:\n',''.join((f.upper() if i in excluded_residues else f.lower()) for i,f in enumerate(fasta_one_letter))
    print 
    print 'UHB_residues_type1_included:\n',''.join((f.upper() if i in UHB_residues_type1_included else f.lower()) for i,f in enumerate(fasta_one_letter))
    print
    print 'UHB_residues_type2_included:\n',''.join((f.upper() if i in UHB_residues_type2_included else f.lower()) for i,f in enumerate(fasta_one_letter))

    sequence = list(sequence) 
    for num in excluded_residues:
        sequence[num] = 'NON'                  # abbreviation for residues excluded from membrane potential
                                               # the residue will have membrane potential equal to 0 everywhere
                                               # the index for NON in the potential library is 0  
    for num in UHB_residues_type1_included:
        sequence[num] = sequence[num]+'UHB1'   # abbreviation for Unsatisfied HBond type1 
                                               # sequence is comprised of 3-letter codes of aa
    for num in UHB_residues_type2_included:
        sequence[num] = sequence[num]+'UHB2'   # abbreviation for Unsatisfied HBond type2 

    sequence = np.array(sequence)
    #print 'sequence: length ', len(sequence), '\n', sequence
    
    z_lib = np.linspace(z_lib_min, z_lib_max, z_energy.shape[-1])

    import scipy.interpolate
    def extrapolated_spline(x0,y0):
        spline = scipy.interpolate.InterpolatedUnivariateSpline(x0,y0)
        def f(x, spline=spline):
            return np.select(
                    [(x<x0[0]),              (x>x0[-1]),              np.ones_like(x,dtype='bool')], 
                    [np.zeros_like(x)+y0[0], np.zeros_like(x)+y0[-1], spline(x)])
        return f

    energy_splines = [extrapolated_spline(z_lib,ene) for ene in z_energy]

    half_thickness = membrane_thickness/2
    z = np.linspace(-half_thickness - 15., half_thickness + 15., int((membrane_thickness+30.)/0.3)+1)

    # if z>0, then use the spline evaluated at z-half_thickness, else use the spline evaluated at -z-half_thickness
    z_transformed     = np.where((z>=0.), z-half_thickness, -z-half_thickness)
    membrane_energies = np.array([spline(z_transformed) for spline in energy_splines])
    resname_to_num    = dict([(nm,i) for i,nm in enumerate(resnames)])

    residue_id_filtered = [i for i,x in enumerate(sequence) if sequence[i] != 'NON'] 
    sequence_filtered   = [x for i,x in enumerate(sequence) if sequence[i] != 'NON'] 
    energy_index        = np.array([resname_to_num[aa] for aa in sequence_filtered])

    #print sequence_filtered
    #print len(residue_id_filtered), residue_id_filtered
    #print len(energy_index),energy_index

    create_array(grp, 'residue_id', np.array(residue_id_filtered))
    create_array(grp, 'restype',    energy_index)
    create_array(grp, 'energy',     membrane_energies * scale)
    grp.energy._v_attrs.z_min = z[0]
    grp.energy._v_attrs.z_max = z[-1]


def parse_segments(s):
    ''' Parse segments of the form 10-30,50-60 '''
    import argparse
    import re

    if re.match('^([0-9]+(-[0-9]+)?)(,[0-9]+(-[0-9]+)?)*$', s) is None:
        raise argparse.ArgumentTypeError('segments must be of the form 10-30,45,72-76 or similar')

    def parse_seg(x):
        atoms = x.split('-')
        if len(atoms) == 1:
            return np.array([int(atoms[0])])
        elif len(atoms) == 2:
            return np.arange(int(atoms[0]),1+int(atoms[1]))  # inclusive on both ends
        else:
            raise RuntimeError('the impossible happened.  oops.')

    ints = np.concatenate([parse_seg(a) for a in s.split(',')])
    ints = np.array(sorted(set(ints)))   # remove duplicates and sort
    return ints


def parse_float_pair(s):
    import argparse
    import re

    args = s.split(',')
    if len(args) != 2:
        raise argparse.ArgumentTypeError('must be in the form -2.0,-1.0 or similar (exactly 2 numbers)')

    return (float(args[0]), float(args[1]))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Prepare input file',
            usage='use "%(prog)s --help" for more information')
    parser.add_argument('--fasta', required=True,
            help='[required] FASTA sequence file')
    parser.add_argument('--n-system', type=int, default=1, required=False, 
            help='[required] number of systems to prepare')
    parser.add_argument('--output', default='system.h5', required=True,
            help='path to output the created .h5 file (default system.h5)')
    parser.add_argument('--no-backbone', dest='backbone', default=True, action='store_false',
            help='do not use rigid nonbonded for backbone N, CA, C, and CB')
    parser.add_argument('--backbone-dependent-point', default=None,
            help='use backbone-depedent sidechain location library')
    parser.add_argument('--rotamer-placement', default=None, 
            help='rotameric sidechain library')
    parser.add_argument('--fix-rotamer', default='', 
            help='Table of fixed rotamers for specific sidechains.  A header line must be present and the first '+
            'three columns of that header must be '+
            '"residue restype rotamer", but there can be additional, ignored columns.  The restype must '+
            'match the corresponding restype in the FASTA file (intended to prevent errors).  It is permissible '+
            'to fix only a subset of rotamers.  The value of rotamer must be an integer, corresponding to the '+ 
            'numbering in the --rotamer-placement file.  Such a file can be created with PDB_to_initial_structure '+
            '--output-chi1.')
    parser.add_argument('--rotamer-interaction', default=None, 
            help='rotamer sidechain pair interaction parameters')
    parser.add_argument('--rotamer-solve-damping', default=0.4, type=float,
            help='damping factor to use for solving sidechain placement problem')
    parser.add_argument('--sidechain-radial', default=None,
            help='use sidechain radial potential library')
    parser.add_argument('--sidechain-radial-exclude-residues', default=[], type=parse_segments,
            help='Residues that do not participate in the --sidechain-radial potential (same format as --restraint-group)')
    parser.add_argument('--bond-stiffness', default=48., type=float,
            help='Bond spring constant in units of energy/A^2 (default 48)')
    parser.add_argument('--angle-stiffness', default=175., type=float,
            help='Angle spring constant in units of 1/dot_product (default 175)')
    parser.add_argument('--rama-library', default='',
            help='smooth Rama probability library')
    parser.add_argument('--torus-dbn-library', default='',
            help='TorusDBN Rama probability function')
    parser.add_argument('--rama-sheet-library', default=None,
            help='smooth Rama probability library for sheet structures')
    parser.add_argument('--helical-energy-shift', default=None, type=float,
            help='Energy shift to add to the helical basin (defined as -180.<phi<0 and -100.<psi<50, '+
            'slightly smoothed at edges).')
    parser.add_argument('--rama-sheet-mixing-energy', default=None, type=float,
            help='reference energy for sheets when mixing with coil library.  More negative numbers mean more '+
            'sheet content in the final structure.  Default is no sheet mixing.')
    parser.add_argument('--hbond-energy', default=0., type=float,
            help='energy for forming a protein-protein hydrogen bond.  Default is no HBond energy.')
    parser.add_argument('--hbond-exclude-residues', default=[], type=parse_segments,
            help='Residues to have neither hydrogen bond donors or acceptors') 
    parser.add_argument('--helix-energy-perturbation', default=None,
            help='hbond energy perturbation file for helices')
    parser.add_argument('--z-flat-bottom', default='', 
            help='Table of Z-flat-bottom springs.  Each line must contain 4 fields and the first line '+
            'must contain "residue z0 radius spring_constant".  The restraint is applied to the CA atom '+
            'of each residue.')
    parser.add_argument('--tension', default='', 
            help='Table of linear tensions.  Each line must contain 4 fields and the first line '+
            'must contain "residue tension_x tension_y tension_z".  The residue will be pulled in the '+
            'direction (tension_x,tension_y,tension_z) by its CA atom.  The magnitude of the tension vector '+
            'sets the force.  Units are kT/Angstrom.')
    parser.add_argument('--initial-structures', default='', 
            help='Pickle file for initial structures for the simulation.  ' +
            'If there are not enough structures for the number of replicas ' +
            'requested, structures will be recycled.  If not provided, a ' +
            'freely-jointed chain with good bond lengths and angles but bad dihedrals will be used ' +
            'instead.')
    parser.add_argument('--target-structures', default='', 
            help='Pickle file for target structures for the simulation.  ' +
            'This option controls the structure used for restraint group and '+
            'other structure-specific potentials.  If not provided, the initial '
            'structure is used as a default.')
    parser.add_argument('--restraint-group', default=[], action='append', type=parse_segments,
            help='Path to file containing whitespace-separated residue numbers (first residue is number 0).  '+
            'Each atom in the specified residues will be randomly connected to atoms in other residues by ' +
            'springs with equilibrium distance given by the distance of the atoms in the initial structure.  ' +
            'Multiple restraint groups may be specified by giving the --restraint-group flag multiple times '
            'with different filenames.')
    parser.add_argument('--restraint-spring-constant', default=4., type=float,
            help='Spring constant used to restrain atoms in a restraint group (default 4.) ')
    parser.add_argument('--dihedral-range', default='',
            help='Path to text file that defines a dihedral angle energy function.  The first line of the file should ' +
            'be a header containing "index angletype start end width energy", where angletype is either "phi" or "psi" and the remaining lines should contain '+
            'space separated values.  The form of the interaction is '+
            'energy*(1/(1+exp[-(x-x_lowboundary)/width]))*(1/(1+exp[(x-x_upboundary)/width])). '+
            'x is the dihedral angle, x_lowboundary and x_upboundary are the low and up boundaries.')
    parser.add_argument('--contact-energies', default='', 
            help='Path to text file that defines a contact energy function.  The first line of the file should ' +
            'be a header containing "residue1 residue2 r0 width energy", and the remaining lines should contain '+
            'space separated values.  The form of the interaction is '+
            'energy/(1+exp((|x_residue1-x_residue2|-r0)/width)).  The location x_residue is the centroid of ' +
            'sidechain, typically a few angstroms above the CB.')
    parser.add_argument('--environment-potential', default='',
            help='Path to many-body environment potential')
    parser.add_argument('--reference-state-rama', default='',
            help='Do not use this unless you know what you are doing.')
    parser.add_argument('--membrane-thickness', default=None, type=float,
            help='Thickness of the membrane in angstroms for use with --membrane-potential.')
    parser.add_argument('--membrane-potential', default='',
            help='Parameter file (.h5 format) for membrane potential. User must also supply --membrane-thickness.' + 
                 'There are 3 types of residue-specific types in the potential file now. ' +   
                 'Basic types  : XXX; Derived types: XXXUHB1, XXXUHB2. (XXX is the three-letter code for an amino acid) ' +
                 'UHB1: used for the Helical residues at both ends of a helix (usually 4 at each end), energy = XXX + UHB1. ' +
                 'UHB2: used for the non-Helical residues or unspecified non-hbonded residues, energy = XXX + UHB2.') 
    parser.add_argument('--membrane-potential-scale',            default=1.0,type=float,
            help='scale the membrane potentials. User must also supply --membrane-potential.')  
    parser.add_argument('--membrane-potential-exclude-residues', default=[], type=parse_segments,
            help='Residues that do not participate in the --membrane-potential(same format as --restraint-group).' +
                 'User must also supply --membrane-potential.')      
    parser.add_argument('--membrane-potential-unsatisfied-hbond-residues-type1',default=[], type=parse_segments,
            help='Residues that have 1 unsatisfied hydrogen bond, which will be marked as XXXUHB1 in --membrane-potential. ' +
                 'Normally, this argument is only turned on when user wants to determine the burial orientation of a given membrane protein ' +
                 'and the residues with unsatisfied hbonds are awared of (same format as --restraint-group). ' + 
                 'User must also supply --membrane-potential.')
    parser.add_argument('--membrane-potential-unsatisfied-hbond-residues-type2',default=[], type=parse_segments,
            help='Residues that have 2 unsatisfied hydrogen bonds, which will be marked as XXXUHB2 in --membrane-potential. ' +
                 'NOTE: the indices here should not have overlap with those in --membrane-potential-unsatisfied-hbond-residues-type1.')
    parser.add_argument('--cavity-radius', default=0., type=float,
            help='Enclose the whole simulation in a radial cavity centered at the origin to achieve finite concentration '+
            'of protein.  Necessary for multichain simulation (though this mode is unsupported.')
    parser.add_argument('--debugging-only-disable-basic-springs', default=False, action='store_true',
            help='Disable basic springs (like bond distance and angle).  Do not use this.')


    args = parser.parse_args()
    if args.restraint_group and not (args.initial_structures or args.target_structures):
        parser.error('must specify --initial-structures or --target-structures to use --restraint-group')

    if args.sidechain_radial and not args.backbone_dependent_point:
        parser.error('--sidechain-radial requires --backbone-dependent-point')

    fasta_seq_with_cpr = read_fasta(open(args.fasta,'U'))
    fasta_seq = np.array([(x if x != 'CPR' else 'PRO') for x in fasta_seq_with_cpr])  # most potentials don't care about CPR
    require_affine = False
    require_rama = False
    require_backbone_point = False

    global n_system, n_atom, t, potential
    n_system = args.n_system
    n_atom = 3*len(fasta_seq)
    
    t = tb.open_file(args.output,'w')
    
    input = t.create_group(t.root, 'input')
    create_array(input, 'sequence', obj=fasta_seq_with_cpr)
    
    if args.initial_structures:
        init_pos = cPickle.load(open(args.initial_structures))
        assert init_pos.shape == (n_atom, 3, init_pos.shape[-1])

    pos = np.zeros((n_atom, 3, n_system), dtype='f4')
    for i in range(n_system):
        pos[:,:,i] = init_pos[...,i%init_pos.shape[-1]] if args.initial_structures else random_initial_config(len(fasta_seq))
    create_array(input, 'pos', obj=pos)

    if args.target_structures:
        target = cPickle.load(open(args.target_structures))
        assert target.shape == (n_atom, 3, target.shape[-1])
        if target.shape[-1] != 1: 
            parser.error('Only a single target structure is supported, but your file contains multiple')
    else:
        target = pos.copy()
    
    potential = t.create_group(input,  'potential')

    if not args.debugging_only_disable_basic_springs:
        write_dist_spring(args)
        write_angle_spring(args)
        write_dihedral_spring(fasta_seq_with_cpr)

    if args.rotamer_interaction:
        if args.rotamer_placement is None:
            parser.error('--rotamer_placement is required, based on other options.')
        require_rama = True
        require_affine = True
        write_rotamer_placement(fasta_seq, args.rotamer_placement,args.fix_rotamer)

    if args.environment_potential:
        if args.rotamer_placement is None:
            parser.error('--rotamer_placement is required, based on other options.')
        write_weighted_placement(fasta_seq, args.rotamer_placement)
        write_environment(fasta_seq, args.environment_potential)

    if args.hbond_energy:
        write_infer_H_O  (fasta_seq, args.hbond_exclude_residues)
        write_count_hbond(fasta_seq, args.hbond_energy, args.rotamer_interaction)

    args_group = t.create_group(input, 'args')
    for k,v in sorted(vars(args).items()):
        args_group._v_attrs[k] = v
    args_group._v_attrs['invocation'] = ' '.join(sys.argv[:])

    if args.rama_library:
        require_rama = True
        write_rama_map_pot(fasta_seq_with_cpr, args.rama_library, args.rama_sheet_mixing_energy, args.helical_energy_shift)
    elif args.torus_dbn_library:
        require_rama = True
        write_torus_dbn(fasta_seq_with_cpr, args.torus_dbn_library)
    else:
        print>>sys.stderr, 'WARNING: running without any Rama potential !!!'

    # hack to fix reference state issues for Rama potential
    if args.reference_state_rama:
        # define correction
        ref_state_cor =  np.log(cPickle.load(open(args.reference_state_rama)))
        ref_state_cor -= ref_state_cor.mean()

        grp = t.create_group(potential, 'rama_map_pot_ref')
        grp._v_attrs.arguments = np.array(['rama_coord'])
        grp._v_attrs.log_pot = 0

        create_array(grp, 'residue_id',   obj=np.arange(len(fasta_seq)))
        create_array(grp, 'rama_map_id',  obj=np.zeros(len(fasta_seq), dtype='i4'))
        create_array(grp, 'rama_pot',     obj=ref_state_cor[None])

    if args.cavity_radius:
        write_cavity_radial(args.cavity_radius)

    if args.backbone:
        require_affine = True
        write_backbone_pair(fasta_seq)

    if args.z_flat_bottom:
        write_z_flat_bottom(parser,fasta_seq, args.z_flat_bottom)

    if args.tension:
        write_tension(parser,fasta_seq, args.tension)

    if args.rotamer_interaction:
        # must be after write_count_hbond if hbond_coverage is used
        write_rotamer(fasta_seq, args.rotamer_interaction, args.rotamer_solve_damping)

    if args.sidechain_radial:
        require_backbone_point = True
        write_sidechain_radial(fasta_seq, args.sidechain_radial, args.sidechain_radial_exclude_residues)

    if args.membrane_potential:
        if args.membrane_thickness is None:
            parser.error('--membrane-potential requires --membrane-thickness')
        require_backbone_point = True
        write_membrane_potential(fasta_seq, 
                                 args.membrane_potential, 
	                         args.membrane_potential_scale, 
	                         args.membrane_thickness,
	                         args.membrane_potential_exclude_residues, 
	                         args.membrane_potential_unsatisfied_hbond_residues_type1,
	                         args.membrane_potential_unsatisfied_hbond_residues_type2)

    if require_backbone_point:
        if args.backbone_dependent_point is None:
            parser.error('--backbone-dependent-point is required, based on other options.')
        require_affine = True
        require_rama = True
        write_backbone_dependent_point(fasta_seq, args.backbone_dependent_point)

    if require_rama:
        write_rama_coord()

    if args.contact_energies:
        require_backbone_point = True
        write_contact_energies(parser, fasta_seq, args.contact_energies)

    if require_affine:
        write_affine_alignment(len(fasta_seq))

    if args.restraint_group:
        print
        print 'Restraint groups (uppercase letters are restrained residues)'
        fasta_one_letter = ''.join(one_letter_aa[x] for x in fasta_seq)

        for i,rg in enumerate(args.restraint_group):
            restrained_residues = set(rg)
            assert np.amax(list(restrained_residues)) < len(fasta_seq)
            print 'group_%i: %s'%(i, ''.join((f.upper() if i in restrained_residues else f.lower()) 
                                              for i,f in enumerate(fasta_one_letter)))
            make_restraint_group(i,restrained_residues,target[:,:,0], args.restraint_spring_constant)
	    

    # if we have the necessary information, write pivot_sampler
    if require_rama and 'rama_map_pot' in potential:
        grp = t.create_group(input, 'pivot_moves')
        pivot_atom = potential.rama_coord.id[:]
        non_terminal_residue = np.array([not(np.int64(-1).astype(pivot_atom.dtype) in tuple(x)) 
            for x in pivot_atom])

        create_array(grp, 'proposal_pot',  potential.rama_map_pot.rama_pot[:])
        create_array(grp, 'pivot_atom',    pivot_atom[non_terminal_residue])
        create_array(grp, 'pivot_restype', potential.rama_map_pot.rama_map_id[:][non_terminal_residue])
        create_array(grp, 'pivot_range',   np.column_stack((grp.pivot_atom[:,4]+1,np.zeros(sum(non_terminal_residue),'i')+n_atom)))

    t.close()


if __name__ == '__main__':
    main()

