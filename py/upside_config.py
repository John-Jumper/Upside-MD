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

def highlight_residues(name, fasta, residues_to_highlight):
    fasta_one_letter = [one_letter_aa[x] for x in fasta]
    residues_to_highlight = set(residues_to_highlight)
    print '%s:  %s' % (name, ''.join((f.upper() if i in residues_to_highlight else f.lower()) for i,f in enumerate(fasta_one_letter)))

def vmag(x):
    assert x.shape[-1] == 3
    return np.sqrt(x[...,0]**2+x[...,1]**2+x[...,2]**2)

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


def write_AFM(parser, fasta, AFM_table, time_initial, time_step):
    fields = [ln.split() for ln in open(AFM_table, 'U')]
    header = 'residue spring_const tip_pos_x tip_pos_y tip_pos_z pulling_vel_x pulling_vel_y pulling_vel_z'
    actual_header = [x.lower() for x in fields[0]]
    if actual_header != header.split():
        parser.error('First line of tension table must be "%s" but is "%s"'
                %(header," ".join(actual_header)))
    if not all(len(f)==len(fields[0]) for f in fields):
        parser.error('Invalid format for AFM file')
    fields = fields[1:]
    n_spring = len(fields)

    g = t.create_group(t.root.input.potential, 'AFM')
    g._v_attrs.arguments = np.array(['pos'])

    atom             = np.zeros((n_spring,), dtype='i')
    spring_const     = np.zeros((n_spring,))
    starting_tip_pos = np.zeros((n_spring,3))
    pulling_vel      = np.zeros((n_spring,3))

    for i,f in enumerate(fields):
        res = int(f[0])
        msg = 'AFM energy specified for residue %i (zero is first residue) but there are only %i residues in the FASTA'
        if not (0 <= res < len(fasta)):
            raise ValueError(msg % (res, len(fasta)))
        atom[i]             = int(f[0])*3 + 1  # restrain the CA atom in each residue
        spring_const[i]     = f[1]
        starting_tip_pos[i] = [float(x) for x in (f[2],f[3],f[4])]
        pulling_vel[i]      = [float(x) for x in (f[5],f[6],f[7])]

    create_array(g, 'atom',             obj=atom)
    create_array(g, 'spring_const',     obj=spring_const)
    create_array(g, 'starting_tip_pos', obj=starting_tip_pos)
    create_array(g, 'pulling_vel',      obj=pulling_vel)
    g.pulling_vel._v_attrs.time_initial = time_initial
    g.pulling_vel._v_attrs.time_step    = time_step


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


def write_environment(fasta, environment_library, sc_node_name, pl_node_name):
    with tb.open_file(environment_library) as lib:
        energies    = lib.root.energies[:]
        energies_x_offset = lib.root.energies._v_attrs.offset
        energies_x_inv_dx = lib.root.energies._v_attrs.inv_dx

        restype_order = dict([(str(x),i) for i,x in enumerate(lib.root.restype_order[:])])

        coverage_param = lib.root.coverage_param[:]
        assert coverage_param.shape == (len(restype_order),1,4)
        # params are r0,r_sharpness, dot0, dot_sharpness

    # Place CB
    pgrp = t.create_group(potential, 'placement_fixed_point_vector_only_CB')
    pgrp._v_attrs.arguments = np.array(['affine_alignment'])
    ref_pos = np.zeros((4,3))
    ref_pos[0] = (-1.19280531, -0.83127186,  0.)        # N
    ref_pos[1] = ( 0.,          0.,          0.)        # CA
    ref_pos[2] = ( 1.25222632, -0.87268266,  0.)        # C
    ref_pos[3] = ( 0.,          0.94375626,  1.2068012) # CB
    # FIXME this places the CB in a weird location since I should have
    #  used ref_pos[:3].mean(axis=0,keepdims=1) instead.  I cannot change
    #  this without re-running training.  Thankfully, it is a fairly small
    #  mistake and probably irrelevant with contrastive divergence training.
    ref_pos -= ref_pos.mean(axis=0,keepdims=1)

    placement_data = np.zeros((1,6))
    placement_data[0,0:3] = ref_pos[3]
    placement_data[0,3:6] = (ref_pos[3]-ref_pos[2])/vmag(ref_pos[3]-ref_pos[2])

    create_array(pgrp, 'affine_residue',  np.arange(len(fasta)))
    create_array(pgrp, 'layer_index',     np.zeros(len(fasta),dtype='i'))
    create_array(pgrp, 'placement_data',  placement_data)

    # Bring position and probability together for the side chains
    wgrp = t.create_group(potential, 'weighted_pos')
    wgrp._v_attrs.arguments = np.array([sc_node_name, pl_node_name])
    sc_node = t.get_node(t.root.input.potential, sc_node_name)
    n_sc = sc_node.affine_residue.shape[0]
    create_array(wgrp, 'index_pos',   np.arange(n_sc))
    create_array(wgrp, 'index_weight', np.arange(n_sc))

    # Compute SC coverage of the CB
    cgrp = t.create_group(potential, 'environment_coverage')
    cgrp._v_attrs.arguments = np.array(['placement_fixed_point_vector_only_CB','weighted_pos'])

    # group1 is the source CB
    create_array(cgrp, 'index1', np.arange(len(fasta)))
    create_array(cgrp, 'type1', np.array([restype_order[s] for s in fasta]))  # one type per CB type
    create_array(cgrp, 'id1',    np.arange(len(fasta)))

    # group 2 is the weighted points to interact with
    create_array(cgrp, 'index2', np.arange(n_sc))
    create_array(cgrp, 'type2',  0*np.arange(n_sc))   # for now coverage is very simple, so no types on SC
    create_array(cgrp, 'id2',    sc_node.affine_residue[:])

    create_array(cgrp, 'interaction_param', coverage_param)

    # # Transform coverage to [0,1] scale (1 indicates the most buried)
    # tgrp = t.create_group(potential, 'uniform_transform_environment')
    # tgrp._v_attrs.arguments = np.array(['environment_coverage'])
    # create_array(tgrp, 'bspline_coeff', coverage_transform)
    # tgrp.bspline_coeff._v_attrs.spline_offset = coverage_transform_offset
    # tgrp.bspline_coeff._v_attrs.spline_inv_dx = coverage_transform_inv_dx

    # # Linearly couple the transform to energies
    # egrp = t.create_group(potential, 'linear_coupling_uniform_environment')
    # egrp._v_attrs.arguments = np.array(['uniform_transform_environment'])
    # create_array(egrp, 'couplings', energies)
    # create_array(egrp, 'coupling_types', [restype_order[s] for s in fasta])

    # Couple an energy to the coverage coordinates
    egrp = t.create_group(potential, 'nonlinear_coupling_environment')
    egrp._v_attrs.arguments = np.array(['environment_coverage'])
    create_array(egrp, 'coeff', energies)
    egrp.coeff._v_attrs.spline_offset = energies_x_offset
    egrp.coeff._v_attrs.spline_inv_dx = energies_x_inv_dx
    create_array(egrp, 'coupling_types', [restype_order[s] for s in fasta])


def write_count_hbond(fasta, hbond_energy, coverage_library, loose_hbond, sc_node_name):
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
    create_array(igrp, 'interaction_param', np.array([[
        [(0.5   if loose_hbond else 1.4  ), 1./0.10,
         (3.1   if loose_hbond else 2.5  ), 1./0.125,
         (0.182 if loose_hbond else 0.682), 1./0.05,
         0.,   0.]]]))

    if sc_node_name:  # only create hbond_coverage if there are rotamer side chains
        cgrp = t.create_group(potential, 'hbond_coverage')
        cgrp._v_attrs.arguments = np.array(['protein_hbond',sc_node_name])

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
        sc_node = t.get_node(t.root.input.potential, sc_node_name)
        rseq      = sc_node.beadtype_seq[:]
        sc_resnum = sc_node.affine_residue[:]
        create_array(cgrp, 'index2', np.arange(len(rseq)))
        create_array(cgrp, 'type2',  np.array([bead_num[s] for s in rseq]))
        create_array(cgrp, 'id2',    sc_resnum)

        grp = t.create_group(potential, 'placement_fixed_point_vector_scalar')
        grp._v_attrs.arguments = np.array(['affine_alignment'])
        create_array(grp, 'affine_residue',  np.arange(3*n_res)/3)
        create_array(grp, 'layer_index',     np.arange(3*n_res)%3)
        create_array(grp, 'placement_data',  hydrophobe_placement)

        cgrp = t.create_group(potential, 'hbond_coverage_hydrophobe')
        cgrp._v_attrs.arguments = np.array(['placement_fixed_point_vector_scalar',sc_node_name])

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
        rseq = sc_node.beadtype_seq[:]
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


def read_rama_maps_and_weights(seq, rama_group, mode='mixture', allow_CPR=True):
    assert mode in ['mixture', 'product']
    restype = rama_group._v_attrs.restype
    dirtype = rama_group._v_attrs.dir
    ridx_dict = dict([(x,i) for i,x in enumerate(restype)])
    didx = dict([(x,i) for i,x in enumerate(dirtype)])
    ridx = lambda resname, keep_cpr=True: (ridx_dict[resname] if resname!='CPR' or keep_cpr else ridx_dict['PRO'])

    dimer_pot    = rama_group.dimer_pot[:]
    dimer_weight = rama_group.dimer_weight[:]

    assert len(seq) >= 3   # avoid bugs

    # cis-proline is only CPR when it is the central residue, otherwise just use PRO

    V = lambda r,d,n: dimer_pot   [ridx(r,allow_CPR), didx[d], ridx(n,False)]
    W = lambda r,d,n: dimer_weight[ridx(r,allow_CPR), didx[d], ridx(n,False)]

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


def read_weighted_maps(seq, rama_library_h5, sheet_mixing=None, mode='mixture'):
    with tb.open_file(rama_library_h5) as tr:
        coil_pots, coil_weights = read_rama_maps_and_weights(seq, tr.root.coil, mode=mode)

        if sheet_mixing is None:
            return coil_pots
        else:
            sheet_pots, sheet_weights = read_rama_maps_and_weights(seq, tr.root.sheet, allow_CPR=False)
            return mixture_potential([coil_weights, sheet_weights*np.exp(-sheet_mixing)],
                                     [coil_pots,    sheet_pots])


def write_torus_dbn(seq, torus_dbn_library):
    # FIXME use omega emission to handle CPR code
    with tb.open_file(torus_dbn_library) as data:
        dbn_aa_num = dict((x,i) for i,x in enumerate(data.root.restype_order[:]))
        # basin_param order is log_norm,kappa_phi,mu_phi,kappa_psi,mu_psi,kappa_phi_minus_psi
        basin_param = data.root.basin_param[:]
        aa_basin_energy = data.root.aa_basin_energy[:]
        transition_energy = data.root.transition_energy[:]
    restypes = np.array([dbn_aa_num[s] for s in seq])

    # Old-style parsing, more similar to original TorusDBN format
    # dbn_aa_num = dict((x,i) for i,x in enumerate(data.root.restype_order[:]))
    # log_normalization = data.root.TORUS_LOGNORMCONST[:]
    # kappa = data.root.TORUS_KAPPA[:]
    # mu = data.root.TORUS_MU[:]
    # aa_emission_energy  = -np.log(data.root.AA_EMISSION[:].T)
    # cis_emission_energy = -np.log(data.root.CIS_EMISSION[:])
    # transition_energies = -np.log(data.root.HIDDEN_TRANSITION[:])
    # n_state = transition_energies.shape[0]

    # # Add type to handle cis-proline
    # CPR_prior = aa_emission_energy[dbn_aa_num['PRO']] + cis_emission_energy[:,1]
    # dbn_aa_num['CPR'] = len(dbn_aa_num)
    # aa_emission_energy = np.concatenate((aa_emission_energy,CPR_prior[None,:]),axis=0)

    # basin_param = np.zeros((n_state,6),'f4')
    # basin_param[:,0] = log_normalization.ravel()
    # basin_param[:,1] = kappa[:,0]
    # basin_param[:,2] = mu   [:,0]
    # basin_param[:,3] = kappa[:,1]
    # basin_param[:,4] = mu   [:,1]
    # basin_param[:,5] = kappa[:,2]

    egrp = t.create_group(potential, 'torus_dbn')
    egrp._v_attrs.arguments = np.array(['rama_coord'])

    # since Rama angles are not valid for the first and last angles,
    # don't confuse the HMM by including them
    create_array(egrp, 'id',                    np.arange(1,len(seq)-1))
    create_array(egrp, 'restypes',              restypes[1:-1])
    create_array(egrp, 'prior_offset_energies', aa_basin_energy)
    create_array(egrp, 'basin_param',           basin_param)

    hgrp = t.create_group(potential, 'fixed_hmm')
    hgrp._v_attrs.arguments = np.array(['torus_dbn'])

    create_array(hgrp, 'index', np.arange(egrp.id.shape[0]))
    create_array(hgrp, 'transition_energy', transition_energy)


def write_rama_map_pot(seq, rama_library_h5, sheet_mixing_energy=None, secstr_bias='', mode='mixture'):
    grp = t.create_group(potential, 'rama_map_pot')
    grp._v_attrs.arguments = np.array(['rama_coord'])

    rama_pot = read_weighted_maps(seq, rama_library_h5, sheet_mixing_energy, mode)

    if sheet_mixing_energy is not None:
        # support finite differencing for potential derivative
        eps = 1e-2
        grp._v_attrs.sheet_eps = eps
        create_array(grp, 'more_sheet_rama_pot', read_weighted_maps(seq, rama_library_h5, sheet_mixing_energy+eps))
        create_array(grp, 'less_sheet_rama_pot', read_weighted_maps(seq, rama_library_h5, sheet_mixing_energy-eps))

    if secstr_bias:
        assert len(rama_pot.shape) == 3
        phi = np.linspace(-np.pi,np.pi,rama_pot.shape[1],endpoint=False)[:,None]
        psi = np.linspace(-np.pi,np.pi,rama_pot.shape[2],endpoint=False)[None,:]
        sigmoid_lessthan = lambda a,b: 1./(1.+np.exp(-(b-a)/(10.*deg)))

        helical_basin = sigmoid_lessthan(phi,0.*deg) * sigmoid_lessthan(-100.*deg,psi) * sigmoid_lessthan(psi,50.*deg)
        sheet_basin   = sigmoid_lessthan(phi,0.*deg) * (sigmoid_lessthan(psi,-100.*deg) + sigmoid_lessthan(50.*deg,psi))

        f = (ln.split() for ln in open(secstr_bias))
        assert f.next() == 'residue secstr energy'.split()
        for residue,secstr,energy in f:
            residue = int(residue)
            energy = float(energy)

            if secstr == 'helix':
                rama_pot[residue] += energy * helical_basin
            elif secstr == 'sheet':
                rama_pot[residue] += energy *   sheet_basin
            else:
                raise ValueError('secstr in secstr-bias file must be helix or sheet')

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

def write_CB(fasta):
    # Place CB
    pgrp = t.create_group(potential, 'placement_fixed_point_only_CB')
    pgrp._v_attrs.arguments = np.array(['affine_alignment'])
    ref_pos = np.zeros((4,3))
    ref_pos[0] = (-1.19280531, -0.83127186,  0.)        # N
    ref_pos[1] = ( 0.,          0.,          0.)        # CA
    ref_pos[2] = ( 1.25222632, -0.87268266,  0.)        # C
    ref_pos[3] = ( 0.,          0.94375626,  1.2068012) # CB
    ref_pos -= ref_pos[:3].mean(axis=0,keepdims=1)

    placement_data = np.zeros((1,3))
    placement_data[0,0:3] = ref_pos[3]

    create_array(pgrp, 'affine_residue',  np.arange(len(fasta)))
    create_array(pgrp, 'layer_index',     np.zeros(len(fasta),dtype='i'))
    create_array(pgrp, 'placement_data',  placement_data)


def write_contact_energies(parser, fasta, contact_table):
    fields = [ln.split() for ln in open(contact_table,'U')]
    header_fields = 'residue1 residue2 group energy distance transition_width'.split()
    if [x.lower() for x in fields[0]] != header_fields:
        parser.error('First line of contact energy table must be "%s"'%(" ".join(header_fields)))
    if not all(len(f)==len(header_fields) for f in fields):
        parser.error('Invalid format for contact file')
    fields = fields[1:]
    n_contact = len(fields)

    g = t.create_group(t.root.input.potential, 'contact')
    g._v_attrs.arguments = np.array(['placement_fixed_point_only_CB'])

    id     = np.zeros((n_contact,2), dtype='i')
    group  = np.zeros((n_contact,),  dtype='i')
    group_energy = dict()
    dist   = np.zeros((n_contact,))
    width  = np.zeros((n_contact,))

    for i,f in enumerate(fields):
        id[i] = (int(f[0]), int(f[1]))
        msg = 'Contact energy specified for residue %i (zero is first residue) but there are only %i residues in the FASTA'
        if not (0 <= id[i,0] < len(fasta)): raise ValueError(msg % (id[i,0], len(fasta)))
        if not (0 <= id[i,1] < len(fasta)): raise ValueError(msg % (id[i,1], len(fasta)))

        group[i]= int(f[2])
        energy  = float(f[3])

        if group[i] not in group_energy:
            group_energy[group[i]] = energy
        if energy != group_energy[group[i]]:
            raise ValueError(('All contacts in a group must have the same energy '+
                'but there are multiple energies for group %i (at least %f and %f)')
                % (group[i], group_energy[group[i]], energy))

        dist[i]   = float(f[4])
        width[i]  = float(f[5])  # compact_sigmoid cuts off at distance +/- width

        if width[i] <= 0.: raise ValueError('Cannot have negative contact transition_width')

    # 0-based indexing sometimes trips up users, so give them a quick check
    highlight_residues('residues that participate in any --contact potential in uppercase',
            fasta, id.ravel())
    if np.amax(group_energy.values()) > 0.:
        print ('\nWARNING: Some contact energies are positive (repulsive).\n'+
                 '         Please ignore this warning if you intentionally have repulsive contacts.')

    group_energy_array = np.array([group_energy.get(ng,0.)
        for ng in range(1+np.amax(group_energy.keys()))])

    create_array(g, 'group_energy', obj=group_energy_array)
    create_array(g, 'id',           obj=id)
    create_array(g, 'group_id',     obj=group)
    create_array(g, 'distance',     obj=dist)
    create_array(g, 'width',        obj=width)


def write_rama_coord():
    grp = t.create_group(potential, 'rama_coord')
    grp._v_attrs.arguments = np.array(['pos'])
    n_res = n_atom/3
    N_id = 3*np.arange(n_res)
    id = np.column_stack((N_id-1,N_id,N_id+1,N_id+2,N_id+3))
    id[id>=n_atom] = -1  # last atom is non-existent (as is first)
                         #   and non-existence is indicated by -1
    create_array(grp, 'id', id)


def write_sidechain_radial(fasta, library, excluded_residues, suffix=''):
    g = t.create_group(t.root.input.potential, 'radial'+suffix)
    g._v_attrs.arguments = np.array(['placement_fixed_point_only_CB'])
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


def write_rotamer_placement(fasta, placement_library, dynamic_placement, dynamic_1body, fix_rotamer):
    def compute_chi1_state(angles):
        chi1_state = np.ones(angles.shape, dtype='i')
        chi1_state[(   0.*deg<=angles)&(angles<120.*deg)] = 0
        chi1_state[(-120.*deg<=angles)&(angles<  0.*deg)] = 2
        return chi1_state

    with tb.open_file(placement_library) as data:
        restype_num = dict((aa,i) for i,aa in enumerate(data.root.restype_order[:]))

        if dynamic_placement:
            placement_pos = data.root.rotamer_center[:].transpose((2,0,1,3)) # put layer index first
        else:
            placement_pos = data.root.rotamer_center_fixed[:]

        if dynamic_1body:
            placement_energy = -np.log(data.root.rotamer_prob[:].transpose((2,0,1)))[...,None]
        else:
            placement_energy = data.root.rotamer_prob_fixed[:][...,None]

        start_stop = data.root.rotamer_start_stop_bead[:]
        find_restype =                       data.root.restype_and_chi_and_state[:,0].astype('i')
        find_chi1 =                          data.root.restype_and_chi_and_state[:,1]
        find_chi1_state = compute_chi1_state(data.root.restype_and_chi_and_state[:,1])
        find_chi2 =                          data.root.restype_and_chi_and_state[:,2]
        find_state =                         data.root.restype_and_chi_and_state[:,3].astype('i')

    fix = dict()
    if fix_rotamer:
        fields = [x.split() for x in list(open(fix_rotamer))]

        header = 'residue restype chain resnum chi1 chi2'
        actual_header = [x.lower() for x in fields[0]]
        if actual_header != header.split():
            raise RuntimeError('First line of fix-rotamer table must be "%s" but is "%s" for file %s'
                    %(header," ".join(actual_header),fix_rotamer))

        for residue, restype, chain, resnum, chi1, chi2 in fields[1:]:
            if fasta[int(residue)] != (restype if restype != 'CPR' else 'PRO'):
                raise RuntimeError("fix-rotamer file does not match FASTA"
                    + ", residue %i should be %s but fix-rotamer file has %s"%(
                        int(residue), fasta[int(residue)], restype))
            chi1 = float(chi1)*deg  # convert to radians internally
            chi2 = float(chi2)*deg

            if restype == 'GLY' or restype == 'ALA':
                fix_state = 0
            else:
                if np.isnan(chi1): continue
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
                    if np.isnan(chi2): continue
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
    ref_chi1_state = []

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

    sc_node_name = 'placement%s_point_vector_only' % ('' if dynamic_placement else '_fixed')
    grp = t.create_group(potential, sc_node_name)
    grp._v_attrs.arguments = np.array(['affine_alignment'] + (['rama_coord'] if dynamic_placement else []))
    create_array(grp, 'rama_residue',    rama_residue)
    create_array(grp, 'affine_residue',  affine_residue)
    create_array(grp, 'layer_index',     layer_index)
    create_array(grp, 'placement_data',  placement_pos[...,:6])
    create_array(grp, 'beadtype_seq',    beadtype_seq)
    create_array(grp, 'id_seq',          np.array(id_seq))
    create_array(grp, 'fix_rotamer',     np.array(sorted(fix.items())))
    # create_array(grp, 'ref_chi1_state',  np.array(ref_chi1_state))
    # create_array(grp, 'find_chi1',       find_chi1)

    pl_node_name = 'placement%s_scalar' % ('' if dynamic_1body else '_fixed')
    grp = t.create_group(potential, pl_node_name)
    grp._v_attrs.arguments = np.array(['affine_alignment']+(['rama_coord'] if dynamic_1body else []))
    create_array(grp, 'rama_residue',    rama_residue)
    create_array(grp, 'affine_residue',  affine_residue)
    create_array(grp, 'layer_index',     layer_index)
    create_array(grp, 'placement_data',  placement_energy)

    return sc_node_name, pl_node_name


def write_rotamer(fasta, interaction_library, damping, sc_node_name, pl_node_name):
    g = t.create_group(t.root.input.potential, 'rotamer')
    args = [sc_node_name,pl_node_name]
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

    sc_node = t.get_node(t.root.input.potential, sc_node_name)
    rseq = sc_node.beadtype_seq[:]
    create_array(pg, 'index', np.arange(len(rseq)))
    create_array(pg, 'type',  np.array([bead_num[s] for s in rseq]))
    create_array(pg, 'id',    sc_node.id_seq[:])


def write_membrane_potential(
        fasta_seq, membrane_potential_fpath, membrane_thickness, membrane_exclude_residues, hbond_exclude_residues):

    grp = t.create_group(t.root.input.potential, 'membrane_potential')
    grp._v_attrs.arguments = np.array(['placement_fixed_point_only_CB', 'environment_coverage', 'protein_hbond'])

    with tb.open_file(membrane_potential_fpath) as lib:
        resnames        = lib.root.names[:]
        cb_energy       = lib.root.cb_energy[:]
        cb_z_min        = lib.root.cb_energy._v_attrs.z_min
        cb_z_max        = lib.root.cb_energy._v_attrs.z_max
        thickness       = lib.root.cb_energy._v_attrs.thickness
        uhb_energy      = lib.root.uhb_energy[:]
        uhb_z_min       = lib.root.uhb_energy._v_attrs.z_min
        uhb_z_max       = lib.root.uhb_energy._v_attrs.z_max
        cov_midpoint    = lib.root.cov_midpoint[:]
        cov_sharpness   = lib.root.cov_sharpness[:]
    
    #<----- ----- ----- ----- donor/acceptor res ids ----- ----- ----- ----->#
    # Note: hbond_excluded_residues is the same as in the function write_infer_H_O.
    n_res                = len(fasta_seq)
    donor_residue_ids    = np.array([i for i in range(n_res) if i>0       and i not in hbond_exclude_residues and fasta_seq[i]!='PRO'])
    acceptor_residue_ids = np.array([i for i in range(n_res) if i<n_res-1 and i not in hbond_exclude_residues])

    #<----- ----- ----- ----- make energy splines ----- ----- ----- ----->#
    import scipy.interpolate
    def extrapolated_spline(x0, y0):
        spline = scipy.interpolate.InterpolatedUnivariateSpline(x0,y0)
        def f(x, spline=spline):
            return np.select(
                    [(x<x0[0]),              (x>x0[-1]),              np.ones_like(x,dtype='bool')],
                    [np.zeros_like(x)+y0[0], np.zeros_like(x)+y0[-1], spline(x)])
        return f

    cb_z_lib           = np.linspace(cb_z_min, cb_z_max, cb_energy.shape[-1])
    cb_energy_splines  = [extrapolated_spline(cb_z_lib, ene) for ene in cb_energy]

    uhb_z_lib          = np.linspace(uhb_z_min, uhb_z_max, uhb_energy.shape[-1])
    uhb_energy_splines = [extrapolated_spline(uhb_z_lib, ene) for ene in uhb_energy]

    #<----- ----- ----- ----- make energy splines ----- ----- ----- ----->#
    # This step is necessary in case the supplied membrane thickness is not eaual to the thickness in the membrane potential file.
    default_half_thickness = thickness/2.
    half_thickness         = membrane_thickness/2.
    z_                     = np.linspace(-half_thickness - 15., half_thickness + 15., int((membrane_thickness+30.)/0.25)+1)

    # ensure that the potential is continuous at 0
    # spline(z-(half_thickness-default_half_thickness)) may not equal to spline(z+(half_thickness-default_half_thickness))
    membrane_cb_energies = np.zeros((len(cb_energy_splines), len(z_)))
    for ispl, spline in enumerate(cb_energy_splines):
        if half_thickness < default_half_thickness:
            delta_t = default_half_thickness - half_thickness
            delta_s = spline(delta_t) - spline(-delta_t)
            membrane_cb_energies[ispl] = np.select([(z_ < 0), (z_ >= 0.)],
                                                   [spline(z_-delta_t) + 0.5*delta_s, spline(z_+delta_t) - 0.5*delta_s])
        elif half_thickness > default_half_thickness:
            delta_t = half_thickness - default_half_thickness
            membrane_cb_energies[ispl] = np.select([
                (z_ <  -delta_t),
                (z_ >= -delta_t) & (z_ <= delta_t),
                (z_ >   delta_t)],
                [spline(z_+delta_t), spline(0), spline(z_-delta_t)])
        else:
            membrane_cb_energies[ispl] = spline(z_)

    membrane_uhb_energies = np.zeros((len(uhb_energy_splines), len(z_)))
    for ispl, spline in enumerate(uhb_energy_splines):
        if half_thickness < default_half_thickness:
            delta_t = default_half_thickness - half_thickness
            delta_s = spline(delta_t) - spline(-delta_t)
            membrane_uhb_energies[ispl] = np.select([(z_ < 0), (z_ >= 0.)],
                                                    [spline(z_-delta_t) + 0.5*delta_s, spline(z_+delta_t) - 0.5*delta_s])
        elif half_thickness > default_half_thickness:
            delta_t = half_thickness - default_half_thickness
            membrane_uhb_energies[ispl] = np.select([
                (z_ <  -delta_t),
                (z_ >= -delta_t) & (z_ <= delta_t),
                (z_ >   delta_t)],
                [spline(z_+delta_t), spline(0), spline(z_-delta_t)])
        else:
            membrane_uhb_energies[ispl] = spline(z_)

    #<----- ----- ----- ----- cb energy indices ----- ----- ----- ----->#
    # Note: there's a residue type, NON, in resnames for those excluded from membrane potential.
    # And there's a potential profile in cb_energy for NON, which is all zeros. 
    if set(membrane_exclude_residues).difference(range(len(fasta_seq))) != set():
        raise ValueError('Residue number', set(membrane_exclude_residues).difference(range(len(fasta_seq))), 'not valid')
    highlight_residues('membrane_exclude_residues', fasta_seq, membrane_exclude_residues)

    sequence = list(fasta_seq)
    for num in membrane_exclude_residues:
        sequence[num] = 'NON' 
    sequence = np.array(sequence)

    resname_to_num  = dict([(aa,i) for i,aa in enumerate(resnames)])
    residue_id      = np.array([i for i,aa in enumerate(sequence)])
    cb_energy_index = np.array([resname_to_num[aa] for aa in sequence])

    #<----- ----- ----- ----- write to grp ----- ----- ----- ----->#
    create_array(grp,             'cb_index', residue_id)
    create_array(grp,            'env_index', residue_id)
    create_array(grp,         'residue_type', cb_energy_index)
    create_array(grp,         'cov_midpoint', cov_midpoint)
    create_array(grp,        'cov_sharpness', cov_sharpness)
    create_array(grp,            'cb_energy', membrane_cb_energies)
    create_array(grp,           'uhb_energy', membrane_uhb_energies)
    create_array(grp,    'donor_residue_ids', donor_residue_ids)
    create_array(grp, 'acceptor_residue_ids', acceptor_residue_ids)
    grp. cb_energy._v_attrs.z_min = z_[ 0]
    grp. cb_energy._v_attrs.z_max = z_[-1]
    grp.uhb_energy._v_attrs.z_min = z_[ 0]
    grp.uhb_energy._v_attrs.z_max = z_[-1]


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

def chain_endpts(n_res, chain_first_residue, i):
    n_chains = chain_first_residue.size+1
    if i == 0:
        first_res = 0
        next_first_res = chain_first_residue[i]
    elif i == n_chains-1:
        first_res = chain_first_residue[i-1]
        next_first_res = n_res
    else:
        first_res = chain_first_residue[i-1]
        next_first_res = chain_first_residue[i]

    return first_res, next_first_res


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Prepare input file',
            usage='use "%(prog)s --help" for more information')
    parser.add_argument('--fasta', required=True,
            help='[required] FASTA sequence file')
    parser.add_argument('--output', default='system.h5', required=True,
            help='path to output the created .h5 file (default system.h5)')
    parser.add_argument('--target-structure', default='',
            help='Add target .initial.pkl structure for later analysis.  This information is written under '+
            '/target and is never read by Upside.  The /target group may be useful for later analysis.')
    parser.add_argument('--no-backbone', dest='backbone', default=True, action='store_false',
            help='do not use rigid nonbonded for backbone N, CA, C, and CB')
    parser.add_argument('--rotamer-placement', default=None,
            help='rotameric sidechain library')
    parser.add_argument('--dynamic-rotamer-placement', default=False, action='store_true',
            help='Use dynamic rotamer placement (not recommended)')
    parser.add_argument('--dynamic-rotamer-1body', default=False, action='store_true',
            help='Use dynamic rotamer 1body')
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
    parser.add_argument('--rama-library-combining-rule', default='mixture',
            help='How to combine left and right coil distributions in Rama library '+
            '(mixture or product).  Default is mixture.')
    # parser.add_argument('--torus-dbn-library', default='',
    #         help='TorusDBN Rama probability function')
    parser.add_argument('--rama-sheet-library', default=None,
            help='smooth Rama probability library for sheet structures')
    parser.add_argument('--secstr-bias', default='',
            help='Bias file for secondary structure.  First line of the file must be "residue secstr energy".  '+
            'secstr must be one of "helix" or "sheet".  Bias is implemented by a simple Rama bias, hence coil bias '+
            'is not implemented.')
    parser.add_argument('--rama-sheet-mixing-energy', default=None, type=float,
            help='reference energy for sheets when mixing with coil library.  More negative numbers mean more '+
            'sheet content in the final structure.  Default is no sheet mixing.')
    parser.add_argument('--hbond-energy', default=0., type=float,
            help='energy for forming a protein-protein hydrogen bond.  Default is no HBond energy.')

    parser.add_argument('--hbond-exclude-residues', default=[], type=parse_segments,
            help='Residues to have neither hydrogen bond donors or acceptors')
    parser.add_argument('--chain-break-from-file', default='',
            help='File with indices of chain first residues recorded during initial structure generation to automate --hbond-exclude-residues.')

    parser.add_argument('--loose-hbond-criteria', default=False, action='store_true',
            help='Use far more permissive angles and distances to judge HBonding.  Do not use for simulation. '+
            'This is only useful for static backbone training when crystal or NMR structures have poor '+
            'hbond geometry.')
    parser.add_argument('--z-flat-bottom', default='',
            help='Table of Z-flat-bottom springs.  Each line must contain 4 fields and the first line '+
            'must contain "residue z0 radius spring_constant".  The restraint is applied to the CA atom '+
            'of each residue.')
    parser.add_argument('--tension', default='',
            help='Table of linear tensions.  Each line must contain 4 fields and the first line '+
            'must contain "residue tension_x tension_y tension_z".  The residue will be pulled in the '+
            'direction (tension_x,tension_y,tension_z) by its CA atom.  The magnitude of the tension vector '+
            'sets the force.  Units are kT/Angstrom.')
    
    parser.add_argument('--ask-before-using-AFM', default='',
            help='Table of tip positions and pulling velocitis for mimicing AFM pulling experiment in the constant velocity mode. ' +
            'Each line must contain 8 fields and the first line must contain ' +
            '"residue spring_const tip_pos_x tip_pos_y tip_pos_z pulling_vel_x pulling_vel_y pulling_vel_z". ' +
            'The residue will be pulled in the direction (pulling_vel_x, pulling_vel_y, pulling_vel_z) by its CA atom, ' +
            'which is attached to the tip at (tip_pos_x, tip_pos_y, tip_pos_z). ' +
            'The magnitude of the pulling velocity vector sets the pulling speed. The unit is: angstrom/time_step. ' +
            'The spring_const is in the unit of kT/angstrom^2. At T = 298.15 K, it equals 41.14 pN/angstrom. ' + 
            'Note: consult with the developer before using this AFM function.')
    parser.add_argument('--AFM-time-initial', default=0., type=float,
            help='Time initial for AFM pulling simulation. The default value is 0. ' +
            'WARNING: do not change this value unless the simulation is a continuation of a previous one. ' +
            'To set the time initial, check the /root/output/time_estimate in the output h5 file. ' )
    parser.add_argument('--AFM-time-step', default=0.009, type=float,
            help='Time step for AFM pulling simulation. The default value is 0.009. ' +
            'WARNING: this should be the same as the global time step, which is set to 0.009 by default. Change this value accordingly.')

    parser.add_argument('--initial-structure', default='',
            help='Pickle file for initial structure for the simulation.  ' +
            'If there are not enough structures for the number of replicas ' +
            'requested, structures will be recycled.  If not provided, a ' +
            'freely-jointed chain with good bond lengths and angles but bad dihedrals will be used ' +
            'instead.')

    parser.add_argument('--restraint-group', default=[], action='append', type=parse_segments,
            help='List of residues in the protein.  The residue list should be of a form like ' +
            '--restraint-group 10-13,17,19-21 and that list would specify all the atoms in '+
            'residues 10,11,12,13,17,19,20,21. '+
            'Each atom in the specified residues will be randomly connected to atoms in other residues by ' +
            'springs with equilibrium distance given by the distance of the atoms in the initial structure.  ' +
            'Multiple restraint groups may be specified by giving the --restraint-group flag multiple times '
            'with different residue lists.  The strength of the restraint is given by --restraint-spring-constant')
    parser.add_argument('--apply-restraint-group-to-each-chain', action='store_true',
            help='Use indices of chain first residues recorded during PDB_to_initial_structure to automate'+
            ' --restraint-group for chains. Requires --chain-break-from-file.')

    parser.add_argument('--restraint-spring-constant', default=4., type=float,
            help='Spring constant used to restrain atoms in a restraint group (default 4.) ')
    parser.add_argument('--contact-energies', default='',
            help='Path to text file that defines a contact energy function.  The first line of the file should ' +
            'be a header containing "residue1 residue2 energy distance transition_width", and the remaining '+
            'lines should contain space separated values.  The form of the interaction is approximately '+
            'sigmoidal but the potential is constant outside (distance-transition_width,distance+transition_width).'+
            '  This potential is approximately twice as sharp as a standard sigmoid with the same width as the '+
            'specified transition_width.  The location x_residue is approximately the CB position of the '+
            'residue.')
    parser.add_argument('--environment-potential', default='',
            help='Path to many-body environment potential')
    parser.add_argument('--reference-state-rama', default='',
            help='Do not use this unless you know what you are doing.')

    parser.add_argument('--membrane-thickness', default=None, type=float,
            help='Thickness of the membrane in angstroms for use with --membrane-potential.')
    parser.add_argument('--membrane-potential', default='',
            help='Parameter file (.h5 format) for membrane potential. User must also supply --membrane-thickness.')
    parser.add_argument('--membrane-exclude-residues', default=[], type=parse_segments,
            help='Residues that do not participate in the --membrane-potential (same format as --restraint-group).' +
                 'User must also supply --membrane-potential.')

    parser_grp1 = parser.add_mutually_exclusive_group()
    parser_grp1.add_argument('--cavity-radius', default=0., type=float,
            help='Enclose the whole simulation in a radial cavity centered at the origin to achieve finite concentration '+
            'of protein.  Necessary for multichain simulation (though this mode is unsupported.')
    parser_grp1.add_argument('--debugging-only-heuristic-cavity-radius', default=0., type=float,
        help='Set the cavity radius to this provided scale factor times the max distance between com\'s and atoms of the chains.')
    parser_grp1.add_argument('--cavity-radius-from-config', default='', help='Config file with cavity radius set. Useful for applying'+
            ' the same heuristic cavity of bound complex config to unbound counterpart')

    parser.add_argument('--make-unbound', action='store_true',
            help='Separate chains into different corners of a cavity that you set with one of the cavity options.')

    parser.add_argument('--debugging-only-disable-basic-springs', default=False, action='store_true',
            help='Disable basic springs (like bond distance and angle).  Do not use this.')

    args = parser.parse_args()
    if args.restraint_group and not args.initial_structure:
        parser.error('must specify --initial-structures to use --restraint-group')

    if args.apply_restraint_group_to_each_chain and not args.chain_break_from_file:
        parser.error('--apply-restraint-group-to-each-chain requires --chain-break-from-file')

    if args.make_unbound and not args.chain_break_from_file:
        parser.error('--make-unbound requires --chain-break-from-file')

    fasta_seq_with_cpr = read_fasta(open(args.fasta,'U'))
    fasta_seq = np.array([(x if x != 'CPR' else 'PRO') for x in fasta_seq_with_cpr])  # most potentials don't care about CPR
    require_affine = False
    require_rama = False
    require_backbone_point = False

    global n_atom, t, potential
    n_res = len(fasta_seq)
    n_atom = 3*n_res

    t = tb.open_file(args.output,'w')

    input = t.create_group(t.root, 'input')
    create_array(input, 'sequence', obj=fasta_seq_with_cpr)

    if args.initial_structure:
        init_pos = cPickle.load(open(args.initial_structure))
        assert init_pos.shape == (n_atom, 3, 1)

    if args.target_structure:
        def f():
            # little function closure to protect the namespace from ever seeing the target structure
            target_pos = cPickle.load(open(args.target_structure))
            assert target_pos.shape == (n_atom, 3, 1)
            g_target = t.create_group(t.root, 'target')
            t.create_array(t.root.target, 'pos', obj=target_pos[:,:,0])
        f()


    pos = np.zeros((n_atom, 3, 1), dtype='f4')
    if args.initial_structure:
        pos[:,:,0] = init_pos[...,0]
    else:
        pos[:,:,0] = random_initial_config(len(fasta_seq))
    create_array(input, 'pos', obj=pos)

    potential = t.create_group(input,  'potential')

    if not args.debugging_only_disable_basic_springs:
        write_dist_spring(args)
        write_angle_spring(args)
        write_dihedral_spring(fasta_seq_with_cpr)

    sc_node_name = ''
    if args.rotamer_placement:
        require_rama = True
        require_affine = True
        sc_node_name, pl_node_name = write_rotamer_placement(
                fasta_seq, args.rotamer_placement,
                args.dynamic_rotamer_placement, args.dynamic_rotamer_1body,
                args.fix_rotamer)

    if args.chain_break_from_file:
        try:
            with open(args.chain_break_from_file) as infile:
                chain_dat = list(infile)
            # chain_first_residue = np.loadtxt(args.chain_break_from_file, ndmin=1, dtype='int32')
        except IOError:
            chain_first_residue = np.array([], dtype='int32')
            n_chains = 1
        else:
            if len(chain_dat) > 1:
                has_rl_info = True
            else:
                has_rl_info = False
            chain_first_residue = chain_dat[0].split()
            chain_first_residue = np.array(chain_first_residue, dtype='int32')
            print "chain_first_residue:", chain_first_residue
            n_chains = chain_first_residue.size+1
            if has_rl_info:
                rl_chains = np.array(chain_dat[-1].split(), dtype='int32')
                # rl_chains = [int(i) for i in rl_chains]
                print "rl_chains:", rl_chains

        print
        print "n_chains"
        print n_chains

        if chain_first_residue.size:
            break_grp = t.create_group("/input","chain_break","Indicates that multi-chain simulation and removal of bonded potential terms accross chains requested")
            t.create_array(break_grp, "chain_first_residue", chain_first_residue, "Contains array of chain first residues, apart from residue 0")
            if has_rl_info:
                t.create_array(break_grp, "rl_chains", rl_chains, "Numbers of receptor and ligand chains")    

            required_hbond_exclude_res = [i+j for i in chain_first_residue for j in [-1,0]]
            if args.hbond_exclude_residues:
                args.hbond_exclude_residues = np.unique(np.append(args.hbond_exclude_residues, required_hbond_exclude_res))
            else:
                args.hbond_exclude_residues = np.array(required_hbond_exclude_res)

            print
            print "hbond_exclude_residues"
            print args.hbond_exclude_residues

    if args.hbond_energy:
        write_infer_H_O  (fasta_seq, args.hbond_exclude_residues)
        write_count_hbond(fasta_seq, args.hbond_energy, args.rotamer_interaction, args.loose_hbond_criteria, sc_node_name)

    if args.environment_potential:
        if args.rotamer_placement is None:
            parser.error('--rotamer-placement is required, based on other options.')
        write_environment(fasta_seq, args.environment_potential, sc_node_name, pl_node_name)

    args_group = t.create_group(input, 'args')
    for k,v in sorted(vars(args).items()):
        args_group._v_attrs[k] = v
    args_group._v_attrs['invocation'] = ' '.join(sys.argv[:])

    if args.rama_library:
        require_rama = True
        write_rama_map_pot(fasta_seq_with_cpr, args.rama_library, args.rama_sheet_mixing_energy,
                args.secstr_bias, args.rama_library_combining_rule)
    # elif args.torus_dbn_library:
    #     require_rama = True
    #     write_torus_dbn(fasta_seq_with_cpr, args.torus_dbn_library)
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

    if args.debugging_only_heuristic_cavity_radius:
        if n_chains < 2:
            print>>sys.stderr, 'WARNING: --debugging-only-heuristic-cavity-radius requires at least 2 chains. Skipping setting up cavity'
        else:
            com_list = []
            com_dist_list = []

            for i in xrange(n_chains):
                first_res, next_first_res = chain_endpts(n_res, chain_first_residue, i)
                com_list.append(pos[first_res*3:next_first_res*3,:,0].mean(axis=0))

            # Distance between chain com
            # for i in xrange(n_chains):
            #     print
            #     print "com_list"
            #     print com_list[i]
            #     for j in xrange(n_chains):
            #         if j > i:
            #             com_dist_list.append(vmag(com_list[i]-com_list[j]))

            # args.cavity_radius = 2.*max(com_dist_list)
            # print
            # print "old cavity_radius"
            # print args.cavity_radius
            # com_dist_list = []

            # Distance between chain com and all atoms
            for i in xrange(n_chains):
                for j in xrange(n_atom):
                        com_dist_list.append(vmag(com_list[i]-pos[j,:,0]))

            # Max distance between all atoms

            args.cavity_radius = args.debugging_only_heuristic_cavity_radius*max(com_dist_list)
            print
            print "cavity_radius"
            print args.cavity_radius

    if args.cavity_radius_from_config:
        if n_chains < 2:
            print>>sys.stderr, 'WARNING: --cavity-radius-from-config requires at least 2 chains. Skipping setting up cavity'
        elif args.debugging_only_heuristic_cavity_radius:
            print>>sys.stderr, 'WARNING: Overwriting heuristic cavity with the one from --cavity-radius-from-config'
        else:
            t_cavity = tb.open_file(args.cavity_radius_from_config,'r')
            args.cavity_radius = t_cavity.root.input.potential.cavity_radial.radius[0]
            t_cavity.close()
            print
            print "cavity_radius"
            print args.cavity_radius


    if args.cavity_radius:
        write_cavity_radial(args.cavity_radius)

    if args.make_unbound:
        if n_chains < 2 or n_chains > 8:
            print>>sys.stderr, 'WARNING: --make-unbound requires at least 2 and no more than 8 chains. Skipping separating chains'
        elif not args.cavity_radius:
            print>>sys.stderr, 'WARNING: --make-unbound requires setting a cavity radius. Skipping separating chains'
        else:
            print
            print "making unbound"

            displacement = np.array([[-1.,0.,0.], [1.,0.,0.],
                                     [0.,-1.,0.], [0.,1.,0.],
                                     [0.,0.,-1.], [0.,0.,1.],])
            if not has_rl_info: # separate all chains
                for j in xrange(n_chains):
                    first_res, next_first_res = chain_endpts(n_res, chain_first_residue, j)
                    #com = pos[first_res*3:next_first_res*3,:,0].mean(axis=0)
                    pos[first_res*3:next_first_res*3,:,0] = (pos[first_res*3:next_first_res*3,:,0] +
                            displacement[j]*0.5*args.cavity_radius) #- displacement[j]*com
            else: # keep receptor and ligand chains together
                # move receptor chains
                first_res = chain_endpts(n_res, chain_first_residue, 0)[0]
                next_first_res = chain_endpts(n_res, chain_first_residue, rl_chains[0]-1)[1]
                pick_disp = random.choice([0, 2, 4])
                pos[first_res*3:next_first_res*3,:,0] = pos[first_res*3:next_first_res*3,:,0] + displacement[pick_disp]*0.5*args.cavity_radius

                # move ligand chains
                first_res = chain_endpts(n_res, chain_first_residue, rl_chains[0])[0]
                next_first_res = chain_endpts(n_res, chain_first_residue, n_chains-1)[1]
                pick_disp = random.choice([1, 3, 5])
                pos[first_res*3:next_first_res*3,:,0] = pos[first_res*3:next_first_res*3,:,0] + displacement[pick_disp]*0.5*args.cavity_radius
            t.root.input.pos[:] = pos
            target = pos.copy()

    if args.backbone:
        require_affine = True
        write_backbone_pair(fasta_seq)

    if args.z_flat_bottom:
        write_z_flat_bottom(parser,fasta_seq, args.z_flat_bottom)
    
    if args.tension and args.ask_before_using_AFM:
        print 'Nope, you cannot pull the protein using two modes. Choose one.'
    elif args.tension and not args.ask_before_using_AFM:
        write_tension(parser, fasta_seq, args.tension)
    elif args.ask_before_using_AFM and not args.tension:
        write_AFM(parser, fasta_seq, args.ask_before_using_AFM, args.AFM_time_initial, args.AFM_time_step)

    if args.rotamer_interaction:
        # must be after write_count_hbond if hbond_coverage is used
        write_rotamer(fasta_seq, args.rotamer_interaction, args.rotamer_solve_damping, sc_node_name, pl_node_name)

    if args.sidechain_radial:
        require_backbone_point = True
        write_sidechain_radial(fasta_seq, args.sidechain_radial, args.sidechain_radial_exclude_residues)

    if args.membrane_potential:
        if args.membrane_thickness is None:
            parser.error('--membrane-potential requires --membrane-thickness')
        require_backbone_point = True
        write_membrane_potential(fasta_seq,
                                 args.membrane_potential,
                                 args.membrane_thickness,
                                 args.membrane_exclude_residues, 
                                 args.hbond_exclude_residues)

    if args.contact_energies:
        require_backbone_point = True
        write_contact_energies(parser, fasta_seq, args.contact_energies)

    if require_backbone_point:
        require_affine = True
        write_CB(fasta_seq)

    if require_rama:
        write_rama_coord()

    if require_affine:
        write_affine_alignment(len(fasta_seq))

    if args.apply_restraint_group_to_each_chain and n_chains > 1:
        if has_rl_info:
            # receptor chains
            first_res = chain_endpts(n_res, chain_first_residue, 0)[0]
            next_first_res = chain_endpts(n_res, chain_first_residue, rl_chains[0]-1)[1]
            args.restraint_group.append(np.arange(first_res, next_first_res))

            # ligand chains
            first_res = chain_endpts(n_res, chain_first_residue, rl_chains[0])[0]
            next_first_res = chain_endpts(n_res, chain_first_residue, n_chains-1)[1]
            args.restraint_group.append(np.arange(first_res, next_first_res))
        else:
            for i in xrange(n_chains):
                first_res, next_first_res = chain_endpts(n_res, chain_first_residue, i)
                args.restraint_group.append(np.arange(first_res, next_first_res))

        print
        print "restraint_group"
        print args.restraint_group

    if args.restraint_group:
        print
        print 'Restraint groups (uppercase letters are restrained residues)'
        fasta_one_letter = ''.join(one_letter_aa[x] for x in fasta_seq)
        print
        print "Restraint spring constant: {}".format(args.restraint_spring_constant)

        for i,restrained_residues in enumerate(args.restraint_group):
            assert np.amax(list(restrained_residues)) < len(fasta_seq)
            highlight_residues('group_%i'%i, fasta_seq, restrained_residues)
            make_restraint_group(i,set(restrained_residues),pos[:,:,0], args.restraint_spring_constant)

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
