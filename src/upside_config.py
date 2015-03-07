#!/usr/bin/env python

import numpy as np
import tables 
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

default_filter = tables.Filters(complib='zlib', complevel=5, fletcher32=True)

base_sc_ref = {
 'ALA': np.array([-0.01648328,  1.50453228,  1.20193768]),
 'ARG': np.array([-0.27385093,  3.43874264,  2.24442499]),
 'ASN': np.array([-0.27119135,  2.28878532,  1.32214314]),
 'ASP': np.array([-0.19836569,  2.23864046,  1.36505725]),
 'CYS': np.array([-0.17532601,  1.92513503,  1.34296652]),
 'GLN': np.array([-0.28652696,  2.84800873,  1.60009894]),
 'GLU': np.array([-0.26377398,  2.80887008,  1.69621717]),
 'GLY': np.array([ -1.56136239e-02,   5.46052464e-01,  -5.67664281e-19]),
 'HIS': np.array([-0.32896151,  2.66635893,  1.42411271]),
 'ILE': np.array([-0.23956042,  2.26489309,  1.49776818]),
 'LEU': np.array([-0.23949426,  2.67123263,  1.3032201 ]),
 'LYS': np.array([-0.26626635,  3.18256448,  1.85836641]),
 'MET': np.array([-0.21000946,  2.79544428,  1.52568726]),
 'PHE': np.array([-0.27214755,  2.83761534,  1.45094383]),
 'PRO': np.array([-1.10993493,  0.89959734,  1.41005877]),
 'SER': np.array([-0.00692474,  1.56683138,  1.475341  ]),
 'THR': np.array([-0.14662723,  1.80061252,  1.42785569]),
 'TRP': np.array([-0.01433503,  3.07506159,  1.56167948]),
 'TYR': np.array([-0.2841611 ,  3.02555746,  1.50123341]),
 'VAL': np.array([-0.02436993,  1.97251406,  1.32782961])}

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

    grp._v_attrs.energy_scale = 4.
    grp._v_attrs.dist_cutoff = 6.
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


def write_count_hbond(fasta, hbond_energy, helix_energy_perturbation, excluded_residues):
    n_res = len(fasta)
    if hbond_energy > 0.:
        print '\n**** WARNING ****  hydrogen bond formation energy set to repulsive value\n'


    # note that proline is not an hbond donor since it has no NH
    excluded_residues = set(excluded_residues)
    donor_residues    = np.array([i for i in range(n_res) if i>0       and i not in excluded_residues and fasta[i]!='PRO'])
    acceptor_residues = np.array([i for i in range(n_res) if i<n_res-1 and i not in excluded_residues])

    print
    print 'hbond, %i donors, %i acceptors in sequence' % (len(donor_residues), len(acceptor_residues))

    H_bond_length = 0.88
    O_bond_length = 1.24

    if helix_energy_perturbation is None:
        don_bonus = np.zeros(len(   donor_residues))
        acc_bonus = np.zeros(len(acceptor_residues))
    else:
        import pandas as pd
        bonus = pd.read_csv(helix_energy_perturbation)
        d = dict(zip(bonus['aa'],zip(bonus['U_donor'],bonus['U_acceptor'])))
        don_bonus = np.array([d[fasta[nr]][0] for nr in    donor_residues])
        acc_bonus = np.array([d[fasta[nr]][1] for nr in acceptor_residues])

    # FIXME yes, this is scandalous
    # I really need to separate the Infer and the HBondEnergy, but I am busy right now

    grp = t.create_group(potential, 'infer_H_O')
    grp._v_attrs.arguments = np.array(['pos'])

    donors    = t.create_group(grp, 'donors')
    acceptors = t.create_group(grp, 'acceptors')

    create_array(donors,    'bond_length', obj=H_bond_length*np.ones(len(   donor_residues)))
    create_array(acceptors, 'bond_length', obj=O_bond_length*np.ones(len(acceptor_residues)))

    create_array(donors,    'id', obj=np.array((-1,0,1))[None,:] + 3*donor_residues   [:,None])
    create_array(acceptors, 'id', obj=np.array(( 1,2,3))[None,:] + 3*acceptor_residues[:,None])


    grp = t.create_group(potential, 'hbond_energy')
    grp._v_attrs.arguments = np.array(['infer_H_O'])
    grp._v_attrs.hbond_energy = hbond_energy

    donors    = t.create_group(grp, 'donors')
    acceptors = t.create_group(grp, 'acceptors')

    create_array(donors,    'residue_id', obj=   donor_residues)
    create_array(acceptors, 'residue_id', obj=acceptor_residues)

    create_array(donors,    'helix_energy_bonus', obj=don_bonus)
    create_array(acceptors, 'helix_energy_bonus', obj=acc_bonus)
    return


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
    create_array(grp, 'bonded_atoms', obj=np.concatenate((bonded_atoms,np.zeros(len(pairs),dtype='bool')),axis=0))


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
    bonded_atoms = np.ones(id.shape[0], dtype='bool')

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

def write_dihedral_spring():
    # this is primarily used for omega bonds
    grp = t.create_group(potential, 'dihedral_spring')
    grp._v_attrs.arguments = np.array(['pos'])
    id = np.arange(1,n_atom-3,3)  # start at CA atom
    id = np.column_stack((id,id+1,id+2,id+3))

    create_array(grp, 'id', obj=id)
    create_array(grp, 'equil_dist',   obj=180*deg*np.ones(id.shape[0]))
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




def exact_minimum_chi_square_fixed_marginal(row_marginal, col_marginal, counts, pseudoprob=10.0):
    N = row_marginal.shape[0]
    N = col_marginal.shape[0]

    assert row_marginal.shape == (N,)
    assert col_marginal.shape == (N,)
    assert counts.shape == (N,N)

    row_marginal = row_marginal / row_marginal .sum()
    col_marginal = col_marginal / col_marginal.sum()

    # add pseudocount as in common in such methods
    # this prevents problems with exact zero counts

    counts = counts + pseudoprob * row_marginal[:,None] * col_marginal[None,:]
    freq = counts / counts.sum()

    import cvxopt.base
    matrix = cvxopt.base.matrix

    quadratic_matrix = matrix(np.diag(1./freq.reshape((N*N,))))
    linear_vector = matrix(-2. * np.ones(N*N))

    def marginal_matrix(n, row_or_col):
        x = np.zeros((N,N))
        if row_or_col == 'row':
            x[n,:] = 1.
        elif row_or_col == 'col':
            x[:,n] = 1.
        else:
            raise ValueError
        return x

    # last constraint is redundant, so it is removed
    constraint_matrix = matrix(np.concatenate((
            [marginal_matrix(i, 'row').reshape((N*N)) for i in range(N)],
            [marginal_matrix(j, 'col').reshape((N*N)) for j in range(N)]))[:-1])

    constraint_values = matrix(np.concatenate((row_marginal, col_marginal))[:-1])

    inequality_matrix  = matrix(-np.eye(N*N))  # this is a maximum value constraint
    inequality_cutoffs = matrix( np.zeros(N*N))

    import cvxopt.solvers
    cvxopt.solvers.options['show_progress'] = False
    result = cvxopt.solvers.qp(
            quadratic_matrix,  linear_vector, 
            inequality_matrix, inequality_cutoffs, 
            constraint_matrix, constraint_values)
    assert result['status'] == 'optimal'
    joint_prob = np.array(result['x']).reshape((N,N))
    # FIXME add non-negativity constraints to joint probabilities
    joint_prob[joint_prob<0.] = 0.
    return joint_prob


def minimum_chi_square_fixed_marginal(row_marginal, col_marginal, counts, pseudoprob=10.0, tune_pseudoprob=False):
    # this method is based on Deming and Stephen (1940) 
    # in this method, p_{ij} = n_{ij}/n * (1 + lambda^r_i + lambda^r_j)
    # this equation has a symmetry lambda^r_i += alpha, lambda^c_j -= alpha, so the linear system
    # will be degenerate

    if tune_pseudoprob:
        # in tuning mode, the pseudoprob is increased until all the probability estimates are positive
        # this gives an automated way to ensure that the result is a valid probability distribution
        prob_estimate = minimum_chi_square_fixed_marginal(row_marginal, col_marginal, counts, pseudoprob)
        while np.amin(prob_estimate) < 0.:
            pseudoprob += 1.
            prob_estimate = minimum_chi_square_fixed_marginal(row_marginal, col_marginal, counts, pseudoprob)
        # print '%.2f'%(pseudoprob/counts.sum())
        return prob_estimate

    N = row_marginal.shape[0]
    N = col_marginal.shape[0]

    assert row_marginal.shape == (N,)
    assert col_marginal.shape == (N,)
    assert counts.shape == (N,N)

    row_marginal = row_marginal   / row_marginal .sum()
    col_marginal = col_marginal / col_marginal.sum()

    # add pseudocount as in common in such methods
    # this prevents problems with exact zero counts

    counts = counts + pseudoprob * row_marginal[:,None] * col_marginal[None,:]
    freq = counts / counts.sum()
    freq_row_marginal = freq.sum(axis=1)
    freq_col_marginal = freq.sum(axis=0)

    LHS = np.zeros((2*N,2*N))
    LHS[:N,:N] =  np.diag(freq_row_marginal);  LHS[:N,N:] = freq; 
    LHS[N:,:N] =  freq.T;                      LHS[N:,N:] = np.diag(freq_col_marginal);

    RHS = np.zeros((2*N,))
    RHS[:N] = row_marginal - freq_row_marginal
    RHS[N:] = col_marginal - freq_col_marginal

    # The matrix is small, so we can just explicitly invert.  Due to the
    # degeneracy, we will use the pseuodoinverse instead of a regular inverse.

    lambd = np.dot(np.linalg.pinv(LHS), RHS)
    prob = freq * (1. + lambd[:N][:,None] + lambd[N:][None,:])

    # now we have a good estimate for the frequencies, but not perfect, because
    # some of the frequencies could be negative
    return prob


def make_trans_matrices(seq, monomer_basin_prob, dimer_counts):
    N = len(seq)
    trans_matrices = np.zeros((N-1, 5,5))
    prob_matrices  = np.zeros((N-1, 5,5))
    count_matrices = np.zeros((N-1, 5,5))
    assert monomer_basin_prob.shape == (N,5)
    # normalize basin probabilities
    monomer_basin_prob = monomer_basin_prob / monomer_basin_prob.sum(axis=1)[:,None]

    for i in range(N-1):
        count = dimer_counts[(seq[i],seq[i+1])]
        prob = exact_minimum_chi_square_fixed_marginal(
                monomer_basin_prob[i], 
                monomer_basin_prob[i+1], 
                count, pseudoprob = 0.1)   

        # transition matrix is correlation after factoring out independent component
        trans_matrices[i] = prob / (monomer_basin_prob[i][:,None] * monomer_basin_prob[i+1][None,:])
        prob_matrices[i]  = prob
        count_matrices[i] = count

    return trans_matrices, prob_matrices, count_matrices


def populate_rama_maps(seq, rama_library_h5):
    rama_maps = np.zeros((len(seq), 72,72))
    t=tables.open_file(rama_library_h5)
    rama = t.root.rama[:]
    restype = t.root.rama._v_attrs.restype
    dirtype = t.root.rama._v_attrs.dir
    ridx = dict([(x,i) for i,x in enumerate(restype)])
    didx = dict([(x,i) for i,x in enumerate(dirtype)])
    rama_maps[0] = rama[ridx[seq[0]], didx['right'], ridx[seq[1]]]
        
    f = lambda r,d,n: rama[ridx[r], didx[d], ridx[n]]
    for i,l,c,r in zip(range(1,len(seq)-1), seq[:-2], seq[1:-1], seq[2:]):
        rama_maps[i] = f(c,'left',l) + f(c,'right',r) - f(c,'right','ALL')
        
    rama_maps[len(seq)-1] = f(seq[len(seq)-1], 'left', seq[len(seq)-2])
    rama_maps -= -np.log(np.exp(-1.0*rama_maps).sum(axis=-1).sum(axis=-1))[...,None,None]
    t.close()

    return dict(rama_maps = rama_maps, phi=np.arange(-180,180,5), psi=np.arange(-180,180,5))


def write_rama_map_pot(seq, rama_library_h5):
    grp = t.create_group(potential, 'rama_map_pot')
    grp._v_attrs.arguments = np.array(['rama_coord'])

    rama_pot = populate_rama_maps(seq, rama_library_h5)['rama_maps']
    assert rama_pot.shape[0] == len(seq)

    # let's remove the average energy from each Rama map 
    # so that the Rama potential emphasizes its variation

    rama_pot -= (rama_pot*np.exp(-rama_pot)).sum(axis=-1).sum(axis=-1)[:,None,None]

    create_array(grp, 'residue_id',   obj=np.arange(len(seq)))
    create_array(grp, 'rama_map_id',  obj=np.arange(rama_pot.shape[0]))
    create_array(grp, 'rama_pot',     obj=rama_pot)


def write_hmm_pot(sequence, rama_library_h5, dimer_counts=None):
    grp = t.create_group(potential, 'rama_hmm_pot')
    grp._v_attrs.arguments = np.array(['pos'])
    # first ID is previous C
    id = np.arange(2,n_atom-4,3)
    id = np.column_stack((id,id+1,id+2,id+3,id+4))
    n_states = 5
    n_bin=72
    rama_deriv = np.zeros((id.shape[0],n_states,n_bin,n_bin,3))

    d=populate_rama_maps(sequence, rama_library_h5)

    import scipy.interpolate as interp
    phi = np.linspace(-np.pi,np.pi,n_bin,endpoint=False) + 2*np.pi/n_bin/2
    psi = np.linspace(-np.pi,np.pi,n_bin,endpoint=False) + 2*np.pi/n_bin/2

    sharpness = 2. ; # parameters set basin sharpness, 4. is fairly diffuse
    basin_cond_prob = basin_cond_prob_fcns(sharpness, sharpness)  
    assert len(basin_cond_prob) == n_states

    def find_deriv(i):
        rmap = np.tile(d['rama_maps'][i], (3,3))  # tiling helps to ensure periodicity
        h=d['phi']/180.*np.pi
        s=d['psi']/180.*np.pi
        rmap_spline = interp.RectBivariateSpline(
                np.concatenate((h-2*np.pi, h, h+2*np.pi)),
                np.concatenate((s-2*np.pi, s, s+2*np.pi)),
                rmap*1.0)

        eps = 1e-8
        vals = []
        for basin in range(n_states):
            # the new axes are to make the broadcasting rules agree
            lprob_fcn = lambda x,y: rmap_spline(x,y) - np.log(basin_cond_prob[basin](x[:,None],y[None,:]))
            p  = np.exp(-lprob_fcn(phi,psi))
            dx = (lprob_fcn(phi+eps,psi    ) - lprob_fcn(phi-eps,psi    ))/(2.*eps)
            dy = (lprob_fcn(phi    ,psi+eps) - lprob_fcn(phi,    psi-eps))/(2.*eps)
            vals.append(np.concatenate((p[...,None],dx[...,None],dy[...,None]), axis=-1))

        # concatenate over basins
        return np.concatenate([x[None] for x in vals], axis=0)

    for nr in 1+np.arange(rama_deriv.shape[0]):
        rama_deriv[nr-1] = find_deriv(nr).transpose((0,2,1,3))

    # P(phi,b) = P(phi) * P(b|phi)
    # P(b|phi) = f(phi, b) / sum_b' f(phi,b)

    # # normalize the prob at each site (but this will ruin potential calculation)
    # rama_deriv[...,0] /= rama_deriv[...,0].sum(axis=1)[:,None]

    # scale the probabilities for each residue
    # print rama_deriv[...,0].mean(axis=1).mean(axis=1).mean(axis=1)
    rama_deriv[...,0] /= rama_deriv[...,0].mean(axis=1).mean(axis=1).mean(axis=1)[:,None,None,None] + 1e-8;

    idx_to_map = np.arange(id.shape[0])

    if dimer_counts is not None:
        trans_matrices, prob_matrices, count_matrices = make_trans_matrices(
                sequence[1:-1], # exclude termini
                rama_deriv[...,0].sum(axis=-1).sum(axis=-1),
                dimer_counts)
    else:
        # completely uncorrelated transition matrices
        trans_matrices = np.ones((id.shape[0]-1, n_states, n_states))

    grp._v_attrs.sharpness = sharpness
    create_array(grp, 'id',             obj=id)
    create_array(grp, 'rama_deriv',     obj=rama_deriv.astype('f4').transpose((0,1,3,2,4)))
    create_array(grp, 'rama_pot',       obj=d['rama_maps'])
    create_array(grp, 'idx_to_map',     obj=idx_to_map)
    create_array(grp, 'trans_matrices', obj=trans_matrices.astype('f4'))
    if dimer_counts is not None:
        create_array(grp, 'prob_matrices',  obj=prob_matrices)
        create_array(grp, 'count_matrices', obj=count_matrices)

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


def write_basin_correlation_pot(sequence, rama_pot, rama_map_id, dimer_basin_library):
    np.set_printoptions(precision=3, suppress=True)
    assert len(rama_pot.shape) == 3
    grp = t.create_group(potential, 'basin_correlation_pot')
    grp._v_attrs.arguments = np.array(['rama_coord'])

    basin_width = 10.*deg; 
    basin_sharpness = 1./basin_width

    rama_prob = np.exp(-rama_pot)
    rama_prob *= 1./rama_prob.sum(axis=-1,dtype='f8').sum(axis=-1)[:,None,None]

    basins = deg * np.array([
            ((-180.,   0.), (-100.,  50.)),   # alpha_R
            ((-180.,-100.), (  50., 260.)),   # beta
            ((-100.,   0.), (  50., 260.)),   # PPII
            ((   0., 180.), ( -50., 100.)),   # alpha_L
            ((   0., 180.), ( 100., 310.))])  # gamma

    basin_center     = 0.5*(basins[:,:,1]+basins[:,:,0])
    basin_half_width = 0.5*(basins[:,:,1]-basins[:,:,0])

    create_array(grp, 'basin_center',     obj=basin_center)
    create_array(grp, 'basin_half_width', obj=basin_half_width)
    grp.basin_half_width._v_attrs.sharpness = basin_sharpness

    phi_grid = np.linspace(-180.,180., rama_prob.shape[1], endpoint=False)*deg
    psi_grid = np.linspace(-180.,180., rama_prob.shape[2], endpoint=False)*deg
    PSI,PHI = np.meshgrid(phi_grid,psi_grid)
    rama = np.concatenate((PHI[...,None], PSI[...,None]), axis=-1)

    rama_basin_prob = rama_box(rama, basin_center, basin_half_width, basin_sharpness)
    marginal_basin_prob = (rama_basin_prob[None] * rama_prob[...,None]).sum(axis=1).sum(axis=1)


    connection_matrices, prob_matrices, count_matrices = make_trans_matrices(
            sequence[1:-1], # exclude termini
            np.array([marginal_basin_prob[i] for i in rama_map_id[1:-1]]),
            cPickle.load(open(dimer_basin_library)))
    connection_matrices = np.clip(connection_matrices, 1e-3, 5.)  # avoid zeros and wild numbers in energy

    left_res = np.arange(1,len(sequence)-2)
    create_array(grp, 'residue_id',           obj=np.column_stack((left_res, left_res+1)))
    create_array(grp, 'connection_matrices',  obj=connection_matrices.astype('f4'))
    create_array(grp, 'connection_matrix_id', obj=np.arange(grp.connection_matrices.shape[0]))
    create_array(grp, 'prob_matrices',        obj=prob_matrices)
    create_array(grp, 'marginal_basin_prob',  obj=marginal_basin_prob)


def read_fasta(file_obj):
    lines = list(file_obj)
    assert lines[0][0] == '>'
    one_letter_seq = ''.join(x.strip().replace('\r','') for x in lines[1:])
    seq = np.array([three_letter_aa[a] for a in one_letter_seq])
    return seq

def write_dihedral_angle_energies(parser, n_res, dihedral_angle_table):
    fields = [ln.split() for ln in open(dihedral_angle_table,'U')]
    if [x.lower() for x in fields[0]] != 'index angle_type start end width energy'.split():
        parser.error('First line of dihedral angle energy table must be "index angle_type start end width energy"')
    if not all(len(f)==6 for f in fields):
        parser.error('Invalid format for dihedral angle energy file')
    fields = fields[1:]
    n_elem = len(fields)

    grp = t.create_group(t.root.input.potential, 'dihedral_range')
    grp._v_attrs.arguments = np.array(['pos'])

    id          = np.zeros((n_elem,4), dtype = 'i')
    angle_range = np.zeros((n_elem,2))
    scale       = np.zeros((n_elem,))
    energy      = np.zeros((n_elem,))
 
    for i,f in enumerate(fields):
        res_num = int(f[0])
        good_res_num = (f[1]=='phi' and 0 < res_num <= n_res-1) or (f[1]=='psi' and 0 <= res_num < n_res-1) 

        if not good_res_num:
            raise ValueError("Cannot constrain dihedral angles for residue %i"%res_num)
           
        if f[1] == 'phi':
            id[i,0] = 3*res_num - 1    # prev C
 	    id[i,1] = 3*res_num + 0    # N
 	    id[i,2] = 3*res_num + 1    # CA
 	    id[i,3] = 3*res_num + 2    # C
        elif f[1] == 'psi':
            id[i,0] = 3*res_num + 0    # N
            id[i,1] = 3*res_num + 1    # CA
            id[i,2] = 3*res_num + 2    # C
            id[i,3] = 3*res_num + 3    # next N
        else:
 	   raise ValueError('angle type %s not understood'%f[1])
 
 	angle_range[i,0] = float(f[2])*np.pi/180.0
 	angle_range[i,1] = float(f[3])*np.pi/180.0
        if angle_range[i,0] > angle_range[i,1]:
            raise ValueError("Lower dihedral angle bound for residue %i %s is greater than upper bound" % (res_num, f[1]))
        scale[i]       = 1./(float(f[4])*np.pi/180.)
        energy[i]      = float(f[5])
 
    create_array(grp, 'id',          obj=id)
    create_array(grp, 'angle_range', obj=angle_range)
    create_array(grp, 'scale',       obj=scale)
    create_array(grp, 'energy',      obj=energy)
def write_contact_energies(parser, fasta, contact_table):
    fields = [ln.split() for ln in open(contact_table,'U')]
    if [x.lower() for x in fields[0]] != 'residue1 residue2 r0 width energy'.split():
        parser.error('First line of contact energy table must be "residue1 residue2 r0 width energy"')
    if not all(len(f)==5 for f in fields):
        parser.error('Invalid format for contact file')
    fields = fields[1:]
    n_contact = len(fields)

    g = t.create_group(t.root.input.potential, 'contact')
    g._v_attrs.arguments = np.array(['affine_alignment'])
    g._v_attrs.cutoff = 6.   # in units of width

    id         = np.zeros((n_contact,2), dtype='i')
    sc_ref_pos = np.zeros((n_contact,2,3))
    r0         = np.zeros((n_contact,))
    scale      = np.zeros((n_contact,))
    energy     = np.zeros((n_contact,))

    for i,f in enumerate(fields):
        id[i] = (int(f[0]), int(f[1]))
        msg = 'Contact energy specified for residue %i (zero is first residue) but there are only %i residues in the FASTA'
        if not (0 <= id[i,0] < len(fasta)): raise ValueError(msg % (id[i,0], len(fasta)))
        if not (0 <= id[i,1] < len(fasta)): raise ValueError(msg % (id[i,1], len(fasta)))
        sc_ref_pos[i] = (base_sc_ref[fasta[id[i,0]]], base_sc_ref[fasta[id[i,1]]])

        r0[i]     =    float(f[2])
        scale[i]  = 1./float(f[3])
        energy[i] =    float(f[4])

    if energy.max() > 0.:
        print ('\nWARNING: Some contact energies are positive (repulsive).\n'+
                 '         Please ignore this warning if you intendent to have repulsive contacts.')

    create_array(g, 'id',         obj=id)
    create_array(g, 'sc_ref_pos', obj=sc_ref_pos)
    create_array(g, 'r0',         obj=r0)
    create_array(g, 'scale',      obj=scale)
    create_array(g, 'energy',     obj=energy)


def write_sidechain_potential(fasta, library):
    g = t.create_group(t.root.input.potential, 'sidechain')
    g._v_attrs.arguments = np.array(['affine_alignment'])
    t.create_external_link(g, 'sidechain_data', os.path.abspath(library)+':/params')
    create_array(g, 'restype', map(str,fasta))

    # quick check to ensure the external link worked
    assert t.get_node('/input/potential/sidechain/sidechain_data/LYS').corner_location.shape == (3,)

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
    grp = t.create_group(potential, 'backbone_dependent_point')
    grp._v_attrs.arguments = np.array(['rama_coord','affine_alignment'])

    data = tables.open_file(library)

    n_restype = len(aa_num)
    n_bin = data.get_node('/ALA/center').shape[0]-1  # ignore periodic repeat of last bin
    point_map = np.zeros((n_restype,n_bin,n_bin, 3),dtype='f4')  

    for rname,idx in sorted(aa_num.items()):
        point_map[idx] = data.get_node('/%s/center'%rname)[:-1,:-1]
    data.close()

    create_array(grp, 'rama_residue',       np.arange(len(fasta)))
    create_array(grp, 'alignment_residue',  np.arange(len(fasta)))
    create_array(grp, 'restype',            np.array([aa_num[s] for s in fasta]))
    create_array(grp, 'backbone_point_map', point_map)


def write_sidechain_radial(fasta, library, scale_energy, scale_radius, excluded_residues, suffix=''):
    g = t.create_group(t.root.input.potential, 'radial'+suffix)
    g._v_attrs.arguments = np.array(['backbone_dependent_point'])
    for res_num in excluded_residues:
        if not (0<=res_num<len(fasta)):
            raise ValueError('Residue number %i is invalid'%res_num)

    residues = sorted(set(np.arange(len(fasta))).difference(excluded_residues))

    create_array(g, 'restype', obj=map(str,fasta[residues]))
    create_array(g, 'id',      obj=np.array(residues))

    params = tables.open_file(library)
    data = t.create_group(g, 'data')
    data._v_attrs.cutoff = 8.
    create_array(data, 'names',      obj=params.root.params.names[:])
    create_array(data, 'energy',     obj=params.root.params.energy[:] * scale_energy)
    create_array(data, 'r0_squared', obj=params.root.params.r0_squared[:] * scale_radius**2)
    # scale has units 1/A^2
    create_array(data, 'scale',      obj=params.root.params.scale[:]      / scale_radius)
    params.close()


def write_membrane_potential(sequence, potential_library_path, scale, membrane_thickness,excluded_residues, unsatisfiedHB_residues):
    grp = t.create_group(t.root.input.potential, 'membrane_potential')
    grp._v_attrs.arguments = np.array(['backbone_dependent_point'])

    potential_library = tables.open_file(potential_library_path)
    resnames  = potential_library.root.names[:]
    z_energy  = potential_library.root.z_energy[:]
    z_lib_min = potential_library.root.z_energy._v_attrs.z_min
    z_lib_max = potential_library.root.z_energy._v_attrs.z_max
    potential_library.close()

    for res_num in excluded_residues + unsatisfiedHB_residues:
        if not (0<=res_num<len(sequence)):
           raise ValueError('Residue number %i is invalid'%res_num)
    
    residues_included               = sorted(set(np.arange(len(sequence))).difference(excluded_residues))
    unsatisfiedHB_residues_included = sorted(set(unsatisfiedHB_residues)  .difference(excluded_residues))
    sequence_included               = [sequence[i] for i in residues_included]
    for num in unsatisfiedHB_residues_included:
        sequence_included[num]      = 'UHB'   # abbreviation for Unsatisfied HBond
                                              # sequence is comprised of 3-letter codes of aa
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

    resname_to_num = dict([(nm,i) for i,nm in enumerate(resnames)])
    energy_index   = np.array([resname_to_num[aa] for aa in sequence_included])

    create_array(grp, 'residue_id', residues_included)
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Prepare input file',
            usage='use "%(prog)s --help" for more information')
    parser.add_argument('--fasta', required=True,
            help='[required] FASTA sequence file')
    parser.add_argument('--n-system', type=int, default=1, required=False, 
            help='[required] number of systems to prepare')
    parser.add_argument('--output', default='system.h5',
            help='path to output the created .h5 file (default system.h5)')
    # parser.add_argument('--residue-radius', type=float, default=0.,
    #         help='radius of residue for repulsive interaction (1 kT value)')
    parser.add_argument('--backbone', default=False, action='store_true',
            help='use rigid nonbonded for backbone N, CA, C, and CB')
    parser.add_argument('--backbone-dependent-point', default=None,
            help='use backbone-depedent sidechain location library')
    parser.add_argument('--sidechain-radial', default=None,
            help='use sidechain radial potential library')
    parser.add_argument('--sidechain-radial-exclude-residues', default=[], type=parse_segments,
            help='Residues that do not participate in the --sidechain-radial potential (same format as --restraint-group)')
    parser.add_argument('--sidechain-radial-scale-energy', default=1.0, type=float,
            help='scale the sidechain radial energies')
    parser.add_argument('--sidechain-radial-scale-inverse-energy', default=0.0, type=float,
            help='scale the sidechain radial inverse energies (default 0.)')
    parser.add_argument('--sidechain-radial-scale-inverse-radius', default=0.7, type=float,
            help='scale the sidechain radial inverse energies (default 0.7)')
    # parser.add_argument('--sidechain-library', default=None, 
    #         help='use sidechain density potential')
    parser.add_argument('--bond-stiffness', default=48., type=float,
            help='Bond spring constant in units of energy/A^2 (default 48)')
    parser.add_argument('--angle-stiffness', default=175., type=float,
            help='Angle spring constant in units of 1/dot_product (default 175)')
    parser.add_argument('--rama-library', default='',
            help='smooth Rama probability library')
    parser.add_argument('--dimer-basin-library', default='',
            help='dimer basin probability library')
    parser.add_argument('--hbond-energy', default=0., type=float,
            help='energy for forming a hydrogen bond')
    parser.add_argument('--hbond-exclude-residues', default=[], type=parse_segments,
            help='Residues to have neither hydrogen bond donors or acceptors') 
    parser.add_argument('--helix-energy-perturbation', default=None,
            help='hbond energy perturbation file for helices')
    parser.add_argument('--z-flat-bottom', default='', 
            help='Table of Z-flat-bottom springs.  Each line must contain 4 fields and the first line '+
            'must contact "residue z0 radius spring_constant".  The restraint is applied to the CA atom '+
            'of each residue.')
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
    parser.add_argument('--reference-state-rama', default='',
            help='Do not use this unless you know what you are doing.')
    parser.add_argument('--membrane-thickness', default=None, type=float,
            help='Thickness of the membrane in angstroms for use with --membrane-potential.')
    parser.add_argument('--membrane-potential', default='',
            help='Parameter file for membrane potential.  User must also supply --membrane-thickness.')
    parser.add_argument('--membrane-potential-scale',            default=1.0,type=float,
            help='scale the membrane potentials. User must also supply --membrane-potential.')	
    parser.add_argument('--membrane-potential-exclude-residues', default=[], type=parse_segments,
            help='Residues that do not participate in the --membrane-potential(same format as --restraint-group).' +
            'User must also supply --membrane-potential.')	
    parser.add_argument('--membrane-potential-unsatisfied-hbond-residues',default=[], type=parse_segments,
            help='Residues that have unsatisfied hydrogen bond, which will be marked as UHB in --membrane-potential. ' +
            'Normally, this argument is only turned on when user wants to determine the burial orientation of a given membrane protein ' +
	    'and the residues with unsatisfied hbonds are awared of (same format as --restraint-group). ' + 
	    'User must also supply --membrane-potential.')
    parser.add_argument('--cavity-radius', default=0., type=float,
            help='Enclose the whole simulation in a radial cavity centered at the origin to achieve finite concentration '+
            'of protein.  Necessary for multichain simulation (though this mode is unsupported.')


    args = parser.parse_args()
    if args.restraint_group and not (args.initial_structures or args.target_structures):
        parser.error('must specify --initial-structures or --target-structures to use --restraint-group')

    if args.sidechain_radial and not args.backbone_dependent_point:
        parser.error('--sidechain-radial requires --backbone-dependent-point')

    fasta_seq = read_fasta(open(args.fasta,'U'))
    do_alignment = False
    require_rama = False
    require_backbone_point = False

    global n_system, n_atom, t, potential
    n_system = args.n_system
    n_atom = 3*len(fasta_seq)
    
    t = tables.open_file(args.output,'w')
    
    input = t.create_group(t.root, 'input')
    create_array(input, 'sequence', obj=fasta_seq)
    
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

    write_dist_spring(args)
    write_angle_spring(args)
    write_dihedral_spring()
    if args.hbond_energy!=0.: 
        write_count_hbond(fasta_seq, args.hbond_energy, args.helix_energy_perturbation, args.hbond_exclude_residues)

    args_group = t.create_group(input, 'args')
    for k,v in sorted(vars(args).items()):
        args_group._v_attrs[k] = v

    if args.rama_library:
        require_rama = True
        write_rama_map_pot(fasta_seq, args.rama_library)

        if args.dimer_basin_library:
            write_basin_correlation_pot(fasta_seq,
                    potential.rama_map_pot.rama_pot[:], potential.rama_map_pot.rama_map_id[:], 
                    args.dimer_basin_library)

    if args.dihedral_range:
        write_dihedral_angle_energies(parser, len(fasta_seq), args.dihedral_range)

    if args.cavity_radius:
        write_cavity_radial(args.cavity_radius)

    if args.backbone:
        do_alignment = True
        write_backbone_pair(fasta_seq)

    if args.z_flat_bottom:
        write_z_flat_bottom(parser,fasta_seq, args.z_flat_bottom)

    if args.sidechain_radial:
        require_backbone_point = True
        write_sidechain_radial(fasta_seq, args.sidechain_radial, args.sidechain_radial_scale_energy, 1.0,
                args.sidechain_radial_exclude_residues)

        if args.sidechain_radial_scale_energy != 0.:
            write_sidechain_radial(fasta_seq, args.sidechain_radial, 
                    -args.sidechain_radial_scale_energy*args.sidechain_radial_scale_inverse_energy, 
                    args.sidechain_radial_scale_inverse_radius,
                    args.sidechain_radial_exclude_residues, '_inverse')

    if args.membrane_potential:
        if args.membrane_thickness is None:
            parser.error('--membrane-potential requires --membrane-thickness')
        require_backbone_point = True
        write_membrane_potential(fasta_seq, args.membrane_potential, args.membrane_potential_scale, args.membrane_thickness,
	        args.membrane_potential_exclude_residues, args.membrane_potential_unsatisfied_hbond_residues)

    if require_backbone_point:
        if args.backbone_dependent_point is None:
            parser.error('--backbone-dependent-point is required, based on other options.')
        do_alignment = True
        require_rama = True
        write_backbone_dependent_point(fasta_seq, args.backbone_dependent_point)

    if require_rama:
        write_rama_coord()

    if args.contact_energies:
        do_alignment = True
        write_contact_energies(parser, fasta_seq, args.contact_energies)

    if do_alignment:
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

    # hack to fix reference state issues for Rama potential
    if args.reference_state_rama:
        ref_state_pot = -np.log(cPickle.load(open(args.reference_state_rama)))
        ref_state_pot -= ref_state_pot.mean()
        potential.rama_map_pot.rama_pot[:] = potential.rama_map_pot.rama_pot[:] - ref_state_pot
    
    t.close()


if __name__ == '__main__':
    main()

