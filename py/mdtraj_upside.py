# VERY IMPORTANT all distances must be in nanometers for MDTraj
import numpy as np
import mdtraj.core.element as el
import mdtraj as md
import sys

from mdtraj.formats.registry import FormatRegistry
angstrom=0.1  # conversion to nanometer from angstrom

print 'Very Important: All distances are in nanometers for MDTraj'

def vmag(x):
    assert x.shape[-1] == 3
    return np.sqrt(x[...,0]**2 + x[...,1]**2 + x[...,2]**2)

def vhat(x):
    return x / vmag(x)[...,None]

def _output_groups(t):
    i=0
    while 'output_previous_%i'%i in t.root:
        yield t.get_node('/output_previous_%i'%i)
        i += 1
    if 'output' in t.root: 
        yield t.get_node('/output')
        i += 1

def traj_from_upside(seq, time, pos, chain_first_residue=[0]):
    H_bond_length = 0.88
    O_bond_length = 1.24

    n_frame = len(pos)
    n_res = len(seq)
    seq = np.array([('PRO' if x == 'CPR' else x) for x in seq])

    chain_first_residue = set(chain_first_residue).union(set([0,n_res]))
    assert all(x<=n_res for x in chain_first_residue)
    assert 0 in chain_first_residue

    assert pos.shape == (n_frame, 3*n_res, 3)
    assert seq.shape == (n_res,)

    topo = md.Topology()
    # ch = topo.add_chain()
    # res = topo.add_residue((restype if restype!='CPR' else 'PRO'), ch[chain_idx], resSeq=i)

    # Main atoms
    expanded_pos_columns = []
    atoms = []
    last_C_atom = None  # needed for adding bonds
    atom_num = 0

    for nr in range(n_res):
        if nr in chain_first_residue:
            current_chain = topo.add_chain()

        res = topo.add_residue(seq[nr], current_chain, resSeq=nr)
        atoms.append(topo.add_atom('N', el.nitrogen, res, atom_num)); atom_num+=1; N =atoms[-1]
        atoms.append(topo.add_atom('CA',el.carbon,   res, atom_num)); atom_num+=1; CA=atoms[-1]
        atoms.append(topo.add_atom('C', el.carbon,   res, atom_num)); atom_num+=1; C =atoms[-1]

        expanded_pos_columns.append(pos[:,3*nr:3*(nr+1)])
        N_pos  = expanded_pos_columns[-1][:,0]
        CA_pos = expanded_pos_columns[-1][:,1]
        C_pos  = expanded_pos_columns[-1][:,2]

        if nr not in chain_first_residue:
            topo.add_bond(last_C,N)
        topo.add_bond(N ,CA)
        topo.add_bond(CA,C)

        # Add NH
        if nr not in chain_first_residue and seq[nr] != 'PRO':
            atoms.append(topo.add_atom('NH', el.hydrogen, res, atom_num)); atom_num+=1; H =atoms[-1]
            topo.add_bond(N,H)

            last_C_pos = pos[:,3*nr-1]
            H_pos = N_pos - H_bond_length*vhat(vhat(last_C_pos-N_pos) + vhat(CA_pos-N_pos))
            expanded_pos_columns.append(H_pos[:,None].astype('f4'))

        # Add CB
        if seq[nr] != 'GLY':
            atoms.append(topo.add_atom('CB', el.carbon, res, atom_num)); atom_num+=1; CB=atoms[-1]
            topo.add_bond(CA,CB)

            extend_dir = vhat(vhat(CA_pos-N_pos)+vhat(CA_pos-C_pos))
            cross_dir  = np.cross(N_pos-CA_pos, C_pos-CA_pos)
            CB_pos = CA_pos + 0.94375626*extend_dir + 0.5796686718421049*cross_dir
            expanded_pos_columns.append(CB_pos[:,None].astype('f4'))

        # Add O
        if nr+1 not in chain_first_residue:
            atoms.append(topo.add_atom('O', el.oxygen, res, atom_num)); atom_num+=1; O =atoms[-1]
            topo.add_bond(C,O)

            next_N_pos = pos[:,3*nr+3]
            O_pos = C_pos - O_bond_length*vhat(vhat(CA_pos-C_pos) + vhat(next_N_pos-C_pos))
            expanded_pos_columns.append(O_pos[:,None].astype('f4'))

        last_C = C
    xyz = np.concatenate(expanded_pos_columns,axis=1)

    # There is some weird bug related to the indices of the topology object.  Basically, the 
    # indices seem to be messed up by the fact when I didn't add them in residue order.  I will
    # continue to try to avoid issues by making a copy, which fixes numbering issues.
    topo = topo.copy()

    # VERY IMPORTANT all distances must be in nanometers for MDTraj
    return md.Trajectory(xyz=xyz*angstrom, topology=topo, time=time)


@FormatRegistry.register_loader('.up')
def load_upside_traj(fname, stride=1, atom_indices=None, target_pos_only=False):
    assert atom_indices is None
    import tables as tb
    with tb.open_file(fname) as t:
        last_time = 0.
        start_frame = 0
        total_frames_produced = 0
        xyz = []
        time = []
        if target_pos_only:
            xyz.append(t.root.target.pos[:,:,0])
            time.append(np.zeros(1,dtype='f4'))
            last_time = time[-1]
            total_frames_produced = 1
            start_frame=1
        else:
            for g_no, g in enumerate(_output_groups(t)):
                # take into account that the first frame of each pos is the same as the last frame before restart
                # attempt to land on the stride
                sl = slice(start_frame,None,stride)
                xyz.append(g.pos[sl,0])
                time.append(g.time[sl]+last_time)
                last_time = g.time[-1]+last_time
                total_frames_produced += g.pos.shape[0]-(1 if g_no else 0)  # correct for first frame
                start_frame = 1 + stride*(total_frames_produced%stride>0) - total_frames_produced%stride
        xyz = np.concatenate(xyz,axis=0)
        time = np.concatenate(time,axis=0)

        seq = t.root.input.sequence[:]

        # Check for chain breaks in config file
        chain_first_residue = np.array([0], dtype='int32')
        if 'chain_break' in t.root.input:
            chain_first_residue = np.append(chain_first_residue, t.root.input.chain_break.chain_first_residue[:])

    return traj_from_upside(seq, time, xyz, chain_first_residue=chain_first_residue)

def load_upside_data(fname, output_names):
    pass

    # return dict from output_names to numpy array of values

def load_upside_rep(fnames, rep_select, stride=1):
    import tables as tb
    for i, fn in enumerate(fnames):
        if i == 0:
            with tb.open_file(fn) as t:
                last_time = 0.
                start_frame = 0
                total_frames_produced = 0
                xyz = []
                time = []
                for g_no, g in enumerate(_output_groups(t)):
                    # take into account that the first frame of each pos is the same as the last frame before restart
                    # attempt to land on the stride
                    sl = slice(start_frame,None,stride)
                    xyz.append(g.pos[sl,0])
                    time.append(g.time[sl]+last_time)
                    last_time = g.time[-1]+last_time
                    total_frames_produced += g.pos.shape[0]-(1 if g_no else 0)  # correct for first frame
                    start_frame = 1 + stride*(total_frames_produced%stride>0) - total_frames_produced%stride
                xyz = np.concatenate(xyz,axis=0)
                time = np.concatenate(time,axis=0)

                seq = t.root.input.sequence[:]

                # Check for chain breaks in config file
                chain_first_residue = np.array([0], dtype='int32')
                if 'chain_break' in t.root.input:
                    chain_first_residue = np.append(chain_first_residue, t.root.input.chain_break.chain_first_residue[:])
        else:
            with tb.open_file(fn) as t:
                start_frame = 0
                total_frames_produced = 0
                xyz2 = []
                replica_idx = []
                for g_no, g in enumerate(_output_groups(t)):
                    # take into account that the first frame of each pos is the same as the last frame before restart
                    # attempt to land on the stride
                    sl = slice(start_frame,None,stride)
                    xyz2.append(g.pos[sl,0])
                    replica_idx.append(g.replica_index[sl,0])
                    total_frames_produced += g.pos.shape[0]-(1 if g_no else 0)  # correct for first frame
                    start_frame = 1 + stride*(total_frames_produced%stride>0) - total_frames_produced%stride
            xyz2 = np.concatenate(xyz2, axis=0)
            replica_idx = np.concatenate(replica_idx, axis=0)
            replica_idx = (replica_idx == rep_select)
            
            xyz[replica_idx] = xyz2[replica_idx]
        
    return traj_from_upside(seq, time, xyz, chain_first_residue=chain_first_residue)        

def ca_contact_pca(traj, n_pc, cutoff_angstroms=8., variance_scaled=True):
    from sklearn.decomposition import TruncatedSVD
    m =  (10.*md.compute_contacts(traj,scheme='ca')[0]<cutoff_angstroms)
    m = m-m.mean(axis=0)
    trunc_svd = TruncatedSVD(n_pc).fit(m)
    pc = trunc_svd.transform(m)*(trunc_svd.explained_variance_ratio_ if variance_scaled else 1.)
    del m
    return pc


def kmeans_cluster(pc, rmsd, n_clusters):
    import sklearn.cluster
    assert len(pc.shape) == 2
    assert len(pc) == len(rmsd)

    km = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(pc)
    n_labels = 1+np.max(km.labels_)
    assert n_labels == n_clusters
    label_rmsd = np.array([np.mean(rmsd[km.labels_==i]) for i in range(n_labels)])
    label_order = np.argsort(label_rmsd)
    # permute the labels so that they are in order of average RMSD
    labels=np.choose(km.labels_[:],np.argsort(label_order)[:,None])
    # label_rmsd = np.array([np.mean(r[labels==i]) for i in range(n_labels)])
    return labels


def compute_upside_values(config_path, traj, outputs=dict(), named_values=dict()):
    import upside_engine as ue

    # extract N,CA,C positions for measurements
    bb_atoms =  [(a.serial,a.index) for a in traj.topology.atoms if a.name in ('N','CA','C')]
    # Must convert distances to angstroms
    pos = 10.*traj.xyz[:,np.array([index for serial,index in sorted(bb_atoms)])]
    assert pos.shape[1] == traj.n_residues*3

    # dists = np.sqrt(np.sum((pos[:,1:]-pos[:,:-1])**2, axis=-1))
    # print [(arr.mean(), np.std(arr)) for arr in (dists[:,0::3], dists[:,1::3], dists[:,2::3])]

    ret = dict()
    ret['energy'] = []
    for nm in list(outputs)+list(named_values):
        ret[nm] = []

    if len(ret) != 1+len(outputs)+len(named_values):
        raise RuntimeError('Some of the output or value names are repeated')

    engine = ue.Upside(config_path)
    assert engine.n_atom == pos.shape[1]
    for x in pos:
        # must compute energy before any other quantities
        ret['energy'].append(engine.energy(x))

        for nm, node_name in outputs.items():
            ret[nm].append(engine.get_output(node_name))

        for nm, (value_shape, node_name, log_name) in named_values.items():
            ret[nm].append(
                    engine.get_value_by_name(
                        value_shape, node_name, log_name))
    for k,v in ret.items():
        ret[k] = np.stack(v, axis=0)

    return ret

def pick_representative_point(coord, sigma_fraction=0.2):
    from sklearn.neighbors import KernelDensity
    assert len(coord.shape) == 2
    sigma = np.sqrt(np.var(coord,axis=0).mean(axis=0))
    
    # Now we will perform kernel density estimation at each data point.
    # We will use a fairly large bandwidth, because we want the mode of the 
    # gross structure of the probability, not dependent on the tiny but sharp
    # clusters.
    
    kde = KernelDensity(bandwidth=sigma_fraction*sigma,rtol=1e-2)
    density = np.exp(kde.fit(coord).score_samples(coord));
    return np.argmax(density)

def pick_all_representative_points(coord, labels, sigma_fraction=0.1):
    assert len(labels.shape) == 1
    assert coord.shape == (labels.shape[0],coord.shape[1])
    n_label = 1+np.max(labels)
    ret = []
    for i in range(n_label):
        good = labels==i
        all_idx = np.arange(len(coord))[good]
        ret.append(all_idx[pick_representative_point(coord[good])])
    return np.array(ret)

def select(traj, sel_text):
    return traj.atom_slice(traj.topology.select(sel_text))

def replex_demultiplex(list_of_replex_traj, replica_index):
    n_frame = list_of_replex_traj[0].n_frames
    n_atom  = list_of_replex_traj[0].n_atoms




def ca_interfacial_rmsd_angstroms(traj, native, group1, group2, ca_cutoff_angstroms=10., verbose=True):
    native = native[0]  # ensure only a single frame is passed

    res_group1, res_group2 = [np.array(sorted(set([native.topology.atom(i).residue.index for i in g])))
            for g in (group1,group2)]

    contact_pairs = np.array([(i,j) for i in res_group1 for j in res_group2])
    is_contact = (10.*md.compute_contacts(native,scheme='ca', contacts=contact_pairs)[0]<ca_cutoff_angstroms)[0]
    contacts = contact_pairs[is_contact]
            
    interface_residues = sorted(set(contacts[:,0]).union(set(contacts[:,1])))
    if verbose:
        print '%i interface residues (%i,%i)' % (
                len(interface_residues), len(set(contacts[:,0])), len(set(contacts[:,1])))
    interface_atom_indices = np.array([a.index for a in native.topology.atoms
                                               if  a.residue.index in interface_residues])

    return 10.*md.rmsd(traj.atom_slice(interface_atom_indices), native.atom_slice(interface_atom_indices))
    

def ca_rmsd_angstroms(traj, native, cut_tails=False, verbose=True):
    ''' Computes RMSD of the CA atoms in angstroms, rather than the MDTraj default of nanometers.  If 
    cut_tails is True, the secondary structure of the native will be computed first, and all 
    all residues before the first helix or sheet and all residues after the last helix or sheet will
    be excluded from the RMSD.  This is convenient to avoid computing the RMSD of unstructured tails.
    If verbose is True, the default, then the number of excluded residues on each end will be printed for
    diagnostics.  You must pass full structures, not CA structures, to this function.'''
    ca_traj = select(traj, 'name CA')
    native_dssp = md.compute_dssp(native)[0]

    first_residue = 0
    last_residue = len(native_dssp)
    if cut_tails:
        while first_residue<len(native_dssp) and native_dssp[first_residue ] in ('C','NA'): 
            first_residue += 1
        while last_residue>=0                and native_dssp[last_residue-1] in ('C','NA'):
            last_residue  -= 1
        if verbose:
            print ('RMSD excluded %i residues from N-terminus and %i residues from C-terminus, leaving %i residues'%(
                    first_residue, len(native_dssp)-last_residue, last_residue-first_residue))

    # convert RMSD to angstroms
    sel = 'name CA and resid %i to %i'%(first_residue,last_residue-1)
    return 10.*md.rmsd(select(traj, sel), select(native,sel))
