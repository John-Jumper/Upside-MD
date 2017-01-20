# VERY IMPORTANT all distances must be in nanometers for MDTraj
import tables as tb
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


@FormatRegistry.register_loader('.up')
def load_upside_traj(fname, stride=1, input_pos_only=False):
    with tb.open_file(fname) as t:
        last_time = 0.
        start_frame = 0
        total_frames_produced = 0
        xyz = []
        time = []
        if input_pos_only:
            xyz.append(t.root.input.pos[:,:,0])
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

        g = t.root.input.potential.infer_H_O.donors
        n_don = g.bond_length.shape[0]
        hb_length = g.bond_length[:]
        hb_atoms  = g.id[:]
        hb_residue= g.residue[:]

        g = t.root.input.potential.infer_H_O.acceptors
        n_acc = g.bond_length.shape[0]
        hb_length = np.concatenate([hb_length ,g.bond_length[:]], axis=0)
        hb_atoms  = np.concatenate([hb_atoms  ,g.id[:]], axis=0)
        hb_residue= np.concatenate([hb_residue,g.residue[:]], axis=0)
        is_donor = np.arange(n_don+n_acc)<n_don

        hb_xyz = [(xyz[:,i[1]] - b*vhat(vhat(xyz[:,i[0]]-xyz[:,i[1]]) + vhat(xyz[:,i[2]]-xyz[:,i[1]])))[:,None]
                  for i,b in zip(hb_atoms, hb_length)]
        xyz = np.concatenate([xyz]+hb_xyz,axis=1)

        topo = md.Topology()

        # Check for chain breaks in config file
        chain_first_residue = np.array([0], dtype='int32')
        if 'chain_break' in t.root.input:
            chain_first_residue = np.append(chain_first_residue, t.root.input.chain_break.chain_first_residue[:])

        ch = [topo.add_chain() for i in xrange(chain_first_residue.size)] 

        residues = []
        atoms = []

        # Main atoms
        chain_idx = 0
        for i,restype in enumerate(seq):
            if chain_first_residue.size > 1 and i in chain_first_residue[1:]: chain_idx += 1
            res = topo.add_residue((restype if restype!='CPR' else 'PRO'), ch[chain_idx], resSeq=i)
            residues.append(res)
            atoms.append(topo.add_atom('N', el.nitrogen, res, 3*i+0))
            atoms.append(topo.add_atom('CA',el.carbon,   res, 3*i+1))
            atoms.append(topo.add_atom('C', el.carbon,   res, 3*i+2))

        # Main chain bonds
        omitted_bonds = set([3*nr-1 for nr in chain_first_residue[1:]])
        for i in xrange(len(atoms)-1):
            if i not in omitted_bonds:
                topo.add_bond(atoms[i],atoms[i+1])  # don't add bond at chain_break

        # Hydrogens and oxygens
        for i in xrange(n_don+n_acc):
            atoms.append(topo.add_atom(
                ('NH' if is_donor[i] else 'O'), 
                (el.hydrogen if is_donor[i] else el.oxygen),
                residues[hb_residue[i]],
                3*len(seq)+i))
            topo.add_bond(atoms[hb_atoms[i,1]],atoms[-1])

        # Now we need to permute the trajectory data for the topology order
        xyz = xyz[:,np.array([a.index for a in topo.atoms])]

        # There is some weird bug related to the indices of the topology object.  Basically, the 
        # indices seem to be messed up by the fact that I did not add them in residue order.
        # Making a copy of the Topology object fixes the numbering issue
        topo = topo.copy()

        # VERY IMPORTANT all distances must be in nanometers for MDTraj
        return md.Trajectory(xyz=xyz*angstrom, topology=topo, time=time)


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

    engine = ue.Upside(pos.shape[1], config_path)
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
