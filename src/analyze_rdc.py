#!/usr/bin/env python
import numpy as np

def vmag(x):
    assert x.shape[-1] == 3
    return np.sqrt(x[...,0]**2 + x[...,1]**2 + x[...,2]**2)

def vhat(x):
    return x/vmag(x)[...,None]

def infer_hydrogens(C,N,CA):
    assert C.shape == N.shape == CA.shape == (len(N),3)
    H = N - 0.88 * vhat(vhat(CA-N) + vhat(C-N))
    return H


def principal_axes(pos):
    assert pos.shape == (len(pos),3)
    pos = (pos - pos.mean(axis=0)).astype('f8')

    inertial_tensor_diag = np.mean(np.sum(pos**2,axis=-1)) * np.eye(3)
    inertial_tensor_off  = -(pos[:,None,:]*pos[:,:,None]).mean(axis=0)  # outer product
    inertial_tensor      = inertial_tensor_diag + inertial_tensor_off

    # eigenvectors are columns of evecs
    evals, evecs = np.linalg.eigh(inertial_tensor)
    assert np.all(evals>=0.)  # inertial tensor must be non-negative definite

    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:,order]

    return (evals[0],evecs[:,0]), (evals[1],evecs[:,1]), (evals[2],evecs[:,2])


def RDC(pos):
    pos = pos.astype('f8')
    pos = pos - pos.mean(axis=0)

    N  = pos[0::3]
    CA = pos[1::3]
    C  = pos[2::3]
    assert len(N) == len(CA) == len(C)

    # location of first hydrogen is not determined by the algorithms I use
    H = infer_hydrogens(C[:-1], N[1:], CA[1:])
    H_dir = H - N[1:]; H_dir /= np.sqrt(np.sum(H_dir**2,axis=-1))[:,None]

    P2 = lambda cos_theta: 1.5*cos_theta**2 - 0.5
    return tuple([(eigval, P2(np.dot(H_dir,axis))) for eigval,axis in principal_axes(pos)])


def read_field(name, fields):
    return fields[fields.index(name)+1]


def parse_vtf(fname):
    import re
    topology_block = True
    atoms = []
    bonds = []
    curr_block = None

    for ln in open(fname):
        fields = ln.split()
        if not fields: continue

        key = fields[0]
        # if key in ('atom','bond','timestep'): print fields

        if key == 'atom':
            atoms.append(tuple(fields))
        elif key == 'bond':
            bonds.append(tuple(fields))
        elif key == 'timestep':
            assert read_field('timestep',fields) == 'ordered'

            if topology_block:  # just switched from topology block
                topology_block = False

                n_system  = len(set(read_field('segid',atom) for atom in atoms))
                n_residue = len(set(read_field('resid',atom) for atom in atoms))
                sequence  = np.array([read_field('resname',atom) for atom in atoms
                                      if read_field('name',atom) == 'CA' and read_field('segid',atom) == 's0'])
                n_atom = 3*n_residue
                assert len(atoms) == n_atom*n_system

                yield dict(n_system=n_system,n_residue=n_residue,n_atom=n_atom,atoms=atoms,bonds=bonds)
                curr_block = []
            else:
                frame = np.array(curr_block,dtype='f4').reshape((n_system,n_atom,3))
                yield frame
                curr_block = []
        else:
            curr_block.append(tuple(map(float,fields)))

    # finish last block
    frame = np.array(curr_block).reshape((n_system,n_atom,3)).astype('f4')
    yield frame

    
def parse_vtf_quick(fname):
    iter = parse_vtf(fname)
    d = iter.next()
    d['pos'] = np.array(list(iter))
    return d
    
def dihe_value(a1,a2,a3,a4):
    assert a1.shape == a2.shape == a3.shape == a4.shape
    b1 = a2-a1; b1 *= 1./np.sqrt(np.sum(b1**2, axis=-1))[...,None]
    b2 = a3-a2; b2 *= 1./np.sqrt(np.sum(b2**2, axis=-1))[...,None]
    b3 = a4-a3; b3 *= 1./np.sqrt(np.sum(b3**2, axis=-1))[...,None]

    cross = lambda a,b: np.cross(a,b, axisa=-1, axisb=-1, axisc=-1)
    dot   = lambda a,b: np.sum(a*b,axis=-1)

    a = cross(b1,b2)
    b = cross(b2,b3)
    c = cross(a,b)

    return np.arctan2(dot(c,b2), dot(a,b))


def main():
    import sys
    fname, outname = sys.argv[1:]
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    size = world.Get_size()

    import tables

    t = tables.open_file(fname)
    pos_arr = t.root.output.pos
    n_frame, n_atom, three, n_system = pos_arr.shape
    first_frame = n_frame/2
    n_residue = n_atom/3
    assert 3*n_residue == n_atom
    assert three == 3

    my_results = np.zeros((n_residue, n_system))
    my_rg = np.zeros(n_system)
    for ns in range(rank, n_system, size):
        # exclude first half of trajectory as equilibration
        pos = pos_arr[first_frame:, :, :, ns]

        for frame in pos:
            my_results[1:,ns] += RDC(frame.astype('f8'))[2][1]
        my_rg[ns] = np.sqrt(np.var(pos, axis=1, dtype='f8').sum(axis=1).mean(axis=0))

    t.close()
    my_results /= (n_frame - first_frame)
    results = np.zeros_like(my_results)
    rg      = np.zeros_like(my_rg)
    world.Reduce(my_results, results)
    world.Reduce(my_rg,      rg)

    if not rank:
        print fname, np.sqrt(np.mean(rg**2))
        results /= n_system
        import cPickle as cp
        f=open(outname,'w')
        cp.dump(dict(rdc=results,rg=rg),f,-1)
        print results[...,0]
        f.close()

if __name__ == '__main__':
    main()
