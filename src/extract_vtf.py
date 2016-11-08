#!/usr/bin/env python
import tables
import sys
import numpy as np

H_bond=0.88
O_bond=1.24

def vmag(x):# special version for systems
    assert x.shape[-2] == 3
    return np.sqrt(x[...,0,:]**2 + x[...,1,:]**2 + x[...,2,:]**2)

def vhat(x):  # special version for systems
    return x / vmag(x)[...,None,:]

def print_traj_vtf(fname, sequence, traj, bond_id):
    vtf = open(fname,'w')
    n_timestep, n_atom, three, n_system = traj.shape
    assert three == 3

    for ns in xrange(n_system):
        for na in xrange(n_atom):
            print >>vtf, "atom %i name %s resid %i resname %s segid s%i" % (
                    n_atom*ns+na, ['N','CA','C'][na%3], na/3, sequence[na/3], ns)
    
    for a,b in bond_id:
        for ns in xrange(n_system):
            print >>vtf, "bond %i:%i" % (n_atom*ns+a,n_atom*ns+b)

    for frame in traj:
        print >>vtf, "\ntimestep ordered"
        for ns in xrange(n_system):
            for na in xrange(n_atom):
                print >>vtf, "%.3f %.3f %.3f" % (frame[na,0,ns], frame[na,1,ns], frame[na,2,ns])
    print >>vtf
    vtf.close()


def print_augmented_vtf(fname, sequence, traj, stride=1):
    n_timestep, n_atom, three, n_system = traj.shape
    assert three == 3
    assert n_atom%3 ==0
    n_res = n_atom/3

    vtf = open(fname,'w')

    N  = traj[:,0::3].astype('f4')
    CA = traj[:,1::3].astype('f4')
    C  = traj[:,2::3].astype('f4')
    H  = N[:,1: ] - H_bond * vhat(vhat(C [:,:-1]-N[:,1: ]) + vhat(CA[:,1: ]-N[:,1: ]))
    O  = C[:,:-1] - O_bond * vhat(vhat(CA[:,:-1]-C[:,:-1]) + vhat(N [:,1: ]-C[:,:-1]))

    # write structure information
    atom_id = 0
    prev_C = None
    for ns in xrange(n_system):
        for nr in xrange(n_res):
            res = 'resid %i resname %s segid s%i' % (nr, sequence[nr], ns)

            vtf.write('atom %i name N  %s\n' % (atom_id+0, res))
            vtf.write('atom %i name CA %s\n' % (atom_id+1, res))
            vtf.write('atom %i name C  %s\n' % (atom_id+2, res))

            if prev_C is not None: 
                vtf.write('bond %i:%i\n' % (prev_C,atom_id+0)) # prevC->N  bond
            vtf.write('bond %i:%i\n' % (atom_id+0,atom_id+1))  # N->CA bond
            vtf.write('bond %i:%i\n' % (atom_id+1,atom_id+2))  # CA->C bond

            if nr==0 or sequence[nr]=='PRO':
                vtf.write('atom %i name O  %s\n' % (atom_id+3, res))
                vtf.write('bond %i:%i\n' % (atom_id+2, atom_id+3))  # C->O bond
                consumed = 4
            elif nr==n_res-1:
                vtf.write('atom %i name H  %s\n' % (atom_id+3, res))
                vtf.write('bond %i:%i\n' % (atom_id+0, atom_id+3))  # N->H bond
                consumed = 4
            else:
                vtf.write('atom %i name H  %s\n' % (atom_id+3, res))
                vtf.write('atom %i name O  %s\n' % (atom_id+4, res))
                vtf.write('bond %i:%i\n' % (atom_id+0, atom_id+3))  # N->H bond
                vtf.write('bond %i:%i\n' % (atom_id+2, atom_id+4))  # C->O bond
                consumed = 5;

            prev_C = atom_id+2
            atom_id += consumed

    for f in range(len(traj)):
        vtf.write("\ntimestep ordered\n")
        for ns in xrange(n_system):
            for nr in xrange(n_res):
                vtf.write("%.3f %.3f %.3f\n" % (N [f,nr,0,ns], N [f,nr,1,ns], N [f,nr,2,ns]))
                vtf.write("%.3f %.3f %.3f\n" % (CA[f,nr,0,ns], CA[f,nr,1,ns], CA[f,nr,2,ns]))
                vtf.write("%.3f %.3f %.3f\n" % (C [f,nr,0,ns], C [f,nr,1,ns], C [f,nr,2,ns]))
                if nr>0 and sequence[nr]!='PRO': 
                    vtf.write("%.3f %.3f %.3f\n" % (H[f,nr-1,0,ns], H[f,nr-1,1,ns], H[f,nr-1,2,ns]))
                if nr<n_res-1: vtf.write("%.3f %.3f %.3f\n" % (O[f,nr,0,ns], O[f,nr,1,ns], O[f,nr,2,ns]))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_h5', help='Input simulation file')
    parser.add_argument('output_vtf', help='Output trajectory file')
    parser.add_argument('--stride', type=int, default=1, help='Stride for reading file')
    args = parser.parse_args()

    t=tables.open_file(args.input_h5); 
    
    n_res = t.root.input.sequence.shape[0]
    # print_traj_vtf(args.output_vtf, t.root.input.sequence[:], t.root.output.pos[:], 
    #         np.column_stack((np.arange(3*n_res)[:-1], np.arange(3*n_res)[1:])))


    output_paths = []
    i = 0
    while 'output_previous_%i'%i in t.root:
        output_paths.append('/output_previous_%i'%i)
        i+=1
    if 'output' in t.root:  # 'output' is the *last* produced output
        output_paths.append('/output')

    start_frame = 0
    total_frames_produced = 0
    pos = []
    stride = args.stride
    for opath in output_paths:
        g = t.get_node(t.get_node(opath))
        pos.append(g.pos[start_frame::stride].transpose((0,2,3,1)))
        # take into account that the first frame of each pos is the same as the last frame before restart
        # attempt to land on the stride
        total_frames_produced += g.pos.shape[0]-1  # correct for first frame
        start_frame = 1 + stride*(total_frames_produced%stride>0) - total_frames_produced%stride
        print opath, total_frames_produced, 'cumulative frames found'
    pos = np.concatenate(pos, axis=0)
    print pos.shape[0], 'frames for output', pos.shape[0]/30., 'seconds at 30 frames/second video'
    print_augmented_vtf(args.output_vtf, t.root.input.sequence[:], pos)
    t.close()

if __name__ == '__main__':
    main()
