#!/usr/bin/env python

import tables
import numpy as np
import argparse

def vmag(x):
    ''' compute the magnitude of vectors in 3 dimensions '''
    assert x.shape[-1] == 3  # last dimension must be length 3
    return np.sqrt(x[...,0]**2 + x[...,1]**2 + x[...,2]**2)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, 
        help='[required] path to .h5 config file containing output trajectory')
parser.add_argument('--output', required=True, 
        help='[required] path for output .txt file')
parser.add_argument('--equil-fraction', default=0.25, type=float,
        help='fraction of frames to discard as equilibration (default 0.25)')
parser.add_argument('--bin-spacing', default=0.25, type=float,
        help='Distance in angstroms between bin centers for histogram of P(r) (default 0.25)')
parser.add_argument('--plot-histogram', default=False, action='store_true', 
        help='Plot the histogram of P(r) (default 0.25)')
args = parser.parse_args()

config = tables.open_file(args.config)  # open .h5 file for reading

# Give a good error message if the user forgot to run the simulation
if 'output' not in config.root._v_children:
    print 'No "output" group in "%s".  Did you forget to run the simulation after making the config?' % args.config

# Read data
traj = config.root.output.pos[:]        # read all trajectory data, shape is (n_frame, n_system, n_atom, 3)
                                        # n_system is probably 1
config.close()
n_frame, n_system, n_atom, _ = traj.shape
print 'Read %i frames and %i atoms' % (n_frame, n_atom)

# Keep only frames after --equil-fraction
equil_frames = int(n_frame * args.equil_fraction)
print 'Discarding %i frames as equilibration' % equil_frames
traj = traj[equil_frames:]   

# Compute distance matrix
# Note that this can consume a lot of memory for large trajectories, but it 
#  would be easy to modify to run in constant memory if needed
distance_matrix = vmag(traj[:,:,np.newaxis,:] - traj[:,:,:,np.newaxis])

# Take only the upper triangle of the distance matrix to avoid double-counting and self-interaction
indices1, indices2 = np.triu_indices(n_atom,1)
distance_matrix = distance_matrix[:,:,indices1,indices2]  # shape is (n_frame_after_equil, n_system, n_pairs)

# Compute the bin edges 
max_distance = distance_matrix.max()
max_bin = int(np.ceil(max_distance / args.bin_spacing))
edges = args.bin_spacing * np.arange(max_bin+1)   # compute bin edges (0*dx, 1*dx, ..., max_bin*dx)

# Compute histogram
flattened_distance_matrix = distance_matrix.reshape((-1,))  # flatten to a single, very-long vector
P_of_r, edges = np.histogram(flattened_distance_matrix, bins=edges, density=True)  # normalize so that integral of P_of_r is 1
bin_centers = 0.5*(edges[1:] + edges[:-1])   # use center of bins for plotting

output_file = open(args.output,'w')
print >>output_file, '%10s %10s' % ('distance','frequency')
for r,p in zip(bin_centers, P_of_r):
    print >>output_file, '%10.3f %10.8f' % (r,p)
output_file.close()


if args.plot_histogram:
    import matplotlib.pyplot as plt
    plt.plot(bin_centers, P_of_r)
    plt.xlabel('distance (A)')
    plt.ylabel('frequency')
    plt.title('P(r) for %s' % args.config)
    plt.show()

Rg = np.sqrt(((traj-traj.mean(axis=-2)[:,:,None,:])**2).sum(axis=-1).mean(axis=-1).mean(axis=0))
print traj.shape
print Rg.shape

np.set_printoptions(precision=4, suppress=True)
print 'Rg = %s A' % (Rg if len(Rg.shape)>1 else np.float32(Rg[0]))
