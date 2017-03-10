import sys
import re
import pandas as pd
from StringIO import StringIO
from glob import glob
import shutil

import tables as tb
import ctypes as ct
import numpy as np
import cPickle as cp
import tempfile

data_dir = '/scratch/midway/jumper/'

from mpi4py import MPI
world = MPI.COMM_WORLD
rank,size = world.rank, world.size

np.set_printoptions(precision=3, linewidth=100, suppress=True)

lib = ct.cdll.LoadLibrary('libsidechain_interaction.so')

lib.interaction_function.restype = None
lib.interaction_function.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_double, ct.c_int, 
        ct.c_double, ct.c_double, ct.c_uint]

def pick_not_none(x):
    res, = [y for y in x if y is not None]
    return res

class StaticScheduler(object):
    def __init__(self, rank, size):
        self.iteration = 0
        self.next_iteration_to_process = None
        return

    def my_work(self):
        do_it = (self.iteration%size) == rank
        self.iteration += 1
        return do_it

scheduler = StaticScheduler(rank,size)

class DynamicScheduler(object):
    def __init__(self, rank, size):
        self.rank == rank
        self.size == size
        self.iteration = 0
        return

    def my_work(self):
        # unless size == 1, reserve 1 process as a scheduler
        if size == 1: return True  

        SCHEDULER_PROCESS_CANNOT_BE_IN_COMMUNICATOR_WITH_WORKERS
        FIXME_SCHEDULE_WILL_HANG_IF_BARRIER_IS_CALLED
        if not rank:
            process = world.recv(MPI.ANY_SOURCE)
            world.send(self.iteration, process)
            self.iteration += 1
            return False  # never do the work on rank 0
        else:
            if self.iteration < self.next_iteration_to_process:
                self.iteration += 1
                return False
            elif self.iteration == self.next_iteration_to_process:
                self.iteration += 1
                self.next_iteration_to_process = None
                return True
            elif self.next_iteration_to_process is None:
                world.send(rank, 0)
                self.next_iteration_to_process = world.recv(0)
                do_it = (self.next_iteration_to_process == self.iteration)
                self.iteration += 1
                return do_it
            elif self.next_iteration_to_process == -1:
                return False
            else:
                raise RuntimeError("Something impossible happened.  Must be an error in the code.")


def interaction_values(dists, energy_scale, sigma1, sigma2, n_pairs=10000, seed=314159):
    dists = np.ascontiguousarray(dists, dtype='double')
    interaction = np.zeros_like(dists)

    lib.interaction_function(interaction.ctypes, interaction.shape[0], dists.ctypes, 
            energy_scale, n_pairs, sigma1, sigma2, seed)
    return interaction

def interaction_function(dists, energy_scale, sigma1, sigma2, n_pairs=10000, seed=314159):
    dists = np.ascontiguousarray(dists, dtype='double')
    interaction = interaction_values(dists,energy_scale, sigma1,sigma2, n_pairs, seed)
    import scipy.interpolate
    return scipy.interpolate.InterpolatedUnivariateSpline(dists, interaction)


def interaction_field(weights,points, dx, f, cutoff): 
    assert points.shape == (points.shape[0],3)
    corner = points.min(axis=0) - cutoff - dx  # make sure to exceed cutoff
    side_length = (points.max(axis=0) + cutoff + dx) - corner
    nbin = np.ceil(side_length/dx).astype('i')

    field = np.zeros(tuple(nbin)+(4,), dtype='f8')
    for ix in range(nbin[0]):
        if not scheduler.my_work(): continue
        for iy in range(nbin[1]):
            for iz in range(nbin[2]):
                # use corner of each bin to define potential
                loc = corner + (np.array((ix,iy,iz), dtype='f8'))*dx
                disp = loc-points
                dist = np.sqrt(disp[:,0]**2+disp[:,1]**2+disp[:,2]**2)
                field[ix,iy,iz,:3] = ((weights * f(dist,1) / dist)[:,None] * disp).sum(axis=0)  # gradient
                field[ix,iy,iz, 3] = ( weights * f(dist)).sum(axis=0)  # value
    return corner, field


def interaction_field_for_gaussian_mixture(weights, centers, sigma_field, sigma_point, 
        dx=None, energy_scale=None, cutoff=None):
    assert dx is not None
    assert energy_scale is not None
    assert cutoff is not None

    dists = np.linspace(0, 10*cutoff, 500)
    f = interaction_function(dists, energy_scale, sigma_field, sigma_point)
    return interaction_field(weights, centers, dx, f, cutoff), f


def test_deriv(f, x, eps=1e-8):
    deriv = np.zeros_like(x)
    for index in zip(*[a.ravel() for a in np.indices(x.shape)]):
        xcopy = x.copy()
        xcopy[index] += eps
        deriv[index] = (f(xcopy)-f(x))/eps
    return deriv



default_filter = tb.Filters(complib='zlib', complevel=5, fletcher32=True)

def create_array(tbl, grp, nm, obj=None):
    return tbl.create_carray(grp, nm, obj=obj, filters=default_filter)


def main():
    # FIXME switch to argparse
    import argparse
    parser = argparse.ArgParse()
    parser.add_argument('--output',       required=True, help="output .h5 file [overwritten]")
    parser.add_argument('--input',        required=True, help="input .pkl file contain gaussian mixture information")
    parser.add_argument('--dx',           required=True, type=float, help="grid spacing for interaction fields")
    parser.add_argument('--energy-scale', required=True, type=float, help="scale for interactions in Mayer-function")
    parser.add_argument('--cutoff',       default=5.,    type=float, help="Mayer-function cutoff")
    args = parser.parse_args()

    t = tb.open_file(args.output, 'w')

    sc_grp = t.create_group(t.root, 'sidechain')
    sc_data = t.create_group(sc_grp, 'sidechain_data')

    # FIXME need to double weight backbone points due to 1/2-ness in interaction
    # FIXME need to fit backbone points

    mixtures = cp.load(open(in_fit_fname))
    sigma_density, = set(x[1][2] for x in mixtures.values()) # all density sigmas must be the same

    import time
    tstart = time.time()

    for rt in sorted(mixtures.keys()):
        (weights_field, centers_field, sigma_field), (weights_density,centers_density,_) = mixtures[k]
        print rt; sys.stdout.flush()
        g = t.create_group(sc_data, 'sc%02i'%i)

        corner,field = interaction_field_for_gaussian_mixture(weights_field, centers_field, sigma_field, 
                sigma_density, dx=args.dx, energy_scale=args.energy_scale, cutoff=cutoff)

        create_array(t, g, 'corner_location', obj=corner)
        create_array(t, g, 'interaction',     obj=field)

        kernels = np.zeros((len(r),4))  # 4th component is the weight
        kernels[:,:3] = centers_density
        kernels[:, 3] = weights_density

        create_array(t, g, 'kernels',  obj=kernels)
        g.interaction._v_attrs.bin_side_length = args.dx
        print time.time() - tstart

    t.close()



def main_test():
    fname,dx = sys.argv[1:]
    dx = float(dx)
    t = tb.open_file(fname, 'a')

    ingrp = t.root.input.force.affine_pairs
    sc_grp = t.create_group(t.root.input.force, 'sidechain')
    sc_data = t.create_group(sc_grp, 'sidechain_data')

    ref_pos = [x[np.isfinite(x[:,0])] for x in ingrp.ref_pos[:]]

    rp_tuples = [tuple(map(tuple,r)) for r in ref_pos]
    rp_tuples_dict = dict([(r,i) for i,r in enumerate(set(rp_tuples))])
    create_array(t, sc_grp, 'restype', ['sc%02i'%rp_tuples_dict[r] for r in rp_tuples])

    dist_cutoff = ingrp._v_attrs.dist_cutoff
    energy_scale = ingrp._v_attrs.energy_scale 
    sc_grp._v_attrs.dist_cutoff = dist_cutoff

    # I should look for unique ref_pos to reduce the number of stored maps
    import time
    tstart = time.time()
    for r,i in rp_tuples_dict.items():
        print i, ; sys.stdout.flush()
        r = np.array(r)
        g = t.create_group(sc_data, 'sc%02i'%i)
        corner, field = interaction_field(r, dx)
        create_array(t, g, 'corner_location', obj=corner)
        create_array(t, g, 'interaction',     obj=field)
        kernels = np.zeros((len(r),4))  # 4th component is the weight
        kernels[:,:3] = r
        kernels[:, 3] = 0.5*energy_scale   # not normal, but makes sure weight is applied 
                                           # also note double counting of interactions
        create_array(t, g, 'kernels',  obj=kernels)
        g.interaction._v_attrs.bin_side_length = dx
        print time.time() - tstart

    # t.root.input.force.affine_pairs._f_remove(recursive=True)
    t.close()
    






def convert_groups(pattern, string, *converters):
    groups = re.search(pattern,string).groups()
    assert len(groups) == len(converters)
    return tuple([f(x) for f,x in zip(converters,groups)])

import theano
import logging
logging.getLogger('theano.gof.compilelock').setLevel(logging.WARNING)
# tmpdir = tempfile.mkdtemp()
# theano.config.compiledir = tmpdir 
import theano.tensor as T

import tables as tb


def theano_init():
    theano.config.floatX = 'float64'
    lweights = T.vector('w')
    centers = T.matrix('c')
    point   = T.matrix('p')
    lsigma   = T.scalar('lsigma')
    
    inv_sigma = 1/T.exp(lsigma)
    weights   = T.exp(lweights)
    
    prefactor    = inv_sigma**3 / np.float32(np.sqrt(2*np.pi))**3
    disp         = point[:,None,:] - centers[None,:,:]
    exponent     = -np.float32(0.5)*inv_sigma**2 * (disp[:,:,0]**2+disp[:,:,1]**2+disp[:,:,2]**2)
    unnorm_probs = T.dot(T.exp(exponent), weights/weights.sum())
    probs        = prefactor * unnorm_probs
    
    llikelihood = -T.log(probs).sum(dtype='float64')
    
    obj_body = llikelihood + (weights.sum()-1)**2
    
    obj_core = theano.function([lweights,centers,point,lsigma], obj_body , 
                               allow_input_downcast=True)
    
    obj_deriv_core = theano.function([lweights,centers,point,lsigma],
                                     theano.grad(obj_body, [lweights,centers,lsigma]), 
                                     allow_input_downcast=True)
    return obj_core, obj_deriv_core


def unpack(x):
    n_gaussians = len(x)/4
    assert x.shape == (4*n_gaussians,)
    lweights = x[:n_gaussians].astype('f4')
    centers = x[n_gaussians:4*n_gaussians].reshape(n_gaussians,3).astype('f4')
    return lweights, centers


def guess_gaussian_parameters(points, n_gaussians):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_gaussians)
    km.fit(points)
    init_centers = np.array(km.cluster_centers_,dtype='f4')
    
    init_weights = np.zeros(n_gaussians,dtype='f4')
    squared_dist = np.zeros(n_gaussians) 
    for p,l in zip(points,km.labels_): 
        init_weights[l] += np.float32(1.)   # count points in each cluster
        squared_dist[l] += np.sum((p-km.cluster_centers_[l])**2)  # find cluster radius
    squared_dist /= (init_weights + 1e-10)
    init_sigma = 4.*np.sqrt(squared_dist.mean())
    print 'init_sigma', init_sigma; sys.stdout.flush()
    init_weights /= init_weights.sum()

    assert init_centers.shape == (n_gaussians,3)
    return init_weights, init_centers, init_sigma


def fit_gaussian_mixture(points, n_gaussians, init_params):
    from scipy.optimize import minimize
    nsplits = len(points)/1000
    nsplits = max(1,nsplits)
    init_weights, init_centers, init_sigma = init_params
    init = np.concatenate(
            [np.log(init_weights), init_centers.reshape((-1,))])
    lsigma = np.log(init_sigma)
    
    def obj(x):
        lweights,centers = unpack(x)
        # process in batches to avoid large memory requirements
        core_val = sum(obj_core(lweights, centers, y, lsigma) 
                       for y in np.array_split(points, nsplits))
        return core_val  # include penalty to ensure sum(weights) is near 1
    
    def obj_deriv(x):
        lweights,centers = unpack(x)
        deriv = np.zeros(len(x))
        for y in np.array_split(points, nsplits):
            d_lweights, d_centers, d_lsigma = obj_deriv_core(lweights, centers, y, lsigma)
            deriv += np.concatenate([d_lweights, d_centers.reshape((-1,))])
        return deriv
    
    return minimize(obj, init, jac=obj_deriv, options=dict(maxiter=1000), method='L-BFGS-B')


def init_guess_map_creation(t, ng):
    my_init_guess_map = dict()
    for rt in sorted(t.root.residues._v_children.keys()):
        if not scheduler.my_work(): continue
        print rt
        x = t.get_node('/residues/%s/pos'%rt)[:,:]
        # select and double up non-diffuse atoms (N,CA,C,O,CB) -- should fix handling of O
        x = np.concatenate((x[:,:5].reshape((-1,3)), x.reshape((-1,3))), axis=0)
        my_init_guess_map[rt] = guess_gaussian_parameters(x[::11], ng)

    init_guess = dict()
    for x in world.allgather(my_init_guess_map):
        init_guess.update(x)
    return init_guess


def init_routines(t, n_gaussian_field, max_density_gaussians):
    init_guess_field = init_guess_map_creation(t, n_gaussian_field)

    my_init_guess_density = []
    for ng in range(max_density_gaussians):
        my_init_guess_density.append(dict())
        for rt in t.root.residues._v_children.keys():
            nd = t.get_node('/residues/%s/pos'%rt)[:,:]
            if nd.shape[1] <= 5: continue  # ALA and GLY
            if not scheduler.my_work(): continue
            if rt == 'ARG': print ng+1
            x = t.get_node('/residues/%s/pos'%rt)[:,5:].reshape((-1,3))
            my_init_guess_density[-1][rt] = guess_gaussian_parameters(x[::11],ng+1)

    # merge results
    init_guess_density = []
    for ng in range(max_density_gaussians):
        if not rank: print ng
        init_guess_density.append(dict())
        for x in world.allgather(my_init_guess_density[ng]):
            init_guess_density[-1].update(x)

    return init_guess_field, init_guess_density


def main_guess():
    t=tb.open_file(data_dir+'residues.h5')
    init_field, init_density = init_routines(t, 100, 40)
    if not rank: 
        f=open(data_dir+'init_guess.pkl','w')
        cp.dump((init_field,init_density), f, -1)
        f.close()
    t.close()


def main_optimize_field():
    t=tb.open_file(data_dir+'residues.h5')
    init_field, init_density = cp.load(open(data_dir+'init_guess.pkl'))

    my_field_opt = dict()
    for rt,(init_weights,init_centers,init_sigma) in sorted(init_field.items()):
        x = t.get_node('/residues/%s/pos'%rt)[:,:]
        # select and double up non-diffuse atoms (N,CA,C,O,CB) -- should fix handling of O
        x = np.concatenate((x[:,:5].reshape((-1,3)), x.reshape((-1,3))), axis=0)
        my_field_opt[rt] = []

        for r in np.linspace(0.05,1.,19):
            if scheduler.my_work():
                if r == 0.1: 
                    print rt, r
                    sys.stdout.flush()
                my_field_opt[rt].append((init_sigma*r,
                        fit_gaussian_mixture(x[5::110], init_weights.shape[0],
                            (init_weights, init_centers, init_sigma*r))))
            else:
                my_field_opt[rt].append(None)
    
    all_maps = world.allgather(my_field_opt)
    field_opt = dict()
    for rt in sorted(init_field):
        sigma_scans = [d[rt] for d in all_maps]
        field_opt[rt] = [pick_not_none(x) for x in zip(*sigma_scans)]

    if not rank: 
        f=open(data_dir+'field_opt.pkl','w')
        cp.dump(field_opt, f, -1)
        f.close()
    t.close()



    # work = [(rt,init_guess[rt][0], init_guess[rt][1], r*init_guess[rt][2])
    #         for r in np.linspace(0.1,2,40) for rt in init_guess]

    # my_results = []
    # for rt, init_weights, init_centers, sigma in work[rank::size]:
    #     import time
    #     tstart = time.clock()
    #     atoms = t.get_node('/residues/%s/pos'%rt)[:,5:].reshape((-1,3))[5::11]
    #     res = fit_gaussian_mixture(atoms, init_weights.shape[0], 
    #             (init_weights, init_centers, sigma))
    #     my_results.append((rt, init_weights, init_centers, sigma, res))
    #     print '%s %.2f %.2f %.2f' % (rt, sigma, res.fun/len(atoms), time.clock()-tstart)
    #     sys.stdout.flush()
    # t.close()

    # my_results_coll = world.gather(my_results)

    # if not rank:
    #     all_results = []
    #     for d in my_results_coll: 
    #         all_results.extend(d)

    #     f=open('/scratch/midway/jumper/sc-optimization.pkl','w')
    #     cp.dump(all_results,f,-1)
    #     f.close()


if __name__ == '__main__':
        obj_core, obj_deriv_core = theano_init()
        main_optimize()
        # main_guess()

