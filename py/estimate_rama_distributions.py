#!/usr/bin/env python

import os,sys
import tables as tb
import numpy as np
from sklearn.neighbors.kde import KernelDensity
import cPickle as cp

deg = np.pi/180.

def add_image_points(rama, padding):
    assert rama.shape == (rama.shape[0], 2)
    assert np.all(np.abs(rama) < np.pi+1e-3)  # must be in the range (-pi,pi)

    new_rama = np.concatenate([rama + np.array((i*2*np.pi,j*2*np.pi), dtype='f4')
            for i in [-1,0,1] for j in [-1,0,1]], axis=0)

    new_rama = new_rama[np.all(np.abs(new_rama) < np.pi+padding,axis=-1)]

    return new_rama


def rama_density(rama, bw_no_std):
    print rama.shape
    assert rama.shape == (rama.shape[0], 2)
    bins = (-180.+np.arange(72)*5.)*deg

    eval_points = np.column_stack([a.ravel() for a in np.meshgrid(bins,bins)])

    bw = bw_no_std / np.sqrt(np.var(rama,axis=0).sum(axis=-1))  # correct for scikit automatic bandwidth selection

    kde = KernelDensity(kernel='gaussian', bandwidth=bw, rtol=1e-2).fit(rama)
    return np.exp(kde.score_samples(eval_points).reshape((bins.shape[0],bins.shape[0])))


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('input_h5', help='Simulation output')
    parser.add_argument('output_pkl', help='path to put output density pickles')
    parser.add_argument('--system', type=int, required=True, help='system to analyze')
    parser.add_argument('--bandwidth', default=0.2, type=float, help='Bandwidth for kernel density estimate')
    parser.add_argument('--periodic-padding', type=float, default=80., 
            help='periodic padding of angles (in degrees, default 80.)')

    args = parser.parse_args(sys.argv[1:])

    with tb.open_file(args.input_h5) as t:
        rama = t.root.output.rama[:]

    n_frame,n_res,two = rama.shape;  assert two == 2
    densities = np.zeros((n_res,72,72))

    for nr in range(n_res):
        densities[nr] = rama_density(
                add_image_points(rama[:,nr], args.periodic_padding*deg), 
                args.bandwidth)

    with open(args.output_pkl,'w') as f:
        cp.dump(densities,f,-1)

if __name__ == '__main__':
    main()
