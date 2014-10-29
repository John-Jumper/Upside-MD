import tables as tb
import numpy as np
import sys

default_filter = tb.Filters(complib='zlib', complevel=5, fletcher32=True)

def create_array(tbl, grp, nm, obj=None):
    return tbl.create_carray(grp, nm, obj=obj, filters=default_filter)

def nonbonded_kernel_over_r(r_mag2):
    # V(r) = 1/(1+exp(s*(r**2-d**2)))
    # V(d+width) is approximately V(d) + V'(d)*width
    # this equals 1/2 - (1/2)^2 * 2*s*r * width = (1/2) * (1 - s*(r*width))
    # Based on this, to get a characteristic scale of width,
    #   s should be 1/(wall_radius * width)

    # V'(r) = -2*s*r * z/(1+z)^2 where z = exp(s*(r**2-d**2))
    # V'(r)/r = -2*s*z / (1+z)^2

    wall = 3.2;  # corresponds to vdW *diameter*
    wall_squared = wall*wall;  
    width = 0.15;
    scale_factor = 1./(wall*width);  # ensure character

    # overflow protection prevents NaN
    # z = min(np.exp(scale_factor * (r_mag2-wall_squared)), 1e12);
    z = np.clip(np.exp(scale_factor * (r_mag2-wall_squared)), 0., 1e12);
    w = 1./(1. + z);  # include protection from 0

    deriv_over_r = -2.*scale_factor * z * (w*w);
    return w,deriv_over_r  #  value, derivative_over_r pair


def pairwise_interaction(x_minus_y):
    assert x_minus_y.shape == (3,)
    r_mag2 = x_minus_y[0]**2 + x_minus_y[1]**2 + x_minus_y[2]**2
    germ = np.zeros(4)
    if r_mag2 > 4.*4.: return germ   # mimic cutoff
    val,deriv_over_r = nonbonded_kernel_over_r(r_mag2)
    germ[:3] = deriv_over_r * x_minus_y
    germ[ 3] = val
    return germ


def interaction_field(points, dx):
    assert points.shape == (points.shape[0],3)
    corner = points.min(axis=0) - 4 - 2*dx  # make sure to exceed cutoff
    side_length = points.max(axis=0) + 4 + 2*dx - corner
    nbin = np.ceil(side_length/dx).astype('i')

    field = np.zeros(tuple(nbin)+(4,), dtype='f8')
    for ix in range(nbin[0]):
        for iy in range(nbin[1]):
            for iz in range(nbin[2]):
                # use corner of each bin to define potential
                loc = corner + (np.array((ix,iy,iz), dtype='f8'))*dx
                for p in points:
                    field[ix,iy,iz,:] += pairwise_interaction(loc-p)
    return corner, field


def main():
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

#    t.root.input.force.affine_pairs._f_remove(recursive=True)  # FIXME debug
    t.close()
    

if __name__ == '__main__':
    main()
