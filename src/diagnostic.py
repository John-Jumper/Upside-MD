#!/usr/bin/env python
import tables
import numpy as np
import os,sys

def vmag(x):
    return np.sqrt(x[...,0]**2 + x[...,1]**2 + x[...,2]**2)

def hot_frames(tbl, outlier_cutoff = 7.):
    ke = tbl.root.output.kinetic_energy[:]
    # compute interquartile range over second half of trajectory
    x = ke[len(ke)/2:].reshape((-1,))
    hot_frames = x > 2.  # simple cutoff

    hot_frames_per_traj = hot_frames[len(hot_frames)/2:].mean(dtype='f8',axis=0)
    return hot_frames_per_traj.mean(), np.median(x)

# def hot_frames(tbl, outlier_cutoff = 7.):
#     ke = tbl.root.output.kinetic_energy[:]
#     # compute interquartile range over second half of trajectory
#     x = ke[len(ke)/2:].reshape((-1,))
#     ke_iqr = np.percentile(x, 75) - np.percentile(x, 25)
#     ke_med = np.median(x)
# 
#     # assume gaussian distribution to convert IQR to standard deviation
#     approx_sigma = ke_iqr / 1.349
# 
#     approx_Z = (ke - ke_med) / approx_sigma
#     hot_frames = approx_Z > outlier_cutoff
# 
#     hot_frames_per_traj = hot_frames[len(hot_frames)/2:].mean(dtype='f8',axis=0)
#     return hot_frames_per_traj.mean()


def robust_distance_autocorr(tbl):
    pos_arr = tbl.root.output.pos
    n_frame, n_atom, three, n_system = pos_arr.shape
    assert three == 3

    median_atom = n_atom/2
    min_chain_disp = 12

    chain_disp_list = [min_chain_disp/2 * 2**i for i in range(n_atom)]
    chain_disp_list = [x for x in chain_disp_list if 0 <= median_atom-x and median_atom+x < n_atom]

    pairs = dict((2*cdisp, (median_atom-cdisp, median_atom+cdisp)) for cdisp in chain_disp_list)

    lags = []; i=0
    while 2**i < n_frame/8:   # leave some room so that the correlation function is reliable
        lags.append(2**i)
        i+=1
    lags = np.array([0] + lags)
    n_lags = lags.shape[0]

    # I want a robust estimate of the autocorrelation time that is insensitive to 
    # outlier frames or even outlier trajectories
    # I was use the insight in Ma and Genton, 1998 that the covariances in the autocorrelation
    # function can be replaced by variances, since
    # cov(A,B) = (1/4) (var(A+B) - var(A-B)) 
    # and the variances can be replaced by a robust scale estimator, like median absolute difference

    all_medians = []
    for ns in range(n_system):
        medians = dict((cd, np.zeros((n_lags,2,n_system))) for cd in pairs)
        traj = pos_arr[n_frame/2:,:,:,ns].astype('f4')

        for cd, (p1,p2) in sorted(pairs.items()):
            dist_seq = vmag(traj[:,p2] -traj[:,p1])

            for il,l in enumerate(lags):
                x = dist_seq[:-l] if l>0 else dist_seq
                y = dist_seq[l:]
                
                x_p_y = x+y
                x_m_y = x-y

                # the following is robust against outlier frames
                medians[cd][il,0] = np.median(np.abs(x_p_y - np.median(x_p_y)))
                medians[cd][il,1] = np.median(np.abs(x_m_y - np.median(x_m_y)))
        all_medians.append(medians)

    # now take further medians in order to be robust against outlier trajectories
    medians = dict()
    for cd in all_medians[0]: 
        medians[cd] = np.median([x[cd] for x in all_medians], axis=0)

    # estimate "covariances"
    autocorr = dict((cd, m[:,1] - m[:,0]) for cd,m in medians.items())

    frame_dt = tbl.root.output.time[1]  - tbl.root.output.time[0]

    # normalize to get "correlations"
    for cd in autocorr: 
        autocorr[cd] = np.column_stack((lags*frame_dt, autocorr[cd][:] / autocorr[cd][0]))

    # fit with a spline to extract correlation time (1/e time for convenience)
    import scipy.interpolate 
    Spline = scipy.interpolate.InterpolatedUnivariateSpline
    autocorr_splines = dict((cd, Spline(ac[:,0], ac[:,1] - 1./np.e, k=3)) for cd,ac in autocorr.items())

    autocorr_times = np.zeros((len(autocorr_splines),2))
    for i,(cd,ac_spl) in enumerate(sorted(autocorr_splines.items())):
        autocorr_times[i,0] = cd
        roots = ac_spl.roots()
        autocorr_times[i,1] = roots[0] if roots else np.NAN

    finite_autocorr_times = autocorr_times[np.isfinite(autocorr_times[:,1])]
    return autocorr_times, np.exp(np.mean(np.log(finite_autocorr_times)))

def process_file((fn, max_fn_length)):
    tbl = tables.open_file(fn)
    try:
        if 'output' in tbl.root:
            hf,temp = hot_frames(tbl)
            s = '%8.0f %.7f %.4f %.2f' % (tbl.root.output.time[-1], hf, temp, robust_distance_autocorr(tbl)[1])
        else:
            s = 'missing'
    except:
        s = 'failed'
    tbl.close()
    return '%-*s %s' % (max_fn_length, fn, s)

def main():
    fnames = sys.argv[1:]
    max_fn_length = max(len(x) for x in fnames)

    import multiprocessing
    pool = multiprocessing.Pool()
    for result in pool.imap(process_file, [(fn,max_fn_length) for fn in fnames]):
        print result

if __name__ == '__main__':
    main()
