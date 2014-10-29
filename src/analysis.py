import numpy as np
import ctypes as ct
clib = ct.cdll.LoadLibrary('analysis.so')

clib.rdc_prediction_simple.restype  = None
clib.rdc_prediction_simple.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p]

clib.compressed_rdc_prediction.restype  = None
clib.compressed_rdc_prediction.argtypes = [
        ct.c_void_p, ct.c_void_p, ct.c_float, ct.c_int, 
        ct.c_int, ct.c_void_p,
        ct.c_int, ct.c_int, ct.c_void_p]

clib.helicity.restype  = None
clib.helicity.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p]

clib.detailed_helicity.restype  = None
clib.detailed_helicity.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p]


def detailed_helicity(is_proline, traj):
    n_res, = is_proline.shape
    n_frame, _1, _2 = traj.shape
    assert traj.shape == (n_frame, n_res*3, 3)

    is_proline = np.ascontiguousarray(is_proline, np.intc)
    traj       = np.ascontiguousarray(traj,       'f4')

    helicity = np.zeros((n_frame,n_res), dtype='f4')
    clib.detailed_helicity(helicity.ctypes, is_proline.ctypes, n_frame, n_res, traj.ctypes)
    return helicity


def rdc_prediction_simple(traj):
    traj = np.require(traj, dtype='f4', requirements='C')
    n_frame, n_atom, three = traj.shape
    n_res = n_atom/3
    assert three == 3
    assert n_res*3 == n_atom

    result = np.zeros((n_res,), dtype='f4')

    clib.rdc_prediction_simple(result.ctypes, n_frame, n_atom, traj.ctypes)
    return result


def compressed_rdc_prediction(traj, d_height=0.2, n_height=1000, n_rot=100):
    traj = np.require(traj, dtype='f4', requirements='C')
    n_frame, n_atom, three = traj.shape
    n_res = n_atom/3
    assert three == 3
    assert n_res*3 == n_atom

    rdc_table = np.zeros((n_height,n_res,), dtype='f8')
    rdc_count = np.zeros((n_height,), dtype='f8')

    quat = np.random.randn(n_rot,4).astype('f4')
    quat /= np.sqrt(np.sum(quat**2,axis=1))[:,None]

    clib.compressed_rdc_prediction(
            rdc_table.ctypes, rdc_count.ctypes, d_height, n_height,
            n_rot, quat.ctypes,
            n_frame, n_atom, traj.ctypes)

    width = d_height * (np.arange(n_height) + 0.5)

    def rdc_value(h):
        weights = h - width_vals
        weights[weights<0.] = 0.
        return np.dot(weights,rdc_table) / np.dot(weights,rdc_count)
       
    return rdc_table, rdc_count, width, rdc_value


def helicity(traj):
    traj = np.require(traj, dtype='f4', requirements='C')
    n_frame, n_atom, three = traj.shape
    n_res = n_atom/3
    assert three == 3
    assert n_res*3 == n_atom

    result = np.zeros((traj.shape[0],n_res,), dtype='i1')

    clib.helicity(result.ctypes, n_frame, n_res, traj.ctypes)
    return result


def main():
    import tables
    import sys
    import cPickle as cp

    for fn in sys.argv[1:]:
        t = tables.open_file(fn)
        if 'output' not in t.root:
            t.close()
            continue
        good_frames = np.isfinite(t.root.output.kinetic[:])
        first_bad_frame = len(good_frames) if np.all(good_frames) else np.nonzero(np.logical_not(good_frames))[0].min()
        if not all(good_frames): print first_bad_frame, len(good_frames)

        pos = t.root.output.pos[int(0.25*first_bad_frame):first_bad_frame]


        sequence = t.root.input.sequence[:]
        is_proline = np.array([x=='PRO' for x in sequence], np.intc)
        detail = detailed_helicity(is_proline, pos)
        pred = compressed_rdc_prediction(pos)
        simple_pred = rdc_prediction_simple(pos)
        hel= helicity(pos)
        t.close()

        print fn, 'helicity %.3f %.3f %.2f'%(hel.mean(dtype='f8'), detail[:,:-4].mean(dtype='f8'), 
                hel.mean(dtype='f8')/detail[:,:-4].mean(dtype='f8'))

        f = open(fn+'.dat.pkl','w')
        cp.dump(dict(
            sequence   = sequence,
            n_frames   = pos.shape[0],
            rdc_table  = pred[0],
            rdc_count  = pred[1],
            detailed_helicity = detail,
            width      = pred[2],
            simple_rdc = simple_pred,
            helicity   = hel), f, -1)
        f.close()


if __name__ == '__main__':
    main()
