import tables as tb
import numpy as np
import tempfile
import subprocess as sp
import os
import theano
import theano.tensor as T
import cPickle as cp
import concurrent.futures
import ctypes

upside_calc = ctypes.cdll.LoadLibrary('/home/jumper/Dropbox/code/upside/obj/libupside_calculation.so')
upside_path = '../obj'

upside_calc.new_rotamer_construct_and_solve.argtypes = [
        ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_int, ctypes.c_float,
        ctypes.c_void_p]
upside_calc.new_rotamer_construct_and_solve.restype = ctypes.c_void_p

upside_calc.free_energy_and_parameter_deriv.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
upside_calc.free_energy_and_parameter_deriv.restype = ctypes.c_float

upside_calc.delete_rotamer_construct_and_solve.argtypes = [
        ctypes.c_void_p]
upside_calc.delete_rotamer_construct_and_solve.restype = None

mode = 'PosBead'
if mode == 'PosDirBead':
    D = 7
    P_symm = 6
    P_swap = 6

elif mode == 'PosBead':
    D = 4
    P_symm = 6
    P_swap = 0

P = P_symm + P_swap

executor = concurrent.futures.ThreadPoolExecutor(4)


class RotamerObjectHolder(object):
    def __init__(self, rotamer_object):
        self.rotamer_object = rotamer_object

    def energy(self,interactions):
        assert interactions.shape == (20,20,P)
        deriv = np.zeros(interactions.shape, dtype='f4')
        interactions = interactions.astype('f4').copy()
        value = upside_calc.free_energy_and_parameter_deriv(self.rotamer_object, 
                deriv.ctypes.data, interactions.ctypes.data)
        return float(value),deriv.astype('f8')

    def __del__(self):
        upside_calc.delete_rotamer_construct_and_solve(self.rotamer_object)


def imat_unpack_expr(param_expr, n_restype=20):
    n_restype = 20
    upper_tri        = (n_restype*(n_restype+1))/2
    strict_upper_tri = (n_restype*(n_restype-1))/2
    diag             = n_restype

    i = [0]
    def read_param(size):
        ret = param_expr[i[0]:i[0]+size]
        i[0] += size
        return ret

    inner_energy = T.exp(read_param(upper_tri))  # inner_energy is repulsive
    inner_radius = T.exp(read_param(upper_tri))  # positive
    inner_scale  = T.exp(read_param(upper_tri))  # positive

    if mode == 'PosDirBead':
        energy_gauss =       read_param(upper_tri)
        dist_loc     = T.exp(read_param(upper_tri))
        dist_scale   = T.exp(read_param(upper_tri))

        dp1_steric_upper = read_param(strict_upper_tri)
        dp2_steric_upper = read_param(strict_upper_tri)
        dp_steric_diag   = read_param(diag)

        dp1_loc_gauss_upper = read_param(strict_upper_tri)
        dp2_loc_gauss_upper = read_param(strict_upper_tri)
        dp_loc_gauss_diag   = read_param(diag)

        dp1_scale_gauss_upper = T.exp(read_param(strict_upper_tri))  # scales are positive only
        dp2_scale_gauss_upper = T.exp(read_param(strict_upper_tri))
        dp_scale_gauss_diag   = T.exp(read_param(diag))

        symm_terms  = [inner_energy, inner_radius, inner_scale, energy_gauss, dist_loc, dist_scale]
        upper_terms = [dp1_steric_upper,      dp2_steric_upper,
                       dp1_loc_gauss_upper,   dp2_loc_gauss_upper, 
                       dp1_scale_gauss_upper, dp2_scale_gauss_upper] 
        diag_terms  = [dp_steric_diag, dp_loc_gauss_diag, dp_scale_gauss_diag]
                 
        return T.stack(*symm_terms), T.stack(*upper_terms), T.stack(*diag_terms)

    else:
        outer_energy =       read_param(upper_tri)  # outer_energy likely attractive
        outer_radius = T.exp(read_param(upper_tri))+inner_radius  # > inner_radius
        outer_scale  = T.exp(read_param(upper_tri))  # positive

        terms = [inner_energy, inner_radius, inner_scale, outer_energy, outer_radius, outer_scale]
        return T.stack(*terms),


def imat_pack(packed_params, upper_params=None, diag_params=None, n_restype=20):
    inner_energy = np.log(packed_params[0])
    inner_radius = np.log(packed_params[1])
    inner_scale  = np.log(packed_params[2])
    params = [inner_energy, inner_radius, inner_scale]

    if mode == 'PosDirBead':
        params.extend([packed_params[3], np.log(packed_params[4]), np.log(packed_params[5]),
                       upper_params[0],upper_params[1],diag_params[0],
                       upper_params[2],upper_params[3],diag_params[1],
                       np.log(upper_params[4]),np.log(upper_params[5]),np.log(diag_params[2])])
    else:
        outer_energy =        packed_params[3]
        outer_radius = np.log(packed_params[4]-packed_params[1])
        outer_scale  = np.log(packed_params[5])
        
        params.extend([outer_energy, outer_radius, outer_scale])

    return np.concatenate(params,axis=0)


lparam = T.dvector('lparam')

def energy_and_deriv_for_rotamer_objects(rotamer_objects, *param):
        interactions = np.zeros((20,20,P))

        def make_symm(f):
            ret = np.zeros((20,20))
            ret[np.triu_indices(20)] = f
            ret = ret + ret.T - np.diag(np.diagonal(ret))
            return ret

        def make_swap(upper,diag):
            ret = np.zeros((20,20,2))
            ret[np.triu_indices(20,k=1)] = upper.T
            ret = ret + ret[:,:,::-1].transpose((1,0,2)) + np.diag(diag)[...,None]
            return ret

        for i in range(P_symm):
            interactions[:,:,i] = make_symm(param[0][i])

        for s in range(P_swap/2):
            interactions[:,:,P_symm+2*s:P_symm++2*s+2] = make_swap(param[1][2*s:2*s+2],param[2][s])

        number_of_tasks = min(len(rotamer_objects),32)

        def accum_task(start_idx):
            energy = 0.
            deriv = np.zeros((20,20,P))

            for rfree,rfixed in rotamer_objects[start_idx::number_of_tasks]:
                er,dr = rfree .energy(interactions)
                ei,di = rfixed.energy(interactions)
                energy += ei-er
                deriv  += di-dr

            return energy,deriv

        energy = 0.
        deriv = np.zeros((20,20,P))
        for e,d in executor.map(accum_task, range(number_of_tasks)):
            energy += e
            deriv  += d

        if mode == 'PosDirBead':
            flat_deriv = np.zeros((P_symm,21*20/2)), np.zeros((P_swap,19*20/2)), np.zeros((P_swap/2,20))
        elif mode == 'PosBead':
            flat_deriv = np.zeros((P_symm,21*20/2)),

        for i in range(P_symm):
            flat_deriv[0][i] = deriv[:,:,i][np.triu_indices(20)]

        for i in range(P_swap):
            flat_deriv[1][i] = deriv[:,:,P_symm+i][np.triu_indices(20,k=1)].T
        for i in range(P_swap/2):
            flat_deriv[2][i] = deriv[:,:,P_symm+2*i].diagonal() + deriv[:,:,P_symm+2*i+1].diagonal()

        return energy, flat_deriv


class UpsideRotamerEnergyGap(theano.Op):
    def __init__(self, rotamer_data_files, n_res_limits = None):
        n_res = []
        for rdf in rotamer_data_files:
            d = cp.load(open(rdf))
            n_res.append(d['restype1'].shape[0]+d['restype3'].shape[0])
        n_res = np.array(n_res)

        indices = np.arange(len(n_res)) 
        
        if n_res_limits is not None:
            mask = (n_res_limits[0]<=n_res) & (n_res<n_res_limits[1])
            n_res = n_res[mask]
            indices = indices[mask]

        # sort by n_res to help load balancing
        indices = indices[np.argsort(n_res)]
        n_res   = n_res  [np.argsort(n_res)]

        self.total_n_res = n_res.sum()
        self.indices = indices
        self.n_res   = n_res

        def construct_rotamer_objects(i):
            # we need a unique temporary path, not temporary file
            d = cp.load(open(rotamer_data_files[i]))

            assert d['fixed_rotamer3'].shape == (d['restype3'].shape[0],)
            assert d['pos1'].shape[-1] ==   D
            assert d['pos3'].shape[-1] == 3*D

            rotamer_object_free = RotamerObjectHolder(upside_calc.new_rotamer_construct_and_solve(
                    20,
                    int(d['restype1'].shape[0]), d['restype1'].ctypes.data, d['pos1'].ctypes.data,
                    int(d['restype3'].shape[0]), d['restype3'].ctypes.data, d['pos3'].ctypes.data,
                    d['damping'], d['max_iter'], d['tol'],
                    None))

            fix = d['fixed_rotamer3'].astype('i4').copy()
            rotamer_object_fixed = RotamerObjectHolder(upside_calc.new_rotamer_construct_and_solve(
                    20,
                    int(d['restype1'].shape[0]), d['restype1'].ctypes.data, d['pos1'].ctypes.data,
                    int(d['restype3'].shape[0]), d['restype3'].ctypes.data, d['pos3'].ctypes.data,
                    d['damping'], d['max_iter'], d['tol'],
                    fix.ctypes.data))

            return (rotamer_object_free, rotamer_object_fixed)

        self.rotamer_objects = list(executor.map(construct_rotamer_objects, self.indices))

    def make_node(self, *param):
        if mode == 'PosBead':
            p, = param
            return theano.Apply(self, [T.as_tensor_variable(p)], [T.dscalar()])
        elif mode == 'PosDirBead':
            p,u,d = param
            return theano.Apply(self, [T.as_tensor_variable(x) for x in (p,u,d)], [T.dscalar()])

    def perform(self, node, inputs_storage, output_storage):
        energy, flat_deriv = energy_and_deriv_for_rotamer_objects(
                self.rotamer_objects, *inputs_storage)

        result = np.array(float(energy)*1./self.total_n_res)
        output_storage[0][0] = result

    def grad(self, inputs, output_gradients):
        grad_func = UpsideRotamerEnergyGapGrad(self.total_n_res,self.rotamer_objects)
        if mode=='PosBead':
            return [output_gradients[0] * grad_func(inputs[0])]
        else:
            return [output_gradients[0] * x for x in grad_func(*inputs)]


class UpsideRotamerEnergyGapGrad(theano.Op):
    def __init__(self, total_n_res, rotamer_objects):
        self.total_n_res = total_n_res
        self.rotamer_objects = rotamer_objects

    def make_node(self, *param):
        if mode == 'PosBead':
            p, = param
            return theano.Apply(self, [T.as_tensor_variable(p)], [T.dmatrix()])
        elif mode == 'PosDirBead':
            p,u,d = param
            return theano.Apply(self, [T.as_tensor_variable(x) for x in (p,u,d)], 
                    [T.dmatrix(),T.dmatrix(),T.dmatrix()])

    def perform(self, node, inputs_storage, output_storage):
        energy, flat_deriv = energy_and_deriv_for_rotamer_objects(
                self.rotamer_objects, *inputs_storage)

        for i in range(len(output_storage)):
            output_storage[i][0] = flat_deriv[i]*(1./self.total_n_res)


def convert_structure_to_rotamer_data(fasta, init_structure, rotamer_lib):
    res = dict()
    tmp_path = os.tempnam()

    try:
        open(tmp_path,'w').close()  # ensure some file at that path
        # configure for upside runnign
        sp.check_call([
            os.path.join(upside_path,'upside_config'), 
            '--fasta', fasta,
            '--initial-structures', init_structure,
            '--output', tmp_path,
            '--rotamer', rotamer_lib,
            '--debugging-only-disable-basic-springs'])

        # we can now run to get the rotamer placement information
        sp.check_output([os.path.join(upside_path,"upside"), 
            '--config', tmp_path,
            '--duration', '3e-2',
            '--frame-interval', '1',
            '--overwrite-output', 
            '--log-level', 'extensive'])

        # read out the data
        with tb.open_file(tmp_path) as t:
            res['restype1'] = t.root.output.rotamer_restype1[:].astype('i4').copy()
            res['restype3'] = t.root.output.rotamer_restype3[:].astype('i4').copy()

            # we have to be careful here; somethings go wrong if there 
            # are zero residues of a given type
            if t.root.output.rotamer_pos1.shape[0] != 0:
                res['pos1'] = t.root.output.rotamer_pos1[0,0].astype('f4').copy()
            else:
                res['pos1'] = np.zeros((0,4),dtype='f4')

            if t.root.output.rotamer_pos3.shape[0] != 0:
                res['pos3'] = t.root.output.rotamer_pos3[0,0].astype('f4').copy()
            else:
                res['pos3'] = np.zeros((0,12),dtype='f4')

            res['damping']  = float(t.root.input.potential.rotamer._v_attrs.damping)
            res['max_iter'] = int  (t.root.input.potential.rotamer._v_attrs.max_iter)
            res['tol']      = float(t.root.input.potential.rotamer._v_attrs.tol)

    finally:
        os.remove(tmp_path)

    return res
