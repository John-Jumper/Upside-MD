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


executor = concurrent.futures.ThreadPoolExecutor(4)


class RotamerObjectHolder(object):
    def __init__(self, rotamer_object):
        self.rotamer_object = rotamer_object

    def energy(self,interactions):
        assert interactions.shape == (20,20,6)
        deriv = np.zeros(interactions.shape, dtype='f4')
        interactions = interactions.astype('f4').copy()
        value = upside_calc.free_energy_and_parameter_deriv(self.rotamer_object, 
                deriv.ctypes.data, interactions.ctypes.data)
        return float(value),deriv.astype('f8')

    def __del__(self):
        upside_calc.delete_rotamer_construct_and_solve(self.rotamer_object)


def interaction_matrix_from_param_vector_expr(param_vector):
    inner_energy = T.exp(param_vector[0,:])  # inner_energy is repulsive, so positive
    inner_radius = T.exp(param_vector[1,:])  # positive
    inner_width  = T.exp(param_vector[2,:])  # positive
        
    outer_energy =       param_vector[3,:]  # outer_energy is probably attractive
    outer_radius = T.exp(param_vector[4,:])+inner_radius  # > inner_radius
    outer_width  = T.exp(param_vector[5,:])  # positive
    # FIXME I should penalize heavily cases where width > radius since derivative is non-smooth at origin
    
    return T.stack(inner_energy, inner_radius, inner_width, outer_energy, outer_radius, outer_width)

lparam = T.dmatrix('lparam')
to_matrix = theano.function([lparam],interaction_matrix_from_param_vector_expr(lparam))

def energy_and_deriv_for_rotamer_objects(rotamer_objects, flat_interactions):
        interactions = np.zeros((20,20,6))
        inds = np.triu_indices(20)
        interactions[inds] = flat_interactions.T

        for i in range(6):
            interactions[:,:,i] = (interactions[:,:,i] + interactions[:,:,i].T -
                    np.diag(np.diagonal(interactions[:,:,i])))

        number_of_tasks = min(len(rotamer_objects),32)

        def accum_task(start_idx):
            energy = 0.
            deriv = np.zeros((20,20,6))

            for rfree,rfixed in rotamer_objects[start_idx::number_of_tasks]:
                er,dr = rfree .energy(interactions)
                ei,di = rfixed.energy(interactions)
                energy += ei-er
                deriv  += di-dr

            return energy,deriv

        energy = 0.
        deriv = np.zeros((20,20,6))
        for e,d in executor.map(accum_task, range(number_of_tasks)):
            energy += e
            deriv  += d

        flat_deriv = np.zeros((6,21*20/2))
        for i in range(6):
            flat_deriv[i] = deriv[:,:,i][inds]

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

    def make_node(self, param):
        return theano.Apply(self, [T.as_tensor_variable(param)], [T.dscalar()])

    def perform(self, node, inputs_storage, output_storage):
        energy, flat_deriv = energy_and_deriv_for_rotamer_objects(
                self.rotamer_objects, inputs_storage[0])

        result = np.array(float(energy)*1./self.total_n_res)
        output_storage[0][0] = result

    def grad(self, inputs, output_gradients):
        grad_func = UpsideRotamerEnergyGapGrad(self.total_n_res,self.rotamer_objects)
        return [output_gradients[0] * grad_func(inputs[0])]


class UpsideRotamerEnergyGapGrad(theano.Op):
    def __init__(self, total_n_res, rotamer_objects):
        self.total_n_res = total_n_res
        self.rotamer_objects = rotamer_objects

    def make_node(self, param):
        # FIXME I need more type checking and size checking here
        return theano.Apply(self, [T.as_tensor_variable(param)], [T.dmatrix()])

    def perform(self, node, inputs_storage, output_storage):
        energy, flat_deriv = energy_and_deriv_for_rotamer_objects(
                self.rotamer_objects, inputs_storage[0])

        output_storage[0][0] = flat_deriv*(1./self.total_n_res)



#     def nesterov_sgd_pass(self, init_p, init_momentum, momentum_persistence, step_size, mini_batch_size):
#         pass  # FIXME should I use size-matching here?



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
