import tables as tb
import numpy as np
import tempfile
import subprocess as sp
import os

upside_path = '../obj'

class RotamerEnergy(object):
    def __init__(self, config_file, tmp_dir = None):
        self.config_file = config_file
        with tb.open_file(self.config_file) as t:
            self.n_restype = t.root.input.potential.rotamer.energy.shape[0]
        self.tmp_dir = tmp_dir
        return

    def germ(self, parameters):
        for x in parameters: assert x.shape == ((self.n_restype*(self.n_restype+1))/2,)
        parameter_matrices = []

        for i in range(6):
            m = np.zeros((self.n_restype,self.n_restype))
            m[np.triu_indices_from(m)] = parameters[i]
            # symmetrize without over-counting diagonal
            m = m + m.T - np.diag(m.diagonal())
            parameter_matrices.append(m)

        with tempfile.NamedTemporaryFile(suffix='.h5', dir=self.tmp_dir) as new_config:
            # copy reference config to new config
            with open(self.config_file, 'rb') as old_config:
                new_config.write(old_config.read())
                new_config.flush()

            # load the new paramters
            with tb.open_file(new_config.name,'a') as t:
                g = t.root.input.potential.rotamer

                g.energy[:,:,0] = parameter_matrices[0]
                g.radius[:,:,0] = parameter_matrices[1]
                g.width [:,:,0] = parameter_matrices[2]

                g.energy[:,:,1] = parameter_matrices[3]
                g.radius[:,:,1] = parameter_matrices[4]
                g.width [:,:,1] = parameter_matrices[5]

            # evaluate the energy and derivative
            sp.check_output([os.path.join(upside_path,"upside"), 
                '--config', new_config.name,
                '--duration', '3e-2',
                '--frame-interval', '1',
                '--overwrite-output', 
                '--log-level', 'extensive'])

            # dump the parameters and energy
            with tb.open_file(new_config.name) as t:
                potential = t.root.output.potential[0,0]
                d         = t.root.output.rotamer_interaction_parameter_gradient[0,0]

            triu = lambda m: m[np.triu_indices_from(m)]
            deriv_flat = np.row_stack(x[np.triu_indices_from(x)] for x in d.transpose((2,0,1)))

        return potential, deriv_flat


    def spot_check_derivative(self, parameters, rt1, rt2, eps=(0.03,0.03,0.03,  0.03,0.03,0.03)):
        a,b = np.triu_indices(self.n_restype)
        idx, = ((a==rt1) & (b==rt2)).nonzero()[0]

        p = list(parameters)

        def update_vec(v,i,dx):
            v = v.copy()
            v[i] += dx
            return v

        pot, der = self.germ(p)

        der_est = np.zeros(6)
        for i in range(6):
            new_p = [(p[j] if j!=i else update_vec(p[j],idx, eps[i])) for j in range(len(p))]
            p_up   = self.germ(new_p)[0]

            new_p = [(p[j] if j!=i else update_vec(p[j],idx,-eps[i])) for j in range(len(p))]
            p_down = self.germ(new_p)[0]

            der_est[i] = (p_up-p_down)/(2.*eps[i])

        return [x[idx] for x in der], der_est


def configure_rotamer_potentials(
        file_obj_free, file_obj_fixed, 
        fasta_path, init_structure_path, rotamer_lib, fixed_rotamers = None, tmp_dir = '/dev/shm'):

    # we need a unique temporary path, not temporary file
    tmp_path = os.tempnam(tmp_dir)

    try:
        open(tmp_path,'w').close()  # ensure some file at that path
        sp.check_call([
            os.path.join(upside_path,'upside_config'), 
            '--fasta', fasta_path,
            '--initial-structures', init_structure_path,
            '--output', tmp_path,
            '--rotamer', rotamer_lib,
            '--debugging-only-disable-basic-springs'])

        # we can now copy the tmp_path into the file object
        with open(tmp_path) as tmp_file:
            file_obj_free.write(tmp_file.read())
            file_obj_free.flush()
    finally:
        os.remove(tmp_path)

    if fixed_rotamers is not None:
        assert fixed_rotamers.dtype in (np.int, np.int64, np.int32)

        file_obj_free.seek(0)
        file_obj_fixed.write(file_obj_free.read())
        file_obj_fixed.flush()

        with tb.open_file(file_obj_fixed.name, 'a') as t:
            t.create_array(t.root.input.potential.rotamer, 'fixed_rotamers', obj=fixed_rotamers)

last_value_point = [None]
last_deriv_point = [None]

import theano
import theano.tensor as T
class RotamerEnergyGapGrad(theano.Op):
    def __init__(self, energy_obj_free, energy_obj_fixed):
        self.energy_obj_free  = energy_obj_free
        self.energy_obj_fixed = energy_obj_fixed

    def make_node(self, param):
        return theano.Apply(self, [T.as_tensor_variable(param)], [T.dmatrix()])

    def perform(self, node, inputs_storage, output_storage):
        last_deriv_point[0] = inputs_storage[0].copy()
        energy_free,  deriv_free  = self.energy_obj_free .germ(inputs_storage[0])
        energy_fixed, deriv_fixed = self.energy_obj_fixed.germ(inputs_storage[0])
        output_storage[0][0] = (deriv_fixed - deriv_free).astype('f8')


class RotamerEnergyGap(theano.Op):
    def __init__(self, fasta_path, init_structure_path, rotamer_lib, fixed_rotamers, tmp_dir=None):
        self.config_free  = tempfile.NamedTemporaryFile(suffix='.h5', dir=tmp_dir)
        self.config_fixed = tempfile.NamedTemporaryFile(suffix='.h5', dir=tmp_dir)

        configure_rotamer_potentials(self.config_free, self.config_fixed,
                fasta_path, init_structure_path, rotamer_lib, fixed_rotamers, tmp_dir)

        self.energy_obj_free  = RotamerEnergy(self.config_free.name)
        self.energy_obj_fixed = RotamerEnergy(self.config_fixed.name)

    def make_node(self, param):
        # FIXME I need more type checking and size checking here
        return theano.Apply(self, [T.as_tensor_variable(param)], [T.dscalar()])

    def perform(self, node, inputs_storage, output_storage):
        last_deriv_point[0] = inputs_storage[0].copy()
        energy_free,  deriv_free  = self.energy_obj_free .germ(inputs_storage[0])
        energy_fixed, deriv_fixed = self.energy_obj_fixed.germ(inputs_storage[0])
        output_storage[0][0] = float(energy_fixed - energy_free)

    def grad(self, inputs, output_gradients):
        grad_func = RotamerEnergyGapGrad(self.energy_obj_free, self.energy_obj_fixed)
        return [output_gradients[0] * grad_func(inputs[0])]



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

from mpi4py import MPI
world = MPI.COMM_WORLD
rank  = world.Get_rank()
size  = world.Get_size()

class MultiProteinEnergyGap(object):
    def __init__(self, fasta, init_structure, rotamer_lib, fixed_rotamer, tmp_dir=None, n_res_limits = None):
        n_res = []
        if not rank:
            for fr in fixed_rotamer:
                with open(fr) as f:
                    n_res.append(len(f.readlines()))
            n_res = np.array(n_res)
            residue_order = np.argsort(n_res)
            
            if n_res_limits is not None:
                mask = (n_res_limits[0]<=n_res) & (n_res<n_res_limits[1])
                n_res = n_res[mask]
                residue_order = residue_order[mask]

            n_res, residue_order = world.bcast((n_res,residue_order),root=0)
        else:
            n_res, residue_order = world.bcast(None,root=0)

        self.total_n_res = n_res.sum()

        self.indices = residue_order[rank::size]
        self.n_res   = n_res[self.indices]
        self.energy_deriv_functions = []

        for i in self.indices:
            expr = RotamerEnergyGap(fasta[i],init_structure[i],rotamer_lib,fixed_rotamer[i], tmp_dir)(
                    interaction_matrix_from_param_vector_expr(lparam))
            self.energy_deriv_functions.append((
                theano.function([lparam], expr),
                theano.function([lparam], T.grad(expr,lparam))))

        return

    def full_value(self, p):
        my_value = np.zeros(1)
        for val,der in self.energy_deriv_functions:
            my_value[0] += val(p)
        value = np.zeros_like(my_value)
        world.Allreduce(my_value, value)
        return value*(1./self.total_n_res)

    def full_deriv(self, p):
        my_deriv = np.zeros(p.shape, dtype='f8')
        for val,der in self.energy_deriv_functions:
            my_deriv += der(p)
        deriv = np.zeros_like(my_deriv)
        world.Allreduce(my_deriv, deriv)
        return deriv*(1./self.total_n_res)

    def nesterov_sgd_pass(self, init_p, init_momentum, momentum_persistence, step_size, mini_batch_size):
        pass  # FIXME should I use size-matching here?

