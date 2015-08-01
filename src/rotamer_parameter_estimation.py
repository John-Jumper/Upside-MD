import tables as tb
import numpy as np
import tempfile
import subprocess as sp
import os
import theano
import theano.tensor as T
import cPickle as cp
import scipy.optimize as opt

n_restype = 20

lparam = T.dvector('lparam')
func = lambda expr: (theano.function([lparam],expr),expr)
i = [0]
def read_param(shape):
    size = np.prod(shape)
    ret = lparam[i[0]:i[0]+size].reshape(shape)
    i[0] += size
    return ret

def unpack_param_maker():
    def read_symm():
        x = read_param((n_restype,n_restype))
        return 0.5*(x + x.T)

    inner_energy = T.exp(read_symm())
    inner_radius = T.exp(read_symm())
    inner_scale  = T.exp(read_symm())

    outer_energy = read_symm()
    outer_radius = inner_radius + T.exp(read_symm())
    outer_scale  = T.exp(read_symm())

    rot = T.stack(inner_energy, inner_radius, inner_scale, outer_energy, outer_radius, outer_scale
            ).transpose((1,2,0))

    def read_cov():
        return read_param((2,n_restype))

    cov_radius      = T.exp(read_cov())
    cov_scale       = T.exp(read_cov())
    cov_angle       = read_cov()
    cov_angle_scale = T.exp(read_cov())
    cov_energy      = read_cov()

    cov = T.stack(cov_radius, cov_scale, cov_angle, cov_angle_scale, cov_energy).transpose((1,2,0))

    return func(rot), func(cov)

(unpack_rot,unpack_rot_expr), (unpack_cov,unpack_cov_expr) = unpack_param_maker()

def pack_param(loose_rot,loose_cov, check_accuracy=True):
    discrep,     discrep_expr = func(T.sum((unpack_rot_expr - loose_rot)**2) +
                                     T.sum((unpack_cov_expr - loose_cov)**2))
    d_discrep, d_discrep_expr = func(T.grad(discrep_expr,lparam))

    # solve the resulting equations so I don't have to work out the formula
    results = opt.minimize(
            (lambda x: discrep(x)),
            np.zeros(n_restype*n_restype*6+2*n_restype*5),
            method = 'L-BFGS-B',
            jac = (lambda x: d_discrep(x)))

    if not check_accuracy and not (discrep(results.x) < 1e-4):
        raise ValueError('Failed to converge')

    return results.x


def bind_param_and_evaluate(pos_fix_free, node_names, param_matrices):
    energy = np.zeros(2)
    deriv = [np.zeros((2,)+pm.shape) for pm in param_matrices]

    for pos, fix, free in pos_fix_free:
        for nm, pm in zip(node_names, param_matrices):
            fix .set_param(pm, nm) 
            free.set_param(pm, nm) 

        energy[0] += fix .energy(pos)
        energy[1] += free.energy(pos)

        for d,nm in zip(deriv,node_names):
            d[0] += fix .get_param_deriv(d[0].shape, nm)
            d[1] += free.get_param_deriv(d[0].shape, nm)

    return energy, deriv


class UpsideEnergyGap(theano.Op):
    def __init__(self, protein_data, node_names):
        self.protein_data = [None,None]
        self.change_protein_data(protein_data)   # (total_n_res, pos_fix_free)
        self.node_names   = list(node_names)

    def make_node(self, *param):
        assert len(param) == len(self.node_names)
        return theano.Apply(self, [T.as_tensor_variable(x) for x in param], [T.dvector()])

    def perform(self, node, inputs_storage, output_storage):
        total_n_res, pos_fix_free = self.protein_data
        energy, deriv = bind_param_and_evaluate(pos_fix_free, self.node_names, inputs_storage)
        output_storage[0][0] = (energy/total_n_res).astype('f8')

    def grad(self, inputs, output_gradients):
        grad_func = UpsideEnergyGapGrad(self.protein_data, self.node_names)  # grad will have linked data
        gf = grad_func(*inputs)
        if len(inputs) == 1: gf = [gf]  # single inputs cause problems
        return [T.tensordot(output_gradients[0], x, axes=(0,0)) for x in gf]

    def change_protein_data(self, new_protein_data):
        self.protein_data[0] = 1*new_protein_data[0]
        self.protein_data[1] = list(new_protein_data[1])


class UpsideEnergyGapGrad(theano.Op):
    def __init__(self, protein_data, node_names):
        self.protein_data = protein_data
        self.node_names   = list(node_names)

    def make_node(self, *param):
        assert len(param) == len(self.node_names)
        return theano.Apply(self, 
                [T.as_tensor_variable(p) for p in param], 
                [T.dtensor4() for p in param])

    def perform(self, node, inputs_storage, output_storage):
        total_n_res, pos_fix_free = self.protein_data
        energy, deriv = bind_param_and_evaluate(pos_fix_free, self.node_names, inputs_storage)

        for i in range(len(output_storage)):
            output_storage[i][0] = (deriv[i]*(1./total_n_res)).astype('f8')


def sgd_sweep(state, mom, mu, eps, minibatches, change_batch_function, d_obj, nesterov=True):
    for mb in minibatches:
        change_batch_function(mb)
        state, mom = state + mom, mu*mom - eps*d_obj(state+mu*mom if nesterov else state)
    return state, mom
