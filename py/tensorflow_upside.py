import tables as tb
import numpy as np
import tempfile
import subprocess as sp
import os
import cPickle as cp
import collections
import uuid
import predict_chi1

gensym_salt = str(uuid.uuid4()).replace('-','')
gensym_count = [0]
deg = np.pi/180.

def _unique_name(prefix):
    nm = '%s_%i_%s' %(prefix, gensym_count[0], gensym_salt)
    gensym_count[0] += 1
    return nm

def numpy_reduce_inplace(comm, arrays, root=0):
    # FIXME consolidate gathers with the same dtype to reduce latency
    for a in arrays:
        a_copy = a.copy()
        comm.Reduce(a_copy,a, root=root)

class UpsideEnergy(object):
    def __init__(self, n_restype, name_list, path_list, chi1_file_list, mpi_comm=None):
        import pandas as pd
        self.mpi_comm = mpi_comm
        self.is_master = self.mpi_comm is None or self.mpi_comm.rank==0

        assert len(name_list) == len(set(name_list)) # no duplicates

        import upside_engine as ue
        self.systems = dict()

        assert len(name_list) == len(path_list) == len(chi1_file_list)
        name_path_list = list(zip(name_list,path_list,chi1_file_list))

        if self.mpi_comm is not None:
            name_path_list = self.mpi_comm.scatter(
                    [name_path_list[i::self.mpi_comm.size] for i in range(self.mpi_comm.size)])
            n_restype = self.mpi_comm.bcast(n_restype)
        self.n_restype = n_restype

        for name,path,chi1_file in name_path_list:
            d = dict()
            d['path'] = path
            with tb.open_file(path) as t:
                d['predict_chi1'] = predict_chi1.Chi1Predict(t.root.input.args._v_attrs.rotamer_placement)
                d['pos'] = t.root.input.pos[:,:,0]
                d['seq'] = t.root.input.sequence[:]
                d['residue_num'] = t.root.input.potential.placement_fixed_point_vector_only.affine_residue[:]
                d['n_res'] = len(d['seq'])
                chi1_true = pd.read_csv(chi1_file, delim_whitespace=1)
                chi1_true = chi1_true[np.isfinite(chi1_true.chi1)]
                d['chi1_state'] = predict_chi1.compute_chi1_state(chi1_true.chi1.as_matrix()*deg)
                d['chi1_residue'] = chi1_true.residue.as_matrix()
                # d['chi1'] = chi1_true.chi1.as_matrix()
            d['engine'] = ue.Upside(path)
            self.systems[name] = d

        self.grad_name = None

        if not self.is_master:
            self._worker_loop()
        return

    def _worker_loop(self,):
            # if we are not on rank 0, we never really exit the constructor, instead
            #   waiting in an infinite loop to assist in computations as workers
            while True:
                # post bcat for all nodes to wait for rank 0
                rpc = self.mpi_comm.bcast(None)
                if rpc[0] == 'shutdown':
                    self.shutdown()
                    break
                elif rpc[0] == 'evaluate':
                    # be careful here about the keyword argument energy_sens
                    self.evaluate(*rpc[1:-1], energy_sens=rpc[-1])
                elif rpc[0] == 'evaluate_chi1_loss':
                    self.evaluate_chi1_loss(*rpc[1:])
                else:
                    raise RuntimeError('bad command')


    def evaluate_tensorflow(self, system_names, param_names, *param_tensors):
        name=None
        # FIXME this does not use the solution from computing the value to speed 
        #  computing the gradient.  I should figure out how to cache the parameter derivatives
        #  to make this easier.

        from tensorflow.python.framework import ops
        import tensorflow as tf

        if self.grad_name is None:
            self.grad_name = _unique_name('UpsideEnergyGrad')

            @tf.RegisterGradient(self.grad_name)
            def grad(op, energy_sens, n_res_sens, self=self):
                inputs = tuple(op.inputs)
                with ops.control_dependencies([energy_sens.op]):
                    param_deriv = tf.py_func(
                            (lambda energy_sens, system_names, param_names, *param_tensors:
                        self.evaluate(system_names, param_names, param_tensors, energy_sens=energy_sens)[-1]),
                            (energy_sens,)+inputs,
                            [tf.float32]*(len(inputs)-2))
                    return [None,None] + param_deriv

        g = tf.get_default_graph()
        with g.gradient_override_map(dict(PyFunc=self.grad_name)):
            return tf.py_func(
                    (lambda system_names, param_names, *param_tensors:
                        self.evaluate(system_names, param_names, param_tensors)),
                    (system_names,param_names) + param_tensors,
                    [tf.float32, tf.int32],
                    name=name)

    def evaluate_chi1_loss_tensorflow(self, system_names, param_names, *param_tensors):
        import tensorflow as tf
        return tf.py_func(
                (lambda system_names, param_names, *param_tensors:
                    self.evaluate_chi1_loss(system_names, param_names, param_tensors)),
                (system_names,param_names) + param_tensors,
                tf.float32)

    def evaluate_chi1_loss(self, system_names, param_names, param_tensors):
        if self.mpi_comm is not None and self.is_master:
            # activate the workers with send to match their waiting collectives
            self.mpi_comm.bcast(('evaluate_chi1_loss', system_names, param_names,param_tensors,))

        my_results = np.zeros((self.n_restype,2), dtype='i8')

        for i,name in enumerate(system_names):
            if name not in self.systems:
                continue

            d = self.systems[name]
            engine = d['engine']
            predictor = d['predict_chi1']

            for pnm, pt in zip(param_names, param_tensors):
                engine.set_param(pt, pnm)

            engine.energy(d['pos'])
            chi1_prob = predictor.predict_chi1(d['seq'], d['residue_num'], engine.get_sens('hbond_coverage')[:,0])
            # print d['seq']
            # print d['chi1_residue']
            # for qq in  zip(d['seq'][d['chi1_residue']], chi1_prob[d['chi1_residue']], d['chi1'], d['chi1_state']):
            #     print qq
            my_results += predictor.compute_zero_one_stats(
                    d['seq'][d['chi1_residue']], chi1_prob[d['chi1_residue']], d['chi1_state'])

        if self.mpi_comm is not None:
            numpy_reduce_inplace(self.mpi_comm, [my_results])

        if self.is_master:
            return my_results.astype('f4')


    def evaluate(self, system_names, param_names, param_tensors, energy_sens=None):
        if energy_sens is not None:
            energy_sens = np.asarray(energy_sens)

        if self.mpi_comm is not None and self.is_master:
            # activate the workers with send to match their waiting collectives
            self.mpi_comm.bcast(('evaluate', system_names, param_names,param_tensors,energy_sens,))

        n_sys = len(system_names)
        assert system_names.shape == (n_sys,)
        if energy_sens is not None:
            assert energy_sens.shape == (n_sys,)
        assert len(param_names) == len(param_tensors)

        energy = np.zeros(n_sys, dtype='f4')
        n_res  = np.zeros(n_sys, dtype='i4')  # return number of residues in each system

        # param_deriv contains the parameter derivatives contracted with the 
        #  energy sensitivities
        if energy_sens is not None:
            # accumulate in double precision to avoid numerical errors
            param_deriv = dict((pnm, np.zeros(pt.shape,dtype='f8'))
                    for pnm,pt in zip(param_names, param_tensors))

        for i,name in enumerate(system_names):
            if name not in self.systems:
                continue

            d = self.systems[name]
            engine = d['engine']
            n_res[i] = d['n_res']

            for pnm, pt in zip(param_names, param_tensors):
                engine.set_param(pt, pnm)

            energy[i] = engine.energy(d['pos'])

            if energy_sens is not None:
                for pnm, pt in zip(param_names, param_tensors):
                    param_deriv[pnm] += energy_sens[i]*engine.get_param_deriv(pt.shape, pnm)

        if self.mpi_comm is not None:
            numpy_reduce_inplace(self.mpi_comm,
                    [energy,n_res] +
                    (list(param_deriv.values()) if energy_sens is not None else []))

        if self.is_master:
            if np.any(n_res==0):
                raise ValueError('missing n_res, maybe missing name in name list')

            if energy_sens is None:
                return energy, n_res
            else:
                return energy, n_res, [param_deriv[pnm].astype('f4') for pnm in param_names]

    def shutdown(self):
        if self.mpi_comm is not None and self.is_master:
            self.mpi_comm.bcast(['shutdown'])
        self.systems = None


# # handle a mix of scalars and lists
# def read_comp(x,i):
#     try:
#         return x[i]
#     except:
#         return x  # scalar case
# 
# class AdamSolver(object):
#     ''' See Adam optimization paper (Kingma and Ba, 2015) for details. Beta2 is reduced by 
#     default to handle the shorter training expected on protein problems.  alpha is roughly 
#     the largest possible step size.'''
#     def __init__(self, n_comp, alpha=1e-2, beta1=0.8, beta2=0.96, epsilon=1e-6):
#         self.n_comp  = n_comp
# 
#         self.alpha   = alpha
#         self.beta1   = beta1
#         self.beta2   = beta2
#         self.epsilon = epsilon
# 
#         self.step_num = 0
#         self.grad1    = [0. for i in range(n_comp)]  # accumulator of gradient
#         self.grad2    = [0. for i in range(n_comp)]  # accumulator of gradient**2
# 
#     def update_for_d_obj(self,):
#         return [0. for x in self.grad1]  # This method is used in Nesterov SGD, not Adam
# 
#     def update_step(self, grad):
#         r = read_comp
#         self.step_num += 1
#         t = self.step_num
# 
#         u = [None]*len(self.grad1)
#         for i,g in enumerate(grad):
#             b=r(self.beta1,i); self.grad1[i] = b*self.grad1[i] + (1.-b)*g    ; grad1corr = self.grad1[i]/(1-b**t)
#             b=r(self.beta2,i); self.grad2[i] = b*self.grad2[i] + (1.-b)*g**2 ; grad2corr = self.grad2[i]/(1-b**t)
#             u[i] = -r(self.alpha,i) * grad1corr / (np.sqrt(grad2corr) + r(self.epsilon,i))
# 
#         return u
# 
#     def log_state(self, direc):
#         with open(os.path.join(direc, 'solver_state.pkl'),'w') as f: 
#             cp.dump(dict(step_num=self.step_num, grad1=self.grad1, grad2=self.grad2, solver=str(self)), f, -1)
# 
#     def __repr__(self):
#         return 'AdamSolver(%i, alpha=%r, beta1=%r, beta2=%r, epsilon=%r)'%(
#                 self.n_comp,self.alpha,self.beta1,self.beta2,self.epsilon)
# 
#     def __str__(self):
#         return 'AdamSolver(%i, alpha=%s, beta1=%s, beta2=%s, epsilon=%s)'%(
#                 self.n_comp,self.alpha,self.beta1,self.beta2,self.epsilon)
# 
# 
# class SGD_Solver(object):
#     def __init__(self, n_comp, mu=0.9, learning_rate = 0.1, nesterov=True):
#         self.n_comp = n_comp
# 
#         self.mu = mu
#         self.learning_rate = learning_rate
#         self.nesterov = nesterov
# 
#         self.momentum = [0. for i in range(n_comp)]
# 
#     def update_for_d_obj(self,):
#         if self.nesterov:
#             return [read_comp(self.mu,i)*self.momentum[i] for i in range(self.n_comp)]
#         else:
#             return [0. for i in range(n_comp)]
# 
#     def update_step(self, grad):
#         self.momentum = [read_comp(self.mu,i)*self.momentum[i] - read_comp(self.learning_rate,i)*grad[i] 
#                 for i in range(self.n_comp)]
#         return [1.*x for x in self.momentum]  # make sure the user doesn't smash the momentum


