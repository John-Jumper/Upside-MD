import sys
import time
import tables as tb
import numpy as np
import tempfile
import subprocess as sp
import os
import cPickle as cp
import collections
import uuid
import predict_chi1
import upside_engine as ue

import threading

mpi_lock = threading.Lock()

gensym_salt = str(uuid.uuid4()).replace('-','')
gensym_count = [0]
deg = np.pi/180.


# IMPORTANT NOTE
# Tensorflow will multithread the evaluation of the graph by default
# This is incompatible with MPI since it interleaves calls, so we need locks.

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
    def __init__(self, n_restype, name_list, path_list, chi1_file_list, condiv_dict=dict(), mpi_comm=None):
        self.mpi_comm = mpi_comm
        import pandas as pd
        self.is_master = self.mpi_comm is None or self.mpi_comm.rank==0

        self.condiv_dict = dict(condiv_dict)
        self.condiv_deriv_cache = dict()
        self.condiv_log = []

        assert len(name_list) == len(set(name_list)) # no duplicates

        import upside_engine as ue
        self.systems = dict()

        assert len(name_list) == len(path_list) == len(chi1_file_list)
        name_path_list = list(zip(name_list,path_list,chi1_file_list))

        if self.mpi_comm is not None:
            name_path_list = self.mpi_comm.scatter(
                    [name_path_list[i::self.mpi_comm.size] for i in range(self.mpi_comm.size)])
            n_restype = self.mpi_comm.bcast(n_restype)
            self.condiv_dict = self.mpi_comm.bcast(condiv_dict)
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

        self.condiv_grad_name = None
        self.grad_name = None

        if not self.is_master:
            self._worker_loop()
        return

    def _worker_loop(self,):
        # if we are not on rank 0, we never really exit the constructor, instead
        #   waiting in an infinite loop to assist in computations as workers
        while True:
            # post bcast for all nodes to wait for rank 0
            rpc = self.mpi_comm.bcast(None)
            if rpc[0] == 'shutdown':
                self.shutdown()
                break
            elif rpc[0] == 'condiv':
                # be careful here about the keyword argument energy_sens
                self.condiv(*rpc[1:-1], energy_sens=rpc[-1])
            elif rpc[0] == 'evaluate':
                # be careful here about the keyword argument energy_sens
                self.evaluate(*rpc[1:-1], energy_sens=rpc[-1])
            elif rpc[0] == 'evaluate_chi1_loss':
                self.evaluate_chi1_loss(*rpc[1:])
            else:
                raise RuntimeError('bad command')


    def condiv_tensorflow(self, duration, system_names, param_names, *param_tensors):
        name=None

        from tensorflow.python.framework import ops
        import tensorflow as tf

        if self.condiv_grad_name is None:
            self.condiv_grad_name = _unique_name('CondivGrad')

            @tf.RegisterGradient(self.condiv_grad_name)
            def grad(op, energy_sens, rmsd_sens, bad_solve_sens):
                # NB we produce no derivative for rmsd_sens
                inputs = tuple(op.inputs)
                with ops.control_dependencies([energy_sens.op]):
                    def df(energy_sens, duration, system_names, param_names, *param_tensors):
                        return self.condiv(duration, system_names, param_names, param_tensors, energy_sens=energy_sens)

                    param_deriv = tf.py_func(df,
                            (energy_sens,)+inputs,
                            [tf.float32]*(len(inputs)-3))
                    return [None,None,None] + param_deriv

        g = tf.get_default_graph()
        with g.gradient_override_map(dict(PyFunc=self.condiv_grad_name)):
            def f(duration, system_names, param_names, *param_tensors):
                return self.condiv(duration, system_names, param_names, param_tensors)

            return tf.py_func(f,
                    (duration,system_names,param_names) + param_tensors,
                    [tf.float32, tf.float32, tf.float32],
                    name=name)

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
            def grad(op, energy_sens, n_res_sens):
                inputs = tuple(op.inputs)
                with ops.control_dependencies([energy_sens.op]):
                    def df(energy_sens, system_names, param_names, *param_tensors):
                        return self.evaluate(system_names, param_names, param_tensors, energy_sens=energy_sens)[-1]
                    param_deriv = tf.py_func(df,
                            (energy_sens,)+inputs,
                            [tf.float32]*(len(inputs)-2))
                    return [None,None] + param_deriv

        g = tf.get_default_graph()
        with g.gradient_override_map(dict(PyFunc=self.grad_name)):
            def f(system_names, param_names, *param_tensors):
                return self.evaluate(system_names, param_names, param_tensors)
            return tf.py_func(f,
                    (system_names,param_names) + param_tensors,
                    [tf.float32, tf.int32],
                    name=name)

    def evaluate_chi1_loss_tensorflow(self, system_names, param_names, *param_tensors):
        import tensorflow as tf
        def f(system_names, param_names, *param_tensors):
            return self.evaluate_chi1_loss(system_names, param_names, param_tensors)

        return tf.py_func(f,
                (system_names,param_names) + param_tensors,
                tf.float32)

    def evaluate_chi1_loss(self, system_names, param_names, param_tensors):
        with mpi_lock:
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
                my_results += predictor.compute_zero_one_stats(
                        d['seq'][d['chi1_residue']], chi1_prob[d['chi1_residue']], d['chi1_state'])

            if self.mpi_comm is not None:
                numpy_reduce_inplace(self.mpi_comm, [my_results])

            if self.is_master:
                return my_results.astype('f4')


    def condiv(self, duration, system_names, param_names, param_tensors, energy_sens=None):
        import run_upside as ru
        with mpi_lock:
            if energy_sens is not None:
                energy_sens = np.asarray(energy_sens)

            if self.mpi_comm is not None and self.is_master:
                # activate the workers with send to match their waiting collectives
                self.mpi_comm.bcast(('condiv', duration, system_names, param_names,param_tensors,energy_sens,), 0)

            rank,size = (self.mpi_comm.rank,self.mpi_comm.size) if self.mpi_comm is not None else (0,1)

            n_sys = len(system_names)
            assert system_names.shape == (n_sys,)
            if energy_sens is not None:
                assert energy_sens.shape == (n_sys,)
            assert len(param_names) == len(param_tensors)

            # we need a parameter hash to know if we are executing on parameters we have used before
            param_hash = tuple(zip(param_names, [hash(x.tostring()) for x in param_tensors]))
            param_dict = dict(zip(param_names, param_tensors))

            if energy_sens is None:
                energy = np.zeros((n_sys,), dtype='f8')
                rmsd   = np.zeros((n_sys,2), dtype='f8')
                bad_solve = np.zeros((n_sys,), dtype='f8')

                for sys_num, nm in enumerate(system_names):
                    if sys_num%size != rank: continue

                    self.condiv_log.append((nm,self.condiv_dict[nm].get_weighted_frames(duration, param_dict)))
                    config_path,target,weights,frames,bs = self.condiv_log[-1][-1]
                    # I will compute RMSD using the sign of weights to control which RMSD to add to
                    trmsd = ru.traj_rmsd(frames,target)
                    restrain_weights = np.where(weights>0,  weights, 0.)
                    free_weights     = np.where(weights<0, -weights, 0.)

                    restrain_weights /= restrain_weights.sum()
                    free_weights /= free_weights.sum()

                    rmsd[sys_num] = (np.dot(trmsd, restrain_weights), np.dot(trmsd, free_weights))
                    bad_solve[sys_num] = bs

                    assert len(weights) == len(frames)
                    inv_n_res = 3./len(target)

                    # store 3 param deriv -- energy_deriv, restrain_rmsd_deriv, free_rmsd_deriv
                    param_deriv = dict((pn,(np.zeros(pt.shape),np.zeros(pt.shape),np.zeros(pt.shape),))
                            for pn,pt in param_dict.items())

                    engine = ue.Upside(config_path)
                    for w,rw,fw,pos,r in zip(weights,restrain_weights, free_weights, frames, trmsd):
                        energy[sys_num] += float(w*engine.energy(pos))*inv_n_res
                        for pnm, pt in zip(param_names, param_tensors):
                            pd = engine.get_param_deriv(pt.shape, pnm)
                            pd_energy, pd_rrestrain, pd_rfree = param_deriv[pnm]

                            pd_energy    += (w*inv_n_res)*pd
                            pd_rrestrain += (r*rw)*pd
                            pd_rfree     += (r*fw)*pd
                    self.condiv_deriv_cache[nm] = (param_hash, param_deriv)

                if self.mpi_comm is not None:
                    numpy_reduce_inplace(self.mpi_comm, [energy,rmsd,bad_solve])
                if self.is_master:
                    if len(energy): print 'mean energy gap %5.3f rmsd %5.3f' % (energy.mean(), rmsd[:,1].mean())
                    return energy.astype('f4'), rmsd.astype('f4'), bad_solve.astype('f4')
            else:
                # We are in derivative mode
                param_deriv = dict((pn,np.zeros(pt.shape, dtype='f8')) for pn,pt in param_dict.items())
                n_found = np.zeros(n_sys, dtype='i4')
                for sys_num, nm in enumerate(system_names):
                    if nm in self.condiv_deriv_cache:
                        cache_param_hash, cache_param_deriv = self.condiv_deriv_cache[nm]
                        if cache_param_hash != param_hash: continue
                        n_found[sys_num] += 1
                        for pnm in param_names:
                            pd_energy, pd_rrestrain, pd_rfree = cache_param_deriv[pnm]

                            param_deriv[pnm] += energy_sens[sys_num]*pd_energy
                            param_deriv[pnm] += rmsd_sens[sys_num,0]*pd_rrestrain +
                            param_deriv[pnm] += rmsd_sens[sys_num,1]*pd_rfree)

                if self.mpi_comm is not None:
                    numpy_reduce_inplace(self.mpi_comm, [n_found] + list(param_deriv.values()))

                if self.is_master:
                    if not np.all(n_found==1):  # we must find each system exactly once
                        raise ValueError('missing systems, maybe value not called before derivative')
                    return [param_deriv[pnm].astype('f4') for pnm in param_names]


    def evaluate(self, system_names, param_names, param_tensors, energy_sens=None):
        with mpi_lock:
            if energy_sens is not None:
                energy_sens = np.asarray(energy_sens)

            if self.mpi_comm is not None and self.is_master:
                # activate the workers with send to match their waiting collectives
                self.mpi_comm.bcast(('evaluate', system_names, param_names,param_tensors,energy_sens,))

            n_sys = len(system_names)
            assert system_names.shape == (n_sys,)
            if energy_sens is not None:
                assert energy_sens.shape == (n_sys,)

            if not self.mpi_comm.rank:
                try:
                    assert len(param_names) == len(param_tensors)
                except AssertionError:
                    print 'BAD', len(param_names), len(param_tensors)
                    print 'system_names', system_names[:3]
                    print 'param names', param_names[:3]
                    print 'param tensors', [x.shape for x in param_tensors][:3]
                    print 'energy_sens', energy_sens
                    print
                    raise AssertionError

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
