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
from mpi4py import MPI
import mpi_collective_object as mco

import threading

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

@mco.obj.register
class UpsideEnsemble(object):
    def __init__(self, condiv_dict):
        self.condiv_dict = dict(condiv_dict)
        self.grad_name = None
        obj = self.condiv_dict.values()[0]
        self.n_observable = obj.n_observable
        self.frame_shape = obj.frame_shape

    @mco.obj.collective
    def ensemble_deriv(self,
            weight_sens, observable_sens,
            weights, observables, seq, traj, atom_boundaries,
            system_names, param_names, *param_tensors
            ):
        # derivative of i-th weight is effectively -weight*d_energy, so we can just augment the energy
        # sensitivity which is always column 0 for each system
        augmented_observable_sens = observable_sens.copy()
        augmented_observable_sens[:,0] -= weights*weight_sens 
        param_dict = dict(zip(param_names, param_tensors))
        param_sens = dict((pnm,np.zeros(pval.shape,dtype='f8')) for pnm,pval in param_dict.items())

        for sys_num, nm in enumerate(system_names):
            if sys_num%self.comm.size != self.comm.rank: continue
            sys_traj = traj[...,atom_boundaries[sys_num]:atom_boundaries[sys_num+1],:]

            sys_param_sens = self.condiv_dict[nm].param_sens(
                    param_dict,
                    augmented_observable_sens[sys_num],
                    weights[sys_num],
                    observables[sys_num],
                    sys_traj)
            for pnm in param_sens:
                param_sens[pnm] += sys_param_sens[pnm]

        numpy_reduce_inplace(self.comm, param_sens.values())
        return [param_sens[pnm].astype('f4') for pnm in param_names]

    @mco.obj.collective
    def ensemble(self, system_names, param_names, *param_tensors):
        tstart = time.time()
        n_sys = len(system_names)
        assert system_names.shape == (n_sys,)
        assert len(param_names) == len(param_tensors)
        param_dict = dict(zip(param_names, param_tensors))

        weights      = np.zeros((n_sys,) + self.frame_shape, dtype='f4')
        observables  = np.zeros((n_sys,self.n_observable) + self.frame_shape, dtype='f4')
        traj_list = []
        atom_counts = np.zeros((n_sys,), dtype='i4')

        for sys_num, nm in enumerate(system_names):
            if sys_num%self.comm.size != self.comm.rank: continue
            d = self.condiv_dict[nm]
            weights[sys_num], observables[sys_num], sys_traj,sys_seq = d.get_weighted_frames(param_dict)
            assert sys_traj.shape == self.frame_shape + (d.n_atom,3)
            assert sys_seq.shape == (d.n_atom/3,)
            traj_list.append((sys_num,sys_traj.astype('f4'),np.array(sys_seq,dtype='S3')))
            atom_counts[sys_num] = d.n_atom

        numpy_reduce_inplace(self.comm, [weights,observables,atom_counts])
        traj_list = self.comm.gather(traj_list)

        if not self.comm.rank:
            # flatten traj lists and sort by system number
            traj_list = sorted([t for l in traj_list for t in l])

            # concatenate on atom index
            if n_sys:
                num,traj,seq = zip(*traj_list)
                traj = np.concatenate(traj, axis=-2)
                seq  = np.concatenate(seq)
            else:
                traj = np.zeros(self.frame_shape + (0,3), dtype='f4')

            # Now figure out the atom boundaries of each system
            # so that the traj may be split later
            atom_boundaries = np.zeros(n_sys+1, dtype='i4')
            atom_boundaries[1:] = np.cumsum(atom_counts)

            return weights, observables, seq, traj, atom_boundaries


    def ensemble_tensorflow(self, system_names, param_names, *param_tensors):
        name=None

        from tensorflow.python.framework import ops
        import tensorflow as tf

        args = (system_names,param_names) + tuple(param_tensors)

        if self.grad_name is None:
            self.grad_name = _unique_name('UpsideEnsembleGrad')

            @tf.RegisterGradient(self.grad_name)
            def grad(op, weight_sens, observable_sens, seq_sens, traj_sens, atom_boundaries_sens):
                param_deriv = tf.py_func(
                        self.ensemble_deriv,
                        (weight_sens, observable_sens) + tuple(op.outputs) + tuple(op.inputs),
                        [tf.float32]*len(param_tensors))
                return [None,None] + param_deriv

        with tf.get_default_graph().gradient_override_map(dict(PyFunc=self.grad_name)):
            return tf.py_func(
                    self.ensemble,
                    args,
                    [tf.float32, tf.float32, tf.string, tf.float32, tf.int32],
                    name=name)


@mco.obj.register
class UpsideEnergy(object):
    def __init__(self, n_restype, name_list, path_list, chi1_file_list):
        import pandas as pd

        assert len(name_list) == len(set(name_list)) # no duplicates

        import upside_engine as ue
        self.systems = dict()

        assert len(name_list) == len(path_list) == len(chi1_file_list)
        name_path_list = list(zip(name_list,path_list,chi1_file_list))

        name_path_list = self.comm.scatter(
                [name_path_list[i::self.comm.size] for i in range(self.comm.size)])
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

    def energy_tensorflow(self, system_names, param_names, *param_tensors):
        name=None
        # FIXME this does not use the solution from computing the value to speed 
        #  computing the gradient.  I should figure out how to cache the parameter derivatives
        #  to make this easier.

        from tensorflow.python.framework import ops
        import tensorflow as tf
        args = (system_names,param_names) + param_tensors

        if self.grad_name is None:
            self.grad_name = _unique_name('UpsideEnergyGrad')

            @tf.RegisterGradient(self.grad_name)
            def grad(op, energy_sens, n_res_sens):
                param_deriv = tf.py_func(
                        self.energy,
                        (energy_sens,)+args,
                        [tf.float32]*len(param_tensors))
                return [None,None,None] + param_deriv

        with tf.get_default_graph().gradient_override_map(dict(PyFunc=self.grad_name)):
            return tf.py_func(
                    self.energy,
                    (np.zeros((),dtype='f4'),)+args,
                    [tf.float32, tf.int32],
                    name=name)

    def chi1_loss_tensorflow(self, system_names, param_names, *param_tensors):
        import tensorflow as tf
        return tf.py_func(
                self.chi1_loss,
                (system_names,param_names) + param_tensors,
                tf.float32)

    @mco.obj.collective
    def chi1_loss(self, system_names, param_names, *param_tensors):
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

        numpy_reduce_inplace(self.comm, [my_results])
        if not self.comm.rank:
            return my_results.astype('f4')


    @mco.obj.collective
    def energy(self, energy_sens, system_names, param_names, *param_tensors):
        if energy_sens.shape == ():  # sentinel value for tensorflow needs to be a tensor
            energy_sens = None
        if energy_sens is not None:
            energy_sens = np.asarray(energy_sens)

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

        numpy_reduce_inplace(self.comm,
                [energy,n_res] +
                (list(param_deriv.values()) if energy_sens is not None else []))

        if not self.comm.rank:
            if np.any(n_res==0):
                raise ValueError('missing n_res, maybe missing name in name list')

            if energy_sens is None:
                return energy, n_res
            else:
                return [param_deriv[pnm].astype('f4') for pnm in param_names]
