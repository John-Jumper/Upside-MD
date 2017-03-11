import numpy as np
import ctypes as ct
import shutil
import tables as tb
import os
import time

# We have to do a dirty trick to get the correct path to libupside.so
# It is likely possible to modify the build system to drop a config file 
# that contains the necessary paths.  Since I do a lot of work concurrently
# in the source tree, I don't want to depend on the build system to copy the
# python executables into obj/.  This would make it easy not to commit changes
# back when live-editing.

# FIXME This could fail if the user links directly into the py directory
# instead of going through upside/py.  
py_source_dir = os.path.dirname(__file__)
obj_dir = os.path.join(py_source_dir, '..', 'obj')
calc = ct.cdll.LoadLibrary(os.path.join(obj_dir, 'libupside.so'))

calc.construct_deriv_engine.restype  = ct.c_void_p
calc.construct_deriv_engine.argtypes = [ct.c_int, ct.c_char_p, ct.c_bool]

calc.free_deriv_engine.restype  = None
calc.free_deriv_engine.argtypes = [ct.c_void_p]

calc.evaluate_energy.restype  = ct.c_int
calc.evaluate_energy.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]

calc.evaluate_deriv.restype  = ct.c_int
calc.evaluate_deriv.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]

calc.set_param.restype  = ct.c_int
calc.set_param.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_char_p]

calc.get_param_deriv.restype  = ct.c_int
calc.get_param_deriv.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_char_p]

calc.get_param.restype  = ct.c_int
calc.get_param.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_char_p]

calc.get_output_dims.restype  = ct.c_int
calc.get_output_dims.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_char_p]

calc.get_sens.restype  = ct.c_int
calc.get_sens.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_char_p]

calc.get_output.restype  = ct.c_int
calc.get_output.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_char_p]

calc.get_value_by_name.restype  = ct.c_int
calc.get_value_by_name.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_char_p, ct.c_char_p]

calc.get_clamped_value_and_deriv.restype  = ct.c_int
calc.get_clamped_value_and_deriv.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p]

calc.clamped_spline_value.restype  = ct.c_int
calc.clamped_spline_value.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p]

calc.clamped_spline_solve.restype  = ct.c_int
calc.clamped_spline_solve.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p]

calc.get_clamped_coeff_deriv.restype  = ct.c_int
calc.get_clamped_coeff_deriv.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_float]


def in_process_upside(args):
    # A bit of paranoia to make sure this doesn't interfere with python handlers
    # I more sophisticated strategy would modify the signal handlers to interoperate with
    # the Python handlers
    # Additionally, add a first argument because the program name is expected in argv[0]
    exec_args = ['python_library', '--disable-signal-handler'] + args
    exec_c_strs = [ct.c_char_p(s) for s in exec_args]

    # c_str_length = 1+max(len(a) for a in args)
    # print 'arg length %i' % c_str_length

    # build the strings into a char** for C
    c_arr_t = (ct.c_char_p * len(exec_args))
    c_arr = c_arr_t()
    for i,a in enumerate(exec_c_strs):
        c_arr[i] = a

    calc.upside_main.restype  = ct.c_int
    # calc.upside_main.argtypes = [ct.c_int, ct.POINTER(ct.c_char_p)]
    calc.upside_main.argtypes = [ct.c_int, c_arr_t]

    return calc.upside_main(len(exec_args), c_arr)


def clamped_spline_value(bspline_coeff, x):
    x = np.require(x, dtype='f4', requirements='C')
    assert len(x.shape) == 1

    bspline_coeff = np.require(bspline_coeff, dtype='f4', requirements='C')
    assert len(bspline_coeff.shape) == 1

    result = np.zeros(len(x), dtype='f4')

    if calc.clamped_spline_value(
            len(bspline_coeff),
            result.ctypes.data, 
            bspline_coeff.ctypes.data,
            len(x),
            x.ctypes.data): raise RuntimeError("spline evaluation error")
    return result

def clamped_spline_solve(values):
    values = np.require(values, dtype='f4', requirements='C')
    assert len(values.shape) == 1

    bspline_coeff = np.zeros(len(values)+2, dtype='f4')

    if calc.clamped_spline_solve(
            len(bspline_coeff),
            bspline_coeff.ctypes.data,
            values.ctypes.data): raise RuntimeError("spline solve error")
    return bspline_coeff

def clamped_value_and_deriv(bspline_coeff, x):
    x = np.require(x, dtype='f4', requirements='C')
    assert len(x.shape) == 1

    bspline_coeff = np.require(bspline_coeff, dtype='f4', requirements='C')
    assert len(bspline_coeff.shape) == 1

    result = np.zeros((len(x),2), dtype='f4')

    if calc.get_clamped_value_and_deriv(
            len(bspline_coeff),
            result.ctypes.data, 
            bspline_coeff.ctypes.data,
            len(x),
            x.ctypes.data): raise RuntimeError("spline evaluation error")
    return result


def clamped_coeff_deriv(bspline_coeff, x):
    x = np.asarray(x, dtype='f4')
    assert len(x.shape) == 1

    bspline_coeff = np.require(bspline_coeff, dtype='f4', requirements='C')
    assert len(bspline_coeff.shape) == 1

    result = np.zeros((len(x),len(bspline_coeff)), dtype='f4')

    for i,y in enumerate(x):
        if calc.get_clamped_coeff_deriv(
                len(bspline_coeff),
                result[i].ctypes.data, 
                bspline_coeff.ctypes.data,
                y): raise RuntimeError("spline evaluation error")
    return result


class Upside(object):
    def __init__(self, config_file_path, quiet=True):
        self.config_file_path = str(config_file_path)
        with tb.open_file(self.config_file_path) as t:
            self.initial_pos = t.root.input.pos[:,:,0]
            self.n_atom = self.initial_pos.shape[0]
            self.sequence = t.root.input.sequence[:]
        self.engine = calc.construct_deriv_engine(self.n_atom, self.config_file_path, bool(quiet))
        if self.engine is None: raise RuntimeError('Unable to initialize upside engine')

    def __repr__(self):
        return 'Upside(%r, %r)'%(self.n_atom, self.config_file_path)

    def energy(self, pos):
        pos = np.require(pos, dtype='f4', requirements='C')
        assert pos.shape == (self.n_atom,3)
        energy = np.zeros(1, dtype='f4')
        retcode = calc.evaluate_energy(energy.ctypes.data, self.engine, pos.ctypes.data)
        if retcode: raise RuntimeError('Unable to evaluate energy')
        return energy[0]

    def deriv(self, pos):
        pos = np.require(pos, dtype='f4', requirements='C')
        assert pos.shape == (self.n_atom,3)
        deriv = np.zeros_like(pos)
        retcode = calc.evaluate_deriv(deriv.ctypes.data, self.engine, pos.ctypes.data)
        if retcode: raise RuntimeError('Unable to evaluate derivative')
        return deriv

    def set_param(self, param, node_name):
        param_size = param.shape
        param = np.require(param.ravel(), dtype='f4', requirements='C')  # flatten and make contiguous
        retcode = calc.set_param(int(param.shape[0]), param.ctypes.data, self.engine, node_name)
        if retcode: raise RuntimeError('Unable to set param with size %s for node %s'%(param_size,node_name))

    def get_param_deriv(self, param_shape, node_name):
        n_param = int(np.prod(param_shape))
        deriv = np.zeros(param_shape, dtype='f4')
        retcode = calc.get_param_deriv(n_param, deriv.ctypes.data, self.engine, node_name)
        if retcode: raise RuntimeError('Unable to get param deriv')
        return deriv

    def get_param(self, param_shape, node_name):
        n_param = int(np.prod(param_shape))
        param = np.zeros(param_shape, dtype='f4')
        retcode = calc.get_param(n_param, param.ctypes.data, self.engine, node_name)
        if retcode: raise RuntimeError('Unable to get param')
        return param

    def get_sens(self, node_name):
        n_elem = np.zeros(1,dtype=np.intc)
        elem_width = np.zeros(1,dtype=np.intc)
        retcode = calc.get_output_dims(n_elem.ctypes.data, elem_width.ctypes.data, self.engine, node_name)
        if retcode: raise RuntimeError('Unable to get output dims')
        output_shape = (int(n_elem[0]), int(elem_width[0]))

        n_output = int(np.prod(output_shape))
        output = np.zeros(output_shape, dtype='f4')
        retcode = calc.get_sens(n_output, output.ctypes.data, self.engine, node_name)
        if retcode: raise RuntimeError('Unable to get output')
        return output

    def get_output(self, node_name):
        n_elem = np.zeros(1,dtype=np.intc)
        elem_width = np.zeros(1,dtype=np.intc)
        retcode = calc.get_output_dims(n_elem.ctypes.data, elem_width.ctypes.data, self.engine, node_name)
        if retcode: raise RuntimeError('Unable to get output dims')
        output_shape = (int(n_elem[0]), int(elem_width[0]))

        n_output = int(np.prod(output_shape))
        output = np.zeros(output_shape, dtype='f4')
        retcode = calc.get_output(n_output, output.ctypes.data, self.engine, node_name)
        if retcode: raise RuntimeError('Unable to get output')
        return output

    def get_value_by_name(self, value_shape, node_name, log_name):
        n_param = int(np.prod(value_shape))
        value = np.zeros(value_shape, dtype='f4')
        retcode = calc.get_value_by_name(n_param, value.ctypes.data, self.engine, node_name, log_name)
        if retcode: raise RuntimeError('Unable to get value by name')
        return value

    def __del__(self):
        calc.free_deriv_engine(self.engine)

def get_rotamer_graph(engine):
    n_node, n_edge = engine.get_value_by_name((2,),         'rotamer', 'graph_nodes_edges_sizes').astype('i')
    node_prob      = engine.get_value_by_name((n_node,3),   'rotamer', 'graph_node_prob')
    edge_prob      = engine.get_value_by_name((n_edge,3,3), 'rotamer', 'graph_edge_prob')
    edge_indices   = engine.get_value_by_name((n_edge,2),   'rotamer', 'graph_edge_indices').astype('i')
    return node_prob, edge_prob, edge_indices

def freeze_nodes(new_h5_path, old_h5_path, nodes_to_freeze, additional_nodes_to_delete, quiet=False):
    '''Replace computation nodes with constant nodes that give the same answer on the initial structure'''
    shutil.copyfile(old_h5_path, new_h5_path)

    engine = Upside(old_h5_path)
    pos = engine.initial_pos

    en = engine.energy(pos)  # required to fill output
    if not quiet: print 'energy', en
    freeze = dict((nm,engine.get_output(nm)) for nm in nodes_to_freeze)

    with tb.open_file(new_h5_path, 'a') as tn:
        for nm in list(nodes_to_freeze) + list(additional_nodes_to_delete):
            tn.get_node('/input/potential/%s'%nm)._f_remove(recursive=True)

        for nm,value in freeze.items():
            g = tn.create_group(tn.root.input.potential, 'constant_'+nm)
            g._v_attrs.arguments = []
            tn.create_array(g, 'value', obj=value)

        for node in tn.root.input.potential:
            node._v_attrs.arguments = np.array(
                    [('constant_'+nm if nm in freeze else nm) 
                        for nm in node._v_attrs.arguments])

    new_en = Upside(new_h5_path).energy(pos)
    if not quiet: print 'new_energy', new_en
