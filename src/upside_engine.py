import numpy as np
import ctypes as ct

calc = ct.cdll.LoadLibrary('/home/jumper/Dropbox/code/upside/obj/libupside_calculation.so')

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

class Upside(object):
    def __init__(self, n_atom, config_file_path, quiet=True):
        self.config_file_path = str(config_file_path)
        self.n_atom = int(n_atom)
        self.engine = calc.construct_deriv_engine(self.n_atom, self.config_file_path, bool(quiet))
        if self.engine is None: raise RuntimeError('Unable to initialize upside engine')

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
        param = np.require(param.ravel(), dtype='f4', requirements='C')  # flatten and make contiguous
        retcode = calc.set_param(int(param.shape[0]), param.ctypes.data, self.engine, node_name)
        if retcode: raise RuntimeError('Unable to set param (was upside compiled for parameter derivatives?)')

    def get_param_deriv(self, param_shape, node_name):
        n_param = int(np.prod(param_shape))
        deriv = np.zeros(param_shape, dtype='f4')
        retcode = calc.get_param_deriv(n_param, deriv.ctypes.data, self.engine, node_name)
        if retcode: raise RuntimeError('Unable to get param deriv (was upside compiled for parameter derivatives?)')
        return deriv

    def get_param(self, param_shape, node_name):
        n_param = int(np.prod(param_shape))
        param = np.zeros(param_shape, dtype='f4')
        retcode = calc.get_param(n_param, param.ctypes.data, self.engine, node_name)
        if retcode: raise RuntimeError('Unable to get param (was upside compiled for parameter derivatives?)')
        return param

    def __del__(self):
        calc.free_deriv_engine(self.engine)
