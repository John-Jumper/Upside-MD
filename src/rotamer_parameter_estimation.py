import tables as tb
import numpy as np
import tempfile
import subprocess as sp
import os

class RotamerEnergy(object):
    def __init__(self, config_file, upside_path = '../obj', tmp_dir = None):
        self.config_file = config_file
        with tb.open_file(self.config_file) as t:
            self.n_restype = t.root.input.potential.rotamer.energy.shape[0]
        self.upside_path = upside_path
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
            sp.check_output([os.path.join(self.upside_path,"upside"), 
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
            deriv_flat = tuple(x[np.triu_indices_from(x)] for x in d.transpose((2,0,1)))

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






