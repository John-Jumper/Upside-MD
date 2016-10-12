''' Very opinionated convenience script for running upside jobs on midway '''
import collections
import numpy as np
import subprocess as sp
import os, sys
import json,uuid

params_dir = os.path.expanduser('~/upside-parameters/')
upside_dir = os.path.expanduser('~/upside/')


def upside_config(fasta, output, dimer=False, backbone=True, rotamer=True, 
                  sidechain=False, hbond=None, sheet_mix_energy=None, helical_energy_shift=None,
                  sidechain_scale=None, inverse_scale=0., inverse_radius_scale=None, init=None, 
                  rama_pot=params_dir+'rama_libraries.h5',
                  fix_rotamer = '',
                  dynamic_1body = False,
                  environment=None,
                  torus_dbn = None,
                  placement=params_dir+'rotamer-extended-with-direc.h5',
                  reference_rama=None, restraint_groups=[], restraint_spring=None, hbond_coverage_radius=None,
                  rotamer_interaction_param='/home/jumper/optimized_param4_env.h5'):
    
    args = [upside_dir + 'src/upside_config.py', '--fasta=%s'%fasta, '--output=%s'%output]

    if init:
        args.append('--initial-structures=%s'%init)
    if torus_dbn is not None:
        args.append('--torus-dbn-library=%s'%torus_dbn)
    if rama_pot is not None:
        args.append('--rama-library=%s'%rama_pot)
    if sheet_mix_energy is not None:
        args.append('--rama-sheet-mixing-energy=%f'%sheet_mix_energy)
    if helical_energy_shift is not None:
        args.append('--helical-energy-shift=%f'%helical_energy_shift)
    if dimer:
        args.append('--dimer-basin-library=%s'%(params_dir+'TCB_count_matrices.pkl'))
    if not backbone:
        args.append('--no-backbone')
    if hbond:
        args.append('--hbond-energy=%f'%hbond)
    if reference_rama:
        args.append('--reference-state-rama=%s'%reference_rama)
    for rg in restraint_groups:
        args.append('--restraint-group=%s'%rg)
    if restraint_spring is not None:
        args.append('--restraint-spring-constant=%f'%restraint_spring)
        
    if rotamer:
        args.append('--rotamer-placement=%s'%placement)
    if dynamic_1body:
        args.append('--dynamic-rotamer-1body')
    if rotamer:
        args.append('--rotamer-interaction=%s'%rotamer_interaction_param)
    if fix_rotamer:
        args.append('--fix-rotamer=%s'%fix_rotamer)

    if environment:
        args.append('--environment=%s'%environment)
    
    if sidechain:
        args.append('--sidechain-radial=%s'%(params_dir+'radial-MJ-1996.h5'))
    if sidechain_scale is not None: 
        args.append('--sidechain-radial-scale-energy=%f'%sidechain_scale)
    if inverse_scale: 
        args.append('--sidechain-radial-scale-inverse-energy=%f'%inverse_scale)
        args.append('--sidechain-radial-scale-inverse-radius=%f'%inverse_radius_scale)
        
    return ' '.join(args) + '\n' + sp.check_output(args)


def compile():
    return sp.check_output(['/bin/bash', '-c', 'cd %s; make -j4'%(upside_dir+'obj')])


UpsideJob = collections.namedtuple('UpsideJob', 'job config output'.split())


def run_upside(queue, config, duration, frame_interval, n_threads=1, hours=36, temperature=1., seed=None,
               replica_interval=None, anneal_factor=1., anneal_duration=-1., mc_interval=None, 
               time_step = None, swap_sets = None,
               log_level='detailed', account=None):
    if isinstance(config,str): config = [config]
    
    upside_args = [upside_dir+'obj/upside', '--duration', '%f'%duration, '--frame-interval', '%f'%frame_interval] + config

    try:
        upside_args.extend(['--temperature', ','.join(map(str,temperature))])
    except TypeError:  # not iterable
        upside_args.extend(['--temperature', str(temperature)])

    if replica_interval is not None:
        upside_args.extend(['--replica-interval', '%f'%replica_interval])
        for s in swap_sets:
            upside_args.extend(['--swap-set', s])
    if mc_interval is not None:
        upside_args.extend(['--monte-carlo-interval', '%f'%mc_interval])
    if anneal_factor != 1.:
        upside_args.extend(['--anneal-factor', '%f'%anneal_factor])
    if anneal_duration != -1.:
        upside_args.extend(['--anneal-duration', '%f'%anneal_duration])
    upside_args.extend(['--log-level', log_level])
    
    if time_step is not None:
        upside_args.extend(['--time-step', str(time_step)])

    upside_args.extend(['--seed','%li'%(seed if seed is not None else np.random.randint(1<<31))])
    
    output_path = config[0]+'.output'

    if queue == '': 
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(n_threads)
        output_file = open(output_path,'w')
        job = sp.Popen(upside_args, stdout=output_file, stderr=output_file)
    elif queue == 'srun':
        # set num threads carefully so that we don't overwrite the rest of the environment
        # setting --export on srun will blow away the rest of the environment
        # afterward, we will undo the change

        old_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)

        try:
            os.environ['OMP_NUM_THREADS'] = str(n_threads)
            args = ['srun', '--ntasks=1', '--nodes=1', '--cpus-per-task=%i'%n_threads, 
                    '--slurmd-debug=0', '--output=%s'%output_path] + upside_args
            job = sp.Popen(args, close_fds=True)
        finally:
            if old_omp_num_threads is None:
                del os.environ['OMP_NUM_THREADS']
            else:
                os.environ['OMP_NUM_THREADS'] = old_omp_num_threads
    else:
        args = ['sbatch', '-p', queue, '--time=0-%i'%hours, '--ntasks=1', 
                '--cpus-per-task=%i'%n_threads, '--export=OMP_NUM_THREADS=%i'%n_threads,
                '--output=%s'%output_path, '--parsable', '--wrap', ' '.join(upside_args)]
        if account is not None:
            args.append('--account=%s'%account)
        job = sp.check_output(args).strip()

    return UpsideJob(job,config,output_path)


def status(job):
    try:
        job_state = sp.check_output(['/usr/bin/env', 'squeue', '-j', job.job, '-h', '-o', '%t']).strip()
    except sp.CalledProcessError:
        job_state = 'FN'
        
    if job_state == 'PD':
        status = ''
    else:
        status = sp.check_output(['/usr/bin/env','tail','-n','%i'%1, job.output])[:-1]
    return '%s %s' % (job_state, status)


def read_hb(tr):
    n_res = tr.root.input.pos.shape[0]/3
    don_res =  tr.root.input.potential.infer_H_O.donors.id[:,1] / 3
    acc_res = (tr.root.input.potential.infer_H_O.acceptors.id[:,1]-2) / 3
    
    n_hb = tr.root.output.hbond.shape[1]
    hb_raw   = tr.root.output.hbond[:]
    hb = np.zeros((hb_raw.shape[0],n_res,2,2))

    hb[:,don_res,0,0] =    hb_raw[:,:len(don_res)]
    hb[:,don_res,0,1] = 1.-hb_raw[:,:len(don_res)]

    hb[:,acc_res,1,0] =    hb_raw[:,len(don_res):]
    hb[:,acc_res,1,1] = 1.-hb_raw[:,len(don_res):]
    
    return hb

def read_constant_hb(tr, n_res):
    don_res = tr.root.input.potential.infer_H_O.donors.residue[:]
    acc_res = tr.root.input.potential.infer_H_O.acceptors.residue[:]
    
    n_hb = tr.root.output.hbond.shape[2]
    hb_raw   = tr.root.output.hbond[:,0]

    hb = np.zeros((hb_raw.shape[0],n_res,2,3))

    hb[:,don_res,0,0] = hb_raw[:,:len(don_res),0]
    hb[:,don_res,0,1] = hb_raw[:,:len(don_res),1]
    hb[:,don_res,0,2] = 1.-hb_raw[:,:len(don_res)].sum(axis=-1)

    hb[:,acc_res,1,0] = hb_raw[:,len(don_res):,0]
    hb[:,acc_res,1,1] = hb_raw[:,len(don_res):,1]
    hb[:,acc_res,1,2] = 1.-hb_raw[:,len(don_res):].sum(axis=-1)
    
    return hb
    


def rmsd_transform(target, model):
    assert target.shape == model.shape == (model.shape[0],3)
    base_shift_target = target.mean(axis=0)
    base_shift_model  = model .mean(axis=0)
    
    target = target - target.mean(axis=0)
    model = model   - model .mean(axis=0)

    R = np.dot(target.T, model)
    U,S,Vt = np.linalg.svd(R)
    if np.linalg.det(np.dot(U,Vt))<0.:
        Vt[-1] *= -1.  # fix improper rotation
    rot = np.dot(U,Vt)
    shift = base_shift_target - np.dot(rot, base_shift_model)
    return rot, shift


def structure_rmsd(a,b):
    rot,trans = rmsd_transform(a,b)
    diff = a - (trans+np.dot(b,rot.T))
    return np.sqrt((diff**2).sum(axis=-1).mean(axis=-1))


def traj_rmsd(traj, native):
    return np.array([structure_rmsd(x,native) for x in traj])


def vmag(x):
    assert x.shape[-1] == 3
    return np.sqrt(x[...,0]**2 + x[...,1]**2 + x[...,2]**2)


def vhat(x):
    return x / vmag(x)[...,None]


def compact_sigmoid(x, sharpness):
    y = x*sharpness;
    result = 0.25 * (y+2) * (y-1)**2
    result = np.where((y< 1), result, np.zeros_like(result))
    result = np.where((y>-1), result, np.ones_like (result))
    return result


def compute_topology(t):
    seq = t.root.input.sequence[:]
    infer = t.root.input.potential.infer_H_O
    n_donor = infer.donors.id.shape[0]
    n_acceptor = infer.acceptors.id.shape[0]
    id = np.concatenate((infer.donors.id[:],infer.acceptors.id[:]), axis=0)
    bond_length = np.concatenate((infer.donors.bond_length[:],infer.acceptors.bond_length[:]),axis=0)
    
    def augment_pos(pos, id=id, bond_length=bond_length):
        prev = pos[id[:,0]]
        curr = pos[id[:,1]]
        nxt  = pos[id[:,2]]
        
        virtual = curr + bond_length[:,None] * vhat(vhat(curr-nxt) + vhat(curr-prev))
        new_pos = np.concatenate((pos,virtual), axis=0)
        return json.dumps([map(float,x) for x in new_pos])  # convert to json form
    
    n_atom = 3*len(seq)
    backbone_names = ['N','CA','C']
    
    backbone_atoms = [dict(name=backbone_names[i%3], residue_num=i/3, element=backbone_names[i%3][:1]) 
                      for i in range(n_atom)]
    virtual_atoms  = [dict(name=('H' if i<n_donor else 'O'), residue_num=int(id[i,1]/3), 
                           element=('H' if i<n_donor else 'O'))
                     for i in range(n_donor+n_acceptor)]
    backbone_bonds = [[i,i+1] for i in range(n_atom-1)]
    virtual_bonds  = [[int(id[i,1]), n_atom+i] for i in range(n_donor+n_acceptor)]
    
    topology = json.dumps(dict(
        residues = [dict(resname=str(s), resid=i) for i,s in enumerate(seq)],
        atoms = backbone_atoms + virtual_atoms,
        bonds = backbone_bonds + virtual_bonds,
    ))
    
    return topology, augment_pos


def display_structure(topo_aug, pos, size=(600,600)):
    import IPython.display as disp
    id_string = uuid.uuid4()
    return disp.Javascript(lib='/files/js/protein-viewer.js', 
                    data='render_structure(element, "%s", %i, %i, %s, %s);'%
                       (id_string, size[0], size[1], topo_aug[0], topo_aug[1](pos))), id_string

def swap_table2d(nx,ny):
    idx = lambda xy: xy[0]*ny + xy[1]
    good = lambda xy: (0<=xy[0]<nx and 0<=xy[1]<ny)
    swap = lambda i,j: '%i-%i'%(idx(i),idx(j)) if good(i) and good(j) else None
    horiz0 = [swap((a,b),(a+1,b)) for a in range(0,nx,2) for b in range(0,ny)]
    horiz1 = [swap((a,b),(a+1,b)) for a in range(1,nx,2) for b in range(0,ny)]
    vert0  = [swap((a,b),(a,b+1)) for a in range(0,nx)   for b in range(0,ny,2)]
    vert1  = [swap((a,b),(a,b+1)) for a in range(0,nx)   for b in range(1,ny,2)]
    sets = (horiz0,horiz1,vert0,vert1)
    sets = [[y for y in x if y is not None] for x in sets]
    return [','.join(x) for x in sets if x]
