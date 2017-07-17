#!/usr/bin/env python

from multiprocessing import Pool

import numpy as np
import tables as tb
import collections
from glob import glob
import re
import sys,os
import cPickle as cp
from glob import glob
import pandas as pd
from gzip import GzipFile
import time

upside_dir = os.path.expanduser('~/upside/')

if upside_dir + 'src' not in sys.path: sys.path = [upside_dir+'src'] + sys.path
import run_upside as ru

deg = np.pi/180.

def process_file(a):
    x,skip,equil_fraction,do_traj = a
    protein = os.path.basename(x).split('_')[0]

    for n_try in range(3):
        try:
            with tb.open_file(x) as t:
                # print t.root._v_children.keys(), 'output' in t.root, 'output_previous_0' in t.root
                output_names = []
                i = 0
                while 'output_previous_%i'%i in t.root:
                    output_names.append('output_previous_%i'%i)
                    i += 1
                if 'output' in t.root: 
                    output_names.append('output')
                if not output_names:
                    return None

                last_time = 0.

                df_list = []
                for onm in output_names:
                    sl = slice(skip,None,skip)
                    n = t.get_node('/'+onm)
                    sim_time = n.time[sl] + last_time
                    last_time = sim_time[-1]

                    pos=n.pos[sl,0]
                    pot=n.potential[sl,0]
                    T=n.temperature[0,0]

                    df = pd.DataFrame(dict(
                        time=sim_time,
                        energy=pot,
                        N_res = pos.shape[1]//3,
                        protein=protein,
                        initial="init_"+str(t.root.input.args._v_attrs.initial_structures),
                        T=T+np.zeros_like(pot),
                        Temp=np.array(['T=%.3f'%T]*len(sim_time)),
                        HBond=0.5*(n.hbond[sl]>0.05).sum(axis=1),  # 0.5 takes care of double counting
                        Rg = np.sqrt(np.var(pos,axis=1).sum(axis=-1)),
                        ))

                    df['RMSD'] = ru.traj_rmsd(pos[:,9:-9],t.root.target.pos[9:-9])

                    if do_traj:
                        # copy in the position with the object dtype 
                        df['pos'] = pd.Series(list(pos[:,1::3].astype('f4').copy()), dtype=np.object) 

                    if 'replica_index' in n:
                        df['replica'] = n.replica_index[sl,0]
                        df['method']  = 'replex'
                    else:
                        df['replica'] = 0
                        df['method']  = 'constantT'
                    print x, onm
                    df_list.append(df)
                df = pd.concat(df_list)

                df['filename'] = x
                df['frame'] = np.arange(len(df['time']))
                df['phase'] = np.where(((df['frame']<df['frame'].max()*equil_fraction).as_matrix()),
                        'equilibration','production')
                return df
        except Exception as e:
            print e
            # There is a decent chance that the exception is due to Upside writing to the .h5 concurrently
            # We will try again after waiting to allow the .h5 to get a consistent state
            time.sleep(2)  
            continue

    print x, 'FAILURE'
    return None

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-j', default=1, type=int, help = 'number of processes to use')
    parser.add_argument('--output-csv-gz',  required=True, help='Path to output compressed CSV output')
    parser.add_argument('--output-traj-h5', default='',    help='Path to output trajectories in .h5 format')
    parser.add_argument('--skip', default=32, type=int,    help='Analyze every n-th frame (default 32)')
    parser.add_argument('--equil-fraction', default=1./3., type=float,
            help='Fraction of simulation to call equilibration (default 0.333)')
    parser.add_argument('--exclude-pattern', default='', 
            help='regular expression pattern to exclude configs from analysis')
    parser.add_argument('configs', nargs='+', help='Upside trajectories to analyze')
    args = parser.parse_args()

    print args.configs
    do_traj = bool(args.output_traj_h5)

    if args.exclude_pattern:
        configs = [x for x in args.configs if not re.search(args.exclude_pattern, x)]
    else:
        configs = list(args.configs)
        
    pool = Pool(processes=args.j)
    all_output = list(pool.map(process_file, [(c,args.skip,args.equil_fraction,do_traj) for c in configs]))
    df = pd.concat([x for x in all_output if x is not None], ignore_index=True)
    print 'number of read failures', len([x for x in all_output if x is None])
    print df.index

    if do_traj:
        import tables as tb
        with tb.open_file(args.output_traj_h5,'w') as t:
            filt = tb.Filters(complib='zlib', complevel=5, fletcher32=True)

            for protein, df_protein in df.groupby('protein'):
                print 'traj', protein
                g = t.create_group(t.root, protein)
                t.create_earray(g, 'traj',  obj=np.array(list(df_protein.pos), dtype='f4'), filters=filt)
                t.create_earray(g, 'index', obj=np.array(df_protein.index.values, dtype='i4'), filters=filt)
        del df['pos'] # do not put position in CSV file
    
    print 'CSV output'
    with GzipFile(args.output_csv_gz,'wt') as f:
        df.to_csv(f)


if __name__ == '__main__':
    main()
