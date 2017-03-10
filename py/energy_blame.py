#!/usr/bin/env python
import numpy as np
import tables as tb
import cPickle as cp
import run_upside as ru
import sys
import pandas as pd
import os
import re

np.set_printoptions(precision=3,suppress=True)

three_letter_aa = dict(
        A='ALA', C='CYS', D='ASP', E='GLU',
        F='PHE', G='GLY', H='HIS', I='ILE',
        K='LYS', L='LEU', M='MET', N='ASN',
        P='PRO', Q='GLN', R='ARG', S='SER',
        T='THR', V='VAL', W='TRP', Y='TYR')
restypes = np.array(sorted(three_letter_aa.values()))

def seq_to_1hot(seq):
    ret = np.column_stack([1.*(seq==rt) for rt in restypes])
    assert ret.sum() == len(seq)
    return ret

def read_traj(s, path):
    d=dict()
    with tb.open_file(path) as t:
        o = t.root.output
        d['seq']    = t.root.input.sequence[:]
        d['seq'][d['seq']=='CPR'] = 'PRO'
        print 'n_res', len(d['seq'])
        d['pos']    = o.pos[s:,0]
        d['pot']    = o.potential[s:,0]
        d['strain'] = o.rotamer_1body_energy0[s:]
        d['cov']    = o.rotamer_1body_energy1[s:]
        d['hydro']  = o.rotamer_1body_energy2[s:]
        d['sc']     = o.rotamer_free_energy  [s:]
        d['pair']   = d['sc'] - d['strain'] - d['cov'] #- d['env']
        d['rama']   = o.rama_map_potential[s:]
        d['hb']     = read_hb(t)[s:]
        d['Rg']     = np.sqrt(np.var(d['pos'],axis=1).sum(axis=-1))
        phe = t.root.input.potential.hbond_energy._v_attrs.protein_hbond_energy
        d['hb_energy'] = phe*d['hb'][...,0].sum(axis=-1)
        d['env']    = o.nonlinear_coupling[s:]

    return d


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

terms = 'restype pair cov hydro rama strain hb env'.split()
def extract_blame(native_fname, denovo_fname, rmsd_cutoff, denovo_equil_fraction):
    try:
        d = read_traj(0, native_fname)  # skip first frame due to weird rmsd
        print os.path.basename(native_fname),
    except tb.NoSuchNodeError:
        return None
    if denovo_fname:
        try:
            dd = read_traj(0, denovo_fname)  # skip first frame due to weird rmsd
            for k in d:
                if k != 'seq':
                    n_skip = int(dd[k].shape[0]*denovo_equil_fraction+0.999)  # round up
                    d[k] = np.concatenate([d[k],dd[k][n_skip:]],axis=0)
            print os.path.basename(denovo_fname)
        except tb.NoSuchNodeError:
            print
            pass
    else:
        print
    k = 9
    rmsd = ru.traj_rmsd(d['pos'][1:,k:-k], d['pos'][0,k:-k])
    good = rmsd<rmsd_cutoff
    bad  = np.logical_not(good) # & (rmsd<8.)
    if not good.sum() or not bad.sum(): return None

    s = seq_to_1hot(d['seq'])/len(d['seq'])
    
    f = lambda x: np.dot(x[good].mean(axis=0),s) - np.dot(x[bad].mean(axis=0),s)
    c = np.column_stack([f(x[1:]) for x in (
        d['pair'], d['cov'], d['hydro'], d['rama'], d['strain'], d['hb_energy'], d['env'])])
    df = pd.DataFrame.from_records([(s,)+tuple(r) for s,r in zip(restypes,c)], columns=terms)
    df['file'] = [os.path.splitext(os.path.basename(native_fname))[0]]*len(df)
    return df



def plot_energies(df, fname):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    ax1 = plt.subplot2grid((1,3),(0,0),colspan=2)
    ax2 = plt.subplot2grid((1,3),(0,2))

    colors = []
    for t in terms[1:]:
        p = ax1.plot(df[t], label=t)[0]
        colors.append(p.get_color())  # make the bar plots the same colors

    ax1.set_ylabel('energy/residue')
    ax1.set_xticks(np.arange(len(restypes)))
    ax1.set_xticklabels(list(restypes), rotation=90)

    # on second axes, we will plot the overall sums
    sum_interact = df.sum()
    x = np.arange(len(sum_interact))
    for i in x:
        b = ax2.bar(x[i:i+1], list(sum_interact)[i:i+1], label=list(sum_interact.index)[i],)
        b[0].set_color(colors[i])
    ax2.set_xticks(x+0.5)
    ax2.set_xticklabels(list(sum_interact.index), rotation=90)

    plt.tight_layout()
    plt.savefig(fname)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', default='', help='filename for plotting')
    parser.add_argument('--dump-csv', default='', help='filename to dump csv')
    parser.add_argument('--rmsd-cutoff', default=4., type=float, help='RMSD cutoff for good structure (default 4.)')
    parser.add_argument('--de-novo-equil-fraction', default=0.5, type=float, help='fraction of de novo simulation to exclude as equilibration (default 0.5)')
    parser.add_argument('--native-pattern', default='native', help='regex pattern to substitute from to convert native simulation filenames to de novo simulation filenames (default "native")')
    parser.add_argument('--de-novo-pattern', default=None, help='regex pattern to substitute to convert native simulation filenames to de novo simulation filenames (e.g. "denovo")')
    parser.add_argument('native_files', nargs='+', help='file paths for simulations from native.  Blame will be averaged across files')
    args = parser.parse_args(sys.argv[1:])

    recs = []
    for fn in args.native_files:
        native = fn
        denovo = None if args.de_novo_pattern is None else re.sub(args.native_pattern, args.de_novo_pattern, native)
        recs.append(extract_blame(native, denovo, args.rmsd_cutoff, args.de_novo_equil_fraction))
    recs = [x for x in recs if x is not None]
    df = pd.concat(recs)  #.to_csv(index=False)
    if args.dump_csv: df.to_csv(args.dump_csv, index=False)
    # dfm = df.groupby('restype').median()
    dfm = df.groupby('restype').mean()
    pd.options.display.float_format=lambda x:"% .3f"%x
    print 10*dfm
    print
    print 10*dfm.sum()
    print 
    print 'total', dfm.sum().sum()
    if args.plot: plot_energies(dfm, args.plot)

    
if __name__ == '__main__':
    main()


