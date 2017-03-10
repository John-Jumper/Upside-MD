#!/usr/bin/env python

import tables as tb
import cPickle as cp
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file to modify with target information')
    parser.add_argument('--replace', default=False, action='store_true', help='replace target information with new target')
    parser.add_argument('--target-structure', default='', required=True,
            help='Target .initial.pkl structure')
    args = parser.parse_args()

    with tb.open_file(args.config,'a') as t:
        if args.replace and 'target' in t.root:
            t.root.target._f_remove(recursive=True)
        pos = cp.load(open(args.target_structure))
        assert pos.shape == t.root.input.pos.shape
        g = t.create_group(t.root,'target')
        t.create_array(g, 'pos', obj=pos[:,:,0])

if __name__ == '__main__':
    main()
