#!/usr/bin/env python

import tables

def summarize(node, indentation=0, min_name_length=0):
    print '%s%-*s' % (' ' * indentation, min_name_length, node._v_name),
    if 'shape' in dir(node):
        print '%s %s' % (node.shape, node.dtype)
    else:
        print

    indentation += 4

    attr_names = sorted(node._v_attrs._v_attrnamesuser)
    max_len = max([0] + [len(a) for a in attr_names])

    for nm in attr_names:
        print "%s%-*s := %s" % (' ' * indentation, max_len, nm, node._v_attrs[nm])

    if '_v_children' in dir(node):
        if attr_names: print  # extra space to separate the attr_list and the elements
        max_len = max([0] + [len(a) for a in node._v_children.keys()])
        for nm,child in sorted(node._v_children.items()):
            summarize(child, indentation=indentation, min_name_length=max_len)
        print  # extra space after groups

    indentation -= 4

def main():
    import sys
    fnames = sys.argv[1:]
    for fn in fnames:
        tbl = tables.open_file(fn)
        summarize(tbl.root)
        tbl.close()

if __name__ == '__main__':
    main()
