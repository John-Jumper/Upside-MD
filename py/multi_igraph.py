import os, sys, re
import numpy as np
import tables as tb
sys.path.append(os.path.expanduser("~/upside/py/"))
from upside_config import write_cavity_radial

def add_suffix(t, suffix):
    for group in t.walk_groups():
        if group._v_name not in ["/", "input", "chain_break", "pivot_moves", "potential",
                                 "cavity_radial", "acceptors", "donors", "pair_interaction"]:
            # Regexp pattern to check for existing suffix:
            pattern = r"_\d+$"
            search_obj = re.search(pattern, group._v_name)

            if not search_obj:
                group._f_rename(group._v_name + suffix)

                # Add suffix to arguments as well and change pos args to slice
                try:
                    attrs_args = group._v_attrs.arguments
                except AttributeError:
                    continue
                else:
                    attrs_args_new = []
                    for arg in attrs_args:
                        if arg == "pos":
                            attrs_args_new.append("slice" + suffix)
                        else:
                            attrs_args_new.append(arg + suffix)
                    group._v_attrs.arguments = np.array(attrs_args_new)

def main(h5files):
    """Combine Upside .h5 config files such that they have separate interaction graphs
    
    Args:
        h5files (:obj:`list` of :obj:`str`): .h5 file path list with base combined file as first
                                            elem, seperate chain files, and output as the last elem

    Note: must run Upside with "--log-level basic" due to hardcoded logger names leading to the
    separate graph nodes conflicting over the same logger  
    """
    assert len(h5files) > 3 # need at least three input files and one output
    fin_list = h5files[:-1]
    fout = h5files[-1]

    nres_list = [0]

    with tb.open_file(fout, 'w') as t:
        # Use the first file as the base by copying over the input group

        for i, fin in enumerate(fin_list):
            with tb.open_file(fin) as tin:

                # Use the first combined file as the base by copying over the input group
                if i == 0:
                    tin.root.input._f_copy(t.root, recursive=True)

                    # Remove nodes that will be spliced into seperate igraphs
                    t.root.input.args._f_remove(recursive=True)
                    for group_name in tin.root.input.potential._v_groups:
                        if group_name != "cavity_radial":
                            grp = t.get_node("/input/potential/%s" % group_name)
                            grp._f_remove(recursive=True)
                else:
                    nres_list.append(len(tin.root.input.sequence[:]))

                    for group_name in tin.root.input._v_groups:
                        if group_name not in ["potential", "pivot_moves"]:
                            grp = tin.get_node("/input/%s" % group_name)
                            grp._f_copy(t.root.input, recursive=True)
                    for group_name in tin.root.input.potential._v_groups:
                        if group_name != "cavity_radial":
                            grp = tin.get_node("/input/potential/%s" % group_name)
                            grp._f_copy(t.root.input.potential, recursive=True)
                    add_suffix(t, "_%d" % (i-1))

                    # FIXME: append pivot_moves from sep chains instead of using base combo
        
        # Create slice nodes (computational node == h5 group) with arrays of atom
        # indices from which the separate graphs are built
        for i, nres in enumerate(nres_list[1:]):
            g = t.create_group("/input/potential", "slice_%d" % i, "Initiate a separate interaction graph based on specified atom ids")
            g._v_attrs.arguments = np.array(['pos'])
            t.create_carray(g, "id",  obj=np.arange(nres*3)+np.sum(nres_list[:i+1])*3)