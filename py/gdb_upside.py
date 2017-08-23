import numpy as np
np.set_printoptions(precision=3, suppress=True)

gdb_float_t = gdb.lookup_type('float')
gdb_int_t   = gdb.lookup_type('int')

def gdb_eval(x):
    return gdb.parse_and_eval(x) if isinstance(x,str) else x

def gdb_int(i):
    return int(gdb_eval(i))

def gdb_float(i):
    return float(gdb_eval(i))

def vec_array(vec, n_row, n_col):
    vec   = gdb_eval(vec)
    n_row = gdb_eval(n_row)
    n_col = gdb_eval(n_col)

    real_type_str = str(vec.type.strip_typedefs().unqualified())

    if real_type_str == 'VecArrayStorage':
        n_elem = int(vec['n_elem'])
        assert n_elem >= n_row
    elif real_type_str == 'VecArray':
        pass
    else:
        raise ValueError('wrong variable type %s'%real_type_str)

    x = vec['x'].cast(gdb_float_t.pointer())
    row_width = int(vec['row_width'])

    assert row_width >= n_col

    ret = np.zeros((n_row,n_col), dtype='f4')
    for ne in range(n_row):
        for nc in range(n_col):
            ret[ne,nc] = x[ne*row_width + nc]

    return ret

def float4(x):
    x = gdb_eval(x)
    y = x['v']['vec']
    return np.array([float(y[i]) for i in range(4)],dtype='f4')

def multilog(*tuples):
    for f, full_str in tuples:
        for s in full_str.split():
            print(s, f(s))

def py2_dict_pickle(path):
    import pickle as cp
    with open(path,'rb') as f:
        d=cp.load(f)
    print(d)
    return dict((k.decode(),v) for k,v in d.items())

def unique_to_ptr(unique_ptr, gdb_base_type=gdb_float_t):
    # Performs a dance to convert a unique ptr to a raw pointer in GDB
    # This would be much easier if GDB supported .get() on the Value type
    # This function assumes the raw pointer is the first element of the struct
    #   so that &unique_ptr == &unique_ptr.get()
    unique_ptr = gdb_eval(unique_ptr)
    return unique_ptr.address.reinterpret_cast(gdb_base_type.pointer().pointer()).dereference()

def read_array(ptr, n, dtype='f4'):
    ptr = gdb_eval(ptr)
    n = gdb_int(n)

    ret = np.zeros(n, dtype=dtype)
    for i in range(n):
        ret[i] = ptr[i]
    return ret

def round_up(n,r):
    return r*((n+r-1)//r)

def read_edges(edge_holder):
    edge_holder = gdb_eval(edge_holder)
    n_edge = gdb_int(edge_holder['nodes_to_edge']['n_edge'])
    n_rot1 = gdb_int(edge_holder['n_rot1'])
    n_rot2 = gdb_int(edge_holder['n_rot2'])
    n_rot1u = round_up(n_rot1,4)
    n_rot2u = round_up(n_rot2,4)
    ei1 = read_array(unique_to_ptr(edge_holder['edge_indices1'], gdb_int_t), n_edge, 'i4')
    ei2 = read_array(unique_to_ptr(edge_holder['edge_indices2'], gdb_int_t), n_edge, 'i4')
    cur_belief = vec_array(edge_holder['cur_belief'], n_edge, n_rot1u+n_rot2u)

    prob = vec_array(edge_holder['prob'], n_edge, n_rot1*n_rot2u)
    prob = prob.reshape(n_edge, n_rot1, n_rot2u)[:,:,:n_rot2]

    d = dict(
            inds=np.column_stack([ei1,ei2]),
            cur_belief1 = cur_belief[:,:n_rot1],
            cur_belief2 = cur_belief[:,n_rot1u:n_rot1u+n_rot2],
            prob = prob)

    d['cur_belief1'] /= d['cur_belief1'].sum(axis=1,keepdims=1)
    d['cur_belief2'] /= d['cur_belief2'].sum(axis=1,keepdims=1)
    return d
    # ei2 = read_array(unique_to_ptr(edge_holder['edge_indices1'], gdb_int_t), n_edge, 'i4')

