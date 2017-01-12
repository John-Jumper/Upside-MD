#ifndef INTERACTION_GRAPH_H
#define INTERACTION_GRAPH_H

#include <vector>
#include "deriv_engine.h"
#include "vector_math.h"
#include "h5_support.h"
#include "timing.h"
#include <algorithm>
#include "Float4.h"


template <typename T>
inline T* operator+(const std::unique_ptr<T[]>& ptr, int i) {
    // little function to make unique_ptr for an array do pointer arithmetic
    return ptr.get()+i;
}

template <typename T>
void fill_n(std::unique_ptr<T[]> &ptr, int n_elem, const T& value) {
    std::fill_n(ptr.get(), n_elem, value);
}

template <typename T>
static T&& message(const std::string& s, T&& x) {
    printf("%s", s.c_str());
    return std::forward<T>(x);
}

template <bool symmetric>
struct PairlistComputation {
    public:
        struct alignas(16) CoarsePacket4 {
            alignas(16) float x[4][4];
            alignas(16) float y[4][4];
            alignas(16) float z[4][4];
            alignas(16) float coarse_x[4];
            alignas(16) float coarse_y[4];
            alignas(16) float coarse_z[4];
            alignas(16) float coarse_r[4];

            Vec<3,Float4> read_fine(int i) const {
                return make_vec3(Float4(x[i]), Float4(y[i]), Float4(z[i]));
            }
            Vec<3,Float4> read_constant_fine(int i, int j) const {
                return make_vec3(Float4(x[i][j]), Float4(y[i][j]), Float4(z[i][j]));
            }
            Vec<4,Float4> read_coarse() const {
                return make_vec4(Float4(coarse_x), Float4(coarse_y), Float4(coarse_z), Float4(coarse_r));
            }
            Vec<4,Float4> read_constant_coarse(int i) const {
                return make_vec4(Float4(coarse_x[i]), Float4(coarse_y[i]), Float4(coarse_z[i]), Float4(coarse_r[i]));
            }
        };

        const int n_elem1, n_elem2;


        std::unique_ptr<int32_t[]>  edge_indices1, edge_indices2;
        std::unique_ptr<int32_t[]>  edge_id1,      edge_id2;
        int n_edge;

    public:
        PairlistComputation(int n_elem1_, int n_elem2_, int max_n_edge):
            n_elem1(n_elem1_), n_elem2(n_elem2_),

            edge_indices1(new_aligned<int32_t>(max_n_edge, 16)),
            edge_indices2(new_aligned<int32_t>(max_n_edge, 16)),
            edge_id1     (new_aligned<int32_t>(max_n_edge, 16)),
            edge_id2     (new_aligned<int32_t>(max_n_edge, 16)),

            n_edge(0)

        {}

        void find_edges(float cutoff,
                        const float* aligned_pos1, const int pos1_stride, int* id1, 
                        const float* aligned_pos2, const int pos2_stride, int* id2) {
            int32_t offset[4] = {0,1,2,3};
            auto cutoff2 = Float4(sqr(cutoff));

            int ne = 0;
            for(int32_t i1=0; i1<n_elem1; i1+=4) {
                Float4 v0(aligned_pos1+(i1+0)*pos1_stride), // aligned_pos1 size was rounded up so that this never runs past the end
                       v1(aligned_pos1+(i1+1)*pos1_stride), 
                       v2(aligned_pos1+(i1+2)*pos1_stride), 
                       v3(aligned_pos1+(i1+3)*pos1_stride);
                transpose4(v0,v1,v2,v3); // v3 will be unused at the end
                auto  x1 = make_vec3(v0,v1,v2);

                auto  my_id1 = Int4(id1+i1);
                auto  i1_vec = Int4(i1) + Int4(offset,Alignment::unaligned);

                for(int32_t i2=symmetric?i1+1:0; i2<n_elem2; ++i2) {
                    const float* p = (symmetric?aligned_pos1:aligned_pos2)+i2*pos2_stride;
                    auto  x2 = make_vec3(Float4(p[0]), Float4(p[1]),  Float4(p[2]));
                    auto near = mag2(x1-x2)<cutoff2;
                    if(near.none()) continue;

                    auto my_id2 = Int4((symmetric?id1:id2)[i2]);  // might as well start the load
                    auto i2_vec = Int4(i2);

                    Int4 is_hit = symmetric 
                        ? (i1_vec<i2_vec) & near.cast_int()
                        :                   near.cast_int();
                    int is_hit_bits = is_hit.movemask();

                    // i2_vec and my_id2 is constant, so we don't have to left pack
                    i2_vec                       .store(edge_indices2+ne, Alignment::unaligned);
                    my_id2                       .store(edge_id2     +ne, Alignment::unaligned);
                    i1_vec.left_pack(is_hit_bits).store(edge_indices1+ne, Alignment::unaligned);
                    my_id1.left_pack(is_hit_bits).store(edge_id1     +ne, Alignment::unaligned);
                    ne += popcnt_nibble(is_hit_bits);
                }
            }
            n_edge = ne;
            for(int i=ne; i<round_up(ne,4); ++i) {
                // we need something sane to fill out the last group of 4 so just duplicate the interactions
                // with sensitivity 0.
                edge_indices1[i] = edge_indices1[i-i%4];
                edge_indices2[i] = edge_indices2[i-i%4]; // just put something sane here
            }
        }

        typedef Int4(*acceptable_id_pair_t)(const Int4&,const Int4&);
        template<acceptable_id_pair_t acceptable_id_pair>
        void filter_for_acceptable_id() {
            int ne=0;
            alignas(16) int32_t offset_v[4] = {0,1,2,3};
            Int4 offset(offset_v);
            Int4 n_edge4(n_edge);

            for(int i_edge=0; i_edge<n_edge; i_edge+=4) {
                auto i1 = Int4(edge_indices1+i_edge);
                auto i2 = Int4(edge_indices2+i_edge);
                auto eid1 = Int4(edge_id1+i_edge);
                auto eid2 = Int4(edge_id2+i_edge);

                auto pair_acceptable = (acceptable_id_pair(eid1,eid2)&(Int4(i_edge)+offset<n_edge4)).movemask();
                // Now pack and write.  Since ne <= i_edge always
                // we may write without worrying about collisions
                
                i1  .left_pack(pair_acceptable).store(edge_indices1+ne, Alignment::unaligned);
                i2  .left_pack(pair_acceptable).store(edge_indices2+ne, Alignment::unaligned);
                eid1.left_pack(pair_acceptable).store(edge_id1     +ne, Alignment::unaligned);
                eid2.left_pack(pair_acceptable).store(edge_id2     +ne, Alignment::unaligned);
                ne += popcnt_nibble(pair_acceptable);
            }
            n_edge = ne;
        }
};


template<typename IType>
struct InteractionGraph{
    constexpr static const bool symmetric  = IType::symmetric;
    constexpr static const bool simd_width = IType::simd_width;
    constexpr static const int  align      = maxint(4,simd_width);
    constexpr static const int  align_bytes= 4*align;
    constexpr static const int  n_dim1     = IType::n_dim1, n_dim1a = round_up(n_dim1, align);
    constexpr static const int  n_dim2     = IType::n_dim2, n_dim2a = round_up(n_dim2, align);
    constexpr static const int  n_param    = IType::n_param;

    CoordNode* pos_node1;
    CoordNode* pos_node2;
    std::vector<index_t> loc1, loc2;

    int   n_elem1, n_elem2;
    int   n_type1, n_type2;
    int   max_n_edge;
    float cutoff;

    int n_edge;

    std::unique_ptr<int32_t[]>  types1, types2; // pair type is type[0]*n_types2 + type[1]
    std::unique_ptr<int32_t[]>  id1,    id2;    // used to avoid self-interaction

    // buffers to copy position data to ensure contiguity
    std::unique_ptr<float[]> pos1, pos2;

    // per edge data
    PairlistComputation<IType::symmetric> pairlist;
    int32_t* edge_indices1; // pointers to pairlist-maintained arrays
    int32_t* edge_indices2;  
    int32_t* edge_id1;
    int32_t* edge_id2;
    std::unique_ptr<float[]>    edge_value;
    std::unique_ptr<float[]>    edge_deriv;  // this may become a SIMD-type vector
    std::unique_ptr<float[]>    edge_sensitivity; // must be filled by user of this class

    std::unique_ptr<float[]> interaction_param;

    std::unique_ptr<float[]> pos1_deriv, pos2_deriv;

    #ifdef PARAM_DERIV
    std::vector<Vec<n_param>> edge_param_deriv;
    VecArrayStorage           interaction_param_deriv;
    #endif

    InteractionGraph(hid_t grp, CoordNode* pos_node1_, CoordNode* pos_node2_ = nullptr):
        pos_node1(pos_node1_), pos_node2(pos_node2_),

        n_elem1(h5::get_dset_size(1,grp,symmetric?"index":"index1")[0]), 
        n_elem2(symmetric ? n_elem1 : h5::get_dset_size(1,grp,"index2")[0]), 

        n_type1(h5::get_dset_size(3,grp,"interaction_param")[0]),
        n_type2(h5::get_dset_size(3,grp,"interaction_param")[1]),

        max_n_edge(round_up(
                    h5::read_attribute<int>(grp, ".", "max_n_edge", 
                        n_elem1*n_elem2/(symmetric?2:1)),
                        16)),

        types1(new_aligned<int32_t>(n_elem1,16)), types2(new_aligned<int32_t>(n_elem2,16)),
        id1   (new_aligned<int32_t>(n_elem1,16)), id2   (new_aligned<int32_t>(n_elem2,16)),

        pos1(new_aligned<float>(round_up(n_elem1,16)*n_dim1a,             align_bytes)),
        pos2(new_aligned<float>(round_up(symmetric?16:n_elem2,16)*n_dim2a, align_bytes)),

        pairlist(n_elem1,n_elem2,max_n_edge),
        edge_indices1(pairlist.edge_indices1.get()),
        edge_indices2(pairlist.edge_indices2.get()),
        edge_id1      (pairlist.edge_id1.get()),
        edge_id2      (pairlist.edge_id2.get()),

        edge_value      (new_aligned<float>  (max_n_edge,                 align_bytes)),
        edge_deriv      (new_aligned<float>  (max_n_edge*(n_dim1+n_dim2), align_bytes)),
        edge_sensitivity(new_aligned<float>  (max_n_edge,                 align_bytes)),

        interaction_param(new_aligned<float>(n_type1*n_type2*n_param, 4)),

        pos1_deriv(new_aligned<float>(round_up(n_elem1,16)*n_dim1a,             maxint(4,simd_width))),
        pos2_deriv(new_aligned<float>(round_up(symmetric?16:n_elem2,16)*n_dim2a, maxint(4,simd_width)))

        #ifdef PARAM_DERIV
        ,interaction_param_deriv(n_param, n_type1*n_type2)
        #endif
    {
        using namespace h5;
        auto suffix1 = [](const char* base) {return base + std::string(symmetric?"":"1");};
        bool s = symmetric;

        fill_n(edge_sensitivity, max_n_edge, 0.f);
        fill_n(pos1, round_up(n_elem1,16)*n_dim1a, 1e20f); // just put dummy values far from all points
        fill_n(pos2, round_up(symmetric?16:n_elem2,16)*n_dim2a, 1e20f);
        fill_n(pos1_deriv, round_up(n_elem1,16)*n_dim1a, 0.f);
        fill_n(pos2_deriv, round_up(symmetric?16:n_elem2,16)*n_dim2a, 0.f);

        if(!(s ^ bool(pos_node2)))
            throw std::string("second node must be null iff symmetric interaction");

        check_elem_width_lower_bound(*pos_node1, n_dim1);
        if(!s) check_elem_width_lower_bound(*pos_node2, n_dim2);

        check_size(grp, "interaction_param", n_type1, n_type2, n_param);
        traverse_dset<3,float>(grp, "interaction_param", [&](size_t nt1, size_t nt2, size_t np, float x) {
                interaction_param[(nt1*n_type2+nt2)*n_param+np] = x;});
        update_cutoffs();

        check_size(grp, suffix1("index").c_str(), n_elem1); if(!s) check_size(grp, "index2", n_elem2);
        check_size(grp, suffix1("type").c_str(),  n_elem1); if(!s) check_size(grp, "type2",  n_elem2);
        check_size(grp, suffix1("id").c_str(),    n_elem1); if(!s) check_size(grp, "id2",    n_elem2);

        for(int i=0; i<round_up(n_elem1,16); ++i) id1[i] = 0;  // padding
        traverse_dset<1,int>(grp,suffix1("index").c_str(),[&](size_t nr,int x){loc1.push_back(x);});
        traverse_dset<1,int>(grp,suffix1("type" ).c_str(),[&](size_t nr,int x){types1[nr]=x;});
        traverse_dset<1,int>(grp,suffix1("id"   ).c_str(),[&](size_t nr,int x){id1   [nr]=x;});

        if(!s) {
            for(int i=0; i<round_up(n_elem2,16); ++i) id2[i] = 0;  // padding
            traverse_dset<1,int>(grp,"index2",[&](size_t nr,int x){loc2.push_back(x);});
            traverse_dset<1,int>(grp,"type2", [&](size_t nr,int x){types2[nr]=x;});
            traverse_dset<1,int>(grp,"id2",   [&](size_t nr,int x){id2   [nr]=x;});
        } else {
            for(int nr: range(n_elem2)) types2[nr] = types1[nr];
            for(int nr: range(n_elem2)) id2   [nr] = id1   [nr];
        }
    }

    void update_cutoffs() {
        cutoff = 0.f;
        for(int nt1: range(n_type1)) {
            for(int nt2: range(n_type2)) {
                cutoff = std::max(cutoff, IType::cutoff(interaction_param+(nt1*n_type2+nt2)*n_param));

                if(IType::symmetric && !IType::is_compatible(interaction_param+(nt1*n_type2+nt2)*IType::n_param, 
                                                             interaction_param+(nt2*n_type2+nt1)*IType::n_param)){
                    throw std::string("incompatibile parameters");
                }
            }
        }
    }

    #ifdef PARAM_DERIV
    std::vector<float> get_param() const {
        return {interaction_param.get(), interaction_param.get()+n_type1*n_type2*n_param};
    }

    std::vector<float> get_param_deriv() const {
        std::vector<float> ret; ret.reserve(n_type1*n_type2*n_param);
        for(int i: range(n_type1*n_type2))
            for(int d: range(n_param))
                ret.push_back(interaction_param_deriv(d,i));
        return ret;
    }

    void set_param(const std::vector<float>& new_param) {
        if(new_param.size() != size_t(n_type1*n_type2*IType::n_param)) throw std::string("Bad param size");
        std::copy(begin(new_param), end(new_param), interaction_param.get());
        update_cutoffs();
    }
    #endif

    std::vector<float> count_edges_by_type() {
        // must be called after compute_edges 
        // return type is really an int vector but float is convenient for other interfaces
        std::vector<float> retval(n_type1*n_type2, 0.f);
        for(int ne=0; ne<n_edge; ++ne) {
            auto i1 = edge_indices1[ne];
            auto i2 = edge_indices2[ne];
            auto t1 = types1[i1];
            auto t2 = types2[i2];
            auto interaction_index = t1*n_type2 + t2;

            retval[interaction_index] += 1.f;
        }
        return retval;
    }

    void compute_edges() {
        // Copy in the data to packed arrays to ensure contiguity
        {
            VecArray posv = pos_node1->output;
            for(int ne=0; ne<n_elem1; ++ne) 
                store_vec(pos1.get()+ne*n_dim1a, load_vec<n_dim1>(posv, loc1[ne]));
        }
        if(!symmetric) {
            VecArray posv = pos_node2->output;
            for(int ne=0; ne<n_elem2; ++ne) 
                store_vec(pos2.get()+ne*n_dim2a, load_vec<n_dim2>(posv, loc2[ne]));
        }

        // First find all the edges
        {
            Timer timer1("find_edges");
            pairlist.find_edges(cutoff,
                                pos1.get(), n_dim1a, id1.get(),
                                pos2.get(), n_dim2a, id2.get());
            timer1.stop();

            Timer timer2("filter_edges");
            if(IType::filter_for_acceptable_id)
                pairlist.template filter_for_acceptable_id<IType::acceptable_id_pair>();
            timer2.stop();

            n_edge = pairlist.n_edge;
        }
        // printf("n_edge for n_dim1 %i n_dim2 %i n_elem1 %i n_elem2 %i is %i\n", n_dim1, n_dim2, n_elem1, n_elem2, n_edge);

        // Compute edge values
        #ifdef PARAM_DERIV
        edge_param_deriv.clear();
        #endif

        for(int ne=0; ne<n_edge; ne+=4) {
            auto i1 = Int4(edge_indices1+ne);
            auto i2 = Int4(edge_indices2+ne);

            auto t1 = Int4(types1.get(),i1);
            auto t2 = Int4(types2.get(),i2);

            auto interaction_offset = (t1*Int4(n_type2) + t2)*Int4(n_param);
            const float* interaction_ptr[4] = {
                interaction_param+interaction_offset.x(),
                interaction_param+interaction_offset.y(),
                interaction_param+interaction_offset.z(),
                interaction_param+interaction_offset.w()};

            auto coord1 = aligned_gather_vec<n_dim1>(pos1.get(),                 i1*Int4(n_dim1a));
            auto coord2 = aligned_gather_vec<n_dim2>((symmetric?pos1:pos2).get(),i2*Int4(n_dim2a));

            Vec<n_dim1,Float4> d1;
            Vec<n_dim2,Float4> d2;

            // print_vector("i1 ", i1);
            // print_vector("i2 ", i2);
            // print_vector("t1 ", t1);
            // print_vector("t2 ", t2);
            // print_vector("id1", Int4(edge_id1+ne));
            // print_vector("id2", Int4(edge_id2+ne));
            IType::compute_edge(d1,d2, interaction_ptr, coord1,coord2).store(edge_value+ne);
            store_vec(edge_deriv + ne*(n_dim1+n_dim2),          d1);
            store_vec(edge_deriv + ne*(n_dim1+n_dim2)+4*n_dim1, d2);

            #ifdef PARAM_DERIV
            for(int i: range(4)) {
                Vec<n_dim1> c1; for(int d: range(n_dim1)) c1[d] = extract_float(coord1[d],i);
                Vec<n_dim2> c2; for(int d: range(n_dim2)) c2[d] = extract_float(coord2[d],i);

                edge_param_deriv.push_back(make_zero<n_param>());
                IType::param_deriv(edge_param_deriv.back(), interaction_ptr[i], c1,c2);
            }
            #endif
        }
    }


    void propagate_derivatives() {
        // Finally put the data where it is needed.
        // This function must be called after the user sets edge_sensitivity

        // The edge_sensitivity of elements at location n_edge and beyond must
        // be zero since these are not real edges.  This is an implementation detail
        // that the user does not know about, so we will set these edge sensistitivies to 
        // zero ourselves.
        for(int ne=n_edge; ne<round_up(n_edge,4); ++ne) edge_sensitivity[ne] = 0.f;

        // Zero accumulation buffers
        fill_n(pos1_deriv, n_elem1*n_dim1a, 0.f);
        if(!symmetric) fill_n(pos2_deriv, n_elem2*n_dim2a, 0.f);
        #ifdef PARAM_DERIV
        fill(interaction_param_deriv, 0.f);
        #endif

        // Accumulate derivatives
        for(int ne=0; ne<n_edge; ne+=4) {
            auto i1 = Int4(edge_indices1+ne);
            auto i2 = Int4(edge_indices2+ne);
            auto sens = Float4(edge_sensitivity+ne);

            auto d1 = sens*load_vec<n_dim1>(edge_deriv + ne*(n_dim1+n_dim2), Alignment::aligned);
            auto d2 = sens*load_vec<n_dim2>(edge_deriv + ne*(n_dim1+n_dim2)+4*n_dim1, Alignment::aligned);

            aligned_scatter_update_vec_destructive(pos1_deriv.get(),                       i1*Int4(n_dim1a), d1);
            aligned_scatter_update_vec_destructive((symmetric?pos1_deriv:pos2_deriv).get(),i2*Int4(n_dim2a), d2);

            #ifdef PARAM_DERIV
            for(int i: range(4)) {
                int t1 = types1[edge_indices1[ne+i]];
                int t2 = types2[edge_indices2[ne+i]];
                update_vec(interaction_param_deriv, t1*n_type2+t2, extract_float(sens,i)*edge_param_deriv[ne+i]);
            }
            #endif
        }

        // Push derivatives to slots
        {
            VecArray pos1_sens = pos_node1->sens;
            for(int i1=0; i1<n_elem1; ++i1)
                update_vec(pos1_sens, loc1[i1], load_vec<n_dim1>(pos1_deriv+i1*n_dim1a));
        }
        if(!symmetric) {
            VecArray pos2_sens = pos_node2->sens;
            for(int i2=0; i2<n_elem2; ++i2)
                update_vec(pos2_sens, loc2[i2], load_vec<n_dim2>(pos2_deriv+i2*n_dim2a));
        }
    }
};
#endif
