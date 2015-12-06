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


template<typename IType>
struct InteractionGraph{
    constexpr static const bool symmetric  = IType::symmetric;
    constexpr static const int  simd_width = IType::simd_width;
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
    std::unique_ptr<int32_t[]>  edge_indices1, edge_indices2;
    std::unique_ptr<int32_t[]>  edge_id1,      edge_id2;
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
                        4)),

        types1(new_aligned<int32_t>(n_elem1,4)), types2(new_aligned<int32_t>(n_elem2,4)),
        id1   (new_aligned<int32_t>(n_elem1,4)), id2   (new_aligned<int32_t>(n_elem2,4)),

        pos1(new_aligned<float>(round_up(n_elem1,4)*n_dim1a,             align_bytes)),
        pos2(new_aligned<float>(round_up(symmetric?4:n_elem2,4)*n_dim2a, align_bytes)),

        edge_indices1   (new_aligned<int32_t>(max_n_edge,                 align_bytes)),
        edge_indices2   (new_aligned<int32_t>(max_n_edge,                 align_bytes)),
        edge_id1        (new_aligned<int32_t>(max_n_edge,                 align_bytes)),
        edge_id2        (new_aligned<int32_t>(max_n_edge,                 align_bytes)),
        edge_value      (new_aligned<float>  (max_n_edge,                 align_bytes)),
        edge_deriv      (new_aligned<float>  (max_n_edge*(n_dim1+n_dim2), align_bytes)),
        edge_sensitivity(new_aligned<float>  (max_n_edge,                 align_bytes)),

        interaction_param(new_aligned<float>(n_type1*n_type2*n_param, simd_width)),

        pos1_deriv(new_aligned<float>(round_up(n_elem1,4)*n_dim1a,             maxint(4,simd_width))),
        pos2_deriv(new_aligned<float>(round_up(symmetric?4:n_elem2,4)*n_dim2a, maxint(4,simd_width)))

        #ifdef PARAM_DERIV
        ,interaction_param_deriv(n_param, n_type1*n_type2)
        #endif
    {
        using namespace h5;
        auto suffix1 = [](const char* base) {return base + std::string(symmetric?"":"1");};
        bool s = symmetric;

        fill_n(edge_sensitivity, max_n_edge, 0.f);
        fill_n(pos1, round_up(n_elem1,4)*n_dim1a, 1e20f); // just put dummy values far from all points
        fill_n(pos2, round_up(symmetric?4:n_elem2,4)*n_dim2a, 1e20f);
        fill_n(pos1_deriv, round_up(n_elem1,4)*n_dim1a, 0.f);
        fill_n(pos2_deriv, round_up(symmetric?4:n_elem2,4)*n_dim2a, 0.f);

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

        traverse_dset<1,int>(grp,suffix1("index").c_str(),[&](size_t nr,int x){loc1.push_back(x);});
        traverse_dset<1,int>(grp,suffix1("type" ).c_str(),[&](size_t nr,int x){types1[nr]=x;});
        traverse_dset<1,int>(grp,suffix1("id"   ).c_str(),[&](size_t nr,int x){id1   [nr]=x;});

        if(!s) {
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
            int32_t offset[4] = {0,1,2,3};
            auto cutoff2 = Float4(sqr(cutoff));

            int ne = 0;
            for(int32_t i1=0; i1<n_elem1; i1+=4) {
                Float4 v0(pos1+(i1+0)*n_dim1a), // pos1 size was rounded up so that this never runs past the end
                       v1(pos1+(i1+1)*n_dim1a), 
                       v2(pos1+(i1+2)*n_dim1a), 
                       v3(pos1+(i1+3)*n_dim1a);
                transpose4(v0,v1,v2,v3); // v3 will be unused at the end
                auto  x1 = make_vec3(v0,v1,v2);

                auto  my_id1 = Int4(id1+i1);
                auto  i1_vec = Int4(i1) + Int4(offset,Alignment::unaligned);

                for(int32_t i2=symmetric?i1+1:0; i2<n_elem2; ++i2) {
                    float* p = (symmetric?pos1:pos2).get()+i2*n_dim2a;
                    auto  x2 = make_vec3(Float4(p[0]), Float4(p[1]),  Float4(p[2]));
                    auto near = mag2(x1-x2)<cutoff2;
                    if(near.none()) continue;

                    auto my_id2 = Int4((symmetric?id1:id2)[i2]);  // might as well start the load
                    auto i2_vec = Int4(i2);
                    auto pair_acceptable = IType::acceptable_id_pair(my_id1,my_id2);

                    Int4 is_hit = symmetric 
                        ? ((i1_vec<i2_vec) & near.cast_int()) & pair_acceptable
                        :                    near.cast_int()  & pair_acceptable;
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
                edge_indices1[i] = 0;
                edge_indices2[i] = 0; // just put something sane here
            }
        }
        // printf("n_edge for n_dim1 %i n_dim2 %i is %i\n", n_dim1, n_dim2, n_edge);

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
            IType::compute_edge(d1,d2, interaction_ptr, coord1,coord2).store(edge_value+ne);

            store_vec(edge_deriv + ne*(n_dim1+n_dim2),          d1);
            store_vec(edge_deriv + ne*(n_dim1+n_dim2)+4*n_dim1, d2);

            #ifdef PARAM_DERIV
            NOT UPDATED YET;
            edge_param_deriv.push_back(make_zero<n_param>());
            IType::param_deriv(edge_param_deriv.back(), interaction_param+interaction*n_param, 
                               coord1, coord2);
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
            auto t1 = types1[i1];
            auto t2 = types2[i2];
            update_vec(interaction_param_deriv, t1*n_type2+t2, sens*edge_param_deriv[ne]);
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
