#include "deriv_engine.h"
#include "timing.h"
#include "coord.h"

using namespace h5;
using namespace std;


namespace {

struct BasinCorrelationParams {
    CoordPair residue[2];
    int       connection_matrix_id;
};

struct Basin {
    float2 center;
    float2 half_width;
};

template<int N_BASIN>
void basin_correlation_pot(
        float* restrict potential,
        const CoordArray rama,
        const BasinCorrelationParams* restrict params,
        const float* connection_matrices,
        float basin_sharpness, Basin* basins,
        int n_term, int n_system) 
{
    #pragma omp parallel for schedule(static,1)
    for(int ns=0; ns<n_system; ++ns) {
        if(potential) potential[ns] = 0.f;

        for(int nt=0; nt<n_term; ++nt) {
            Coord<2> rama1(rama, ns, params[nt].residue[0]);
            Coord<2> rama2(rama, ns, params[nt].residue[1]);

            float basin_prob1[N_BASIN];
            float basin_prob2[N_BASIN];

            float2 basin_prob_deriv1[N_BASIN];
            float2 basin_prob_deriv2[N_BASIN];

            for(int nb=0; nb<N_BASIN; ++nb) {
                float3 prob1 = rama_box(make_float2(rama1.v[0], rama1.v[1]), 
                        basins[nb].center, basins[nb].half_width, basin_sharpness);
                basin_prob1[nb] = prob1.x;
                basin_prob_deriv1[nb] = make_float2(prob1.y, prob1.z);

                float3 prob2 = rama_box(make_float2(rama2.v[0], rama2.v[1]), 
                        basins[nb].center, basins[nb].half_width, basin_sharpness);
                basin_prob2[nb] = prob2.x;
                basin_prob_deriv2[nb] = make_float2(prob2.y, prob2.z);
            }


            // FIXME there are two common cases worth optimizing for
            // (1) The dbasin/drama is exactly zero for one of the residues
            // (2) There is only one nonzero entry in one of the basin_prob vectors
            // I should optimize for both cases.  Many times these conditions will occur
            // at the same time, but I should not assume that for correctness

            const float* mat = connection_matrices + params[nt].connection_matrix_id*N_BASIN*N_BASIN;

            float mat_times_basin_prob2[N_BASIN];
            for(int nb=0; nb<N_BASIN; ++nb) {
                mat_times_basin_prob2[nb] = 0.f;
                for(int i=0; i<N_BASIN; ++i)
                    mat_times_basin_prob2[nb] += mat[nb*N_BASIN+i]*basin_prob2[i];
            }

            float basin_prob1_times_mat[N_BASIN];
            for(int nb=0; nb<N_BASIN; ++nb) {
                basin_prob1_times_mat[nb] = 0.f;
                for(int i=0; i<N_BASIN; ++i)
                    basin_prob1_times_mat[nb] += basin_prob1[i]*mat[i*N_BASIN+nb];
            }

            float total_prob = 0.f;
            for(int nb=0; nb<N_BASIN; ++nb) 
                total_prob += basin_prob1[nb] * mat_times_basin_prob2[nb];
            float prefactor = -1.f/total_prob;
            if(potential) potential[ns] += -log(total_prob);

            float2 deriv1 = make_float2(0.f,0.f);
            for(int nb=0; nb<N_BASIN; ++nb) deriv1 += basin_prob_deriv1[nb] * mat_times_basin_prob2[nb];
            rama1.d[0][0] = deriv1.x*prefactor; rama1.d[0][1] = deriv1.y*prefactor; 

            float2 deriv2 = make_float2(0.f,0.f);
            for(int nb=0; nb<N_BASIN; ++nb) deriv2 += basin_prob_deriv2[nb] * basin_prob1_times_mat[nb];
            rama2.d[0][0] = deriv2.x*prefactor; rama2.d[0][1] = deriv2.y*prefactor; 

            rama1.flush();
            rama2.flush();
        }
    }
}
}


template<int N_BASIN>
struct BasinCorrelationPot : public PotentialNode
{
    int n_term;
    CoordNode& rama;
    vector<BasinCorrelationParams> params;

    int n_matrices;
    vector<float> connection_matrices;
    vector<Basin> basins;
    float basin_sharpness;

    BasinCorrelationPot(hid_t grp, CoordNode& rama_):
        PotentialNode(rama_.n_system),
        n_term    (get_dset_size(2, grp, "residue_id")[0]), 
        rama(rama_), 
        params(n_term),
        n_matrices(get_dset_size(3, grp, "connection_matrices")[0]),
        basins(N_BASIN),
        basin_sharpness(read_attribute<float>(grp, "basin_half_width", "sharpness"))
    {
        check_size(grp, "basin_center",         N_BASIN, 2);
        check_size(grp, "basin_half_width",     N_BASIN, 2);

        check_size(grp, "residue_id",           n_term, 2);
        check_size(grp, "connection_matrix_id", n_term);
        check_size(grp, "connection_matrices",  n_matrices, N_BASIN, N_BASIN);

        traverse_dset<2,float> (grp, "basin_center", [&](size_t nb, size_t phipsi, float x) {
                if(phipsi==0) basins[nb].center.x = x; else basins[nb].center.y = x;});
        traverse_dset<2,float> (grp, "basin_half_width", [&](size_t nb, size_t phipsi, float x) {
                if(phipsi==0) basins[nb].half_width.x = x; else basins[nb].half_width.y = x;});

        traverse_dset<2,int>   (grp, "residue_id",  [&](size_t nt, size_t nr, int x) {params[nt].residue[nr].index = x;});
        traverse_dset<1,int>   (grp, "connection_matrix_id", [&](size_t i, int x) {params[i].connection_matrix_id = x;});
        traverse_dset<3,double>(grp, "connection_matrices",  [&](size_t i, size_t nb1, size_t nb2, double x) {
                connection_matrices.push_back(x);});

        for(size_t nr=0; nr<2; ++nr) for(auto &p: params) rama.slot_machine.add_request(1, p.residue[nr]);
    }

    virtual void compute_value(ComputeMode mode) {
        Timer timer(string("basin_correlation_pot"));
        basin_correlation_pot<N_BASIN>((mode==PotentialAndDerivMode ? potential.data() : nullptr),
                rama.coords(), params.data(), connection_matrices.data(), basin_sharpness, basins.data(),
                n_term, n_system);
    }

    virtual double test_value_deriv_agreement() {
        vector<vector<CoordPair>> coord_pairs(1);
        for(auto &p: params) for(int nr=0; nr<2; ++nr) coord_pairs.back().push_back(p.residue[nr]);
        return compute_relative_deviation_for_node<2>(*this, rama, coord_pairs);
    }
};
static RegisterNodeType<BasinCorrelationPot<5>,1> basin_correlation_pot_node("basin_correlation_pot");
