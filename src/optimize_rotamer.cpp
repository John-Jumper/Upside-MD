#include "vector_math.h"
#include <dai/alldai.h>
#include <vector>
#include <string>
#include "spline.h"
#include <tuple>
#include "h5_support.h"
#include "affine.h"
#include <algorithm>
#include <chrono>

using namespace std;
using namespace h5;

void 
three_atom_alignment(
        float * restrict rigid_body, //length 7 = 3 translation + 4 quaternion rotation
        float * restrict deriv,    // length 3x3x7
        const float * restrict atom1, const float*  restrict atom2, const float*  restrict atom3,  // each length 3
        const float*  restrict ref_geom);  // length 9

struct PairHistogram {
    int n_type;
    int n_bin;
    double dx;
    double inv_dx;
    double cutoff;
    vector<double> count;

    PairHistogram(int n_type_, int n_bin_, double dx_):
        n_type(n_type_), n_bin(n_bin_), dx(dx_), inv_dx(1.f/dx), 
        count(n_type*n_type*n_bin, 0.) {}

    void accumulate(int rt1, int rt2, double value, double weight) {
        double normed_value = value*inv_dx;
        if(normed_value >= n_bin) return;
        count[min(rt1,rt2)*n_type*n_bin + max(rt1,rt2)*n_bin + int(normed_value)] += weight;
    }
};


struct SidechainInteractionParams {
    float cutoff2;
    float radius_outer, scale_outer, energy_outer;
    float radius_inner, scale_inner, energy_inner;

    void update_cutoff2() {
        float cutoff2_inner = sqr(radius_inner + 1.f/scale_inner);
        float cutoff2_outer = sqr(radius_outer + 1.f/scale_outer);
        cutoff2 = cutoff2_inner<cutoff2_outer ? cutoff2_outer : cutoff2_inner;
    }

    bool operator==(const SidechainInteractionParams& other) const {
        return cutoff2      == other.cutoff2      && 
               radius_inner == other.radius_inner && radius_outer == other.radius_outer &&
               energy_inner == other.energy_inner && energy_outer == other.energy_outer &&
               scale_inner  == other.scale_inner  && scale_outer  == other.scale_outer;
    }

    bool operator!=(const SidechainInteractionParams& other) {return !(*this==other);}
};


struct SidechainParams {
    int restype;
    int start_loc;
};


void rotamer_sidechain(
        PairHistogram &hist,
        float*                      marginal_prob,
        char*                       inference_string,

        VecArray                    rotamer_pos_and_prob,  // dimension 4,n_total_rotamer
        SidechainParams*            params,                // length n_res
        SidechainInteractionParams* interactions,          // size (n_restype, n_restype)

        int n_total_rotamer, int n_restype, int n_res) {

    auto tstart = chrono::high_resolution_clock::now();
    vector<dai::Var> vars;
    vector<dai::Factor> factors;

    int max_rot = 0;
    for(int nr=0; nr<n_res; ++nr) {
        // create variables
        int start = params[nr].start_loc, stop = params[nr+1].start_loc;
        if(stop-start>max_rot) max_rot = stop-start;
        vars.emplace_back(nr, stop-start);

        // set 1-body probabilities
        factors.emplace_back(vars.back());
        for(int nrot=0; nrot<stop-start; ++nrot) 
            factors.back().set(nrot, rotamer_pos_and_prob(3, nrot+start));
    }

    // Note the column-major order to match the DAI library
    vector<double> iprob(max_rot*max_rot);

    long n_interaction = 0;
    for(int nr1=0; nr1<n_res; ++nr1) {
        int n_rot1 = params[nr1+1].start_loc - params[nr1].start_loc;
        int nt1 = params[nr1].restype;

        for(int nr2=nr1+1; nr2<n_res; ++nr2) {
            int n_rot2 = params[nr2+1].start_loc - params[nr2].start_loc;
            int nt2 = params[nr2].restype;

            auto &p = interactions[nt1*n_restype+nt2];
            float cutoff2 = p.cutoff2;

            long old_n_interaction = n_interaction;
            for(int no1=0; no1<n_rot1; ++no1) {
                float3 x1 = load_vec<3>(rotamer_pos_and_prob, params[nr1].start_loc+no1);

                for(int no2=0; no2<n_rot2; ++no2) {
                    float3 x2 = load_vec<3>(rotamer_pos_and_prob, params[nr2].start_loc+no2);

                    if(mag2(x1-x2) >= cutoff2) {
                        iprob[no1 + n_rot1*no2] = 1.f;
                        continue;
                    }

                    ++n_interaction;

                    float dist = mag(x2-x1);
                    float2 en = p.energy_outer*compact_sigmoid(dist-p.radius_outer, p.scale_outer)
                               +p.energy_inner*compact_sigmoid(dist-p.radius_inner, p.scale_inner);
                    
                    // FIXME should I work at a varying sidechain temperature or always T=1?
                    iprob[no1 + n_rot1*no2] = expf(-en.x());
                }
            }

            if(n_interaction>old_n_interaction) 
                factors.emplace_back(dai::VarSet(vars[nr1], vars[nr2]), iprob.data());
        }
    }

    dai::FactorGraph fg(factors);
    auto tfinish = chrono::high_resolution_clock::now();
    // if(n_res == 74) fg.WriteToFile("factor_graph.txt");
    
    dai::InfAlg* ia = dai::newInfAlgFromString(string(inference_string), fg);
    auto tsolve_start = chrono::high_resolution_clock::now();
    ia->init();
    ia->run();
    auto tsolve_finish = chrono::high_resolution_clock::now();

    for(int nr=0; nr<n_res; ++nr) {
        int start = params[nr].start_loc, stop = params[nr+1].start_loc;
        auto p = ia->beliefV(nr).normalized().p();
        if(int(p.size()) != stop-start) throw string("impossible");

        for(int nrot=0; nrot<stop-start; ++nrot) 
            marginal_prob[start+nrot] = p[nrot];
    }


    auto thist_start = chrono::high_resolution_clock::now();
    vector<int> have_pair(n_res*n_res, 0);
    vector<dai::Factor> beliefs = ia->beliefs();

    std::vector<double> pair_prob;
    pair_prob.reserve(max_rot*max_rot);
    for(auto& b: beliefs) {
        auto& vars = b.vars();
        if(vars.size() != 2) continue;

        pair_prob.clear();
        auto p = b.normalized().p();
        for(double x: p)
            pair_prob.push_back(x);

        // extract residue numbers
        int nr1 = vars.front().label();
        int nr2 = vars.back() .label();

        have_pair[nr1*n_res + nr2] = 1;

        int n_rot1 = params[nr1+1].start_loc - params[nr1].start_loc;
        int nt1 = params[nr1].restype;

        int n_rot2 = params[nr2+1].start_loc - params[nr2].start_loc;
        int nt2 = params[nr2].restype;

        for(int no1=0; no1<n_rot1; ++no1) {
            float3 x1 = load_vec<3>(rotamer_pos_and_prob, params[nr1].start_loc+no1);
            for(int no2=0; no2<n_rot2; ++no2) {
                float3 x2 = load_vec<3>(rotamer_pos_and_prob, params[nr2].start_loc+no2);
                double weight = pair_prob[no1 + n_rot1*no2];
                hist.accumulate(nt1, nt2, mag(x1-x2), weight);
            }
        }
    }


    for(int nr1: range(n_res)) {
        int n_rot1 = params[nr1+1].start_loc - params[nr1].start_loc;
        int nt1 = params[nr1].restype;

        for(int nr2: range(nr1+1,n_res)) {
            if(have_pair[nr1*n_res+nr2]) continue; // statistics already accumulated

            int n_rot2 = params[nr2+1].start_loc - params[nr2].start_loc;
            int nt2 = params[nr2].restype;

            auto pr1 = marginal_prob + params[nr1].start_loc;
            auto pr2 = marginal_prob + params[nr2].start_loc;
            pair_prob.resize(n_rot1*n_rot2);

            // if no joint distribution is found, approximate with an independent distribution
            for(int i1: range(n_rot1)) 
                for(int i2: range(n_rot2)) 
                    pair_prob[i1 + n_rot1*i2] = pr1[i1]*pr2[i2];
            // printf("belief not found for %i %i %f\n", nr1,nr2, accumulate(begin(pair_prob), end(pair_prob), 0.));

            for(int no1=0; no1<n_rot1; ++no1) {
                float3 x1 = load_vec<3>(rotamer_pos_and_prob, params[nr1].start_loc+no1);
                for(int no2=0; no2<n_rot2; ++no2) {
                    float3 x2 = load_vec<3>(rotamer_pos_and_prob, params[nr2].start_loc+no2);
                    double weight = pair_prob[no1 + n_rot1*no2];
                    hist.accumulate(nt1, nt2, mag(x1-x2), weight);
                }
            }
        }
    }
    auto thist_finish = chrono::high_resolution_clock::now();

    printf("build %7.1f us  solve %8.0f us  hist %8.0f us  interactions/res %8.2f ", 
            chrono::duration<double>(tfinish-tstart            ).count()*1e6, 
            chrono::duration<double>(tsolve_finish-tsolve_start).count()*1e6, 
            chrono::duration<double>(thist_finish-thist_start).count()*1e6, 

            n_interaction*1./n_res);
}


struct RotamerIndices {
    int start, stop;
};



struct RotamerSidechain {
    int n_restype;
    vector<SidechainInteractionParams> interactions;
    map<string,int> index_from_restype;

    vector<RotamerIndices> rotamer_indices;  // start and stop
    LayeredPeriodicSpline2D<4> backbone_location_and_prob_fcn;  // order (x,y,z,prob)

    // sequence specific data
    vector<string> sequence;
    vector<SidechainParams> params;

    RotamerSidechain(hid_t grp, vector<string> sequence_):
        n_restype(get_dset_size(1, grp, "restype_order")[0]), 
        interactions(n_restype*n_restype),

        rotamer_indices(n_restype),
        backbone_location_and_prob_fcn(
                get_dset_size(3,grp, "rotamer_prob")[2],
                get_dset_size(3,grp, "rotamer_prob")[0],
                get_dset_size(3,grp, "rotamer_prob")[1])
    {
        check_size(grp, "energy", n_restype, n_restype, 2);  // 2 is for inner or outer energy
        check_size(grp, "radius", n_restype, n_restype, 2); 
        check_size(grp, "width",  n_restype, n_restype, 2);

        traverse_dset<3,float>(grp, "energy", [&](size_t rt1, size_t rt2, int is_outer, float x) {
                auto& p = interactions[rt1*n_restype+rt2];
                (is_outer ? p.energy_outer : p.energy_inner) = x;});

        traverse_dset<3,float>(grp, "radius", [&](size_t rt1, size_t rt2, int is_outer, float x) {
                auto& p = interactions[rt1*n_restype+rt2];
                (is_outer ? p.radius_outer : p.radius_inner) = x;});

        traverse_dset<3,float>(grp, "width", [&](size_t rt1, size_t rt2, int is_outer, float x) {
                auto& p = interactions[rt1*n_restype+rt2];
                (is_outer ? p.scale_outer  : p.scale_inner)  = 1.f/x;});

        for(auto& p: interactions) 
            p.update_cutoff2();

        for(int rt1: range(n_restype))
            for(int rt2: range(n_restype))
                if(interactions[rt1*n_restype + rt2] != interactions[rt2*n_restype + rt1]) 
                    throw string("interaction matrix must be symmetric");

        traverse_string_dset<1>(grp, "restype_order", [&](size_t idx, string nm) {index_from_restype[nm] = idx;});

        check_size(grp, "rotamer_start_stop", n_restype, 2);
        traverse_dset<2,int>(grp, "rotamer_start_stop", [&](size_t rt, size_t is_stop, int x) {
                (is_stop ? rotamer_indices[rt].stop : rotamer_indices[rt].start) = x;});
        // for(auto& p: rotamer_indices) p.stop = min(p.start+10,p.stop); // FIXME DEBUG

        auto& spl = backbone_location_and_prob_fcn;
        check_size(grp, "rotamer_prob",   spl.nx, spl.ny, spl.n_layer);
        check_size(grp, "rotamer_center", spl.nx, spl.ny, spl.n_layer, 3);

        vector<double> data_to_fit(spl.n_layer*spl.nx*spl.ny*4);
        traverse_dset<4,double>(grp, "rotamer_center", [&](size_t ix, size_t iy, size_t i_pt, size_t d, double x) {
                data_to_fit.at(((i_pt*spl.nx + ix)*spl.ny + iy)*4 + d) = x;});
        traverse_dset<3,double>(grp, "rotamer_prob",   [&](size_t ix, size_t iy, size_t i_pt,           double x) {
                data_to_fit.at(((i_pt*spl.nx + ix)*spl.ny + iy)*4 + 3) = x;});

        spl.fit_spline(data_to_fit.data());

        // traverse_string_dset<1>(grp, "sequence", [&](size_t nr, string resname) {sequence.push_back(resname);});
        set_sequence(sequence_);
    }

    void set_sequence(const vector<string> sequence_) {
        sequence = sequence_;
        params.clear();

        int n_rot = 0;
        for(auto&& s: sequence) {
            int rt = index_from_restype[s];
            params.emplace_back();
            params.back().restype = rt;
            params.back().start_loc = n_rot;
            n_rot += rotamer_indices[rt].stop-rotamer_indices[rt].start;
        }

        // add dummy record for last stop location
        params.emplace_back();
        params.back().restype = -1;
        params.back().start_loc = n_rot;
    }


    vector<float> rotamer_prob(PairHistogram& hist, char* inference_string, VecArray rigid_body, VecArray rama) {
        auto tstart = chrono::high_resolution_clock::now();
        int n_rotamer = params.back().start_loc;
        vector<float> loc_and_prob_storage(n_rotamer*4, 0.f);
        VecArray loc_and_prob(loc_and_prob_storage.data(), n_rotamer);

        // add a litte paranoia to make sure there are no rounding problems
        const float scale_x = backbone_location_and_prob_fcn.nx * (0.5f/M_PI_F - 1e-7f);
        const float scale_y = backbone_location_and_prob_fcn.ny * (0.5f/M_PI_F - 1e-7f);
        const float shift = M_PI_F;

        int n_rot_placed = 0;
        for(int nr=0; nr<int(params.size()-1); ++nr) {
            Vec<7> body = load_vec<7>(rigid_body, nr);
            float U[9]; quat_to_rot(U, body.v+3);
            float3 t = make_vec3(body[0], body[1], body[2]);

            params[nr].start_loc = n_rot_placed;
            float3 avg_pos = make_vec3(0.f, 0.f, 0.f); float total_prob = 0.f;
            auto inds = rotamer_indices[params[nr].restype];
            for(int no=inds.start; no<inds.stop; ++no) {
                float result[12];
                backbone_location_and_prob_fcn.evaluate_value_and_deriv(result, no, 
                        (rama(0,nr)+shift)*scale_x, (rama(1,nr)+shift)*scale_y);
                float3 pt  = apply_affine(U,t, make_vec3(result[3*0+2], result[3*1+2], result[3*2+2]));
                float prob = result[3*3+2];

                avg_pos += prob * pt;
                total_prob += prob;

                if(prob>0.01f) {
                    loc_and_prob(0,n_rot_placed) = pt[0];
                    loc_and_prob(1,n_rot_placed) = pt[1];
                    loc_and_prob(2,n_rot_placed) = pt[2];
                    loc_and_prob(3,n_rot_placed) = prob;
                    ++n_rot_placed;
                } 
            }
            // loc_and_prob(0,n_rot_placed) = avg_pos[0];
            // loc_and_prob(1,n_rot_placed) = avg_pos[1];
            // loc_and_prob(2,n_rot_placed) = avg_pos[2];
            // loc_and_prob(3,n_rot_placed) = total_prob;
            // ++n_rot_placed;
        }
        params.back().start_loc = n_rot_placed;

        auto tfinish = chrono::high_resolution_clock::now();
        printf("place %6.1f (%6.2f rot/res) ", chrono::duration<double>(tfinish-tstart).count()*1e6, n_rot_placed*1./(params.size()-1));

        vector<float> marginal_prob(n_rot_placed);
        // try{
        rotamer_sidechain(hist, marginal_prob.data(), inference_string, 
                loc_and_prob, params.data(), interactions.data(), n_rotamer, n_restype,  params.size()-1);
        // }catch(dai::Exception& e) {
        //     return marginal_prob;
        // }
        printf("finished\n");
        // for(int i: range(marginal_prob.size())) 
        //     printf("%.4f %.4f\n", marginal_prob[i], loc_and_prob(3,i));

        return marginal_prob;
    }
};


int main(int argc, char** argv) try {
    if(argc!=5) throw string("Improper number of arguments");
    char* potential_fname  = argv[1];
    char* structure_fname  = argv[2];
    char* output_fname     = argv[3];
    char* inference_string = argv[4];

    auto config     = open_file(potential_fname, H5F_ACC_RDONLY);
    auto structures = open_file(structure_fname, H5F_ACC_RDONLY);

    // initialize with fake sequence
    RotamerSidechain node(config.get(), vector<string>());

    float ref_geom[9] = {-1.21261231f, -0.26328702f,  0.f, 
                         -0.019807f  ,  0.56798484f,  0.f,
                          1.23241932f, -0.30469782f,  0.f};

    int n_processed = 0;
    PairHistogram hist(node.n_restype, 150, 0.1);
    for(const auto& structure_name: node_names_in_group(structures.get(), "/")) {
        auto structure_grp = open_group(structures.get(), structure_name.c_str());
        // if(get_dset_size(1, structure_grp.get(), "sequence")[0] >= 100) continue;

        vector<string> sequence;
        traverse_string_dset<1>(structure_grp.get(), "sequence", [&](size_t rt, string nm) {sequence.push_back(nm);});
        node.set_sequence(sequence);
        int n_residue = sequence.size();
        printf("%4i %s %3i ", n_processed++, structure_name.c_str(), n_residue);
        fflush(stdout);

        // FIXME load atoms and align them
        vector<float> rigid_body_data(n_residue*7);
        VecArray rigid_body(rigid_body_data.data(), n_residue);

        check_size(structure_grp.get(), "pos", n_residue*3, 3);
        float atom[9];
        traverse_dset<2,float>(structure_grp.get(), "pos", [&](size_t na, size_t d, float x) {
                int idx = (na%3)*3+d;
                atom[idx] = x;
                if(idx==8) {  // once 3 atoms are loaded, we can do the alignment
                    float r[7];
                    float dr[3*3*7];
                    three_atom_alignment(r, dr, atom+0, atom+3, atom+6, ref_geom);
                    for(int i: range(7)) rigid_body(i,na/3) = r[i];
                }});

        vector<float> rama_data(n_residue*2);
        VecArray rama(rama_data.data(), n_residue);
        check_size(structure_grp.get(), "rama", sequence.size(), 2);
        traverse_dset<2,float>(structure_grp.get(), "rama", [&](size_t nr, size_t d, float x) {
                rama(d,nr) = x;});

        auto marginal_prob = node.rotamer_prob(hist, inference_string, rigid_body, rama);
        // if(structure_name == "prot_1B4FA") break;
    }

    {
        auto out = open_file(output_fname, H5F_ACC_TRUNC);
        auto hist_array = create_earray(out.get(), "pair_histogram", H5T_NATIVE_DOUBLE, {0, hist.n_type, hist.n_bin}, {1,1,hist.n_bin});
        append_to_dset(hist_array.get(), hist.count, 0);

        auto bin_edges = vector<double>(hist.n_bin+1);
        for(int i: range(hist.n_bin+1)) bin_edges[i] = i*hist.dx;
        auto bin_edges_array = create_earray(out.get(), "bin_edges", H5T_NATIVE_DOUBLE, {0}, {hist.n_bin+1});
        append_to_dset(bin_edges_array.get(), bin_edges, 0);
    }

    return 0;
} catch (const string& e) {
    fprintf(stderr, "\n\nERROR: %s\n", e.c_str());
    return 1;
}
