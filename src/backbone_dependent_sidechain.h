#ifndef BACKBONE_DEPENDENT_SIDECHAIN_H
#define BACKBONE_DEPENDENT_SIDECHAIN_H
#include "md.h"
#include "coord.h"

struct BackboneSCParam {
    CoordPair rama_residue;
    CoordPair alignment_residue;
    int       restype;
};

struct BackbonePointMap {
    int    n_bin;
    float* germ;
};

void backbone_dependent_point(
        SysArray   output,
        CoordArray rama,
        CoordArray alignment,
        BackboneSCParam* param,
        BackbonePointMap map,
        int n_term, int n_system);
#endif
