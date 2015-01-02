#!/usr/bin/env python

import tables as tb
import numpy as np
import sys

t = tb.open_file('/rust/work/residues.h5')

def dihedral(x1,x2,x3,x4):
    '''four atom dihedral angle in radians'''
    b1 = x2-x1
    b2 = x3-x2
    b3 = x4-x3
    b2b3 = np.cross(b2,b3)
    b2mag = np.sqrt(np.sum(b2**2, axis=-1))
    return np.arctan2(
            b2mag * (b1*b2b3).sum(axis=-1),
            (np.cross(b1,b2) * b2b3).sum(axis=-1))

def angle(x,y,z):
    r1 = x-y; r1 *= 1./np.sqrt(np.sum(r1**2,axis=-1))[...,None]
    r2 = z-y; r2 *= 1./np.sqrt(np.sum(r2**2,axis=-1))[...,None]
    return np.arccos(np.sum(r1*r2,axis=-1))

deg = np.pi/180.

def ga(resname, aname):
    g = t.get_node('/residues/%s'%resname)

    # handle ARG as a special case because of the symmetry of NH1 and NH2
    # ensure last chi angle is < pi/2 to break symmetry
    if resname == 'ARG' and aname in ('NH1','NH2'):
        CDi = g._v_attrs.atoms.index('CD')
        NEi = g._v_attrs.atoms.index('NE')
        CZi = g._v_attrs.atoms.index('CZ')
        NH1 = g.pos[:,g._v_attrs.atoms.index('NH1')]
        NH2 = g.pos[:,g._v_attrs.atoms.index('NH2')]

        Nchi = dihedral(g.pos[:,CDi], g.pos[:,NEi], g.pos[:,CZi], NH1)
        good = np.abs(Nchi) < np.pi/2
        if aname == 'NH1': return np.where(good[:,None], NH1, NH2)
        if aname == 'NH2': return np.where(good[:,None], NH2, NH1)

    anum = g._v_attrs.atoms.index(aname)
    return g.pos[:,anum]


def dist(x,y):
    return np.sqrt(np.sum((x-y)**2,axis=-1))

def sdist(resname, atoms):
    l,r = atoms.split()
    return dist(ga(resname,l),ga(resname,r))

def angle_center(theta):
    theta = np.asarray(theta)
    theta = theta % (2*np.pi)
    theta[theta>np.pi] -= 2*np.pi
    return theta


def find_graph(resname):
    g = t.get_node('/residues/'+resname)
    atoms = map(str,g._v_attrs.atoms[:])
    r = resname

    order = [0,1]
    edges = [(0,1)]

    avg_dist_func = lambda i,j: np.median(dist(ga(resname,atoms[i]), ga(resname,atoms[j])))

    while len(order) < len(atoms):
        assert len(edges) == len(order)-1

        for curr_atom_num in order:
            curr_atom = atoms[curr_atom_num]
            if curr_atom == 'N': continue  # otherwise proline is a problem
            for i,a1 in enumerate(atoms):
                if i in order: continue
                avg_dist = avg_dist_func(curr_atom_num, i)
                # if r == 'PRO': print '%s %s %s %.2f' % (resname, curr_atom, a1, avg_dist)
                is_bond = avg_dist < 1.9

                if is_bond: 
                    order.append(i)
                    edges.append((curr_atom_num,i))


    # for i,j in edges:
    #     print '%s -> %s %.2f' % (atoms[i],atoms[j], avg_dist_func(i,j))

    # for i1,j1 in edges:
    #     for i2,j2 in edges:
    #         if j1!=i2: continue
    #         print '%s -> %s -> %s %.1f' % (atoms[i1], atoms[i2], atoms[j2], 
    #                 np.median(angle(ga(r,atoms[i1]), ga(r,atoms[i2]), ga(r,atoms[j2])))/deg)

    print >>sys.stderr, 'if(name == "%s") {' % resname
    chis = dict()
    placed_atoms = 5
    for i1,j1 in edges:
        # if r == 'PRO': print '%s -> %s' % (atoms[i1],atoms[j1])
        for i2,j2 in edges:
            if j1!=i2: continue
            for i3,i4 in edges:
                if j2!=i3: continue

                a1,a2,a3,a4 = [atoms[i] for i in (i1,i2,i3,i4)]
                if a4 == 'O': continue
                x1,x2,x3,x4 = [ga(r,a)  for a in (a1,a2,a3,a4)]

                # print '%3s -> %3s -> %3s -> %3s' % (a1,a2,a3,a4)
                bonds  = dist (   x3,x4); bond_dist  = np.mean(bonds);  bond_sd  = np.std(bonds); del bonds
                angles = angle(x2,x3,x4); angle_dist = np.mean(angles); angle_sd = np.std(angles); del angles
                dihes  = np.exp(1j * dihedral(x1,x2,x3,x4))
                dihe_dist = np.angle(dihes.mean())
                dihes_centered = np.angle(dihes * np.exp(-1j * dihe_dist))
                dihe_sd  = np.std(dihes_centered)

                if dihe_sd > 15.*deg:   # detect rotatable bond
                    chi_text = ''
                    for chi_name, chi_dihe in sorted(chis.items()):
                        dev = dihes * np.conjugate(chi_dihe)
                        dev_dist = np.angle(dev.mean())
                        dev_sd = np.std(np.angle(dev * np.exp(-1j * dev_dist)))
                        if dev_sd < 15.*deg:
                            chi_text = '%s%s%5.1ff*deg' % (chi_name, ('-' if dev_dist < 0. else '+'), np.abs(dev_dist/deg))
                            dihe_sd = dev_sd
                            break
                    if not chi_text:
                        chi_text = 'chi[%i]' % len(chis)
                        chis[chi_text] = dihes
                        print >>sys.stderr, '    %s = dihedral(pos.row(%i), pos.row(%i), pos.row(%i), pos.row(%i));' % (chi_text, i1,i2,i3,i4)
                else:
                    chi_text = '% 6.1ff*deg' % (dihe_dist/deg)

                print '    Affine3f %-3s = %-3s*make_tab(%17s, %.1ff*deg, %.2ff); ret.row(%2i) = %-3s.translation().transpose(); // dev %4.1f %.1f %.2f' %(
                        a4, a3, chi_text, angle_dist/deg, bond_dist,  placed_atoms, a4, dihe_sd/deg, angle_sd/deg, bond_sd)
                placed_atoms += 1
    print >>sys.stderr, '}\n'

def main():
    print r'''#include <Eigen/Dense>
#include <string>
#include <array>
#include <map>
#include <cmath>
#include <Eigen/Geometry>

#include <iostream>

using namespace std;
using namespace Eigen;

static const float deg = 4.*atan(1.) / 180.;

inline Affine3f make_tab(float phi, float theta, float bond) 
{
    Affine3f out;
    float cp(cos(phi)),   sp(sin(phi));
    float ct(cos(theta)), st(sin(theta));
    float l(bond);

    out(0,0)=   -ct; out(0,1)=    -st; out(0,2)=   0; out(0,3)=   -l*ct;
    out(1,0)= cp*st; out(1,1)= -cp*ct; out(1,2)= -sp; out(1,3)= l*cp*st;
    out(2,0)= sp*st; out(2,1)= -sp*ct; out(2,2)=  cp; out(2,3)= l*sp*st;
    //  out(3,0)=     0; out(3,1)=      0; out(3,2)=   0; out(3,3)=       1;

    return out;
}

Affine3f place_bb(MatrixX3f& ret, float psi, bool include_CB=true) {
    Affine3f a = Affine3f::Identity();
    a(0,0)=0.8191292; a(0,1)=-0.3103239; a(0,2)= 0.4824173; a(0,3)=-1.2079210;
    a(1,0)=0.5736088; a(1,1)= 0.4423396; a(1,2)=-0.6894263; a(1,3)=-0.2636016;
    a(2,0)=0.0005532; a(2,1)= 0.8414480; a(2,2)= 0.5403378; a(2,3)=-0.0009170;

    Affine3f N  = a *make_tab(          0.f,        0.f,   0.f);  ret.row(0) = N .translation().transpose();
    Affine3f CA = N *make_tab(          0.f,        0.f, 1.45f);  ret.row(1) = CA.translation().transpose();
    Affine3f C  = CA*make_tab(   122.7f*deg, 110.3f*deg, 1.53f);  ret.row(2) = C .translation().transpose();
    Affine3f O  = C *make_tab(psi+180.f*deg, 120.5f*deg, 1.23f);  ret.row(3) = O .translation().transpose();
    Affine3f CB = CA*make_tab(          0.f, 110.6f*deg, 1.53f);  if(include_CB) ret.row(4) = CB.translation().transpose();

    return CB;
}
'''
    all_resnames = 'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()
    for r in all_resnames:
        g = t.get_node('/residues/%s'%r)
        print 'static void %s_res_func(MatrixX3f& ret, float psi, const array<float,4> &chi) {' % r
        n_atom = len(g._v_attrs.atoms[:])
        print '    ret.resize(%i,3);' % n_atom
        if n_atom > 5:
            print '    Affine3f CB  = place_bb(ret, psi);'
        elif n_atom == 5:
            print '    place_bb(ret, psi);'
        elif n_atom == 4:
            print '    place_bb(ret, psi, false);'

        find_graph(r)
        print '}'
        print

    print '''
typedef void (*ResFuncPtr)(MatrixX3f& ret, float psi, const array<float,4> &chi);

map<string,ResFuncPtr>& res_func_map() {
    static map<string,ResFuncPtr> m;
    if(!m.size()) {'''
    for r in all_resnames:
        print '        m["%s"] = &%s_res_func;' % (r,r)
    print '''    }

    return m;
}
'''
    # N  = t.root.model_geom[0]
    # CA = t.root.model_geom[1]
    # C  = t.root.model_geom[2]
    # CB = t.root.model_geom[3]
    # print 'N -> CA %f' % dist(N,CA)
    # # print 'CA -> O %f' % dist(CA,O)
    # print 'N -> CA -> C  %f %f' % (dist(CA,C),angle(N,CA,C)/deg)
    # print 'N -> CA -> CB  %f %f' % (dist(CA,CB),angle(N,CA,CB)/deg)

if __name__ == '__main__':
    main()


t.close()
