#include <Eigen/Dense>
#include <string>
#include <array>
#include <map>
#include <cmath>

typedef void (*ResFuncPtr)(Eigen::MatrixX3f& ret, float psi, const std::array<float,4> &chi);
std::map<std::string,ResFuncPtr>& res_func_map();
