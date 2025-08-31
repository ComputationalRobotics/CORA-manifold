#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "StiefelProduct.h"
#include "ScaledStiefelProduct.h"

int main() {
  using CORA::Matrix;
  using CORA::Vector;
  using CORA::Scalar;

  // ---- 简单尺寸：p>=k ----
  const size_t p = 3;   // 行
  const size_t k = 3;   // 每块列数
  const size_t n = 5;   // 块数
  CORA::ScaledStiefelProduct M(k, p, n);

  std::default_random_engine gen(123);
  std::normal_distribution<Scalar> N01(0.0, 1.0);
  Matrix A_R(p, k*n);
  for (int i = 0; i < p; ++i)
    for (int j = 0; j < (k*n); ++j)
      A_R(i,j) = i + j + 1;

  Matrix V_R(p, k*n);
  for (int i = 0; i < p; ++i)
    for (int j = 0; j < (k*n); ++j)
      V_R(i,j) = 1;

  Vector A_s(n);
  for (int i = 0; i < n; ++i)
    A_s(i) = i + 1;

  Vector V_s(n);
  V_s << 1, -1, 1, -1, 1;

  Vector X = M.projectSToPositive_Tangent(A_s,V_s);

  std::cout << X << std::endl;

  Matrix Y = M.projectRToStiefel_Tangent(A_R, V_R);

  std::cout << Y << std::endl;

  auto Z = M.random_sample(1);

  std::cout << Z.R << std::endl;
  std::cout << Z.s << std::endl;

  return 0;
}
