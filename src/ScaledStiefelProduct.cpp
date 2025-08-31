#include <Eigen/QR>
#include <iostream>

#include "ScaledStiefelProduct.h"
namespace CORA {

Matrix ScaledStiefelProduct::projectRToStiefel(const Matrix &A_R) const {
  // We use a generalization of the well-known SVD-based projection for the
  // orthogonal and special orthogonal groups; see for example Proposition 7
  // in the paper "Projection-Like Retractions on Matrix
  // Manifolds" by Absil and Malick.

  Matrix P(p_, k_ * n_);

  // check that A is the correct size
  if (A_R.rows() != p_ || A_R.cols() != k_ * n_) {
    throw std::runtime_error("Error in StiefelProduct::projectToManifold: "
                             "Shape of A is: " +
                             std::to_string(A_R.rows()) + " x " +
                             std::to_string(A_R.cols()) +
                             " but should be: " + std::to_string(p_) + " x " +
                             std::to_string(k_ * n_));
  }

  for (size_t i = 0; i < n_; ++i) {
    auto c0 = static_cast<Index>(i * k_);

    // 取第 i 个块 Ai \in R^{p_ x k_}
    Matrix Ai = A_R.block(0, c0, p_, k_);

    // thin QR（Householder），Q_full 是 p_ x p_ 的隐式表示
    Eigen::HouseholderQR<Matrix> qr(Ai);

    // 取 thin Q ：前 k_ 列
    Matrix Qi = qr.householderQ() * Matrix::Identity(p_, k_);

    // 取上三角 R 的 k_×k_ 部分
    Matrix Ri = qr.matrixQR().topLeftCorner(k_, k_)
                  .template triangularView<Eigen::Upper>();

    // 唯一化：把 R 的对角线调成非负，把符号乘到 Q 的列上
    Vector s = Vector::Ones(static_cast<Index>(k_));
    for (Index j = 0; j < static_cast<Index>(k_); ++j) {
      // 零按 +1 处理（与 MATLAB s(s==0)=1 一致）
      if (Ri(j,j) < Scalar(0)) s(j) = Scalar(-1);
    }

    // Qi <- Qi * diag(s)
    Qi.noalias() = Qi * s.asDiagonal();

    P.block(0, c0, p_, k_) = Qi;
  }

  return P; // 每块列正交，且“正对角”唯一化
}

Vector ScaledStiefelProduct::projectSToPositive(const Vector& A_s, const Vector& V_s) const {

    //check size
    if (A_s.rows() != n_){
        throw std::runtime_error("Error in ScaledStiefelProduct::projectSToPositive");
    }

    Vector P = (A_s.array() * (V_s.array() / A_s.array()).exp()).matrix();

    return P;
}

Matrix ScaledStiefelProduct::SymBlockDiagProduct(const Matrix &A, const Matrix &BT,
                                           const Matrix &C) const {
  // Preallocate result matrix
  Matrix R(p_, k_ * n_);
  Matrix P(k_, k_);
  Matrix S(k_, k_);

  for (auto i = 0; i < n_; ++i) {
    auto start_col = static_cast<Index>(i * k_);
    // Compute block product Bi' * Ci
    P = BT.block(start_col, 0, k_, p_) * C.block(0, start_col, p_, k_);
    // Symmetrize this block
    S = .5 * (P + P.transpose());
    // Compute Ai * S and set corresponding block of R
    R.block(0, start_col, p_, k_) = A.block(0, start_col, p_, k_) * S;
  }
  return R;
}

ScaledStiefelProduct::Point ScaledStiefelProduct::random_sample(
    const std::default_random_engine::result_type &seed) const {

  std::default_random_engine generator(seed);
  std::normal_distribution<Scalar> g;

  ScaledStiefelProduct::Point X;
  X.R.resize(static_cast<Index>(p_), static_cast<Index>(k_ * n_));
  X.s = Vector::Ones(static_cast<Index>(n_)); // or sample positive values

  for (size_t i = 0; i < n_; ++i) {
    X.s(i) = std::abs(g(generator));
  }
  for (size_t r = 0; r < p_; ++r)
    for (size_t c = 0; c < k_ * n_; ++c)
      X.R(static_cast<Index>(r), static_cast<Index>(c)) = g(generator);
  return projectToManifold(X.R,X.s,Vector::Zero(static_cast<Index>(n_)));
}
} // namespace CORA
