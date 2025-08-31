/**
 * ScaledStiefelProduct: product manifold (Positive^n) × (St(k,p))^n
 *
 * Storage format:
 *   - Point X = { s ∈ R_{>0}^n, R ∈ St(k,p)^n } stored separately (no concatenated sR).
 *   - Tangent T = { ds ∈ R^n, Z ∈ T_R St(k,p)^n } stored separately.
 *
 * Core API (no decomposition from a concatenated Y):
 *   - projectRToStiefel(A_R): project arbitrary R-part to (St(k,p))^n (blockwise).
 *   - projectSToPositive(a_s): project arbitrary scale vector to R_{>0}^n.
 *   - projectToManifold(A_R, a_s): project both parts at once → Point {s, R}.
 *   - projectToTangentSpace(X, V_R, v_s): ambient pair → Tangent {ds, Z}.
 *   - random_sample(): sample {s, R} with s>0 and R ∈ St(k,p)^n.
 *   - assemble(X): (optional) build concatenated Y = [ s_i R_i ] ∈ R^{p×(k n)}.
 *
 * Notes:
 *   - Stiefel blocks are p×k with orthonormal columns.
 *   - Positive part uses a simple truncation projection; swap in log-domain if desired.
 *   - SymBlockDiagProduct acts on the R-part and is kept for compatibility with
 *     your existing algebra (e.g., SE-Sync style symmetric block-diagonal ops).
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <random>
#include <limits>
#include <cassert>

#include "CORA_types.h"
#include "MatrixManifold.h"

namespace CORA {

class ScaledStiefelProduct {
public:
  using Index = Eigen::Index;

  struct Point {
    Vector s;  // length n_
    Matrix R;  // p_ * k_ * n_, block i is p x k
  };

  struct Tangent {
    Vector ds; // length n_
    Matrix Z;  // p_ * k_ * n_), blockwise Stiefel tangents
  };

private:
  size_t k_{}; // columns per Stiefel frame
  size_t p_{}; // ambient dimension
  size_t n_{}; // number of blocks

public:
  // Constructors / mutators / accessors
  ScaledStiefelProduct() = default;
  ScaledStiefelProduct(size_t k, size_t p, size_t n) : k_(k), p_(p), n_(n) {}

  void set_k(size_t k) { k_ = k; }
  void set_p(size_t p) { p_ = p; }
  void set_n(size_t n) { n_ = n; }
  void addNewFrame() { n_++; }
  void incrementRank() { p_++; }
  void setRank(size_t p) { p_ = p; }

  size_t get_k() const { return k_; }
  size_t get_p() const { return p_; }
  size_t get_n() const { return n_; }

  // ===== Product-manifold API (separate storage) =====

  // Project R-part to (St(k,p))^n, block by block.
  Matrix projectRToStiefel(const Matrix& A_R) const;

  // Project scale vector to Positive^n by truncation (replace ≤eps with eps).
  // Replace with a log-domain map if you want a geodesically natural projection.
  Vector projectSToPositive(const Vector& A_s, const Vector& V_s) const;

  // Project both parts at once to the product manifold.
  Point projectToManifold(const Matrix& A_R, const Vector& A_s, const Vector& V_s) const {
    Point X;
    X.R = projectRToStiefel(A_R);
    X.s = projectSToPositive(A_s, V_s);
    return X;
  }

  Matrix SymBlockDiagProduct(const Matrix &A, const Matrix &BT, const Matrix &C) const;

  //  Project R-part to Tangent space of A_R, block by block.
  Matrix projectRToStiefel_Tangent(const Matrix& A_R, const Matrix& V_R) const {
    return V_R - SymBlockDiagProduct(A_R, A_R.transpose(), V_R);
  };

  // The same for s-part
  Vector projectSToPositive_Tangent(const Matrix& A_s, const Matrix& V_s) const{
    return V_s;
  };

  Point projectToTangentSpace(const Point& A, const Point& V) const{
    Point X;
    X.R = projectRToStiefel_Tangent(A.R, V.R);
    X.s = projectSToPositive_Tangent(A.s, V.s);
    return X;
  };

  // Random sample on the product manifold:
  Point random_sample(const std::default_random_engine::result_type& seed =
                        std::default_random_engine::default_seed) const;

};

} // namespace CORA
