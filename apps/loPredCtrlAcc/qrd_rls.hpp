/** \file qrd_rls.hpp
 * \brief Multi-output QR information-factor RLS with an opt-in exact triangular invariant.
 */

#ifndef PC_QRDRLS_HPP
#define PC_QRDRLS_HPP

#include <Eigen/Dense>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
using namespace mx::improc;
#include "utils.hpp"
#include "recursive_least_squares.hpp"

namespace DDSPC
{

/// Retained method names; the existing update always uses Givens rotations.
enum class QRD_RLS_Method
{
    Householder,
    Givens
};

/// Recursive least squares sharing a QR information factor across output rows.
/** R^T R = gamma^k*rho*I + sum_j gamma^(k-j)*phi_j*phi_j^T.
 * The optional triangular invariant explicitly zeros eliminated entries so rounding remnants
 * cannot reenter later adjacent-row QR updates. Causal offline controllers enable it;
 * default construction preserves legacy numerics. Public legacy field names are retained for compatibility.
 */
class QRDRecursiveLeastSquares
{
  private:
    /// Preserve exact structural zeros after Givens elimination when explicitly enabled.
    bool m_enforceTriangular;

    /// Keep Legacy rotation arithmetic independent of the causal structural-zero branch.
    template <bool enforceTriangular /**< [in] Apply the causal structural-zero rule. */>
    void apply_givens_impl();

  public:
    /// Regression column count.
    int _num_features;

    /// Simultaneously fitted output row count.
    int _num_predictors;

    /// Per-observation forgetting factor.
    realT _gamma;

    /// Square root of the forgetting factor, applied to R.
    realT _lambda;

    /// Square root of the initial information strength.
    realT _delta;

    /// Prior information rho, despite the retained legacy name.
    realT _initial_covariance;

    /// Retained unused method selector; updates always use Givens rotations.
    QRD_RLS_Method method_;

    /// Upper triangular information factor, p by p.
    Matrix R_;

    /// Identified output-by-feature coefficients.
    Matrix prediction_matrix;

    /// Augmented [sqrt(gamma)*R; phi^T] workspace.
    Matrix A_;

    /// Pre-update prediction error shared with the image overload.
    Matrix err;

    /// Pre-update predicted output column.
    Matrix prediction_output;

    /// Intermediate solution of R^T z = phi.
    Matrix z;

    /// Posterior information-inverse times phi, equal to the RLS gain.
    Matrix g;

    /// Allocate a zero-weight model with R=sqrt(rho)*I.
    QRDRecursiveLeastSquares( int   num_predictors /**< [in] Output row count. */,
                              int   num_features /**< [in] Feature column count. */,
                              realT forgetting_factor /**< [in] Per-update gamma. */,
                              realT inverse_covariance /**< [in] Initial information strength rho. */,
                              bool  enforce_triangular = false /**< [in] Opt in to exact structural zeros. */ );

    /// Release Eigen-owned workspaces.
    ~QRDRecursiveLeastSquares();

    /// Calculate a stable two-row elimination rotation.
    void givens( realT  a /**< [in] Upper entry. */,
                 realT  b /**< [in] Entry to eliminate. */,
                 realT &c /**< [out] Cosine. */,
                 realT &s /**< [out] Signed sine. */ );

    /// Dispatch once to Legacy or causal adjacent-row rotations, preserving each arithmetic path.
    void apply_givens();

    /// Forward solve R^T z=x using the information factor.
    void solve_Rt( Matrix &x /**< [in] Regressor column. */, Matrix &z /**< [out] Forward solution. */ );

    /// Back solve R*g=z using the information factor.
    void solve_R( Matrix &z /**< [in] Forward solution. */, Matrix &g /**< [out] Gain column. */ );

    /// Update from Eigen image columns with the same RLS equations.
    void update( eigenImage<realT> *x /**< [in] Regressor column. */,
                 eigenImage<realT> *y /**< [in] Target column. */ );

    /// Update from matrix columns with one information factor shared by all outputs.
    void update( Matrix *x /**< [in] Regressor column. */, Matrix *y /**< [in] Target column. */ );

    /// Update with the same rotations/solves using preallocated causal rank-one scratch.
    void update_preallocated( Matrix &x /**< [in] Finite regressor; mutable type only for retained solve interfaces. */,
                              const Matrix &y /**< [in] Finite target. */,
                              Matrix       &weightUpdate /**< [in,out] Output-by-feature scratch. */ );

    /// Reset weights and restore the initial information factor.
    void reset();

    /// Compatibility stub that prints the filename but does not save state.
    void save_state( const std::string &filename /**< [in] Requested legacy basename. */ );

    /// Compatibility stub that prints the filename but does not restore state.
    void load_state( const std::string &filename /**< [in] Requested legacy basename. */ );
};

} // namespace DDSPC

#endif // PC_QRDRLS_HPP
