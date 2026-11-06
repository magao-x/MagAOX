/** \file recursive_least_squares.hpp
 * \brief Operational multi-output RLS and an opt-in fixed-workspace causal update.
 */
#ifndef PCRLS_HPP
#define PCRLS_HPP

#include <Eigen/Dense>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
using namespace mx::improc;
#include "utils.hpp"

namespace DDSPC
{

/// Shared covariance for multi-output identification; public legacy names preserve compatibility.
class RecursiveLeastSquares
{
  public:
    realT  _gamma;              ///< Per-update forgetting factor.
    realT  _inverse_gamma;      ///< Reciprocal forgetting factor.
    realT  _initial_covariance; ///< Initial information strength, with P0=I/rho.
    int    _num_features;       ///< Regressor dimension.
    int    _num_predictors;     ///< Output dimension.
    Matrix K;                   ///< Row gain shared by all outputs.
    Matrix err;                 ///< A-priori prediction error.
    Matrix xtP;                 ///< Scaled regressor transpose times covariance.
    Matrix prediction_matrix;   ///< Identified output-by-feature weights.
    Matrix inverse_covariance;  ///< Shared inverse information matrix.
    Matrix prediction_output;   ///< Reusable output prediction in the opt-in causal path.

    /// Allocate zero weights and P0=I/rho.
    RecursiveLeastSquares( int   num_predictors /**< [in] Output rows. */,
                           int   num_features /**< [in] Feature rows. */,
                           realT forgetting_factor /**< [in] Per-update gamma. */,
                           realT initial_covariance /**< [in] Information strength rho. */ );

    /// Release Eigen-owned state.
    ~RecursiveLeastSquares();

    /// Preserve the operational image-column update and its arithmetic.
    void update( eigenImage<realT> *x /**< [in] Regressor. */, eigenImage<realT> *y /**< [in] Target. */ );

    /// Preserve the operational matrix-column update and its arithmetic.
    void update( Matrix *x /**< [in] Regressor. */, Matrix *y /**< [in] Target. */ );

    /// Apply the same RLS equations with caller-owned fixed scratch; causal opt-in only.
    void update_preallocated( const Matrix &x /**< [in] Finite feature column. */,
                              const Matrix &y /**< [in] Finite target column. */,
                              Matrix       &normalization /**< [in,out] One-by-one normalization-product scratch. */,
                              Matrix       &weightUpdate /**< [in,out] Output-by-feature scratch. */,
                              Matrix       &covarianceUpdate /**< [in,out] Feature-square scratch. */ );

    /// Restore zero weights and the original prior.
    void reset();

    /// Write the unchanged legacy numerical artifact.
    void save_state( const std::string &filename /**< [in] Artifact basename. */ );

    /// Load the unchanged legacy numerical artifact.
    void load_state( const std::string &filename /**< [in] Artifact basename. */ );
};

} // namespace DDSPC
#endif // PCRLS_HPP
