/** \file operational_law.hpp
 * \brief Offline re-expression of the existing Legacy scaled control algebra on explicit causal coefficients.
 */
#ifndef loPredCtrlAcc_operational_law_hpp
#define loPredCtrlAcc_operational_law_hpp

#include "../ar_controller.hpp"

namespace DDSPC
{
/// Return canonical future-input gain using Legacy scaling and its separate current-WFS feedback term.
/** This does not reuse or repair Legacy histories, accumulate outputs, change a learner, or publish commands. */
Matrix
operational_legacy_gain( const Matrix &theta /**< [in] Canonical [A B D C] fit, newest-first. */,
                         int           modes /**< [in] Controlled residual/input modes. */,
                         int           history /**< [in] WFS history; past input count is history minus one. */,
                         realT regularization /**< [in] Legacy dimensionless factor multiplying maxCoeff(C^T C). */,
                         realT feedback /**< [in] Added current-WFS gain, not a predictive-gain multiplier. */ );
} //namespace DDSPC
#endif
