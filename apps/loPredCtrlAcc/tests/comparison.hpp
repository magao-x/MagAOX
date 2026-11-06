/** \file comparison.hpp
 * \brief Matched comparisons under an explicitly synthetic held-command plant.
 */
#ifndef loPredCtrlAcc_comparison_hpp
#define loPredCtrlAcc_comparison_hpp
#include <array>
namespace libXWCTest::loPredCtrlAccTest
{
/// Frozen-fit forecasts and separate simulated feedback outcomes, ordered Legacy, standard, assisted.
struct ComparisonResult
{
    std::array<double, 3> m_prediction{}; ///< Held-out conditional prediction RMSE.
    std::array<double, 3> m_residual{};   ///< Synthetic feedback residual RMS, identical exogenous realization.
    std::array<double, 3> m_command{};    ///< Actual synthetic absolute input RMS, including clipping/holds.
    double                m_dNorm{ 0 };   ///< Assisted D block norm.
};
/// Train on identical histories, then freeze fits; no application/stream/device interfaces are used.
ComparisonResult compare_models( int  horizon /**< [in] One or three decision steps. */,
                                 int  scenario /**< [in] 0: informative sensor, 1: D=0, 2: unrelated sensor. */,
                                 bool qrd /**< [in] QR rather than conventional RLS. */ );
} // namespace libXWCTest::loPredCtrlAccTest
#endif
