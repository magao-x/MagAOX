/** \file ddspc_comparison_test.cpp
 * \brief Matched-plant comparisons including negative controls for uninformative sensors.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "comparison.hpp"
#include <cmath>
#include <iostream>

namespace libXWCTest::loPredCtrlAccTest
{
/// Compare frozen fits and synthetic feedback under identical training inputs and exogenous realizations.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Matched causal replay compares all three DDSPC formulations", "[ddspc][comparison]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::PredictiveController::append_sample();
    DDSPC::PredictiveController::update_system();
    DDSPC::PredictiveController::get_reference_controller();
#endif
    // clang-format on
    for( bool qrd : { false, true } )
        for( int horizon : { 1, 3 } )
            for( int scenario : { 0, 1, 2 } )
            {
                CAPTURE( qrd, horizon, scenario );
                const auto result = compare_models( horizon, scenario, qrd );
                std::cout << "Comparison qrd=" << qrd << " F=" << horizon << " scenario=" << scenario;
                for( int k = 0; k < 3; ++k )
                {
                    REQUIRE( std::isfinite( result.m_prediction[k] ) );
                    REQUIRE( std::isfinite( result.m_residual[k] ) );
                    REQUIRE( result.m_command[k] <= 1 );
                    std::cout << " | " << k << " pred=" << result.m_prediction[k]
                              << " residual=" << result.m_residual[k] << " applied=" << result.m_command[k];
                }
                std::cout << " Dnorm=" << result.m_dNorm << '\n';
                if( scenario == 0 )
                {
                    // Independent innovations beyond a_i remain unpredictable at F>1.
                    if( horizon == 1 )
                        REQUIRE( result.m_prediction[2] < 0.1 * result.m_prediction[1] );
                    REQUIRE( result.m_residual[2] < result.m_residual[1] );
                    REQUIRE( result.m_dNorm > 0.5 );
                }
                else
                {
                    REQUIRE( result.m_prediction[2] > 0.8 * result.m_prediction[1] );
                    REQUIRE( result.m_prediction[2] < 1.3 * result.m_prediction[1] );
                }
            }
}

/// Every reported metric must repeat under identical data and arithmetic.
/** \ingroup loPredCtrlAcc_unit_test */
TEST_CASE( "Offline DDSPC comparison is deterministic", "[ddspc][comparison]" )
{
    const auto first = compare_models( 3, 0, false ), second = compare_models( 3, 0, false );
    REQUIRE( first.m_prediction == second.m_prediction );
    REQUIRE( first.m_residual == second.m_residual );
    REQUIRE( first.m_command == second.m_command );
    REQUIRE( first.m_dNorm == second.m_dNorm );
}
} // namespace libXWCTest::loPredCtrlAccTest
