/** \file ogTracker_test.cpp
 * \brief Catch2 tests for the ogTracker app.
 */
#include "../../../tests/catch2/catch.hpp"

#include "../ogTracker.hpp"

using namespace MagAOX::app;

namespace ogTracker_test
{

SCENARIO( "calibration folder naming is stable", "[ogTracker]" )
{
    GIVEN( "sparkle parameter values" )
    {
        const std::string folder = ogTracker::formatCalibFolder( 22.8f, 45.2f, 0.02f, 2000.0f );
        REQUIRE( folder == "sep22_ang45_amp0.020_freq2000" );
    }
}

SCENARIO( "ring buffer start index wraps correctly", "[ogTracker]" )
{
    REQUIRE( ogTracker::ringStartIndex( 3, 3, 10 ) == 0 );
    REQUIRE( ogTracker::ringStartIndex( 1, 4, 10 ) == 7 );
}

SCENARIO( "RMS normalization follows ref_rms scaling", "[ogTracker]" )
{
    Eigen::MatrixXf proj( 3, 2 );
    proj << 1.0f, 2.0f, -1.0f, -2.0f, 1.0f, 2.0f;

    const Eigen::VectorXf rms = ogTracker::rmsPerMode( proj );
    REQUIRE( rms.size() == 2 );
    REQUIRE( rms[0] == Approx( 1.0f ) );
    REQUIRE( rms[1] == Approx( 2.0f ) );

    Eigen::VectorXf ref( 2 );
    ref << 0.5f, 4.0f;
    const Eigen::VectorXf norm = ogTracker::normalizeByReference( rms, ref, 1e-8f );
    REQUIRE( norm[0] == Approx( 2.0f ) );
    REQUIRE( norm[1] == Approx( 0.5f ) );
}

} // namespace ogTracker_test

