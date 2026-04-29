/** \file ogTracker_test.cpp
 * \brief Catch2 tests for the ogTracker app.
 *
 * \ingroup ogTracker_unit_test
 */
#include "../../../tests/catch2/catch.hpp"
#include "../../../tests/testXWC.hpp"

#include "../ogTracker.hpp"

/** \defgroup ogTracker_unit_test ogTracker Unit Tests
 * \brief Unit tests for the `ogTracker` application helper logic.
 *
 * \ingroup application_unit_test
 */

namespace libXWCTest
{
namespace ogTrackerTest
{

/** \brief Verify sparkle parameters map to the expected calibration folder name. */
TEST_CASE( "calibration folder naming is stable", "[ogTracker]" )
{
    const std::string folder = MagAOX::app::ogTracker::formatCalibFolder( 22.8f, 45.2f, 0.02f, 2000.0f );
    REQUIRE( folder == "sep22_ang45_amp0.020_freq2000" );
}

/** \brief Verify circular-buffer window indexing wraps correctly at the beginning of the ring. */
TEST_CASE( "circular window start index wraps correctly", "[ogTracker]" )
{
    REQUIRE( MagAOX::app::ogTracker::cbWindowStartIndex( 2, 3, 10 ) == 0 );
    REQUIRE( MagAOX::app::ogTracker::cbWindowStartIndex( 0, 4, 10 ) == 7 );
}

/** \brief Verify temporal ordering is preserved when reading from a wrapped pointer circular buffer. */
TEST_CASE( "pointer circular-buffer extraction follows temporal order", "[ogTracker]" )
{
    using cbT = MagAOX::app::ogTracker::frameCircBuffT;

    cbT cb;
    cb.maxEntries( 4 );

    std::vector<std::vector<float>> frames( 5, std::vector<float>( 1, 0.0f ) );
    for( int i = 0; i < 5; ++i )
    {
        frames[static_cast<size_t>( i )][0] = static_cast<float>( i );
        cb.nextEntry( frames[static_cast<size_t>( i )].data() );
    }

    const int count  = static_cast<int>( cb.size() );
    const int latest = static_cast<int>( cb.latest() );
    REQUIRE( count == 4 );

    const int start = MagAOX::app::ogTracker::cbWindowStartIndex( latest, count, count );
    REQUIRE( ( *cb.at( static_cast<MagAOX::app::ogTracker::cbIndexT>( start ), 0 ) ) == Approx( 1.0f ) );
    REQUIRE( ( *cb.at( static_cast<MagAOX::app::ogTracker::cbIndexT>( start ), 1 ) ) == Approx( 2.0f ) );
    REQUIRE( ( *cb.at( static_cast<MagAOX::app::ogTracker::cbIndexT>( start ), 2 ) ) == Approx( 3.0f ) );
    REQUIRE( ( *cb.at( static_cast<MagAOX::app::ogTracker::cbIndexT>( start ), 3 ) ) == Approx( 4.0f ) );
}

/** \brief Verify per-mode RMS and normalization follow reference RMS scaling. */
TEST_CASE( "RMS normalization follows ref_rms scaling", "[ogTracker]" )
{
    Eigen::MatrixXf proj( 3, 2 );
    proj << 1.0f, 2.0f, -1.0f, -2.0f, 1.0f, 2.0f;

    const Eigen::VectorXf rms = MagAOX::app::ogTracker::rmsPerMode( proj );
    REQUIRE( rms.size() == 2 );
    REQUIRE( rms[0] == Approx( 1.0f ) );
    REQUIRE( rms[1] == Approx( 2.0f ) );

    Eigen::VectorXf ref( 2 );
    ref << 0.5f, 4.0f;
    const Eigen::VectorXf norm = MagAOX::app::ogTracker::normalizeByReference( rms, ref, 1e-8f );
    REQUIRE( norm[0] == Approx( 2.0f ) );
    REQUIRE( norm[1] == Approx( 0.5f ) );
}

} // namespace ogTrackerTest
} // namespace libXWCTest
