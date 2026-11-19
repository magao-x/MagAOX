/** \file t2wOffloader_test.cpp
 * \brief Catch2 tests for the t2wOffloader app.
 *
 * \ingroup t2wOffloader_files
 */

#include "../../../tests/testXWC.hpp"

#include "../t2wOffloader.hpp"

#include <limits>

using namespace MagAOX::app;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace MagAOX
{
namespace app
{
/// Expose the tweeter offload state without opening any shmims.
class t2wOffloader_test : public t2wOffloader
{
  public:
    using t2wOffloader::m_deltavalDM;

    using t2wOffloader::m_fps;

    using t2wOffloader::m_syncoMainHz;

    using t2wOffloader::m_syncoSteps;

    using t2wOffloader::m_maxModes;

    using t2wOffloader::m_numModes;

    using t2wOffloader::m_tModesOrtho;

    using t2wOffloader::m_tweeter;

    /// Set up three orthonormal 4x4 tweeter modes and the per-mode offload changes.
    t2wOffloader_test()
    {
        m_maxModes = 3;

        m_tModesOrtho.resize( 4, 4, 3 );
        m_tModesOrtho.setZero();
        m_tModesOrtho.image( 0 )( 0, 0 ) = 1;
        m_tModesOrtho.image( 1 )( 1, 1 ) = 0.5; // a mode spread over several pixels
        m_tModesOrtho.image( 1 )( 1, 2 ) = 0.5;
        m_tModesOrtho.image( 1 )( 2, 1 ) = 0.5;
        m_tModesOrtho.image( 1 )( 2, 2 ) = 0.5;
        m_tModesOrtho.image( 2 )( 3, 3 ) = 1;

        m_deltavalDM.resize( 3, 1 );
        m_deltavalDM( 0, 0 ) = 0.5;
        m_deltavalDM( 1, 0 ) = -0.25;
        m_deltavalDM( 2, 0 ) = 2;

        m_tweeter.resize( 4, 4 );
    }
};
} // namespace app
} // namespace MagAOX
/// \endcond

namespace libXWCTest
{

/** \defgroup t2wOffloader_unit_test t2wOffloader Unit Tests
 * \brief Unit tests for the t2wOffloader application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `t2wOffloader` unit tests.
/** \ingroup t2wOffloader_unit_test
 */
namespace t2wOffloaderTest
{

/// Verify the placeholder t2wOffloader test harness instantiates the app cleanly.
/**
 * \ingroup t2wOffloader_unit_test
 */
TEST_CASE( "t2wOffloader placeholder harness instantiates the app", "[t2wOffloader]" )
{
// clang-format off
    #ifdef T2WOFFLOADER_TEST_DOXYGEN_REF
    t2wOffloader();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        t2wOffloader app;

        REQUIRE( true );
    }
}

/// Verify the modal offload change is back-projected onto the tweeter modes.
/** Checks the sum over modes, the m_numModes and m_maxModes limits, and that each call replaces the previous shape.
 *
 * \ingroup t2wOffloader_unit_test
 */
TEST_CASE( "t2wOffloader back-projects the offload change onto the tweeter", "[t2wOffloader]" )
{
// clang-format off
    #ifdef T2WOFFLOADER_TEST_DOXYGEN_REF
    t2wOffloader::calcTweeterOffload;
    #endif
    // clang-format on

    t2wOffloader_test app;

    SECTION( "all offloaded modes are summed" )
    {
        app.m_numModes = 3;
        app.calcTweeterOffload();

        REQUIRE( app.m_tweeter( 0, 0 ) == 0.5f );
        REQUIRE( app.m_tweeter( 1, 1 ) == -0.125f );
        REQUIRE( app.m_tweeter( 2, 2 ) == -0.125f );
        REQUIRE( app.m_tweeter( 3, 3 ) == 2.0f );
        REQUIRE( app.m_tweeter.sum() == 2.0f );
    }

    SECTION( "modes at or beyond m_numModes are excluded" )
    {
        app.m_numModes = 2;
        app.calcTweeterOffload();

        REQUIRE( app.m_tweeter( 0, 0 ) == 0.5f );
        REQUIRE( app.m_tweeter( 3, 3 ) == 0.0f );
    }

    SECTION( "modes at or beyond m_maxModes are excluded" )
    {
        app.m_numModes = 3;
        app.m_maxModes = 1;
        app.calcTweeterOffload();

        REQUIRE( app.m_tweeter( 0, 0 ) == 0.5f );
        REQUIRE( app.m_tweeter.sum() == 0.5f );
    }

    SECTION( "zero modes gives a zero shape" )
    {
        app.m_numModes = 0;
        app.calcTweeterOffload();

        REQUIRE( app.m_tweeter.abs().sum() == 0.0f );
    }

    SECTION( "each call replaces the previous shape" )
    {
        app.m_numModes = 3;
        app.calcTweeterOffload();
        app.calcTweeterOffload();

        REQUIRE( app.m_tweeter( 3, 3 ) == 2.0f );
        REQUIRE( app.m_tweeter.sum() == 2.0f );
    }
}

/// Verify the averager navg chosen for a requested offload rate.
/**
 * \ingroup t2wOffloader_unit_test
 */
TEST_CASE( "t2wOffloader chooses navg for a requested offload rate", "[t2wOffloader]" )
{
    // clang-format off
    #ifdef T2WOFFLOADER_TEST_DOXYGEN_REF
    t2wOffloader::navgForRate(0,0);
    #endif
    // clang-format on

    SECTION( "250 Hz at 2 kHz and at 8 kHz" )
    {
        REQUIRE( t2wOffloader::navgForRate( 2000, 250 ) == 8 );
        REQUIRE( t2wOffloader::navgForRate( 8000, 250 ) == 32 );
    }

    SECTION( "rates that do not divide the DM rate round to the nearest navg" )
    {
        REQUIRE( t2wOffloader::navgForRate( 2000, 300 ) == 7 );  // 285.7 Hz achieved
        REQUIRE( t2wOffloader::navgForRate( 8000, 300 ) == 27 ); // 296.3 Hz achieved
    }

    SECTION( "navg is at least 1" )
    {
        REQUIRE( t2wOffloader::navgForRate( 2000, 5000 ) == 1 );
    }

    SECTION( "a rate of 0 or an unknown DM rate leaves navg unmanaged" )
    {
        REQUIRE( t2wOffloader::navgForRate( 2000, 0 ) == 0 );
        REQUIRE( t2wOffloader::navgForRate( 0, 250 ) == 0 );
        REQUIRE( t2wOffloader::navgForRate( 2000, std::numeric_limits<float>::quiet_NaN() ) == 0 );
    }
}

/// Verify the DM command rate follows dmSyncopation's active steps.
/**
 * \ingroup t2wOffloader_unit_test
 */
TEST_CASE( "t2wOffloader DM rate follows dmSyncopation", "[t2wOffloader]" )
{
    // clang-format off
    #ifdef T2WOFFLOADER_TEST_DOXYGEN_REF
    t2wOffloader::dmRate();
    #endif
    // clang-format on

    t2wOffloader_test app;
    app.m_fps = 2000;

    SECTION( "synco inactive uses the camera rate" )
    {
        REQUIRE( app.dmRate() == 2000 );
    }

    SECTION( "synco active at 4 steps multiplies the main rate" )
    {
        app.m_syncoSteps  = 4;
        app.m_syncoMainHz = 2000;
        REQUIRE( app.dmRate() == 8000 );
        REQUIRE( t2wOffloader::navgForRate( app.dmRate(), 250 ) == 32 );
    }

    SECTION( "an unknown synco main rate falls back to the camera rate" )
    {
        app.m_syncoSteps  = 4;
        app.m_syncoMainHz = 0;
        REQUIRE( app.dmRate() == 2000 );
    }
}

} // namespace t2wOffloaderTest

} // namespace libXWCTest
