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

/// \cond
class t2wOffloaderHarness : public t2wOffloader
{
  public:
    /// Initialize one tweeter actuator and one in-memory woofer command pixel.
    t2wOffloaderHarness();

    /// Send an offloading switch command to the app.
    int setOffloading( bool enabled /**< [in] Requested offloading state. */ );

    /// Feed a single actuator value to the frame callback.
    int processActuator( float command /**< [in] Averaged tweeter actuator command. */ );

    /// Return the most recently published woofer actuator command.
    float wooferCommand() const;

  private:
    IMAGE_METADATA m_wooferMetadata{};   ///< Metadata for the in-memory output with no attached semaphores.
    float          m_wooferCommand{ 0 }; ///< Storage for the in-memory woofer command pixel.
};

t2wOffloaderHarness::t2wOffloaderHarness()
{
    m_configName = "test_t2wOffloader";
    createStandardIndiToggleSw( m_indiP_offloadToggle, "offload" );

    m_numModes = 0;
    m_gain     = 0.5F;
    m_width    = 1;
    m_height   = 1;
    m_typeSize = sizeof( float );

    m_twRespM.setOnes( 1, 1 );
    m_woofer.setZero( 1, 1 );

    m_dmWidth            = 1;
    m_dmHeight           = 1;
    m_dmStream           = {};
    m_dmStream.md        = &m_wooferMetadata;
    m_dmStream.array.raw = &m_wooferCommand;
}

int t2wOffloaderHarness::setOffloading( bool enabled )
{
    auto request = m_indiP_offloadToggle;
    request["toggle"].setSwitchState( enabled ? pcf::IndiElement::On : pcf::IndiElement::Off );
    return newCallBack_m_indiP_offloadToggle( request );
}

int t2wOffloaderHarness::processActuator( float command )
{
    return processImage( &command, dev::shmimT{} );
}

float t2wOffloaderHarness::wooferCommand() const
{
    return m_wooferCommand;
}
/// \endcond

/// Verify actuator offloading holds its correction while off and resumes integration from that correction.
/**
 * \ingroup t2wOffloader_unit_test
 */
TEST_CASE( "t2wOffloader resumes actuator integration after toggling off and on", "[t2wOffloader]" )
{
    // clang-format off
    #ifdef T2WOFFLOADER_TEST_DOXYGEN_REF
    t2wOffloader::newCallBack_m_indiP_offloadToggle( pcf::IndiProperty() );
    t2wOffloader::processImage( nullptr, dev::shmimT{} );
    #endif
    // clang-format on

    t2wOffloaderHarness app;

    REQUIRE( app.setOffloading( true ) == 0 );
    REQUIRE( app.processActuator( 2.0F ) == 0 );
    REQUIRE( app.wooferCommand() == Approx( 1.0F ) );

    REQUIRE( app.setOffloading( false ) == 0 );
    REQUIRE( app.processActuator( 6.0F ) == 0 );
    CHECK( app.wooferCommand() == Approx( 1.0F ) );

    REQUIRE( app.setOffloading( true ) == 0 );
    REQUIRE( app.processActuator( 0.0F ) == 0 );
    CHECK( app.wooferCommand() == Approx( 1.0F ) );

    REQUIRE( app.processActuator( 2.0F ) == 0 );
    CHECK( app.wooferCommand() == Approx( 2.0F ) );
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
