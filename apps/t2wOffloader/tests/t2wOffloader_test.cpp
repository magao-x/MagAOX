/** \file t2wOffloader_test.cpp
 * \brief Catch2 tests for the t2wOffloader app.
 *
 * \ingroup t2wOffloader_files
 */

#include "../../../tests/testXWC.hpp"

#include "../t2wOffloader.hpp"

using namespace MagAOX::app;

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

} // namespace t2wOffloaderTest

} // namespace libXWCTest
