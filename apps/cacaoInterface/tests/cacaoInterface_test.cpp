/** \file cacaoInterface_test.cpp
 * \brief Catch2 tests for the cacaoInterface app.
 *
 * \ingroup cacaoInterface_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

// Exercise the callback bodies as well as their property validators.
#undef XWCTEST_INDI_CALLBACK_VALIDATION
#include "../cacaoInterface.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup cacaoInterface_unit_test cacaoInterface Unit Tests
 * \brief Unit tests for the cacaoInterface application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `cacaoInterface` unit tests.
/** \ingroup cacaoInterface_unit_test
 */
namespace cacaoInterfaceTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class cacaoInterface_test : public cacaoInterface
{
  public:
    /// Configure INDI properties for direct callback tests.
    cacaoInterface_test( const std::string &device /**< [in] Device name. */ );

    /// Return whether offloading is linked to the HO loop.
    bool offloadLinked() const;
};

cacaoInterface_test::cacaoInterface_test( const std::string &device )
{
    m_configName = device;

    XWCTEST_SETUP_INDI_NEW_PROP( loopState );
    XWCTEST_SETUP_INDI_NEW_PROP( loopGain );
    XWCTEST_SETUP_INDI_NEW_PROP( loopZero );
    XWCTEST_SETUP_INDI_NEW_PROP( multCoeff );
    XWCTEST_SETUP_INDI_NEW_PROP( maxLim );

    createStandardIndiToggleSw( m_indiP_offloadLink, "offload_link" );
    createStandardIndiRequestSw( m_indiP_offloadOverride, "offload_override" );
}

bool cacaoInterface_test::offloadLinked() const
{
    return m_offloadLinked;
}
/// \endcond

/// Verify the cacaoInterface INDI callback validators accept only the expected properties.
/**
 * \ingroup cacaoInterface_unit_test
 */
TEST_CASE( "cacaoInterface INDI callbacks validate device and property names", "[cacaoInterface]" )
{
    // clang-format off
    #ifdef CACAOINTERFACE_TEST_DOXYGEN_REF
    cacaoInterface::newCallBack_m_indiP_loopState( pcf::IndiProperty() );
    cacaoInterface::newCallBack_m_indiP_loopGain( pcf::IndiProperty() );
    cacaoInterface::newCallBack_m_indiP_loopZero( pcf::IndiProperty() );
    cacaoInterface::newCallBack_m_indiP_multCoeff( pcf::IndiProperty() );
    cacaoInterface::newCallBack_m_indiP_maxLim( pcf::IndiProperty() );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, loopState );
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, loopGain );
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, loopZero );
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, multCoeff );
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, maxLim );
}

/// Verify a separate offload change releases the link in either HO loop state.
/** \ingroup cacaoInterface_unit_test
 */
TEST_CASE( "cacaoInterface releases the offload link on a manual override", "[cacaoInterface]" )
{
    // clang-format off
    #ifdef CACAOINTERFACE_TEST_DOXYGEN_REF
    cacaoInterface::newCallBack_m_indiP_offloadLink( pcf::IndiProperty() );
    cacaoInterface::newCallBack_m_indiP_offloadOverride( pcf::IndiProperty() );
    #endif
    // clang-format on

    cacaoInterface_test app( "holoop" );
    app.observeLoopState( GENERATE( false, true ) );

    pcf::IndiProperty link( pcf::IndiProperty::Switch );
    link.setDevice( "holoop" );
    link.setName( "offload_link" );
    link.add( pcf::IndiElement( "toggle", pcf::IndiElement::On ) );

    REQUIRE( app.newCallBack_m_indiP_offloadLink( link ) == 0 );
    REQUIRE( app.offloadLinked() );

    pcf::IndiProperty overrideLink( pcf::IndiProperty::Switch );
    overrideLink.setDevice( "holoop" );
    overrideLink.setName( "offload_override" );
    overrideLink.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );

    REQUIRE( app.newCallBack_m_indiP_offloadOverride( overrideLink ) == 0 );
    CHECK_FALSE( app.offloadLinked() );

    REQUIRE( app.newCallBack_m_indiP_offloadLink( link ) == 0 );
    CHECK( app.offloadLinked() );
}

} // namespace cacaoInterfaceTest

} // namespace libXWCTest
