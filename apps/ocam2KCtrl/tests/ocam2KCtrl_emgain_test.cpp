/** \file ocam2KCtrl_emgain_test.cpp
 * \brief Offline tests of the OCAM unity-gain request and serial failure reporting.
 * \ingroup ocam2KCtrl_emgain_unit_test
 */

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
#define OCAM2KCTRL_TEST_SUPPORT_ONLY
#include "ocam2KCtrl_test.cpp"
#undef OCAM2KCTRL_TEST_SUPPORT_ONLY
#include "../../../tests/outletAppTest.hpp"
/// \endcond

#include "../../../tests/testXWC.hpp"

namespace libXWCTest
{
/** \addtogroup ocam2KCtrl_emgain_unit_test */
namespace ocam2KCtrlTest
{

/// The off request sends unity gain while warm and protected, and propagates serial errors.
/** \ingroup ocam2KCtrl_emgain_unit_test */
TEST_CASE( "ocam2KCtrl turns EM gain off without temperature lock or protection reset", "[ocam2KCtrl]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    ocam2KCtrl::setEMGainOff(); ocam2KCtrl::setEMGain();
    dev::stdCamera<ocam2KCtrl>::newCallBack_emgain_off();
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<ocam2KCtrl_test> app;
    app.driver();
    setPoweredOn( app );
    app.state( stateCodes::OPERATING );
    app.m_emGainRequiresTemp  = true;
    app.m_tempControlOnTarget = false;
    app.m_tempControlStatus   = false;
    app.m_protectionReset     = false;
    app.m_emGainSet           = 100;
    app.m_maxEMGain           = 600;
    app.createStandardIndiNumber<float>( app.m_indiP_emGain, "emgain", 0, 600, 1, "%0.0f" );
    app.createStandardIndiRequestSw( app.m_indiP_emGainOff, "emgain_off" );
    pcf::IndiProperty request( pcf::IndiProperty::Switch, app.m_indiP_emGainOff.getDevice(), "emgain_off" );
    request.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );

    SECTION( "each request sends gain one and resets the switch" )
    {
        for( int repeat = 0; repeat < 2; ++repeat )
        {
            queueSerialResponse( "gain set\n" );
            REQUIRE( app.newCallBack_stdCamera( request ) == 0 );
            REQUIRE( app.m_emGainSet == 1 );
            REQUIRE( app.m_indiP_emGain["target"].get<float>() == 1 );
            REQUIRE( app.m_indiP_emGainOff["request"].getSwitchState() == pcf::IndiElement::Off );
            REQUIRE( app.m_indiP_emGainOff.getState() == INDI_IDLE );
            REQUIRE_FALSE( app.m_protectionReset );
        }
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "gain 1", "gain 1" } );
    }
    SECTION( "a failed serial command is reported by the request property" )
    {
        queueSerialResponse( "", -1 );
        REQUIRE( app.newCallBack_stdCamera( request ) == -1 );
        REQUIRE( g_edtStubState.serialCommands == std::vector<std::string>{ "gain 1" } );
        REQUIRE( app.m_emGainSet == 1 );
        REQUIRE( app.m_indiP_emGainOff["request"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( app.m_indiP_emGainOff.getState() == INDI_ALERT );
    }
}
} // namespace ocam2KCtrlTest
} // namespace libXWCTest
