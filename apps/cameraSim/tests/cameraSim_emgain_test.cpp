/** \file cameraSim_emgain_test.cpp
 * \brief Offline tests of the camera simulator's standard EM-gain-off request.
 * \ingroup cameraSim_emgain_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/outletAppTest.hpp"
#include <future>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
#define MagAOXApp outletTestApp
#define telemeter outletTestTelemeter
#include "../cameraSim.hpp"
#undef telemeter
#undef MagAOXApp
/// \endcond

namespace libXWCTest
{
/** \addtogroup cameraSim_emgain_unit_test */
namespace cameraSimTest
{
/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Expose only the simulated gain and cooling state needed by the offline request test.
class GainFixture : public outletHarness::Controller<MagAOX::app::cameraSim>
{
    /// Signals the harmless test thread used by the production framegrabber liveness check.
    std::promise<void> m_finish;

  public:
    /// Attach a waiting test thread without starting acquisition or telemetry.
    GainFixture();

    /// Release and join the waiting thread before destroying the application.
    ~GainFixture();

    using cameraSim::m_ccdTemp;
    using cameraSim::m_ccdTempSetpt;
    using cameraSim::m_ccdTempValid;
    using cameraSim::m_emGain;
    using cameraSim::m_emGainMaxTemp;
    using cameraSim::m_emGainRequiresTemp;
    using cameraSim::m_emGainSet;
    using cameraSim::m_emGainTemp;
    using cameraSim::m_emGainTempMode;
    using cameraSim::m_indiP_emGain;
    using cameraSim::m_indiP_emGainOff;
    using cameraSim::m_tempControlOnTarget;
    using cameraSim::m_tempControlStatus;
    using cameraSim::m_tempControlStatusSet;
    using cameraSim::powerOnDefaults;
    using cameraSim::setTempControl;
    using cameraSim::setTempSetPt;
};

GainFixture::GainFixture()
{
    outletHarness::g_faults = {};
    m_fgThread              = std::thread( [completion = m_finish.get_future()]() { completion.wait(); } );
}

GainFixture::~GainFixture()
{
    m_finish.set_value();
    m_fgThread.join();
}
/// \endcond

/// The standard request resets both simulated and requested gain even without a temperature lock.
/** \ingroup cameraSim_emgain_unit_test */
TEST_CASE( "cameraSim standard EM-gain-off request selects unity", "[cameraSim]" )
{
    // clang-format off
    #ifdef CAMERASIM_TEST_DOXYGEN_REF
    MagAOX::app::cameraSim::setEMGainOff(); MagAOX::app::cameraSim::setEMGain();
    MagAOX::app::dev::stdCamera<MagAOX::app::cameraSim>::newCallBack_emgain_off();
    #endif
    // clang-format on
    GainFixture app;
    app.driver();
    app.state( MagAOX::app::stateCodes::OPERATING );
    app.m_emGainRequiresTemp  = true;
    app.m_tempControlOnTarget = false;
    app.m_tempControlStatus   = false;
    app.createStandardIndiNumber<float>( app.m_indiP_emGain, "emgain", 0, 600, 1, "%0.0f" );
    app.createStandardIndiRequestSw( app.m_indiP_emGainOff, "emgain_off" );
    pcf::IndiProperty request( pcf::IndiProperty::Switch, app.m_indiP_emGainOff.getDevice(), "emgain_off" );
    request.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );
    for( float previous : { 100.0f, 1.0f } )
    {
        app.m_emGain    = previous;
        app.m_emGainSet = previous;
        REQUIRE( app.newCallBack_stdCamera( request ) == 0 );
        REQUIRE( app.m_emGain == 1 );
        REQUIRE( app.m_emGainSet == 1 );
        REQUIRE( app.m_indiP_emGain["target"].get<float>() == 1 );
        REQUIRE( app.m_indiP_emGainOff["request"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( app.m_indiP_emGainOff.getState() == INDI_IDLE );
    }
}
/// Maximum-temperature protection accepts fresh simulated measurements and invalidates them on state changes.
/** \ingroup cameraSim_emgain_unit_test */
TEST_CASE( "cameraSim maximum-temperature guard waits for its next measurement cycle", "[cameraSim]" )
{
    // clang-format off
    #ifdef CAMERASIM_TEST_DOXYGEN_REF
    MagAOX::app::cameraSim::appLogic(); MagAOX::app::cameraSim::configureAcquisition();
    MagAOX::app::cameraSim::powerOnDefaults(); MagAOX::app::cameraSim::setTempSetPt();
    MagAOX::app::cameraSim::setTempControl();
    MagAOX::app::dev::stdCamera<MagAOX::app::cameraSim>::newCallBack_emgain();
    #endif
    // clang-format on
    using namespace MagAOX::app;
    GainFixture app;
    app.driver();
    app.m_emGainRequiresTemp = true;
    app.m_emGainTempMode     = "maxTemp";
    app.m_emGainTemp         = -50;
    app.m_emGainMaxTemp      = -45;
    REQUIRE( static_cast<dev::stdCamera<cameraSim> &>( app ).appStartup() == 0 );
    app.state( stateCodes::OPERATING );
    app.m_ccdTempSetpt      = -50;
    app.m_tempControlStatus = true;
    auto requestGain        = [&]()
    {
        pcf::IndiProperty request( pcf::IndiProperty::Number, app.m_indiP_emGain.getDevice(), "emgain" );
        request.add( pcf::IndiElement( "target" ) );
        request["target"].set( 100 );
        return app.newCallBack_stdCamera( request );
    };
    REQUIRE( requestGain() == -1 );
    REQUIRE( app.appLogic() == 0 );
    REQUIRE_FALSE( app.m_ccdTempValid ); // The initial -999 sentinel is never a usable cold reading.
    REQUIRE( requestGain() == -1 );
    app.m_ccdTemp = -46;
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_ccdTempValid );
    REQUIRE_FALSE( app.m_tempControlOnTarget );
    REQUIRE( requestGain() == 0 );
    REQUIRE( app.m_emGain == 100 );

    SECTION( "the threshold is inclusive and invalid or warmer readings reject new gain" )
    {
        app.m_ccdTemp = -45;
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( requestGain() == 0 );
        app.m_ccdTemp = -44.9;
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( requestGain() == -1 );
        app.m_ccdTemp = std::numeric_limits<float>::quiet_NaN();
        REQUIRE( app.appLogic() == 0 );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
    }
    SECTION( "temperature commands and power defaults require another measurement cycle" )
    {
        REQUIRE( app.setTempSetPt() == 0 );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( requestGain() == 0 );
        app.m_tempControlStatusSet = true;
        REQUIRE( app.setTempControl() == 0 );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( requestGain() == 0 );
        REQUIRE( app.powerOnDefaults() == 0 );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
    }
    SECTION( "acquisition configuration invalidates the preceding measurement" )
    {
        REQUIRE( app.powerOnDefaults() == 0 );
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( app.m_ccdTempValid );
        REQUIRE( app.configureAcquisition() == 0 );
        REQUIRE_FALSE( app.m_ccdTempValid );
        REQUIRE( requestGain() == -1 );
        REQUIRE( app.appLogic() == 0 );
        REQUIRE( requestGain() == 0 );
    }
}
} // namespace cameraSimTest
} // namespace libXWCTest
