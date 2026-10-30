/** \file cameraSim_emgain_test.cpp
 * \brief Offline tests of the camera simulator's standard EM-gain-off request.
 * \ingroup cameraSim_emgain_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/outletAppTest.hpp"
#include "../cameraSim.hpp"

namespace libXWCTest
{
/** \addtogroup cameraSim_emgain_unit_test */
namespace cameraSimTest
{
/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Expose only the simulated gain and cooling state needed by the offline request test.
class GainFixture : public outletHarness::Controller<MagAOX::app::cameraSim>
{
  public:
    using cameraSim::m_emGain;
    using cameraSim::m_emGainRequiresTemp;
    using cameraSim::m_emGainSet;
    using cameraSim::m_indiP_emGain;
    using cameraSim::m_indiP_emGainOff;
    using cameraSim::m_tempControlOnTarget;
    using cameraSim::m_tempControlStatus;
};
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
} // namespace cameraSimTest
} // namespace libXWCTest
