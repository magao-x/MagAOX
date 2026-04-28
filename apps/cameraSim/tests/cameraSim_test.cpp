/** \file cameraSim_test.cpp
 * \brief Catch2 tests for the cameraSim app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup cameraSim_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../cameraSim.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup cameraSim_unit_test cameraSim Unit Tests
 * \brief Unit tests for the cameraSim application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `cameraSim` unit tests.
/** \ingroup cameraSim_unit_test
 */
namespace cameraSimTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class cameraSim_test : public cameraSim
{
  public:
    cameraSim_test( const std::string device )
    {
        m_configName = device;
        m_hasFocus   = true;

        m_indiP_focus = pcf::IndiProperty( pcf::IndiProperty::Switch );
        m_indiP_focus.setDevice( m_configName );
        m_indiP_focus.setName( "focus" );
        m_indiP_focus.setState( INDI_IDLE );
        m_indiP_focus.add( pcf::IndiElement( "state", pcf::IndiElement::Off ) );

        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, reconfigure )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, temp_ccd )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, temp_controller )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, readout_speed )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, vshift_speed )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, emgain )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, exptime )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, fps )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, synchro )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, mode )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_crop_mode )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_region_x )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_region_y )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_region_w )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_region_h )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_region_bin_x )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_region_bin_y )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_region_check )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_set )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_set_full )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_set_full_bin )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_load_last )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_set_last )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, roi_set_default )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, shutter )
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_temp, goto_focus )
    }

    void configureFocusHelper( const std::string &device,
                               const std::string &property,
                               const std::string &element,
                               bool               onMeansInFocus )
    {
        m_focusStateHelperConfigured = true;
        m_focusStateSource           = device + "." + property;
        m_focusStateElement          = element;
        m_focusStateOnMeansInFocus   = onMeansInFocus;
        m_focusStateSourceIndex      = 0;
        m_focusMonitoredPropertyKeys = { m_focusStateSource };
        m_indiP_focusMonitoredProperties.resize( 1 );
        m_indiP_focusMonitoredProperties[0].setDevice( device );
        m_indiP_focusMonitoredProperties[0].setName( property );
    }

    int cacheFocusProperty( const pcf::IndiProperty &ipRecv )
    {
        return setCallBack_focusMonitored( ipRecv );
    }

    bool helperFocusState()
    {
        return checkFocusSwitchState();
    }

    pcf::IndiElement::SwitchStateType publishedFocusState()
    {
        return m_indiP_focus["state"].getSwitchState();
    }
};

class focusHelper_test : public MagAOXApp<>, public dev::stdCamera<focusHelper_test>
{
    friend class dev::stdCamera<focusHelper_test>;

  public:
    static constexpr bool c_stdCamera_hasFocus = true;

    focusHelper_test() : MagAOXApp<>( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
    {
        m_configName = "camtest";
        m_hasFocus   = true;

        m_indiP_focus = pcf::IndiProperty( pcf::IndiProperty::Switch );
        m_indiP_focus.setDevice( m_configName );
        m_indiP_focus.setName( "focus" );
        m_indiP_focus.setState( INDI_IDLE );
        m_indiP_focus.add( pcf::IndiElement( "state", pcf::IndiElement::Off ) );
    }

    void configureFocusHelper( const std::string &device,
                               const std::string &property,
                               const std::string &element,
                               bool               onMeansInFocus )
    {
        m_focusStateHelperConfigured = true;
        m_focusStateSource           = device + "." + property;
        m_focusStateElement          = element;
        m_focusStateOnMeansInFocus   = onMeansInFocus;
        m_focusStateSourceIndex      = 0;
        m_focusMonitoredPropertyKeys = { m_focusStateSource };
        m_indiP_focusMonitoredProperties.resize( 1 );
        m_indiP_focusMonitoredProperties[0].setDevice( device );
        m_indiP_focusMonitoredProperties[0].setName( property );
    }

    int cacheFocusProperty( const pcf::IndiProperty &ipRecv )
    {
        return setCallBack_focusMonitored( ipRecv );
    }

    bool checkFocus()
    {
        return checkFocusSwitchState();
    }

    int gotoFocus()
    {
        return 0;
    }

    int appStartup() override
    {
        return 0;
    }

    int appLogic() override
    {
        return 0;
    }

    int appShutdown() override
    {
        return 0;
    }

    pcf::IndiElement::SwitchStateType publishedFocusState()
    {
        return m_indiP_focus["state"].getSwitchState();
    }
};
/// \endcond

/// Verify the cameraSim stdCamera callback validators accept only the expected properties.
/**
 * \ingroup cameraSim_unit_test
 */
TEST_CASE( "cameraSim INDI callbacks validate device and property names", "[cameraSim]" )
{
    // clang-format off
    #ifdef CAMERASIM_TEST_DOXYGEN_REF
    cameraSim::newCallBack_stdCamera( pcf::IndiProperty() );
    #endif
    // clang-format on

    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, reconfigure );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, temp_ccd );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, temp_controller );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, readout_speed );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, vshift_speed );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, emgain );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, exptime );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, fps );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, synchro );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_crop_mode );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_region_x );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_region_y );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_region_w );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_region_h );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_region_bin_x );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_region_bin_y );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_region_check );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_set );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_set_full );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_set_full_bin );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_load_last );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_set_last );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, roi_set_default );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, shutter );
    XWCTEST_INDI_ARBNEW_CALLBACK( cameraSim, newCallBack_stdCamera, goto_focus );
}

/// Verify the stdCamera focus helper supports configurable polarity and tracks monitored property updates.
/**
 * \ingroup cameraSim_unit_test
 */
TEST_CASE( "cameraSim stdCamera focus helper tracks monitored switch properties", "[cameraSim]" )
{
    SECTION( "configured element On means out of focus" )
    {
        focusHelper_test app;
        app.configureFocusHelper( "sre", "caution", "focus-mismatch", false );

        pcf::IndiProperty focusProp( pcf::IndiProperty::Switch );
        focusProp.setDevice( "sre" );
        focusProp.setName( "caution" );
        focusProp.add( pcf::IndiElement( "focus-mismatch", pcf::IndiElement::On ) );

        REQUIRE( app.cacheFocusProperty( focusProp ) == 0 );
        REQUIRE( app.publishedFocusState() == pcf::IndiElement::Off );

        focusProp["focus-mismatch"].setSwitchState( pcf::IndiElement::Off );

        REQUIRE( app.cacheFocusProperty( focusProp ) == 0 );
        REQUIRE( app.publishedFocusState() == pcf::IndiElement::On );
    }

    SECTION( "configured element On means in focus" )
    {
        focusHelper_test app;
        app.configureFocusHelper( "sre", "caution", "focus-ok", true );

        pcf::IndiProperty focusProp( pcf::IndiProperty::Switch );
        focusProp.setDevice( "sre" );
        focusProp.setName( "caution" );
        focusProp.add( pcf::IndiElement( "focus-ok", pcf::IndiElement::On ) );

        REQUIRE( app.cacheFocusProperty( focusProp ) == 0 );
        REQUIRE( app.publishedFocusState() == pcf::IndiElement::On );

        focusProp["focus-ok"].setSwitchState( pcf::IndiElement::Off );

        REQUIRE( app.cacheFocusProperty( focusProp ) == 0 );
        REQUIRE( app.publishedFocusState() == pcf::IndiElement::Off );
    }
}

} // namespace cameraSimTest

} // namespace libXWCTest
