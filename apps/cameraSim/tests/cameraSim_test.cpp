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

    /// Configure the stdCamera focus-state helper for a monitored switch element.
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
    static constexpr bool c_stdCamera_hasFocus     = true;
    static constexpr bool c_stdCamera_tempControl  = false;
    static constexpr bool c_stdCamera_readoutSpeed = false;
    static constexpr bool c_stdCamera_vShiftSpeed  = false;
    static constexpr bool c_stdCamera_emGain       = false;
    static constexpr bool c_stdCamera_usesModes    = false;
    static constexpr bool c_stdCamera_usesROI      = false;

  protected:
    pcf::IndiProperty m_lastSentProperty; ///< Captures the last INDI command sent through the goto-focus helper.

    int m_sendNewPropertyResult{ 0 }; ///< Return value used by the test sendNewProperty override.

  public:
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

    ~focusHelper_test() noexcept override = default;

    void configureGotoFocusHelper( const std::vector<std::string> &properties,
                                   const std::string              &format,
                                   const std::string              &targetProperty )
    {
        m_focusGotoHelperConfigured = true;
        m_focusGotoSourceProperties = properties;
        m_focusGotoFormat           = format;
        m_focusGotoTargetProperty   = targetProperty;
        m_focusGotoSourceIndices.clear();
        m_focusMonitoredPropertyKeys.clear();
        m_indiP_focusMonitoredProperties.clear();

        REQUIRE( indi::parseIndiKey( m_focusGotoTargetDevice, m_focusGotoTargetName, m_focusGotoTargetProperty ) == 0 );

        for( size_t n = 0; n < properties.size(); ++n )
        {
            std::string devName;
            std::string propName;

            REQUIRE( indi::parseIndiKey( devName, propName, properties[n] ) == 0 );

            m_focusGotoSourceIndices.push_back( static_cast<int>( n ) );
            m_focusMonitoredPropertyKeys.push_back( properties[n] );
            m_indiP_focusMonitoredProperties.emplace_back();
            m_indiP_focusMonitoredProperties[n].setDevice( devName );
            m_indiP_focusMonitoredProperties[n].setName( propName );
        }
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

    /// Cache a monitored INDI property through the stdCamera helper callback.
    int cacheFocusProperty( const pcf::IndiProperty &ipRecv )
    {
        return setCallBack_focusMonitored( ipRecv );
    }

    /// Evaluate the cached focus-state helper result.
    bool checkFocus()
    {
        return checkFocusSwitchState();
    }

    /// Satisfy the stdCamera derived-class interface for unit testing.
    int gotoFocus()
    {
        return 0;
    }

    /// Capture the last helper-issued INDI command instead of sending it.
    int sendNewProperty( const pcf::IndiProperty &ipSend )
    {
        m_lastSentProperty = ipSend;
        return m_sendNewPropertyResult;
    }

    /// Set the return code that the sendNewProperty test hook should report.
    void setSendNewPropertyResult( int result )
    {
        m_sendNewPropertyResult = result;
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

    /// Retrieve the last INDI property captured from sendGotoFocusCommand.
    const pcf::IndiProperty &lastSentProperty() const
    {
        return m_lastSentProperty;
    }

    /// Retrieve the configured goto-focus format string after config parsing.
    const std::string &gotoFocusFormat() const
    {
        return m_focusGotoFormat;
    }

    /// Expose stdCamera configuration setup for the config-file unit test.
    int setupConfig( mx::app::appConfigurator &config )
    {
        return dev::stdCamera<focusHelper_test>::setupConfig( config );
    }

    /// Expose stdCamera configuration loading for the config-file unit test.
    int loadConfig( mx::app::appConfigurator &config )
    {
        return dev::stdCamera<focusHelper_test>::loadConfig( config );
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

/// Verify the stdCamera goto-focus helper formats and dispatches the expected preset command.
/**
 * \ingroup cameraSim_unit_test
 */
TEST_CASE( "cameraSim stdCamera goto-focus helper dispatches preset commands", "[cameraSim]" )
{
    focusHelper_test app;
    app.configureGotoFocusHelper(
        { "stagebs.presetName", "fwfpm.filterName", "stagescibs.presetName" }, "{}-{}-{}", "stagesci1.presetName" );

    pcf::IndiProperty prop1( pcf::IndiProperty::Switch );
    prop1.setDevice( "stagebs" );
    prop1.setName( "presetName" );
    prop1.add( pcf::IndiElement( "65-35", pcf::IndiElement::On ) );
    prop1.add( pcf::IndiElement( "ha-ir", pcf::IndiElement::Off ) );

    pcf::IndiProperty prop2( pcf::IndiProperty::Switch );
    prop2.setDevice( "fwfpm" );
    prop2.setName( "filterName" );
    prop2.add( pcf::IndiElement( "open", pcf::IndiElement::On ) );
    prop2.add( pcf::IndiElement( "lyotsm", pcf::IndiElement::Off ) );

    pcf::IndiProperty prop3( pcf::IndiProperty::Switch );
    prop3.setDevice( "stagescibs" );
    prop3.setName( "presetName" );
    prop3.add( pcf::IndiElement( "ri", pcf::IndiElement::On ) );
    prop3.add( pcf::IndiElement( "out", pcf::IndiElement::Off ) );

    REQUIRE( app.cacheFocusProperty( prop1 ) == 0 );
    REQUIRE( app.cacheFocusProperty( prop2 ) == 0 );
    REQUIRE( app.cacheFocusProperty( prop3 ) == 0 );

    SECTION( "successful dispatch sends the formatted preset selection" )
    {
        REQUIRE( app.sendGotoFocusCommand() == 0 );
        REQUIRE( app.lastSentProperty().getDevice() == "stagesci1" );
        REQUIRE( app.lastSentProperty().getName() == "presetName" );
        REQUIRE( app.lastSentProperty().find( "65-35-open-ri" ) );
        REQUIRE( app.lastSentProperty()["65-35-open-ri"].getSwitchState() == pcf::IndiElement::On );
    }

    SECTION( "dispatch failures are propagated to the caller" )
    {
        app.setSendNewPropertyResult( -1 );
        REQUIRE( app.sendGotoFocusCommand() == -1 );
    }
}

/// Verify the stdCamera goto-focus helper strips wrapping quotes from configured format strings.
/**
 * \ingroup cameraSim_unit_test
 */
TEST_CASE( "cameraSim stdCamera goto-focus helper strips quoted format strings", "[cameraSim]" )
{
    mx::app::writeConfigFile( "/tmp/cameraSim_focusHelper.conf",
                              { "focus.gotoFocus",
                                "focus.gotoFocus",
                                "focus.gotoFocus",
                                "focus.gotoFocus",
                                "focus.gotoFocus",
                                "focus.gotoFocus" },
                              { "numSwitches", "property1", "property2", "property3", "format", "targetProperty" },
                              { "3",
                                "stagebs.presetName",
                                "fwfpm.filterName",
                                "stagescibs.presetName",
                                "\"{}-{}-{}\"",
                                "stagesci1.presetName" } );

    mx::app::appConfigurator config;
    focusHelper_test         app;

    REQUIRE( app.setupConfig( config ) == 0 );
    config.readConfig( "/tmp/cameraSim_focusHelper.conf" );
    REQUIRE( app.loadConfig( config ) == 0 );
    REQUIRE( app.gotoFocusFormat() == "{}-{}-{}" );
    REQUIRE_FALSE( app.gotoFocusFormat().find( '\"' ) != std::string::npos );
}

} // namespace cameraSimTest

} // namespace libXWCTest
