/** \file stdCamera_test.cpp
 * \brief Offline tests of shared camera temperature limits, startup targets, and EM-gain protection and disable
 * requests.
 * \ingroup stdCamera_unit_test
 */

#include "../../../../tests/testXWC.hpp"
#include "../../../../tests/outletAppTest.hpp"

#include <chrono>
#include <future>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace temperatureHarness
{
/// Threadless app exercising the real shared camera interface.
template <bool tempLimits, bool emGain, bool tempControl>
class Camera : public MagAOX::app::outletTestApp<>,
               public MagAOX::app::dev::stdCamera<Camera<tempLimits, emGain, tempControl>>
{
    friend class MagAOX::app::dev::stdCamera<Camera>;

    /// Shared camera interface for this static test configuration.
    typedef MagAOX::app::dev::stdCamera<Camera> stdCameraT;

  public:
    static constexpr bool c_stdCamera_tempControl     = tempControl; ///< Expose temperature controls.
    static constexpr bool c_stdCamera_tempLimits      = tempLimits;  ///< Enable configurable temperature limits.
    static constexpr bool c_stdCamera_temp            = true;        ///< Expose temperature reporting.
    static constexpr bool c_stdCamera_readoutSpeed    = false;       ///< Disable readout-speed control.
    static constexpr bool c_stdCamera_vShiftSpeed     = false;       ///< Disable vertical-shift control.
    static constexpr bool c_stdCamera_fanSpeed        = false;       ///< Disable fan control.
    static constexpr bool c_stdCamera_emGain          = emGain;      ///< Expose EM-gain control.
    static constexpr bool c_stdCamera_exptimeCtrl     = false;       ///< Disable exposure-time control.
    static constexpr bool c_stdCamera_fpsCtrl         = false;       ///< Disable frame-rate control.
    static constexpr bool c_stdCamera_fps             = false;       ///< Disable frame-rate reporting.
    static constexpr bool c_stdCamera_synchro         = false;       ///< Disable synchronization control.
    static constexpr bool c_stdCamera_usesModes       = false;       ///< Disable camera modes.
    static constexpr bool c_stdCamera_usesROI         = false;       ///< Disable ROI control.
    static constexpr bool c_stdCamera_cropMode        = false;       ///< Disable crop mode.
    static constexpr bool c_stdCamera_hasShutter      = false;       ///< Disable shutter control.
    static constexpr bool c_stdCamera_usesStateString = false;       ///< Disable persistent state descriptions.

    using stdCameraT::emGainTempReady;
    using stdCameraT::invalidateTempReading;
    using stdCameraT::m_ccdTemp;
    using stdCameraT::m_ccdTempSetpt;
    using stdCameraT::m_ccdTempValid;
    using stdCameraT::m_configMaxTemp;
    using stdCameraT::m_configMinTemp;
    using stdCameraT::m_emGain;
    using stdCameraT::m_emGainMaxTemp;
    using stdCameraT::m_emGainRequiresTemp;
    using stdCameraT::m_emGainSet;
    using stdCameraT::m_emGainTemp;
    using stdCameraT::m_emGainTempMode;
    using stdCameraT::m_indiP_emGain;
    using stdCameraT::m_indiP_emGainOff;
    using stdCameraT::m_indiP_emGainTempReady;
    using stdCameraT::m_indiP_temp;
    using stdCameraT::m_indiP_tempcont;
    using stdCameraT::m_maxTemp;
    using stdCameraT::m_minTemp;
    using stdCameraT::m_startupTemp;
    using stdCameraT::m_tempControlOnTarget;
    using stdCameraT::m_tempControlStatus;
    using stdCameraT::onPowerOff;
    using stdCameraT::updateEMGainTempReady;
    using stdCameraT::updateTempReading;

    /// Number of derived setter calls.
    unsigned m_setCalls{ 0 };

    /// Injected failure of the derived setter.
    int m_setResult{ 0 };

    /// Number of EM-gain requests dispatched to the derived setter.
    unsigned m_gainCalls{ 0 };

    /// Number of gain-off requests dispatched to the camera-specific interface.
    unsigned m_gainOffCalls{ 0 };

    /// Camera-specific gain value selected by the gain-off operation.
    float m_gainOffValue{ 1 };

    /// Injected result of the camera-specific gain-off operation.
    int m_gainOffResult{ 0 };

    /// Reconfiguration flag required by the shared callback dispatcher.
    bool m_reconfig{ false };

    /// Construct with the default power-on temperature.
    Camera();

    /// Register the shared configuration keys.
    void setupConfig() override;

    /// Register shared INDI properties without a worker thread.
    int appStartup() override;

    /// Run the actual shared power-on logic.
    int appLogic() override;

    /// Shut down the threadless shared interface.
    int appShutdown() override;

    /// Leave power-on target initialization to stdCamera.
    int powerOnDefaults();

    /// Capture a temperature request without contacting a camera.
    int setTempSetPt();

    /// Provide the unused temperature-controller interface.
    int setTempControl();

    /// Capture an EM-gain request without contacting a camera.
    int setEMGain();

    /// Select the fake camera's gain-off value without contacting hardware.
    int setEMGainOff();

    /// Discard camera telemetry without starting a logger.
    template <class telemT>
    int telem( const typename telemT::messageT &message /**< [in] Serialized telemetry. */ );
};

template <bool tempLimits, bool emGain, bool tempControl>
Camera<tempLimits, emGain, tempControl>::Camera() : outletTestApp<>( "test", false )
{
    m_startupTemp = -55;
}

template <bool tempLimits, bool emGain, bool tempControl>
void Camera<tempLimits, emGain, tempControl>::setupConfig()
{
    REQUIRE( stdCameraT::setupConfig( config ) == 0 );
}

template <bool tempLimits, bool emGain, bool tempControl>
int Camera<tempLimits, emGain, tempControl>::appStartup()
{
    return stdCameraT::appStartup();
}

template <bool tempLimits, bool emGain, bool tempControl>
int Camera<tempLimits, emGain, tempControl>::appLogic()
{
    return stdCameraT::appLogic();
}

template <bool tempLimits, bool emGain, bool tempControl>
int Camera<tempLimits, emGain, tempControl>::appShutdown()
{
    return stdCameraT::appShutdown();
}

template <bool tempLimits, bool emGain, bool tempControl>
int Camera<tempLimits, emGain, tempControl>::powerOnDefaults()
{
    return 0;
}

template <bool tempLimits, bool emGain, bool tempControl>
int Camera<tempLimits, emGain, tempControl>::setTempSetPt()
{
    ++m_setCalls;
    return m_setResult;
}

template <bool tempLimits, bool emGain, bool tempControl>
int Camera<tempLimits, emGain, tempControl>::setTempControl()
{
    return 0;
}

template <bool tempLimits, bool emGain, bool tempControl>
int Camera<tempLimits, emGain, tempControl>::setEMGain()
{
    ++m_gainCalls;
    return 0;
}

template <bool tempLimits, bool emGain, bool tempControl>
int Camera<tempLimits, emGain, tempControl>::setEMGainOff()
{
    ++m_gainOffCalls;
    if( m_gainOffResult >= 0 )
        m_emGainSet = m_gainOffValue;
    return m_gainOffResult;
}

template <bool tempLimits, bool emGain, bool tempControl>
template <class telemT>
int Camera<tempLimits, emGain, tempControl>::telem( const typename telemT::messageT & )
{
    return 0;
}

/// Shared camera with a real private INDI transport.
template <bool tempLimits = true, bool emGain = false, bool tempControl = true>
class Fixture : public outletHarness::Controller<Camera<tempLimits, emGain, tempControl>>
{
  public:
    /// Initialize the shared properties and power state.
    Fixture();

    /// Load actual configuration through stdCamera.
    int load();

    /// Request a temperature through the real shared callback.
    int request( float target /**< [in] Requested temperature, in C. */ );

    /// Request raw INDI number text without silently converting malformed input.
    int requestValue( const std::string &target /**< [in] Number text received from INDI. */ );

    /// Request EM gain through the real shared callback.
    int requestGain( float target /**< [in] Requested EM gain. */ );

    /// Request the camera-specific gain-off operation through the shared property dispatcher.
    int requestGainOff();

    /// Request raw EM-gain text through the selected INDI element.
    int requestGainValue( const std::string &target /**< [in] Number text received from INDI. */,
                          const std::string &element = "target" /**< [in] Target or current request element. */ );

    /// Request a temperature-controller state through the shared callback.
    int requestController( bool enabled /**< [in] Whether cooling should be enabled. */ );

    /// Execute a power cycle through the shared state machine.
    int powerOn();
};

template <bool tempLimits, bool emGain, bool tempControl>
Fixture<tempLimits, emGain, tempControl>::Fixture()
{
    outletHarness::g_faults = {};
    this->driver();
    REQUIRE( this->appStartup() == 0 );
    this->configText( "" );
    REQUIRE( load() == 0 );
    REQUIRE( this->powerOnTemp() == 0 );
}

template <bool tempLimits, bool emGain, bool tempControl>
int Fixture<tempLimits, emGain, tempControl>::load()
{
    return MagAOX::app::dev::stdCamera<Camera<tempLimits, emGain, tempControl>>::loadConfig( this->config );
}

template <bool tempLimits, bool emGain, bool tempControl>
int Fixture<tempLimits, emGain, tempControl>::request( float target )
{
    return requestValue( std::to_string( target ) );
}

template <bool tempLimits, bool emGain, bool tempControl>
int Fixture<tempLimits, emGain, tempControl>::requestValue( const std::string &target )
{
    pcf::IndiProperty received(
        pcf::IndiProperty::Number, this->m_indiP_temp.getDevice(), this->m_indiP_temp.getName() );
    received.add( pcf::IndiElement( "target" ) );
    received["target"].set( target );
    return this->newCallBack_temp( received );
}

template <bool tempLimits, bool emGain, bool tempControl>
int Fixture<tempLimits, emGain, tempControl>::requestGain( float target )
{
    return requestGainValue( std::to_string( target ) );
}

template <bool tempLimits, bool emGain, bool tempControl>
int Fixture<tempLimits, emGain, tempControl>::requestGainOff()
{
    pcf::IndiProperty received( pcf::IndiProperty::Switch, this->configName(), "emgain_off" );
    received.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );
    return this->newCallBack_stdCamera( received );
}

template <bool tempLimits, bool emGain, bool tempControl>
int Fixture<tempLimits, emGain, tempControl>::requestGainValue( const std::string &target, const std::string &element )
{
    pcf::IndiProperty received(
        pcf::IndiProperty::Number, this->m_indiP_emGain.getDevice(), this->m_indiP_emGain.getName() );
    received.add( pcf::IndiElement( element ) );
    received[element].set( target );
    return this->newCallBack_emgain( received );
}

template <bool tempLimits, bool emGain, bool tempControl>
int Fixture<tempLimits, emGain, tempControl>::requestController( bool enabled )
{
    pcf::IndiProperty received(
        pcf::IndiProperty::Switch, this->m_indiP_tempcont.getDevice(), this->m_indiP_tempcont.getName() );
    received.add( pcf::IndiElement( "toggle" ) );
    received["toggle"].setSwitchState( enabled ? pcf::IndiElement::On : pcf::IndiElement::Off );
    return this->newCallBack_temp_controller( received );
}

template <bool tempLimits, bool emGain, bool tempControl>
int Fixture<tempLimits, emGain, tempControl>::powerOn()
{
    this->m_powerState       = 1;
    this->m_powerTargetState = 1;
    this->m_powerOnWait      = 0;
    this->m_powerOnCounter   = 0;
    this->state( MagAOX::app::stateCodes::POWERON );
    return this->appLogic();
}
} // namespace temperatureHarness
/// \endcond

namespace libXWCTest
{
/** \addtogroup stdCamera_unit_test */
namespace stdCameraTest
{
using MagAOX::app::stateCodes;
using temperatureHarness::Fixture;

/// Startup settings are reapplied on power cycles and must lie within the effective inclusive bounds.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera validates and restores its configured power-on target", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::setupConfig(); MagAOX::app::dev::stdCamera::loadConfig();
    MagAOX::app::dev::stdCamera::powerOnTemp(); MagAOX::app::dev::stdCamera::appLogic();
    #endif
    // clang-format on
    Fixture<> f;
    f.configText( "[camera]\nminTemp=-40\nmaxTemp=10\nstartupTemp=-30\n" );
    REQUIRE( f.load() == 0 );
    REQUIRE( f.m_minTemp == -40 );
    REQUIRE( f.m_maxTemp == 10 );
    for( int cycle = 0; cycle < 2; ++cycle )
    {
        f.m_ccdTempSetpt = 5;
        REQUIRE( f.powerOn() == 0 );
        REQUIRE( f.m_ccdTempSetpt == -30 );
        REQUIRE( f.m_indiP_temp["target"].get<float>() == -30 );
    }
    f.m_startupTemp = 11;
    REQUIRE( f.powerOnTemp() == -1 );
    REQUIRE( f.m_ccdTempSetpt == -30 );
}

/// Camera ranges can narrow configuration and refresh from the user settings without accumulating old limits.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera intersects camera and user temperature limits", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::setTempLimits(); MagAOX::app::dev::stdCamera::validateTempSetPt();
    #endif
    // clang-format on
    Fixture<> f;
    REQUIRE( f.setTempLimits( -80, 30 ) == 0 );
    REQUIRE( f.m_minTemp == -55 );
    REQUIRE( f.m_maxTemp == 20 );
    REQUIRE( f.setTempLimits( -40, 10 ) == 0 );
    REQUIRE( f.m_minTemp == -40 );
    REQUIRE( f.m_maxTemp == 10 );
    REQUIRE( f.setTempLimits( -80, 30 ) == 0 );
    REQUIRE( f.m_minTemp == -55 );
    REQUIRE( f.m_maxTemp == 20 );
    REQUIRE( f.setTempLimits( 21, 30 ) == -1 );
    REQUIRE( f.m_minTemp == -55 );
    REQUIRE( f.m_maxTemp == 20 );
    REQUIRE( f.validateTempSetPt( -55 ) == 0 );
    REQUIRE( f.validateTempSetPt( 20 ) == 0 );
    for( float target : { std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity() } )
        REQUIRE( f.validateTempSetPt( target ) == -1 );
}

/// Invalid numeric requests never reach the derived setter, and setter failures restore the prior target.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera rejects requests and restores failed targets", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::newCallBack_temp();
    #endif
    // clang-format on
    Fixture<> f;
    f.state( stateCodes::OPERATING );
    REQUIRE( f.request( 20 ) == 0 );
    REQUIRE( f.m_setCalls == 1 );
    REQUIRE( f.request( 21 ) == -1 );
    REQUIRE( f.m_setCalls == 1 );
    REQUIRE( f.m_ccdTempSetpt == 20 );
    REQUIRE( f.m_indiP_temp["target"].get<float>() == 20 );
    REQUIRE( f.state() == stateCodes::OPERATING );
    f.m_setResult = -1;
    REQUIRE( f.request( -30 ) == -1 );
    REQUIRE( f.m_setCalls == 2 );
    REQUIRE( f.m_ccdTempSetpt == 20 );
    REQUIRE( f.m_indiP_temp["target"].get<float>() == 20 );
    REQUIRE( f.m_indiP_temp.getState() == INDI_ALERT );
    for( const char *target : { "nan", "inf", "-inf", "1e100", "20junk", "" } )
    {
        REQUIRE( f.requestValue( target ) == -1 );
        REQUIRE( f.m_setCalls == 2 );
        REQUIRE( f.m_ccdTempSetpt == 20 );
    }
}
/// Cameras that have not enabled configurable limits retain their existing target and startup-override behavior.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera preserves temperature behavior for cameras without configurable limits", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::powerOnTemp(); MagAOX::app::dev::stdCamera::newCallBack_temp();
    #endif
    // clang-format on
    REQUIRE_FALSE( MagAOX::app::dev::stdCameraHasTempLimits<MagAOX::app::outletTestApp<>>::value );
    Fixture<false> f;
    f.m_startupTemp  = -999;
    f.m_ccdTempSetpt = 5;
    REQUIRE( f.powerOnTemp() == 0 );
    REQUIRE( f.m_ccdTempSetpt == 5 );
    REQUIRE( f.request( 25 ) == 0 );
    REQUIRE( f.m_ccdTempSetpt == 25 );
}
/// The opt-in guard is exposed only for cameras with both temperature control and EM gain.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera configures EM-gain temperature protection only for capable cameras", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::setupConfig(); MagAOX::app::dev::stdCamera::loadConfig();
    MagAOX::app::dev::stdCamera::newCallBack_emgain();
    #endif
    // clang-format on
    SECTION( "capable cameras preserve legacy behavior unless enabled" )
    {
        Fixture<true, true> f;
        REQUIRE( f.config.m_targets.count( "camera.emGainRequiresTemp" ) == 1 );
        REQUIRE( f.config.m_targets.count( "camera.emGainTemp" ) == 1 );
        REQUIRE( f.config.m_targets.count( "camera.emGainTempMode" ) == 1 );
        REQUIRE( f.config.m_targets.count( "camera.emGainMaxTemp" ) == 1 );
        REQUIRE_FALSE( f.m_emGainRequiresTemp );
        REQUIRE( f.requestGain( 100 ) == 0 );
        REQUIRE( f.m_gainCalls == 1 );
        f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\n" );
        REQUIRE( f.load() == 0 );
        REQUIRE( f.m_emGainRequiresTemp );
        REQUIRE( f.requestGain( 100 ) == -1 );
        REQUIRE( f.m_gainCalls == 1 );
        f.configText( "[camera]\nemGainRequiresTemp=false\n" );
        REQUIRE( f.load() == 0 );
        REQUIRE_FALSE( f.m_emGainRequiresTemp );
        REQUIRE( f.requestGain( 100 ) == 0 );
        REQUIRE( f.m_gainCalls == 2 );
    }
    SECTION( "cameras without EM gain do not expose the guard" )
    {
        Fixture<> f;
        REQUIRE( f.config.m_targets.count( "camera.emGainRequiresTemp" ) == 0 );
        REQUIRE( f.config.m_targets.count( "camera.emGainTemp" ) == 0 );
        REQUIRE( f.config.m_targets.count( "camera.emGainTempMode" ) == 0 );
        REQUIRE( f.config.m_targets.count( "camera.emGainMaxTemp" ) == 0 );
        REQUIRE_FALSE( f.m_indiP_emGainTempReady.find( "toggle" ) );
    }
    SECTION( "cameras without temperature control retain EM gain" )
    {
        Fixture<false, true, false> f;
        REQUIRE( f.config.m_targets.count( "camera.emGainRequiresTemp" ) == 0 );
        REQUIRE( f.config.m_targets.count( "camera.emGainTemp" ) == 0 );
        REQUIRE( f.config.m_targets.count( "camera.emGainTempMode" ) == 0 );
        REQUIRE( f.config.m_targets.count( "camera.emGainMaxTemp" ) == 0 );
        REQUIRE_FALSE( f.m_indiP_emGainTempReady.find( "toggle" ) );
        REQUIRE( f.requestGain( 100 ) == 0 );
        REQUIRE( f.m_gainCalls == 1 );
    }
}

/// Gain-off is a momentary standard request switch exposed only by EM-gain-capable cameras.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera exposes gain-off only for EM-gain-capable cameras", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::appStartup(); MagAOX::app::dev::stdCamera::newCallBack_stdCamera();
    #endif
    // clang-format on
    SECTION( "EM gain with temperature control" )
    {
        Fixture<true, true> f;
        REQUIRE( f.m_indiP_emGainOff.getType() == pcf::IndiProperty::Switch );
        REQUIRE( f.m_indiP_emGainOff.getName() == "emgain_off" );
        REQUIRE( f.m_indiP_emGainOff.getDevice() == f.configName() );
        REQUIRE( f.m_indiP_emGainOff["request"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( f.m_indiP_emGainOff.getState() == INDI_IDLE );
        REQUIRE( f.m_indiNewCallBacks.count( f.m_indiP_emGainOff.createUniqueKey() ) == 1 );
        REQUIRE( f.requestGainOff() == 0 );
        REQUIRE( f.m_gainOffCalls == 1 );
    }
    SECTION( "EM gain without temperature control" )
    {
        Fixture<false, true, false> f;
        REQUIRE( f.m_indiNewCallBacks.count( f.m_indiP_emGainOff.createUniqueKey() ) == 1 );
        REQUIRE( f.requestGainOff() == 0 );
        REQUIRE( f.m_gainOffCalls == 1 );
    }
    SECTION( "camera without EM gain" )
    {
        Fixture<>         f;
        pcf::IndiProperty received( pcf::IndiProperty::Switch, f.configName(), "emgain_off" );
        received.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );
        REQUIRE( f.m_indiNewCallBacks.count( received.createUniqueKey() ) == 0 );
        REQUIRE( f.newCallBack_stdCamera( received ) == -1 );
        REQUIRE( f.m_gainOffCalls == 0 );
    }
}

/// The dedicated safe-lowering action delegates the off value to the camera and can be repeated while warm or faulted.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera gain-off uses the camera-specific safe value despite temperature interlocks", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::newCallBack_stdCamera(); MagAOX::app::dev::stdCamera::newCallBack_emgain_off();
    MagAOX::app::dev::stdCamera::setEMGainOff();
    #endif
    // clang-format on
    Fixture<true, true> f;
    f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\n" );
    REQUIRE( f.load() == 0 );
    f.m_ccdTempSetpt        = -30;
    f.m_tempControlStatus   = false;
    f.m_tempControlOnTarget = false;
    f.m_emGain              = 100;
    f.m_emGainSet           = 100;
    f.m_indiP_emGain["current"].set( 100 );
    f.m_indiP_emGain["target"].set( 100 );
    f.state( stateCodes::ERROR );
    REQUIRE( f.requestGain( 200 ) == -1 );
    REQUIRE( f.m_gainCalls == 0 );
    f.messages();
    for( float offValue : { 0.0f, 1.0f } )
    {
        f.m_gainOffValue = offValue;
        for( int repeat = 0; repeat < 2; ++repeat )
        {
            REQUIRE( f.requestGainOff() == 0 );
            REQUIRE( f.m_emGainSet == offValue );
            REQUIRE( f.m_indiP_emGain["target"].get<float>() == offValue );
            REQUIRE( f.m_indiP_emGain["current"].get<float>() == 100 );
            REQUIRE( f.m_emGain == 100 );
            REQUIRE( f.m_indiP_emGainOff["request"].getSwitchState() == pcf::IndiElement::Off );
            REQUIRE( f.m_indiP_emGainOff.getState() == INDI_IDLE );
            REQUIRE( f.m_indiP_emGain.getState() == INDI_IDLE );
            REQUIRE( f.m_gainCalls == 0 );
            std::vector<pcf::IndiProperty> acknowledgments;
            for( auto &message : f.messages() )
            {
                if( message.getProperty().getName() == "emgain_off" )
                    acknowledgments.push_back( message.getProperty() );
            }
            REQUIRE( acknowledgments.size() == 2 );
            REQUIRE( acknowledgments[0]["request"].getSwitchState() == pcf::IndiElement::On );
            REQUIRE( acknowledgments[0].getState() == INDI_BUSY );
            REQUIRE( acknowledgments[1]["request"].getSwitchState() == pcf::IndiElement::Off );
            REQUIRE( acknowledgments[1].getState() == INDI_IDLE );
        }
    }
    REQUIRE( f.m_gainOffCalls == 4 );
}

/// Missing or Off requests do nothing, and mismatched properties cannot invoke the camera-specific operation.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera gain-off validates the request before dispatching", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::newCallBack_emgain_off(); MagAOX::app::dev::stdCamera::newCallBack_stdCamera();
    #endif
    // clang-format on
    Fixture<true, true> f;
    pcf::IndiProperty   received( pcf::IndiProperty::Switch, f.configName(), "emgain_off" );
    REQUIRE( f.newCallBack_emgain_off( received ) == 0 );
    received.add( pcf::IndiElement( "request", pcf::IndiElement::Off ) );
    REQUIRE( f.newCallBack_emgain_off( received ) == 0 );
    REQUIRE( f.newCallBack_stdCamera( received ) == 0 );
    REQUIRE( f.m_gainOffCalls == 0 );
    received["request"].setSwitchState( pcf::IndiElement::On );
    received.setName( "invalid_gain_off" );
    REQUIRE( f.newCallBack_emgain_off( received ) == -1 );
    REQUIRE( f.newCallBack_stdCamera( received ) == -1 );
    received.setName( "emgain_off" );
    received.setDevice( "another-camera" );
    REQUIRE( f.newCallBack_emgain_off( received ) == -1 );
    REQUIRE( f.newCallBack_stdCamera( received ) == -1 );
    REQUIRE( f.m_gainOffCalls == 0 );
    REQUIRE( f.m_indiP_emGainOff["request"].getSwitchState() == pcf::IndiElement::Off );
}

/// Device failures clear the momentary request, report Alert, and permit a later retry.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera gain-off reports device failures and recovers on retry", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::newCallBack_emgain_off(); MagAOX::app::dev::stdCamera::setEMGainOff();
    #endif
    // clang-format on
    Fixture<true, true> f;
    f.m_emGainSet     = 100;
    f.m_gainOffResult = -7;
    f.m_indiP_emGain["target"].set( 100 );
    REQUIRE( f.requestGainOff() == -7 );
    REQUIRE( f.m_gainOffCalls == 1 );
    REQUIRE( f.m_emGainSet == 100 );
    REQUIRE( f.m_indiP_emGain["target"].get<float>() == 100 );
    REQUIRE( f.m_indiP_emGain.getState() == INDI_ALERT );
    REQUIRE( f.m_indiP_emGainOff["request"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( f.m_indiP_emGainOff.getState() == INDI_ALERT );
    f.m_gainOffResult = 0;
    REQUIRE( f.requestGainOff() == 0 );
    REQUIRE( f.m_gainOffCalls == 2 );
    REQUIRE( f.m_indiP_emGain["target"].get<float>() == 1 );
    REQUIRE( f.m_indiP_emGain.getState() == INDI_IDLE );
    REQUIRE( f.m_indiP_emGainOff["request"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( f.m_indiP_emGainOff.getState() == INDI_IDLE );
}

/// Enabling the EM-gain guard requires an explicit, finite setpoint within the camera's configured limits.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera validates the required EM-gain temperature setpoint", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::setupConfig(); MagAOX::app::dev::stdCamera::loadConfig();
    #endif
    // clang-format on
    SECTION( "missing required setpoint is rejected" )
    {
        Fixture<true, true> f;
        f.configText( "[camera]\nemGainRequiresTemp=true\n" );
        REQUIRE( f.load() == -1 );
    }
    SECTION( "invalid required setpoints are rejected" )
    {
        for( const char *target : { "nan", "inf", "-inf", "1e100", "-55junk", "", "-56", "21" } )
        {
            CAPTURE( target );
            Fixture<true, true> f;
            f.configText( std::string( "[camera]\nemGainRequiresTemp=true\nemGainTemp=" ) + target + "\n" );
            REQUIRE( f.load() == -1 );
        }
    }
    SECTION( "required setpoint obeys a narrowed configured range" )
    {
        Fixture<true, true> f;
        f.configText( "[camera]\nminTemp=-40\nstartupTemp=-30\nemGainRequiresTemp=true\nemGainTemp=-55\n" );
        REQUIRE( f.load() == -1 );
    }
    SECTION( "inclusive setpoint limits are accepted" )
    {
        for( float target : { -55.0f, 20.0f } )
        {
            Fixture<true, true> f;
            f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=" + std::to_string( target ) + "\n" );
            REQUIRE( f.load() == 0 );
            REQUIRE( f.m_emGainTemp == target );
        }
    }
}

/// Locked behavior remains the default, while maximum-temperature mode requires an explicit compatible threshold.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera configures the EM-gain temperature enforcement mode", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::setupConfig(); MagAOX::app::dev::stdCamera::loadConfig();
    MagAOX::app::dev::stdCamera::newCallBack_emgain();
    #endif
    // clang-format on
    SECTION( "default locked mode retains existing behavior without a measured-temperature requirement" )
    {
        Fixture<true, true> f;
        f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\n" );
        REQUIRE( f.load() == 0 );
        REQUIRE( f.m_emGainTempMode == "locked" );
        REQUIRE_FALSE( f.m_ccdTempValid );
        f.state( stateCodes::OPERATING );
        f.m_tempControlStatus   = true;
        f.m_tempControlOnTarget = true;
        REQUIRE( f.requestGain( 100 ) == 0 );
    }
    SECTION( "unknown enforcement modes are rejected" )
    {
        for( const char *mode : { "invalid", "maxtemp", "lockedjunk", "" } )
        {
            CAPTURE( mode );
            Fixture<true, true> f;
            f.configText( std::string( "[camera]\nemGainTempMode=" ) + mode + "\n" );
            REQUIRE( f.load() == -1 );
        }
    }
    SECTION( "malformed or non-finite explicit maxima are rejected even with protection disabled" )
    {
        for( const char *maximum : { "nan", "inf", "-inf", "1e100", "-30junk", "" } )
        {
            CAPTURE( maximum );
            Fixture<true, true> f;
            f.configText( std::string( "[camera]\nemGainMaxTemp=" ) + maximum + "\n" );
            REQUIRE( f.load() == -1 );
        }
    }
    SECTION( "enabled maximum-temperature mode requires its maximum" )
    {
        Fixture<true, true> f;
        f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\nemGainTempMode=maxTemp\n" );
        REQUIRE( f.load() == -1 );
    }
    SECTION( "maximum cannot be colder than the required setpoint" )
    {
        Fixture<true, true> f;
        f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\nemGainTempMode=maxTemp\n"
                      "emGainMaxTemp=-56\n" );
        REQUIRE( f.load() == -1 );
    }
    SECTION( "maximum equal to the required setpoint is valid" )
    {
        Fixture<true, true> f;
        f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\nemGainTempMode=maxTemp\n"
                      "emGainMaxTemp=-55\n" );
        REQUIRE( f.load() == 0 );
        REQUIRE( f.m_emGainTempMode == "maxTemp" );
        REQUIRE( f.m_emGainMaxTemp == -55 );
    }
    SECTION( "the master flag still disables the temperature restriction" )
    {
        Fixture<true, true> f;
        f.configText( "[camera]\nemGainRequiresTemp=false\nemGainTempMode=maxTemp\n" );
        REQUIRE( f.load() == 0 );
        REQUIRE_FALSE( f.m_ccdTempValid );
        REQUIRE( f.requestGain( 100 ) == 0 );
    }
}

/// Maximum-temperature mode admits an unlocked cold camera and rejects a locked camera above the inclusive threshold.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera enforces an inclusive measured EM-gain temperature maximum", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::loadConfig(); MagAOX::app::dev::stdCamera::updateTempReading();
    MagAOX::app::dev::stdCamera::newCallBack_emgain();
    #endif
    // clang-format on
    Fixture<true, true> f;
    f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\nemGainTempMode=maxTemp\n"
                  "emGainMaxTemp=-30\n" );
    REQUIRE( f.load() == 0 );
    f.state( stateCodes::OPERATING );
    f.m_tempControlStatus   = true;
    f.m_tempControlOnTarget = false;
    for( float temperature : { -40.0f, std::nextafter( -30.0f, -std::numeric_limits<float>::infinity() ), -30.0f } )
    {
        CAPTURE( temperature );
        f.updateTempReading( temperature );
        REQUIRE( f.m_ccdTempValid );
        REQUIRE( f.requestGain( 100 ) == 0 );
    }
    REQUIRE( f.m_gainCalls == 3 );
    f.m_tempControlOnTarget = true;
    f.updateTempReading( std::nextafter( -30.0f, std::numeric_limits<float>::infinity() ) );
    REQUIRE( f.requestGain( 200 ) == -1 );
    REQUIRE( f.m_gainCalls == 3 );
    REQUIRE( f.m_emGainSet == 100 );
    REQUIRE( f.m_indiP_emGain["target"].get<float>() == 100 );
    REQUIRE( f.m_indiP_emGain.getState() == INDI_ALERT );
    for( float target : { 0.0f, 1.0f } )
        REQUIRE( f.requestGain( target ) == 0 );
    REQUIRE( f.requestGainOff() == 0 );
    REQUIRE( f.m_gainOffCalls == 1 );
}

/// A cold valid reading cannot override an incorrect setpoint, inactive cooling, or an unavailable camera.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera retains camera prerequisites in maximum-temperature mode", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::updateTempReading(); MagAOX::app::dev::stdCamera::newCallBack_emgain();
    MagAOX::app::dev::stdCamera::newCallBack_emgain_off();
    #endif
    // clang-format on
    Fixture<true, true> f;
    f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\nemGainTempMode=maxTemp\n"
                  "emGainMaxTemp=-30\n" );
    REQUIRE( f.load() == 0 );
    f.state( stateCodes::READY );
    f.m_tempControlStatus = true;
    f.updateTempReading( -40 );
    REQUIRE( f.requestGain( 100 ) == 0 );
    SECTION( "wrong requested setpoint" )
    {
        f.m_ccdTempSetpt = -40;
    }
    SECTION( "inactive cooling" )
    {
        f.m_tempControlStatus = false;
    }
    SECTION( "camera error state" )
    {
        f.state( stateCodes::ERROR );
    }
    SECTION( "non-finite required setpoint" )
    {
        f.m_emGainTemp = std::numeric_limits<float>::quiet_NaN();
    }
    SECTION( "non-finite requested setpoint" )
    {
        f.m_ccdTempSetpt = std::numeric_limits<float>::quiet_NaN();
    }
    SECTION( "non-finite runtime maximum" )
    {
        f.m_emGainMaxTemp = std::numeric_limits<float>::quiet_NaN();
    }
    SECTION( "runtime maximum colder than the required setpoint" )
    {
        f.m_emGainMaxTemp = -60;
        f.updateTempReading( -65 );
    }
    REQUIRE( f.requestGain( 200 ) == -1 );
    REQUIRE( f.m_gainCalls == 1 );
    REQUIRE( f.m_emGainSet == 100 );
    REQUIRE( f.m_indiP_emGain["target"].get<float>() == 100 );
    REQUIRE( f.requestGain( 0 ) == 0 );
    REQUIRE( f.requestGain( 1 ) == 0 );
    REQUIRE( f.requestGainOff() == 0 );
}

/// Invalid, stale, sentinel, and non-finite readings must never satisfy a maximum-temperature restriction.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera requires a valid measured temperature for maximum-temperature mode", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::updateTempReading(); MagAOX::app::dev::stdCamera::invalidateTempReading();
    MagAOX::app::dev::stdCamera::newCallBack_emgain();
    #endif
    // clang-format on
    Fixture<true, true> f;
    f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\nemGainTempMode=maxTemp\n"
                  "emGainMaxTemp=-30\n" );
    REQUIRE( f.load() == 0 );
    f.state( stateCodes::OPERATING );
    f.m_tempControlStatus   = true;
    f.m_tempControlOnTarget = true;
    REQUIRE_FALSE( f.m_ccdTempValid );
    REQUIRE( f.requestGain( 100 ) == -1 );
    REQUIRE( f.m_gainCalls == 0 );
    for( float temperature : { -999.0f,
                               std::nextafter( -273.15f, -std::numeric_limits<float>::infinity() ),
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity() } )
    {
        CAPTURE( temperature );
        f.updateTempReading( temperature );
        REQUIRE_FALSE( f.m_ccdTempValid );
        REQUIRE( f.requestGain( 100 ) == -1 );
        REQUIRE( f.m_gainCalls == 0 );
    }
    f.updateTempReading( -273.15f );
    REQUIRE( f.m_ccdTempValid );
    f.updateTempReading( -40 );
    REQUIRE( f.requestGain( 100 ) == 0 );
    f.invalidateTempReading();
    REQUIRE_FALSE( f.m_ccdTempValid );
    REQUIRE( f.requestGain( 200 ) == -1 );
    REQUIRE( f.m_gainCalls == 1 );
    for( float temperature : { -999.0f,
                               std::nextafter( -273.15f, -std::numeric_limits<float>::infinity() ),
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity() } )
    {
        f.m_ccdTemp      = temperature;
        f.m_ccdTempValid = true;
        REQUIRE( f.requestGain( 200 ) == -1 );
        REQUIRE( f.m_gainCalls == 1 );
    }
    REQUIRE( f.requestGain( 0 ) == 0 );
    REQUIRE( f.requestGain( 1 ) == 0 );
    REQUIRE( f.requestGainOff() == 0 );
}

/// Shared temperature and power transitions invalidate readings until a subsequent measurement is reported.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera invalidates measured temperature across control and power transitions", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::newCallBack_temp(); MagAOX::app::dev::stdCamera::newCallBack_temp_controller();
    MagAOX::app::dev::stdCamera::powerOnTemp(); MagAOX::app::dev::stdCamera::onPowerOff();
    MagAOX::app::dev::stdCamera::newCallBack_emgain(); MagAOX::app::dev::stdCamera::updateTempReading();
    #endif
    // clang-format on
    Fixture<true, true> f;
    f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\nemGainTempMode=maxTemp\n"
                  "emGainMaxTemp=-30\n" );
    REQUIRE( f.load() == 0 );
    f.state( stateCodes::OPERATING );
    f.m_tempControlStatus = true;
    f.updateTempReading( -40 );
    REQUIRE( f.requestGain( 100 ) == 0 );
    SECTION( "successful target change" )
    {
        REQUIRE( f.request( -50 ) == 0 );
        REQUIRE_FALSE( f.m_ccdTempValid );
        REQUIRE( f.request( -55 ) == 0 );
    }
    SECTION( "controller disable request" )
    {
        REQUIRE( f.requestController( false ) == 0 );
    }
    SECTION( "controller enable request" )
    {
        REQUIRE( f.requestController( true ) == 0 );
    }
    SECTION( "power-on defaults" )
    {
        REQUIRE( f.powerOnTemp() == 0 );
    }
    SECTION( "power-off" )
    {
        REQUIRE( f.onPowerOff() == 0 );
    }
    REQUIRE_FALSE( f.m_ccdTempValid );
    f.state( stateCodes::OPERATING );
    f.m_tempControlStatus = true;
    REQUIRE( f.requestGain( 200 ) == -1 );
    REQUIRE( f.m_gainCalls == 1 );
    f.updateTempReading( -30 );
    REQUIRE( f.requestGain( 200 ) == 0 );
    REQUIRE( f.m_gainCalls == 2 );
}

/// The read-only readiness switch uses the same predicate as amplified-gain requests and updates across invalidation.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera publishes the enforced EM-gain temperature readiness", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::appStartup(); MagAOX::app::dev::stdCamera::emGainTempReady();
    MagAOX::app::dev::stdCamera::updateEMGainTempReady(); MagAOX::app::dev::stdCamera::updateINDI();
    MagAOX::app::dev::stdCamera::newCallBack_emgain(); MagAOX::app::dev::stdCamera::newCallBack_temp();
    MagAOX::app::dev::stdCamera::onPowerOff();
    #endif
    // clang-format on
    SECTION( "readiness starts false when the property is first published" )
    {
        outletHarness::Controller<temperatureHarness::Camera<true, true, true>> f;
        f.driver();
        REQUIRE( f.appStartup() == 0 );
        REQUIRE( f.m_indiP_emGainTempReady.getName() == "emgain_temp_ready" );
        REQUIRE( f.m_indiP_emGainTempReady.getPerm() == pcf::IndiProperty::ReadOnly );
        REQUIRE( f.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::Off );
    }
    SECTION( "published maximum-temperature readiness agrees with gain enforcement" )
    {
        Fixture<true, true> f;
        f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\nemGainTempMode=maxTemp\n"
                      "emGainMaxTemp=-30\n" );
        REQUIRE( f.load() == 0 );
        f.state( stateCodes::OPERATING );
        f.m_tempControlStatus   = true;
        f.m_tempControlOnTarget = false;
        f.updateTempReading( -40 );
        REQUIRE( f.updateINDI() == 0 );
        REQUIRE( f.emGainTempReady() );
        REQUIRE( f.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::On );
        REQUIRE( f.requestGain( 100 ) == 0 );

        f.m_tempControlOnTarget = true;
        f.updateTempReading( -20 );
        REQUIRE( f.requestGain( 200 ) == -1 );
        REQUIRE_FALSE( f.emGainTempReady() );
        REQUIRE( f.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::Off );

        f.updateTempReading( -40 );
        f.invalidateTempReading();
        REQUIRE( f.updateINDI() == 0 );
        REQUIRE_FALSE( f.emGainTempReady() );
        REQUIRE( f.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( f.requestGain( 200 ) == -1 );

        f.updateTempReading( -40 );
        f.updateEMGainTempReady();
        REQUIRE( f.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::On );
        REQUIRE( f.request( -50 ) == 0 );
        REQUIRE_FALSE( f.emGainTempReady() );
        REQUIRE( f.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( f.request( -55 ) == 0 );
        f.updateTempReading( -40 );
        REQUIRE( f.updateINDI() == 0 );
        REQUIRE( f.emGainTempReady() );
        REQUIRE( f.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::On );
        REQUIRE( f.onPowerOff() == 0 );
        REQUIRE_FALSE( f.emGainTempReady() );
        REQUIRE( f.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::Off );

        f.m_emGainRequiresTemp = false;
        f.updateEMGainTempReady();
        REQUIRE( f.emGainTempReady() );
        REQUIRE( f.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::On );
    }
}

/// Readiness refresh must skip a busy INDI mutex and publish on the next uncontended application pass.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera readiness refresh does not wait for a busy INDI mutex", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::appLogic(); MagAOX::app::dev::stdCamera::updateEMGainTempReady();
    #endif
    // clang-format on
    Fixture<true, true> f;
    f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\nemGainTempMode=maxTemp\n"
                  "emGainMaxTemp=-30\n" );
    REQUIRE( f.load() == 0 );
    f.updateEMGainTempReady();
    f.state( stateCodes::READY );
    f.m_tempControlStatus = true;
    f.updateTempReading( -40 );
    REQUIRE( f.emGainTempReady() );
    REQUIRE( f.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::Off );

    std::promise<void> entered;
    auto               started = entered.get_future();
    std::future<int>   result;
    bool               enteredInTime       = false;
    bool               returnedWhileLocked = false;
    { //mutex scope
        std::unique_lock<std::mutex> lock( f.m_indiMutex );
        result        = std::async( std::launch::async,
                             [&]
                             {
                                 entered.set_value();
                                 return f.appLogic();
                             } );
        enteredInTime = started.wait_for( std::chrono::seconds( 2 ) ) == std::future_status::ready;
        if( enteredInTime )
            returnedWhileLocked = result.wait_for( std::chrono::milliseconds( 250 ) ) == std::future_status::ready;
    }
    // Always release the mutex before assertions or joining, including when a blocking implementation regresses.
    REQUIRE( result.wait_for( std::chrono::seconds( 2 ) ) == std::future_status::ready );
    REQUIRE( result.get() == 0 );
    REQUIRE( enteredInTime );
    REQUIRE( returnedWhileLocked );
    REQUIRE( f.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::Off );
    REQUIRE( f.appLogic() == 0 );
    REQUIRE( f.m_indiP_emGainTempReady["toggle"].getSwitchState() == pcf::IndiElement::On );
}

/// A locked camera must be configured to the dedicated EM-gain setpoint, independently of startup temperature.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera requires lock at the configured EM-gain temperature", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::loadConfig(); MagAOX::app::dev::stdCamera::powerOnTemp();
    MagAOX::app::dev::stdCamera::newCallBack_temp(); MagAOX::app::dev::stdCamera::newCallBack_emgain();
    #endif
    // clang-format on
    Fixture<true, true> f;
    f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-40\nstartupTemp=-30\n" );
    REQUIRE( f.load() == 0 );
    REQUIRE( f.powerOnTemp() == 0 );
    REQUIRE( f.m_ccdTempSetpt == -30 );
    REQUIRE( f.m_emGainTemp == -40 );
    f.state( stateCodes::OPERATING );
    f.m_tempControlStatus   = true;
    f.m_tempControlOnTarget = true;
    REQUIRE( f.requestGain( 1 ) == 0 );

    for( float setpoint : { -30.0f, -50.0f } )
    {
        f.m_ccdTempSetpt = setpoint;
        for( float measured : { setpoint, -40.0f } )
        {
            CAPTURE( setpoint, measured );
            f.m_ccdTemp = measured;
            REQUIRE( f.requestGain( 100 ) == -1 );
            REQUIRE( f.m_gainCalls == 1 );
            REQUIRE( f.m_emGainSet == 1 );
            REQUIRE( f.m_indiP_emGain["target"].get<float>() == 1 );
            REQUIRE( f.m_indiP_emGain.getState() == INDI_ALERT );
        }
    }
    for( float target : { 0.0f, 1.0f } )
        REQUIRE( f.requestGain( target ) == 0 );
    REQUIRE( f.m_gainCalls == 3 );
    REQUIRE( f.request( -40 ) == 0 );
    REQUIRE_FALSE( f.m_tempControlOnTarget );
    REQUIRE( f.requestGain( 100 ) == -1 );
    REQUIRE( f.m_gainCalls == 3 );
    f.m_ccdTemp             = -40;
    f.m_tempControlOnTarget = true;
    REQUIRE( f.requestGain( 100 ) == 0 );
    REQUIRE( f.m_gainCalls == 4 );
    REQUIRE( f.m_emGainSet == 100 );
}

/// Corrupted non-finite required or requested temperatures cannot satisfy the EM-gain guard.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera rejects non-finite EM-gain temperature state", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::newCallBack_emgain();
    #endif
    // clang-format on
    Fixture<true, true> f;
    f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\n" );
    REQUIRE( f.load() == 0 );
    f.state( stateCodes::OPERATING );
    f.m_tempControlStatus   = true;
    f.m_tempControlOnTarget = true;
    REQUIRE( f.requestGain( 1 ) == 0 );
    for( float value : { std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity() } )
    {
        f.m_emGainTemp   = value;
        f.m_ccdTempSetpt = -55;
        REQUIRE( f.requestGain( 100 ) == -1 );
        f.m_emGainTemp   = -55;
        f.m_ccdTempSetpt = value;
        REQUIRE( f.requestGain( 100 ) == -1 );
        f.m_emGainTemp = value;
        REQUIRE( f.requestGain( 100 ) == -1 );
        REQUIRE( f.m_gainCalls == 1 );
        REQUIRE( f.m_emGainSet == 1 );
        REQUIRE( f.m_indiP_emGain["target"].get<float>() == 1 );
    }
}

/// Off-target or disabled cooling rejects gain above unity without changing accepted requests or commanding hardware.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera refuses amplified EM gain until temperature is on target", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::newCallBack_emgain();
    #endif
    // clang-format on
    Fixture<true, true> f;
    f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\n" );
    REQUIRE( f.load() == 0 );
    f.state( stateCodes::OPERATING );
    f.m_emGain    = 100;
    f.m_emGainSet = 100;
    f.m_indiP_emGain["target"].set( 100 );

    SECTION( "cooling has not reached the target" )
    {
        f.m_tempControlStatus   = true;
        f.m_tempControlOnTarget = false;
    }
    SECTION( "cooling is disabled even if the previous status was on target" )
    {
        f.m_tempControlStatus   = false;
        f.m_tempControlOnTarget = true;
    }
    SECTION( "a disconnected or failed camera cannot reuse its previous temperature lock" )
    {
        f.m_tempControlStatus   = true;
        f.m_tempControlOnTarget = true;
        f.state( stateCodes::ERROR );
    }
    SECTION( "neither cooling nor an on-target status is available" )
    {
        f.m_tempControlStatus   = false;
        f.m_tempControlOnTarget = false;
    }
    for( float target : { 1.1f, 2.0f, 50.0f, 100.0f, 200.0f } )
    {
        CAPTURE( target );
        REQUIRE( f.requestGain( target ) == -1 );
        REQUIRE( f.m_gainCalls == 0 );
        REQUIRE( f.m_emGainSet == 100 );
        REQUIRE( f.m_indiP_emGain["target"].get<float>() == 100 );
        REQUIRE( f.m_indiP_emGain.getState() == INDI_ALERT );
    }
    for( float target : { 1.0f, 0.0f } )
    {
        REQUIRE( f.requestGain( target ) == 0 );
        REQUIRE( f.m_emGainSet == target );
        REQUIRE( f.m_indiP_emGain["target"].get<float>() == target );
    }
    REQUIRE( f.m_gainCalls == 2 );
    f.m_tempControlStatus   = true;
    f.m_tempControlOnTarget = true;
    f.state( stateCodes::READY );
    REQUIRE( f.requestGain( 200 ) == 0 );
    REQUIRE( f.m_gainCalls == 3 );
    REQUIRE( f.m_emGainSet == 200 );
}

/// Both accepted INDI element conventions use the same guard and malformed numbers cannot bypass it.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera validates EM-gain request elements before changing the target", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::newCallBack_emgain();
    #endif
    // clang-format on
    Fixture<true, true> f;
    f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\n" );
    REQUIRE( f.load() == 0 );
    f.state( stateCodes::OPERATING );
    REQUIRE( f.requestGain( 1 ) == 0 );
    for( const char *element : { "target", "current" } )
    {
        for( const char *target : { "nan", "inf", "-inf", "1e100", "20junk", "", "20" } )
        {
            CAPTURE( element, target );
            REQUIRE( f.requestGainValue( target, element ) == -1 );
            REQUIRE( f.m_gainCalls == 1 );
            REQUIRE( f.m_emGainSet == 1 );
            REQUIRE( f.m_indiP_emGain["target"].get<float>() == 1 );
            REQUIRE( f.m_indiP_emGain.getState() == INDI_ALERT );
        }
    }
    REQUIRE( f.requestGainValue( "0", "current" ) == 0 );
    REQUIRE( f.m_gainCalls == 2 );
    REQUIRE( f.m_emGainSet == 0 );

    pcf::IndiProperty received( pcf::IndiProperty::Number, f.m_indiP_emGain.getDevice(), f.m_indiP_emGain.getName() );
    received.add( pcf::IndiElement( "current" ) );
    received.add( pcf::IndiElement( "target" ) );
    received["current"].set( 100 );
    received["target"].set( 1 );
    REQUIRE( f.newCallBack_emgain( received ) == 0 );
    REQUIRE( f.m_gainCalls == 3 );
    REQUIRE( f.m_emGainSet == 1 );
    received["current"].set( 1 );
    received["target"].set( 100 );
    REQUIRE( f.newCallBack_emgain( received ) == -1 );
    REQUIRE( f.m_gainCalls == 3 );
    REQUIRE( f.m_emGainSet == 1 );
}

/// New temperature settings and power transitions invalidate prior temperature lock before another gain command.
/** \ingroup stdCamera_unit_test */
TEST_CASE( "stdCamera does not reuse stale temperature lock for EM gain", "[stdCamera]" )
{
    // clang-format off
    #ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera::newCallBack_temp(); MagAOX::app::dev::stdCamera::newCallBack_temp_controller();
    MagAOX::app::dev::stdCamera::powerOnTemp(); MagAOX::app::dev::stdCamera::onPowerOff();
    MagAOX::app::dev::stdCamera::newCallBack_emgain();
    #endif
    // clang-format on
    Fixture<true, true> f;
    f.configText( "[camera]\nemGainRequiresTemp=true\nemGainTemp=-55\n" );
    REQUIRE( f.load() == 0 );
    f.state( stateCodes::OPERATING );
    f.m_tempControlStatus   = true;
    f.m_tempControlOnTarget = true;
    REQUIRE( f.requestGain( 100 ) == 0 );

    SECTION( "a changed temperature target invalidates lock" )
    {
        REQUIRE( f.request( -30 ) == 0 );
    }
    SECTION( "disabling the controller invalidates lock" )
    {
        REQUIRE( f.requestController( false ) == 0 );
    }
    SECTION( "a power-on invalidates lock" )
    {
        REQUIRE( f.powerOn() == 0 );
    }
    SECTION( "a power-off invalidates lock" )
    {
        REQUIRE( f.onPowerOff() == 0 );
    }
    REQUIRE_FALSE( f.m_tempControlOnTarget );
    REQUIRE( f.requestGain( 200 ) == -1 );
    REQUIRE( f.m_gainCalls == 1 );
    REQUIRE( f.m_emGainSet == 100 );
}

} // namespace stdCameraTest
} // namespace libXWCTest
