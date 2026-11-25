/** \file stdCamera_test.cpp
 * \brief Offline tests of shared camera temperature limits, startup targets, and request rollback.
 * \ingroup stdCamera_unit_test
 */

#include "../../../../tests/testXWC.hpp"
#include "../../../../tests/outletAppTest.hpp"

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace temperatureHarness
{
/// Threadless app exercising the real shared camera interface.
template <bool tempLimits>
class Camera : public MagAOX::app::outletTestApp<>, public MagAOX::app::dev::stdCamera<Camera<tempLimits>>
{
    friend class MagAOX::app::dev::stdCamera<Camera>;

    /// Shared camera interface for this static test configuration.
    typedef MagAOX::app::dev::stdCamera<Camera> stdCameraT;

  public:
    static constexpr bool c_stdCamera_tempControl     = true;       ///< Expose temperature controls.
    static constexpr bool c_stdCamera_tempLimits      = tempLimits; ///< Enable configurable temperature limits.
    static constexpr bool c_stdCamera_temp            = true;       ///< Expose temperature reporting.
    static constexpr bool c_stdCamera_readoutSpeed    = false;      ///< Disable readout-speed control.
    static constexpr bool c_stdCamera_vShiftSpeed     = false;      ///< Disable vertical-shift control.
    static constexpr bool c_stdCamera_fanSpeed        = false;      ///< Disable fan control.
    static constexpr bool c_stdCamera_emGain          = false;      ///< Disable EM-gain control.
    static constexpr bool c_stdCamera_exptimeCtrl     = false;      ///< Disable exposure-time control.
    static constexpr bool c_stdCamera_fpsCtrl         = false;      ///< Disable frame-rate control.
    static constexpr bool c_stdCamera_fps             = false;      ///< Disable frame-rate reporting.
    static constexpr bool c_stdCamera_synchro         = false;      ///< Disable synchronization control.
    static constexpr bool c_stdCamera_usesModes       = false;      ///< Disable camera modes.
    static constexpr bool c_stdCamera_usesROI         = false;      ///< Disable ROI control.
    static constexpr bool c_stdCamera_cropMode        = false;      ///< Disable crop mode.
    static constexpr bool c_stdCamera_hasShutter      = false;      ///< Disable shutter control.
    static constexpr bool c_stdCamera_usesStateString = false;      ///< Disable persistent state descriptions.

    using stdCameraT::m_ccdTempSetpt;
    using stdCameraT::m_configMaxTemp;
    using stdCameraT::m_configMinTemp;
    using stdCameraT::m_indiP_temp;
    using stdCameraT::m_maxTemp;
    using stdCameraT::m_minTemp;
    using stdCameraT::m_startupTemp;

    /// Number of derived setter calls.
    unsigned m_setCalls{ 0 };

    /// Injected failure of the derived setter.
    int m_setResult{ 0 };

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

    /// Discard camera telemetry without starting a logger.
    template <class telemT>
    int telem( const typename telemT::messageT &message /**< [in] Serialized telemetry. */ );
};

template <bool tempLimits>
Camera<tempLimits>::Camera() : outletTestApp<>( "test", false )
{
    m_startupTemp = -55;
}

template <bool tempLimits>
void Camera<tempLimits>::setupConfig()
{
    REQUIRE( stdCameraT::setupConfig( config ) == 0 );
}

template <bool tempLimits>
int Camera<tempLimits>::appStartup()
{
    return stdCameraT::appStartup();
}

template <bool tempLimits>
int Camera<tempLimits>::appLogic()
{
    return stdCameraT::appLogic();
}

template <bool tempLimits>
int Camera<tempLimits>::appShutdown()
{
    return stdCameraT::appShutdown();
}

template <bool tempLimits>
int Camera<tempLimits>::powerOnDefaults()
{
    return 0;
}

template <bool tempLimits>
int Camera<tempLimits>::setTempSetPt()
{
    ++m_setCalls;
    return m_setResult;
}

template <bool tempLimits>
int Camera<tempLimits>::setTempControl()
{
    return 0;
}

template <bool tempLimits>
template <class telemT>
int Camera<tempLimits>::telem( const typename telemT::messageT & )
{
    return 0;
}

/// Shared camera with a real private INDI transport.
template <bool tempLimits = true>
class Fixture : public outletHarness::Controller<Camera<tempLimits>>
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

    /// Execute a power cycle through the shared state machine.
    int powerOn();
};

template <bool tempLimits>
Fixture<tempLimits>::Fixture()
{
    outletHarness::g_faults = {};
    this->driver();
    REQUIRE( this->appStartup() == 0 );
    this->configText( "" );
    REQUIRE( load() == 0 );
    REQUIRE( this->powerOnTemp() == 0 );
}

template <bool tempLimits>
int Fixture<tempLimits>::load()
{
    return MagAOX::app::dev::stdCamera<Camera<tempLimits>>::loadConfig( this->config );
}

template <bool tempLimits>
int Fixture<tempLimits>::request( float target )
{
    return requestValue( std::to_string( target ) );
}

template <bool tempLimits>
int Fixture<tempLimits>::requestValue( const std::string &target )
{
    pcf::IndiProperty received(
        pcf::IndiProperty::Number, this->m_indiP_temp.getDevice(), this->m_indiP_temp.getName() );
    received.add( pcf::IndiElement( "target" ) );
    received["target"].set( target );
    return this->newCallBack_temp( received );
}

template <bool tempLimits>
int Fixture<tempLimits>::powerOn()
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
} // namespace stdCameraTest
} // namespace libXWCTest
