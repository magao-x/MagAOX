/** \file stdCamera_test.hpp
 * \brief Test harness for the MagAOX::app::dev::stdCamera device mixin.
 *
 * One harness class template, stdCameraHarness<capsT>, drives the real stdCamera code
 * against a fake camera. capsT is a plain struct of the stdCamera capability flags and
 * decides which features the harness turns on. Two configurations are used:
 *
 * - stdCameraFullHarness turns every capability on and mixes in the telemeter.
 * - stdCameraReportOnlyHarness only reports temperature and FPS, with everything else off.
 *
 * The derived class interface a real camera app implements is stubbed. Each stub counts its
 * calls, copies the requested value into the current value where one exists, and returns a
 * result the test can set in advance. The common parts of every dev:: harness, such as the
 * FIFO-less indiDriver and the registration fault injection, come from appHarnessBase in
 * testHarnessCommon.hpp.
 *
 * \ingroup testing
 */

#include "../../../../tests/catch2/catch.hpp"

#include "../../MagAOXApp.hpp"
#include "../stdCamera.hpp"
#include "../telemeter.hpp"
#include "testHarnessCommon.hpp"

// LCOV_EXCL_START

namespace stdCamera_tests
{

/// Capability set with every stdCamera feature on and the telemeter mixed in.
struct stdCameraFullCaps
{
    static constexpr const char *configName = "stdcamtest";
    static constexpr bool        telemeter  = true;

    static constexpr bool tempControl     = true;
    static constexpr bool temp            = true;
    static constexpr bool readoutSpeed    = true;
    static constexpr bool vShiftSpeed     = true;
    static constexpr bool fanSpeed        = true;
    static constexpr bool led             = true;
    static constexpr bool analogGain      = true;
    static constexpr bool hasFocus        = true;
    static constexpr bool emGain          = true;
    static constexpr bool exptimeCtrl     = true;
    static constexpr bool fpsCtrl         = true;
    static constexpr bool fps             = true;
    static constexpr bool synchro         = true;
    static constexpr bool usesModes       = true;
    static constexpr bool usesROI         = true;
    static constexpr bool cropMode        = true;
    static constexpr bool hasShutter      = true;
    static constexpr bool usesStateString = true;
};

/// Capability set that only reports temperature and FPS. Everything else is off.
/** This reaches the `else if( derivedT::c_stdCamera_temp )` and `c_stdCamera_fps` branches
 * and the disabled skip branch of every other feature.
 */
struct stdCameraReportOnlyCaps
{
    static constexpr const char *configName = "stdcamreportonlytest";
    static constexpr bool        telemeter  = false;

    static constexpr bool tempControl     = false;
    static constexpr bool temp            = true;
    static constexpr bool readoutSpeed    = false;
    static constexpr bool vShiftSpeed     = false;
    static constexpr bool fanSpeed        = false;
    static constexpr bool led             = false;
    static constexpr bool analogGain      = false;
    static constexpr bool hasFocus        = false;
    static constexpr bool emGain          = false;
    static constexpr bool exptimeCtrl     = false;
    static constexpr bool fpsCtrl         = false;
    static constexpr bool fps             = true;
    static constexpr bool synchro         = false;
    static constexpr bool usesModes       = false;
    static constexpr bool usesROI         = false;
    static constexpr bool cropMode        = false;
    static constexpr bool hasShutter      = false;
    static constexpr bool usesStateString = false;
};

/// Stand-in for the telemeter mixin when a capability set does not use it.
/** It answers the calls the harness forwards with success, so the harness code does not
 * need a special case per configuration.
 */
struct noTelemeter
{
    int setupConfig( mx::app::appConfigurator & )
    {
        return 0;
    }
    int loadConfig( mx::app::appConfigurator & )
    {
        return 0;
    }
    int appStartup()
    {
        return 0;
    }
    int appShutdown()
    {
        return 0;
    }
    template <class telT>
    int checkRecordTimes( const telT & )
    {
        return 0;
    }
};

/// The stdCamera test harness. capsT selects the capability flags. See the file comment.
template <class capsT>
struct stdCameraHarness
    : public MagAOX::app::dev::testHarness::appHarnessBase,
      public MagAOX::app::dev::stdCamera<stdCameraHarness<capsT>>,
      public std::conditional_t<capsT::telemeter, MagAOX::app::dev::telemeter<stdCameraHarness<capsT>>, noTelemeter>
{
    typedef MagAOX::app::dev::testHarness::appHarnessBase                                              baseT;
    typedef MagAOX::app::dev::stdCamera<stdCameraHarness<capsT>>                                       stdCameraT;
    typedef std::conditional_t<capsT::telemeter, MagAOX::app::dev::telemeter<stdCameraHarness<capsT>>, noTelemeter> telemeterT;

    friend stdCameraT;
    friend telemeterT;

    // The stdCamera capability flags, taken from capsT. stdCamera reads these as derivedT::c_stdCamera_*.
    static constexpr bool c_stdCamera_tempControl     = capsT::tempControl;
    static constexpr bool c_stdCamera_temp            = capsT::temp;
    static constexpr bool c_stdCamera_readoutSpeed    = capsT::readoutSpeed;
    static constexpr bool c_stdCamera_vShiftSpeed     = capsT::vShiftSpeed;
    static constexpr bool c_stdCamera_fanSpeed        = capsT::fanSpeed;
    static constexpr bool c_stdCamera_led             = capsT::led;
    static constexpr bool c_stdCamera_analogGain      = capsT::analogGain;
    static constexpr bool c_stdCamera_hasFocus        = capsT::hasFocus;
    static constexpr bool c_stdCamera_emGain          = capsT::emGain;
    static constexpr bool c_stdCamera_exptimeCtrl     = capsT::exptimeCtrl;
    static constexpr bool c_stdCamera_fpsCtrl         = capsT::fpsCtrl;
    static constexpr bool c_stdCamera_fps             = capsT::fps;
    static constexpr bool c_stdCamera_synchro         = capsT::synchro;
    static constexpr bool c_stdCamera_usesModes       = capsT::usesModes;
    static constexpr bool c_stdCamera_usesROI         = capsT::usesROI;
    static constexpr bool c_stdCamera_cropMode        = capsT::cropMode;
    static constexpr bool c_stdCamera_hasShutter      = capsT::hasShutter;
    static constexpr bool c_stdCamera_usesStateString = capsT::usesStateString;

    // Make the protected state of stdCamera public so tests can read and set it.
    using stdCameraT::m_analogGainName;
    using stdCameraT::m_analogGainNameLabels;
    using stdCameraT::m_analogGainNameSet;
    using stdCameraT::m_analogGainNames;
    using stdCameraT::m_analogGainValid;
    using stdCameraT::m_cameraModes;
    using stdCameraT::m_ccdTempSetpt;
    using stdCameraT::m_cropMode;
    using stdCameraT::m_cropModeSet;
    using stdCameraT::m_currentROI;
    using stdCameraT::m_defaultFanSpeed;
    using stdCameraT::m_defaultLEDState;
    using stdCameraT::m_defaultReadoutSpeed;
    using stdCameraT::m_defaultVShiftSpeed;
    using stdCameraT::m_default_bin_x;
    using stdCameraT::m_default_bin_y;
    using stdCameraT::m_default_h;
    using stdCameraT::m_default_w;
    using stdCameraT::m_default_x;
    using stdCameraT::m_default_y;
    using stdCameraT::m_emGainSet;
    using stdCameraT::m_expTime;
    using stdCameraT::m_expTimeSet;
    using stdCameraT::m_fanSpeedControlEnabled;
    using stdCameraT::m_fanSpeedName;
    using stdCameraT::m_fanSpeedNameLabels;
    using stdCameraT::m_fanSpeedNameSet;
    using stdCameraT::m_fanSpeedNames;
    using stdCameraT::m_fanSpeedValid;
    using stdCameraT::m_focusGotoFormat;
    using stdCameraT::m_focusGotoHelperConfigured;
    using stdCameraT::m_focusGotoSourceIndices;
    using stdCameraT::m_focusGotoTargetDevice;
    using stdCameraT::m_focusGotoTargetName;
    using stdCameraT::m_focusMonitoredPropertyKeys;
    using stdCameraT::m_focusStateElement;
    using stdCameraT::m_focusStateHelperConfigured;
    using stdCameraT::m_focusStateOnMeansInFocus;
    using stdCameraT::m_focusStateSourceIndex;
    using stdCameraT::m_fpsSet;
    using stdCameraT::m_full_bin_x;
    using stdCameraT::m_full_currbin_h;
    using stdCameraT::m_full_currbin_w;
    using stdCameraT::m_full_currbin_x;
    using stdCameraT::m_full_currbin_y;
    using stdCameraT::m_full_w;
    using stdCameraT::m_full_x;
    using stdCameraT::m_hasFocus;
    using stdCameraT::m_indiP_analogGain;
    using stdCameraT::m_indiP_cropMode;
    using stdCameraT::m_indiP_emGain;
    using stdCameraT::m_indiP_exptime;
    using stdCameraT::m_indiP_fanSpeed;
    using stdCameraT::m_indiP_focus;
    using stdCameraT::m_indiP_focusMonitoredProperties;
    using stdCameraT::m_indiP_fps;
    using stdCameraT::m_indiP_gotoFocus;
    using stdCameraT::m_indiP_led;
    using stdCameraT::m_indiP_mode;
    using stdCameraT::m_indiP_readoutSpeed;
    using stdCameraT::m_indiP_reconfig;
    using stdCameraT::m_indiP_roi_x;
    using stdCameraT::m_indiP_shutter;
    using stdCameraT::m_indiP_shutterStatus;
    using stdCameraT::m_indiP_stateString;
    using stdCameraT::m_indiP_synchro;
    using stdCameraT::m_indiP_temp;
    using stdCameraT::m_indiP_tempcont;
    using stdCameraT::m_indiP_vShiftSpeed;
    using stdCameraT::m_lastROI;
    using stdCameraT::m_ledState;
    using stdCameraT::m_ledStateSet;
    using stdCameraT::m_ledStateValid;
    using stdCameraT::m_maxEMGain;
    using stdCameraT::m_modeName;
    using stdCameraT::m_nextMode;
    using stdCameraT::m_nextROI;
    using stdCameraT::m_readoutSpeedName;
    using stdCameraT::m_readoutSpeedNameSet;
    using stdCameraT::m_shutterState;
    using stdCameraT::m_shutterStatus;
    using stdCameraT::m_startupMode;
    using stdCameraT::m_startupTemp;
    using stdCameraT::m_synchro;
    using stdCameraT::m_synchroSet;
    using stdCameraT::m_tempControlOnTarget;
    using stdCameraT::m_tempControlStatus;
    using stdCameraT::m_tempControlStatusSet;
    using stdCameraT::m_vShiftSpeedName;
    using stdCameraT::m_vShiftSpeedNameSet;

    // Bring the trueFalseT tag-dispatch overloads of stdCamera back into scope. The no-argument
    // derived-class interface stubs below have the same names and would otherwise hide them.
    using stdCameraT::checkFocus;
    using stdCameraT::checkNextROI;
    using stdCameraT::gotoFocus;
    using stdCameraT::setAnalogGain;
    using stdCameraT::setCropMode;
    using stdCameraT::setEMGain;
    using stdCameraT::setExpTime;
    using stdCameraT::setFPS;
    using stdCameraT::setFanSpeed;
    using stdCameraT::setLED;
    using stdCameraT::setNextROI;
    using stdCameraT::setReadoutSpeed;
    using stdCameraT::setShutter;
    using stdCameraT::setSynchro;
    using stdCameraT::setTempControl;
    using stdCameraT::setTempSetPt;
    using stdCameraT::setVShiftSpeed;
    using stdCameraT::stateString;
    using stdCameraT::stateStringValid;

    // Call counters. Tests assert on these to see which stubs ran.
    int m_powerOnDefaultsCalls{ 0 };
    int m_setTempControlCalls{ 0 };
    int m_setTempSetPtCalls{ 0 };
    int m_setReadoutSpeedCalls{ 0 };
    int m_setVShiftSpeedCalls{ 0 };
    int m_setFanSpeedCalls{ 0 };
    int m_setAnalogGainCalls{ 0 };
    int m_setLEDCalls{ 0 };
    int m_setEMGainCalls{ 0 };
    int m_setExpTimeCalls{ 0 };
    int m_setFPSCalls{ 0 };
    int m_setSynchroCalls{ 0 };
    int m_setCropModeCalls{ 0 };
    int m_checkNextROICalls{ 0 };
    int m_setNextROICalls{ 0 };
    int m_setShutterCalls{ 0 };
    int m_lastShutterSS{ -100 };
    int m_gotoFocusCalls{ 0 };

    // Forced return values. A test sets one nonzero to exercise an error branch.
    int         m_setTempSetPtResult{ 0 };
    int         m_setTempControlResult{ 0 };
    int         m_setReadoutSpeedResult{ 0 };
    int         m_setVShiftSpeedResult{ 0 };
    int         m_setFanSpeedResult{ 0 };
    int         m_setAnalogGainResult{ 0 };
    int         m_setLEDResult{ 0 };
    int         m_setEMGainResult{ 0 };
    int         m_setExpTimeResult{ 0 };
    int         m_setFPSResult{ 0 };
    int         m_setSynchroResult{ 0 };
    int         m_setCropModeResult{ 0 };
    int         m_checkNextROIResult{ 0 };
    int         m_setNextROIResult{ 0 };
    int         m_setShutterResult{ 0 };
    bool        m_checkFocusResult{ false };
    int         m_gotoFocusResult{ 0 };
    std::string m_stateStringResult{ "STATESTRING" };
    bool        m_stateStringValidResult{ true };

    /// stdCamera sets this on the derived class when a reconfiguration is requested.
    bool m_reconfig{ false };

    // Captures for the goto-focus helper. The stubbed sendNewProperty() stores the outgoing
    // property here instead of sending it to a real INDI driver.
    pcf::IndiProperty m_lastSentProperty;
    int               m_sendNewPropertyResult{ 0 };

    /// Set up a fake camera. Defaults exist only for the capabilities that are on.
    stdCameraHarness() : baseT( capsT::configName )
    {
        if constexpr( capsT::readoutSpeed )
        {
            m_defaultReadoutSpeed    = "slow";
            this->m_readoutSpeedNames      = { "slow", "fast" };
            this->m_readoutSpeedNameLabels = { "Slow", "Fast" };
        }
        if constexpr( capsT::vShiftSpeed )
        {
            m_defaultVShiftSpeed    = "std";
            this->m_vShiftSpeedNames      = { "std", "fast" };
            this->m_vShiftSpeedNameLabels = { "Standard", "Fast" };
        }
        if constexpr( capsT::fanSpeed )
        {
            m_fanSpeedNames      = { "low", "high" };
            m_fanSpeedNameLabels = { "Low", "High" };
            m_defaultFanSpeed    = "low"; // valid, so tests unrelated to fan speed validation need not set it
        }
        if constexpr( capsT::analogGain )
        {
            m_analogGainNames      = { "low", "high" };
            m_analogGainNameLabels = { "Low", "High" };
        }
        if constexpr( capsT::emGain )
        {
            m_maxEMGain = 100;
        }
        if constexpr( capsT::usesROI )
        {
            this->m_minROIx = 0;
            this->m_maxROIx = 1023; // a 1024 by 1024 sensor
            this->m_minROIy = 0;
            this->m_maxROIy = 1023;

            m_default_x     = 511.5;
            m_default_y     = 511.5;
            m_default_w     = 1024;
            m_default_h     = 1024;
            m_default_bin_x = 1;
            m_default_bin_y = 1;

            m_full_x     = 511.5;
            this->m_full_y     = 511.5;
            m_full_w     = 1024;
            this->m_full_h     = 1024;
            m_full_bin_x = 1;
            this->m_full_bin_y = 1;

            m_currentROI.x     = m_default_x;
            m_currentROI.y     = m_default_y;
            m_currentROI.w     = m_default_w;
            m_currentROI.h     = m_default_h;
            m_currentROI.bin_x = m_default_bin_x;
            m_currentROI.bin_y = m_default_bin_y;
            m_nextROI          = m_currentROI;
        }
    }

    ~stdCameraHarness() noexcept override
    {
    }

    // Disambiguating wrappers. MagAOXApp, stdCamera, and telemeter all declare setupConfig() and
    // loadConfig(). These call the stdCamera version and then the telemeter version.
    int setupConfig( mx::app::appConfigurator &config )
    {
        if( stdCameraT::setupConfig( config ) < 0 )
        {
            return -1;
        }

        if constexpr( capsT::usesROI )
        {
            // stdCamera::loadConfig() reads camera.full_x, full_y, full_w, full_h, full_bin_x, and
            // full_bin_y through config(). That only resolves keys registered with config.add(), and
            // stdCamera never registers them. The documentation says derivedT must set them before
            // appStartup(). A real camera app that wants them in a config file adds them itself, so
            // this harness adds them here.
            config.add( "camera.full_x", "", "camera.full_x", argType::Required, "camera", "full_x", false, "float", "" );
            config.add( "camera.full_y", "", "camera.full_y", argType::Required, "camera", "full_y", false, "float", "" );
            config.add( "camera.full_w", "", "camera.full_w", argType::Required, "camera", "full_w", false, "int", "" );
            config.add( "camera.full_h", "", "camera.full_h", argType::Required, "camera", "full_h", false, "int", "" );
            config.add( "camera.full_bin_x", "", "camera.full_bin_x", argType::Required, "camera", "full_bin_x", false, "int", "" );
            config.add( "camera.full_bin_y", "", "camera.full_bin_y", argType::Required, "camera", "full_bin_y", false, "int", "" );
        }

        return telemeterT::setupConfig( config );
    }

    int loadConfig( mx::app::appConfigurator &config )
    {
        if( stdCameraT::loadConfig( config ) < 0 )
        {
            return -1;
        }
        return telemeterT::loadConfig( config );
    }

    // Run only the stdCamera startup logic. This does not start the telemeter log thread. Starting
    // a real background log thread in every Catch2 SECTION re-execution would be needlessly slow.
    // The few tests that record telemetry call startTelemetry() explicitly.
    int appStartup() override
    {
        return stdCameraT::appStartup();
    }

    /// Start the telemeter log thread. Only tests that call recordCamera() or telem() need this.
    int startTelemetry()
    {
        return telemeterT::appStartup();
    }

    int appLogic() override
    {
        return stdCameraT::appLogic();
    }

    int appShutdown() override
    {
        stdCameraT::appShutdown();
        return telemeterT::appShutdown();
    }

    int onPowerOff() override
    {
        return stdCameraT::onPowerOff();
    }

    int whilePowerOff() override
    {
        return stdCameraT::whilePowerOff();
    }

    /// Forward the telemeter record-time check for the stdcam telemetry type.
    int checkRecordTimes()
    {
        return telemeterT::checkRecordTimes( MagAOX::logger::telem_stdcam() );
    }

    /// Stub for sendNewProperty(). Captures the outgoing goto-focus command instead of sending it.
    int sendNewProperty( const pcf::IndiProperty &ipSend )
    {
        m_lastSentProperty = ipSend;
        return m_sendNewPropertyResult;
    }

    // The stdCamera derived-class interface. A real camera app talks to hardware here. Each stub
    // counts its calls, copies the requested value into the current value where one exists, and
    // returns the forced result.
    int powerOnDefaults()
    {
        ++m_powerOnDefaultsCalls;
        return 0;
    }

    int setTempControl()
    {
        ++m_setTempControlCalls;
        m_tempControlStatus    = m_tempControlStatusSet;
        m_tempControlOnTarget  = true;
        this->m_tempControlStatusStr = m_tempControlStatusSet ? "COOLING" : "OFF";
        return m_setTempControlResult;
    }

    int setTempSetPt()
    {
        ++m_setTempSetPtCalls;
        this->m_ccdTemp = m_ccdTempSetpt;
        return m_setTempSetPtResult;
    }

    int setReadoutSpeed()
    {
        ++m_setReadoutSpeedCalls;
        m_readoutSpeedName = m_readoutSpeedNameSet;
        return m_setReadoutSpeedResult;
    }

    int setVShiftSpeed()
    {
        ++m_setVShiftSpeedCalls;
        m_vShiftSpeedName = m_vShiftSpeedNameSet;
        return m_setVShiftSpeedResult;
    }

    int setFanSpeed()
    {
        ++m_setFanSpeedCalls;
        m_fanSpeedName  = m_fanSpeedNameSet;
        m_fanSpeedValid = true;
        return m_setFanSpeedResult;
    }

    int setAnalogGain()
    {
        ++m_setAnalogGainCalls;
        m_analogGainName  = m_analogGainNameSet;
        m_analogGainValid = true;
        return m_setAnalogGainResult;
    }

    int setLED()
    {
        ++m_setLEDCalls;
        m_ledState      = m_ledStateSet;
        m_ledStateValid = true;
        return m_setLEDResult;
    }

    int setEMGain()
    {
        ++m_setEMGainCalls;
        this->m_emGain = m_emGainSet;
        return m_setEMGainResult;
    }

    int setExpTime()
    {
        ++m_setExpTimeCalls;
        m_expTime = m_expTimeSet;
        return m_setExpTimeResult;
    }

    int setFPS()
    {
        ++m_setFPSCalls;
        this->m_fps = m_fpsSet;
        return m_setFPSResult;
    }

    int setSynchro()
    {
        ++m_setSynchroCalls;
        m_synchro = m_synchroSet;
        return m_setSynchroResult;
    }

    int checkNextROI()
    {
        ++m_checkNextROICalls;
        return m_checkNextROIResult;
    }

    int setNextROI()
    {
        ++m_setNextROICalls;
        m_currentROI = m_nextROI;
        return m_setNextROIResult;
    }

    int setCropMode()
    {
        ++m_setCropModeCalls;
        m_cropMode = m_cropModeSet;
        return m_setCropModeResult;
    }

    /// Record the requested shutter state. Zero means closed and anything else means open.
    int setShutter( int ss )
    {
        ++m_setShutterCalls;
        m_lastShutterSS = ss;
        m_shutterState  = ( ss == 0 ) ? 0 : 1;
        return m_setShutterResult;
    }

    bool checkFocus()
    {
        // Mirror the pattern documented for checkFocusSwitchState(). When the focus-state helper is
        // configured, defer to it. Otherwise return the result the test has set.
        if( m_focusStateHelperConfigured )
        {
            return this->checkFocusSwitchState();
        }
        return m_checkFocusResult;
    }

    int gotoFocus()
    {
        ++m_gotoFocusCalls;
        return m_gotoFocusResult;
    }

    std::string stateString()
    {
        return m_stateStringResult;
    }

    bool stateStringValid()
    {
        return m_stateStringValidResult;
    }

    /// Telemeter hook. Force a stdcam telemetry record when the telemeter asks for one.
    int recordTelem( const MagAOX::logger::telem_stdcam * )
    {
        return this->recordCamera( true );
    }

    // Wrappers for the private creator overloads that no stdCamera code path calls. appStartup()
    // only calls createReadoutSpeed() and createVShiftSpeed() with the true tag, and it builds the
    // fan-speed switch inline, so these overloads can only run through a direct call.
    int exposeCreateReadoutSpeedFalse() // the false overload is behind a c_stdCamera_readoutSpeed guard
    {
        mx::meta::trueFalseT<false> f;
        return this->createReadoutSpeed( f );
    }

    int exposeCreateVShiftSpeedFalse() // the false overload is behind a c_stdCamera_vShiftSpeed guard
    {
        mx::meta::trueFalseT<false> f;
        return this->createVShiftSpeed( f );
    }

    int exposeCreateFanSpeedTrue() // appStartup() never calls createFanSpeed() with either tag
    {
        mx::meta::trueFalseT<true> t;
        return this->createFanSpeed( t );
    }

    int exposeCreateFanSpeedFalse() // appStartup() never calls createFanSpeed() with either tag
    {
        mx::meta::trueFalseT<false> f;
        return this->createFanSpeed( f );
    }
};

/// Harness with every capability on and the telemeter mixed in.
using stdCameraFullHarness = stdCameraHarness<stdCameraFullCaps>;

/// Harness that only reports temperature and FPS.
using stdCameraReportOnlyHarness = stdCameraHarness<stdCameraReportOnlyCaps>;

} // namespace stdCamera_tests

// LCOV_EXCL_STOP
