/** \file MagAOXApp_test.hpp
  * \brief Shared Catch2 test harness for MagAOX::app::MagAOXApp.
  *
  * This header is used by MagAOXApp_test.cpp, MagAOXAppExecute_test.cpp, and the
  * per-fault translation units that re-include it under a fault namespace.
  *
  * APP_XWCTEST_BASE selects the base class of the concrete test app. When
  * XWCTEST_NAMESPACE is not defined the base is the production MagAOXApp<true>.
  * When a fault translation unit defines XWCTEST_NAMESPACE before including
  * MagAOXApp.hpp, the base is the copy of MagAOXApp<true> that was re-included inside
  * that namespace with one XWCTEST_* fault macro enabled. The namespace nesting below
  * mirrors that scheme, so each fault translation unit gets its own harness types and
  * there are no one-definition-rule collisions.
  *
  * The MagAOXApp_test harness exposes protected members of MagAOXApp through thin
  * public wrappers. It adds flags that make appStartup(), appLogic(), appShutdown(),
  * onPowerOff(), and whilePowerOff() fail on demand. It does not stub out logging,
  * INDI, or the file system. Tests that need those use real files under /tmp and a
  * real indiDriver object that has no FIFOs and no indiserver to talk to.
  */
#ifndef app_tests_MagAOXApp_test_hpp
#define app_tests_MagAOXApp_test_hpp

#include "../dev/tests/testHarnessCommon.hpp"

namespace libXWCTest
{
namespace appTest
{
namespace MagAOXAppTest
{

#undef APP_XWCTEST_BASE
#ifdef XWCTEST_NAMESPACE
    #define APP_XWCTEST_BASE MagAOX::app::XWCTEST_NAMESPACE::MagAOXApp<true>
#else
    #define APP_XWCTEST_BASE MagAOX::app::MagAOXApp<true>
#endif

#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE
{
#endif

/// Test harness for MagAOXApp<true>.
/** It stubs the appStartup(), appLogic(), and appShutdown() hooks with flags that make each
  * one fail on demand, and exposes protected MagAOXApp members through public wrappers.
  *
  * \ingroup MagAOXApp_unit_test
  */
struct MagAOXApp_test : public APP_XWCTEST_BASE
{
    MagAOXApp_test( bool gitmod = false ) : MagAOXApp( "sha1", gitmod )
    {
    }

    bool appStartupFail{ false };
    bool appLogicFail{ false };
    bool appShutdownFail{ false };

    // Thread bookkeeping for the threadStart() tests. The thread function that
    // threadStart() launches only receives the thisPtr argument, so the thread id and the
    // init flag must be reachable through the app object. Real derived apps expose them
    // the same way, for example dev::dm with m_satThreadID and m_satThreadInit.
    bool  m_testThreadInit{ false };
    pid_t m_testThreadID{ 0 };

    void addUnusedConfig()
    {
        config.add( "name2", "", "name2", argType::Required, "", "", true, "string", "" );
        config.m_sources = true;
    }

    void setup( int argc, char **argv )
    {
        APP_XWCTEST_BASE::setup( argc, argv );
    }

    virtual int appStartup()
    {
        if( appStartupFail )
        {
            return -1;
        }

        return 0;
    }
    int m_appLogicCallCount{ 0 }; ///< Number of times appLogic() has run.
    /// Call number of appLogic() on which to simulate an external power off.
    /** If greater than zero, the appLogic() call with this number forces m_powerState
      * back to 0 and makes the next onPowerOff() call fail. This exercises the branch in
      * the main loop of execute() that handles power going off while state() is not
      * POWEROFF. Reaching that branch needs power to come on, appLogic() to run at least
      * once, and power to go off again. The simple bool flags such as appLogicFail
      * cannot express that sequence.
      */
    int m_flipPowerOffOnCall{ 0 };

    virtual int appLogic()
    {
        if( appLogicFail )
        {
            return -1;
        }

        ++m_appLogicCallCount;
        if( m_flipPowerOffOnCall > 0 && m_appLogicCallCount == m_flipPowerOffOnCall )
        {
            m_powerState   = 0;
            onPowerOffFail = true;
        }

        return 0;
    }
    virtual int appShutdown()
    {
        if( appShutdownFail )
        {
            return -1;
        }

        return 0;
    }

    // Public wrappers for the protected INDI startup functions.
    int callCreateINDIFIFOS()
    {
        return APP_XWCTEST_BASE::createINDIFIFOS();
    }

    int callStartINDI()
    {
        return APP_XWCTEST_BASE::startINDI();
    }

    void callSendGetPropertySetList( bool all )
    {
        APP_XWCTEST_BASE::sendGetPropertySetList( all );
    }

    std::string configPathGlobal()
    {
        return APP_XWCTEST_BASE::m_configPathGlobal;
    }

    std::string configPathUser()
    {
        return APP_XWCTEST_BASE::m_configPathUser;
    }

    std::string configPathLocal()
    {
        return APP_XWCTEST_BASE::m_configPathLocal;
    }

    std::string &invokedName()
    {
        return APP_XWCTEST_BASE::invokedName;
    }

    bool &doHelp()
    {
        return APP_XWCTEST_BASE::doHelp;
    }

    bool configOnly()
    {
        return APP_XWCTEST_BASE::m_configOnly;
    }

    void setPowerMgtEnabled( bool pme )
    {
        m_powerMgtEnabled = pme;
    }

    void setConfigName( const std::string &cn )
    {
        m_configName = cn;

        m_indiDriver =
            MagAOX::app::dev::testHarness::makeFifolessIndiDriver<APP_XWCTEST_BASE>( this, m_configName );
    }

    void setConfigBase( const std::string &cb )
    {
        m_configBase = cb;
    }

    // Public wrappers for the protected INDI send functions and the clearFSMAlert callback.
    template <typename T>
    int sendNewProperty( const pcf::IndiProperty &ipSend, const std::string &el, const T &newVal )
    {
        return APP_XWCTEST_BASE::sendNewProperty( ipSend, el, newVal );
    }

    int sendNewProperty( const pcf::IndiProperty &ipSend )
    {
        return APP_XWCTEST_BASE::sendNewProperty( ipSend );
    }

    int sendNewStandardIndiToggle( const std::string &device, const std::string &property, bool onoff )
    {
        return APP_XWCTEST_BASE::sendNewStandardIndiToggle( device, property, onoff );
    }

    int newCallBack_clearFSMAlert( const pcf::IndiProperty &ipRecv )
    {
        return APP_XWCTEST_BASE::newCallBack_clearFSMAlert( ipRecv );
    }

    int called_back{ 0 };

    void setAlert()
    {
        m_stateAlert = true;
    }

    void doFSMClearAlert()
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.setDevice( configName() );
        ip.setName( "fsm_clear_alert" );
        ip.add( pcf::IndiElement( "request" ) );
        ip["request"].setSwitchState( pcf::IndiElement::On );

        st_newCallBack_clearFSMAlert( this, ip );
    }

    std::string powerDevice()
    {
        return m_powerDevice;
    }

    std::string powerChannel()
    {
        return m_powerChannel;
    }

    std::string powerElement()
    {
        return m_powerElement;
    }

    std::string powerTargetElement()
    {
        return m_powerTargetElement;
    }

    int powerOnWait()
    {
        return m_powerOnWait;
    }

    // When set, the matching power hook returns -1 so execute() takes its error branch.
    bool onPowerOffFail{ false };
    bool whilePowerOffFail{ false };

    int onPowerOff()
    {
        if( onPowerOffFail )
        {
            return -1;
        }
        return APP_XWCTEST_BASE::onPowerOff();
    }

    int whilePowerOff()
    {
        if( whilePowerOffFail )
        {
            return -1;
        }
        return APP_XWCTEST_BASE::whilePowerOff();
    }

    bool powerOnWaitElapsed()
    {
        return APP_XWCTEST_BASE::powerOnWaitElapsed();
    }

    int powerState()
    {
        return APP_XWCTEST_BASE::powerState();
    }

    void configurePowerManagement( const std::string &device, const std::string &channel )
    {
        m_indiP_powerChannel = pcf::IndiProperty( pcf::IndiProperty::Text );
        m_powerDevice        = device;
        m_indiP_powerChannel.setDevice( device );

        m_powerChannel = channel;
        m_indiP_powerChannel.setName( channel );
    }

    void configurePowerOnWait( unsigned long powerOnWait, int powerOnCounter, int loopPause )
    {
        m_powerOnWait    = powerOnWait;
        m_powerOnCounter = powerOnCounter;
        m_loopPause      = loopPause;
    }

    int setPowerState( const std::string &state, const std::string target )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );
        ip.setDevice( m_powerDevice );
        ip.setName( m_powerChannel );
        ip.add( pcf::IndiElement( "state" ) );
        ip["state"].setValue( state );

        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].setValue( target );

        return setCallBack_m_indiP_powerChannel( ip );
    }

    /// Set the power state through the static callback wrapper.
    /** This calls the static wrapper that INDI_SETCALLBACK_DECL generates, in the same way
      * that doFSMClearAlert() calls st_newCallBack_clearFSMAlert. setPowerState() above
      * calls the real handler directly, so without this the static wrapper is never
      * exercised.
      */
    int setPowerStateViaStaticWrapper( const std::string &state, const std::string target )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );
        ip.setDevice( m_powerDevice );
        ip.setName( m_powerChannel );
        ip.add( pcf::IndiElement( "state" ) );
        ip["state"].setValue( state );

        ip.add( pcf::IndiElement( "target" ) );
        ip["target"].setValue( target );

        return st_setCallBack_m_indiP_powerChannel( this, ip );
    }

    int setSigTermHandler()
    {
        return APP_XWCTEST_BASE::setSigTermHandler();
    }

    void _handlerSigTerm( int signum, siginfo_t *siginf, void *ucont )
    {
        APP_XWCTEST_BASE::_handlerSigTerm( signum, siginf, ucont );
    }

    int setEuidReal()
    {
        return APP_XWCTEST_BASE::setEuidReal();
    }

    int setEuidReal( int euidr, bool set = true )
    {
        m_euidReal = euidr;

        if( set )
        {
            return APP_XWCTEST_BASE::setEuidReal();
        }

        return 0;
    }

    int setEuidCalled()
    {
        return APP_XWCTEST_BASE::setEuidCalled();
    }

    int setEuidCalled( int euidc )
    {
        m_euidCalled = euidc;
        return APP_XWCTEST_BASE::setEuidCalled();
    }

    int lockPID()
    {
        return APP_XWCTEST_BASE::lockPID();
    }

    int unlockPID()
    {
        return APP_XWCTEST_BASE::unlockPID();
    }

    /// Exercise the early returns in the elevatedPrivileges RAII guard.
    /** The guard elevates in its constructor. A second elevate() and a second restore()
      * each take the early-return branch for a state that is already applied.
      */
    void testElevatedPrivilegesDoubleGuard()
    {
        APP_XWCTEST_BASE::elevatedPrivileges ep( this );
        ep.elevate(); // Already elevated, so this takes the early return.
        ep.restore();
        ep.restore(); // Already restored, so this takes the early return.
    }

    // Public wrappers for the protected updateIfChanged family and indiTargetUpdate().
    template <typename T>
    void updateIfChanged( pcf::IndiProperty &p, const std::string &el, const T &newVal )
    {
        APP_XWCTEST_BASE::updateIfChanged( p, el, newVal );
    }

    template <typename T>
    int indiTargetUpdate( pcf::IndiProperty &localProperty, T &localTarget, const pcf::IndiProperty &remoteProperty, bool setBusy )
    {
        return APP_XWCTEST_BASE::template indiTargetUpdate( localProperty, localTarget, remoteProperty, setBusy );
    }

    void updateIfChanged( pcf::IndiProperty &p, const std::string &el, const char *newVal )
    {
        APP_XWCTEST_BASE::updateIfChanged( p, el, newVal );
    }

    template <typename T>
    void updatesIfChanged( pcf::IndiProperty &p,
                           const std::vector<const char *> &els,
                           const std::vector<T> &newVals,
                           pcf::IndiProperty::PropertyStateType newState = pcf::IndiProperty::Ok )
    {
        APP_XWCTEST_BASE::updatesIfChanged( p, els, newVals, newState );
    }

    template <typename T>
    void updateIfChanged( pcf::IndiProperty &p,
                          const std::vector<std::string> &els,
                          const std::vector<T> &newVals,
                          pcf::IndiProperty::PropertyStateType newState = pcf::IndiProperty::Ok )
    {
        APP_XWCTEST_BASE::updateIfChanged( p, els, newVals, newState );
    }

    void updateSwitchIfChanged( pcf::IndiProperty &p,
                                const std::string &el,
                                const pcf::IndiElement::SwitchStateType &newVal,
                                pcf::IndiProperty::PropertyStateType ipState = pcf::IndiProperty::Ok )
    {
        APP_XWCTEST_BASE::updateSwitchIfChanged( p, el, newVal, ipState );
    }

    std::string setPropertyKey( const pcf::IndiProperty &prop )
    {
        return prop.createUniqueKey();
    }

    void resetSetPropertyRetry( const std::string &key )
    {
        APP_XWCTEST_BASE::resetIndiSetPropertyRetry( m_indiSetCallBacks.at( key ) );
    }

    bool shouldRequestSetProperty( const std::string &key, bool all, const std::chrono::steady_clock::time_point &now )
    {
        return APP_XWCTEST_BASE::indiSetPropertyShouldRequest( m_indiSetCallBacks.at( key ), all, now );
    }

    void noteSetPropertyRequested( const std::string &key, const std::chrono::steady_clock::time_point &now )
    {
        APP_XWCTEST_BASE::noteIndiSetPropertyRequested( m_indiSetCallBacks.at( key ), now );
    }

    uint32_t setPropertyRetryCount( const std::string &key )
    {
        return m_indiSetCallBacks.at( key ).m_retryCount;
    }

    std::chrono::steady_clock::duration setPropertyRetryDelay( const std::string &key )
    {
        return m_indiSetCallBacks.at( key ).m_retryDelay;
    }

    std::chrono::steady_clock::time_point setPropertyNextRetry( const std::string &key )
    {
        return m_indiSetCallBacks.at( key ).m_nextRetry;
    }

    bool setPropertyMissingLogged( const std::string &key )
    {
        return m_indiSetCallBacks.at( key ).m_missingLogged;
    }

    void markSetPropertyReceived( const std::string &key, bool received )
    {
        m_indiSetCallBacks.at( key ).m_defReceived = received;
    }
};

int callback( void *app, const pcf::IndiProperty &ipRecv )
{
    static_cast<void>( ipRecv ); // be unused

    MagAOXApp_test *appt = static_cast<MagAOXApp_test *>( app );

    appt->called_back = 1;

    return 0;
}

#ifdef XWCTEST_NAMESPACE
} // namespace XWCTEST_NAMESPACE
#endif

} // namespace MagAOXAppTest
} // namespace appTest
} // namespace libXWCTest

#endif // app_tests_MagAOXApp_test_hpp
