#ifndef app_tests_MagAOXApp_test_hpp
#define app_tests_MagAOXApp_test_hpp

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

struct MagAOXApp_test : public APP_XWCTEST_BASE
{
    MagAOXApp_test( bool gitmod = false ) : MagAOXApp( "sha1", gitmod )
    {
    }

    bool appStartupFail{ false };
    bool appLogicFail{ false };
    bool appShutdownFail{ false };

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
    virtual int appLogic()
    {
        if( appLogicFail )
        {
            return -1;
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

        m_indiDriver = new MagAOX::app::indiDriver<APP_XWCTEST_BASE>( this, m_configName, "0", "0" );
    }

    void setConfigBase( const std::string &cb )
    {
        m_configBase = cb;
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

    int onPowerOff()
    {
        return APP_XWCTEST_BASE::onPowerOff();
    }

    int whilePowerOff()
    {
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
