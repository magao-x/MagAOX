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

    ~MagAOXApp_test() noexcept (true)
    {}

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

    void p_handlerSigTerm( int signum, siginfo_t *siginf, void *ucont )
    {
        _handlerSigTerm( signum, siginf, ucont );
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

    int p_elevatePriveleges()
    {
        elevatedPrivileges elPriv( this );
        elPriv.elevate();
        elPriv.restore();

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

    // Thread
    int m_threadPrio{ 0 }; ///< Priority of the framegrabber thread, should normally be > 00.

    std::string m_cpuset; ///< The cpuset to assign the framegrabber thread to.  Not used if empty, the default.

    bool m_threadInit{ true }; ///< Synchronizer for thread startup, to allow priority setting to finish.

    pid_t m_threadID{ 0 }; ///< The ID of the thread.

    pcf::IndiProperty m_threadProp; ///< The property to hold the thread details.

    std::thread m_thread; ///< A separate thread

    bool m_threadRunning {false};
    bool m_threadStopped {false};
    int m_threadError {0};

    int threadStartTest()
    {
        if( threadStart( m_thread,
                         m_threadInit,
                         m_threadID,
                         m_threadProp,
                         m_threadPrio,
                         m_cpuset,
                         "thread",
                         this,
                         threadStarter ) < 0 )
        {

            return -1;
        }


        while(m_threadError == 0 && m_threadRunning == false)
        {
            sleep(1);
        }

        if(m_threadError != 0)
        {
            return m_threadError;
        }

        if(m_threadRunning != true)
        {
            return -3;
        }

        m_threadRunning = false;

        while(m_threadError == 0 && m_threadStopped == false)
        {
            sleep(1);
        }

        if(m_threadError != 0)
        {
            return m_threadError;
        }

        XWCAPP_THREAD_STOP(m_thread);

        return 0;
    }
    /// Thread starter, called by MagAOXApp::threadStart on thread construction.  Calls threadExec.
    static void threadStarter( MagAOXApp_test *o /**< [in] a pointer to aninstance (normally this) */ )
    {
        o->threadExec();
    }

    /// Execute framegrabbing.
    void threadExec()
    {
        // Get the thread PID immediately so the caller can return.
        m_threadID = syscall( SYS_gettid );

        // Wait for the thread starter to finish initializing this thread.
        int n = 0;
        while( m_threadInit == true && n < 5)
        {
            sleep( 1 );
            ++n;
        }

        if(n >= 5)
        {
            m_threadError = -2;
            return;
        }

        std::cerr << "threading\n";

        m_threadRunning = true;

        while(m_threadRunning)
        {
            sleep( 1 );
        }

        m_threadRunning = false;
        m_threadStopped = true;
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
