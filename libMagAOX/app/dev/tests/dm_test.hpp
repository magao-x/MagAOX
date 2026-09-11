/** \file dm_test.hpp
  * \brief Test harness for the MagAOX::app::dev::dm device mixin.
  *
  * Declares one harness class template, dmHarness<realT>. It derives from appHarnessBase
  * in testHarnessCommon.hpp and mixes in dm<dmHarness<realT>, realT> and a real
  * shmimMonitor base. realT is the DM data type. The names, calibration directory, and
  * config name for each realT come from dmHarnessTraits<realT>. Two instantiations are used:
  *
  * - dmTest is dmHarness<float>. It drives every dm<> code path.
  * - dmTestBadType is dmHarness<int>. int has no ImageStreamTypeCode<> specialization, so
  *   m_dmDataType is 0 and appStartup() must reject it. This reaches the unsupported data
  *   type error path.
  *
  * The hardware hooks initDM(), zeroDM(), releaseDM(), and commandDM() are stubs whose return
  * values a test can set. Tests drive the mixin through its lifecycle functions, its public
  * API, and its INDI callbacks with hand-built pcf::IndiProperty objects. The harness only
  * exposes what those routes cannot reach. That is protected dm<> state that tests assert on,
  * the shmimMonitor stream size, the shutdown flag and saturation thread handle, and a few
  * MagAOXApp<false> functions that MagAOXApp_test.cpp cannot reach, because that file only
  * instantiates MagAOXApp<true>.
  *
  * The common parts of every dev:: harness, such as the FIFO-less indiDriver installed by
  * setupRealDriver(), the registration fault injection, and public access to the power
  * management state, come from appHarnessBase.
  *
  * \ingroup dm_tests
  */

#include "../../MagAOXApp.hpp"
#include "../dm.hpp"
#include "../shmimMonitor.hpp"
#include "testHarnessCommon.hpp"

// LCOV_EXCL_START

#ifndef XWCTEST_NAMESPACE
    #define MAPPNS MagAOX::app::dev
#else
    #define MAPPNS MagAOX::app::dev::XWCTEST_NAMESPACE

#endif

namespace dm_tests
{

#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE
{
#endif

/// Names used by dmHarness<realT>. Each supported realT has a specialization.
template <class realT>
struct dmHarnessTraits;

/// Names for the float harness, dmTest.
template <>
struct dmHarnessTraits<float>
{
    static constexpr const char *configName  = "dmtest";
    static constexpr const char *calibDir    = "/tmp/dmtest_calibs";
    static constexpr const char *calibRelDir = "dmtest";
};

/// Names for the int harness, dmTestBadType.
template <>
struct dmHarnessTraits<int>
{
    static constexpr const char *configName  = "dmtestbadtype";
    static constexpr const char *calibDir    = "/tmp/dmtestbadtype_calibs";
    static constexpr const char *calibRelDir = "dmtestbadtype";
};

/// Test harness for dev::dm. realT is the DM data type. See the file comment.
/**
 * \ingroup dm_tests
 */
template <class realT>
struct dmHarness : public MagAOX::app::dev::testHarness::appHarnessBase,
                   public MAPPNS::dm<dmHarness<realT>, realT>,
                   public MAPPNS::shmimMonitor<dmHarness<realT>>
{
    typedef MagAOX::app::dev::testHarness::appHarnessBase baseT;
    typedef MAPPNS::dm<dmHarness<realT>, realT>           dmT;
    typedef dmHarnessTraits<realT>                        traitsT;

    friend dmT;

    // Protected dm<> state that tests assert on. dm<> has no public getters for these.
    using dmT::m_flatCommands;
    using dmT::m_testCommands;
    using dmT::m_flatLoaded;
    using dmT::m_testLoaded;
    using dmT::m_flatSet;
    using dmT::m_testSet;
    using dmT::m_flatCurrent;
    using dmT::m_testCurrent;

    // Protected saturation maps. Tests fill them so a later clearSat() has something to clear.
    using dmT::m_instSatMap;
    using dmT::m_accumSatMap;

    /// Set the config name and calibration directories for this realT. Also install the no-op
    /// SIGUSR1 handler, because appShutdown() interrupts the saturation thread with SIGUSR1
    /// and the default disposition would end the test process. MagAOXApp::execute() normally
    /// installs it.
    dmHarness() : baseT( traitsT::configName )
    {
        this->m_calibDir     = traitsT::calibDir;
        this->m_calibRelDir  = traitsT::calibRelDir;

        struct sigaction act;
        memset( &act, 0, sizeof( act ) );
        act.sa_sigaction = &MagAOX::app::sigUsr1Handler;
        act.sa_flags     = SA_SIGINFO;
        sigemptyset( &act.sa_mask );
        sigaction( SIGUSR1, &act, 0 );
    }

    /// Stop the saturation thread if a test started it with appStartup() and did not shut it
    /// down, for instance because a REQUIRE failed. A joinable std::thread would otherwise
    /// call std::terminate() when it is destroyed.
    ~dmHarness() noexcept override
    {
        requestShutdown();
        dmT::appShutdown();
    }

    int setupConfig( mx::app::appConfigurator &config )
    {
        return dmT::setupConfig( config );
    }

    int loadConfig( mx::app::appConfigurator &config )
    {
        return dmT::loadConfig( config );
    }

    int appStartup()
    {
        return dmT::appStartup();
    }

    int appLogic()
    {
        return dmT::appLogic();
    }

    int appShutdown()
    {
        return dmT::appShutdown();
    }

    /// The next three functions forward the MagAOXApp hooks to the dm mixin. The forwarders
    /// are needed because MagAOXApp and shmimMonitor declare the same names, so an unqualified
    /// call on the harness would be ambiguous.
    int onPowerOff()
    {
        return dmT::onPowerOff();
    }

    int whilePowerOff()
    {
        return dmT::whilePowerOff();
    }

    int updateINDI()
    {
        return dmT::updateINDI();
    }

    /// Return values of the stub hardware hooks initDM(), zeroDM(), and releaseDM().
    /// A test sets one of them to -1 to make that hook fail.
    int m_initDMRV{ 0 };
    int initDM()
    {
        return m_initDMRV;
    }

    int m_zeroDMRV{ 0 };
    int zeroDM()
    {
        return m_zeroDMRV;
    }

    int m_releaseDMRV{ 0 };
    int releaseDM()
    {
        return m_releaseDMRV;
    }

    /// Set the shmimMonitor stream size that allocate() checks against the configured DM.
    /// This is protected shmimMonitor state that only its own stream thread sets.
    void setSize( int w, int h, int d )
    {
        this->m_width    = w;
        this->m_height   = h;
        this->m_dataType = d;
    }

    /// Value written into m_instSatMap by commandDM(), settable by tests.
    uint8_t m_testSatValue{ 0 };

    /// If true, commandDM() returns -1 to exercise the processImage() error path.
    bool m_commandDMFail{ false };

    /// Stub for the hardware hook that dm<>::processImage() calls to command the DM.
    /// It fills the instantaneous saturation map with m_testSatValue, or returns -1
    /// when m_commandDMFail is set.
    int commandDM( void *curr_src )
    {
        static_cast<void>( curr_src );
        if( m_commandDMFail )
        {
            return -1;
        }
        if( m_instSatMap.rows() > 0 && m_instSatMap.cols() > 0 )
        {
            m_instSatMap.setConstant( m_testSatValue );
        }
        return 0;
    }

    /// Set the shutdown flag the way a signal handler would. m_shutdown is protected with no
    /// public setter, and the saturation thread loops only exit once it is set.
    void requestShutdown()
    {
        this->m_shutdown = 1;
    }

    /// Call MagAOXApp<false>::startINDI(). It is protected, and trivially succeeds when
    /// _useINDI is false. MagAOXApp_test.cpp cannot reach this branch.
    int startINDI()
    {
        return MagAOX::app::MagAOXApp<false>::startINDI();
    }

    /// Call MagAOXApp<false>::createINDIFIFOS(). It is protected, and trivially succeeds when
    /// INDI is compiled out. MagAOXApp_test.cpp cannot reach this branch.
    int callCreateINDIFIFOS()
    {
        return MagAOX::app::MagAOXApp<false>::createINDIFIFOS();
    }

    /// Call MagAOXApp<false>::sendNewStandardIndiToggle(). It is protected, the dm mixin never
    /// calls it, and it trivially succeeds when _useINDI is false. MagAOXApp_test.cpp cannot
    /// reach this branch.
    int callSendNewStandardIndiToggle( const std::string &device, const std::string &property, bool onoff )
    {
        return MagAOX::app::MagAOXApp<false>::sendNewStandardIndiToggle( device, property, onoff );
    }

    /// Swap the saturation thread object out and overwrite the copy with an empty thread.
    /// The handle is leaked on purpose. appLogic() has already reaped the thread with
    /// pthread_tryjoin_np(), and destroying a joinable std::thread would call
    /// std::terminate(). shmimMonitor_test.cpp uses the same pattern.
    void abandonSatThread()
    {
        std::thread tmp;
        tmp.swap( this->m_satThread );
        new( &tmp ) std::thread();
    }
};

/// The float harness. This drives every dm<> code path.
using dmTest = dmHarness<float>;

/// The int harness. appStartup() must reject int as an unsupported DM data type.
using dmTestBadType = dmHarness<int>;

#ifdef XWCTEST_NAMESPACE
} // namespace XWCTEST_NAMESPACE
#endif

} // namespace dm_tests

// LCOV_EXCL_STOP
