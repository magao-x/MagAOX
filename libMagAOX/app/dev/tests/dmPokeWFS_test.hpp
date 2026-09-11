/** \file dmPokeWFS_test.hpp
  * \brief Test harness for the MagAOX::app::dev::dmPokeWFS device mixin.
  *
  * Declares dmPokeWFSTest, an appHarnessBase that mixes in dmPokeWFS and two real
  * shmimMonitor bases, one for the WFS camera stream and one for the dark stream.
  * The harness stubs out the telemeter interface with a call counter and replaces
  * runSensor() and analyzeSensor() with controllable stubs that can return canned
  * values, sleep, or call the real basicRunSensor(). Tests drive the mixin through
  * its lifecycle functions, its public shmimMonitor interface, the runSensor() and
  * analyzeSensor() stubs, and its INDI callbacks called with hand built properties.
  * The few protected members exposed below are the ones no such path reaches. The
  * common parts of every dev:: harness, such as the constructor that sets the config
  * name, the FIFO-less indiDriver, and the registration fault injection, come from
  * appHarnessBase in testHarnessCommon.hpp.
  *
  * \ingroup dmPokeWFS_tests
  */

#include "../../MagAOXApp.hpp"
#include "../dmPokeWFS.hpp"
#include "../shmimMonitor.hpp"
#include "testHarnessCommon.hpp"

// LCOV_EXCL_START

#ifndef XWCTEST_NAMESPACE
    #define MAPPNS MagAOX::app::dev
#else
    #define MAPPNS MagAOX::app::dev::XWCTEST_NAMESPACE
#endif

namespace dmPokeWFS_tests
{

#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE
{
#endif

/// Test harness for dev::dmPokeWFS.
/** An appHarnessBase that mixes in dmPokeWFS and two shmimMonitor bases. It stubs
 * the telemeter interface and the runSensor() and analyzeSensor() hooks.
 *
 * \ingroup dmPokeWFS_tests
 */
struct dmPokeWFSTest : public MagAOX::app::dev::testHarness::appHarnessBase,
                       public MAPPNS::dmPokeWFS<dmPokeWFSTest>,
                       public MAPPNS::shmimMonitor<dmPokeWFSTest, MAPPNS::dmPokeWFS<dmPokeWFSTest>::wfsShmimT>,
                       public MAPPNS::shmimMonitor<dmPokeWFSTest, MAPPNS::dmPokeWFS<dmPokeWFSTest>::darkShmimT>
{
    typedef MagAOX::app::dev::testHarness::appHarnessBase baseT;

    friend class MAPPNS::dmPokeWFS<dmPokeWFSTest>;
    typedef MAPPNS::dmPokeWFS<dmPokeWFSTest> dmPokeWFST;

    friend class MAPPNS::shmimMonitor<dmPokeWFSTest, dmPokeWFST::wfsShmimT>;
    typedef MAPPNS::shmimMonitor<dmPokeWFSTest, dmPokeWFST::wfsShmimT> shmimMonitorT;

    friend class MAPPNS::shmimMonitor<dmPokeWFSTest, dmPokeWFST::darkShmimT>;
    typedef MAPPNS::shmimMonitor<dmPokeWFSTest, dmPokeWFST::darkShmimT> darkShmimMonitorT;

    dmPokeWFSTest() : baseT( "dmPokeWFStest" )
    {
    }

    ~dmPokeWFSTest() noexcept
    {
    }

    // The two shmimMonitor accessors are part of the interface dmPokeWFS requires of derivedT.

    shmimMonitorT &shmimMonitor()
    {
        return *static_cast<shmimMonitorT *>( this );
    }

    darkShmimMonitorT &darkShmimMonitor()
    {
        return *static_cast<darkShmimMonitorT *>( this );
    }

    int setupConfig( mx::app::appConfigurator &config )
    {
        return dmPokeWFST::setupConfig( config );
    }

    int loadConfig( mx::app::appConfigurator &config )
    {
        return dmPokeWFST::loadConfig( config );
    }

    int appStartup()
    {
        return dmPokeWFST::appStartup();
    }

    int appLogic()
    {
        return dmPokeWFST::appLogic();
    }

    int appShutdown()
    {
        return dmPokeWFST::appShutdown();
    }

    /// Stub of the telemeter interface. dmPokeWFS::recordPokeLoop() calls
    /// derived().template telem<telem_pokeloop>(msg). No real dev::telemeter is used here.
    /// The stub only counts calls.
    int m_telemCount{ 0 };

    template <class telT>
    int telem( const typename telT::messageT &msg )
    {
        static_cast<void>( msg );
        ++m_telemCount;
        return 0;
    }

    // The runSensor() and analyzeSensor() interface that dmPokeWFS requires. Tests control
    // the behavior through the members below. These stubs are also the only way to reach
    // the protected basicRunSensor() and updateMeasurement().

    bool m_useRealRunSensor{ false }; ///< If true, runSensor() calls the real basicRunSensor().
    int  m_runSensorRV{ 0 };          ///< Canned return value when not using the real sensor.
    bool m_lastFirstRun{ false };     ///< The firstRun argument of the last runSensor() call.
    int  m_runSensorCalls{ 0 };       ///< Number of runSensor() calls so far.
    int  m_runSensorSleepMs{ 0 }; ///< If greater than zero, runSensor() sleeps this long so a test can observe the measuring state.

    /// Stub runSensor(). Records the call, optionally sleeps, then returns the canned value
    /// or the result of the real basicRunSensor().
    int runSensor( bool firstRun )
    {
        m_lastFirstRun = firstRun;
        ++m_runSensorCalls;
        if( m_runSensorSleepMs > 0 )
        {
            mx::sys::milliSleep( m_runSensorSleepMs );
        }
        if( m_useRealRunSensor )
        {
            return basicRunSensor();
        }
        return m_runSensorRV;
    }

    int   m_analyzeSensorRV{ 0 };    ///< If negative, analyzeSensor() returns this without updating.
    float m_testDeltaX{ 0 };         ///< Delta x that analyzeSensor() reports.
    float m_testDeltaY{ 0 };         ///< Delta y that analyzeSensor() reports.
    int   m_analyzeSensorCalls{ 0 }; ///< Number of analyzeSensor() calls so far.

    /// Stub analyzeSensor(). Either fails with the canned value or reports the test deltas.
    int analyzeSensor()
    {
        ++m_analyzeSensorCalls;
        if( m_analyzeSensorRV < 0 )
        {
            return m_analyzeSensorRV;
        }
        return updateMeasurement( m_testDeltaX, m_testDeltaY );
    }

    /// Telemeter interface used by real apps. It is not exercised directly here.
    int checkRecordTimes()
    {
        return 0;
    }

    // shmimMonitor only sets its size and data type when its own thread connects to a
    // live stream. These tests run allocate() and processImage() directly, so they set
    // the size by hand.

    void setWfsSize( uint32_t w, uint32_t h, uint8_t dtype )
    {
        shmimMonitorT::m_width    = w;
        shmimMonitorT::m_height   = h;
        shmimMonitorT::m_dataType = dtype;
    }

    void setDarkSize( uint32_t w, uint32_t h, uint8_t dtype )
    {
        darkShmimMonitorT::m_width    = w;
        darkShmimMonitorT::m_height   = h;
        darkShmimMonitorT::m_dataType = dtype;
    }

    // Protected dmPokeWFS state with no public getter. Tests read it to synchronize with
    // the wfs thread and to check the results of a measurement.
    using dmPokeWFST::m_measuring;
    using dmPokeWFST::m_counter;
    using dmPokeWFST::m_deltaX;
    using dmPokeWFST::m_deltaY;
    using dmPokeWFST::m_rawImage;
    using dmPokeWFST::m_pokeImage;

    // The harness is a MagAOXApp<false>, so handleNewProperty() and handleSetProperty()
    // never dispatch. Tests call the callbacks and their static trampolines directly.
    using dmPokeWFST::newCallBack_m_indiP_poke_amp;
    using dmPokeWFST::newCallBack_m_indiP_nPokeImages;
    using dmPokeWFST::newCallBack_m_indiP_nPokeAverage;
    using dmPokeWFST::setCallBack_m_indiP_wfsFps;
    using dmPokeWFST::newCallBack_m_indiP_single;
    using dmPokeWFST::newCallBack_m_indiP_continuous;
    using dmPokeWFST::newCallBack_m_indiP_stop;
    using dmPokeWFST::st_newCallBack_m_indiP_poke_amp;
    using dmPokeWFST::st_newCallBack_m_indiP_nPokeImages;
    using dmPokeWFST::st_newCallBack_m_indiP_nPokeAverage;
    using dmPokeWFST::st_setCallBack_m_indiP_wfsFps;
    using dmPokeWFST::st_newCallBack_m_indiP_single;
    using dmPokeWFST::st_newCallBack_m_indiP_continuous;
    using dmPokeWFST::st_newCallBack_m_indiP_stop;

    // The telemeter interface is stubbed, so nothing else calls recordTelem(). The wfs
    // thread only calls recordPokeLoop() after a change, so the unchanged path needs a
    // direct call too.
    using dmPokeWFST::recordTelem;
    using dmPokeWFST::recordPokeLoop;

    /// m_shutdown is protected in MagAOXApp and only its signal handlers set it. The wfs
    /// thread's outer loop exits only once it is nonzero.
    void requestShutdown()
    {
        m_shutdown = 1;
    }

    /// Initialize the image and wfs semaphores without appStartup(), which would also start
    /// the three real threads. This lets basicRunSensor() and processImage() run alone.
    void initSemaphoresForTest()
    {
        sem_init( &m_imageSemaphore, 0, 0 );
        sem_init( &m_wfsSemaphore, 0, 0 );
    }

    /// Post the wfs semaphore with neither m_single nor m_continuous set. Both callbacks
    /// always set one of the flags before posting, so this is the only way to reach the
    /// defensive branch in wfsThreadExec() that makes the thread return outright.
    void postWfsSemaphoreWithNoModeSet()
    {
        m_single     = 0;
        m_continuous = 0;
        sem_post( &m_wfsSemaphore );
    }

    // Direct control of the std::thread bookkeeping in shmimMonitor and dmPokeWFS. This
    // mirrors the setSmThread() and abandonSmThread() pattern in shmimMonitor_test.cpp.
    // The two shmimMonitor threads only exit on shutdown, and then both exit together, so
    // the appLogic() branch for one exited monitor thread while the other is alive is
    // reachable only with test threads installed here. The abandon helpers swap the
    // std::thread of a thread appLogic() has already joined into a local and placement
    // construct a fresh std::thread over that local, so no destructor ever sees the stale
    // id and calls std::terminate().

    /// Install a test thread as the WFS shmimMonitor thread.
    void setWfsMonitorThread( std::thread &&t )
    {
        shmimMonitorT::m_smThread = std::move( t );
    }

    /// Drop the WFS shmimMonitor thread object without joining it.
    void abandonWfsMonitorThread()
    {
        std::thread tmp;
        tmp.swap( shmimMonitorT::m_smThread );
        new( &tmp ) std::thread();
    }

    /// Join the WFS shmimMonitor thread if it is joinable.
    void joinWfsMonitorThread()
    {
        if( shmimMonitorT::m_smThread.joinable() )
            shmimMonitorT::m_smThread.join();
    }

    /// Install a test thread as the dark shmimMonitor thread.
    void setDarkMonitorThread( std::thread &&t )
    {
        darkShmimMonitorT::m_smThread = std::move( t );
    }

    /// Drop the dark shmimMonitor thread object without joining it.
    void abandonDarkMonitorThread()
    {
        std::thread tmp;
        tmp.swap( darkShmimMonitorT::m_smThread );
        new( &tmp ) std::thread();
    }

    /// Join the dark shmimMonitor thread if it is joinable.
    void joinDarkMonitorThread()
    {
        if( darkShmimMonitorT::m_smThread.joinable() )
            darkShmimMonitorT::m_smThread.join();
    }

    /// Drop the real wfs measurement thread object after appLogic() has joined it, so
    /// appShutdown() does not signal and join the stale id again.
    void abandonWfsMeasurementThread()
    {
        std::thread tmp;
        tmp.swap( m_wfsThread );
        new( &tmp ) std::thread();
    }
};

#ifdef XWCTEST_NAMESPACE
} // namespace XWCTEST_NAMESPACE
#endif

} // namespace dmPokeWFS_tests

// LCOV_EXCL_STOP
