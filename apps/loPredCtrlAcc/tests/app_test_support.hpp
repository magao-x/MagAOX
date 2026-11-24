/** \file app_test_support.hpp
 * \brief Stream-free lifecycle doubles for testing the real loPredCtrlAcc header and INDI callbacks.
 */
#ifndef loPredCtrlAcc_app_test_support_hpp
#define loPredCtrlAcc_app_test_support_hpp

#include <atomic>
#include <cstdlib>
#include <map>
#include <sstream>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <semaphore.h>
#include <ImageStreamIO/ImageStreamIO.h>

#include "../../../libMagAOX/app/indiMacros.hpp"
#include "../../../libMagAOX/app/indiUtils.hpp"
#include "../../../libMagAOX/app/stateCodes.hpp"

// This translation unit deliberately cannot instantiate native threads, ImageStreamIO, or INDI drivers.
#define libMagAOX_hpp
#define MAGAOX_CURRENT_SHA1 "offline-test"
#define MAGAOX_REPO_MODIFIED 0
#define app_appConfigurator_hpp // The unchanged producer helper uses the explicit fixture configurator below.
#define _DATATYPE_FLOAT 9
#define _DATATYPE_UINT16 3 // ImageStruct.h value; the fixture does not link ImageStreamIO.

/// \cond TEST_SUPPORT
namespace mx::app
{
enum class argType
{
    Required,
    Optional
};
class appConfigurator
{
  public:
    /// Configured scalar strings used only by the offline fixture.
    std::map<std::string, std::string> m_values;

    /// Accept a configuration declaration without registering external resources.
    template <typename... T>
    void add( T... /**< [in] Unused configuration metadata. */ );

    /// Read a fixture scalar, leaving absent optional values unchanged.
    template <typename T>
    void operator()( T &value /**< [out] Destination. */, const std::string &key /**< [in] Key. */ );
};

template <typename... T>
inline void appConfigurator::add( T... /**< [in] Unused configuration metadata. */ )
{
}

template <typename T>
inline void appConfigurator::operator()( T &value /**< [out] Destination. */, const std::string &key /**< [in] Key. */ )
{
    if( m_values.count( key ) )
    {
        std::istringstream input( m_values.at( key ) );
        input >> value;
    }
}
} // namespace mx::app

namespace MagAOX::app
{
using mx::app::argType;
struct software_error
{
};
struct software_critical
{
};
struct text_log
{
};
struct telem_fgtimings
{
};
namespace logPrio
{
enum
{
    LOG_NOTICE,
    LOG_WARNING
};
} // namespace logPrio
struct TestLog
{
    std::string m_text; ///< Text of a single-text entry, as text_log; empty for other payloads.

    /// Accept existing logger payload syntax without I/O, keeping the text of a single-text entry.
    template <typename... T>
    TestLog( T... payload /**< [in] Logger payload. */ );
};

template <typename... T>
inline TestLog::TestLog( T... payload )
{
    if constexpr( sizeof...( T ) == 1 && ( std::is_convertible_v<T, std::string> && ... ) )
    {
        m_text = ( std::string( payload ), ... );
    }
    else
    {
        ( static_cast<void>( payload ), ... );
    }
}
struct TestDriver
{
    /// Captured property publications, never sent over a connection.
    std::vector<pcf::IndiProperty> m_sent;

    /// Capture a real INDI property update.
    void sendSetProperty( const pcf::IndiProperty &property /**< [in] Snapshot. */ );
};

inline void TestDriver::sendSetProperty( const pcf::IndiProperty &property /**< [in] Snapshot. */ )
{
    m_sent.push_back( property );
}

template <bool useIndi>
class MagAOXApp
{
  public:
    /// In-memory config used by loadConfigImpl.
    mx::app::appConfigurator config;

    /// Matches the real application's INDI lock boundary.
    std::mutex m_indiMutex;

    /// Captures diagnostic updates without a native driver.
    TestDriver m_driver;

    /// Actual app property registrations, inspected without connecting a native INDI driver.
    std::map<std::string, pcf::IndiProperty *> m_registered;

    /// Driver pointer expected by real INDI utility templates.
    TestDriver *m_indiDriver{ &m_driver };

    /// Shutdown request from the real app's legacy callback.
    bool m_shutdown{ false };

    /// Construct without system initialization.
    MagAOXApp( const char * /**< [in] Version. */, int /**< [in] Modified flag. */ );

    /// Text log entries and their priorities, in order; nothing is written.
    std::vector<std::pair<std::string, int>> m_textLogs;

    /// Record text entries, preserving error returns.
    template <typename T, int result = 0>
    int log( TestLog payload /**< [in] Payload. */, int priority = 0 /**< [in] Priority. */ );

    /// Ignore application state publication.
    void state( int /**< [in] State. */ );

    /// Initialize property identity with the same public schema as MagAOXApp.
    int createProperty( pcf::IndiProperty      &p /**< [out] Property. */,
                        const std::string      &name /**< [in] Name. */,
                        pcf::IndiProperty::Type type /**< [in] Type. */,
                        bool                    readOnly /**< [in] Permission. */ );

    /// Create a number diagnostic.
    int createROIndiNumber( pcf::IndiProperty &p /**< [out] Property. */, const std::string &n /**< [in] Name. */ );

    /// Create a text diagnostic.
    int createROIndiText( pcf::IndiProperty &p /**< [out] Property. */,
                          const std::string &n /**< [in] Name. */,
                          const std::string &element /**< [in] Initial element. */ );

    /// Create the legacy current/target text schema.
    int createStandardIndiText( pcf::IndiProperty &p /**< [out] Property. */,
                                const std::string &n /**< [in] Name. */,
                                const std::string & /**< [in] Label. */,
                                const std::string & /**< [in] Group. */ );

    /// Create the real current/target numeric schema without a server.
    template <typename T>
    int createStandardIndiNumber( pcf::IndiProperty &p /**< [out] Property. */,
                                  const std::string &name /**< [in] Property name. */,
                                  const T           &minimum /**< [in] Unused display minimum. */,
                                  const T           &maximum /**< [in] Unused display maximum. */,
                                  const T           &step /**< [in] Unused display step. */,
                                  const std::string &format /**< [in] Unused display format. */ );

    /// Create a one-of-many selector, with all elements initially off.
    int createStandardIndiSelectionSw( pcf::IndiProperty              &p /**< [out] Property. */,
                                       const std::string              &n /**< [in] Name. */,
                                       const std::vector<std::string> &elements /**< [in] Elements. */,
                                       const std::string & /**< [in] Label. */,
                                       const std::string & /**< [in] Group. */ );

    /// Create a toggle without connecting to a server.
    int createStandardIndiToggleSw( pcf::IndiProperty &p /**< [out] Property. */,
                                    const std::string &n /**< [in] Name. */,
                                    const std::string & = "" /**< [in] Label. */,
                                    const std::string & = "" /**< [in] Group. */ );

    /// Create a request without connecting to a server.
    int createStandardIndiRequestSw( pcf::IndiProperty &p /**< [out] Property. */,
                                     const std::string &n /**< [in] Name. */,
                                     const std::string & /**< [in] Label. */,
                                     const std::string & /**< [in] Group. */ );

    /// Accept callback registration without a server.
    template <typename T>
    int registerIndiPropertyNew( pcf::IndiProperty & /**< [in] Property. */, T /**< [in] Callback. */ );

    /// Accept read-only registration without a server.
    int registerIndiPropertyReadOnly( pcf::IndiProperty & /**< [in] Property. */ );

    /// Assign a subscription identity without subscribing.
    template <typename T>
    int registerIndiPropertySet( pcf::IndiProperty &p /**< [out] Property. */,
                                 const std::string &device /**< [in] Device. */,
                                 const std::string &name /**< [in] Name. */,
                                 T /**< [in] Callback. */ );

    /// Exercise the real batch update utility with a capture-only driver.
    template <typename T>
    void updatesIfChanged( pcf::IndiProperty                   &p /**< [in,out] Property. */,
                           const std::vector<const char *>     &e /**< [in] Elements. */,
                           const std::vector<T>                &v /**< [in] Values. */,
                           pcf::IndiProperty::PropertyStateType state = INDI_OK /**< [in] Diagnostic state. */ );

    /// Exercise the real scalar update utility.
    template <typename T>
    void updateIfChanged( pcf::IndiProperty &p /**< [in,out] Property. */,
                          const std::string &e /**< [in] Element. */,
                          const T           &v /**< [in] Value. */ );

    /// Exercise the real switch update utility.
    void updateSwitchIfChanged( pcf::IndiProperty                   &p /**< [in,out] Property. */,
                                const std::string                   &e /**< [in] Element. */,
                                pcf::IndiElement::SwitchStateType    v /**< [in] Value. */,
                                pcf::IndiProperty::PropertyStateType state /**< [in] Diagnostic state. */ );

    /// Parse a legacy text request without any external side effects.
    int indiTargetUpdate( pcf::IndiProperty & /**< [in,out] Property. */,
                          std::string             &value /**< [out] Target. */,
                          const pcf::IndiProperty &request /**< [in] Request. */,
                          bool /**< [in] Target required. */ );

    /// Start a worker like MagAOXApp::threadStart, without privileges, priority, cpuset or INDI registration.
    template <class thisPtr, class Function>
    int threadStart( std::thread       &thrd /**< [out] Started worker. */,
                     bool              &thrdInit /**< [out] Cleared once setup is complete. */,
                     pid_t             &tpid /**< [out] Reset; the worker fills it. */,
                     pcf::IndiProperty &thProp /**< [in] Unused thread property. */,
                     int                thrdPrio /**< [in] Unused priority. */,
                     const std::string &cpuset /**< [in] Unused cpuset. */,
                     const std::string &thrdName /**< [in] Unused name. */,
                     thisPtr           *thrdThis /**< [in] Worker argument. */,
                     Function         &&thrdStart /**< [in] Static worker entry point. */ );
};

template <bool useIndi>
template <class thisPtr, class Function>
inline int MagAOXApp<useIndi>::threadStart( std::thread &thrd,
                                            bool        &thrdInit,
                                            pid_t       &tpid,
                                            pcf::IndiProperty &,
                                            int,
                                            const std::string &,
                                            const std::string &,
                                            thisPtr   *thrdThis,
                                            Function &&thrdStart )
{
    // Nothing to configure, so setup is complete before the worker starts and reads thrdInit.
    tpid     = 0;
    thrdInit = false;
    thrd     = std::thread( thrdStart, thrdThis );
    return 0;
}

template <bool useIndi>
inline MagAOXApp<useIndi>::MagAOXApp( const char * /**< [in] Version. */, int /**< [in] Modified flag. */ )
{
}

template <bool useIndi>
template <typename T, int result>
inline int MagAOXApp<useIndi>::log( TestLog payload, int priority )
{
    if( !payload.m_text.empty() )
    {
        m_textLogs.emplace_back( payload.m_text, priority );
    }

    return result;
}

template <bool useIndi>
inline void MagAOXApp<useIndi>::state( int /**< [in] State. */ )
{
}

template <bool useIndi>
inline int MagAOXApp<useIndi>::createProperty( pcf::IndiProperty      &p /**< [out] Property. */,
                                               const std::string      &name /**< [in] Name. */,
                                               pcf::IndiProperty::Type type /**< [in] Type. */,
                                               bool                    readOnly /**< [in] Permission. */ )
{
    p = pcf::IndiProperty( type );
    p.setName( name );
    p.setDevice( "loPredCtrlAcc-test" );
    p.setPerm( readOnly ? pcf::IndiProperty::ReadOnly : pcf::IndiProperty::ReadWrite );
    p.setState( INDI_IDLE );
    return 0;
}

template <bool useIndi>
inline int MagAOXApp<useIndi>::createROIndiNumber( pcf::IndiProperty &p /**< [out] Property. */,
                                                   const std::string &n /**< [in] Name. */ )
{
    return createProperty( p, n, pcf::IndiProperty::Number, true );
}

template <bool useIndi>
inline int MagAOXApp<useIndi>::createROIndiText( pcf::IndiProperty &p /**< [out] Property. */,
                                                 const std::string &n /**< [in] Name. */,
                                                 const std::string &element /**< [in] Initial element. */ )
{
    createProperty( p, n, pcf::IndiProperty::Text, true );
    p.add( pcf::IndiElement( element, "" ) );
    return 0;
}

template <bool useIndi>
template <typename T>
inline int MagAOXApp<useIndi>::createStandardIndiNumber(
    pcf::IndiProperty &p, const std::string &name, const T &, const T &, const T &, const std::string & )
{
    createProperty( p, name, pcf::IndiProperty::Number, false );
    p.add( pcf::IndiElement( "current", 0.0 ) );
    p.add( pcf::IndiElement( "target", 0.0 ) );
    return 0;
}

template <bool useIndi>
inline int MagAOXApp<useIndi>::createStandardIndiText( pcf::IndiProperty &p /**< [out] Property. */,
                                                       const std::string &n /**< [in] Name. */,
                                                       const std::string & /**< [in] Label. */,
                                                       const std::string & /**< [in] Group. */ )
{
    createProperty( p, n, pcf::IndiProperty::Text, false );
    p.add( pcf::IndiElement( "current", "" ) );
    p.add( pcf::IndiElement( "target", "" ) );
    return 0;
}

template <bool useIndi>
inline int
MagAOXApp<useIndi>::createStandardIndiSelectionSw( pcf::IndiProperty              &p /**< [out] Property. */,
                                                   const std::string              &n /**< [in] Name. */,
                                                   const std::vector<std::string> &elements /**< [in] Elements. */,
                                                   const std::string & /**< [in] Label. */,
                                                   const std::string & /**< [in] Group. */ )
{
    createProperty( p, n, pcf::IndiProperty::Switch, false );
    p.setRule( pcf::IndiProperty::OneOfMany );
    for( auto &e : elements )
        p.add( pcf::IndiElement( e, pcf::IndiElement::Off ) );
    return 0;
}

template <bool useIndi>
inline int MagAOXApp<useIndi>::createStandardIndiToggleSw( pcf::IndiProperty &p /**< [out] Property. */,
                                                           const std::string &n /**< [in] Name. */,
                                                           const std::string & /**< [in] Label. */,
                                                           const std::string & /**< [in] Group. */ )
{
    createProperty( p, n, pcf::IndiProperty::Switch, false );
    p.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
    return 0;
}

template <bool useIndi>
inline int MagAOXApp<useIndi>::createStandardIndiRequestSw( pcf::IndiProperty &p /**< [out] Property. */,
                                                            const std::string &n /**< [in] Name. */,
                                                            const std::string & /**< [in] Label. */,
                                                            const std::string & /**< [in] Group. */ )
{
    createProperty( p, n, pcf::IndiProperty::Switch, false );
    p.add( pcf::IndiElement( "request", pcf::IndiElement::Off ) );
    return 0;
}

template <bool useIndi>
template <typename T>
inline int MagAOXApp<useIndi>::registerIndiPropertyNew( pcf::IndiProperty &p /**< [in] Property. */,
                                                        T /**< [in] Callback. */ )
{
    m_registered[p.getName()] = &p;
    return 0;
}

template <bool useIndi>
inline int MagAOXApp<useIndi>::registerIndiPropertyReadOnly( pcf::IndiProperty &p /**< [in] Property. */ )
{
    m_registered[p.getName()] = &p;
    return 0;
}

template <bool useIndi>
template <typename T>
inline int MagAOXApp<useIndi>::registerIndiPropertySet( pcf::IndiProperty &p /**< [out] Property. */,
                                                        const std::string &device /**< [in] Device. */,
                                                        const std::string &name /**< [in] Name. */,
                                                        T /**< [in] Callback. */ )
{
    p.setDevice( device );
    p.setName( name );
    return 0;
}

template <bool useIndi>
template <typename T>
inline void MagAOXApp<useIndi>::updatesIfChanged( pcf::IndiProperty                   &p /**< [in,out] Property. */,
                                                  const std::vector<const char *>     &e /**< [in] Elements. */,
                                                  const std::vector<T>                &v /**< [in] Values. */,
                                                  pcf::IndiProperty::PropertyStateType state )
{
    indi::updatesIfChanged( p, e, v, m_indiDriver, state );
}

template <bool useIndi>
template <typename T>
inline void MagAOXApp<useIndi>::updateIfChanged( pcf::IndiProperty &p /**< [in,out] Property. */,
                                                 const std::string &e /**< [in] Element. */,
                                                 const T           &v /**< [in] Value. */ )
{
    indi::updateIfChanged( p, e, v, m_indiDriver );
}

template <bool useIndi>
inline void MagAOXApp<useIndi>::updateSwitchIfChanged( pcf::IndiProperty                &p /**< [in,out] Property. */,
                                                       const std::string                &e /**< [in] Element. */,
                                                       pcf::IndiElement::SwitchStateType v /**< [in] Value. */,
                                                       pcf::IndiProperty::PropertyStateType state )
{
    indi::updateSwitchIfChanged( p, e, v, m_indiDriver, state );
}

template <bool useIndi>
inline int MagAOXApp<useIndi>::indiTargetUpdate( pcf::IndiProperty & /**< [in,out] Property. */,
                                                 std::string             &value /**< [out] Target. */,
                                                 const pcf::IndiProperty &request /**< [in] Request. */,
                                                 bool /**< [in] Target required. */ )
{
    if( !request.find( "target" ) )
        return -1;
    value = request["target"].get<std::string>();
    return 0;
}

namespace dev
{
struct shmimT
{
};
struct TestLifecycle
{
    /// Counts lifecycle calls instead of starting resources.
    int m_loads{ 0 }, m_starts{ 0 }, m_logic{ 0 }, m_stops{ 0 }, m_updates{ 0 };

    /// Configuration declarations are side-effect free.
    void setupConfig( mx::app::appConfigurator & /**< [in] Configuration. */ );

    /// Count configuration loads.
    void loadConfig( mx::app::appConfigurator & /**< [in] Configuration. */ );

    /// Count starts; never create a thread or stream.
    int appStartup();

    /// Count event-loop calls.
    int appLogic();

    /// Count stops.
    int appShutdown();

    /// Count diagnostic updates.
    int updateINDI();
};

inline void TestLifecycle::setupConfig( mx::app::appConfigurator & /**< [in] Configuration. */ )
{
}

inline void TestLifecycle::loadConfig( mx::app::appConfigurator & /**< [in] Configuration. */ )
{
    ++m_loads;
}

inline int TestLifecycle::appStartup()
{
    ++m_starts;
    return 0;
}

inline int TestLifecycle::appLogic()
{
    ++m_logic;
    return 0;
}

inline int TestLifecycle::appShutdown()
{
    ++m_stops;
    return 0;
}

inline int TestLifecycle::updateINDI()
{
    ++m_updates;
    return 0;
}
template <class D, class Tag = shmimT>
class shmimMonitor : public TestLifecycle
{
  public:
    /// Fixture stream shape and type; no ImageStreamIO object exists.
    uint32_t m_width{ 1 }, m_height{ 1 }, m_dataType{ _DATATYPE_FLOAT };

    /// Optional process-private metadata for WFS callback tests; never attached to shared memory.
    IMAGE m_imageStream{};

    /// Fixture name and reader preference.
    std::string m_shmimName;
    bool        m_getExistingFirst{ false };
};
template <class D>
class frameGrabber : public TestLifecycle
{
  public:
    /// Fixture output shape; no output publisher exists.
    uint32_t m_width{ 0 }, m_height{ 0 }, m_dataType{ 0 }, m_typeSize{ sizeof( float ) };

    /// Fixture ownership setting, never used to attach streams.
    bool m_ownShmim{ true };

    /// Timestamp storage expected by the legacy callback.
    timespec m_currImageTimestamp{};

    /// Output stream name, as configured for the real publisher.
    std::string m_shmimName;

    /// Reconfiguration request, observed by tests instead of a publisher thread.
    std::atomic<bool> m_reconfig{ false };

    /// Attached output, a malloc'd copy of m_existing made by openShmim(); null when none.
    IMAGE *m_imageStream{ nullptr };

    /// Fixture stream returned by openShmim(), standing in for an existing shared-memory image; null when absent.
    IMAGE *m_existing{ nullptr };

    /// Ignore timing telemetry.
    int recordFGTimings( bool /**< [in] Force record. */ );

    /// Attach m_existing as the real base attaches an existing stream: 0 on success, 1 when not ready.
    int openShmim();
};

template <class D>
inline int frameGrabber<D>::recordFGTimings( bool /**< [in] Force record. */ )
{
    return 0;
}

template <class D>
inline int frameGrabber<D>::openShmim()
{
    if( !m_existing || !m_existing->md )
        return 1;
    free( m_imageStream );
    m_imageStream  = static_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );
    *m_imageStream = *m_existing; // Shares the fixture metadata, as a mapping shares the stream.
    m_width        = m_existing->md->size[0];
    m_height       = m_existing->md->naxis > 1 ? m_existing->md->size[1] : 1;
    m_dataType     = m_existing->md->datatype;
    return 0;
}
template <class D>
class telemeter : public TestLifecycle
{
  public:
    /// No native telemetry scheduler is run.
    int checkRecordTimes( telem_fgtimings /**< [in] Record type. */ );
};

template <class D>
inline int telemeter<D>::checkRecordTimes( telem_fgtimings /**< [in] Record type. */ )
{
    return 0;
}
} // namespace dev
} // namespace MagAOX::app

// Mirror helper dispatch while replacing only the hardware-owning bases above.
#define SHMIMMONITOR_SETUP_CONFIG( c ) shmimMonitorT::setupConfig( c )
#define SHMIMMONITOR_LOAD_CONFIG( c ) shmimMonitorT::loadConfig( c )
#define SHMIMMONITOR_APP_STARTUP shmimMonitorT::appStartup()
#define SHMIMMONITOR_APP_LOGIC shmimMonitorT::appLogic()
#define SHMIMMONITOR_APP_SHUTDOWN shmimMonitorT::appShutdown()
#define SHMIMMONITOR_UPDATE_INDI shmimMonitorT::updateINDI()
#define SHMIMMONITORT_SETUP_CONFIG( T, c ) T::setupConfig( c )
#define SHMIMMONITORT_LOAD_CONFIG( T, c ) T::loadConfig( c )
#define SHMIMMONITORT_APP_STARTUP( T ) T::appStartup()
#define SHMIMMONITORT_APP_LOGIC( T ) T::appLogic()
#define SHMIMMONITORT_APP_SHUTDOWN( T ) T::appShutdown()
#define SHMIMMONITORT_UPDATE_INDI( T ) T::updateINDI()
#define FRAMEGRABBER_SETUP_CONFIG( c ) frameGrabberT::setupConfig( c )
#define FRAMEGRABBER_LOAD_CONFIG( c ) frameGrabberT::loadConfig( c )
#define FRAMEGRABBER_APP_STARTUP frameGrabberT::appStartup()
#define FRAMEGRABBER_APP_LOGIC frameGrabberT::appLogic()
#define FRAMEGRABBER_APP_SHUTDOWN frameGrabberT::appShutdown()
#define FRAMEGRABBER_UPDATE_INDI frameGrabberT::updateINDI()
#define TELEMETER_SETUP_CONFIG( c ) telemeterT::setupConfig( c )
#define TELEMETER_LOAD_CONFIG( c ) telemeterT::loadConfig( c )
#define TELEMETER_APP_STARTUP telemeterT::appStartup()
#define TELEMETER_APP_LOGIC telemeterT::appLogic()
#define TELEMETER_APP_SHUTDOWN telemeterT::appShutdown()

/// Count semaphore entrypoints so shadow tests verify no output signaling occurs.
struct DdspcTestSemaphore
{
    /// Number of attempted output semaphore operations.
    inline static int m_calls{ 0 };
};
/// Replace semaphore initialization without allocating an OS object.
inline int ddspc_test_sem_init( sem_t * /**< [in] Ignored semaphore. */,
                                int /**< [in] Scope. */,
                                unsigned int /**< [in] Value. */ )
{
    ++DdspcTestSemaphore::m_calls;
    return 0;
}
/// Replace the unreachable legacy wait without blocking.
inline int ddspc_test_sem_timedwait( sem_t * /**< [in] Ignored semaphore. */, const timespec * /**< [in] Timeout. */ )
{
    ++DdspcTestSemaphore::m_calls;
    return -1;
}
/// Capture any accidental output signal without posting an OS semaphore.
inline int ddspc_test_sem_post( sem_t * /**< [in] Ignored semaphore. */ )
{
    ++DdspcTestSemaphore::m_calls;
    return 0;
}
#define sem_init ddspc_test_sem_init
#define sem_timedwait ddspc_test_sem_timedwait
#define sem_post ddspc_test_sem_post
/// \endcond
#endif
