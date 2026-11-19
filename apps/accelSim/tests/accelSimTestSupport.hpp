/** \file accelSimTestSupport.hpp
 * \brief Reuse stream-free application doubles to exercise the real accelSim header and producer.
 * \ingroup accelSim_unit_test
 */
#ifndef accelSimTestSupport_hpp
#define accelSimTestSupport_hpp

#include <chrono>
#include <cstdint>
#include <ctime>
#include <thread>
#include "../../loPredCtrlAcc/tests/app_test_support.hpp"

#undef MAGAOX_CURRENT_SHA1
#undef MAGAOX_REPO_MODIFIED
#ifndef SEMAPHORE_MAXVAL
    #define SEMAPHORE_MAXVAL 10
#endif

/// \cond ACCELSIM_APP_DOUBLE
namespace MagAOX::app
{
namespace logPrio
{
constexpr int LOG_ERROR = LOG_WARNING; ///< No-op logger priority in the shared fixture.
}

/// Add accelSim's framework interface to the existing stream-free application fixture.
template <bool useIndi>
class AccelSimTestApp : public MagAOXApp<useIndi>
{
  public:
    std::string m_configName, m_shmimName; ///< Local identities; no native application is started.
    bool        m_reconfig{ false };       ///< Acquisition reconfiguration request without a framework worker.

    /// Construct only the shared in-memory fixture.
    AccelSimTestApp( const char *version /**< [in] ignored version */,
                     int         modified /**< [in] ignored repository status */ );

    /// Preserve the production virtual lifecycle contract.
    virtual ~AccelSimTestApp() noexcept = default;

    /// Accept production configuration setup.
    virtual void setupConfig();

    /// Accept production configuration loading.
    virtual void loadConfig();

    /// Accept production startup without invoking it in the fixture.
    virtual int appStartup();

    /// Accept production app-loop processing.
    virtual int appLogic();

    /// Accept production shutdown processing.
    virtual int appShutdown();

    /// Observe the shared fixture shutdown request.
    bool shutdown() const;

    /// Start a worker like MagAOXApp::threadStart, without privileges, priority, cpuset or INDI registration.
    template <class thisPtr, class Function>
    int threadStart( std::thread       &thrd /**< [out] started worker */,
                     bool              &thrdInit /**< [out] cleared once setup is complete */,
                     pid_t             &tpid /**< [out] reset; the worker fills it */,
                     pcf::IndiProperty &thProp /**< [in] unused thread property */,
                     int                thrdPrio /**< [in] unused priority */,
                     const std::string &cpuset /**< [in] unused cpuset */,
                     const std::string &thrdName /**< [in] unused name */,
                     thisPtr           *thrdThis /**< [in] worker argument */,
                     Function         &&thrdStart /**< [in] static worker entry point */ );

    /// Register real INDI current/target elements in local memory.
    template <typename T, typename... Args>
    int createStandardIndiNumber( pcf::IndiProperty &property /**< [out] local property */,
                                  const std::string &name /**< [in] property name */,
                                  Args... /**< [in] unused display metadata */ );

    using MagAOXApp<useIndi>::createROIndiNumber;
    using MagAOXApp<useIndi>::updateIfChanged;

    /// Accept the standard diagnostic label and group arguments.
    int createROIndiNumber( pcf::IndiProperty &property /**< [out] local property */,
                            const std::string &name /**< [in] property name */,
                            const std::string & /**< [in] unused label */,
                            const std::string & /**< [in] unused group */ );

    /// Use the actual INDI scalar-update utility with an explicit state.
    template <typename T>
    void updateIfChanged( pcf::IndiProperty                   &property /**< [in,out] local property */,
                          const std::string                   &element /**< [in] element name */,
                          const T                             &value /**< [in] new value */,
                          pcf::IndiProperty::PropertyStateType state /**< [in] new property state */ );
};

template <bool useIndi>
AccelSimTestApp<useIndi>::AccelSimTestApp( const char *version, int modified ) : MagAOXApp<useIndi>( version, modified )
{
}

template <bool useIndi>
void AccelSimTestApp<useIndi>::setupConfig()
{
}

template <bool useIndi>
void AccelSimTestApp<useIndi>::loadConfig()
{
}

template <bool useIndi>
int AccelSimTestApp<useIndi>::appStartup()
{
    return 0;
}

template <bool useIndi>
int AccelSimTestApp<useIndi>::appLogic()
{
    return 0;
}

template <bool useIndi>
int AccelSimTestApp<useIndi>::appShutdown()
{
    return 0;
}

template <bool useIndi>
bool AccelSimTestApp<useIndi>::shutdown() const
{
    return this->m_shutdown;
}

template <bool useIndi>
template <class thisPtr, class Function>
int AccelSimTestApp<useIndi>::threadStart( std::thread &thrd,
                                           bool        &thrdInit,
                                           pid_t       &tpid,
                                           pcf::IndiProperty &,
                                           int,
                                           const std::string &,
                                           const std::string &,
                                           thisPtr   *thrdThis,
                                           Function &&thrdStart )
{
    thrdInit = true;
    tpid     = 0;
    thrd     = std::thread( thrdStart, thrdThis );
    thrdInit = false;
    return 0;
}

template <bool useIndi>
template <typename T, typename... Args>
int AccelSimTestApp<useIndi>::createStandardIndiNumber( pcf::IndiProperty &property, const std::string &name, Args... )
{
    this->createProperty( property, name, pcf::IndiProperty::Number, false );
    property.add( pcf::IndiElement( "current", 0.0 ) );
    property.add( pcf::IndiElement( "target", 0.0 ) );
    return 0;
}

template <bool useIndi>
int AccelSimTestApp<useIndi>::createROIndiNumber( pcf::IndiProperty &property,
                                                  const std::string &name,
                                                  const std::string &,
                                                  const std::string & )
{
    return MagAOXApp<useIndi>::createROIndiNumber( property, name );
}

template <bool useIndi>
template <typename T>
void AccelSimTestApp<useIndi>::updateIfChanged( pcf::IndiProperty                   &property,
                                                const std::string                   &element,
                                                const T                             &value,
                                                pcf::IndiProperty::PropertyStateType state )
{
    indi::updateIfChanged( property, element, value, this->m_indiDriver, state );
}
} // namespace MagAOX::app

/// Read a deterministic same-host clock for the real producer only.
int dmOutputClock( clockid_t clock /**< [in] clock identity */, timespec *value /**< [out] virtual monotonic time */ );

/// Advance the real producer's absolute wait without a runtime timer or device.
int dmOutputSleep( clockid_t       clock /**< [in] ignored clock */,
                   int             flags /**< [in] ignored flags */,
                   const timespec *value /**< [in] absolute virtual deadline */,
                   timespec       *remaining /**< [out] unused remainder */ );

#define clock_gettime dmOutputClock
#define clock_nanosleep dmOutputSleep
#include "../../../libMagAOX/app/dev/dmSyncopationProducer.hpp"
#undef clock_gettime
#undef clock_nanosleep

// Only the framework is doubled; the included accelSim and producer implementations are unchanged.
#define MagAOXApp AccelSimTestApp
/// \endcond
#endif
