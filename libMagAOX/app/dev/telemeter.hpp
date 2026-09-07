/** \file telemeter.hpp
 * \author Jared R. Males
 * \brief Configuration and control of a telemetry logger
 *
 * \ingroup app_files
 *
 */

#ifndef app_telemeter_hpp
#define app_telemeter_hpp

namespace MagAOX
{
namespace app
{
namespace dev
{

#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE
{
#endif


/// A device base class which saves telemetry.
/**
  * CRTP class `derivedT` has the following requirements:
  * - Must be a MagAOXApp
  * - Must include the following friend declaration:
  *   \code
  *       friend class dev::telemeter<DERIVEDNAME>; //replace DERIVEDNAME with derivedT class name
  *   \endcode
  * - Must include the following typedef:
  *   \code
  *       typedef dev::telemeter<DERIVEDNAME> telemeterT; //replace DERIVEDNAME with derivedT class name
  *   \endcode
  * - Must implement the following interface:
  *   \code
  *       int checkRecordTimes()
  *       {
  *            // Must call this variadic template function with each relevant telemetry type exactly like this
  *            return telemeterT::checkRecordTimes( telem_type1(), telem_type2(), ..., telem_typeN());
  *       }
  *   \endcode
  *   where there is one constructor-call argument for each telemetry log type recorded by this device.  The resultant
  *   objects are not used, rather the types are just used for variadic template resolution.
  *
  * - Must provide one overload of the following function for each telemetry type:
  *   \code
  *       int recordTelem( const telem_type1 * )
  *       {
  *          //DO NOT USE telem_type1
  *          return m_tel<telem_type1>( { message entered here } );
  *       }
  *   \endcode
  *   You MUST NOT use the pointer argument, it is for type resolution only -- you
  *   should fill in the telemetry log message using internal values. Note that calls to this function should result
  *   in a telemetry log entry every time -- it is called when the minimum interval has elapsed since the last entry.
  *
  * - Must call this class's setupConfig(), loadConfig(), appStartup(), appLogic(), and appShutdown()
  *   in the corresponding function of `derivedT`, with error checking.
  *   For convenience the following macros are defined to provide error checking:
  *   \code
  *       TELEMETER_SETUP_CONFIG( cfig )
  *       TELEMETER_LOAD_CONFIG( cfig )
  *       TELEMETER_APP_STARTUP
  *       TELEMETER_APP_LOGIC
  *       TELEMETER_APP_SHUTDOWN
  *   \endcode
  *
  * \ingroup appdev
  */
template <class derivedT>
struct telemeter
{
    typedef XWC_DEFAULT_VERBOSITY verboseT;

    /// The log manager type.
    typedef logger::logManager<derivedT, logFileRaw<verboseT>> logManagerT;

    logManagerT m_tel;

    double m_maxInterval{10.0}; ///< The maximum interval, in seconds, between telemetry records. Default is 10.0 seconds.

    pcf::IndiProperty m_indiP_rotateTelem; ///< indi Property to request rotation of the telemetry file.

    pcf::IndiProperty m_indiP_maxLogTime; ///< indi Property to report and set the telemetry file time interval, in minutes.

    telemeter();

    /// Destructor
    /** Explicitly noexcept. The INDI property members above have destructors which are not noexcept,
      * which would otherwise make the implicit destructor of any class deriving from both this and
      * MagAOXApp looser than MagAOXApp's virtual noexcept destructor. MagAOXApp declares its own
      * destructor noexcept for the same reason.
      */
    ~telemeter() noexcept;

    /// The static callback function to be registered for requesting telemetry file rotation
    /** The `void *` is a pointer to the app, which is converted to this telemeter.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_rotateTelem(void *app,                     /**< [in] a pointer to the app, will be
                                                                                   converted to telemeter. */
                                          const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with
                                                                                    the new property request. */
    );

    /// The callback called by the static version, to actually process the telemetry rotation request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_rotateTelem(const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the new
                                                                          property request. */
    );

    /// The static callback function to be registered for setting the telemetry file time interval
    /** The `void *` is a pointer to the app, which is converted to this telemeter.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_maxLogTime(void *app,                      /**< [in] a pointer to the app, will be
                                                                                   converted to telemeter. */
                                         const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with
                                                                                   the new property request. */
    );

    /// The callback called by the static version, to actually process the telemetry interval change.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_maxLogTime(const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the new
                                                                         property request. */
    );

    /// Make a telemetry recording
    /** Wrapper for logManager::log, which updates telT::lastRecord.
     *
     * \tparam logT the log entry type
     * \tparam retval the value returned by this method.
     *
     */
    template <typename telT>
    int telem(const typename telT::messageT &msg /**< [in] the data to log */);

    // Make a telemetry recording, for an empty record
    /* Wrapper for logManager::log, which updates telT::lastRecord.
     *
     * \tparam logT the log entry type
     * \tparam retval the value returned by this method.
     *
     */
    //template <typename telT>
    //int telem(); I think this shouldn't be defined, because empty telem makes no sense.  Delete after 11/27/2025

    /// Setup an application configurator for the device section
    /**
     * \returns 0 on success.
     * \returns -1 on error (nothing implemented yet)
     */
    int setupConfig(appConfigurator &config /**< [in] an application configuration to setup */);

    /// Load the device section from an application configurator
    /**
     *
     * \returns 0 on success
     * \returns -1 on error (nothing implemented yet)
     */
    int loadConfig(appConfigurator &config /**< [in] an application configuration from which to load values */);

    /// Starts the telemetry log thread.
    /**
     * This should be called from `derivedT::appStartup`
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int appStartup();

    /// Perform `telemeter` application logic
    /** This calls `derivedT::checkRecordTimes()`, and should be called from `derivedT::appLogic`, but only
     * when the FSM is in states where telemetry logging makes sense.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int appLogic();

    /// Perform `telemeter` application shutdown
    /** This currently does nothing.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int appShutdown();

    /// Check the time of the last record for each telemetry type and make an entry if needed
    /** This must be called from `derivedT::checkRecordTimes()`, with one template parameter
     * for ach telemetry log type being  recorded.
     *
     * \returns 0 on succcess
     * \returns -1 on error
     */
    template <class telT, class... telTs>
    int checkRecordTimes(const telT &tel, ///< [in] [unused] object of the telemetry type to record
                         telTs... tels    ///< [in] [unused] objects of the additional telemetry types to record
    );

    /// Worker function to actually perform the record time checking logic
    /** Recursively calls itself until the variadic template list is exhausted.
     *
     * \returns 0 on succcess
     * \returns -1 on error
     */
    template <class telT, class... telTs>
    int checkRecordTimes(timespec &ts,    ///<[in] [unused] the timestamp that records are compared to
                         const telT &tel, ///< [in] [unused] objects of the telemetry type to record
                         telTs... tels    ///< [in] [unused] objects of the additional telemetry types to record
    );

    /// Empty function called at the end of the template list
    /**
     * \returns 0 on succcess
     * \returns -1 on error
     */
    int checkRecordTimes(timespec &ts /**<[in] [unused] the timestamp that records are compared to */);

private:
    /// Access the derived class.
    derivedT &derived()
    {
        return *static_cast<derivedT *>(this);
    }
};

template <class derivedT>
telemeter<derivedT>::telemeter()
{
}

template <class derivedT>
telemeter<derivedT>::~telemeter() noexcept
{
}

template <class derivedT>
template <typename telT>
int telemeter<derivedT>::telem(const typename telT::messageT &msg)
{

    m_tel.template log<telT>(msg, logPrio::LOG_TELEM);

    // Set timestamp
    clock_gettime(CLOCK_REALTIME, &telT::lastRecord);

    return 0;
}

/* I think this shouldn't be defined.  Delete after 11/27/2025
template <class derivedT>
template <typename telT>
int telemeter<derivedT>::telem()
{

    m_tel.template log<telT>(logPrio::LOG_TELEM);

    // Set timestamp
    clock_gettime(CLOCK_REALTIME, &telT::lastRecord);

    return 0;
}*/

template <class derivedT>
int telemeter<derivedT>::setupConfig(mx::app::appConfigurator &config)
{
    m_tel.m_configSection = "telemeter";

    m_tel.setupConfig(config);

    config.add("telemeter.maxInterval", "", "telemeter.maxInterval", argType::Required, "telemeter", "maxInterval", false, "double", "The maximum interval, in seconds, between telemetry records. Default is 10.0 seconds.");

    return 0;
}

template <class derivedT>
int telemeter<derivedT>::loadConfig(mx::app::appConfigurator &config)
{
    m_tel.m_logLevel = logPrio::LOG_TELEM;

    // Setup default log path
    std::string tmpstr = mx::sys::getEnv( MAGAOX_env_telem );
    if( tmpstr == "" )
    {
        tmpstr = MAGAOX_telRelPath;
    }
    m_tel.logPath(std::string(derived().basePath()) + "/" + tmpstr);

    m_tel.logExt("bintel");

    m_tel.logName(derived().m_configName);

    m_tel.loadConfig(config);

    config(m_maxInterval, "telemeter.maxInterval");

    return 0;
}

template <class derivedT>
int telemeter<derivedT>::appStartup()
{
    //----------------------------------------//
    //        Set up the INDI properties
    //----------------------------------------//

    // These are registered through the app, which this class is a friend of, so that apps without
    // telemetry do not get telemetry properties.
    derived().createStandardIndiRequestSw(m_indiP_rotateTelem, "telem_rotate", "New Telemetry File", "Logging");
    if (derived().registerIndiPropertyNew(m_indiP_rotateTelem, st_newCallBack_rotateTelem) < 0)
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, "failed to register new telem_rotate property"});
    }

    derived().template createStandardIndiNumber<unsigned>(
        m_indiP_maxLogTime, "telem_maxtime", 0, 525600, 1, "", "Max Telemetry Interval [minutes]", "Logging");
    if (derived().registerIndiPropertyNew(m_indiP_maxLogTime, st_newCallBack_maxLogTime) < 0)
    {
        derivedT::template log<software_error>({__FILE__, __LINE__, "failed to register new telem_maxtime property"});
    }

    // Set the INDI property to the configured interval.
    m_indiP_maxLogTime["current"] = m_tel.maxLogTime();
    m_indiP_maxLogTime["target"] = m_tel.maxLogTime();

    //----------------------------------------//
    //        Begin the telemetry system
    //----------------------------------------//

    m_tel.logThreadStart();

    // clang-format off
    #ifdef XWCTEST_TELEMETER_LOGSTART
    m_tel.logShutdown(true); // LCOV_EXCL_LINE
    sleep(2); // LCOV_EXCL_LINE
    #endif // clang-format on

    // Give up to 2 secs to make sure log thread has time to get started and try to open a file.
    int w = 0;
    while (m_tel.logThreadRunning() == false && w < 20)
    {
        // Sleep for 100 msec
        std::this_thread::sleep_for(std::chrono::duration<unsigned long, std::nano>(100000000));
        ++w;
    }

    if (m_tel.logThreadRunning() == false)
    {
        derivedT::template log<software_critical>({__FILE__, __LINE__, "telemetry thread not running.  exiting."});
        return -1;
    }

    return 0;
}

template <class derivedT>
int telemeter<derivedT>::appLogic()
{
    if( m_tel.logThreadRunning() == false )
    {
        derived().state( stateCodes::FAILURE );

        // Directly ouput the error b/c all other outputs are via the log thread
        std::cerr << "\nCRITICAL: telemetry thread not running.  Exiting.\n\n";

        derived().m_shutdown = 1;

        return -1;
    }

    return derived().checkRecordTimes();
}

template <class derivedT>
int telemeter<derivedT>::st_newCallBack_rotateTelem(void *app, const pcf::IndiProperty &ipRecv)
{
    // MagAOXApp::handleNewProperty always passes its own `this`, so the argument is the app, not
    // this telemeter.
    telemeter<derivedT> *tel = static_cast<derivedT *>(app);
    return tel->newCallBack_rotateTelem(ipRecv);
}

template <class derivedT>
int telemeter<derivedT>::newCallBack_rotateTelem(const pcf::IndiProperty &ipRecv)
{
    if (ipRecv.createUniqueKey() != m_indiP_rotateTelem.createUniqueKey())
    {
        return derivedT::template log<software_error, -1>({__FILE__, __LINE__, "wrong indi property received"});
    }

    if (ipRecv.find("request"))
    {
        if (ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
        {
            // This only sets a flag. The new file is created by the telemetry thread on the next record.
            m_tel.requestRotation();
            derived().updateSwitchIfChanged(m_indiP_rotateTelem, "request", pcf::IndiElement::Off, INDI_IDLE);
        }
    }

    return 0;
}

template <class derivedT>
int telemeter<derivedT>::st_newCallBack_maxLogTime(void *app, const pcf::IndiProperty &ipRecv)
{
    // MagAOXApp::handleNewProperty always passes its own `this`, so the argument is the app, not
    // this telemeter.
    telemeter<derivedT> *tel = static_cast<derivedT *>(app);
    return tel->newCallBack_maxLogTime(ipRecv);
}

template <class derivedT>
int telemeter<derivedT>::newCallBack_maxLogTime(const pcf::IndiProperty &ipRecv)
{
    unsigned target = 0;

    if (derived().indiTargetUpdate(m_indiP_maxLogTime, target, ipRecv, false) < 0)
    {
        return derivedT::template log<software_error, -1>({__FILE__, __LINE__});
    }

    m_tel.maxLogTime(target);

    derivedT::template log<software_info>({__FILE__, __LINE__, "Set telemetry file interval to " + std::to_string(target) + " minutes"});

    derived().updateIfChanged(m_indiP_maxLogTime, "current", target, INDI_IDLE);

    return 0;
}

template <class derivedT>
int telemeter<derivedT>::appShutdown()
{
    return 0;
}

template <class derivedT>
template <class telT, class... telTs>
int telemeter<derivedT>::checkRecordTimes(const telT &tel, telTs... tels)
{
    timespec ts;

    clock_gettime(CLOCK_REALTIME, &ts);
    return checkRecordTimes(ts, tel, tels...);
}

template <class derivedT>
template <class telT, class... telTs>
int telemeter<derivedT>::checkRecordTimes(timespec &ts, const telT &tel, telTs... tels)
{
    // Check if it's been more than maxInterval seconds since the last record.  This is corrected for the pause of the main loop.
    if (((double)ts.tv_sec - ((double)ts.tv_nsec) / 1e9) - ((double)telT::lastRecord.tv_sec - ((double)telT::lastRecord.tv_nsec) / 1e9) > m_maxInterval - ((double)derived().m_loopPause) / 1e9)
    {
        derived().recordTelem(&tel);
    }

    return checkRecordTimes(ts, tels...);
}

template <class derivedT>
int telemeter<derivedT>::checkRecordTimes(timespec &ts)
{
    static_cast<void>(ts); // be unused

    return 0;
}

/// Call telemeter::setupConfig with error checking
/**
  * \param cfig the application configurator
  */
#define TELEMETER_SETUP_CONFIG( cfig )                                                   \
    if (telemeterT::setupConfig( cfig) < 0)                                              \
    {                                                                                    \
        log<software_error>({__FILE__, __LINE__, "Error from telemeterT::setupConfig"}); \
        m_shutdown = true;                                                               \
    }

/// Call telemeter::loadConfig with error checking
/** This must be inside a function that returns int, e.g. the standard loadConfigImpl.
  * \param cfig the application configurator
  */
#define TELEMETER_LOAD_CONFIG( cfig )                                                              \
    if (telemeterT::loadConfig(cfig) < 0)                                                          \
    {                                                                                              \
        return log<software_error, -1>({__FILE__, __LINE__, "Error from telemeterT::loadConfig"}); \
    }

/// Call telemeter::appStartup with error checking
#define TELEMETER_APP_STARTUP                                 \
    if (telemeterT::appStartup() < 0)                         \
    {                                                         \
        return log<software_error, -1>({__FILE__, __LINE__}); \
    }

/// Call telemeter::appLogic with error checking
#define TELEMETER_APP_LOGIC                                   \
    if (telemeterT::appLogic() < 0)                           \
    {                                                         \
        return log<software_error, -1>({__FILE__, __LINE__}); \
    }

/// Call telemeter::appShutdown with error checking
#define TELEMETER_APP_SHUTDOWN                                                           \
    if (telemeterT::appShutdown() < 0)                                                   \
    {                                                                                    \
        log<software_error>({__FILE__, __LINE__, "error from telemeterT::appShutdown"}); \
    }


#ifdef XWCTEST_NAMESPACE
} // namespace XWCTEST_NAMESPACE
#endif


} // namespace dev
} // namespace tty
} // namespace MagAOX

#endif // tty_telemeter_hpp
