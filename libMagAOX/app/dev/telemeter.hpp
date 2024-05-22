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
    /// The log manager type.
    typedef logger::logManager<derivedT, logFileRaw> logManagerT;

    logManagerT m_tel;

    double m_maxInterval{10.0}; ///< The maximum interval, in seconds, between telemetry records. Default is 10.0 seconds.

    telemeter();

    /// Make a telemetry recording
    /** Wrapper for logManager::log, which updates telT::lastRecord.
     *
     * \tparam logT the log entry type
     * \tparam retval the value returned by this method.
     *
     */
    template <typename telT>
    int telem(const typename telT::messageT &msg /**< [in] the data to log */);

    /// Make a telemetry recording, for an empty record
    /** Wrapper for logManager::log, which updates telT::lastRecord.
     *
     * \tparam logT the log entry type
     * \tparam retval the value returned by this method.
     *
     */
    template <typename telT>
    int telem();

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
template <typename telT>
int telemeter<derivedT>::telem(const typename telT::messageT &msg)
{

    m_tel.template log<telT>(msg, logPrio::LOG_TELEM);

    // Set timestamp
    clock_gettime(CLOCK_REALTIME, &telT::lastRecord);

    return 0;
}

template <class derivedT>
template <typename telT>
int telemeter<derivedT>::telem()
{

    m_tel.template log<telT>(logPrio::LOG_TELEM);

    // Set timestamp
    clock_gettime(CLOCK_REALTIME, &telT::lastRecord);

    return 0;
}

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

    m_tel.logPath(std::string(derived().MagAOXPath) + "/" + MAGAOX_telRelPath);

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
    //        Begin the telemetry system
    //----------------------------------------//

    m_tel.logThreadStart();

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
    return derived().checkRecordTimes();
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

} // namespace dev
} // namespace tty
} // namespace MagAOX

#endif // tty_telemeter_hpp
