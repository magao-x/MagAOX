/** \file logMeta.hpp
 * \brief Declares and defines the logMeta class and related classes.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_files
 *
 * History:
 * - 2020-01-02 created by JRM
 */

#ifndef logger_logMeta_hpp
#define logger_logMeta_hpp

#include <cmath>
#include <limits>
#include <sstream>

#include <mx/ioutils/fits/fitsHeaderCard.hpp>
// #define HARD_EXIT
#include "logMap.hpp"

namespace MagAOX
{
namespace logger
{

// This is how the user specifies an item of log meta data (i.e. via a config file)
struct logMetaSpec
{
    std::string          device;    ///< Device name that produced the log entry.
    flatlogs::eventCodeT eventCode; ///< Flatlogs event code used to select the log type.
    std::string          member;    ///< Log member name to extract.
    std::string          keyword;   ///< Optional FITS keyword override.
    std::string          format;    ///< Optional printf-style value format override.
    std::string          comment;   ///< Optional FITS comment override.

    /// Construct an empty metadata specification.
    logMetaSpec()
    {
    }

    /// Construct a complete metadata specification.
    logMetaSpec( const std::string         &dev,  /**< [in] device name that produced the log entry */
                 const flatlogs::eventCodeT ec,   /**< [in] flatlogs event code used to select the log type */
                 const std::string         &memb, /**< [in] log member name to extract */
                 const std::string         &k,    /**< [in] FITS keyword override */
                 const std::string         &f,    /**< [in] printf-style value format override */
                 const std::string         &c     /**< [in] FITS comment override */
                 )
        : device( dev ), eventCode( ec ), member( memb ), keyword( k ), format( f ), comment( c )
    {
    }

    /// Construct a metadata specification using the log type's default FITS keyword, format, and comment.
    logMetaSpec( const std::string         &dev, /**< [in] device name that produced the log entry */
                 const flatlogs::eventCodeT ec,  /**< [in] flatlogs event code used to select the log type */
                 const std::string         &memb /**< [in] log member name to extract */
                 )
        : device( dev ), eventCode( ec ), member( memb )
    {
    }
};

// This is the data returned by the member accessor.
struct logMetaDetail
{
    std::string keyword;             ///< FITS keyword to write for this metadata value.
    std::string comment;             ///< FITS comment describing the metadata value.
    std::string format;              ///< printf-style format used to render numeric values.
    int         valType{ -1 };       ///< FITS value type used to serialize the metadata value.
    int         metaType{ -1 };      ///< Metadata time behavior, one of logMeta::metaTypes.
    void       *accessor{ nullptr }; ///< Type-erased accessor for extracting the member from a log message.
    bool        hierarch{ true };    ///< If true, include the device name in a HIERARCH-style keyword.

    /// Construct an empty metadata detail.
    logMetaDetail()
    {
    }

    /// Construct a metadata detail with explicit keyword, comment, format, type, and accessor.
    logMetaDetail( const std::string &k,  /**< [in] FITS keyword */
                   const std::string &c,  /**< [in] FITS comment */
                   const std::string &f,  /**< [in] printf-style value format */
                   int                vt, /**< [in] FITS value type */
                   int                mt, /**< [in] metadata time behavior */
                   void              *acc /**< [in] type-erased value accessor */
                   )
        : keyword( k ), comment( c ), format( f ), valType( vt ), metaType( mt ), accessor( acc )
    {
    }

    /// Construct a metadata detail with explicit HIERARCH behavior.
    logMetaDetail( const std::string &k,   /**< [in] FITS keyword */
                   const std::string &c,   /**< [in] FITS comment */
                   const std::string &f,   /**< [in] printf-style value format */
                   int                vt,  /**< [in] FITS value type */
                   int                mt,  /**< [in] metadata time behavior */
                   void              *acc, /**< [in] type-erased value accessor */
                   bool               h    /**< [in] true to include the device name in the FITS keyword */
                   )
        : keyword( k ), comment( c ), format( f ), valType( vt ), metaType( mt ), accessor( acc ), hierarch( h )
    {
    }

    /// Construct a metadata detail with default value formatting and explicit HIERARCH behavior.
    logMetaDetail( const std::string &k,   /**< [in] FITS keyword */
                   const std::string &c,   /**< [in] FITS comment */
                   int                vt,  /**< [in] FITS value type */
                   int                mt,  /**< [in] metadata time behavior */
                   void              *acc, /**< [in] type-erased value accessor */
                   bool               h    /**< [in] true to include the device name in the FITS keyword */
                   )
        : keyword( k ), comment( c ), valType( vt ), metaType( mt ), accessor( acc ), hierarch( h )
    {
    }

    /// Construct a metadata detail with only keyword, type, and accessor.
    logMetaDetail( const std::string &k,  /**< [in] FITS keyword */
                   int                vt, /**< [in] FITS value type */
                   int                mt, /**< [in] metadata time behavior */
                   void              *acc /**< [in] type-erased value accessor */
                   )
        : keyword( k ), valType( vt ), metaType( mt ), accessor( acc )
    {
    }

    /// Construct a metadata detail with keyword, type, accessor, and explicit HIERARCH behavior.
    logMetaDetail( const std::string &k,   /**< [in] FITS keyword */
                   int                vt,  /**< [in] FITS value type */
                   int                mt,  /**< [in] metadata time behavior */
                   void              *acc, /**< [in] type-erased value accessor */
                   bool               h    /**< [in] true to include the device name in the FITS keyword */
                   )
        : keyword( k ), valType( vt ), metaType( mt ), accessor( acc ), hierarch( h )
    {
    }
};

/*logMetaDetail logMemberAccessor( flatlogs::eventCodeT ec,
                                 const std::string & memberName
                               );
*/

/// Verify that a log entry's message matches its flatbuffer schema.
/**
 * \returns true if \p log holds an entry whose message passes the schema check for \p ev
 */
bool verifyLogEntry( flatlogs::eventCodeT ev, /**< [in] expected event code for the log entry */
                     logEntryRecord      &log /**< [in] the log entry to verify */
);

/// Check whether two metadata timestamps are separated by no more than the configured gap.
inline bool logMetaGapValid( const flatlogs::timespecX &t0,    /**< [in] first timestamp */
                             const flatlogs::timespecX &t1,    /**< [in] second timestamp */
                             double                     maxGap /**< [in] maximum permitted gap in seconds */
)
{
    flatlogs::timespecX tt0 = t0;
    flatlogs::timespecX tt1 = t1;
    return maxGap < 0 || std::fabs( tt1.asDouble() - tt0.asDouble() ) <= maxGap;
}

/// Build a user-facing reason for a metadata telemetry gap failure.
inline std::string logMetaGapReason( double maxGap /**< [in] maximum permitted gap in seconds */ )
{
    std::ostringstream oss;
    oss << "due to gap in telemetry exceeding " << maxGap << " sec";
    return oss.str();
}

/// Return a metadata lookup failure while optionally recording its reason.
inline int logMetaFail( std::string       *reason, /**< [out] optional reason string */
                        const std::string &msg,    /**< [in] reason to record */
                        int                rv = -1 /**< [in] return value */
)
{
    if( reason != nullptr )
    {
        *reason = msg;
    }

    return rv;
}

/// Normalize an angle into [0, 360) degrees.
inline double logMetaNormalizeAngle360( double angle /**< [in] angle in degrees */ )
{
    double wrapped = std::fmod( angle, 360.0 );
    if( wrapped < 0 )
    {
        wrapped += 360.0;
    }
    if( wrapped == 360.0 )
    {
        wrapped = 0.0;
    }
    return wrapped;
}

/// Normalize an angle into [-180, 180) degrees.
inline double logMetaNormalizeAngle180( double angle /**< [in] angle in degrees */ )
{
    return logMetaNormalizeAngle360( angle + 180.0 ) - 180.0;
}

/// Return the shortest signed angular delta between two angles in degrees.
inline double logMetaAngleDelta( double from, /**< [in] starting angle in degrees */
                                 double to    /**< [in] ending angle in degrees */
)
{
    double delta = std::fmod( to - from, 360.0 );
    if( delta > 180.0 )
    {
        delta -= 360.0;
    }
    else if( delta < -180.0 )
    {
        delta += 360.0;
    }

    return delta;
}

/// Normalize an interpolated angle using the input samples' apparent convention.
inline double logMetaNormalizeInterpolatedAngle( double angle, /**< [in] interpolated angle in degrees */
                                                 double a0,    /**< [in] first endpoint angle in degrees */
                                                 double a1     /**< [in] second endpoint angle in degrees */
)
{
    if( a0 >= -180.0 && a0 <= 180.0 && a1 >= -180.0 && a1 <= 180.0 )
    {
        return logMetaNormalizeAngle180( angle );
    }

    return logMetaNormalizeAngle360( angle );
}

/// Report a skipped unverifiable log entry without aborting metadata processing.
/** Each entry is reported once.
 */
inline void
reportUnverifiableLogEntry( const std::string    &appName, /**< [in] app/device name being searched */
                            flatlogs::eventCodeT  ev,      /**< [in] event code being searched */
                            const logEntryRecord &before,  /**< [in] verified entry before the bad entry, if known */
                            const logEntryRecord &failure, /**< [in] unverifiable entry that was skipped */
                            const logEntryRecord &after,   /**< [in] verified entry after the bad entry, if found */
                            const std::string    &context  /**< [in] search context that encountered the bad entry */
)
{
    std::string sourceFile = !failure.empty() ? failure.m_sourceFile : std::string( "<unknown>" );
    size_t sourceByte = !failure.empty() ? failure.m_sourceOffset : std::numeric_limits<size_t>::max();

    if( !failure.empty() )
    {
        static std::set<std::string> reported;
        std::string reportKey = appName + "|" + std::to_string( ev ) + "|" + sourceFile + "|" +
                                std::to_string( sourceByte ) + "|" + std::to_string( failure.m_ts.time_s ) + "|" +
                                std::to_string( failure.m_ts.time_ns );

        if( reported.count( reportKey ) > 0 )
        {
            return;
        }

        reported.insert( reportKey );
    }

    std::cerr << "Unverifiable log entry skipped while processing FITS metadata: app=" << appName << " ev=" << ev
              << " source=" << sourceFile;

    if( sourceByte != std::numeric_limits<size_t>::max() )
    {
        std::cerr << " sourceByte=" << sourceByte;
    }
    else
    {
        std::cerr << " sourceByte=<unknown>";
    }

    if( !before.empty() )
    {
        std::cerr << " beforeTs=" << logMapDebugTime( before.m_ts );
    }
    else
    {
        std::cerr << " beforeTs=<none>";
    }

    if( !failure.empty() )
    {
        std::cerr << " failureTs=" << logMapDebugTime( failure.m_ts ) << " msgLen=" << failure.m_msgLen
                  << " totalSize=" << failure.m_entrySize;
    }
    else
    {
        std::cerr << " failureTs=<none>";
    }

    if( !after.empty() )
    {
        std::cerr << " afterTs=" << logMapDebugTime( after.m_ts );
    }
    else
    {
        std::cerr << " afterTs=<none>";
    }

    std::cerr << " context=" << context << "\n";
}

/// Find the nearest prior log entry that also passes flatbuffer schema verification.
/** Searches the loaded files for \p appName.
 *
 * \returns the entry found, or an empty record if there is none
 */
template <class verboseT = XWC_DEFAULT_VERBOSITY>
logEntryRecord getPriorVerifiedEntry( logMap<verboseT>          &lm,      /**< [in] loaded log map to search */
                                      const std::string         &appName, /**< [in] app/device name to search */
                                      flatlogs::eventCodeT       ev,      /**< [in] event code to search for */
                                      const flatlogs::timespecX &ts       /**< [in] timestamp to be prior to */
)
{
    logEntryRecord prior;

    auto appBuffer = lm.m_appToBufferMap.find( appName );
    if( appBuffer == lm.m_appToBufferMap.end() || appBuffer->second.m_memory.size() == 0 )
    {
        return prior;
    }

    logInMemory &lim = appBuffer->second;

    logMemoryWalk walk( lim );
    logStep       step;
    size_t        file   = 0;
    size_t        offset = 0;
    while( walk.next( step, file, offset ) )
    {
        if( ts < step.m_entry.ts )
        {
            break;
        }

        if( step.m_entry.eventCode == ev )
        {
            logEntryRecord candidate = lim.record( file, offset, step );
            if( verifyLogEntry( ev, candidate ) )
            {
                prior = std::move( candidate );
            }
            else
            {
                DEBUG_CRUMB( "getPriorVerifiedEntry rejected app=" + appName + " ev=" + std::to_string( ev ) +
                             " logTs=" + logMapDebugTime( step.m_entry.ts ) +
                             " msgLen=" + std::to_string( step.m_entry.msgLen ) +
                             " totalSize=" + std::to_string( step.m_entry.entrySize ) + " " );
            }
        }
    }

    if( !prior.empty() )
    {
        DEBUG_CRUMB( "getPriorVerifiedEntry found app=" + appName + " ev=" + std::to_string( ev ) +
                     " queryTs=" + logMapDebugTime( ts ) + " logTs=" + logMapDebugTime( prior.m_ts ) + " " );
    }
    else
    {
        DEBUG_CRUMB( "getPriorVerifiedEntry no match app=" + appName + " ev=" + std::to_string( ev ) +
                     " queryTs=" + logMapDebugTime( ts ) + " " );
    }

    return prior;
}

/// Find the next log entry that also passes flatbuffer schema verification.
/** Every unverifiable entry passed over, including \p current if it fails, is counted as a recoverable error and
 * reported.
 *
 * \returns the entry found, or an empty record if there is none
 */
template <class verboseT = XWC_DEFAULT_VERBOSITY>
logEntryRecord getNextVerifiedEntry( logMap<verboseT>     &lm,      /**< [in] loaded log map to search */
                                     const logEntryRecord &current, /**< [in] entry before the desired verified entry */
                                     const std::string    &appName, /**< [in] app/device name to search */
                                     const logEntryRecord &before   /**< [in] verified entry before current, if any */
)
{
    logEntryRecord found;

    if( current.empty() )
    {
        return found;
    }

    flatlogs::eventCodeT ev = current.m_eventCode;

    /// An unverifiable entry skipped during the search, with the entry before it.
    struct rejectedLog
    {
        logEntryRecord m_before;  ///< Entry immediately before the unverifiable entry in the search.
        logEntryRecord m_failure; ///< Unverifiable entry skipped during the search.
    };

    std::vector<rejectedLog> rejected;

    logEntryRecord currentCopy = current;
    if( !verifyLogEntry( ev, currentCopy ) )
    {
        rejected.push_back( { before, current } );
    }

    logEntryRecord at = current;
    logEntryRecord candidate;
    while( lm.getNextEntry( candidate, at, appName ) == 0 )
    {
        if( verifyLogEntry( ev, candidate ) )
        {
            for( const rejectedLog &badLog : rejected )
            {
                lm.recordRecoverableError( appName );
                reportUnverifiableLogEntry(
                    appName, ev, badLog.m_before, badLog.m_failure, candidate, "next-verified-search" );
            }

            DEBUG_CRUMB( "getNextVerifiedEntry found app=" + appName + " ev=" + std::to_string( ev ) + " currentTs=" +
                         logMapDebugTime( current.m_ts ) + " logTs=" + logMapDebugTime( candidate.m_ts ) + " " );
            found = std::move( candidate );
            return found;
        }

        DEBUG_CRUMB( "getNextVerifiedEntry rejected app=" + appName + " ev=" + std::to_string( ev ) +
                     " logTs=" + logMapDebugTime( candidate.m_ts ) + " msgLen=" + std::to_string( candidate.m_msgLen ) +
                     " totalSize=" + std::to_string( candidate.m_entrySize ) + " " );

        rejected.push_back( { at, candidate } );
        at = candidate;
    }

    for( const rejectedLog &badLog : rejected )
    {
        lm.recordRecoverableError( appName );
        reportUnverifiableLogEntry(
            appName, ev, badLog.m_before, badLog.m_failure, logEntryRecord(), "next-verified-search" );
    }

    DEBUG_CRUMB( "getNextVerifiedEntry no match app=" + appName + " ev=" + std::to_string( ev ) +
                 " currentTs=" + logMapDebugTime( current.m_ts ) + " " );

    return found;
}

/// Get the value of a state-type metadata item over an exposure.
/** If the value is updated during the exposure, the new value is used, otherwise the value set at the start
 * of the exposure is used.
 *
 * \returns 0 on success
 * \returns -1 if the value is unavailable, with the reason in \p failureReason
 */
template <typename valT, class verboseT = XWC_DEFAULT_VERBOSITY>
int getLogStateVal( valT                      &val,     /**< [out] the value */
                    logMap<verboseT>          &lm,      /**< [in,out] loaded logs to search */
                    const std::string         &appName, /**< [in] app/device name to search */
                    flatlogs::eventCodeT       ev,      /**< [in] event code of the log type holding the value */
                    const flatlogs::timespecX &stime,   /**< [in] exposure start time */
                    const flatlogs::timespecX &atime,   /**< [in] exposure end time */
                    valT ( *getter )( void * ),         /**< [in] gets the value from a log entry's message */
                    double       maxGap        = -1,    /**< [in] maximum allowed metadata gap in seconds */
                    std::string *failureReason = 0      /**< [out] [optional] reason the value is unavailable */
)
{
    logEntryRecord atprior;
    logEntryRecord stprior;

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    if( lm.getPriorEntry( stprior, appName, ev, stime ) != 0 )
    {
        return logMetaFail( failureReason, "due to missing prior telemetry" );
    }

    if( stprior.empty() || stprior.m_eventCode != ev )
    {
        return logMetaFail( failureReason, "due to missing prior telemetry" );
    }
    if( !verifyLogEntry( ev, stprior ) )
    {
        DEBUG_CRUMB( "getLogStateVal prior verify failed app=" + appName + " ev=" + std::to_string( ev ) +
                     " logTs=" + logMapDebugTime( stprior.m_ts ) + " " );
        stprior = getPriorVerifiedEntry( lm, appName, ev, stime );
        if( stprior.empty() )
        {
            return logMetaFail( failureReason, "due to unverifiable prior telemetry" );
        }
    }

    if( !logMetaGapValid( stprior.m_ts, stime, maxGap ) )
    {
        return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
    }

    valT stprV = getter( stprior.message() );

    valT atprV;

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    if( lm.getNextEntry( atprior, stprior, appName ) != 0 )
    {
        if( !logMetaGapValid( stprior.m_ts, atime, maxGap ) )
        {
            return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
        }

        return logMetaFail( failureReason, "due to missing following telemetry" );
    }
    if( atprior.empty() )
    {
        if( logMetaGapValid( stprior.m_ts, atime, maxGap ) )
        {
            val = stprV;
            return 0;
        }

        return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
    }
    if( !verifyLogEntry( ev, atprior ) )
    {
        DEBUG_CRUMB( "getLogStateVal next verify failed app=" + appName + " ev=" + std::to_string( ev ) +
                     " logTs=" + logMapDebugTime( atprior.m_ts ) + " " );
        atprior = getNextVerifiedEntry( lm, atprior, appName, stprior );
        if( atprior.empty() )
        {
            return logMetaFail( failureReason, "due to unverifiable following telemetry" );
        }
    }

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    while( atprior.m_ts < atime )
    {
        if( !logMetaGapValid( stprior.m_ts, atprior.m_ts, maxGap ) )
        {
            return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
        }

        atprV = getter( atprior.message() );
        if( atprV != stprV )
        {
            val = atprV;
            return 0;
        }
        stprior = atprior;
        if( lm.getNextEntry( atprior, stprior, appName ) != 0 )
        {
            if( !logMetaGapValid( stprior.m_ts, atime, maxGap ) )
            {
                return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
            }

            return logMetaFail( failureReason, "due to missing following telemetry" );
        }
        if( atprior.empty() )
        {
            if( logMetaGapValid( stprior.m_ts, atime, maxGap ) )
            {
                val = stprV;
                return 0;
            }

            return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
        }
        if( !verifyLogEntry( ev, atprior ) )
        {
            DEBUG_CRUMB( "getLogStateVal next verify failed app=" + appName + " ev=" + std::to_string( ev ) +
                         " logTs=" + logMapDebugTime( atprior.m_ts ) + " " );
            atprior = getNextVerifiedEntry( lm, atprior, appName, stprior );
            if( atprior.empty() )
            {
                return logMetaFail( failureReason, "due to unverifiable following telemetry" );
            }
        }
    }

    if( !logMetaGapValid( stprior.m_ts, atime, maxGap ) )
    {
        return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
    }

    val = stprV;

    return 0;
}

/// Interpolate the value of a continuous metadata item to the middle of an exposure.
/** Interpolates linearly between the last entry before the middle of the exposure and the next entry after it.
 *
 * \returns 0 on success
 * \returns 1 if the value is unavailable, with the reason in \p failureReason
 */
template <typename valT, class verboseT = XWC_DEFAULT_VERBOSITY>
int getLogContVal( valT                      &val,     /**< [out] the value */
                   logMap<verboseT>          &lm,      /**< [in,out] loaded logs to search */
                   const std::string         &appName, /**< [in] app/device name to search */
                   flatlogs::eventCodeT       ev,      /**< [in] event code of the log type holding the value */
                   const flatlogs::timespecX &stime,   /**< [in] exposure start time */
                   const flatlogs::timespecX &atime,   /**< [in] exposure acquisition/end time */
                   valT ( *getter )( void * ),         /**< [in] gets the value from a log entry's message */
                   double       maxGap        = -1,    /**< [in] maximum allowed metadata gap in seconds */
                   std::string *failureReason = 0,     /**< [out] [optional] reason the value is unavailable */
                   bool         isAngle       = false  /**< [in] true if the value is an angle in degrees */
)
{
    logEntryRecord atafter;
    logEntryRecord stprior;

    flatlogs::timespecX midexp = meanTimespecX( atime, stime );

    // Get log entry before midexp
    if( lm.getPriorEntry( stprior, appName, ev, midexp ) != 0 )
    {
        return logMetaFail( failureReason, "due to missing prior telemetry", 1 );
    }

    if( stprior.empty() || stprior.m_eventCode != ev )
    {
        return logMetaFail( failureReason, "due to missing prior telemetry", 1 );
    }
    if( !verifyLogEntry( ev, stprior ) )
    {
        stprior = getPriorVerifiedEntry( lm, appName, ev, midexp );
        if( stprior.empty() )
        {
            return logMetaFail( failureReason, "due to unverifiable prior telemetry", 1 );
        }
    }

    if( !logMetaGapValid( stprior.m_ts, midexp, maxGap ) )
    {
        return logMetaFail( failureReason, logMetaGapReason( maxGap ), 1 );
    }

    valT stprV = getter( stprior.message() );

    // Get log entry after.
    if( lm.getNextEntry( atafter, stprior, appName ) != 0 )
    {
#ifdef HARD_EXIT
        exit( -1 );
#endif
        return logMetaFail( failureReason, "due to missing following telemetry", 1 );
    }
    if( atafter.empty() )
    {
        return logMetaFail( failureReason, "due to missing following telemetry", 1 );
    }
    if( !verifyLogEntry( ev, atafter ) )
    {
        atafter = getNextVerifiedEntry( lm, atafter, appName, stprior );
        if( atafter.empty() )
        {
            return logMetaFail( failureReason, "due to unverifiable following telemetry", 1 );
        }
    }

    if( !logMetaGapValid( midexp, atafter.m_ts, maxGap ) || !logMetaGapValid( stprior.m_ts, atafter.m_ts, maxGap ) )
    {
        return logMetaFail( failureReason, logMetaGapReason( maxGap ), 1 );
    }

    valT atprV = getter( atafter.message() );

    double st = stprior.m_ts.asDouble();
    double it = midexp.asDouble();
    double et = atafter.m_ts.asDouble();

    double delta = atprV - stprV;
    if( isAngle )
    {
        delta = logMetaAngleDelta( stprV, atprV );
    }

    double interp = stprV + delta / ( et - st ) * ( it - st );
    if( isAngle )
    {
        interp = logMetaNormalizeInterpolatedAngle( interp, stprV, atprV );
    }

    val = static_cast<valT>( interp );

    return 0;
}

/// Manage meta data for a log entry
/** Handles cases where log is a state, i.e. has one of a finite number of values, or is a
 * continuous variable, e.g. a temperature.
 *
 * Contains the information to construct a FITS header card.
 */
struct logMeta
{
    typedef XWC_DEFAULT_VERBOSITY verboseT;

  public:
    enum valTypes
    {
        String           = mx::fits::fitsType<std::string>(),
        Bool             = mx::fits::fitsType<bool>(),
        Char             = mx::fits::fitsType<char>(),
        UChar            = mx::fits::fitsType<unsigned char>(),
        Short            = mx::fits::fitsType<short>(),
        UShort           = mx::fits::fitsType<unsigned short>(),
        Int              = mx::fits::fitsType<int>(),
        UInt             = mx::fits::fitsType<unsigned int>(),
        Long             = mx::fits::fitsType<long>(),
        ULong            = mx::fits::fitsType<unsigned long>(),
        LongLong         = mx::fits::fitsType<long long>(),
        ULongLong        = mx::fits::fitsType<unsigned long long>(),
        Float            = mx::fits::fitsType<float>(),
        Double           = mx::fits::fitsType<double>(),
        Vector_String    = 10000,
        Vector_Bool      = 10002,
        Vector_Char      = 10004,
        Vector_UChar     = 10006,
        Vector_Short     = 10008,
        Vector_UShort    = 10010,
        Vector_Int       = 10012,
        Vector_UInt      = 10014,
        Vector_Long      = 10016,
        Vector_ULong     = 10018,
        Vector_LongLong  = 10020,
        Vector_ULongLong = 10022,
        Vector_Float     = 10024,
        Vector_Double    = 10026
    };

    enum metaTypes
    {
        State,
        Continuous,
        Continuous_Angle
    };

    /// The string written when metadata is expected but unavailable.
    /**
     * \returns the shared unavailable-value sentinel.
     */
    static const std::string &unavailableValue();

  protected:
    logMetaSpec   m_spec;   ///< User/configuration specification for this metadata item.
    logMetaDetail m_detail; ///< Log-type detail used to extract and serialize this metadata item.

    bool        m_isValid{ false };                   ///< True when the metadata member resolved to an accessor.
    std::string m_invalidValue{ unavailableValue() }; ///< Sentinel returned when metadata cannot be read.

    std::string m_unavailableReason; ///< Reason the last metadata lookup returned unavailable, if known.

    /// Build the FITS keyword for this metadata item.
    /**
     * \returns the keyword, including the device prefix when HIERARCH-style output is enabled.
     */
    std::string fitsKeyword() const;

  public:
    /// Construct a metadata item from a specification.
    logMeta( const logMetaSpec &lms /**< [in] the specification of this meta data entry */ );

    /// Get the source device name.
    const std::string &device();

    /// Get the FITS keyword for this metadata item.
    const std::string &keyword();

    /// Get the FITS comment for this metadata item.
    const std::string &comment();

    /// Get the reason the last lookup returned the unavailable sentinel.
    const std::string &unavailableReason() const;

    /// Resolve the log accessor and metadata details for a specification.
    int setLog( const logMetaSpec &lms /**< [in] the specification to resolve */ );

    /// Get the metadata value for an exposure interval.
    std::string value( logMap<verboseT>          &lm,         /**< [in,out] loaded logs to search */
                       const flatlogs::timespecX &stime,      /**< [in] exposure start time */
                       const flatlogs::timespecX &atime,      /**< [in] exposure acquisition/end time */
                       double                     maxGap = -1 /**< [in] maximum allowed metadata gap in seconds */
    );

    /// Get a numeric metadata value for an exposure interval.
    std::string valueNumber( logMap<verboseT>          &lm,         /**< [in,out] loaded logs to search */
                             const flatlogs::timespecX &stime,      /**< [in] exposure start time */
                             const flatlogs::timespecX &atime,      /**< [in] exposure acquisition/end time */
                             double                     maxGap = -1 /**< [in] maximum allowed metadata gap in seconds */
    );

    /// Get a string metadata value for an exposure interval.
    std::string valueString( logMap<verboseT>          &lm,         /**< [in,out] loaded logs to search */
                             const flatlogs::timespecX &stime,      /**< [in] exposure start time */
                             const flatlogs::timespecX &atime,      /**< [in] exposure acquisition/end time */
                             double                     maxGap = -1 /**< [in] maximum allowed metadata gap in seconds */
    );

    /// Build a FITS header card for an unavailable metadata value.
    mx::fits::fitsHeaderCard<verboseT> unavailableCard() const;

    /// Build a FITS header card for this metadata item over an exposure interval.
    mx::fits::fitsHeaderCard<verboseT>
    card( logMap<verboseT>          &lm,         /**< [in,out] loaded logs to search */
          const flatlogs::timespecX &stime,      /**< [in] exposure start time */
          const flatlogs::timespecX &atime,      /**< [in] exposure acquisition/end time */
          double                     maxGap = -1 /**< [in] maximum allowed metadata gap in seconds */
    );
};

} // namespace logger
} // namespace MagAOX

#endif // logger_logMeta_hpp
