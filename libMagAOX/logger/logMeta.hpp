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

/// Verify that a raw log entry matches its flatbuffer schema.
bool verifyLogEntry( flatlogs::eventCodeT ev, /**< [in] expected event code for the raw log entry */
                     char                *log /**< [in] raw log entry buffer to verify */
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
inline void
reportUnverifiableLogEntry( const std::string   &appName, /**< [in] app/device name being searched */
                            flatlogs::eventCodeT ev,      /**< [in] event code being searched */
                            char *before,  /**< [in] verified entry before the bad entry, or null if unknown */
                            char *failure, /**< [in] unverifiable entry that was skipped */
                            char *after,   /**< [in] verified entry after the bad entry, or null if none was found */
                            const std::string &sourceFile, /**< [in] source log file containing the bad entry */
                            size_t             sourceByte, /**< [in] byte offset of the bad entry in the source file */
                            const std::string &context     /**< [in] search context that encountered the bad entry */
)
{
    if( failure != nullptr )
    {
        static std::set<std::string> reported;
        flatlogs::timespecX          failureTs = flatlogs::logHeader::timespec( failure );
        std::string                  reportKey = appName + "|" + std::to_string( ev ) + "|" + sourceFile + "|" +
                                std::to_string( sourceByte ) + "|" + std::to_string( failureTs.time_s ) + "|" +
                                std::to_string( failureTs.time_ns );

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

    if( before != nullptr )
    {
        std::cerr << " beforeTs=" << logMapDebugTime( flatlogs::logHeader::timespec( before ) );
    }
    else
    {
        std::cerr << " beforeTs=<none>";
    }

    if( failure != nullptr )
    {
        std::cerr << " failureTs=" << logMapDebugTime( flatlogs::logHeader::timespec( failure ) )
                  << " msgLen=" << flatlogs::logHeader::msgLen( failure )
                  << " totalSize=" << flatlogs::logHeader::totalSize( failure );
    }
    else
    {
        std::cerr << " failureTs=<none>";
    }

    if( after != nullptr )
    {
        std::cerr << " afterTs=" << logMapDebugTime( flatlogs::logHeader::timespec( after ) );
    }
    else
    {
        std::cerr << " afterTs=<none>";
    }

    std::cerr << " context=" << context << "\n";
}

/// Find the nearest prior log entry that also passes flatbuffer schema verification.
template <class verboseT = XWC_DEFAULT_VERBOSITY>
char *getPriorVerifiedLog( logMap<verboseT>          &lm,      /**< [in] loaded log map to search */
                           const std::string         &appName, /**< [in] app/device name to search */
                           flatlogs::eventCodeT       ev,      /**< [in] event code to search for */
                           const flatlogs::timespecX &ts       /**< [in] timestamp to be prior to */
)
{
    auto appBuffer = lm.m_appToBufferMap.find( appName );
    if( appBuffer == lm.m_appToBufferMap.end() || appBuffer->second.m_memory.size() == 0 )
    {
        return nullptr;
    }

    logInMemory &lim       = appBuffer->second;
    char        *buffer    = lim.m_memory.data();
    char        *bufferEnd = lim.m_memory.data() + lim.m_memory.size();
    char        *prior     = nullptr;

    while( buffer < bufferEnd )
    {
        size_t               totalSize = 0;
        flatlogs::timespecX  minTs;
        flatlogs::timespecX *minTsPtr = nullptr;
        if( prior != nullptr )
        {
            minTs    = flatlogs::logHeader::timespec( prior );
            minTsPtr = &minTs;
        }

        if( !logMapEntrySane( totalSize, buffer, bufferEnd, minTsPtr ) )
        {
            DEBUG_CRUMB( "getPriorVerifiedLog invalid entry app=" + appName + " ev=" + std::to_string( ev ) +
                         " offset=" + std::to_string( buffer - lim.m_memory.data() ) + " totalSize=" +
                         std::to_string( totalSize ) + " memoryBytes=" + std::to_string( lim.m_memory.size() ) + " " );

            char *resynced = logMapResync( buffer, bufferEnd, minTsPtr );
            if( resynced != nullptr )
            {
                DEBUG_CRUMB( "getPriorVerifiedLog resync app=" + appName + " ev=" + std::to_string( ev ) +
                             " offset=" + std::to_string( buffer - lim.m_memory.data() ) +
                             " resyncOffset=" + std::to_string( resynced - lim.m_memory.data() ) );
                buffer = resynced;
                continue;
            }

            DEBUG_CRUMB( "getPriorVerifiedLog resync failed app=" + appName + " ev=" + std::to_string( ev ) +
                         " offset=" + std::to_string( buffer - lim.m_memory.data() ) );
            return prior;
        }

        flatlogs::timespecX logTs = flatlogs::logHeader::timespec( buffer );
        if( ts < logTs )
        {
            break;
        }

        if( flatlogs::logHeader::eventCode( buffer ) == ev )
        {
            if( verifyLogEntry( ev, buffer ) )
            {
                prior = buffer;
            }
            else
            {
                DEBUG_CRUMB( "getPriorVerifiedLog rejected app=" + appName + " ev=" + std::to_string( ev ) +
                             " logTs=" + logMapDebugTime( logTs ) +
                             " msgLen=" + std::to_string( flatlogs::logHeader::msgLen( buffer ) ) +
                             " totalSize=" + std::to_string( totalSize ) + " " );
            }
        }

        buffer += totalSize;
    }

    if( prior != nullptr )
    {
        DEBUG_CRUMB( "getPriorVerifiedLog found app=" + appName + " ev=" + std::to_string( ev ) +
                     " queryTs=" + logMapDebugTime( ts ) +
                     " logTs=" + logMapDebugTime( flatlogs::logHeader::timespec( prior ) ) + " " );
    }
    else
    {
        DEBUG_CRUMB( "getPriorVerifiedLog no match app=" + appName + " ev=" + std::to_string( ev ) +
                     " queryTs=" + logMapDebugTime( ts ) + " " );
    }

    return prior;
}

/// Find the next log entry that also passes flatbuffer schema verification.
template <class verboseT = XWC_DEFAULT_VERBOSITY>
char *getNextVerifiedLog( logMap<verboseT>  &lm,                 /**< [in] loaded log map to search */
                          char              *logCurrent,         /**< [in] entry before the desired verified entry */
                          const std::string &appName,            /**< [in] app/device name to search */
                          char              *logBefore = nullptr /**< [in] verified entry before logCurrent, if any */
)
{
    if( logCurrent == nullptr )
    {
        return nullptr;
    }

    flatlogs::eventCodeT ev = flatlogs::logHeader::eventCode( logCurrent );
    char                *candidate{ nullptr };
    char                *current{ logCurrent };
    struct rejectedLog
    {
        char *m_before{ nullptr };  ///< Entry immediately before the unverifiable entry in the search.
        char *m_failure{ nullptr }; ///< Unverifiable entry skipped during the search.
    };

    std::vector<rejectedLog> rejected;

    auto appBuffer = lm.m_appToBufferMap.find( appName );
    if( appBuffer != lm.m_appToBufferMap.end() && !verifyLogEntry( ev, logCurrent ) )
    {
        rejected.push_back( { logBefore, logCurrent } );
    }

    while( lm.getNextLog( candidate, current, appName ) == 0 )
    {
        if( candidate == nullptr )
        {
            return nullptr;
        }

        if( verifyLogEntry( ev, candidate ) )
        {
            if( appBuffer != lm.m_appToBufferMap.end() )
            {
                for( const rejectedLog &badLog : rejected )
                {
                    lm.recordRecoverableError( appName );
                    reportUnverifiableLogEntry( appName,
                                                ev,
                                                badLog.m_before,
                                                badLog.m_failure,
                                                candidate,
                                                appBuffer->second.sourceFile( badLog.m_failure ),
                                                appBuffer->second.sourceOffset( badLog.m_failure ),
                                                "next-verified-search" );
                }
            }

            DEBUG_CRUMB( "getNextVerifiedLog found app=" + appName + " ev=" + std::to_string( ev ) +
                         " currentTs=" + logMapDebugTime( flatlogs::logHeader::timespec( logCurrent ) ) +
                         " logTs=" + logMapDebugTime( flatlogs::logHeader::timespec( candidate ) ) + " " );
            return candidate;
        }

        DEBUG_CRUMB( "getNextVerifiedLog rejected app=" + appName + " ev=" + std::to_string( ev ) +
                     " logTs=" + logMapDebugTime( flatlogs::logHeader::timespec( candidate ) ) +
                     " msgLen=" + std::to_string( flatlogs::logHeader::msgLen( candidate ) ) +
                     " totalSize=" + std::to_string( flatlogs::logHeader::totalSize( candidate ) ) + " " );

        rejected.push_back( { current, candidate } );
        current = candidate;
    }

    if( appBuffer != lm.m_appToBufferMap.end() )
    {
        for( const rejectedLog &badLog : rejected )
        {
            lm.recordRecoverableError( appName );
            reportUnverifiableLogEntry( appName,
                                        ev,
                                        badLog.m_before,
                                        badLog.m_failure,
                                        nullptr,
                                        appBuffer->second.sourceFile( badLog.m_failure ),
                                        appBuffer->second.sourceOffset( badLog.m_failure ),
                                        "next-verified-search" );
        }
    }

    DEBUG_CRUMB( "getNextVerifiedLog no match app=" + appName + " ev=" + std::to_string( ev ) +
                 " currentTs=" + logMapDebugTime( flatlogs::logHeader::timespec( logCurrent ) ) + " " );

    return nullptr;
}

template <typename valT, class verboseT = XWC_DEFAULT_VERBOSITY>
int getLogStateVal( valT                      &val,
                    logMap<verboseT>          &lm,
                    const std::string         &appName,
                    flatlogs::eventCodeT       ev,
                    const flatlogs::timespecX &stime,
                    const flatlogs::timespecX &atime,
                    valT ( *getter )( void * ),
                    char       **hint          = 0,
                    double       maxGap        = -1,
                    std::string *failureReason = 0 )
{
    char *atprior = nullptr;
    char *stprior = nullptr;

    char *_hint = nullptr;

    if( hint )
        _hint = *hint;
    else
        _hint = 0;

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    if( lm.getPriorLog( stprior, appName, ev, stime, _hint ) != 0 )
    {
        return logMetaFail( failureReason, "due to missing prior telemetry" );
    }

    if( stprior == nullptr || flatlogs::logHeader::eventCode( stprior ) != ev )
    {
        return logMetaFail( failureReason, "due to missing prior telemetry" );
    }
    if( !verifyLogEntry( ev, stprior ) )
    {
        DEBUG_CRUMB( "getLogStateVal prior verify failed app=" + appName + " ev=" + std::to_string( ev ) +
                     " logTs=" + logMapDebugTime( flatlogs::logHeader::timespec( stprior ) ) + " " );
        stprior = getPriorVerifiedLog( lm, appName, ev, stime );
        if( stprior == nullptr )
        {
            return logMetaFail( failureReason, "due to unverifiable prior telemetry" );
        }
    }

    if( !logMetaGapValid( flatlogs::logHeader::timespec( stprior ), stime, maxGap ) )
    {
        return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
    }

    valT stprV = getter( flatlogs::logHeader::messageBuffer( stprior ) );

    valT atprV;

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    if( lm.getNextLog( atprior, stprior, appName ) != 0 )
    {
        if( !logMetaGapValid( flatlogs::logHeader::timespec( stprior ), atime, maxGap ) )
        {
            return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
        }

        return logMetaFail( failureReason, "due to missing following telemetry" );
    }
    if( atprior == nullptr )
    {
        if( logMetaGapValid( flatlogs::logHeader::timespec( stprior ), atime, maxGap ) )
        {
            val = stprV;
            if( hint )
                *hint = stprior;
            return 0;
        }

        return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
    }
    if( !verifyLogEntry( ev, atprior ) )
    {
        DEBUG_CRUMB( "getLogStateVal next verify failed app=" + appName + " ev=" + std::to_string( ev ) +
                     " logTs=" + logMapDebugTime( flatlogs::logHeader::timespec( atprior ) ) + " " );
        atprior = getNextVerifiedLog( lm, atprior, appName, stprior );
        if( atprior == nullptr )
        {
            return logMetaFail( failureReason, "due to unverifiable following telemetry" );
        }
    }

#ifdef DEBUG
    std::cerr << __FILE__ << " " << __LINE__ << "\n";
#endif

    while( flatlogs::logHeader::timespec( atprior ) < atime )
    {
        if( !logMetaGapValid(
                flatlogs::logHeader::timespec( stprior ), flatlogs::logHeader::timespec( atprior ), maxGap ) )
        {
            return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
        }

        atprV = getter( flatlogs::logHeader::messageBuffer( atprior ) );
        if( atprV != stprV )
        {
            val = atprV;
            if( hint )
                *hint = stprior;
            return 0;
        }
        stprior = atprior;
        if( lm.getNextLog( atprior, stprior, appName ) != 0 )
        {
            if( !logMetaGapValid( flatlogs::logHeader::timespec( stprior ), atime, maxGap ) )
            {
                return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
            }

            return logMetaFail( failureReason, "due to missing following telemetry" );
        }
        if( atprior == nullptr )
        {
            if( logMetaGapValid( flatlogs::logHeader::timespec( stprior ), atime, maxGap ) )
            {
                val = stprV;
                if( hint )
                    *hint = stprior;
                return 0;
            }

            return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
        }
        if( !verifyLogEntry( ev, atprior ) )
        {
            DEBUG_CRUMB( "getLogStateVal next verify failed app=" + appName + " ev=" + std::to_string( ev ) +
                         " logTs=" + logMapDebugTime( flatlogs::logHeader::timespec( atprior ) ) + " " );
            atprior = getNextVerifiedLog( lm, atprior, appName, stprior );
            if( atprior == nullptr )
            {
                return logMetaFail( failureReason, "due to unverifiable following telemetry" );
            }
        }
    }

    if( !logMetaGapValid( flatlogs::logHeader::timespec( stprior ), atime, maxGap ) )
    {
        return logMetaFail( failureReason, logMetaGapReason( maxGap ) );
    }

    val = stprV;

    if( hint )
        *hint = stprior;
    return 0;
}

template <typename valT, class verboseT = XWC_DEFAULT_VERBOSITY>
int getLogContVal( valT                      &val,
                   logMap<verboseT>          &lm,
                   const std::string         &appName,
                   flatlogs::eventCodeT       ev,
                   const flatlogs::timespecX &stime,
                   const flatlogs::timespecX &atime,
                   valT ( *getter )( void * ),
                   char       **hint          = 0,
                   double       maxGap        = -1,
                   std::string *failureReason = 0,
                   bool         isAngle       = false )
{
    char *atafter;
    char *stprior;

    char *_hint;
    if( hint )
        _hint = *hint;
    else
        _hint = 0;

    flatlogs::timespecX midexp = meanTimespecX( atime, stime );

    // Get log entry before midexp
    if( lm.getPriorLog( stprior, appName, ev, midexp, _hint ) != 0 )
    {
        return logMetaFail( failureReason, "due to missing prior telemetry", 1 );
    }

    if( stprior == nullptr || flatlogs::logHeader::eventCode( stprior ) != ev )
    {
        return logMetaFail( failureReason, "due to missing prior telemetry", 1 );
    }
    if( !verifyLogEntry( ev, stprior ) )
    {
        DEBUG_CRUMB( "getLogContVal prior verify failed app=" + appName + " ev=" + std::to_string( ev ) +
                     " logTs=" + logMapDebugTime( flatlogs::logHeader::timespec( stprior ) ) + " " );
        stprior = getPriorVerifiedLog( lm, appName, ev, midexp );
        if( stprior == nullptr )
        {
            return logMetaFail( failureReason, "due to unverifiable prior telemetry", 1 );
        }
    }

    if( !logMetaGapValid( flatlogs::logHeader::timespec( stprior ), midexp, maxGap ) )
    {
        return logMetaFail( failureReason, logMetaGapReason( maxGap ), 1 );
    }

    valT stprV = getter( flatlogs::logHeader::messageBuffer( stprior ) );

    // Get log entry after.
    if( lm.getNextLog( atafter, stprior, appName ) != 0 )
    {
#ifdef HARD_EXIT
        exit( -1 );
#endif
        return logMetaFail( failureReason, "due to missing following telemetry", 1 );
    }
    if( atafter == nullptr )
    {
        return logMetaFail( failureReason, "due to missing following telemetry", 1 );
    }
    if( !verifyLogEntry( ev, atafter ) )
    {
        DEBUG_CRUMB( "getLogContVal next verify failed app=" + appName + " ev=" + std::to_string( ev ) +
                     " logTs=" + logMapDebugTime( flatlogs::logHeader::timespec( atafter ) ) + " " );
        atafter = getNextVerifiedLog( lm, atafter, appName, stprior );
        if( atafter == nullptr )
        {
            return logMetaFail( failureReason, "due to unverifiable following telemetry", 1 );
        }
    }

    if( !logMetaGapValid( midexp, flatlogs::logHeader::timespec( atafter ), maxGap ) ||
        !logMetaGapValid( flatlogs::logHeader::timespec( stprior ), flatlogs::logHeader::timespec( atafter ), maxGap ) )
    {
        return logMetaFail( failureReason, logMetaGapReason( maxGap ), 1 );
    }

    valT atprV = getter( flatlogs::logHeader::messageBuffer( atafter ) );

    double st = flatlogs::logHeader::timespec( stprior ).asDouble();
    double it = midexp.asDouble();
    double et = flatlogs::logHeader::timespec( atafter ).asDouble();

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

    if( hint )
        *hint = stprior;

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

    char *m_hint{ nullptr }; ///< Cached log-search hint for repeated lookups of the same metadata item.

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
