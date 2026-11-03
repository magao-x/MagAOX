/** \file logsurgeon.hpp
 * \brief A utility to fix corrupted MagAO-X binary logs.
 *
 * \ingroup files
 */

#ifndef hpp
#define hpp

#include <iostream>
#include <cstring>
#include <set>
#include <utility>
#include <vector>

#include <mx/ioutils/fileUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
using namespace MagAOX::logger;

using namespace flatlogs;

/// Format a log timestamp for diagnostics.
inline std::string logsurgeonTimestamp( const timespecX &ts /**< [in] timestamp to format */ )
{
    timespecX tsCopy = ts;
    return tsCopy.ISO8601DateTimeStrX();
}

/// Store details about the first plausible-but-invalid log entry in a corrupt section.
struct logsurgeonCandidate
{
    bool        m_have{ false };       ///< True if a candidate header was found.
    ssize_t     m_start{ 0 };          ///< Candidate log-entry start byte in the source file.
    ssize_t     m_eventOffset{ 0 };    ///< Candidate event-code byte in the source file.
    ssize_t     m_end{ 0 };            ///< One past candidate end byte, as claimed by the header.
    logPrioT    m_level{ 0 };          ///< Candidate flatlog priority.
    eventCodeT  m_eventCode{ 0 };      ///< Candidate flatlog event code.
    bool        m_eventValid{ false }; ///< True if the candidate event code is recognized.
    timespecX   m_timestamp{ 0, 0 };   ///< Candidate flatlog timestamp.
    msgLenT     m_messageLength{ 0 };  ///< Candidate flatbuffer message length.
    size_t      m_headerSize{ 0 };     ///< Candidate flatlog header size.
    size_t      m_totalSize{ 0 };      ///< Candidate total flatlog entry size.
    std::string m_reason;              ///< Reason the candidate was rejected.

    /// Record a candidate header and rejection reason.
    /** The whole header must be in the buffer.
     */
    void set( char       *buffst,               /**< [in] pointer to the candidate flatlog entry */
              ssize_t     start,                /**< [in] candidate log-entry start byte in the source file */
              ssize_t     eventOffset,          /**< [in] candidate event-code byte in the source file */
              ssize_t     end,                  /**< [in] one past the candidate end byte */
              std::string reason,               /**< [in] reason the candidate was rejected */
              logVersionT version = logVersion1 /**< [in] the file's version */
    )
    {
        m_have        = true;
        m_start       = start;
        m_eventOffset = eventOffset;
        m_end         = end;
        if( version == logVersion2 )
        {
            m_level         = logHeaderV2::logLevel( buffst );
            m_eventCode     = logHeaderV2::eventCode( buffst );
            m_timestamp     = logHeaderV2::timespec( buffst );
            m_messageLength = logHeaderV2::msgLen( buffst );
            m_headerSize    = logHeaderV2::headerSize( buffst );
            m_totalSize     = logHeaderV2::totalSize( buffst );
        }
        else
        {
            m_level         = logHeaderV1::logLevel( buffst );
            m_eventCode     = logHeaderV1::eventCode( buffst );
            m_timestamp     = logHeaderV1::timespec( buffst );
            m_messageLength = logHeaderV1::msgLen( buffst );
            m_headerSize    = logHeaderV1::headerSize( buffst );
            m_totalSize     = logHeaderV1::totalSize( buffst );
        }
        m_eventValid = logCodeValid( m_eventCode );
        m_reason     = reason;
    }

    /// Record a candidate whose header could not be fully decoded.
    void setPartial( ssize_t     start,       /**< [in] candidate log-entry start byte in the source file */
                     ssize_t     eventOffset, /**< [in] candidate event-code byte in the source file */
                     eventCodeT  eventCode,   /**< [in] candidate flatlog event code */
                     ssize_t     end,         /**< [in] one past the candidate end byte */
                     std::string reason       /**< [in] reason the candidate was rejected */
    )
    {
        m_have        = true;
        m_start       = start;
        m_eventOffset = eventOffset;
        m_end         = end;
        m_eventCode   = eventCode;
        m_eventValid  = logCodeValid( m_eventCode );
        m_reason      = reason;
    }

    /// Clear the candidate details.
    void clear()
    {
        m_have = false;
    }
};

/// Check whether a buffer offset holds a decodable flatlog header.
inline bool readCandidateHeader( logsurgeonCandidate &candidate,            /**< [out] candidate header details */
                                 char                *buff,                 /**< [in] full log-file buffer */
                                 ssize_t              fsz,                  /**< [in] size of the log-file buffer */
                                 ssize_t              start,                /**< [in] candidate log-entry start byte */
                                 std::string          reason,               /**< [in] reason to store with the candidate */
                                 logVersionT          version = logVersion1 /**< [in] the file's version */
)
{
    // In v2 the first byte is the version.
    ssize_t prioOffset  = start + ( ( version == logVersion2 ) ? static_cast<ssize_t>( sizeof( logVersionT ) ) : 0 );
    ssize_t eventOffset = prioOffset + static_cast<ssize_t>( sizeof( logPrioT ) );
    if( start < 0 || eventOffset + static_cast<ssize_t>( sizeof( eventCodeT ) ) > fsz )
    {
        return false;
    }

    eventCodeT ec = *( (eventCodeT *)( &buff[eventOffset] ) );

    if( start + static_cast<ssize_t>( entryMinHeadSize( version ) ) > fsz )
    {
        candidate.setPartial( start, eventOffset, ec, fsz, "candidate header extends past EOF before message length" );
        return true;
    }

    char  *buffst     = &buff[start];
    size_t headerSize = entryHeaderSize( buffst, version );
    if( start + static_cast<ssize_t>( headerSize ) > fsz )
    {
        candidate.setPartial(
            start, eventOffset, ec, fsz, "candidate header extends past EOF before full length field" );
        return true;
    }

    ssize_t end = start + static_cast<ssize_t>( entryTotalSize( buffst, version ) );
    candidate.set( buffst, start, eventOffset, end, reason, version );
    return true;
}

/// Return true if a candidate is good.
/** For v1: a full in-file entry of a known type that passes flatbuffer verification.
 * For v2: a full in-file entry (of any type) whose checksum passes.
 */
inline bool verifyCandidateEntry( const logsurgeonCandidate &candidate, /**< [in] candidate header details */
                                  char                      *buff,      /**< [in] full log-file buffer */
                                  ssize_t                    fsz,       /**< [in] size of the log-file buffer */
                                  logVersionT version = logVersion1     /**< [in] the file's version */
)
{
    if( version == logVersion2 )
    {
        if( !candidate.m_have || candidate.m_start < 0 || candidate.m_start >= fsz )
        {
            return false;
        }

        return logHeaderV2::parse( &buff[candidate.m_start], fsz - candidate.m_start ).status ==
               entryStatus::checksumOk;
    }

    if( !candidate.m_have || !candidate.m_eventValid || candidate.m_totalSize == 0 || candidate.m_end > fsz ||
        candidate.m_end <= candidate.m_start ||
        candidate.m_totalSize > static_cast<size_t>( fsz - candidate.m_start ) ||
        candidate.m_messageLength > candidate.m_totalSize )
    {
        return false;
    }

    char *nbuff = (char *)::operator new( candidate.m_totalSize * sizeof( char ) );
    memcpy( nbuff, &buff[candidate.m_start], candidate.m_totalSize );

    bufferPtrT buffPtr = bufferPtrT( nbuff );
    return logVerify( candidate.m_eventCode, logHeaderV1::messageBuffer( buffPtr ), candidate.m_messageLength );
}

/// Print details about a plausible-but-invalid log entry.
inline void printCandidate( std::ostream              &os,       /**< [in,out] stream to print to */
                            const logsurgeonCandidate &candidate /**< [in] candidate details to print */
)
{
    if( !candidate.m_have )
    {
        os << "   Candidate: <none found>\n";
        return;
    }

    os << "   Candidate: start=" << candidate.m_start << " eventByte=" << candidate.m_eventOffset
       << " end=" << candidate.m_end << " level=" << static_cast<int>( candidate.m_level )
       << " code=" << candidate.m_eventCode << " ("
       << ( candidate.m_eventValid ? eventCodeName( candidate.m_eventCode ) : "invalid event code" ) << ")"
       << " timestamp=" << logsurgeonTimestamp( candidate.m_timestamp ) << " msgLen=" << candidate.m_messageLength
       << " headerSize=" << candidate.m_headerSize << " totalSize=" << candidate.m_totalSize
       << " reason=" << candidate.m_reason << "\n";
}

/// Check whether a stretch repair would delete is actually entries of an unknown type.
/** Evaluates chain of v1 entries.
 * From the start of a stretch, the chain counts if it has an unknown-type entry and either runs to
 * \p end or has at least three unknown-type entries.
 * From any later start point, the chain counts if it holds at least three entries, of which at 
 * least one is of an unknown type.
 *
 * \returns true if the chain counts, with the unknown codes added to \p codes.
 */
inline bool unknownTypeChainAt( std::set<eventCodeT> &codes,         /**< [in,out] the unknown codes found */
                                char                 *buff,          /**< [in] full log-file buffer */
                                ssize_t               start,         /**< [in] where the chain starts */
                                ssize_t               end,           /**< [in] one past the end of the stretch */
                                bool                  atStretchStart /**< [in] true if start is the stretch's start */
)
{
    std::set<eventCodeT> found;
    size_t               nPlausible = 0;
    size_t               nUnknown   = 0;
    bool                 havePrev   = false;
    timespecX            prevTs;

    ssize_t pos = start;
    while( pos < end )
    {
        parsedEntry pe = logHeaderV1::parse( &buff[pos], end - pos );
        if( pe.status != entryStatus::noChecksum || !logMapPriorityValid( pe.prio ) )
        {
            break;
        }

        timespecX ts = pe.ts;
        if( havePrev && ts < prevTs )
        {
            break;
        }
        havePrev = true;
        prevTs   = ts;

        ++nPlausible;
        if( !logCodeValid( pe.eventCode ) )
        {
            found.insert( pe.eventCode );
            ++nUnknown;
        }

        pos += pe.entrySize;
    }

    bool counts = false;
    if( atStretchStart )
    {
        counts = ( nUnknown > 0 && ( pos == end || nUnknown >= 3 ) );
    }
    else
    {
        counts = ( nUnknown > 0 && nPlausible >= 3 );
    }

    if( counts )
    {
        codes.insert( found.begin(), found.end() );
    }

    return counts;
}

/// Check whether a stretch repair would delete is actually entries of an unknown type.
/** Evaluates chain of v1 entries.
 * Unknown-type entries can be anywhere in the stretch, so try a chain from every byte of the 
 * stretch (see \ref unknownTypeChainAt).
 *
 * \returns true if the stretch holds such entries, with their codes added to \p codes.
 */
inline bool unknownTypeChain( std::set<eventCodeT> &codes, /**< [in,out] the unknown codes found */
                              char                 *buff,  /**< [in] full log-file buffer */
                              ssize_t               start, /**< [in] start of the stretch */
                              ssize_t               end    /**< [in] one past the end of the stretch */
)
{
    for( ssize_t pos = start; pos < end; ++pos )
    {
        if( unknownTypeChainAt( codes, buff, pos, end, pos == start ) )
        {
            return true;
        }
    }

    return false;
}

/** \defgroup logsurgeon logsurgeon: MagAO-X Log Corrector
 * \brief Read a MagAO-X binary log file and remove corrupted bytes.
 *
 * <a href="../handbook/utils/logsurgeon.html">Utility Documentation</a>
 *
 * \ingroup utils
 *
 */

/** \defgroup files logsurgeon Files
 * \ingroup logsurgeon
 */

/// An application to fix corrupted MagAO-X binary logs.
/** \todo document this
 *
 * \ingroup logsurgeon
 */
class logsurgeon : public mx::app::application
{
  protected:
    std::string m_fname; ///< The full path to the file to check

    bool m_checkOnly{ false }; /** If true then no modification to files on disk occurs, exit code
                                   0 indicates successful verification.  Default is false.*/

    bool m_chainMap{ false }; ///< If true, report length-based traversal diagnostics and exit.

    bool m_dropUnknown{ false }; ///< If true, repair may drop v1 stretches that read as entries of unknown types.

    logVersionT m_version{ logVersion1 }; ///< The format version of the file.

  public:
    enum returnVals
    {
        noerror            = 0,    ///< no errors occurred
        file_not_specified = -1,   ///< no file wa specified
        file_not_found     = -2,   ///< the file was found (or an error occurred opening it)
        errors_found       = -100, ///< errors were found in the file in checkOnly mode
        unknown_types      = -200, ///< repair refused, would drop unknown log types (exit status 56)
        error              = -9999 ///< other errors were found
    };

    virtual void setupConfig();

    virtual void loadConfig();

    virtual int execute();

    const std::string &fname();

    bool checkOnly();

    /// Report length-based traversal diagnostics for one file buffer.
    int mapChain( char   *buff, /**< [in] file buffer to scan */
                  ssize_t fsz   /**< [in] file buffer size */
    );
};

void logsurgeon::setupConfig()
{
    config.add( "file",
                "F",
                "file",
                argType::Required,
                "",
                "file",
                true,
                "string",
                "The single file to process.  If no / are found in name it will look in the specified "
                "directory (or MagAO-X default)." );

    config.add( "check",
                "",
                "",
                argType::Required,
                "",
                "check-only",
                false,
                "bool",
                "Check-only mode config file setting. If true then no modification to files on disk occurs, "
                "exit code 0 indicates successful verification.  Default is false." );

    config.add( "checkCL",
                "C",
                "check-only",
                argType::True,
                "",
                "",
                false,
                "bool",
                "Check-only mode command-line flag. If true then no modification to files on disk occurs, "
                "exit code 0 indicates successful verification. Overrides config file.  Default is false." );

    config.add( "chainMap",
                "M",
                "map-chain",
                argType::True,
                "",
                "",
                false,
                "bool",
                "Report length-based traversal diagnostics and exit without modifying files." );

    config.add( "dropUnknown",
                "",
                "drop-unknown-types",
                argType::True,
                "",
                "",
                false,
                "bool",
                "Allow repair to remove stretches of a v1 file that read as entries of unknown log types "
                "Without this, repair doesn't change the file, since those entries may be real data "
                "from a newer build, and exits with status 56." );
}

void logsurgeon::loadConfig()
{
    config( m_fname, "file" );
    config( m_checkOnly, "check" );

    // Command line always wins
    if( config.isSet( "checkCL" ) )
    {
        m_checkOnly = true;
    }

    if( config.isSet( "chainMap" ) )
    {
        m_chainMap = true;
    }

    if( config.isSet( "dropUnknown" ) )
    {
        m_dropUnknown = true;
    }
}

int logsurgeon::mapChain( char   *buff, /**< [in] file buffer to scan */
                          ssize_t fsz   /**< [in] file buffer size */
)
{
    ssize_t             pos{ 0 };
    size_t              index{ 0 };
    size_t              verified{ 0 };
    size_t              rejected{ 0 };
    size_t              rejectionRuns{ 0 };
    logsurgeonCandidate previous;
    logsurgeonCandidate runPrevious;
    logsurgeonCandidate runFirstRejected;
    logsurgeonCandidate runByteScanNext;
    size_t              runFirstIndex{ 0 };
    size_t              runRejected{ 0 };
    bool                inRejectedRun{ false };

    auto findByteScanNext = [&]( ssize_t start ) -> logsurgeonCandidate
    {
        logsurgeonCandidate byteScanNext;
        for( ssize_t scan = start + 1; scan < fsz; ++scan )
        {
            logsurgeonCandidate scanned;
            if( readCandidateHeader( scanned, buff, fsz, scan, "first later byte-scanned verified entry", m_version ) &&
                verifyCandidateEntry( scanned, buff, fsz, m_version ) )
            {
                byteScanNext = scanned;
                break;
            }
        }
        return byteScanNext;
    };

    auto printRun = [&]( const logsurgeonCandidate &resync, const std::string &resyncLabel )
    {
        ++rejectionRuns;
        std::cerr << "Rejected length-chain run index=" << runFirstIndex << " count=" << runRejected << "\n";
        std::cerr << "   Previous verified chain entry:\n";
        printCandidate( std::cerr, runPrevious );
        std::cerr << "   First rejected chain entry:\n";
        printCandidate( std::cerr, runFirstRejected );
        std::cerr << "   First byte-scanned verified entry:\n";
        printCandidate( std::cerr, runByteScanNext );
        if( runByteScanNext.m_have )
        {
            std::cerr << "   Byte-scan gap: " << runByteScanNext.m_start - runFirstRejected.m_start
                      << " bytes after first rejected start\n";
            if( runByteScanNext.m_start < runFirstRejected.m_end )
            {
                std::cerr << "   Byte-scan entry lies inside first rejected candidate's claimed extent by "
                          << runFirstRejected.m_end - runByteScanNext.m_start << " bytes\n";
            }
        }
        std::cerr << "   " << resyncLabel << ":\n";
        printCandidate( std::cerr, resync );
        std::cerr << "--------------------------------------------------------\n";
    };

    std::cerr << "Length-chain traversal map for " << m_fname << "\n";
    std::cerr << "--------------------------------------------------------\n";

    while( pos < fsz )
    {
        logsurgeonCandidate current;
        if( !readCandidateHeader( current, buff, fsz, pos, "length-chain entry", m_version ) )
        {
            std::cerr << "Chain stopped: no decodable header at start=" << pos << "\n";
            break;
        }

        bool ok = verifyCandidateEntry( current, buff, fsz, m_version );
        if( ok )
        {
            ++verified;

            if( inRejectedRun )
            {
                printRun( current, "Length-chain resynchronizes at verified entry" );
                inRejectedRun = false;
            }
        }
        else
        {
            ++rejected;
            if( !inRejectedRun )
            {
                inRejectedRun    = true;
                runFirstIndex    = index;
                runRejected      = 0;
                runPrevious      = previous;
                runFirstRejected = current;
                runByteScanNext  = findByteScanNext( current.m_start );
            }
            ++runRejected;
        }

        if( current.m_totalSize == 0 || current.m_end <= pos || current.m_end > fsz )
        {
            if( inRejectedRun )
            {
                logsurgeonCandidate stop;
                printRun( stop, "Length-chain does not reach a verified resync before stopping" );
                inRejectedRun = false;
            }
            std::cerr << "Chain stopped: invalid claimed extent at start=" << pos << " end=" << current.m_end << "\n";
            break;
        }

        previous = current;
        pos      = current.m_end;
        ++index;
    }

    if( inRejectedRun )
    {
        logsurgeonCandidate eof;
        printRun( eof, "Length-chain does not reach a verified resync before EOF" );
    }

    std::cerr << "Length-chain summary: entries=" << index << " verified=" << verified << " rejected=" << rejected
              << " rejectionRuns=" << rejectionRuns << " finalByte=" << pos << " fileBytes=" << fsz << "\n";

    if( pos == fsz )
    {
        std::cerr << "Length-chain lands exactly on EOF.\n";
    }
    else
    {
        std::cerr << "Length-chain does not land on EOF.\n";
    }

    return rejected == 0 && pos == fsz ? noerror : errors_found;
}

int logsurgeon::execute()
{
    if( m_fname == "" )
    {
        std::cerr << "Must specify filename with -F option.\n";
        return file_not_specified;
    }

    m_version = logVersionForFile( m_fname );

    FILE *fin;
    fin = fopen( m_fname.c_str(), "rb" );

    if( !fin )
    {
        std::cerr << "Error opening file " << m_fname << "\n";
        return file_not_found;
    }

    ssize_t fsz = mx::ioutils::fileSize( fin );

    char *buff = new char[fsz];

    ssize_t nrd = fread( buff, 1, fsz, fin );
    fclose( fin );

    if( nrd != fsz )
    {
        std::cerr << __FILE__ << " " << __LINE__ << " did not read complete file.\n";
        delete[] buff;

        return error;
    }

    if( m_chainMap )
    {
        int rv = mapChain( buff, fsz );
        delete[] buff;
        return rv;
    }

    ssize_t gcurr      = 0;
    bool    inbad      = false;
    ssize_t lastGoodSt = 0;
    ssize_t lastGoodSz = 0;

    ssize_t totBad = 0;
    ssize_t badSt  = 0;
    ssize_t kpt    = sizeof( logPrioT );

    logsurgeonCandidate badCandidate;

    char *gbuff = new char[fsz];

    // Stretches that repair would drop (v1 only; unknown-type entries).
    std::vector<std::pair<ssize_t, ssize_t>> gaps;

    if( m_version == logVersion1 )
    {
        // Now check each byte to see if it is a valid eventCode,
        // which makes it a potential start of a valid log
        while( kpt + static_cast<ssize_t>( sizeof( eventCodeT ) ) <= fsz )
        {
            eventCodeT ec = *( (eventCodeT *)( &buff[kpt] ) );

            if( logCodeValid( ec ) )
            {
                char *buffst = &buff[kpt - sizeof( logPrioT )];

                ssize_t candStart = kpt - static_cast<ssize_t>( sizeof( logPrioT ) );

                if( candStart + logHeaderV1::minHeadSize > fsz )
                {
                    if( !badCandidate.m_have )
                    {
                        badCandidate.setPartial(
                            candStart, kpt, ec, fsz, "candidate header extends past EOF before message length" );
                    }
                }
                else if( candStart + static_cast<ssize_t>( logHeaderV1::headerSize( buffst ) ) > fsz )
                {
                    // The longer length field itself runs past the end of the file.
                    if( !badCandidate.m_have )
                    {
                        badCandidate.setPartial(
                            candStart, kpt, ec, fsz, "candidate header extends past EOF before full length field" );
                    }
                }
                else
                {
                    msgLenT len    = logHeaderV1::msgLen( buffst );
                    msgLenT totLen = len + logHeaderV1::headerSize( buffst );

                    // Basic check if size isn't too big (i.e. would extend past end of file). Decoding compares the
                    // size with what is left of the file, so a huge length can't wrap the end position around.
                    ssize_t endpt = candStart + static_cast<ssize_t>( totLen );
                    if( logHeaderV1::parse( buffst, fsz - candStart ).status == entryStatus::noChecksum )
                    {
                        // Now we use the flatlogs verifier.
                        char *nbuff = (char *)::operator new( totLen * sizeof( char ) );

                        memcpy( nbuff, buffst, totLen );

                        bufferPtrT buffPtr = bufferPtrT( nbuff );

                        // true means good
                        if( logVerify( ec, logHeaderV1::messageBuffer( buffPtr ), len ) )
                        {
                            // if we pass we check if we're currently in a bad section
                            if( inbad )
                            {
                                // if we were in a bad section we record the end of the bad section
                                inbad              = false;
                                ssize_t nextGoodSt = kpt - static_cast<ssize_t>( sizeof( logPrioT ) );

                                char *lastGBuff = (char *)::operator new( lastGoodSz * sizeof( char ) );

                                memcpy( lastGBuff, &buff[lastGoodSt], lastGoodSz );
                                bufferPtrT lgBuffPtr = bufferPtrT( lastGBuff );

                                gaps.push_back( { lastGoodSt + lastGoodSz, nextGoodSt } );

                                std::cerr << "Found corrupt section: \n";
                                std::cerr << "   Before: ";
                                logStdFormat( std::cerr, logHeaderV1::parse( lgBuffPtr.get(), lastGoodSz ) );
                                std::cerr << "\n";

                                // printLogBuff(lglvl, lgec, logHeader::msgLen(lastGBuff), lgBuffPtr);

                                std::cerr << "   Corrupt: " << badSt << " - " << nextGoodSt << " ("
                                          << nextGoodSt - badSt << " bytes)\n";
                                printCandidate( std::cerr, badCandidate );
                                totBad += nextGoodSt - badSt;

                                std::cerr << "   After:  ";
                                logStdFormat( std::cerr, logHeaderV1::parse( buffPtr.get(), totLen ) );
                                std::cerr << "\n";
                                badCandidate.clear();
                            }

                            // It's good so we write it to the good buffer
                            memcpy( &gbuff[gcurr], &buff[kpt - sizeof( logPrioT )], totLen );

                            lastGoodSt = kpt - sizeof( logPrioT );
                            lastGoodSz = totLen;

                            gcurr += totLen;
                            kpt += totLen;

                            continue;
                        }
                        else if( !badCandidate.m_have )
                        {
                            badCandidate.set( buffst, candStart, kpt, endpt, "flatbuffer verification failed" );
                        }
                    }
                    else if( !badCandidate.m_have )
                    {
                        badCandidate.set( buffst, candStart, kpt, endpt, "candidate extends past EOF" );
                    }
                }
            }

            // If here the one of the checks has failed
            if( inbad == false )
            {
                // a new bad section has started
                badSt = badCandidate.m_have ? badCandidate.m_start : kpt;
                inbad = true;
            }

            ++kpt;
        }

        if( inbad )
        {
            gaps.push_back( { lastGoodSt + lastGoodSz, fsz } );

            std::cerr << "Found corrupt section: \n";

            if( lastGoodSz > 0 )
            {
                char *lastGBuff = (char *)::operator new( lastGoodSz * sizeof( char ) );

                memcpy( lastGBuff, &buff[lastGoodSt], lastGoodSz );
                bufferPtrT lgBuffPtr = bufferPtrT( lastGBuff );

                std::cerr << "   Before: ";
                logStdFormat( std::cerr, logHeaderV1::parse( lgBuffPtr.get(), lastGoodSz ) );
                std::cerr << "\n";
            }
            else
            {
                std::cerr << "   Before: <none found>\n";
            }

            std::cerr << "   Corrupt: " << badSt << " - " << fsz << " (" << fsz - badSt << " bytes)\n";
            printCandidate( std::cerr, badCandidate );
            totBad += fsz - badSt;

            std::cerr << "   After:  <none found before end-of-file>\n";
        }
    }
    else
    {
        // v2: after damage, carry on to the next entry that passes its checksum.
        ssize_t pos = 0;
        while( pos < fsz )
        {
            parsedEntry pe = logHeaderV2::parse( &buff[pos], fsz - pos );

            if( pe.status == entryStatus::checksumOk )
            {
                if( inbad )
                {
                    inbad = false;

                    std::cerr << "Found corrupt section: \n";
                    if( lastGoodSz > 0 )
                    {
                        std::cerr << "   Before: ";
                        logStdFormat( std::cerr, logHeaderV2::parse( &buff[lastGoodSt], lastGoodSz ) );
                        std::cerr << "\n";
                    }
                    else
                    {
                        std::cerr << "   Before: <none found>\n";
                    }

                    std::cerr << "   Corrupt: " << badSt << " - " << pos << " (" << pos - badSt << " bytes)\n";
                    printCandidate( std::cerr, badCandidate );
                    totBad += pos - badSt;

                    std::cerr << "   After:  ";
                    logStdFormat( std::cerr, pe );
                    std::cerr << "\n";
                    badCandidate.clear();
                }

                // The bytes are as written, so the entry is kept even if its message is of unknown type.
                if( logCodeValid( pe.eventCode ) && !logVerify( pe.eventCode, entryMessage( pe ), pe.msgLen ) )
                {
                    std::cerr << "Entry at byte " << pos << " (code=" << pe.eventCode
                              << ") passes its checksum but fails the message check; kept.\n";
                }

                memcpy( &gbuff[gcurr], &buff[pos], pe.entrySize );

                lastGoodSt = pos;
                lastGoodSz = pe.entrySize;

                gcurr += pe.entrySize;
                pos += pe.entrySize;

                continue;
            }

            if( !inbad )
            {
                inbad = true;
                badSt = pos;

                // In v2 the version byte and the priority come before the event code.
                ssize_t    eventOffset = pos + static_cast<ssize_t>( sizeof( logVersionT ) + sizeof( logPrioT ) );
                eventCodeT ec          = 0;
                if( eventOffset + static_cast<ssize_t>( sizeof( eventCodeT ) ) <= fsz )
                {
                    memcpy( &ec, &buff[eventOffset], sizeof( eventCodeT ) );
                }

                if( pe.status == entryStatus::markerMismatch )
                {
                    badCandidate.setPartial( pos, eventOffset, ec, pos, "version byte is not v2" );
                }
                else if( pe.headerSize == 0 )
                {
                    badCandidate.setPartial(
                        pos, eventOffset, ec, fsz, "candidate header extends past EOF before full length field" );
                }
                else if( pe.status == entryStatus::checksumMismatch )
                {
                    badCandidate.set(
                        &buff[pos], pos, eventOffset, pos + pe.entrySize, "checksum does not match", logVersion2 );
                }
                else
                {
                    badCandidate.set( &buff[pos],
                                      pos,
                                      eventOffset,
                                      pos + static_cast<ssize_t>( entryTotalSize( &buff[pos], logVersion2 ) ),
                                      "candidate extends past EOF",
                                      logVersion2 );
                }
            }

            size_t next = findNextEntryV2( buff, fsz, pos, pe );
            pos         = ( next == entryNotFound ) ? fsz : static_cast<ssize_t>( next );
        }

        if( inbad )
        {
            std::cerr << "Found corrupt section: \n";

            if( lastGoodSz > 0 )
            {
                std::cerr << "   Before: ";
                logStdFormat( std::cerr, logHeaderV2::parse( &buff[lastGoodSt], lastGoodSz ) );
                std::cerr << "\n";
            }
            else
            {
                std::cerr << "   Before: <none found>\n";
            }

            std::cerr << "   Corrupt: " << badSt << " - " << fsz << " (" << fsz - badSt << " bytes)\n";
            printCandidate( std::cerr, badCandidate );
            totBad += fsz - badSt;

            std::cerr << "   After:  <none found before end-of-file>\n";
        }
    }

    std::cerr << "--------------------------------------------------------\n";
    std::cerr << "Found " << totBad << " bad bytes ( " << ( 100.0 * totBad ) / fsz << "% bad) \n";
    std::cerr << "Found " << gcurr << " good bytes ( " << ( 100.0 * gcurr ) / fsz << "% good)\n";

    if( totBad == 0 )
    {
        std::cerr << "Taking no action on good file.\n";
    }
    else if( m_checkOnly )
    {
        std::cerr << "Check-only mode set, exiting with error status to indicate failed verification\n";
        delete[] buff;
        delete[] gbuff;
        return errors_found;
    }
    else
    {
        // Repair would drop stretches that read as entries of unkown log types.
        // This may be real data from a newer build, so refuse unless told otherwise.
        if( m_version == logVersion1 && !m_dropUnknown )
        {
            std::set<eventCodeT> codes;
            for( const auto &gap : gaps )
            {
                unknownTypeChain( codes, buff, gap.first, gap.second );
            }

            if( codes.size() > 0 )
            {
                std::cerr << "Repair refused: the stretches it would remove read as entries of log types this build "
                             "doesn't know (codes:";
                for( eventCodeT code : codes )
                {
                    std::cerr << " " << code;
                }
                std::cerr << ").  They are probably real data from a newer build.  No changes made.  Use "
                             "--drop-unknown-types to remove them anyway.\n";
                delete[] buff;
                delete[] gbuff;
                return unknown_types;
            }
        }

        std::string bupPath = m_fname + ".corrupted";

        FILE *fout;
        fout = fopen( bupPath.c_str(), "wb" );

        if( !fout )
        {
            std::cerr << "Error opening corrupted file for writing (" __FILE__ << " " << __LINE__ << ")\n";
            std::cerr << "No further action taken\n";
            delete[] buff;
            delete[] gbuff;
            return error;
        }

        ssize_t fwr = fwrite( buff, sizeof( char ), fsz, fout );

        int fcst = fclose( fout );

        if( fwr != fsz )
        {
            std::cerr << "Error writing backup corrupted file (" __FILE__ << " " << __LINE__ << ")\n";
            std::cerr << "No further action taken\n";
            delete[] buff;
            delete[] gbuff;
            return error;
        }

        if( fcst != 0 )
        {
            std::cerr << "Error closing backup corrupted file (" __FILE__ << " " << __LINE__ << ")\n";
            std::cerr << "No further action taken\n";
            delete[] buff;
            delete[] gbuff;
            return error;
        }

        std::cerr << "Wrote original file to: " << bupPath << "\n";

        fout = fopen( m_fname.c_str(), "wb" );

        if( !fout )
        {
            std::cerr << "Error opening existing file for writing (" __FILE__ << " " << __LINE__ << ")\n";
            std::cerr << "No further action taken\n";

            delete[] buff;
            delete[] gbuff;
            return error;
        }

        fwr = fwrite( gbuff, sizeof( char ), gcurr, fout );

        fcst = fclose( fout );

        if( fwr != gcurr )
        {
            std::cerr << "Error writing corrected file (" __FILE__ << " " << __LINE__ << ")\n";
            delete[] buff;
            delete[] gbuff;
            return error;
        }

        if( fcst != 0 )
        {
            std::cerr << "Error closing corrected file (" __FILE__ << " " << __LINE__ << ")\n";
            delete[] buff;
            delete[] gbuff;
            return error;
        }

        std::cerr << "Wrote corrected file to: " << m_fname << "\n";

        std::cerr << "Surgery Complete\n";
    }
    delete[] buff;
    delete[] gbuff;

    return noerror;
}

const std::string &logsurgeon::fname()
{
    return m_fname;
}

bool logsurgeon::checkOnly()
{
    return m_checkOnly;
}

#endif // hpp
