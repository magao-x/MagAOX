/** \file logMap.cpp
 * \brief Declares and defines the logMap class and related classes.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_files
 *
 */

#include "logMap.hpp"
// Test-only fault hooks. Every XWCTEST_IF_ macro expands to an empty statement unless
// a test defines the matching XWCTEST_ name before including this file.
#include "tests/testMacros.hpp"

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <filesystem>
#include <limits>

#include "../common/exceptions.hpp"

using namespace flatlogs;

namespace MagAOX
{
namespace logger
{

// Test-only. A test can define XWCTEST_NAMESPACE and compile this file a second time
// inside that namespace with one XWCTEST_ fault macro enabled. The faulted copy runs the
// real error handling code, and its hits count toward these same source lines.
// Production builds never define XWCTEST_NAMESPACE.
#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE
{
#endif

int logInMemory::loadFile( file::stdFileName<verboseT> const &lfn )
{
    DEBUG_CRUMB( "logInMemory loadFile begin file=" + lfn.fullName() );

    int fd = open( lfn.fullName().c_str(), O_RDONLY );

    off_t fsz = mx::ioutils::fileSize( fd );
    if( fsz <= 0 )
    {
        close( fd );
        std::cerr << "logInMemory::loadFile(" << lfn.fullName() << ") is empty\n";
        return -1;
    }

    std::vector<char> memory( fsz );

    ssize_t nrd = read( fd, memory.data(), memory.size() );

    close( fd );

    // Test hook. Pretends the read returned nothing so the short read check below fires.
    XWCTEST_IF_LOGINMEMORY_LOADFILE_SHORTREAD( nrd = 0 );

    if( nrd != fsz )
    {
        std::cerr << "logInMemory::loadFile(" << lfn.fullName() << ") did not read all bytes\n";
        return -1;
    }

    char *bufferStart = memory.data();
    char *bufferEnd   = memory.data() + memory.size();
    char *firstGood   = nullptr;
    char *lastGood    = nullptr;
    char *buffer      = bufferStart;

    while( buffer < bufferEnd )
    {
        size_t               totalSize = 0;
        flatlogs::timespecX  minTs;
        flatlogs::timespecX *minTsPtr = nullptr;
        if( lastGood != nullptr )
        {
            minTs    = logHeader::timespec( lastGood );
            minTsPtr = &minTs;
        }

        if( !logMapEntrySane( totalSize, buffer, bufferEnd, minTsPtr ) )
        {
            char *resynced = logMapResync( buffer, bufferEnd, minTsPtr );
            std::cerr << m_reportPrefix << "Invalid log entry skipped while loading log file: source=" << lfn.fullName()
                      << " sourceByte=" << buffer - bufferStart;
            m_recoverableErrors++;
            if( resynced != nullptr )
            {
                std::cerr << " resyncByte=" << resynced - bufferStart << " (" << resynced - buffer
                          << " byte resync span; corrupt section may begin earlier)\n";
                buffer = resynced;
                continue;
            }

            std::cerr << " resyncByte=<none> (" << bufferEnd - buffer
                      << " byte resync span to end-of-file; corrupt section may begin earlier)\n";
            break;
        }

        if( firstGood == nullptr )
        {
            firstGood = buffer;
        }
        lastGood = buffer;
        buffer += totalSize;
    }

    if( firstGood == nullptr || lastGood == nullptr )
    {
        std::cerr << "Possibly corrupt logfile.\n";
        return -1;
    }

    flatlogs::timespecX startTime = logHeader::timespec( firstGood );
    flatlogs::timespecX endTime   = logHeader::timespec( lastGood );

    DEBUG_CRUMB( "logInMemory loadFile read file=" + lfn.fullName() + " bytes=" + std::to_string( memory.size() ) +
                 " start=" + logMapDebugTime( startTime ) + " end=" + logMapDebugTime( endTime ) );

    if( m_memory.size() == 0 )
    {
        m_memory.swap( memory );
        m_startTime = startTime;
        m_endTime   = endTime;
        m_loadedFiles.push_back( { 0, m_memory.size(), lfn.fullName() } );

        DEBUG_CRUMB( "logInMemory loadFile initial file=" + lfn.fullName() +
                     " loadedStart=" + logMapDebugTime( m_startTime ) + " loadedEnd=" + logMapDebugTime( m_endTime ) +
                     " memoryBytes=" + std::to_string( m_memory.size() ) );

        return 0;
    }

    if( startTime < m_startTime )
    {

        if( endTime >= m_startTime )
        {
            std::cerr << "overlapping log files!\n";
            return -1;
        }

        m_memory.insert( m_memory.begin(), memory.begin(), memory.end() );
        for( loadedFile &loaded : m_loadedFiles )
        {
            loaded.m_begin += memory.size();
            loaded.m_end += memory.size();
        }
        m_loadedFiles.insert( m_loadedFiles.begin(), { 0, memory.size(), lfn.fullName() } );
        m_startTime = startTime;
        DEBUG_CRUMB( "logInMemory loadFile prepended file=" + lfn.fullName() +
                     " loadedStart=" + logMapDebugTime( m_startTime ) + " loadedEnd=" + logMapDebugTime( m_endTime ) +
                     " memoryBytes=" + std::to_string( m_memory.size() ) );
        return 0;
    }

    if( startTime > m_endTime )
    {
        size_t loadedBegin = m_memory.size();
        m_memory.insert( m_memory.end(), memory.begin(), memory.end() );
        m_loadedFiles.push_back( { loadedBegin, m_memory.size(), lfn.fullName() } );
        m_endTime = endTime;

        DEBUG_CRUMB( "logInMemory loadFile appended file=" + lfn.fullName() +
                     " loadedStart=" + logMapDebugTime( m_startTime ) + " loadedEnd=" + logMapDebugTime( m_endTime ) +
                     " memoryBytes=" + std::to_string( m_memory.size() ) );

        return 0;
    }

    std::cerr << "Need to implement insert in the middle!\n";
    std::cerr << m_startTime.time_s << " " << m_startTime.time_ns << "\n";
    std::cerr << startTime.time_s << " " << startTime.time_ns << "\n";

    return -1;
}

std::string logInMemory::sourceFile( char *log ) const
{
    if( log == nullptr || m_memory.empty() )
    {
        return "<unknown>";
    }

    const char *bufferStart = m_memory.data();
    const char *bufferEnd   = m_memory.data() + m_memory.size();
    if( log < bufferStart || log >= bufferEnd )
    {
        return "<outside-loaded-buffer>";
    }

    size_t offset = static_cast<size_t>( log - bufferStart );
    for( const loadedFile &loaded : m_loadedFiles )
    {
        if( offset >= loaded.m_begin && offset < loaded.m_end )
        {
            return loaded.m_name;
        }
    }

    return "<unknown-loaded-file>";
}

size_t logInMemory::sourceOffset( char *log ) const
{
    if( log == nullptr || m_memory.empty() )
    {
        return std::numeric_limits<size_t>::max();
    }

    const char *bufferStart = m_memory.data();
    const char *bufferEnd   = m_memory.data() + m_memory.size();
    if( log < bufferStart || log >= bufferEnd )
    {
        return std::numeric_limits<size_t>::max();
    }

    size_t offset = static_cast<size_t>( log - bufferStart );
    for( const loadedFile &loaded : m_loadedFiles )
    {
        if( offset >= loaded.m_begin && offset < loaded.m_end )
        {
            return offset - loaded.m_begin;
        }
    }

    return std::numeric_limits<size_t>::max();
}

#ifdef XWCTEST_NAMESPACE
} // namespace XWCTEST_NAMESPACE
#endif

// The explicit instantiation belongs to the production copy only. Each test namespace
// copy instantiates its own.
#ifndef XWCTEST_NAMESPACE
template class logMap<XWC_DEFAULT_VERBOSITY>;
#endif

} // namespace logger
} // namespace MagAOX
