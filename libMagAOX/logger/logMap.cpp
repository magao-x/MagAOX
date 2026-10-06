/** \file logMap.cpp
 * \brief Declares and defines the logMap class and related classes.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_files
 *
 */

#include "logMap.hpp"

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <filesystem>

#include "../common/exceptions.hpp"

using namespace flatlogs;

namespace MagAOX
{
namespace logger
{

int logInMemory::loadFile( file::stdFileName<verboseT> const &lfn )
{
    DEBUG_CRUMB( "logInMemory loadFile begin file=" + lfn.fullName() );

    if( m_rejectedFiles.count( lfn.fullName() ) > 0 )
    {
        DEBUG_CRUMB( "logInMemory loadFile previously rejected file=" + lfn.fullName() );
        return -1;
    }

    flatlogs::logVersionT version = logVersionForExtension( lfn.extension() );

    int fd = open( lfn.fullName().c_str(), O_RDONLY );

    off_t fsz = mx::ioutils::fileSize( fd );
    if( fsz <= 0 )
    {
        close( fd );
        std::cerr << "logInMemory::loadFile(" << lfn.fullName() << ") is empty\n";
        return rejectFile( lfn );
    }

    std::vector<char> memory( fsz );

    ssize_t nrd = read( fd, memory.data(), memory.size() );

    close( fd );

    if( nrd != fsz )
    {
        std::cerr << "logInMemory::loadFile(" << lfn.fullName() << ") did not read all bytes\n";
        return rejectFile( lfn );
    }

    // A file named as v2 whose first entry isn't v2 is skipped.
    if( version != flatlogs::logVersion1 )
    {
        logStep first;
        logMapStepEntry( first, memory.data(), memory.size(), 0, version, nullptr );
        if( first.m_entry.status == flatlogs::entryStatus::markerMismatch )
        {
            std::cerr << m_reportPrefix << "Log file skipped: its extension says v" << static_cast<int>( version )
                      << " but its first entry is not v" << static_cast<int>( version ) << ": source=" << lfn.fullName()
                      << "\n";
            m_recoverableErrors++;
            return rejectFile( lfn );
        }
    }

    logFileWalk walk( memory.data(), memory.size(), version );

    bool haveGood = false;
    flatlogs::timespecX startTime;
    flatlogs::timespecX endTime;
    std::map<flatlogs::eventCodeT, size_t> unsupported; // count of entries of each unknown event code

    logStep step;
    size_t at = 0;
    size_t resumeAt = 0;
    logWalkEvent event;
    while( ( event = walk.next( step, at, resumeAt ) ) != logWalkEvent::end )
    {
        if( event == logWalkEvent::damage )
        {
            std::cerr << m_reportPrefix << "Invalid log entry skipped while loading log file: source=" << lfn.fullName()
                      << " sourceByte=" << at << " reason=" << logIntegrityName( step.m_integrity );
            m_recoverableErrors++;
            if( resumeAt != flatlogs::entryNotFound )
            {
                std::cerr << " resyncByte=" << resumeAt << " (" << resumeAt - at
                          << " byte resync span; corrupt section may begin earlier)\n";
                continue;
            }

            std::cerr << " resyncByte=<none> (" << memory.size() - at
                      << " byte resync span to end-of-file; corrupt section may begin earlier)\n";
            break;
        }

        if( !haveGood )
        {
            startTime = step.m_entry.ts;
            haveGood  = true;
        }
        endTime = step.m_entry.ts;

        if( !logMapEventCodeKnown( step.m_entry.eventCode ) )
        {
            ++unsupported[step.m_entry.eventCode];
        }
    }

    if( !haveGood )
    {
        std::cerr << m_reportPrefix << "Possibly corrupt logfile: source=" << lfn.fullName() << "\n";
        return rejectFile( lfn );
    }

    for( const auto &code : unsupported )
    {
        std::cerr << m_reportPrefix
                  << "Log entries of a type this build doesn't know were kept but can't be decoded: source="
                  << lfn.fullName() << " eventCode=" << code.first << " count=" << code.second << "\n";
    }

    DEBUG_CRUMB( "logInMemory loadFile read file=" + lfn.fullName() + " bytes=" + std::to_string( memory.size() ) +
                 " start=" + logMapDebugTime( startTime ) + " end=" + logMapDebugTime( endTime ) );

    if( m_memory.size() == 0 )
    {
        m_memory.swap( memory );
        m_startTime = startTime;
        m_endTime = endTime;
        m_loadedFiles.push_back( { 0, m_memory.size(), lfn.fullName(), version } );

        DEBUG_CRUMB( "logInMemory loadFile initial file=" + lfn.fullName() +
                     " loadedStart=" + logMapDebugTime( m_startTime ) + " loadedEnd=" + logMapDebugTime( m_endTime ) +
                     " memoryBytes=" + std::to_string( m_memory.size() ) );

        return 0;
    }

    if( startTime < m_startTime )
    {

        if( endTime >= m_startTime )
        {
            std::cerr << m_reportPrefix << "overlapping log files! source=" << lfn.fullName() << "\n";
            return rejectFile( lfn );
        }

        m_memory.insert( m_memory.begin(), memory.begin(), memory.end() );
        for( loadedFile &loaded : m_loadedFiles )
        {
            loaded.m_begin += memory.size();
            loaded.m_end += memory.size();
        }
        m_loadedFiles.insert( m_loadedFiles.begin(), { 0, memory.size(), lfn.fullName(), version } );
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
        m_loadedFiles.push_back( { loadedBegin, m_memory.size(), lfn.fullName(), version } );
        m_endTime = endTime;

        DEBUG_CRUMB( "logInMemory loadFile appended file=" + lfn.fullName() +
                     " loadedStart=" + logMapDebugTime( m_startTime ) + " loadedEnd=" + logMapDebugTime( m_endTime ) +
                     " memoryBytes=" + std::to_string( m_memory.size() ) );

        return 0;
    }

    std::cerr << m_reportPrefix << "Need to implement insert in the middle! source=" << lfn.fullName() << "\n";
    std::cerr << m_startTime.time_s << " " << m_startTime.time_ns << "\n";
    std::cerr << startTime.time_s << " " << startTime.time_ns << "\n";

    return rejectFile( lfn );
}

int logInMemory::rejectFile( file::stdFileName<verboseT> const &lfn )
{
    m_rejectedFiles.insert( lfn.fullName() );
    return -1;
}

logFileWalk logInMemory::fileWalk( size_t file, size_t offset ) const
{
    if( file >= m_loadedFiles.size() )
    {
        return logFileWalk();
    }

    const loadedFile &loaded = m_loadedFiles[file];

    return logFileWalk( m_memory.data() + loaded.m_begin, loaded.m_end - loaded.m_begin, loaded.m_version, offset );
}

logMemoryWalk::logMemoryWalk( const logInMemory &lim, size_t file, size_t offset )
    : m_lim( &lim ), m_file( file ), m_walk( lim.fileWalk( file, offset ) )
{
}

bool logMemoryWalk::next( logStep &step, size_t &file, size_t &offset )
{
    while( m_file < m_lim->m_loadedFiles.size() )
    {
        size_t at = 0;
        size_t resumeAt = 0;
        logWalkEvent event = m_walk.next( step, at, resumeAt );

        if( event == logWalkEvent::entry )
        {
            file = m_file;
            offset = at;
            return true;
        }

        if( event == logWalkEvent::damage )
        {
            continue;
        }

        // End of this file: continue with the beginning of the next one.
        ++m_file;
        m_walk = m_lim->fileWalk( m_file );
    }

    return false;
}

logEntryRecord logInMemory::record( size_t file, size_t offset, const logStep &step ) const
{
    logEntryRecord rec;

    if( file >= m_loadedFiles.size() || !step.usable() )
    {
        return rec;
    }

    const loadedFile &loaded = m_loadedFiles[file];
    const char *entry  = m_memory.data() + loaded.m_begin + offset;

    // Pad in front so that the message starts on an 8-byte boundary.
    size_t pad = ( 8 - step.m_entry.headerSize % 8 ) % 8;
    rec.m_bytes.assign( pad, 0 );
    rec.m_bytes.insert( rec.m_bytes.end(), entry, entry + step.m_entry.entrySize );
    rec.m_entrySize = step.m_entry.entrySize;
    rec.m_messageOffset = pad + step.m_entry.headerSize;
    rec.m_version = step.m_entry.version;
    rec.m_prio = step.m_entry.prio;
    rec.m_eventCode = step.m_entry.eventCode;
    rec.m_ts = step.m_entry.ts;
    rec.m_msgLen = step.m_entry.msgLen;
    rec.m_integrity = step.m_integrity;
    rec.m_unsupportedType = !logMapEventCodeKnown( step.m_entry.eventCode );
    rec.m_sourceFile = loaded.m_name;
    rec.m_sourceOffset = offset;

    return rec;
}

size_t logInMemory::findFile( const std::string &name ) const
{
    for( size_t n = 0; n < m_loadedFiles.size(); ++n )
    {
        if( m_loadedFiles[n].m_name == name )
        {
            return n;
        }
    }

    return m_loadedFiles.size();
}

template class logMap<XWC_DEFAULT_VERBOSITY>;

} // namespace logger
} // namespace MagAOX
