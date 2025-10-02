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

#include <mx/mxException.hpp>

#include "../common/exceptions.hpp"

using namespace flatlogs;

namespace MagAOX
{
namespace logger
{

int logInMemory::loadFile( file::stdFileName<verboseT> const &lfn )
{
    int fd = open( lfn.fullName().c_str(), O_RDONLY );

    off_t fsz = mx::ioutils::fileSize( fd );

    std::vector<char> memory( fsz );

    ssize_t nrd = read( fd, memory.data(), memory.size() );

    close( fd );

    if( nrd != fsz )
    {
        std::cerr << __FILE__ << " " << __LINE__ << " logInMemory::loadFile(" << lfn.fullName()
                  << ") did not read all bytes\n";
        return -1;
    }

    flatlogs::timespecX startTime = logHeader::timespec( memory.data() );

    size_t st = 0;
    size_t ed = logHeader::totalSize( memory.data() );
    st        = ed;

    while( st < memory.size() )
    {
        ed = logHeader::totalSize( memory.data() + st );
        st = st + ed;
    }

    if( st != memory.size() )
    {
        std::cerr << __FILE__ << " " << __LINE__ << " Possibly corrupt logfile.\n";
        return -1;
    }

    st -= ed;

    flatlogs::timespecX endTime = logHeader::timespec( memory.data() + st );

    if( m_memory.size() == 0 )
    {
        m_memory.swap( memory );
        m_startTime = startTime;
        m_endTime   = endTime;

        std::string timestamp;
        timespec    ts{ endTime.time_s, endTime.time_ns };
        mx::sys::timeStamp( timestamp, ts );

#ifdef DEBUG
        std::cerr << __FILE__ << " " << __LINE__ << " loading: " << lfn.fullName() << " " << timestamp << "\n";
#endif

        return 0;
    }

    if( startTime < m_startTime )
    {

        if( endTime >= m_startTime )
        {
            std::cerr << __FILE__ << " " << __LINE__ << " overlapping log files!\n";
            return -1;
        }

        m_memory.insert( m_memory.begin(), memory.begin(), memory.end() );
        m_startTime = startTime;
        std::cerr << __FILE__ << " " << __LINE__ << " added before!\n";
        return 0;
    }

    if( startTime > m_endTime )
    {
#ifdef DEBUG
        std::cerr << __FILE__ << " " << __LINE__ << " gonna append\n";
#endif

        // m_memory.insert( m_memory.end(), memory.begin(), memory.end() );
        m_endTime = endTime;

#ifdef DEBUG
        std::cerr << __FILE__ << " " << __LINE__ << " added after!\n";
#endif

        return 0;
    }

    std::cerr << __FILE__ << " " << __LINE__ << " Need to implement insert in the middle!\n";
    std::cerr << m_startTime.time_s << " " << m_startTime.time_ns << "\n";
    std::cerr << startTime.time_s << " " << startTime.time_ns << "\n";

    return -1;
}


template class logMap<XWC_DEFAULT_VERBOSITY>;

} // namespace logger
} // namespace MagAOX
