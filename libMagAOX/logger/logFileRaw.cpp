/** \file logFileRaw.cpp
  * \brief Manage a raw log file.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_files
  * 
  */

#include <cstring>
#include "logFileRaw.hpp"

namespace MagAOX
{
namespace logger
{

logFileRaw::logFileRaw()
{
}

logFileRaw::~logFileRaw()
{
   close();
}

int logFileRaw::logPath( const std::string & newPath)
{
   m_logPath = newPath;
   return 0;
}

std::string logFileRaw::logPath()
{
   return m_logPath;
}

int logFileRaw::logName( const std::string & newName)
{
   m_logName = newName;
   return 0;
}

std::string logFileRaw::logName()
{
   return m_logName;
}

int logFileRaw::logExt( const std::string & newExt)
{
   m_logExt = newExt;
   return 0;
}

std::string logFileRaw::logExt()
{
   return m_logExt;
}

int logFileRaw::maxLogSize( size_t newMaxFileSize )
{
   m_maxLogSize = newMaxFileSize;
   return 0;
}

size_t logFileRaw::maxLogSize()
{
   return m_maxLogSize;
}

int logFileRaw::writeLog( flatlogs::bufferPtrT & data )
{
   size_t N = flatlogs::logHeader::totalSize(data);

   //Check if we need a new file
   if(m_currFileSize + N > m_maxLogSize || m_fout == 0)
   {
      flatlogs::timespecX ts = flatlogs::logHeader::timespec(data);
      if( createFile(ts) < 0 ) return -1;
   }

   size_t nwr = fwrite( data.get(), sizeof(char), N, m_fout);

   if(nwr != N*sizeof(char))
   {
      std::cerr << "logFileRaw::writeLog: Error by fwrite.  At: " << __FILE__ << " " << __LINE__ << "\n";
      std::cerr << "logFileRaw::writeLog: errno says: " << strerror(errno) << "\n";
      return -1;
   }

   m_currFileSize += N;

   return 0;
}

int logFileRaw::flush()
{
   ///\todo this probably should be fsync, with appropriate error handling (see fsyncgate)
   if(m_fout) fflush(m_fout);

   return 0;
}

int logFileRaw::close()
{
   if(m_fout) fclose(m_fout);

   return 0;
}

int logFileRaw::createFile(flatlogs::timespecX & ts)
{
   std::string tstamp = ts.timeStamp();

   //Create the standard log name
   std::string fname = m_logPath + "/" + m_logName + "_" + tstamp + "." + m_logExt;

   if(m_fout) fclose(m_fout);

   errno = 0;
   ///\todo handle case where file exists (only if another instance tries at same ns -- pathological)
   m_fout = fopen(fname.c_str(), "wb");

   if(m_fout == 0)
   {
      std::cerr << "logFileRaw::createFile: Error by fopen. At: " << __FILE__ << " " << __LINE__ << "\n";
      std::cerr << "logFileRaw::createFile: errno says: " << strerror(errno) << "\n";
      std::cerr << "logFileRaw::createFile: fname = " << fname << "\n";
      return -1;
   }

   //Reset counters.
   m_currFileSize = 0;

   return 0;
}

} //namespace logger
} //namespace MagAOX

