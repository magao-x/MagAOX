/** \file logFileName.cpp
  * \brief Declares and defines the logFileName class
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_files
  * 
  */

#include "logFileName.hpp"

namespace MagAOX
{
namespace logger
{

logFileName::logFileName()
{
   return;
}

logFileName::logFileName(const std::string & fn) : m_fullName {fn}
{
   parseName();
}

int logFileName::fullName(const std::string & fn)
{
   m_fullName = fn;
   return parseName();
}
   
logFileName & logFileName::operator=(const std::string & fn)
{
   fullName(fn);
   
   return *this;
}
   
std::string logFileName::fullName() const
{
   return m_fullName;
}
      
std::string logFileName::baseName() const
{
   return m_baseName;
}

std::string logFileName::appName() const
{
   return m_appName;
}

int logFileName::year() const
{
   return m_year;
}

int logFileName::month() const
{
   return m_month;
}

int logFileName::day() const
{
   return m_day;
}

int logFileName::hour() const
{
   return m_hour;
}

int logFileName::minute() const
{
   return m_minute;
}

int logFileName::second() const
{
   return m_second;
}

int logFileName::nsec() const
{
   return m_nsec;
}

flatlogs::timespecX logFileName::timestamp() const
{
   return m_timestamp;
}

std::string logFileName::extension() const
{
   return m_extension;
}

bool logFileName::valid() const
{
   return m_valid;
}

int logFileName::parseName()
{
   m_baseName = mx::ioutils::pathFilename(m_fullName);

   size_t ext = m_fullName.rfind('.');
   
   if(ext == std::string::npos)
   {
      std::cerr << "No extension found in: " << m_fullName << "\n";
      m_valid = false;
      return -1;
   }
   
   m_extension = m_fullName.substr(ext+1);
   
   size_t ts = m_fullName.rfind('_', ext);
   
   if(ts == std::string::npos)
   {
      std::cerr << "No app name found in: " << m_fullName << "\n";
      m_valid = false;
      return -1;
   }
   
   size_t ps = m_fullName.rfind('/', ts);
   
   if(ps == std::string::npos) ps = 0;
   else ++ps;
   
   m_appName = m_fullName.substr(ps, ts-ps);
   
   ++ts;
   if(ext-ts != 23)
   {
      std::cerr << "Timestamp wrong size in: " << m_fullName << "\n";
      m_valid = false;
      return -1;
   }
      
   std::string tstamp = m_fullName.substr(ts, ext-ts);

   m_year = std::stoi(tstamp.substr(0,4));
   m_month = std::stoi(tstamp.substr(4,2));
   m_day = std::stoi(tstamp.substr(6,2));
   m_hour = std::stoi(tstamp.substr(8,2));
   m_minute = std::stoi(tstamp.substr(10,2));
   m_second = std::stoi(tstamp.substr(12,2));
   m_nsec = std::stoi(tstamp.substr(14,9));
 
   tm tmst;
   tmst.tm_year = m_year-1900;
   tmst.tm_mon = m_month - 1;
   tmst.tm_mday = m_day;
   tmst.tm_hour = m_hour;
   tmst.tm_min = m_minute;
   tmst.tm_sec = m_second;
   
   m_timestamp.time_s = timegm(&tmst);
   m_timestamp.time_ns = m_nsec;
   
   m_valid = true;
   
   return 0;
}



} //namespace logger
} //namespace MagAOX

