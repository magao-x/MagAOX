/** \file logFileRaw.hpp 
  * \brief Manage a raw log file.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2017-08-28 created by JRM
  */ 

#ifndef logger_logFileRaw_hpp
#define logger_logFileRaw_hpp


#include <iostream>

#include <string>


#include "/home/jrmales/Source/c/mxlib/include/stringUtils.hpp"


#include "../time/timespecX.hpp"
#include "logBuffer.hpp"

namespace MagAOX
{
namespace logger
{

/// A class to manage raw binary log files
/** Manages a binary file containing MagAO-X logs.
  *
  * The log entries are written as a binary stream of a configurable
  * maximum size.  If this size will be exceed by the next entry, then a new file is created.
  *
  * Filenames have a standard form of: [path]/[name]_YYYYMMDDHHMMSSNNNNNNNNN.[ext] where fields in [] are configurable.
  * 
  * The timestamp is from the first entry of the file.
  * 
  */
class logFileRaw
{
   
protected:
   
   /** \name Configurable Parameters
     *@{
     */ 
   std::string m_path {"."}; ///< The base path for the log files.
   std::string m_name {"xlog"}; ///< The base name for the log files.
   std::string m_ext {"rawlog"}; ///< The extension for the log files.

   size_t m_maxFileSize {10485760}; ///< The maximum file size in bytes. Default is 10 MB.
   ///@}
   
   /** \name Internal State
     *@{
     */ 
   
   FILE * m_fout {0}; ///< The file pointer
   
   size_t m_currFileSize {0}; ///< The current file size.
  
   ///@}
   
public:
   
   /// Default constructor
   /** Currently does nothing.
     */
   logFileRaw();

   ///Destructor
   /** Closes the file if open
     */
   ~logFileRaw();
   
   /// Set the path.
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int path( const std::string & newPath /**< [in] the new value of _path */ );
   
   /// Get the path.
   /**
     * \returns the current value of _path.
     */
   std::string path();
   
   /// Set the log name
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int name( const std::string & newName /**< [in] the new value of _name */ );
   
   /// Get the name
   /**
     * \returns the current value of _name. 
     */
   std::string name();
   
   /// Set the maximum file size
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int maxFileSize( size_t newMaxFileSize/**< [in] the new value of _maxFileSize */);
   
   /// Get the maximum file size
   /**
     * \returns the current value of _maxFileSize
     */
   size_t maxFileSize();

   ///Write a log entry to the file
   /** Checks if this write will exceed _maxFileSize, and if so opens a new file.
     * The new file will have the timestamp of this log entry.
     * 
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int writeLog( bufferPtrT & data ///< [in] the log entry to write to disk 
               );

   /// Flush the stream
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int flush();
   
   ///Close the file pointer
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int close();

protected:
   
   ///Create a new file
   /** Closes the current file if open.  Then creates a new file with a name of the form 
     * [path]/[name]_YYYYMMDDHHMMSSNNNNNNNNN.[ext]
     * 
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int createFile(time::timespecX & ts /**< [in] A MagAOX timespec, used to set the timestamp */);
   

};


inline
logFileRaw::logFileRaw()
{
}

inline
logFileRaw::~logFileRaw()
{
   close();
}

inline
int logFileRaw::path( const std::string & newPath)
{
   m_path = newPath;
   return 0;
}

inline
std::string logFileRaw::path()
{
   return m_path;
}

inline
int logFileRaw::name( const std::string & newName)
{
   m_name = newName;
   return 0;
}

inline
std::string logFileRaw::name()
{
   return m_name;
}

inline
int logFileRaw::maxFileSize( size_t newMaxFileSize )
{
   m_maxFileSize = newMaxFileSize;
   return 0;
}

inline
size_t logFileRaw::maxFileSize()
{
   return m_maxFileSize;
}   

inline
int logFileRaw::writeLog( bufferPtrT & data )
{
   msgLenT len = msgLen(data);          
   size_t N = headerSize + len;
            
   //Check if we need a new file
   if(m_currFileSize + N > m_maxFileSize || m_fout == 0)
   {
      time::timespecX ts = timespecX(data);
      createFile(ts);
   }
      
   int nwr = fwrite( data.get(), sizeof(char), N, m_fout);

   if(nwr != N*sizeof(char))
   {
      std::cerr << "logFileRaw::writeLog: Error by fwrite.  At: " << __FILE__ << " " << __LINE__ << "\n";
      return -1;
   }
   
   m_currFileSize += N;
   
   return 0;
}

inline
int logFileRaw::flush()
{
   if(m_fout) fflush(m_fout);
   
   return 0;
}

inline
int logFileRaw::close()
{
   if(m_fout) fclose(m_fout);
   
   return 0;
}

   
inline
int logFileRaw::createFile(time::timespecX & ts)
{
   std::string tstamp = ts.timeStamp();

   //Create the standard log name
   std::string fname = m_path + "/" + m_name + "_" + tstamp + "." + m_ext;

   if(m_fout) fclose(m_fout);
   
   ///\todo handle case where file exists (only if another instance tries at same ns -- pathological)
   m_fout = fopen(fname.c_str(), "wb");
   
   if(m_fout == 0)
   {
      std::cerr << "logFileRaw::createFile: Error by fopen.  At: " << __FILE__ << " " << __LINE__ << "\n";
      return -1;
   }
   
   //Reset counters.
   m_currFileSize = 0;
   
   return 0;
}

} //namespace logger 
} //namespace MagAOX 

#endif //logger_logFileRaw_hpp
