/** \file logFileRaw.hpp
  * \brief Manage a raw log file.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_files
  * 
  * History:
  * - 2017-08-28 created by JRM
  */

#ifndef logger_logFileRaw_hpp
#define logger_logFileRaw_hpp


#include <iostream>

#include <string>


#include <mx/ioutils/stringUtils.hpp>

#include "../common/defaults.hpp"
#include <flatlogs/flatlogs.hpp>

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
   std::string m_logPath {"."}; ///< The base path for the log files.
   std::string m_logName {"xlog"}; ///< The base name for the log files.
   std::string m_logExt {MAGAOX_default_logExt}; ///< The extension for the log files.

   size_t m_maxLogSize {MAGAOX_default_max_logSize}; ///< The maximum file size in bytes. Default is 10 MB.
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
   int logPath( const std::string & newPath /**< [in] the new value of _path */ );

   /// Get the path.
   /**
     * \returns the current value of m_logPath.
     */
   std::string logPath();

   /// Set the log name
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */
   int logName( const std::string & newName /**< [in] the new value of m_logName */ );

   /// Get the name
   /**
     * \returns the current value of _name.
     */
   std::string logName();

   /// Set the log extension
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */
   int logExt( const std::string & newExt /**< [in] the new value of m_logExt */ );

   /// Get the log extension
   /**
     * \returns the current value of m_logExt.
     */
   std::string logExt();

   /// Set the maximum file size
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */
   int maxLogSize( size_t newMaxFileSize/**< [in] the new value of _maxLogSize */);

   /// Get the maximum file size
   /**
     * \returns the current value of m_maxLogSize
     */
   size_t maxLogSize();

   ///Write a log entry to the file
   /** Checks if this write will exceed m_maxLogSize, and if so opens a new file.
     * The new file will have the timestamp of this log entry.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
   int writeLog( flatlogs::bufferPtrT & data ///< [in] the log entry to write to disk
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
   int createFile(flatlogs::timespecX & ts /**< [in] A MagAOX timespec, used to set the timestamp */);


};



} //namespace logger
} //namespace MagAOX

#endif //logger_logFileRaw_hpp
