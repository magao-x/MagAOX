/** \file logManager.hpp
  * \brief The MagAO-X log manager.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_files
  * 
  * History:
  * - 2017-06-27 created by JRM
  */

#ifndef logger_logManager_hpp
#define logger_logManager_hpp

#include <memory>
#include <list>

#include <thread>

#include <mutex>
#include <ratio>

#include <mx/app/appConfigurator.hpp>

#include <flatlogs/flatlogs.hpp>

using namespace flatlogs;

#include "../common/defaults.hpp"

#include "generated/logTypes.hpp"
#include "generated/logStdFormat.hpp"

namespace MagAOX
{
namespace logger
{

/// The standard MagAOX log manager, used for both process logs and telemetry streams.
/** Manages the formatting and queueing of the log entries.
  *
  * A log entry is made using one of the standard log types.  These are formatted into a binary stream and
  * stored in a std::list.  This occurs in the calling thread.  The insertion into the list is mutex-ed, so
  * it is safe to make logs from different threads concurrently.
  *
  * Write-to-disk occurs in a separate thread, which
  * is normally set to the lowest priority so as not to interfere with higher-priority tasks.  The
  * log thread cycles through pending log entries in the list, dispatching them to the logFile.
  *
  * The template parameter logFileT is one of the logFile types, which is used to actually write to disk.
  *
  * Example:
    \code
    logManager<logFileRaw> log;
    log.m_logThreadStart();

    //do stuff

    log<loop_closed>(); //An empty log, which is used to log discrete events.

    //more stuff happens

    log<text_log>("a log entry with a std::string message type");

    \endcode
  *
  * \todo document all the requirements of logFileT
  *
  * \tparam logFileT a logFile type with a writeLog method.
  *
  * \ingroup logger
  */
template<class _parentT, class _logFileT>
struct logManager : public _logFileT
{
   typedef _parentT parentT;
   typedef _logFileT logFileT;
   
protected:
   parentT * m_parent {nullptr};
   
   ///\todo Make these protected members, with appropriate access methods
   //-->
public:
   std::string m_configSection {"logger"}; ///<The configuration files section name.  Default is `logger`.
   
protected:
   std::list<bufferPtrT> m_logQueue; ///< Log entries are stored here, and writen to the file by the log thread.

   std::thread m_logThread; ///< A separate thread for actually writing to the file.
   std::mutex m_qMutex; ///< Mutex for accessing the m_logQueue.

   bool m_logShutdown {false}; ///< Flag to signal the log thread to shutdown.

   unsigned long m_writePause {MAGAOX_default_writePause}; ///< Time, in nanoseconds, to pause between successive batch writes to the file. Default is 1e9. Configure with logger.writePause.

public:
   logPrioT m_logLevel {logPrio::LOG_INFO}; ///< The minimum log level to actually record.  Logs with level below this are rejected. Default is INFO. Configure with logger.logLevel.

protected:
   int m_logThreadPrio {0};

   bool m_logThreadRunning {false};
   //<--end of todo

public:
   /// Default c'tor.
   logManager();

   /// Destructor.
   ~logManager();

   /// Set the logger parent 
   /** The parent is used for interactive presentation of log messages
     */
   void parent( parentT * p /**< [in] pointer to the parent object*/);
   
   /// Get the logger parent
   /**
     * Returns a point to the parent object.
     */ 
   parentT * parent();
   
   /// Set a new value of writePause
   /** Updates m_writePause with new value.
     *
     * \returns 0 on success
     * \returns -1 on error (if wp == 0).
     */
   int writePause( const unsigned long & wp /**< [in] the new value of writePause */);

   /// Get the current value of writePause
   /** \returns the value m_writePause.
     */
   unsigned long writePause();

   /// Set a new value of logLevel
   /** Updates m_logLevel with new value.
     * Will return an error and take no actions if the argument
     * is outside the range of the logLevels enum.
     *
     * \returns 0 on success
     * \returns -1 on error.
     */
   int logLevel( logPrioT newLev /**< [in] the new value of logLevel */);

   /// Get the current value of logLevel
   /** \returns the value m_logLevel
     */
   logPrioT logLevel();

   /// Set a new value of logThreadPrio
   /** Updates m_logThreadPrio with new value.
     * If the argument is < 0, this sets m_logThreadPrio to 0.
     * Will not set > 98, and returns -1 with no changes in this case.
     *
     * \returns 0 on success
     * \returns -1 on error (> 98).
     */
   int logThreadPrio( int newPrio /**< [in] the new value of logThreadPrio */);

   /// Get the current value of logThreadPrio
   /** \returns the value m_logThreadPrio
     */
   int logThreadPrio();

   /// Get status of the log thread running flag 
   /** \returns the current value of m_logThreadRunning 
     */
   bool logThreadRunning();
   
   ///Setup an application configurator for the logger section
   int setupConfig( mx::app::appConfigurator & config /**< [in] an application configuration to setup */);

   ///Load the logger section from an application configurator
   /**
     */
   int loadConfig( mx::app::appConfigurator & config /**< [in] an application configuration from which to load values */);


   ///Thread starter, called by logThreadStart on thread construction.  Calls logThreadExec.
   static void _logThreadStart( logManager * l /**< [in] a pointer to a logger instance (normally this) */);

   /// Start the logger thread.
   int logThreadStart();

   /// Execute the logger thread.
   void logThreadExec();

   /// Create a log formatted log entry, filling in a buffer.
   /** This is where the timestamp of the log entry is set.
     *
     * \tparam logT is a log entry type
     *
     * \returns 0 on success, -1 on error.
     */
   template<typename logT>
   static int createLog( bufferPtrT & logBuffer, ///< [out] a shared_ptr\<logBuffer\>, which will be allocated and populated with the log entry 
                         const typename logT::messageT & msg, ///< [in] the message to log (could be of type emptyMessage) 
                         const logPrioT & level  ///< [in] the level (verbosity) of this log
                       );

   /// Create a log formatted log entry, filling in a buffer.
   /** This version has the timestamp provided.
     *
     * \tparam logT is a log entry type
     *
     * \returns 0 on success, -1 on error.
     */
   template<typename logT>
   static int createLog( bufferPtrT & logBuffer, ///< [out] a shared_ptr\<logBuffer\>, which will be allocated and populated with the log entry 
                         const timespecX & ts, ///< [in] the timestamp of this log entry.
                         const typename logT::messageT & msg, ///< [in] the message to log (could be of type emptyMessage) 
                         const logPrioT & level ///< [in] the level (verbosity) of this log
                       );
   
   /// Make a log entry, including a message.
   /**
     * \tparam logT is a log entry type
     */
   template<typename logT>
   void log( const typename logT::messageT & msg, ///< [in] the message to log
             logPrioT level = logPrio::LOG_DEFAULT ///< [in] [optional] the log level.  The default is used if not specified.
           );

   /// Make a log entry, including a message.
   /**
     * \tparam logT is a log entry type
     */
   template<typename logT>
   void log( timespecX & ts, ///< [in] the timestamp of the log entry
             const typename logT::messageT & msg, ///< [in] the message to log
             logPrioT level = logPrio::LOG_DEFAULT ///< [in] [optional] the log level.  The default is used if not specified.
           );
   
   /// Make a log entry with no message.
   /**
     * \tparam logT is a log entry type
     */
   template<typename logT>
   void log( logPrioT level = logPrio::LOG_DEFAULT /**< [in] [optional] the log level.  The default is used if not specified.*/);

   /// Make a log entry with no message.
   /**
     * \tparam logT is a log entry type
     */
   template<typename logT>
   void log( timespecX & ts, ///< [in] the timestamp of the log entry
             logPrioT level = logPrio::LOG_DEFAULT ///< [in] [optional] the log level.  The default is used if not specified.
           );

};

template<class parentT, class logFileT>
logManager<parentT, logFileT>::logManager()
{
}

template<class parentT, class logFileT>
logManager<parentT, logFileT>::~logManager()
{
   m_logShutdown = true;

   if(m_logThread.joinable()) m_logThread.join();

   //One last check to see if there are any unwritten logs.
   if( !m_logQueue.empty() ) logThreadExec();

}

template<class parentT, class logFileT>
void logManager<parentT, logFileT>::parent( parentT * p )
{
   m_parent = p;
}

template<class parentT, class logFileT>
parentT * logManager<parentT, logFileT>::parent()
{
   return m_parent;
}

template<class parentT, class logFileT>
int logManager<parentT, logFileT>::writePause( const unsigned long & wp )
{
   if(wp == 0) return -1;

   m_writePause = wp;

   return 0;
}

template<class parentT, class logFileT>
unsigned long logManager<parentT, logFileT>::writePause()
{
   return m_writePause;
}

template<class parentT, class logFileT>
int logManager<parentT, logFileT>::logLevel( logPrioT newLev )
{
   

   m_logLevel = newLev;

   return 0;
}

template<class parentT, class logFileT>
logPrioT logManager<parentT, logFileT>::logLevel()
{
   return m_logLevel;
}

template<class parentT, class logFileT>
int logManager<parentT, logFileT>::logThreadPrio( int newPrio )
{
   if(newPrio > 98) return -1; //clamped at 98 for safety.
   if(newPrio < 0) newPrio = 0;

   m_logThreadPrio = newPrio;
   return 0;
}

template<class parentT, class logFileT>
int logManager<parentT, logFileT>::logThreadPrio()
{
   return m_logThreadPrio;
}

template<class parentT, class logFileT>
bool logManager<parentT, logFileT>::logThreadRunning()
{
   return m_logThreadRunning;
}

template<class parentT, class logFileT>
int logManager<parentT, logFileT>::setupConfig( mx::app::appConfigurator & config )
{
   config.add(m_configSection+".logDir","L", "logDir",mx::app::argType::Required, m_configSection, "logDir", false, "string", "The directory for log files");
   config.add(m_configSection+".logExt","", "logExt",mx::app::argType::Required, m_configSection, "logExt", false, "string", "The extension for log files");
   config.add(m_configSection+".maxLogSize","", "maxLogSize",mx::app::argType::Required, m_configSection, "maxLogSize", false, "string", "The maximum size of log files");
   config.add(m_configSection+".writePause","", "writePause",mx::app::argType::Required, m_configSection, "writePause", false, "unsigned long", "The log thread pause time in ns");
   config.add(m_configSection+".logThreadPrio", "", "logThreadPrio", mx::app::argType::Required, m_configSection, "logThreadPrio", false, "int", "The log thread priority");
   config.add(m_configSection+".logLevel","l", "logLevel",mx::app::argType::Required, m_configSection, "logLevel", false, "string", "The log level");

   return 0;
}

template<class parentT, class logFileT>
int logManager<parentT, logFileT>::loadConfig( mx::app::appConfigurator & config )
{
   //-- logDir
   std::string tmp;
   config(tmp, m_configSection+".logDir");
   if(tmp != "") this->logPath(tmp);

   //-- logLevel
   tmp = "";
   config(tmp, m_configSection+".logLevel");
   if(tmp != "")
   {
      logPrioT lev;

      lev = logLevelFromString(tmp);

      if(  lev == logPrio::LOG_DEFAULT ) lev = logPrio::LOG_INFO;
      if( lev == logPrio::LOG_UNKNOWN )
      {
         std::cerr << "Unkown log level specified.  Using default (INFO)\n";
         lev = logPrio::LOG_INFO;
      }
      logLevel(lev);
   }


   //logExt
   config(this->m_logExt, m_configSection+".logExt");

   //maxLogSize
   config(this->m_maxLogSize, m_configSection+".maxLogSize");

   //writePause
   config(m_writePause, m_configSection+".writePause");

   //logThreadPrio
   config(m_logThreadPrio, m_configSection+".logThreadPrio");

   return 0;
}

template<class parentT, class logFileT>
void logManager<parentT, logFileT>::_logThreadStart( logManager * l)
{
   l->logThreadExec();
}

template<class parentT, class logFileT>
int logManager<parentT, logFileT>::logThreadStart()
{
   try
   {
      m_logThread = std::thread( _logThreadStart, this);
   }
   catch( const std::exception & e )
   {
      log<software_error>({__FILE__,__LINE__, 0, 0, std::string("Exception on log thread start: ") + e.what()});
      return -1;
   }
   catch( ... )
   {
      log<software_error>({__FILE__,__LINE__, 0, 0, "Unkown exception on log thread start"});
      return -1;
   }
   
   if(!m_logThread.joinable())
   {
      log<software_error>({__FILE__, __LINE__, 0, 0,  "Log thread did not start"});
      return -1;
   }
   
   //Always set the m_logThread to lowest priority
   sched_param sp;
   sp.sched_priority = m_logThreadPrio;

   int rv = pthread_setschedparam( m_logThread.native_handle(), SCHED_OTHER, &sp);
   
   if(rv != 0)
   {
      log<software_error>({__FILE__, __LINE__, 0, rv, std::string("Error setting thread params: ") + strerror(rv)});
      return -1;
   }
   
   return 0;

}

template<class parentT, class logFileT>
void logManager<parentT, logFileT>::logThreadExec()
{

   m_logThreadRunning = true;
   
   std::unique_lock<std::mutex> lock(m_qMutex, std::defer_lock);

   while(!m_logShutdown || !m_logQueue.empty())
   {
      std::list<bufferPtrT>::iterator beg, it, er, end;

      //Get begin and end once, so we only process logs in the list at this point.
      //Do it locked so we don't have any collisions with new logs (maybe not necessary)
      lock.lock();
      beg = m_logQueue.begin();
      end = m_logQueue.end();
      lock.unlock();

      //Note: we're avoiding use of size() since that could be altered concurrently by a push_back.
      if( beg != end )
      {
         it = beg;
         while( it != end )
         {
            //m_logFile.
            if( this->writeLog( *it ) < 0) 
            {
               m_logThreadRunning = false;
               return;
            }
        
            if(m_parent)
            {
               m_parent->logMessage( *it );
            }
            else if( logHeader::logLevel( *it ) <= logPrio::LOG_NOTICE )
            {
               logStdFormat(std::cerr, *it);
               std::cerr << "\n";
            }
            
            er = it;
            ++it;

            //Erase while locked so we don't collide with a push_back.
            lock.lock();
            m_logQueue.erase(er);
            lock.unlock();
         }
      }

      //m_logFile.
      ///\todo must check this for errors, and investigate how `fsyncgate` impacts us
      this->flush();

      //We only pause if there's nothing to do.
      if(m_logQueue.empty() && !m_logShutdown) std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::nano>(m_writePause));
   }

   m_logThreadRunning = false;
}

template<class parentT, class logFileT>
template<typename logT>
int logManager<parentT, logFileT>::createLog( bufferPtrT & logBuffer,
                                     const typename logT::messageT & msg,
                                     const logPrioT & level
                                   )
{
   //Very first step is to get the current time.
   timespecX ts;
   ts.gettime();

   return logHeader::createLog<logT>(logBuffer, ts, msg, level);
}

template<class parentT, class logFileT>
template<typename logT>
int logManager<parentT, logFileT>::createLog( bufferPtrT & logBuffer,
                                              const timespecX & ts,
                                              const typename logT::messageT & msg,
                                              const logPrioT & level
                                            )
{
   return logHeader::createLog<logT>(logBuffer, ts, msg, level);
}

template<class parentT, class logFileT>
template<typename logT>
void logManager<parentT, logFileT>::log( const typename logT::messageT & msg,
                                         logPrioT level
                                       )
{
   //Step 0 check level.
   if(level == logPrio::LOG_DEFAULT) level = logT::defaultLevel;

   if(level > m_logLevel) return; // We do nothing with this.
   
   //Step 1 create log
   bufferPtrT logBuffer;
   createLog<logT>(logBuffer, msg, level);

   //Step 2 add log to queue
   std::lock_guard<std::mutex> guard(m_qMutex);  //Lock the mutex before pushing back.
   m_logQueue.push_back(logBuffer);
   

}

template<class parentT, class logFileT>
template<typename logT>
void logManager<parentT, logFileT>::log( timespecX & ts,
                                         const typename logT::messageT & msg,
                                         logPrioT level
                                       )
{
   //Step 0 check level.
   if(level == logPrio::LOG_DEFAULT) level = logT::defaultLevel;

   if(level > m_logLevel) return; // We do nothing with this.

   //Step 1 create log
   bufferPtrT logBuffer;
   createLog<logT>(logBuffer, ts, msg, level);

   //Step 2 add log to queue
   std::lock_guard<std::mutex> guard(m_qMutex);  //Lock the mutex before pushing back.
   m_logQueue.push_back(logBuffer);

}

template<class parentT, class logFileT>
template<typename logT>
void logManager<parentT, logFileT>::log( logPrioT level )
{
   log<logT>( emptyMessage(), level );
}

template<class parentT, class logFileT>
template<typename logT>
void logManager<parentT, logFileT>::log( timespecX & ts,
                                logPrioT level 
                              )
{
   log<logT>( ts, emptyMessage(), level );
}

//class logFileRaw;

//extern template struct logManager<logFileRaw>;

} //namespace logger
} //namespace MagAOX

#endif //logger_logger_hpp
