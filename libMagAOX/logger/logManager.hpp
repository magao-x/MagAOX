/** \file logManager.hpp 
  * \brief The MagAO-X log manager.
  * \author Jared R. Males (jaredmales@gmail.com)
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


#include "../time/timespecX.hpp"

#include "logTypes.hpp"

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
  */
template<class logFileT>
struct logManager : public logFileT
{
   ///\todo Make these protected members, with appropriate access methods
   //-->
   std::list<bufferPtrT> m_logQueue; ///< Log entries are stored here, and writen to the file by the log thread.
   
   std::thread m_logThread; ///< A separate thread for actually writing to the file.
   std::mutex m_qMutex; ///< Mutex for accessing the m_logQueue.
   
   bool m_logShutdown {false}; ///< Flag to signal the log thread to shutdown.
   
   unsigned long m_writePause {MAGAOX_default_writePause}; ///< Time, in nanoseconds, to pause between successive batch writes to the file. Default is 1e9. Configure with logger.writePause.
   
   logLevelT m_logLevel {logLevels::INFO}; ///< The minimum log level to actually record.  Logs with level below this are rejected. Default is INFO. Configure with logger.logLevel.
   
   int m_logThreadPrio {0};
   
   //<--end of todo
   
   /// Default c'tor.
   logManager(); 
   
   /// Destructor.
   ~logManager();

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
   
   ///Thread starter, called by m_logThreadStart on thread construction.  Calls m_logThreadExec.
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
   static int createLog( bufferPtrT & logBuffer, /**< [out] a shared_ptr\<logBuffer\>, which will be allocated and populated with the log entry */
                         const typename logT::messageT & msg, /**< [in] the message to log (could be of type emptyMessage) */
                         logLevelT level
                       );

   
   /// Make a log entry, including a message.
   /** 
     * \tparam logT is a log entry type
     */
   template<typename logT>   
   void log( const typename logT::messageT & msg, ///< [in] the message to log
             logLevelT level = logLevels::DEFAULT ///< [in] [optional] the log level.  The default is used if not specified.
           );
   
   /// Make a log entry with no message.
   /**
     * \tparam logT is a log entry type
     */
   template<typename logT>
   void log( logLevelT level = logLevels::DEFAULT /**< [in] [optional] the log level.  The default is used if not specified.*/);
   
   
};

template<class logFileT>
logManager<logFileT>::logManager()
{
}

template<class logFileT>
logManager<logFileT>::~logManager()
{
   m_logShutdown = true;
   
   if(m_logThread.joinable()) m_logThread.join();
   
   //One last check to see if there are any unwritten logs.
   if( m_logQueue.size() > 0 ) logThreadExec();
   
}

template<class logFileT>
int logManager<logFileT>::writePause( const unsigned long & wp )
{
   if(wp == 0) return -1;
   
   m_writePause = wp;
   
   return 0;
}

template<class logFileT>
unsigned long logManager<logFileT>::writePause()
{
   return m_writePause;
}

template<class logFileT>
void logManager<logFileT>::_logThreadStart( logManager * l)
{
   l->logThreadExec();
}

template<class logFileT>
int logManager<logFileT>::logThreadStart()
{
   m_logThread = std::thread( _logThreadStart, this);
   
   //Always set the m_logThread to lowest priority
   sched_param sp;
   sp.sched_priority = m_logThreadPrio;
   
   pthread_setschedparam( m_logThread.native_handle(), SCHED_OTHER, &sp);
   
}
   
template<class logFileT>
void logManager<logFileT>::logThreadExec()
{
   
   std::unique_lock<std::mutex> lock(m_qMutex, std::defer_lock);
   
   while(!m_logShutdown || m_logQueue.size() > 0)
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
            this->writeLog( *it );
            
            er = it;
            ++it;

            //Erase while locked so we don't collide with a push_back.
            lock.lock();
            m_logQueue.erase(er);
            lock.unlock();
         }
      }
      
      //m_logFile.
      this->flush();
      
      //We only pause if there's nothing to do.
      if(m_logQueue.size() == 0 && !m_logShutdown) std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::nano>(m_writePause));
   }
   
}

template<class logFileT>
template<typename logT>
int logManager<logFileT>::createLog( bufferPtrT & logBuffer,
                                 const typename logT::messageT & msg, 
                                 logLevelT level 
                               )
{
   //Very first step is to get the current time.
   time::timespecX ts;
   ts.gettime();
   
   if(level == logLevels::DEFAULT) level = logT::defaultLevel;
   
   //We first allocate the buffer.
   msgLenT len = logT::length(msg);     
   logBuffer = bufferPtrT(new char[headerSize + len]);
   
   //Now load the basics.
   reinterpret_cast<logHeaderT *>(logBuffer.get())->logLevel = level;
   reinterpret_cast<logHeaderT *>(logBuffer.get())->eventCode = logT::eventCode;
   reinterpret_cast<logHeaderT *>(logBuffer.get())->timespecX = ts;
   reinterpret_cast<logHeaderT *>(logBuffer.get())->msgLen = len;
   
   
   //Each log-type is responsible for loading its message
   logT::format( logBuffer.get() + messageOffset, msg);
   
   
}
 
template<class logFileT>
template<typename logT>   
void logManager<logFileT>::log( const typename logT::messageT & msg,
                            logLevelT level
                          )  
{
   //Step 0 check level.
   if(level == logLevels::DEFAULT) level = logT::defaultLevel;
   
   if( level > 0) //Normal logs
   {
      if(level < m_logLevel) return; // We do nothing with this. 
   }
   else //Telemetry logs
   {
      if(level > m_logLevel) return; // We do nothing with this.
   }
   
   //Step 1 create log
   bufferPtrT logBuffer;
   createLog<logT>(logBuffer, msg, level);
   
   //Step 2 add log to queue
   std::lock_guard<std::mutex> guard(m_qMutex);  //Lock the mutex before pushing back.
   m_logQueue.push_back(logBuffer);
   
}

template<class logFileT>
template<typename logT>
void logManager<logFileT>::log( logLevelT level )
{
   log<logT>( emptyMessage(), level );
}



   
} //namespace logger
} //namespace MagAOX

#endif //logger_logger_hpp

