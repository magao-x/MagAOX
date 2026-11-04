/** \file streamCircBuff.hpp
  * \brief The MagAO-X streamCircBuff app header file
  *
  * \ingroup streamCircBuff_files
  */

#ifndef streamCircBuff_hpp
#define streamCircBuff_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup streamCircBuff
  * \brief An application to keep a circular buffer of a stream
  *
  * <a href="../handbook/operating/software/apps/streamCircBuff.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup streamCircBuff_files
  * \ingroup streamCircBuff
  */

namespace MagAOX
{
namespace app
{

/// Class for application to keep a circular buffer of a stream and publish it to another stream
/**
  * \ingroup streamCircBuff
  */
class streamCircBuff : public MagAOXApp<true>,
                       public dev::shmimMonitor<streamCircBuff>,
                       public dev::frameGrabber<streamCircBuff>,
                       public dev::telemeter<streamCircBuff>
{
   friend class dev::shmimMonitor<streamCircBuff>;
   friend class dev::frameGrabber<streamCircBuff>;
   friend class dev::telemeter<streamCircBuff>;

public:

   /** \name app::dev Configurations
     *@{
     */

   static constexpr bool c_frameGrabber_flippable = false; ///< app:dev config to tell framegrabber these images can not be flipped

   ///@}

   /// The base shmimMonitor type
   typedef dev::shmimMonitor<streamCircBuff> shmimMonitorT;

   /// The base frameGrabber type
   typedef dev::frameGrabber<streamCircBuff> frameGrabberT;

   /// The telemeter type
   typedef dev::telemeter<streamCircBuff> telemeterT;


protected:

   /** \name Configurable Parameters
     *@{
     */
   ///@}

   sem_t m_smSemaphore {0}; ///< Semaphore used to synchronize the fg thread and the sm thread.

   std::mutex m_stageMutex; ///< Protects the staged frame and its timing, which the sm thread writes and the fg thread reads.

   std::vector<char> m_staged; ///< Copy of the latest source frame, written by the sm thread.

   timespec m_stagedAtime {0,0}; ///< Source acquisition time of the staged frame.

   timespec m_stagedWritetime {0,0}; ///< Source write time of the staged frame.

   uint64_t m_stagedCnt0 {0}; ///< Source frame counter of the staged frame.

   bool m_stagedFresh {false}; ///< True when the staged frame has not been published yet.

   timespec m_relayWritetime {0,0}; ///< Source write time of the frame being published, used only by the fg thread.

   uint64_t m_relayCnt0 {0}; ///< Source frame counter of the frame being published, used only by the fg thread.

   uint64_t m_cnt0Offset {0}; ///< Added to the source frame counter so the published counter keeps increasing after a source restart.

   std::atomic<uint64_t> m_rewritten {0}; ///< Number of source frames dropped because the source was rewritten during the copy.

   uint64_t m_rewrittenLogged {0}; ///< Value of m_rewritten when last logged.

public:
   /// Default c'tor.
   streamCircBuff();

   /// D'tor, declared and defined for noexcept.
   ~streamCircBuff() noexcept
   {}

   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);

   virtual void loadConfig();

   /// Startup function
   /**
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for streamCircBuff.
   /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
   virtual int appLogic();

   /// Shutdown the app.
   /**
     *
     */
   virtual int appShutdown();


protected:

   //shmimMonitor Interface
   int allocate( const dev::shmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);

   /// Copy the current source frame and its timing to the stage, and wake the fg thread
   /** The frame is dropped if the source was written during the copy.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const dev::shmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );

   /** \name dev::frameGrabber interface
     *
     * @{
     */

   /// Implementation of the framegrabber configureAcquisition interface
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int configureAcquisition();

   /// Implementation of the framegrabber fps interface
   /**
     * \todo this needs to infer the stream fps and return it
     */
   float fps()
   {
      return 1.0;
   }

   /// Implementation of the framegrabber startAcquisition interface
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int startAcquisition();

   /// Implementation of the framegrabber acquireAndCheckValid interface
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int acquireAndCheckValid();

   /// Implementation of the framegrabber loadImageIntoStream interface
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int loadImageIntoStream( void * dest  /**< [in] */);

   /// Implementation of the framegrabber reconfig interface
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int reconfig();

   /// Supply the source write time and frame counter of the frame being published
   /** Called by the framegrabber after loadImageIntoStream.  The published counter is offset so it keeps
     * increasing if the source counter restarts.
     *
     * \returns 0
     */
   int frameGrabberRelayTiming( timespec & writetime, ///< [out] the source write time
                                uint64_t & cnt0       ///< [in/out] on entry the next counter, on exit the counter to publish
                              );

   ///@}

   /** \name Telemeter Interface
     *
     * @{
     */
   int checkRecordTimes();

   int recordTelem( const telem_fgtimings * );

   ///@}

};

streamCircBuff::streamCircBuff() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{

   return;
}

void streamCircBuff::setupConfig()
{
   SHMIMMONITOR_SETUP_CONFIG(config);

   FRAMEGRABBER_SETUP_CONFIG(config);

   TELEMETER_SETUP_CONFIG(config);

}

int streamCircBuff::loadConfigImpl( mx::app::appConfigurator & _config )
{
   SHMIMMONITOR_LOAD_CONFIG(_config);

   FRAMEGRABBER_LOAD_CONFIG(_config);

   TELEMETER_LOAD_CONFIG(config);

   return 0;
}

void streamCircBuff::loadConfig()
{
   loadConfigImpl(config);
}

int streamCircBuff::appStartup()
{
   if(sem_init(&m_smSemaphore, 0,0) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno,0, "Initializing S.M. semaphore"});
      return -1;
   }

   SHMIMMONITOR_APP_STARTUP;

   FRAMEGRABBER_APP_STARTUP;

   TELEMETER_APP_STARTUP;

   state(stateCodes::OPERATING);

   return 0;
}

int streamCircBuff::appLogic()
{
   SHMIMMONITOR_APP_LOGIC;

   FRAMEGRABBER_APP_LOGIC;

   TELEMETER_APP_LOGIC;

   uint64_t rewritten = m_rewritten;
   if(rewritten != m_rewrittenLogged)
   {
      log<text_log>("Source frames dropped because they were rewritten during the copy: " + std::to_string(rewritten - m_rewrittenLogged), logPrio::LOG_WARNING);
      m_rewrittenLogged = rewritten;
   }

   std::unique_lock<std::mutex> lock(m_indiMutex);

   SHMIMMONITOR_UPDATE_INDI;

   FRAMEGRABBER_UPDATE_INDI;

   return 0;
}

int streamCircBuff::appShutdown()
{
   SHMIMMONITOR_APP_SHUTDOWN;
   FRAMEGRABBER_APP_SHUTDOWN;
   TELEMETER_APP_SHUTDOWN;

   return 0;
}

int streamCircBuff::allocate( const dev::shmimT & dummy)
{
   static_cast<void>(dummy);

   { //mutex scope
      std::lock_guard<std::mutex> lock(m_stageMutex);
      m_staged.resize(shmimMonitorT::m_width * shmimMonitorT::m_height * shmimMonitorT::m_typeSize);
      m_stagedFresh = false;
   }

   m_reconfig = true;

   return 0;
}

int streamCircBuff::processImage( void * curr_src,
                                  const dev::shmimT & dummy
                                )
{
   static_cast<void>(dummy);

   IMAGE & src = shmimMonitorT::m_imageStream;

   { //mutex scope
      std::lock_guard<std::mutex> lock(m_stageMutex);

      size_t frameBytes = m_staged.size();

      if(shmimMonitorT::m_depth > 1 && src.cntarray != nullptr)
      {
         //A multi-frame source does not rewrite this frame until it wraps around
         size_t slot = (static_cast<char *>(curr_src) - static_cast<char *>(src.array.raw)) / frameBytes;

         memcpy(m_staged.data(), curr_src, frameBytes);
         m_stagedAtime = src.atimearray[slot];
         m_stagedWritetime = src.writetimearray[slot];
         m_stagedCnt0 = src.cntarray[slot];
      }
      else
      {
         //The source sets write before it writes, and increments cnt0 before clearing write
         uint64_t cnt0 = src.md->cnt0;
         int write = src.md->write;
         std::atomic_thread_fence(std::memory_order_acquire);

         memcpy(m_staged.data(), curr_src, frameBytes);
         m_stagedAtime = src.md->atime;
         m_stagedWritetime = src.md->writetime;

         std::atomic_thread_fence(std::memory_order_acquire);
         if(write != 0 || src.md->write != 0 || src.md->cnt0 != cnt0)
         {
            m_stagedFresh = false;
            ++m_rewritten;
            return 0;
         }

         m_stagedCnt0 = cnt0;
      }

      m_stagedFresh = true;
   }

   //Now tell the f.g. to get going
   if(sem_post(&m_smSemaphore) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
      return -1;
   }

   return 0;
}

int streamCircBuff::configureAcquisition()
{
   std::unique_lock<std::mutex> lock(m_indiMutex);

   ///\todo potential but verrrrry unlikely bug: shmimMonitorT could change these before allocate sets the lock above.  Should use a local set of w/h instead.
   if(shmimMonitorT::m_width==0 || shmimMonitorT::m_height==0)
   {
      //This means we haven't connected to the stream to accumulate. so wait.
      lock.unlock(); //don't hold the lock for a whole second.
      sleep(1);
      return -1;
   }

   frameGrabberT::m_width = shmimMonitorT::m_width;
   frameGrabberT::m_height = shmimMonitorT::m_height;
   frameGrabberT::m_dataType = shmimMonitorT::m_dataType;

   return 0;
}

int streamCircBuff::startAcquisition()
{
   return 0;
}

int streamCircBuff::acquireAndCheckValid()
{
   timespec ts;

   if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
   {
      log<software_critical>({__FILE__,__LINE__,errno,0,"clock_gettime"});
      return -1;
   }

   ts.tv_sec += 1;

   if(sem_timedwait(&m_smSemaphore, &ts) != 0)
   {
      return 1;
   }

   //A frame already published, e.g. after several posts, is not published again
   std::lock_guard<std::mutex> lock(m_stageMutex);
   return m_stagedFresh ? 0 : 1;
}

int streamCircBuff::loadImageIntoStream(void * dest)
{
   std::lock_guard<std::mutex> lock(m_stageMutex);

   if(m_staged.size() != frameGrabberT::m_width * frameGrabberT::m_height * frameGrabberT::m_typeSize)
   {
      return -1; //the source changed, so reconfigure
   }

   memcpy(dest, m_staged.data(), m_staged.size());

   m_currImageTimestamp = m_stagedAtime;
   m_relayWritetime = m_stagedWritetime;
   m_relayCnt0 = m_stagedCnt0;
   m_stagedFresh = false;

   return 0;
}

int streamCircBuff::reconfig()
{
   return 0;
}

int streamCircBuff::frameGrabberRelayTiming( timespec & writetime,
                                             uint64_t & cnt0
                                           )
{
   writetime = m_relayWritetime;

   if(m_relayCnt0 + m_cnt0Offset < cnt0)
   {
      m_cnt0Offset = cnt0 - m_relayCnt0; //the source counter restarted, or the output stream is newer than the source
   }

   cnt0 = m_relayCnt0 + m_cnt0Offset;

   return 0;
}

int streamCircBuff::checkRecordTimes()
{
   return telemeterT::checkRecordTimes(telem_fgtimings());
}

int streamCircBuff::recordTelem( const telem_fgtimings * )
{
   return recordFGTimings(true);
}


} //namespace app
} //namespace MagAOX

#endif //streamCircBuff_hpp
