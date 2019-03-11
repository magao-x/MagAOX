/** \file streamWriter.hpp
  * \brief The MagAO-X Image Stream Writer
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup streamWriter_files
  */

#ifndef streamWriter_hpp
#define streamWriter_hpp


#include <ImageStruct.h>
#include <ImageStreamIO.h>


#include <mx/timeUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"



namespace MagAOX
{
namespace app
{

/** \defgroup streamWriter ImageStreamIO Stream Writing
  * \brief Writes the contents of an ImageStreamIO image stream to disk.
  *
  *  <a href="../apps_html/page_module_streamWriter.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup streamWriter_files ImageStreamIO Stream Writing
  * \ingroup streamWriter
  */

/** MagAO-X application to control writing ImageStreamIO streams to disk.
  *
  * \ingroup streamWriter
  * 
  */
class streamWriter : public MagAOXApp<>
{

protected:

   /** \name configurable parameters 
     *@{
     */ 

   std::string m_rawimageDir; ///< The path where files will be saved.
   
   size_t m_circBuffLength {1000};
   
   size_t m_writeChunkLength {100};
   
   std::string m_streamName;
   
   int m_semaphoreNumber {2}; //The semaphore number to monitor.  Default is 2.
   unsigned m_semWait {500000000}; //The time in nsec to wait on the semaphore.  Max is 999999999. Default is 5e8 nsec.
   
   ///@}
   
   
   size_t m_width {0}; ///< The width of the image
   size_t m_height {0}; ///< The height of the image
   uint8_t m_atype {0}; ///< The ImageStreamIO type code.
   int m_byteDepth {0}; ///< The pixel byte depth
   
    
   char * m_rawImageCircBuff {0};
   
   size_t m_currImage {0};
   size_t m_currChunkStart {0};
   size_t m_nextChunkStart {0};
         
public:

   ///Default c'tor
   streamWriter();

   ///Destructor
   ~streamWriter() noexcept;

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Startup functions
   /** Sets up the INDI vars.
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for the Siglent SDG
   virtual int appLogic();


   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();
   
protected:
   
   /** \name SIGSEGV & SIGBUS signal handling
     * These signals occur as a result of a ImageStreamIO source server resetting (e.g. changing frame sizes).
     * When they occur a restart of the framegrabber and framewriter thread main loops is triggered.
     * 
     * @{
     */ 
   bool m_restart {false};
   
   static streamWriter * m_selfWriter; ///< Static pointer to this (set in constructor).  Used for getting out of the static SIGSEGV handler.

   ///Sets the handler for SIGSEGV and SIGBUS
   /** These are caused by ImageStreamIO server resets.
     */
   int setSigSegvHandler();
   
   ///The handler called when SIGSEGV or SIGBUS is received, which will be due to ImageStreamIO server resets.  Just a wrapper for handlerSigSegv.
   static void _handlerSigSegv( int signum,
                                siginfo_t *siginf,
                                void *ucont
                              );

   ///Handles SIGSEGV and SIGBUS.  Sets m_restart to true.
   void handlerSigSegv( int signum,
                        siginfo_t *siginf,
                        void *ucont
                      );
   ///@}

   /** \name Framegrabber Thread 
     * This thread monitors the ImageStreamIO buffer and copies its images to the circular buffer.
     *
     * @{
     */ 
   int m_fgThreadPrio {1}; ///< Priority of the framegrabber thread, should normally be > 00.

   std::thread m_fgThread; ///< A separate thread for the actual framegrabbings

   ///Thread starter, called by fgThreadStart on thread construction.  Calls fgThreadExec.
   static void _fgThreadStart( streamWriter * s /**< [in] a pointer to an streamWriter instance (normally this) */);

   /// Start the frame grabber thread.
   int fgThreadStart();

   /// Execute the frame grabber main loop.
   void fgThreadExec();

   ///@}
   
   /** \name Stream Writer Thread 
     * This thread writes chunks of the circular buffer to disk.
     *
     * @{
     */ 
   int m_swThreadPrio {1}; ///< Priority of the stream writer thread, should normally be > 0, and <= m_fgThreadPrio.

   sem_t m_swSemaphore; ///< Semaphore used to synchronize the fg thread and the sw thread.
   
   std::thread m_swThread; ///< A separate thread for the actual writing

   ///Thread starter, called by swThreadStart on thread construction.  Calls swThreadExec.
   static void _swThreadStart( streamWriter * s /**< [in] a pointer to an streamWriter instance (normally this) */);

   /// Start the stream writer
   int swThreadStart();

   /// Execute the stream writer main loop.
   void swThreadExec();

   ///@}
   
   //INDI:
protected:
   //declare our properties
   pcf::IndiProperty m_indiP_writing;
   pcf::IndiProperty m_indiP_written;

public:
   INDI_NEWCALLBACK_DECL(streamWriter, m_indiP_writing);

};

//Set self pointer to null so app starts up uninitialized.
streamWriter * streamWriter::m_selfWriter = nullptr;

inline
streamWriter::streamWriter() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = false;
 
   m_selfWriter = this;
   
   return;
}

inline
streamWriter::~streamWriter() noexcept
{
   return;
}

inline
void streamWriter::setupConfig()
{
   config.add("writer.savePath", "", "writer.savePath", argType::Required, "writer", "savePath", false, "string", "The absolute path where images are saved. Will use MagAO-X default if not set.");
   
   config.add("writer.circBuffLength", "", "writer.circBuffLength", argType::Required, "writer", "circBuffLength", false, "size_t", "The length in frames of the circular buffer. Should be an integer multiple of and larger than writeChunkLength.");
   
   config.add("writer.writeChunkLength", "", "writer.writeChunkLength", argType::Required, "writer", "writeChunkLength", false, "size_t", "The length in frames of the chunks to write to disk. Should be smaller than circBuffLength.");
   
   config.add("writer.threadPrio", "", "writer.threadPrio", argType::Required, "writer", "threadPrio", false, "int", "The real-time priority of the stream writer thread.");
   
   config.add("framegrabber.streamName", "", "framegrabber.streamName", argType::Required, "framegrabber", "streamName", false, "int", "The name of the stream to monitor. From /tmp/streamName.im.shm.");
   config.add("framegrabber.semaphoreNumber", "", "framegrabber.semaphoreNumber", argType::Required, "framegrabber", "semaphoreNumber", false, "int", "The number of the semaphore to monitor.");
   config.add("framegrabber.semWait", "", "framegrabber.semWait", argType::Required, "framegrabber", "semWait", false, "int", "The time in nsec to wait on the semaphore.  Max is 999999999. Default is 5e8 nsec.");
   
   config.add("framegrabber.threadPrio", "", "framegrabber.threadPrio", argType::Required, "framegrabber", "threadPrio", false, "int", "The real-time priority of the framegrabber thread.");
}



inline
void streamWriter::loadConfig()
{
   //Set some defaults
   //Setup default log path
   m_rawimageDir = MagAOXPath + "/" + MAGAOX_rawimageRelPath;
   config(m_rawimageDir, "writer.savePath");
   
   config(m_circBuffLength, "writer.circBuffLength");
   config(m_writeChunkLength, "writer.writeChunkLength");
   config(m_swThreadPrio, "writer.threadPrio");
   
   config(m_streamName, "framegrabber.streamName");
   config(m_semaphoreNumber, "framegrabber.semaphoreNumber");
   config(m_semWait, "framegrabber.semWait");
   
   
   config(m_fgThreadPrio, "framegrabber.threadPrio");
   
}



inline
int streamWriter::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP(m_indiP_writing, "writing", pcf::IndiProperty::Switch);
   m_indiP_writing.add (pcf::IndiElement("current"));
   m_indiP_writing.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP_NOCB(m_indiP_written, "written", pcf::IndiProperty::Number);
   m_indiP_written.add (pcf::IndiElement("number"));
   m_indiP_written["number"].set(0);
   m_indiP_written.add (pcf::IndiElement("fps"));
   m_indiP_written["fps"].set(0);

   //Now set up the framegrabber and writer threads.
   // - need SIGSEGV and SIGBUS handling for ImageStreamIO restarts
   // - initialize the semaphore 
   // - start the threads
   
   if(setSigSegvHandler() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return -1;
   }
   
   if(sem_init(&m_swSemaphore, 0,0) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno,0, "Initializing S.W. semaphore"});
      return -1;
   }

   if(fgThreadStart() < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }

   if(swThreadStart() < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
   
   return 0;

}



inline
int streamWriter::appLogic()
{
   //first do a join check to see if other threads have exited.
   if(pthread_tryjoin_np(m_fgThread.native_handle(),0) == 0)
   {
      log<software_error>({__FILE__, __LINE__, "framegrabber thread has exited"});
      
      return -1;
   }
   
   if(pthread_tryjoin_np(m_swThread.native_handle(),0) == 0)
   {
      log<software_error>({__FILE__, __LINE__, "stream thread has exited"});
      
      return -1;
   }
   
   return 0;

}

inline
int streamWriter::appShutdown()
{
   if(m_fgThread.joinable())
   {
      m_fgThread.join();
   }
   
   if(m_swThread.joinable())
   {
      m_swThread.join();
   }
   
   return 0;
}

inline
int streamWriter::setSigSegvHandler()
{
   struct sigaction act;
   sigset_t set;

   act.sa_sigaction = &streamWriter::_handlerSigSegv;
   act.sa_flags = SA_SIGINFO;
   sigemptyset(&set);
   act.sa_mask = set;

   errno = 0;
   if( sigaction(SIGSEGV, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGSEGV failed. Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, 0, logss});

      return -1;
   }

   errno = 0;
   if( sigaction(SIGBUS, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGBUS failed. Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, 0,logss});

      return -1;
   }

   log<text_log>("Installed SIGSEGV/SIGBUS signal handler.", logPrio::LOG_DEBUG);

   return 0;
}

inline
void streamWriter::_handlerSigSegv( int signum,
                                    siginfo_t *siginf,
                                    void *ucont
                                  )
{
   m_selfWriter->handlerSigSegv(signum, siginf, ucont);
}

inline
void streamWriter::handlerSigSegv( int signum,
                                   siginfo_t *siginf,
                                   void *ucont
                                 )
{
   static_cast<void>(signum);
   static_cast<void>(siginf);
   static_cast<void>(ucont);
   
   m_restart = true;

   return;
}

inline
void streamWriter::_fgThreadStart( streamWriter * o)
{
   o->fgThreadExec();
}

inline
int streamWriter::fgThreadStart()
{
   try
   {
      m_fgThread  = std::thread( _fgThreadStart, this);
   }
   catch( const std::exception & e )
   {
      log<software_error>({__FILE__,__LINE__, std::string("Exception on framegrabber thread start: ") + e.what()});
      return -1;
   }
   catch( ... )
   {
      log<software_error>({__FILE__,__LINE__, "Unkown exception on framegrabber thread start"});
      return -1;
   }

   if(!m_fgThread.joinable())
   {
      log<software_error>({__FILE__, __LINE__, "framegrabber thread did not start"});
      return -1;
   }

   //Now set the RT priority.
   
   int prio=m_fgThreadPrio;
   if(prio < 0) prio = 0;
   if(prio > 99) prio = 99;

   sched_param sp;
   sp.sched_priority = prio;

   //Get the maximum privileges available
   if( euidCalled() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, "Setting euid to called failed."});
      return -1;
   }
   
   //We set return value based on result from sched_setscheduler
   //But we make sure to restore privileges no matter what happens.
   errno = 0;
   int rv = 0;
   if(prio > 0) rv = pthread_setschedparam(m_fgThread.native_handle(), MAGAOX_RT_SCHED_POLICY, &sp);
   else rv = pthread_setschedparam(m_fgThread.native_handle(), SCHED_OTHER, &sp);
   
   //Go back to regular privileges
   if( euidReal() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, "Setting euid to real failed."});
   }
   
   if(rv < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__, errno, "Setting F.G. thread scheduler priority to " + std::to_string(prio) + " failed."});
   }
   else
   {
      return log<text_log,0>("F.G. thread scheduler priority (framegrabber.threadPrio) set to " + std::to_string(prio));
   }
   

}

inline
void streamWriter::fgThreadExec()
{

   IMAGE image;
   bool opened = false;
   
   while(m_shutdown == 0)
   {
      /* Initialize ImageStreamIO
       */
      opened = false;
      m_restart = false; //Set this up front, since we're about to restart.
      
      sem_t * sem {nullptr}; ///< The semaphore to monitor for new image data
      
      while(!opened && !m_shutdown && !m_restart)
      {
         if( ImageStreamIO_openIm(&image, m_streamName.c_str()) == 0)
         {
            if(image.md[0].sem <= m_semaphoreNumber) 
            {
               ImageStreamIO_closeIm(&image);
               mx::sleep(1); //We just need to wait for the server process to finish startup.
            }
            else
            {
               opened = true;
            }
         }
         else
         {
            mx::sleep(1); //be patient
         }
      }
      
      if(m_shutdown || !opened) return;
    
      sem = image.semptr[m_semaphoreNumber];
      m_atype = image.md[0].atype;
      m_byteDepth = ImageStreamIO_typesize(image.md[0].atype);
      m_width = image.md[0].size[0];
      m_height = image.md[0].size[1];
      size_t length = image.md[0].size[2];

      
      //Now allocate teh circBuff 
      
      m_rawImageCircBuff = (char *) malloc( xrif::rawImageFrameSz(m_width, m_height, m_byteDepth)*m_circBuffLength );
      
      
      
      uint8_t atype;
      size_t snx, sny, snz;


      long curr_image;
      m_currImage = 0;
      m_currChunkStart = 0;
      m_nextChunkStart = 0;
      
      uint64_t imno = 0;
      //This is the main image grabbing loop.
      while(!m_shutdown && !m_restart)
      {
         timespec ts;
         
         if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
         {
            log<software_critical>({__FILE__,__LINE__,errno,0,"clock_gettime"}); 
            return;
         }
         
         mx::timespecAddNsec(ts, m_semWait);
         
         if(sem_timedwait(sem, &ts) == 0)
         {
            if(image.md[0].size[2] > 0) ///\todo change to naxis?
            {
               curr_image = image.md[0].cnt1 - 1;
               if(curr_image < 0) curr_image = image.md[0].size[2] - 1;
            }
            else curr_image = 0;

            atype = image.md[0].atype;
            snx = image.md[0].size[0];
            sny = image.md[0].size[1];
            snz = image.md[0].size[2];
         
            if( atype!= m_atype || snx != m_width || sny != m_height || snz != length )
            {
               break; //exit the nearest while loop and get the new image setup.
            }
         
            if(m_shutdown || m_restart) break; //Check for exit signals
         
            
            xrif::copyRawImageFrame( m_currImage, m_rawImageCircBuff, (char *) image.array.SI8, image.md[0].atime.ts.tv_sec, image.md[0].atime.ts.tv_nsec, m_width, m_height, m_byteDepth);
            ++imno;
         
            ++m_currImage;
            
            if(m_currImage - m_nextChunkStart == m_writeChunkLength)
            {
               std::cerr << "Write: " << m_nextChunkStart << " to " << m_currImage << "\n";
            
               m_currChunkStart = m_nextChunkStart;
               
               //Now tell the writer to get going
               if(sem_post(&m_swSemaphore) < 0)
               {
                  log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
                  return;
               }
               
               m_nextChunkStart += m_writeChunkLength;
               
               if(m_nextChunkStart >= m_circBuffLength) m_nextChunkStart = 0;
            }
            
            
            if(m_currImage >= m_circBuffLength) m_currImage = 0;
            
         }
         else
         {
            if(image.md[0].sem <= 0) break; //Indicates that the server has cleaned up.
            
            //Check for why we timed out
            if(errno == EINTR) break; //This will indicate time to shutdown, loop will exit normally flags set.
            
            //ETIMEDOUT just means we should wait more.
            //Otherwise, report an error.
            if(errno != ETIMEDOUT)
            {
               log<software_error>({__FILE__, __LINE__,errno, "sem_timedwait"});
               break;
            }
         }
      }

      if(m_rawImageCircBuff)
      {
         free(m_rawImageCircBuff);
         m_rawImageCircBuff = 0;
      }
      
      if(opened) 
      {
         ImageStreamIO_closeIm(&image);
         opened = false;
      }
      
   } //outer loop, will exit if m_shutdown==true
   
   //One more check
   if(m_rawImageCircBuff)
   {
      free(m_rawImageCircBuff);
      m_rawImageCircBuff = 0;
   }
      
   if(opened) ImageStreamIO_closeIm(&image);
   
}


inline
void streamWriter::_swThreadStart( streamWriter * o)
{
   o->swThreadExec();
}

inline
int streamWriter::swThreadStart()
{
   try
   {
      m_swThread  = std::thread( _swThreadStart, this);
   }
   catch( const std::exception & e )
   {
      log<software_error>({__FILE__,__LINE__, std::string("Exception on stream writer thread start: ") + e.what()});
      return -1;
   }
   catch( ... )
   {
      log<software_error>({__FILE__,__LINE__, "Unkown exception on stream writer thread start"});
      return -1;
   }

   if(!m_swThread.joinable())
   {
      log<software_error>({__FILE__, __LINE__, "stream writer thread did not start"});
      return -1;
   }

   //Now set the RT priority.
   
   int prio=m_swThreadPrio;
   if(prio < 0) prio = 0;
   if(prio > 99) prio = 99;

   sched_param sp;
   sp.sched_priority = prio;

   //Get the maximum privileges available
   if( euidCalled() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, "Setting euid to called failed."});
      return -1;
   }
   
   //We set return value based on result from sched_setscheduler
   //But we make sure to restore privileges no matter what happens.
   errno = 0;
   int rv = 0;
   if(prio > 0) rv = pthread_setschedparam(m_swThread.native_handle(), MAGAOX_RT_SCHED_POLICY, &sp);
   else rv = pthread_setschedparam(m_swThread.native_handle(), SCHED_OTHER, &sp);
   
   //Go back to regular privileges
   if( euidReal() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, "Setting euid to real failed."});
   }
   
   if(rv < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__, errno, "Setting S.W.. thread scheduler priority to " + std::to_string(prio) + " failed."});
   }
   else
   {
      return log<text_log,0>("S.W. thread scheduler priority (writer.threadPrio) set to " + std::to_string(prio));
   }
   

}

inline
void streamWriter::swThreadExec()
{
   char * fname = 0;
   while(m_shutdown == 0)
   {
      std::string fnameBase = m_rawimageDir + "/" + m_streamName;
      
      size_t fnameSz = fnameBase.size() + sizeof("_YYYYMMDDHHMMSSNNNNNNNNN.xrif");
      if(!fname) fname = (char*) malloc(fnameSz);
      
      snprintf(fname, fnameSz, "%s_YYYYMMDDHHMMSSNNNNNNNNN.xrif", fnameBase.c_str());
      
      while(!m_shutdown && !m_restart)
      {
         timespec ts;
         
         if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
         {
            log<software_critical>({__FILE__,__LINE__,errno,0,"clock_gettime"}); 
            return;
         }
         
         mx::timespecAddNsec(ts, m_semWait);
         
         if(sem_timedwait(&m_swSemaphore, &ts) == 0)
         {
            double t0 = mx::get_curr_time();
            
            ssize_t totw = xrif::writeBuffer( fname, fnameBase.size()+1, m_currChunkStart, m_writeChunkLength, m_rawImageCircBuff, m_width, m_height, m_atype);

            double t1 = mx::get_curr_time();
            
            std::cerr << "Wrote: " << m_currChunkStart << " to " << m_currChunkStart+m_writeChunkLength << ": " << fname << " " << totw << " bytes in " << t1-t0 << "sec \n";
         }
         else
         {
            
            //Check for why we timed out
            if(errno == EINTR) break; //This will probably indicate time to shutdown, loop will exit normally if flags set.
            
            //ETIMEDOUT just means we should wait more.
            //Otherwise, report an error.
            if(errno != ETIMEDOUT)
            {
               log<software_error>({__FILE__, __LINE__,errno, "sem_timedwait"});
               break;
            }
         }
      }
  } //outer loop, will exit if m_shutdown==true
  
  if(fname) free(fname);
}

INDI_NEWCALLBACK_DEFN(streamWriter, m_indiP_writing)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_writing.getName())
   {
      
      return 0;
   }
   return -1;
}


}//namespace app
} //namespace MagAOX
#endif
