/** \file frameGrabber.hpp
  * \brief The MagAO-X Princeton Instruments EMCCD camera controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup frameGrabber_files
  */

#ifndef frameGrabber_hpp
#define frameGrabber_hpp



#include <xrif/xrif.h>


#include <ImageStruct.h>
#include <ImageStreamIO.h>

#define NOT_WRITING (0)
#define START_WRITING (1)
#define WRITING (2)
#define STOP_WRITING (3)

namespace MagAOX
{
namespace app
{
namespace dev 
{
   



/** MagAO-X generic frame grabber
  *
  * \ingroup frameGrabber
  * 
  * The derived class `derivedT` must expose the following interface
  * \code 
    //Starts the camera acquistion, must also set m_width, m_height, and m_dataType
    int derivedT::startAcquisition();
    
    //Acquires the data, and checks if it is valid.
    //This should set m_currImageTimestamp to the image timestamp.
    int derivedT::acquireAndCheckValid()
    
    //Loads the acquired image into the stream, copying it to the appropriate member of imageStream.array.
    int derivedT::loadImageIntoStream();
    
    //Take any actions needed to reconfigure the system.  Called if m_reconfig is set to true.
    int derivedT::reconfig()
  * \endcode  
  * Each of the above functions should return 0 on success, and -1 on an error.  In most cases, 
  * an appropriate state code, such as NOTCONNECTED, should be set as well.
  *
  * Additionally, the derived class must have a `friend` declaration for frameGrabber, like so:
  * \code
    friend class dev::frameGrabber<derivedT>;
  * \endcode
  * replacing derivedT by the name of the class.
  *
  * 
  */
template<class derivedT>
class frameGrabber 
{

   derivedT * m_parent;
   
protected:

   /** \name Configurable Parameters
    * @{
    */
   std::string m_shmimName {""}; ///< The name of the shared memory image, is used in `/tmp/<shmimName>.im.shm`. Derived classes should set a default.
      
   int m_fgThreadPrio {1}; ///< Priority of the framegrabber thread, should normally be > 00.
    
   uint32_t m_circBuffLength {1000}; ///< Length of the circular buffer, in frames
   
   uint32_t m_writeChunkLength {500}; ///< The number of images to write at a time.  Should normally be < m_circBuffLength.
   
   std::string m_rawimageDir; ///< The path where files will be saved.
   
   int m_swThreadPrio {1}; ///< Priority of the stream writer thread, should normally be > 0, and <= m_fgThreadPrio.
    
   unsigned m_semWait {500000000}; //The time in nsec to wait on the writer semaphore.  Max is 999999999. Default is 5e8 nsec.
   
   
   
   ///@}
   
   uint32_t m_width {0}; ///< The width of the image, once deinterlaced etc.
   uint32_t m_height {0}; ///< The height of the image, once deinterlaced etc.
   
   uint8_t m_dataType{0}; ///< The ImageStreamIO type code.
   size_t m_typeSize {0}; ///< The size of the type, in bytes.  Result of sizeof.
   
   int m_xbinning {0}; ///< The x-binning according to the framegrabber
   int m_ybinning {0}; ///< The y-binning according to the framegrabber
   std::string m_cameraType; ///< The camera type according to the framegrabber
          
   timespec m_currImageTimestamp; ///< The timestamp of the current image.
   
   bool m_reconfig {false}; ///< Flag to set if a camera reconfiguration requires a framegrabber reset.
   
   IMAGE imageStream; ///< The ImageStreamIO shared memory buffer.
   
   int m_writing {NOT_WRITING};
   
   //Writer book-keeping:
   size_t m_currImage {0};
   
   size_t m_currChunkStart {0};
   size_t m_nextChunkStart {0};
   
   size_t m_currSaveStart {0};
   size_t m_currSaveStop {0};
   
   ///The xrif compression handle
   xrif_t xrif {nullptr};
   
public:

   ///Default c'tor
   frameGrabber();

   ///Destructor
   ~frameGrabber() noexcept;

   /// Setup the configuration system
   /**
     * This should be called in `derivedT::setupConfig` as
     * \code
       framegrabber<derivedT>::setupConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void setupConfig(mx::app::appConfigurator & config /**< [out] the derived classes configurator*/);

   /// load the configuration system results
   /**
     * This should be called in `derivedT::loadConfig` as
     * \code
       framegrabber<derivedT>::loadConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void loadConfig(mx::app::appConfigurator & config /**< [in] the derived classes configurator*/);

   /// Startup function
   /** Starts the framegrabber thread
     * This should be called in `derivedT::appStartup` as
     * \code
       framegrabber<derivedT>::appStartup();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appStartup();

   /// Checks the framegrabber thread
   /** This should be called in `derivedT::appLogic` as
     * \code
       framegrabber<derivedT>::appLogic();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appLogic();

   /// Shuts down the framegrabber thread
   /** This should be called in `derivedT::appShutdown` as
     * \code
       framegrabber<derivedT>::appShutdown();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appShutdown();
   
protected:
   

   /** \name Framegrabber Thread
     * This thread actually manages the framegrabbing hardware
     * @{
     */
   
   std::thread m_fgThread; ///< A separate thread for the actual framegrabbings

   ///Thread starter, called by fgThreadStart on thread construction.  Calls fgThreadExec.
   static void _fgThreadStart( frameGrabber * o /**< [in] a pointer to an frameGrabber instance (normally this) */);

   /// Start the log capture.
   int fgThreadStart();

   /// Execute the log capture.
   void fgThreadExec();

   ///@}
  
   
    /** \name Stream Writer Thread 
      * This thread writes chunks of the circular buffer to disk.
      *
      * @{
      */ 
   sem_t m_swSemaphore; ///< Semaphore used to synchronize the fg thread and the sw thread.
   
   std::thread m_swThread; ///< A separate thread for the actual writing

   ///Thread starter, called by swThreadStart on thread construction.  Calls swThreadExec.
   static void _swThreadStart( frameGrabber * o /**< [in] a pointer to an framegrabber instance (normally this) */);

   /// Start the stream writer
   int swThreadStart();

   /// Execute the stream writer main loop.
   void swThreadExec();
      
   ///@}
   
    /** \name INDI 
      *
      *@{
      */ 
protected:
   //declare our properties
   pcf::IndiProperty m_indiP_writing;

   pdf::IndiProperty m_indiP_xrifStats;
   
public:
  
   /// The static callback function to be registered for the writing property
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_writing( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                      const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                    );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_writing( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   

   /// Update the INDI properties for this device controller
   /** You should call this once per main loop.
     * It is not called automatically.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int updateINDI();

   ///@}
};

template<class derivedT>
frameGrabber<derivedT>::frameGrabber() 
{
   m_parent = static_cast<derivedT *>(this);
   
   return;
}

template<class derivedT>
frameGrabber<derivedT>::~frameGrabber() noexcept
{
   return;
}

template<class derivedT>
void frameGrabber<derivedT>::setupConfig(mx::app::appConfigurator & config)
{
   config.add("framegrabber.threadPrio", "", "framegrabber.threadPrio", argType::Required, "framegrabber", "threadPrio", false, "int", "The real-time priority of the fraemgrabber thread.");
   config.add("framegrabber.shmimName", "", "framegrabber.shmimName", argType::Required, "framegrabber", "shmimName", false, "string", "The name of the ImageStreamIO shared memory image. Will be used as /tmp/<shmimName>.im.shm.");
   
   config.add("framegrabber.circBuffLength", "", "framegrabber.circBuffLength", argType::Required, "framegrabber", "circBuffLength", false, "size_t", "The length of the circular buffer. Sets m_circBuffLength, default is 1.");

   config.add("framegrabber.savePath", "", "framegrabber.savePath", argType::Required, "framegrabber", "savePath", false, "string", "The absolute path where images are saved. Will use MagAO-X default if not set.");
      
   config.add("framegrabber.writeChunkLength", "", "framegrabber.writeChunkLength", argType::Required, "framegrabber", "writeChunkLength", false, "size_t", "The length in frames of the chunks to write to disk. Should be smaller than circBuffLength, and must be a whole divisor.");
   config.add("framegrabber.writerThreadPrio", "", "framegrabber.writerThreadPrio", argType::Required, "framegrabber", "writerThreadPrio", false, "int", "The real-time priority of the stream writer thread.");

   config.add("framegrabber.semWait", "", "framegrabber.semWait", argType::Required, "framegrabber", "semWait", false, "int", "The time in nsec to wait on the writer semaphore.  Max is 999999999. Default is 5e8 nsec.");
   
}

template<class derivedT>
void frameGrabber<derivedT>::loadConfig(mx::app::appConfigurator & config)
{
   config(m_fgThreadPrio, "framegrabber.threadPrio");
   config(m_shmimName, "framegrabber.shmimName");
   config(m_circBuffLength, "framegrabber.circBuffLength");
   config(m_rawimageDir,"framegrabber.savePath");
   config(m_writeChunkLength,"framegrabber.writeChunkLength");
   config(m_swThreadPrio,"framegrabber.writerThreadPrio");
   config(m_semWait, "framegrabber.semWait");
}
   

template<class derivedT>
int frameGrabber<derivedT>::appStartup()
{
   //Check if we have a safe writeChunkLengthh
   if( m_circBuffLength % m_writeChunkLength != 0)
   {
      return derivedT::template log<software_critical, -1>({__FILE__,__LINE__, "Write chunk length is not a divisor of circular buffer length."});
   }
   
   int rv = xrif_new(&xrif);
   if( rv != XRIF_NOERROR )
   {
      return derivedT::template log<software_critical, -1>({__FILE__,__LINE__, "xrif handle allocation or initialization error. Code = " + std::to_string(rv)});
   }
   
   //Register the writing INDI property
   m_indiP_writing = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_writing.setDevice(static_cast<derivedT *>(this)->configName());
   m_indiP_writing.setName("writing");
   m_indiP_writing.setPerm(pcf::IndiProperty::ReadWrite);
   m_indiP_writing.setState(pcf::IndiProperty::Idle);
    
   m_indiP_writing.add(pcf::IndiElement("current"));
   m_indiP_writing["current"].set(0);
   m_indiP_writing.add(pcf::IndiElement("target"));
   m_indiP_writing["target"].set(0);
   
   if( static_cast<derivedT *>(this)->registerIndiPropertyNew( m_indiP_writing, st_newCallBack_writing) < 0)
   {
      #ifndef OUTLET_CTRL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   
   
   if(sem_init(&m_swSemaphore, 0,0) < 0)
   {
      derivedT::template log<software_critical>({__FILE__, __LINE__, errno,0, "Initializing S.W. semaphore"});
      return -1;
   }
   
   if(fgThreadStart() < 0)
   {
      derivedT::template log<software_error>({__FILE__, __LINE__});
      return -1;
   }
   
   if(swThreadStart() < 0)
   {
      derivedT::template log<software_error>({__FILE__, __LINE__});
      return -1;
   }
   
   return 0;

}

template<class derivedT>
int frameGrabber<derivedT>::appLogic()
{
   //do a join check to see if other threads have exited.
   if(pthread_tryjoin_np(m_fgThread.native_handle(),0) == 0)
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "framegrabber thread has exited"});
      
      return -1;
   }
   
   if(pthread_tryjoin_np(m_swThread.native_handle(),0) == 0)
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "writer thread has exited"});
      
      return -1;
   }
   
   return 0;

}


template<class derivedT>
int frameGrabber<derivedT>::appShutdown()
{
   if(m_fgThread.joinable())
   {
      m_fgThread.join();
   }
   
   if(m_swThread.joinable())
   {
      m_swThread.join();
   }
   
   if(xrif)
   {
      xrif_delete(xrif);
   }
   return 0;
}



template<class derivedT>
void frameGrabber<derivedT>::_fgThreadStart( frameGrabber * o)
{
   o->fgThreadExec();
}

template<class derivedT>
int frameGrabber<derivedT>::fgThreadStart()
{
   try
   {
      m_fgThread  = std::thread( _fgThreadStart, this);
   }
   catch( const std::exception & e )
   {
      derivedT::template log<software_error>({__FILE__,__LINE__, std::string("Exception on framegrabber thread start: ") + e.what()});
      return -1;
   }
   catch( ... )
   {
      derivedT::template log<software_error>({__FILE__,__LINE__, "Unkown exception on framegrabber thread start"});
      return -1;
   }

   if(!m_fgThread.joinable())
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "framegrabber thread did not start"});
      return -1;
   }

   //Now set the RT priority.
   
   int prio=m_fgThreadPrio;
   if(prio < 0) prio = 0;
   if(prio > 99) prio = 99;

   sched_param sp;
   sp.sched_priority = prio;

   //Get the maximum privileges available
   if( m_parent->euidCalled() < 0 )
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "Setting euid to called failed."});
      return -1;
   }
   
   //We set return value based on result from sched_setscheduler
   //But we make sure to restore privileges no matter what happens.
   errno = 0;
   int rv = 0;
   if(prio > 0) rv = pthread_setschedparam(m_fgThread.native_handle(), MAGAOX_RT_SCHED_POLICY, &sp);
   else rv = pthread_setschedparam(m_fgThread.native_handle(), SCHED_OTHER, &sp);
   
   //Go back to regular privileges
   if( m_parent->euidReal() < 0 )
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "Setting euid to real failed."});
   }
   
   if(rv < 0)
   {
      return derivedT::template log<software_error,-1>({__FILE__, __LINE__, errno, "Setting F.G. thread scheduler priority to " + std::to_string(prio) + " failed."});
   }
   else
   {
      return derivedT::template log<text_log,0>("F.G. thread scheduler priority (framegrabber.threadPrio) set to " + std::to_string(prio));
   }
   

}

template<class derivedT>
void frameGrabber<derivedT>::fgThreadExec()
{
   while(m_parent->shutdown() == 0)
   {
      while(!m_parent->shutdown() && (!( m_parent->state() == stateCodes::READY || m_parent->state() == stateCodes::OPERATING) || m_parent->powerState() <= 0 ) )
      {
         sleep(1);
      }
      
      if(m_parent->shutdown()) continue;
      else 
      {
         //At the end of this, must have m_width, m_height, m_dataType set.
         if(m_parent->startAcquisition() < 0) continue;        
         
         m_typeSize = ImageStreamIO_typesize(m_dataType);
         
         
      }

      /* Initialize ImageStreamIO
       */
      uint32_t imsize[3];
      imsize[0] = m_width; 
      imsize[1] = m_height;
      imsize[2] = m_circBuffLength;
      ImageStreamIO_createIm(&imageStream, m_shmimName.c_str(), 3, imsize, m_dataType, 1, 0);
      imageStream.md[0].cnt1 = m_circBuffLength;
      
      xrif_set_size(xrif, m_width, m_height, 1, m_circBuffLength, m_dataType);
      xrif->compress_on_raw = 0;
      xrif_allocate_reordered(xrif);
      xrif_allocate_compressed(xrif);
      
      //This completes the reconfiguration.
      m_reconfig = false;
                  
      m_currImage = 0;
      m_currChunkStart = 0;
      m_nextChunkStart = 0;
      
      //This is the main image grabbing loop.      
      while(!m_parent->shutdown() && !m_reconfig && m_parent->powerState() > 0)
      {
         //==================
         //Get next image, process validity.
         //====================         
         if(m_parent->acquireAndCheckValid() < 0)
         {
            break;
         }
         
         //Ok, no timeout, so we process the image and publish it.
         imageStream.md[0].write=1;
         
         //Increment cnt1, wrapping for circular buffer.
         uint64_t cnt1 = imageStream.md[0].cnt1 + 1;
         if(cnt1 > m_circBuffLength-1) cnt1 = 0;
         imageStream.md[0].cnt1 = cnt1;

         if(m_parent->loadImageIntoStream((char *) imageStream.array.raw + imageStream.md[0].cnt1*m_width*m_height*m_typeSize) < 0) 
         {
            break;
         }
         
         imageStream.md[0].atime = m_currImageTimestamp;;
         imageStream.md[0].cnt0++;
         
         
         imageStream.md[0].write=0;
         ImageStreamIO_sempost(&imageStream,-1);
 
         //----------------------------
         //Writer book keeping
         m_currImage = imageStream.md[0].cnt1;
         
         switch(m_writing)
         {
            case START_WRITING:
               m_currChunkStart = m_currImage;
               m_nextChunkStart = (m_currImage / m_writeChunkLength)*m_writeChunkLength;
               m_writing = WRITING;
               // fall through
            case WRITING:
               if( m_currImage - m_nextChunkStart == m_writeChunkLength-1 )
               {  
                  m_currSaveStart = m_currChunkStart;
                  m_currSaveStop = m_nextChunkStart + m_writeChunkLength;
               
                  //Now tell the writer to get going
                  if(sem_post(&m_swSemaphore) < 0)
                  {
                     derivedT::template log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
                     return;
                  }
              
                  m_nextChunkStart = ( (m_currImage  + 1) / m_writeChunkLength)*m_writeChunkLength;
                  if(m_nextChunkStart >= m_circBuffLength) m_nextChunkStart = 0;
               
                  m_currChunkStart = m_nextChunkStart;
                
               }
               break;
               
            case STOP_WRITING:
               m_currSaveStart = m_currChunkStart;
               m_currSaveStop = m_currImage + 1;
               //Now tell the writer to get going
               if(sem_post(&m_swSemaphore) < 0)
               {
                  derivedT::template log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
                  return;
               }
               break;
               
            default:
               break;
         }
        
      }
    
      ImageStreamIO_destroyIm( &imageStream );
    
      if(m_reconfig && !m_parent->shutdown())
      {
         m_parent->reconfig();
      }

   } //outer loop, will exit if m_shutdown==true

}


template<class derivedT>
void frameGrabber<derivedT>::_swThreadStart( frameGrabber * o)
{
   o->swThreadExec();
}

template<class derivedT>
int frameGrabber<derivedT>::swThreadStart()
{
   try
   {
      m_swThread  = std::thread( _swThreadStart, this);
   }
   catch( const std::exception & e )
   {
      derivedT::template log<software_error>({__FILE__,__LINE__, std::string("Exception on writer thread start: ") + e.what()});
      return -1;
   }
   catch( ... )
   {
      derivedT::template log<software_error>({__FILE__,__LINE__, "Unkown exception on writer thread start"});
      return -1;
   }

   if(!m_swThread.joinable())
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "writer thread did not start"});
      return -1;
   }

   //Now set the RT priority.
   
   int prio=m_swThreadPrio;
   if(prio < 0) prio = 0;
   if(prio > 99) prio = 99;

   sched_param sp;
   sp.sched_priority = prio;

   //Get the maximum privileges available
   if( m_parent->euidCalled() < 0 )
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "Setting euid to called failed."});
      return -1;
   }
   
   //We set return value based on result from sched_setscheduler
   //But we make sure to restore privileges no matter what happens.
   errno = 0;
   int rv = 0;
   if(prio > 0) rv = pthread_setschedparam(m_swThread.native_handle(), MAGAOX_RT_SCHED_POLICY, &sp);
   else rv = pthread_setschedparam(m_swThread.native_handle(), SCHED_OTHER, &sp);
   
   //Go back to regular privileges
   if( m_parent->euidReal() < 0 )
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "Setting euid to real failed."});
   }
   
   if(rv < 0)
   {
      return derivedT::template log<software_error,-1>({__FILE__, __LINE__, errno, "writer thread scheduler priority to " + std::to_string(prio) + " failed."});
   }
   else
   {
      return derivedT::template log<text_log,0>("writer thread scheduler priority (framegrabber.writerThreadPrio) set to " + std::to_string(prio));
   }
   

}

template<class derivedT>
void frameGrabber<derivedT>::swThreadExec()
{

   while(!m_parent->m_shutdown)
   {
      timespec ts;
       
      if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
      {
         derivedT::template log<software_critical>({__FILE__,__LINE__,errno,0,"clock_gettime"}); 
         return; //will trigger a shutdown
      }
       
      mx::timespecAddNsec(ts, m_semWait);
      
      if(sem_timedwait(&m_swSemaphore, &ts) == 0)
      {
         if(m_writing == NOT_WRITING) continue;
               
         xrif_set_size(xrif, m_width, m_height, 1, (m_currSaveStop-m_currSaveStart), m_dataType);
         xrif_set_raw(xrif, (char *) imageStream.array.raw + m_currSaveStart*m_width*m_height*m_typeSize, (m_currSaveStop-m_currSaveStart)*m_width*m_height*m_typeSize);
         xrif->lz4_acceleration=50;
         xrif_encode(xrif);
         
         
         double ratio = ((double)xrif->compressed_size)/( (m_currSaveStop-m_currSaveStart)*m_width*m_height*m_typeSize);
         
        
         double td = ((double) xrif->ts_compress_done.tv_sec + ((double) xrif->ts_compress_done.tv_nsec)/1e9) -  ((double) xrif->ts_difference_start.tv_sec + ((double) xrif->ts_difference_start.tv_nsec)/1e9);  
         std::cerr << "Compressed: " << m_currSaveStart << " to " << m_currSaveStop << " [" << ratio*100 << "% @ " << (double)(m_currSaveStop-m_currSaveStart) / td <<" fps]\n";
         
         if(m_writing == STOP_WRITING) m_writing = NOT_WRITING;
      }
      else
      {
         //Check for why we timed out
         if(errno == EINTR) continue; //This will probably indicate time to shutdown, loop will exit normally if flags set.
          
         //ETIMEDOUT just means we should wait more.
         //Otherwise, report an error.
         if(errno != ETIMEDOUT)
         {
            derivedT::template log<software_error>({__FILE__, __LINE__,errno, "sem_timedwait"});
            break;
         }
      }
   } //outer loop, will exit if m_shutdown==true
}

template<class derivedT>
int frameGrabber<derivedT>::st_newCallBack_writing( void * app, 
                                                    const pcf::IndiProperty &ipRecv 
                                                  )
{
   return static_cast<derivedT *>(app)->newCallBack_writing(ipRecv);
}


template<class derivedT>
int frameGrabber<derivedT>::newCallBack_writing( const pcf::IndiProperty &ipRecv )
{
   if (ipRecv.getName() == m_indiP_writing.getName())
   {
      int current = -1;
      int target = -1;
      
      if(ipRecv.find("current"))
      {
         current = ipRecv["current"].get<int>();
      }
      
      if(ipRecv.find("target"))
      {
         target = ipRecv["target"].get<int>();
      }
      
      if(target == -1 ) target = current;
      
      if(target == 0 && (m_writing == WRITING || m_writing == START_WRITING))
      {
         m_writing = STOP_WRITING;
      }
      
      if(target == 1 && m_writing == NOT_WRITING)
      {
         m_writing = START_WRITING;
      }
      
      //Lock the m parent's mutex, waiting if necessary
      std::unique_lock<std::mutex> lock( m_parent->m_indiMutex );
      
      //And update target
      indi::updateIfChanged(m_indiP_writing, "target", target, m_parent->m_indiDriver);
      
      return 0;
   }
   
   return -1;
}

template<class derivedT>
int frameGrabber<derivedT>::updateINDI()
{
   if( !m_parent->m_indiDriver ) return 0;
   
   //Only update this if not changing
   if(m_writing == NOT_WRITING || m_writing == WRITING)
   {
      indi::updateIfChanged(m_indiP_writing, "current", (int) (m_writing == WRITING), m_parent->m_indiDriver);
   }
   
   return 0;
}


} //namespace dev
} //namespace app
} //namespace MagAOX
#endif
