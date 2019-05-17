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

#include "../../common/paths.hpp"

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

   derivedT * m_parent; ///< The parent class, used for casting `this`
   
protected:

   /** \name Configurable Parameters
    * @{
    */
   std::string m_shmimName {""}; ///< The name of the shared memory image, is used in `/tmp/<shmimName>.im.shm`. Derived classes should set a default.
      
   int m_fgThreadPrio {2}; ///< Priority of the framegrabber thread, should normally be > 00.
    
   uint32_t m_circBuffLength {1024}; ///< Length of the circular buffer, in frames
   
   uint32_t m_writeChunkLength {512}; ///< The number of images to write at a time.  Should normally be < m_circBuffLength.
   
   std::string m_rawimageDir; ///< The path where files will be saved.  Normally derived from the library config.
   
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
   
   
   
   //Writer book-keeping:
   int m_writing {NOT_WRITING}; ///< Controls whether or not images are being written, and sequences start and stop of writing.
   
   uint64_t m_currChunkStart {0}; ///< The circular buffer starting position of the current to-be-written chunk.
   uint64_t m_nextChunkStart {0}; ///< The circular buffer starting position of the next to-be-written chunk.
   
   uint64_t m_currSaveStart {0}; ///< The circular buffer position at which to start saving.
   uint64_t m_currSaveStop {0}; ///< The circular buffer position at which to stop saving.
   
   bool m_logSaveStart {0}; ///< Flag indicating that the start saving log should entry should be made.
   uint64_t m_currSaveStartFrameNo {0}; ///< The frame number of the image at which saving started (for logging)
   uint64_t m_currSaveStopFrameNo {0}; ///< The frame number of the image at which saving stopped (for logging)
   
   ///The xrif compression handle
   xrif_t xrif {nullptr};
   
   ///Storage for the xrif file header
   char * xrif_header {nullptr};
   
   ///Storage for the iamge timing data for writing
   char * m_timingData {nullptr};
   
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
   
   bool m_fgThreadInit {true}; ///< Synchronizer for thread startup, to allow priority setting to finish.
   
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
    
   bool m_swThreadInit {true}; ///< Synchronizer for thread startup, to allow priority setting to finish.
   
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
   
   pcf::IndiProperty m_indiP_shmimName; ///< Property used to report the shmim buffer name
   
   pcf::IndiProperty m_indiP_frameSize; ///< Property used to report the current frame size

   pcf::IndiProperty m_indiP_bufferSize; ///< Property used to report the current buffer size.
   
   
   pcf::IndiProperty m_indiP_writing; ///< Property used to control whether frames are being written to disk

   pcf::IndiProperty m_indiP_xrifStats; ///< Property to report xrif compression performance.
   
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
   if(xrif) xrif_delete(xrif);
   
   if(xrif_header) free(xrif_header);
   
   if(m_timingData) free(m_timingData);
   
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
   
   
   //Setup default save path before overriding with config
   m_rawimageDir = std::string(MAGAOX_path) + "/" + MAGAOX_rawimageRelPath + "/" + m_parent->configName();
   config(m_rawimageDir,"framegrabber.savePath");
   
   config(m_writeChunkLength,"framegrabber.writeChunkLength");
   config(m_swThreadPrio,"framegrabber.writerThreadPrio");
   config(m_semWait, "framegrabber.semWait");
}
   

template<class derivedT>
int frameGrabber<derivedT>::appStartup()
{
   //Create save directory.
   errno = 0;
   if( mkdir(m_rawimageDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0 )
   {
      if( errno != EEXIST)
      {
         std::stringstream logss;
         logss << "Failed to create image directory (" << m_rawimageDir << ").  Errno says: " << strerror(errno);
         derivedT::template log<software_critical>({__FILE__, __LINE__, errno, 0, logss.str()});

         return -1;
      }

   }
   
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
   
   xrif_header = (char *) malloc( XRIF_HEADER_SIZE * sizeof(char));
      
   //Register the shmimName INDI property
   m_indiP_shmimName = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_shmimName.setDevice(m_parent->configName());
   m_indiP_shmimName.setName("shmimName");
   m_indiP_shmimName.setPerm(pcf::IndiProperty::ReadWrite);
   m_indiP_shmimName.setState(pcf::IndiProperty::Idle);
   m_indiP_shmimName.add(pcf::IndiElement("name"));
   m_indiP_shmimName["name"] = m_shmimName;
   
   if( m_parent->registerIndiPropertyNew( m_indiP_shmimName, nullptr) < 0)
   {
      #ifndef FRAMEGRABBER_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the frameSize INDI property
   m_indiP_frameSize = pcf::IndiProperty(pcf::IndiProperty::Number);
   m_indiP_frameSize.setDevice(m_parent->configName());
   m_indiP_frameSize.setName("frameSize");
   m_indiP_frameSize.setPerm(pcf::IndiProperty::ReadWrite);
   m_indiP_frameSize.setState(pcf::IndiProperty::Idle);
   m_indiP_frameSize.add(pcf::IndiElement("width"));
   m_indiP_frameSize["width"] = 0;
   m_indiP_frameSize.add(pcf::IndiElement("height"));
   m_indiP_frameSize["height"] = 0;
   
   if( m_parent->registerIndiPropertyNew( m_indiP_frameSize, nullptr) < 0)
   {
      #ifndef FRAMEGRABBER_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the writing INDI property
   m_indiP_writing = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_writing.setDevice(m_parent->configName());
   m_indiP_writing.setName("writing");
   m_indiP_writing.setPerm(pcf::IndiProperty::ReadWrite);
   m_indiP_writing.setState(pcf::IndiProperty::Idle);
    
   m_indiP_writing.add(pcf::IndiElement("current"));
   m_indiP_writing["current"].set(0);
   m_indiP_writing.add(pcf::IndiElement("target"));
   m_indiP_writing["target"].set(0);
   
   if( m_parent->registerIndiPropertyNew( m_indiP_writing, st_newCallBack_writing) < 0)
   {
      #ifndef FRAMEGRABBER_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the stats INDI property
   m_indiP_xrifStats = pcf::IndiProperty(pcf::IndiProperty::Number);
   m_indiP_xrifStats.setDevice(m_parent->configName());
   m_indiP_xrifStats.setName("xrif");
   m_indiP_xrifStats.setPerm(pcf::IndiProperty::ReadOnly);
   m_indiP_xrifStats.setState(pcf::IndiProperty::Idle);
    
   m_indiP_xrifStats.add(pcf::IndiElement("ratio"));
   m_indiP_xrifStats["ratio"].set(0);
   
   m_indiP_xrifStats.add(pcf::IndiElement("differenceMBsec"));
   m_indiP_xrifStats["differenceMBsec"].set(0);
   m_indiP_xrifStats.add(pcf::IndiElement("reorderMBsec"));
   m_indiP_xrifStats["reorderMBsec"].set(0);
   
   m_indiP_xrifStats.add(pcf::IndiElement("compressMBsec"));
   m_indiP_xrifStats["compressMBsec"].set(0);
   m_indiP_xrifStats.add(pcf::IndiElement("encodeMBsec"));
   m_indiP_xrifStats["encodeMBsec"].set(0);
   
   m_indiP_xrifStats.add(pcf::IndiElement("differenceFPS"));
   m_indiP_xrifStats["differenceFPS"].set(0);
   m_indiP_xrifStats.add(pcf::IndiElement("reorderFPS"));
   m_indiP_xrifStats["reorderFPS"].set(0);
   
   m_indiP_xrifStats.add(pcf::IndiElement("compressFPS"));
   m_indiP_xrifStats["compressFPS"].set(0);
   m_indiP_xrifStats.add(pcf::IndiElement("encodeFPS"));
   m_indiP_xrifStats["encodeFPS"].set(0);
   
   if( m_parent->registerIndiPropertyNew( m_indiP_xrifStats, nullptr) < 0)
   {
      #ifndef FRAMEGRABBER_TEST_NOLOG
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
      xrif=nullptr;
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
   m_fgThreadInit = true;
   
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
      m_fgThreadInit = false;
      
      return derivedT::template log<text_log,0>("F.G. thread scheduler priority (framegrabber.threadPrio) set to " + std::to_string(prio));
   }
   

}

template<class derivedT>
void frameGrabber<derivedT>::fgThreadExec()
{
   
   
   //Wait fpr the thread starter to finish initializing this thread.
   while(m_fgThreadInit == true && m_parent->m_shutdown == 0)
   {
       sleep(1);
   }
   
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
      
      if(m_shmimName == "") m_shmimName = m_parent->configName();
      
      ImageStreamIO_createIm_gpu(&imageStream, m_shmimName.c_str(), 3, imsize, m_dataType, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL);
       
      imageStream.md->cnt1 = m_circBuffLength;
      
      xrif_set_size(xrif, m_width, m_height, 1, m_writeChunkLength, m_dataType);
      xrif->compress_on_raw = 1;
      xrif_allocate_reordered(xrif);
      xrif_allocate_raw(xrif);

      //Alocate the timing data buffer used for writing.
      if(m_timingData) free(m_timingData);
      m_timingData = (char *) malloc( (sizeof(uint64_t) + 2*sizeof(timespec))*m_writeChunkLength);
      
      
      //This completes the reconfiguration.
      m_reconfig = false;
                  
      m_currChunkStart = 0;
      m_nextChunkStart = 0;
      
      //This is the main image grabbing loop.      
      while(!m_parent->shutdown() && !m_reconfig && m_parent->powerState() > 0)
      {
         //==================
         //Get next image, process validity.
         //====================         
         int isValid = m_parent->acquireAndCheckValid();
         if( isValid < 0)
         {
            break;
         }
         else if( isValid > 0)
         {
            continue;
         }
         
         //Ok, no timeout, so we process the image and publish it.
         imageStream.md->write=1;
         
         //Increment cnt1, wrapping for circular buffer.
         uint64_t cnt1 = imageStream.md->cnt1 + 1;
         if(cnt1 > m_circBuffLength-1) cnt1 = 0;
         imageStream.md->cnt1 = cnt1;

         if(m_parent->loadImageIntoStream((char *) imageStream.array.raw + imageStream.md->cnt1*m_width*m_height*m_typeSize) < 0) 
         {
            break;
         }
         
         //Set the time of last write
         clock_gettime(CLOCK_REALTIME, &imageStream.md->writetime);

         //Set the image acquisition timestamp
         imageStream.md->atime = m_currImageTimestamp;
         
         //Update cnt0
         imageStream.md->cnt0++;
         
         //Update the circular buffers
         imageStream.writetimearray[cnt1] = imageStream.md->writetime;
         imageStream.atimearray[cnt1] = m_currImageTimestamp;
         imageStream.cntarray[cnt1] = imageStream.md->cnt0;
         
         //And post
         imageStream.md->write=0;
         ImageStreamIO_sempost(&imageStream,-1);
 
         //----------------------------
         //Writer book keeping
         uint64_t currImage = imageStream.md->cnt1;
         
         switch(m_writing)
         {
            case START_WRITING:
               m_currChunkStart = currImage;
               m_nextChunkStart = (currImage / m_writeChunkLength)*m_writeChunkLength;
               m_writing = WRITING;
               m_currSaveStartFrameNo = imageStream.md->cnt0;
               m_logSaveStart = true;
               // fall through
            case WRITING:
               if( currImage - m_nextChunkStart == m_writeChunkLength-1 )
               {  
                  m_currSaveStart = m_currChunkStart;
                  m_currSaveStop = m_nextChunkStart + m_writeChunkLength;
                  m_currSaveStopFrameNo = imageStream.md->cnt0;
                  
                  //Now tell the writer to get going
                  if(sem_post(&m_swSemaphore) < 0)
                  {
                     derivedT::template log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
                     return;
                  }
              
                  m_nextChunkStart = ( (currImage  + 1) / m_writeChunkLength)*m_writeChunkLength;
                  if(m_nextChunkStart >= m_circBuffLength) m_nextChunkStart = 0;
               
                  m_currChunkStart = m_nextChunkStart;
                
               }
               break;
               
            case STOP_WRITING:
               m_currSaveStart = m_currChunkStart;
               m_currSaveStop = currImage + 1;
               m_currSaveStopFrameNo = imageStream.md->cnt0;
               
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
   m_swThreadInit = true;
   
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
      m_swThreadInit = false;
      
      return derivedT::template log<text_log,0>("writer thread scheduler priority (framegrabber.writerThreadPrio) set to " + std::to_string(prio));
   }
   

}

template<class derivedT>
void frameGrabber<derivedT>::swThreadExec()
{
   //Wait fpr the thread starter to finish initializing this thread.
   while(m_swThreadInit == true && m_parent->m_shutdown == 0)
   {
       sleep(1);
   }

   
   char * fname = nullptr;
   char * fnameTiming = nullptr;
      
   std::string fnameBase;
   std::string fnameTimingBase;
      
   while(!m_parent->m_shutdown)
   {
      while(!m_parent->shutdown() && (!( m_parent->state() == stateCodes::READY || m_parent->state() == stateCodes::OPERATING) || m_parent->powerState() <= 0 ) )
      {
         if(fname) 
         {
            free(fname);
            fname = nullptr;
         }
         
         if(fnameTiming) 
         {
            free(fnameTiming);
            fnameTiming = nullptr;
         }
         sleep(1);
      }
      
      //This will happen after a reconnection, and could update m_shmimName, etc.
      if(fname == nullptr)
      {
         fnameBase = m_rawimageDir + "/" + m_shmimName + "_";
      
         size_t fnameSz = fnameBase.size() + sizeof("YYYYMMDDHHMMSSNNNNNNNNN.xrif");
         fname = (char*) malloc(fnameSz);
      
         snprintf(fname, fnameSz, "%sYYYYMMDDHHMMSSNNNNNNNNN.xrif", fnameBase.c_str());

      }
      
      if(fnameTiming == nullptr)
      {
         fnameTimingBase = m_rawimageDir + "/" + m_shmimName + "_timing_";
      
         size_t fnameSz = fnameTimingBase.size() + sizeof("YYYYMMDDHHMMSSNNNNNNNNN.dat");
         fnameTiming = (char*) malloc(fnameSz);
      
         snprintf(fnameTiming, fnameSz, "%sYYYYMMDDHHMMSSNNNNNNNNN.dat", fnameTimingBase.c_str());

      }
      
      timespec ts;
       
      if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
      {
         derivedT::template log<software_critical>({__FILE__,__LINE__,errno,0,"clock_gettime"}); 
         
         if(fname) 
         {
            free(fname);
            fname = nullptr;
         }
         if(fnameTiming) 
         {
            free(fnameTiming);
            fnameTiming = nullptr;
         }
         return; //will trigger a shutdown
      }
       
      mx::timespecAddNsec(ts, m_semWait);
      
      if(sem_timedwait(&m_swSemaphore, &ts) == 0)
      {
         if(m_writing == NOT_WRITING) continue;
         
         if(m_logSaveStart) 
         {
            derivedT::template log<saving_start>({1,m_currSaveStartFrameNo});
            m_logSaveStart = false;
         }
         
         timespec tw0, tw1, tw2;
         
         clock_gettime(CLOCK_REALTIME, &tw0);
         
         xrif_set_size(xrif, m_width, m_height, 1, (m_currSaveStop-m_currSaveStart), m_dataType);

         memcpy(xrif->raw_buffer, (char *) imageStream.array.raw + m_currSaveStart*m_width*m_height*m_typeSize, (m_currSaveStop-m_currSaveStart)*m_width*m_height*m_typeSize);
         
         for(size_t i =0; i< m_writeChunkLength; ++i)
         {
            *((uint64_t *) &m_timingData[ (sizeof(uint64_t) + 2*sizeof(timespec))*i + 0]) = imageStream.cntarray[m_currSaveStart + i];
            *((timespec *) &m_timingData[ (sizeof(uint64_t) + 2*sizeof(timespec))*i + sizeof(uint64_t)]) = imageStream.atimearray[m_currSaveStart + i];
            *((timespec *) &m_timingData[ (sizeof(uint64_t) + 2*sizeof(timespec))*i + sizeof(uint64_t)+sizeof(timespec)]) = imageStream.writetimearray[m_currSaveStart + i];
         }
         
         
         xrif->lz4_acceleration=50;
         xrif_encode(xrif);
      
         xrif_write_header( xrif_header, xrif);

         tm uttime;//The broken down time.
   
         //This needs to be first iamge time stamp.
         if(gmtime_r(&ts.tv_sec, &uttime) == 0)
         {
            derivedT::template log<software_critical>({__FILE__,__LINE__,errno,0,"gmtime_r"}); 
            if(fname) 
            {
               free(fname);
               fname = nullptr;
            }
            if(fnameTiming) 
            {
               free(fnameTiming);
               fnameTiming = nullptr;
            }
            return; //will trigger a shutdown
         }
            
         snprintf(fname + fnameBase.size(), 24, "%04i%02i%02i%02i%02i%02i%09i", uttime.tm_year+1900, uttime.tm_mon+1, uttime.tm_mday, 
                                                                uttime.tm_hour, uttime.tm_min, uttime.tm_sec, static_cast<int>(ts.tv_nsec));
         snprintf(fnameTiming + fnameTimingBase.size(), 24, "%04i%02i%02i%02i%02i%02i%09i", uttime.tm_year+1900, uttime.tm_mon+1, uttime.tm_mday, 
                                                                            uttime.tm_hour, uttime.tm_min, uttime.tm_sec, static_cast<int>(ts.tv_nsec));
         
         //Need efficient way to make time file name too
         
         (fname + fnameBase.size())[23] = '.';
         (fnameTiming + fnameTimingBase.size())[23] = '.';
         
         std::cerr << "Would write to: " << fname << "\n";
         std::cerr << "           and: " << fnameTiming << "\n";
         
         clock_gettime(CLOCK_REALTIME, &tw1);
         
         FILE * fp_xrif = fopen(fname, "wb");
         if(fp_xrif == NULL)
         {
            derivedT::template log<software_critical>({__FILE__,__LINE__,errno,0,"failed to open file for writing"}); 
            if(fname) 
            {
               free(fname);
               fname = nullptr;
            }
            if(fnameTiming) 
            {
               free(fnameTiming);
               fnameTiming = nullptr;
            }
            return; //will trigger a shutdown
         }
         
         
         
         size_t bw = fwrite(xrif_header, sizeof(uint8_t), XRIF_HEADER_SIZE, fp_xrif);
         
         if(bw != XRIF_HEADER_SIZE)
         {
            derivedT::template log<software_critical>({__FILE__,__LINE__,errno,0,"failure writing header to file"}); 
            fclose(fp_xrif);
            
            if(fname) 
            {
               free(fname);
               fname = nullptr;
            }
            if(fnameTiming) 
            {
               free(fnameTiming);
               fnameTiming = nullptr;
            }
            return; //will trigger a shutdown
         }
         
         bw = fwrite(xrif->raw_buffer, sizeof(uint8_t), xrif->compressed_size, fp_xrif);

         if(bw != xrif->compressed_size)
         {
            derivedT::template log<software_critical>({__FILE__,__LINE__,errno,0,"failure writing data to file"}); 
            fclose(fp_xrif);
            
            if(fname) 
            {
               free(fname);
               fname = nullptr;
            }
            if(fnameTiming) 
            {
               free(fnameTiming);
               fnameTiming = nullptr;
            }
            return; //will trigger a shutdown
         }
         
         fclose(fp_xrif);
         
         
         FILE * fp_time = fopen(fnameTiming, "wb");
         if(fp_time == NULL)
         {
            derivedT::template log<software_critical>({__FILE__,__LINE__,errno,0,"failed to open timing file for writing"}); 
            if(fname) 
            {
               free(fname);
               fname = nullptr;
            }
            if(fnameTiming) 
            {
               free(fnameTiming);
               fnameTiming = nullptr;
            }
            return; //will trigger a shutdown
         }
         
         bw = fwrite(m_timingData, (sizeof(uint64_t) + 2*sizeof(timespec)), (m_currSaveStop-m_currSaveStart), fp_time);
         
         if(bw != (sizeof(uint64_t) + 2*sizeof(timespec)) * (m_currSaveStop-m_currSaveStart))
         {
            derivedT::template log<software_critical>({__FILE__,__LINE__,errno,0,"failure writing data to file"}); 
            fclose(fp_time);
            if(fname) 
            {
               free(fname);
               fname = nullptr;
            }
            if(fnameTiming) 
            {
               free(fnameTiming);
               fnameTiming = nullptr;
            }
            return; //will trigger a shutdown
         }

         clock_gettime(CLOCK_REALTIME, &tw2);
         
         double wt = ( (double) tw2.tv_sec + ((double) tw2.tv_nsec)/1e9) - ( (double) tw1.tv_sec + ((double) tw1.tv_nsec)/1e9);

         std::cerr << wt << "\n";
         
         //ssize_t bw = write(fd, xrif->raw_buffer, xrif->compressed_size);
         //close fd_xrif
         //open fd_timing
         //close fd_timing
         
         
         if(m_writing == STOP_WRITING) 
         {
            m_writing = NOT_WRITING;
            derivedT::template log<saving_stop>({0,m_currSaveStopFrameNo});
         }
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
   
   
   if(fname) 
   {
      free(fname);
      fname = nullptr;
   }
   if(fnameTiming) 
   {
      free(fnameTiming);
      fnameTiming = nullptr;
   }
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
   
   indi::updateIfChanged(m_indiP_shmimName, "name", m_shmimName, m_parent->m_indiDriver);                     
   indi::updateIfChanged(m_indiP_frameSize, "width", m_width, m_parent->m_indiDriver);
   indi::updateIfChanged(m_indiP_frameSize, "height", m_height, m_parent->m_indiDriver);
   
   //Only update this if not changing
   if(m_writing == NOT_WRITING || m_writing == WRITING)
   {
      indi::updateIfChanged(m_indiP_writing, "current", (int) (m_writing == WRITING), m_parent->m_indiDriver);
      
      if(xrif && m_writing == WRITING)
      {
         indi::updateIfChanged(m_indiP_xrifStats, "ratio", xrif->compression_ratio, m_parent->m_indiDriver);
         
         indi::updateIfChanged(m_indiP_xrifStats, "encodeMBsec", xrif->encode_rate/1048576.0, m_parent->m_indiDriver);
         indi::updateIfChanged(m_indiP_xrifStats, "encodeFPS", xrif->encode_rate/(m_width*m_height*m_typeSize), m_parent->m_indiDriver);
         
         indi::updateIfChanged(m_indiP_xrifStats, "differenceMBsec", xrif->difference_rate/1048576.0, m_parent->m_indiDriver);
         indi::updateIfChanged(m_indiP_xrifStats, "differenceFPS", xrif->difference_rate/(m_width*m_height*m_typeSize), m_parent->m_indiDriver);
         
         indi::updateIfChanged(m_indiP_xrifStats, "reorderMBsec", xrif->reorder_rate/1048576.0, m_parent->m_indiDriver);
         indi::updateIfChanged(m_indiP_xrifStats, "reorderFPS", xrif->reorder_rate/(m_width*m_height*m_typeSize), m_parent->m_indiDriver);

         indi::updateIfChanged(m_indiP_xrifStats, "compressMBsec", xrif->compress_rate/1048576.0, m_parent->m_indiDriver);
         indi::updateIfChanged(m_indiP_xrifStats, "compressFPS", xrif->compress_rate/(m_width*m_height*m_typeSize), m_parent->m_indiDriver);
      }
      else
      {
         indi::updateIfChanged(m_indiP_xrifStats, "ratio", 0.0, m_parent->m_indiDriver);
         indi::updateIfChanged(m_indiP_xrifStats, "encodeMBsec", 0.0, m_parent->m_indiDriver);
         indi::updateIfChanged(m_indiP_xrifStats, "encodeFPS", 0.0, m_parent->m_indiDriver);
         indi::updateIfChanged(m_indiP_xrifStats, "differenceMBsec", 0.0, m_parent->m_indiDriver);
         indi::updateIfChanged(m_indiP_xrifStats, "differenceFPS", 0.0, m_parent->m_indiDriver);
         indi::updateIfChanged(m_indiP_xrifStats, "reorderMBsec", 0.0, m_parent->m_indiDriver);
         indi::updateIfChanged(m_indiP_xrifStats, "reorderFPS", 0.0, m_parent->m_indiDriver);
         indi::updateIfChanged(m_indiP_xrifStats, "compressMBsec", 0.0, m_parent->m_indiDriver);
         indi::updateIfChanged(m_indiP_xrifStats, "compressFPS", 0.0, m_parent->m_indiDriver);
      }
   }
   
   return 0;
}


} //namespace dev
} //namespace app
} //namespace MagAOX
#endif
