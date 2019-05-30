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
    //Configures the camera for acquistion, must also set m_width, m_height, and m_dataType
    //so that the share memory can be allocated
    int derivedT::configureAcquisition();

    //Start acquisition.
    int derivedT::startAcquisition();
    
    //Acquires the data, and checks if it is valid.
    //This should set m_currImageTimestamp to the image timestamp.
    // returns 0 if valid, < 0 on error, > 0 on no data.
    int derivedT::acquireAndCheckValid()
    
    //Loads the acquired image into the stream, copying it to the appropriate member of imageStream.array.
    //This could simply be a memcpy.
    int derivedT::loadImageIntoStream(void * dest);
    
    //Take any actions needed to reconfigure the system.  Called if m_reconfig is set to true.
    int derivedT::reconfig()
  * \endcode  
  * Each of the above functions should return 0 on success, and -1 on an error. 
  * For `acquireAndCheckValid` >0 will indicate no data but not an error.  In most cases, 
  * an appropriate state code, such as NOTCONNECTED, should be set as well.
  *
  *
  * Calls to this class's `setupConfig`, `loadConfig`, `appStartup`, `appLogic` and `appShutdown`
  * functions must be placed in the derived class's functions of the same name.
  */
template<class derivedT>
class frameGrabber 
{
protected:

   /** \name Configurable Parameters
    * @{
    */
   std::string m_shmimName {""}; ///< The name of the shared memory image, is used in `/tmp/<shmimName>.im.shm`. Derived classes should set a default.
      
   int m_fgThreadPrio {2}; ///< Priority of the framegrabber thread, should normally be > 00.
    
   uint32_t m_circBuffLength {1024}; ///< Length of the circular buffer, in frames
   
   uint32_t m_writeChunkLength {512}; ///< The number of images to write at a time.  Should normally be < m_circBuffLength.
   
   std::string m_rawimageDir; ///< The path where files will be saved.  Normally derived from the library config.
   
    
   ///@}
   
   uint32_t m_width {0}; ///< The width of the image, once deinterlaced etc.
   uint32_t m_height {0}; ///< The height of the image, once deinterlaced etc.
   
   uint8_t m_dataType{0}; ///< The ImageStreamIO type code.
   size_t m_typeSize {0}; ///< The size of the type, in bytes.  Result of sizeof.
   
   int m_xbinning {0}; ///< The x-binning according to the framegrabber
   int m_ybinning {0}; ///< The y-binning according to the framegrabber
   std::string m_cameraType; ///< The camera type according to the framegrabber
          
   timespec m_currImageTimestamp {0,0}; ///< The timestamp of the current image.
   
   bool m_reconfig {false}; ///< Flag to set if a camera reconfiguration requires a framegrabber reset.
   
   IMAGE imageStream; ///< The ImageStreamIO shared memory buffer.
   
   timespec m_dummy_ts {0,0};
   uint64_t m_dummy_cnt {0};
   char m_dummy_c {0};
   
   
   
   
public:

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

   ///Thread starter, called by MagAOXApp::threadStart on thread construction.  Calls fgThreadExec.
   static void fgThreadStart( frameGrabber * o /**< [in] a pointer to a frameGrabber instance (normally this) */);

   /// Execute framegrabbing.
   void fgThreadExec();

   
   ///@}
  
   
    
   
    /** \name INDI 
      *
      *@{
      */ 
protected:
   //declare our properties
   
   pcf::IndiProperty m_indiP_shmimName; ///< Property used to report the shmim buffer name
   
   pcf::IndiProperty m_indiP_frameSize; ///< Property used to report the current frame size

public:

   /// Update the INDI properties for this device controller
   /** You should call this once per main loop.
     * It is not called automatically.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int updateINDI();

   ///@}
   
private:
   derivedT & impl()
   {
      return *static_cast<derivedT *>(this);
   }
};

template<class derivedT>
void frameGrabber<derivedT>::setupConfig(mx::app::appConfigurator & config)
{
   config.add("framegrabber.threadPrio", "", "framegrabber.threadPrio", argType::Required, "framegrabber", "threadPrio", false, "int", "The real-time priority of the fraemgrabber thread.");
   config.add("framegrabber.shmimName", "", "framegrabber.shmimName", argType::Required, "framegrabber", "shmimName", false, "string", "The name of the ImageStreamIO shared memory image. Will be used as /tmp/<shmimName>.im.shm.");
   
   config.add("framegrabber.circBuffLength", "", "framegrabber.circBuffLength", argType::Required, "framegrabber", "circBuffLength", false, "size_t", "The length of the circular buffer. Sets m_circBuffLength, default is 1.");

   config.add("framegrabber.savePath", "", "framegrabber.savePath", argType::Required, "framegrabber", "savePath", false, "string", "The absolute path where images are saved. Will use MagAO-X default if not set.");
      
}

template<class derivedT>
void frameGrabber<derivedT>::loadConfig(mx::app::appConfigurator & config)
{
   config(m_fgThreadPrio, "framegrabber.threadPrio");
   m_shmimName = impl().configName();
   config(m_shmimName, "framegrabber.shmimName");
  
   ///\todo handle 0
   config(m_circBuffLength, "framegrabber.circBuffLength");
   
   }
   

template<class derivedT>
int frameGrabber<derivedT>::appStartup()
{
   //Register the shmimName INDI property
   m_indiP_shmimName = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_shmimName.setDevice(impl().configName());
   m_indiP_shmimName.setName("shmimName");
   m_indiP_shmimName.setPerm(pcf::IndiProperty::ReadWrite);
   m_indiP_shmimName.setState(pcf::IndiProperty::Idle);
   m_indiP_shmimName.add(pcf::IndiElement("name"));
   m_indiP_shmimName["name"] = m_shmimName;
   
   if( impl().registerIndiPropertyNew( m_indiP_shmimName, nullptr) < 0)
   {
      #ifndef FRAMEGRABBER_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the frameSize INDI property
   m_indiP_frameSize = pcf::IndiProperty(pcf::IndiProperty::Number);
   m_indiP_frameSize.setDevice(impl().configName());
   m_indiP_frameSize.setName("frameSize");
   m_indiP_frameSize.setPerm(pcf::IndiProperty::ReadWrite);
   m_indiP_frameSize.setState(pcf::IndiProperty::Idle);
   m_indiP_frameSize.add(pcf::IndiElement("width"));
   m_indiP_frameSize["width"] = 0;
   m_indiP_frameSize.add(pcf::IndiElement("height"));
   m_indiP_frameSize["height"] = 0;
   
   if( impl().registerIndiPropertyNew( m_indiP_frameSize, nullptr) < 0)
   {
      #ifndef FRAMEGRABBER_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   if(impl().threadStart( m_fgThread, m_fgThreadInit, m_fgThreadPrio, "framegrabber", this, fgThreadStart) < 0)
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
   
   return 0;

}


template<class derivedT>
int frameGrabber<derivedT>::appShutdown()
{
   if(m_fgThread.joinable())
   {
      try
      {
         m_fgThread.join(); //this will throw if it was already joined
      }
      catch(...)
      {
      }
   }
   
   
   
   return 0;
}



template<class derivedT>
void frameGrabber<derivedT>::fgThreadStart( frameGrabber * o)
{
   o->fgThreadExec();
}


template<class derivedT>
void frameGrabber<derivedT>::fgThreadExec()
{
   timespec writestart;
   
   //Wait fpr the thread starter to finish initializing this thread.
   while(m_fgThreadInit == true && impl().shutdown() == 0)
   {
       sleep(1);
   }
   
   uint32_t imsize[3] = {0,0,0};
   std::string shmimName;
   
   while(impl().shutdown() == 0)
   {
      while(!impl().shutdown() && (!( impl().state() == stateCodes::READY || impl().state() == stateCodes::OPERATING) || impl().powerState() <= 0 ) )
      {
         sleep(1);
      }
      
      if(impl().shutdown()) continue;
      else 
      {
         //At the end of this, must have m_width, m_height, m_dataType set.
         if(impl().configureAcquisition() < 0) continue;        
         
         m_typeSize = ImageStreamIO_typesize(m_dataType);
      }

      /* Initialize ImageStreamIO
       */
      if(m_shmimName == "") m_shmimName = impl().configName();

      if(m_width != imsize[0] || m_height != imsize[1] || m_circBuffLength != imsize[2] || m_shmimName != shmimName)
      {
         imsize[0] = m_width; 
         imsize[1] = m_height;
         imsize[2] = m_circBuffLength;
         shmimName = m_shmimName;
      
         std::cerr << "Creating: " << m_shmimName << " " << m_width << " " << m_height << " " << m_circBuffLength << "\n";
      
         ImageStreamIO_createIm_gpu(&imageStream, m_shmimName.c_str(), 3, imsize, m_dataType, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL);
       
         imageStream.md->cnt1 = m_circBuffLength;
      }
      else
      {
         std::cerr << "Not creating . . .\n";
      }
     
       
      //This completes the reconfiguration.
      m_reconfig = false;
                  
      //At the end of this, must have m_width, m_height, m_dataType set.
      if(impl().startAcquisition() < 0) continue;       
         
      uint64_t next_cnt1 = 0; 
      char * next_dest = (char *) imageStream.array.raw;
      timespec * next_wtimearr = &imageStream.writetimearray[0];
      timespec * next_atimearr = &imageStream.atimearray[0];
      uint64_t * next_cntarr = &imageStream.cntarray[0];
      
      //This is the main image grabbing loop.      
      while(!impl().shutdown() && !m_reconfig && impl().powerState() > 0)
      {
         //==================
         //Get next image, process validity.
         //====================         
         int isValid = impl().acquireAndCheckValid();
         if(isValid != 0)
         {
            if( isValid < 0)
            {
               break;
            }
            else if( isValid > 0)
            {
               continue;
            }
         }
         
         //Ok, no timeout, so we process the image and publish it.
         imageStream.md->write=1;
         
         //Set the time of last write
         clock_gettime(CLOCK_REALTIME, &writestart);
         
         if(impl().loadImageIntoStream(next_dest) < 0) 
         {
            break;
         }
         
         //Set the time of last write
         clock_gettime(CLOCK_REALTIME, &imageStream.md->writetime);

         //Set the image acquisition timestamp
         imageStream.md->atime = m_currImageTimestamp;
         
         //Update cnt1
         imageStream.md->cnt1 = next_cnt1;
          
         //Update cnt0
         imageStream.md->cnt0++;
         
         *next_wtimearr = imageStream.md->writetime;
         *next_atimearr = m_currImageTimestamp;
         *next_cntarr = imageStream.md->cnt0;
         
         //And post
         imageStream.md->write=0;
         ImageStreamIO_sempost(&imageStream,-1);
 
         
         if(imageStream.md[0].cnt0 % 2000 == 0)
         {
            std::cerr << ( (double) imageStream.md->writetime.tv_sec + ((double) imageStream.md->writetime.tv_nsec)/1e9) - ( (double) imageStream.md->atime.tv_sec + ((double) imageStream.md->atime.tv_nsec)/1e9) << " ";
            std::cerr << ( (double) imageStream.md->writetime.tv_sec + ((double) imageStream.md->writetime.tv_nsec)/1e9) - ( (double) writestart.tv_sec + ((double) writestart.tv_nsec)/1e9) << "\n";

         }
         
         //Now we increment pointers outside the time-critical part of the loop.
         next_cnt1 = imageStream.md->cnt1+1;
         if(next_cnt1 >= m_circBuffLength) next_cnt1 = 0;
         
         next_dest = (char *) imageStream.array.raw + next_cnt1*m_width*m_height*m_typeSize;
         next_wtimearr = &imageStream.writetimearray[next_cnt1];
         next_atimearr = &imageStream.atimearray[next_cnt1];
         next_cntarr = &imageStream.cntarray[next_cnt1];
         
         //Touch them to make sure we move
         m_dummy_c = next_dest[0];
         m_dummy_ts.tv_sec = next_wtimearr[0].tv_sec + next_atimearr[0].tv_sec;
         m_dummy_cnt = next_cntarr[0];
         
         
      }
    
      if(m_reconfig && !impl().shutdown())
      {
         impl().reconfig();
      }

   } //outer loop, will exit if m_shutdown==true

   ImageStreamIO_destroyIm( &imageStream );
   
}





template<class derivedT>
int frameGrabber<derivedT>::updateINDI()
{
   if( !impl().m_indiDriver ) return 0;
   
   indi::updateIfChanged(m_indiP_shmimName, "name", m_shmimName, impl().m_indiDriver);                     
   indi::updateIfChanged(m_indiP_frameSize, "width", m_width, impl().m_indiDriver);
   indi::updateIfChanged(m_indiP_frameSize, "height", m_height, impl().m_indiDriver);
   
   
   return 0;
}


} //namespace dev
} //namespace app
} //namespace MagAOX
#endif
