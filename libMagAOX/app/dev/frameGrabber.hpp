/** \file frameGrabber.hpp
  * \brief The MagAO-X generic frame grabber.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup app_files
  */

#ifndef frameGrabber_hpp
#define frameGrabber_hpp

#include <sys/syscall.h>
       
#include <mx/sigproc/circularBuffer.hpp>
#include <mx/math/vectorUtils.hpp>
#include <mx/improc/imageUtils.hpp>

#include <ImageStreamIO/ImageStruct.h>
#include <ImageStreamIO/ImageStreamIO.h>

#include "../../common/paths.hpp"


namespace MagAOX
{
namespace app
{
namespace dev 
{
   



/** MagAO-X generic frame grabber
  *
  * 
  * The derived class `derivedT` must expose the following interface
  * \code 
    //Configures the camera for acquistion, must also set m_width, m_height, and m_dataType
    //so that the shared memory can be allocated
    int derivedT::configureAcquisition();

    //Gets the frames-per-second readout rate 
    //used for the latency statistics
    float derivedT::fps();
    
    //Start acquisition.
    int derivedT::startAcquisition();
    
    //Acquires the data, and checks if it is valid.
    //This should set m_currImageTimestamp to the image timestamp.
    // returns 0 if valid, < 0 on error, > 0 on no data.
    int derivedT::acquireAndCheckValid()
    
    //Loads the acquired image into the stream, copying it to the appropriate member of m_imageStream->array.
    //This could simply be a memcpy.
    int derivedT::loadImageIntoStream(void * dest);
    
    //Take any actions needed to reconfigure the system.  Called if m_reconfig is set to true.
    int derivedT::reconfig()
  * \endcode  
  * Each of the above functions should return 0 on success, and -1 on an error (except fps). 
  * For `acquireAndCheckValid` >0 will indicate no data but not an error.  In most cases, 
  * an appropriate state code, such as NOTCONNECTED, should be set as well.
  *
  * A static configuration variable must be defined in derivedT as
  * \code
  * static constexpr bool c_frameGrabber_flippable =true; //or: false
  * \endcode
  * which determines whether or not the images can be flipped programatically.
  *
  * Calls to this class's `setupConfig`, `loadConfig`, `appStartup`, `appLogic` and `appShutdown`
  * functions must be placed in the derived class's functions of the same name.
  *
  * \ingroup appdev
  */
template<class derivedT>
class frameGrabber 
{
public:
   enum fgFlip { fgFlipNone, fgFlipUD, fgFlipLR, fgFlipUDLR };
   
protected:

   /** \name Configurable Parameters
    * @{
    */
   std::string m_shmimName {""}; ///< The name of the shared memory image, is used in `/tmp/<shmimName>.im.shm`. Derived classes should set a default.
      
   int m_fgThreadPrio {2}; ///< Priority of the framegrabber thread, should normally be > 00.
   std::string m_fgCpuset; ///< The cpuset to assign the framegrabber thread to.  Not used if empty, the default.

   uint32_t m_circBuffLength {1}; ///< Length of the circular buffer, in frames
       
   uint16_t m_latencyCircBuffMaxLength {3600}; ///< Maximum length of the latency measurement circular buffers
   float m_latencyCircBuffMaxTime {5}; ///< Maximum time of the latency meaurement circular buffers
   
   int m_defaultFlip {fgFlipNone};
   
   ///@}
   
   int m_currentFlip {fgFlipNone};
   
   uint32_t m_width {0}; ///< The width of the image, once deinterlaced etc.
   uint32_t m_height {0}; ///< The height of the image, once deinterlaced etc.
   
   uint8_t m_dataType{0}; ///< The ImageStreamIO type code.
   size_t m_typeSize {0}; ///< The size of the type, in bytes.  Result of sizeof.
   
   int m_xbinning {0}; ///< The x-binning according to the framegrabber
   int m_ybinning {0}; ///< The y-binning according to the framegrabber
   
          
   timespec m_currImageTimestamp {0,0}; ///< The timestamp of the current image.
   
   bool m_reconfig {false}; ///< Flag to set if a camera reconfiguration requires a framegrabber reset.
   
   IMAGE * m_imageStream {nullptr}; ///< The ImageStreamIO shared memory buffer.
   
   typedef uint16_t cbIndexT;
   
   mx::sigproc::circularBufferIndex<timespec, cbIndexT> m_atimes;
   mx::sigproc::circularBufferIndex<timespec, cbIndexT> m_wtimes;
   
   std::vector<double> m_atimesD;
   std::vector<double> m_wtimesD;
   std::vector<double> m_watimesD;
   
   timespec m_dummy_ts {0,0};
   uint64_t m_dummy_cnt {0};
   char m_dummy_c {0};
   
   double m_mna;
   double m_vara;  
         
   double m_mnw;   
   double m_varw;  
         
   double m_mnwa;  
   double m_varwa; 
   
   
   
   
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

   /// On power off, sets m_reconfig to true.
   /** This should be called in `derivedT::onPowerOff` as
     * \code
       framegrabber<derivedT>::onPowerOff();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int onPowerOff();
   
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
   
   pid_t m_fgThreadID {0}; ///< The ID of the framegrabber thread.
   
   pcf::IndiProperty m_fgThreadProp; ///< The property to hold the f.g. thread details.
   
   std::thread m_fgThread; ///< A separate thread for the actual framegrabbings

   ///Thread starter, called by MagAOXApp::threadStart on thread construction.  Calls fgThreadExec.
   static void fgThreadStart( frameGrabber * o /**< [in] a pointer to a frameGrabber instance (normally this) */);

   /// Execute framegrabbing.
   void fgThreadExec();

   
   ///@}
  
   void * loadImageIntoStreamCopy( void * dest,
                                   void * src,
                                   size_t width,
                                   size_t height,
                                   size_t szof
                                 );
    
   
    /** \name INDI 
      *
      *@{
      */ 
protected:
   //declare our properties
   
   pcf::IndiProperty m_indiP_shmimName; ///< Property used to report the shmim buffer name
   
   pcf::IndiProperty m_indiP_frameSize; ///< Property used to report the current frame size

   pcf::IndiProperty m_indiP_timing;
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
   
   /** \name Telemeter Interface 
     * @{
     */

   int recordFGTimings( bool force = false );

   /// @}

private:
   derivedT & derived()
   {
      return *static_cast<derivedT *>(this);
   }
};

template<class derivedT>
void frameGrabber<derivedT>::setupConfig(mx::app::appConfigurator & config)
{
   config.add("framegrabber.threadPrio", "", "framegrabber.threadPrio", argType::Required, "framegrabber", "threadPrio", false, "int", "The real-time priority of the framegrabber thread.");

   config.add("framegrabber.cpuset", "", "framegrabber.cpuset", argType::Required, "framegrabber", "cpuset", false, "string", "The cpuset to assign the framegrabber thread to.");

   config.add("framegrabber.shmimName", "", "framegrabber.shmimName", argType::Required, "framegrabber", "shmimName", false, "string", "The name of the ImageStreamIO shared memory image. Will be used as /milk/shm/<shmimName>.im.shm.");
   
   config.add("framegrabber.circBuffLength", "", "framegrabber.circBuffLength", argType::Required, "framegrabber", "circBuffLength", false, "size_t", "The length of the circular buffer. Sets m_circBuffLength, default is 1.");

   config.add("framegrabber.latencyTime", "", "framegrabber.latencyTime", argType::Required, "framegrabber", "latencyTime", false, "float", "The maximum length of time to measure latency timings. Sets  m_latencyCircBuffMaxTime, default is 5.");

   config.add("framegrabber.latencySize", "", "framegrabber.latencySize", argType::Required, "framegrabber", "latencySize", false, "float", "The maximum length of the buffer used to measure latency timings. Sets  m_latencyCircBuffMaxLength, default is 3600.");


   if(derivedT::c_frameGrabber_flippable)
   {
      config.add("framegrabber.defaultFlip", "", "framegrabber.defaultFlip", argType::Required, "framegrabber", "defaultFlip", false, "string", "The default flip of the image.  Options are flipNone, flipUD, flipLR, flipUDLR.  The default is flipNone.");
   }
}

template<class derivedT>
void frameGrabber<derivedT>::loadConfig(mx::app::appConfigurator & config)
{
   config(m_fgThreadPrio, "framegrabber.threadPrio");
   config(m_fgCpuset, "framegrabber.cpuset");
   if(m_shmimName == "") m_shmimName = derived().configName();
   config(m_shmimName, "framegrabber.shmimName");
  
   config(m_circBuffLength, "framegrabber.circBuffLength");

   if(m_circBuffLength < 1)
   {
      m_circBuffLength = 1;
      derivedT::template log<text_log>("circBuffLength set to 1");
   }
   
   config(m_latencyCircBuffMaxTime, "framegrabber.latencyTime");
   if(m_latencyCircBuffMaxTime < 0)
   {
      m_latencyCircBuffMaxTime = 0;
      derivedT::template log<text_log>("latencyTime set to 0 (off)");
   }

   config(m_latencyCircBuffMaxLength, "framegrabber.latencySize");
   

   if(derivedT::c_frameGrabber_flippable)
   {
      std::string flip = "flipNone";
      config(flip, "framegrabber.defaultFlip");
      if(flip == "flipNone")
      {
         m_defaultFlip = fgFlipNone;
      }
      else if(flip == "flipUD")
      {
         m_defaultFlip = fgFlipUD;
      }
      else if(flip == "flipLR")
      {
         m_defaultFlip = fgFlipLR;
      }
      else if(flip == "flipUDLR")
      {
         m_defaultFlip = fgFlipUDLR;
      }
      else
      {
         derivedT::template log<text_log>({std::string("invalid framegrabber flip specification (") + flip + "), setting flipNone"}, logPrio::LOG_ERROR);
         m_defaultFlip = fgFlipNone;
      }
   }
}
   

template<class derivedT>
int frameGrabber<derivedT>::appStartup()
{
   //Register the shmimName INDI property
   m_indiP_shmimName = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_shmimName.setDevice(derived().configName());
   m_indiP_shmimName.setName("fg_shmimName");
   m_indiP_shmimName.setPerm(pcf::IndiProperty::ReadOnly);
   m_indiP_shmimName.setState(pcf::IndiProperty::Idle);
   m_indiP_shmimName.add(pcf::IndiElement("name"));
   m_indiP_shmimName["name"] = m_shmimName;
   
   if( derived().registerIndiPropertyNew( m_indiP_shmimName, nullptr) < 0)
   {
      #ifndef FRAMEGRABBER_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the frameSize INDI property
   m_indiP_frameSize = pcf::IndiProperty(pcf::IndiProperty::Number);
   m_indiP_frameSize.setDevice(derived().configName());
   m_indiP_frameSize.setName("fg_frameSize");
   m_indiP_frameSize.setPerm(pcf::IndiProperty::ReadOnly);
   m_indiP_frameSize.setState(pcf::IndiProperty::Idle);
   m_indiP_frameSize.add(pcf::IndiElement("width"));
   m_indiP_frameSize["width"] = 0;
   m_indiP_frameSize.add(pcf::IndiElement("height"));
   m_indiP_frameSize["height"] = 0;
   
   if( derived().registerIndiPropertyNew( m_indiP_frameSize, nullptr) < 0)
   {
      #ifndef FRAMEGRABBER_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the timing INDI property
   derived().createROIndiNumber( m_indiP_timing, "fg_timing");
   m_indiP_timing.add(pcf::IndiElement("acq_fps"));
   m_indiP_timing.add(pcf::IndiElement("acq_jitter"));
   m_indiP_timing.add(pcf::IndiElement("write_fps"));
   m_indiP_timing.add(pcf::IndiElement("write_jitter"));
   m_indiP_timing.add(pcf::IndiElement("delta_aw"));
   m_indiP_timing.add(pcf::IndiElement("delta_aw_jitter"));

   if( derived().registerIndiPropertyReadOnly( m_indiP_timing ) < 0)
   {
      #ifndef STDCAMERA_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }

   //Start the f.g. thread
   if(derived().threadStart( m_fgThread, m_fgThreadInit, m_fgThreadID, m_fgThreadProp, m_fgThreadPrio, m_fgCpuset, "framegrabber", this, fgThreadStart) < 0)
   {
      derivedT::template log<software_error, -1>({__FILE__, __LINE__});
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
   
   if( derived().state() == stateCodes::OPERATING && m_atimes.size() > 0 )
   {
      if(m_atimes.size() >= m_atimes.maxEntries())
      {
         cbIndexT refEntry = m_atimes.earliest();
         
         m_atimesD.resize(m_atimes.maxEntries()-1);
         m_wtimesD.resize(m_wtimes.maxEntries()-1);
         m_watimesD.resize(m_wtimes.maxEntries()-1);
         
         double a0 = m_atimes.at(refEntry, 0).tv_sec + ((double) m_atimes.at(refEntry, 0).tv_nsec)/1e9;
         double w0 = m_wtimes.at(refEntry, 0).tv_sec + ((double) m_wtimes.at(refEntry, 0).tv_nsec)/1e9;
         for(size_t n=1; n <= m_atimesD.size(); ++n)
         {
            double a = m_atimes.at(refEntry, n).tv_sec + ((double) m_atimes.at(refEntry, n).tv_nsec)/1e9;
            double w = m_wtimes.at(refEntry, n).tv_sec + ((double) m_wtimes.at(refEntry, n).tv_nsec)/1e9;
            m_atimesD[n-1] = a - a0;
            m_wtimesD[n-1] = w - w0;
            m_watimesD[n-1] = w - a;
            a0 = a;
            w0 = w;
         }
         
         m_mna = mx::math::vectorMean(m_atimesD);
         m_vara = mx::math::vectorVariance(m_atimesD, m_mna);
         
         m_mnw = mx::math::vectorMean(m_wtimesD);
         m_varw = mx::math::vectorVariance(m_wtimesD, m_mnw);
         
         m_mnwa = mx::math::vectorMean(m_watimesD);
         m_varwa = mx::math::vectorVariance(m_watimesD, m_mnwa);
         
         recordFGTimings();
      }
      else
      {
         m_mna = 0;
         m_vara = 0;
         m_mnw = 0;
         m_varw = 0;
         m_mnwa = 0;
         m_varwa = 0;
      }
   }
   else
   {
      m_mna = 0;
      m_vara = 0;
      m_mnw = 0;
      m_varw = 0;
      m_mnwa = 0;
      m_varwa = 0;
   }

   return 0;

}

template<class derivedT>
int frameGrabber<derivedT>::onPowerOff()
{ 
   m_mna = 0;
   m_vara = 0;
   m_mnw = 0;  
   m_varw = 0;
   m_mnwa = 0;
   m_varwa = 0;

   m_width = 0;
   m_height = 0;

   updateINDI();
   
   m_reconfig = true;


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
   //Get the thread PID immediately so the caller can return.
   m_fgThreadID = syscall(SYS_gettid);
   
   timespec writestart;
   
   //Wait fpr the thread starter to finish initializing this thread.
   while(m_fgThreadInit == true && derived().shutdown() == 0)
   {
       sleep(1);
   }
   
   uint32_t imsize[3] = {0,0,0};
   std::string shmimName;
   
   while(derived().shutdown() == 0)
   {
      ///\todo this ought to wait until OPERATING, using READY as a sign of "not integrating"
      while(!derived().shutdown() && (!( derived().state() == stateCodes::READY || derived().state() == stateCodes::OPERATING) || derived().powerState() <= 0 ) )
      {
         sleep(1);
      }
      
      if(derived().shutdown()) continue;
      else 
      {
         //At the end of this, must have m_width, m_height, m_dataType set, and derived()->fps must be valid.
         if(derived().configureAcquisition() < 0) continue;        
         
         if(m_latencyCircBuffMaxLength == 0 || m_latencyCircBuffMaxTime == 0)
         {
            m_atimes.maxEntries(0);
            m_wtimes.maxEntries(0);
         }
         else 
         {
            //Set up the latency circ. buffs
            cbIndexT cbSz = m_latencyCircBuffMaxTime * derived().fps();
            if(cbSz > m_latencyCircBuffMaxLength) cbSz = m_latencyCircBuffMaxLength;
            if(cbSz < 3) cbSz = 3; //Make variance meaningful
            m_atimes.maxEntries(cbSz);
            m_wtimes.maxEntries(cbSz);
         }
            
         m_typeSize = ImageStreamIO_typesize(m_dataType);
         

         //Here we resolve currentFlip somehow.
         m_currentFlip = m_defaultFlip;
      }

      /* Initialize ImageStreamIO
       */
      if(m_shmimName == "") m_shmimName = derived().configName();

      if(m_width != imsize[0] || m_height != imsize[1] || m_circBuffLength != imsize[2] || m_shmimName != shmimName || m_imageStream == nullptr)
      {
         if(m_imageStream != nullptr)
         {
            ImageStreamIO_destroyIm(m_imageStream);
            free(m_imageStream);
         }
         
         m_imageStream = (IMAGE *) malloc(sizeof(IMAGE));
         
         imsize[0] = m_width; 
         imsize[1] = m_height;
         imsize[2] = m_circBuffLength;
         shmimName = m_shmimName;
      
         std::cerr << "Creating: " << m_shmimName << " " << m_width << " " << m_height << " " << m_circBuffLength << "\n";
      
         ImageStreamIO_createIm_gpu(m_imageStream, m_shmimName.c_str(), 3, imsize, m_dataType, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL, 0);

         m_imageStream->md->cnt1 = m_circBuffLength - 1;
      }
      else
      {
         std::cerr << "Not creating . . .\n";
      }
     
      //This completes the reconfiguration.
      m_reconfig = false;
                  
      if(derived().startAcquisition() < 0) continue;       
         
      uint64_t next_cnt1 = 0; 
      char * next_dest = (char *) m_imageStream->array.raw;
      timespec * next_wtimearr = &m_imageStream->writetimearray[0];
      timespec * next_atimearr = &m_imageStream->atimearray[0];
      uint64_t * next_cntarr = &m_imageStream->cntarray[0];
      
      //This is the main image grabbing loop.      
      while(!derived().shutdown() && !m_reconfig && derived().powerState() > 0)
      {
         //==================
         //Get next image, process validity.
         //====================         
         int isValid = derived().acquireAndCheckValid();
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
         m_imageStream->md->write=1;
         
         //Set the time of last write
         clock_gettime(CLOCK_REALTIME, &writestart);
         
         if(derived().loadImageIntoStream(next_dest) < 0) 
         {
            break;
         }

         //Set the time of last write
         clock_gettime(CLOCK_REALTIME, &m_imageStream->md->writetime);

         //Set the image acquisition timestamp
         m_imageStream->md->atime = m_currImageTimestamp;
         
         //Update cnt1
         m_imageStream->md->cnt1 = next_cnt1;
          
         //Update cnt0
         m_imageStream->md->cnt0++;
         
         *next_wtimearr = m_imageStream->md->writetime;
         *next_atimearr = m_currImageTimestamp;
         *next_cntarr = m_imageStream->md->cnt0;
         
         //And post
         m_imageStream->md->write=0;
         ImageStreamIO_sempost(m_imageStream,-1);
 
         //Update the latency circ. buffs
         if(m_atimes.maxEntries()  >  0)
         {
            m_atimes.nextEntry(m_imageStream->md->atime);
            m_wtimes.nextEntry(m_imageStream->md->writetime);
         }
         
         //Now we increment pointers outside the time-critical part of the loop.
         next_cnt1 = m_imageStream->md->cnt1+1;
         if(next_cnt1 >= m_circBuffLength) next_cnt1 = 0;
         
         next_dest = (char *) m_imageStream->array.raw + next_cnt1*m_width*m_height*m_typeSize;
         next_wtimearr = &m_imageStream->writetimearray[next_cnt1];
         next_atimearr = &m_imageStream->atimearray[next_cnt1];
         next_cntarr = &m_imageStream->cntarray[next_cnt1];
         
         //Touch them to make sure we move
         m_dummy_c = next_dest[0];
         m_dummy_ts.tv_sec = next_wtimearr[0].tv_sec + next_atimearr[0].tv_sec;
         m_dummy_cnt = next_cntarr[0];
      }
    
      if(m_reconfig && !derived().shutdown())
      {
         derived().reconfig();
      }

   } //outer loop, will exit if m_shutdown==true

   if(m_imageStream != nullptr)
   {
      ImageStreamIO_destroyIm( m_imageStream );
      free(m_imageStream);
      m_imageStream = nullptr;
   }
}

template<class derivedT>
void * frameGrabber<derivedT>::loadImageIntoStreamCopy( void * dest,
                                                        void * src,
                                                        size_t width,
                                                        size_t height,
                                                        size_t szof
                                                      )
{
   if(!derivedT::c_frameGrabber_flippable)
   {
      return memcpy(dest, src, width*height*szof);
   }
   else
   {
      switch(m_currentFlip)
      {
         case fgFlipNone:
            return mx::improc::imcpy(dest, src, width, height, szof);
         case fgFlipUD:
            return mx::improc::imcpy_flipUD(dest, src, width, height, szof);
         case fgFlipLR:
            return mx::improc::imcpy_flipLR(dest, src, width, height, szof);  
         case fgFlipUDLR:
            return mx::improc::imcpy_flipUDLR(dest, src, width, height, szof);
         default:
            return nullptr;
      }
   }
}



template<class derivedT>
int frameGrabber<derivedT>::updateINDI()
{
   if( !derived().m_indiDriver ) return 0;
   
   indi::updateIfChanged(m_indiP_shmimName, "name", m_shmimName, derived().m_indiDriver);                     
   indi::updateIfChanged(m_indiP_frameSize, "width", m_width, derived().m_indiDriver);
   indi::updateIfChanged(m_indiP_frameSize, "height", m_height, derived().m_indiDriver);

   double fpsa = 0;
   double fpsw = 0;
   if(m_mna != 0 ) fpsa = 1.0/m_mna;
   if(m_mnw != 0 ) fpsw = 1.0/m_mnw;

   indi::updateIfChanged<double>(m_indiP_timing, {"acq_fps","acq_jitter","write_fps","write_jitter","delta_aw","delta_aw_jitter"}, 
                        {fpsa, sqrt(m_vara), fpsw, sqrt(m_varw), m_mnwa, sqrt(m_varwa)},derived().m_indiDriver);
   
   return 0;
}

template<class derivedT>
int frameGrabber<derivedT>::recordFGTimings( bool force )
{
   static double last_mna = 0;
   static double last_vara = 0;

   static double last_mnw = 0;
   static double last_varw = 0;
   
   static double last_mnwa = 0;
   static double last_varwa = 0;

   if(force || m_mna != last_mna || m_vara != last_vara ||
                 m_mnw != last_mnw || m_varw != last_varw ||
                   m_mnwa != last_mnwa || m_varwa != last_varwa )
   {
      derived().template telem<telem_fgtimings>({m_mna, sqrt(m_vara), m_mnw, sqrt(m_varw), m_mnwa, sqrt(m_varwa)});

      last_mna = m_mna;
      last_vara = m_vara;
      last_mnw = m_mnw;
      last_varw = m_varw;
      last_mnwa = m_mnwa;
      last_varwa = m_varwa;
   }

   return 0;

}

} //namespace dev
} //namespace app
} //namespace MagAOX
#endif
