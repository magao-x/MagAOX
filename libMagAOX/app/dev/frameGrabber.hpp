/** \file frameGrabber.hpp
  * \brief The MagAO-X Princeton Instruments EMCCD camera controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup frameGrabber_files
  */

#ifndef frameGrabber_hpp
#define frameGrabber_hpp


#include <ImageStruct.h>
#include <ImageStreamIO.h>


//#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch

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

   
   
   uint32_t m_width {0}; ///< The width of the image, once deinterlaced etc.
   uint32_t m_height {0}; ///< The height of the image, once deinterlaced etc.
   uint32_t m_planes {1}; ///< The number of planes in the image circular buffer.
   
   uint8_t m_dataType{0}; ///< The ImageStreamIO type code.
   size_t m_typeSize {0}; ///< The size of the type, in bytes.  Result of sizeof.
   
   int m_xbinning {0}; ///< The x-binning according to the framegrabber
   int m_ybinning {0}; ///< The y-binning according to the framegrabber
   std::string m_cameraType; ///< The camera type according to the framegrabber
          
   timespec m_currImageTimestamp; ///< The timestamp of the current image.
   
   bool m_reconfig {false}; ///< Flag to set if a camera reconfiguration requires a framegrabber reset.
   
   IMAGE imageStream; ///< The ImageStreamIO shared memory buffer.
   
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
   
   std::string m_shmimName {""}; ///< The name of the shared memory image, is used in `/tmp/<shmimName>.im.shm`. Derived classes should set a default.
      
   int m_fgThreadPrio {1}; ///< Priority of the framegrabber thread, should normally be > 00.

   std::thread m_fgThread; ///< A separate thread for the actual framegrabbings

   ///Thread starter, called by fgThreadStart on thread construction.  Calls fgThreadExec.
   static void _fgThreadStart( frameGrabber * o /**< [in] a pointer to an frameGrabber instance (normally this) */);

   /// Start the log capture.
   int fgThreadStart();

   /// Execute the log capture.
   void fgThreadExec();


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
   
   config.add("framegrabber.circBuffLength", "", "framegrabber.circBuffLength", argType::Required, "framegrabber", "circBuffLength", false, "size_t", "The length of the circular buffer. Sets m_planes, default is 1.");
}

template<class derivedT>
void frameGrabber<derivedT>::loadConfig(mx::app::appConfigurator & config)
{
   config(m_fgThreadPrio, "framegrabber.threadPrio");
   config(m_shmimName, "framegrabber.shmimName");
   config(m_planes, "framegrabber.circBuffLength");
}
   

template<class derivedT>
int frameGrabber<derivedT>::appStartup()
{
   
   if(fgThreadStart() < 0)
   {
      derivedT::template log<software_error>({__FILE__, __LINE__});
      return -1;
   }
   
   return 0;

}

template<class derivedT>
int frameGrabber<derivedT>::appLogic()
{
   //first do a join check to see if other threads have exited.
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
      m_fgThread.join();
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
      imsize[2] = m_planes;
      ImageStreamIO_createIm(&imageStream, m_shmimName.c_str(), 3, imsize, m_dataType, 1, 0);
      imageStream.md[0].cnt1 = m_planes;
      
      //This completes the reconfiguration.
      m_reconfig = false;
                  
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
         if(cnt1 > m_planes-1) cnt1 = 0;
         imageStream.md[0].cnt1 = cnt1;

         if(m_parent->loadImageIntoStream((char *) imageStream.array.raw + imageStream.md[0].cnt1*m_width*m_height*m_typeSize) < 0) 
         {
            break;
         }
         
         imageStream.md[0].atime = m_currImageTimestamp;;
         imageStream.md[0].cnt0++;
         
         
         imageStream.md[0].write=0;
         ImageStreamIO_sempost(&imageStream,-1);
 

      }
    
      ImageStreamIO_destroyIm( &imageStream );
    
      if(m_reconfig && !m_parent->shutdown())
      {
         m_parent->reconfig();
      }

   } //outer loop, will exit if m_shutdown==true

}




} //namespace dev
} //namespace app
} //namespace MagAOX
#endif
