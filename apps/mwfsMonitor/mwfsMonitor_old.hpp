/** \file mwfsMonitor.hpp
  * \brief mwfs Monitor
  *
  * \ingroup app_files
  */

#ifndef mwfsMonitor_hpp
#define mwfsMonitor_hpp

#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

struct darkShmimT 
{
   static std::string configSection()
   {
      return "darkShmim";
   };
   
   static std::string indiPrefix()
   {
      return "dark";
   };
};

struct dark2ShmimT 
{
   static std::string configSection()
   {
      return "dark2Shmim";
   };
   
   static std::string indiPrefix()
   {
      return "dark2";
   };
};

/** \defgroup mwfsMonitor ImageStreamIO
  * \brief Monitors a vAPP PSF and pulls out the coefficients for the hMWFS control loop
  *
  * <a href="../handbook/operating/software/apps/mwfsMonitor.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup mwfsMonitor_files ImageStreamIO
  * \ingroup mwfsMonitor
  */

/** MagAO-X application to monitor hMWFS spots
  *
  * \ingroup mwfsMonitor
  * 
  */
class mwfsMonitor : public MagAOXApp<true>, public dev::shmimMonitor<mwfsMonitor>, public dev::shmimMonitor<mwfsMonitor,darkShmimT>, public dev::shmimMonitor<mwfsMonitor,dark2ShmimT>, public dev::frameGrabber<mwfsMonitor>
{

   //Give the test harness access.
   friend class mwfsMonitor_test;

   friend class dev::shmimMonitor<mwfsMonitor>;
   friend class dev::shmimMonitor<mwfsMonitor,darkShmimT>;
   friend class dev::shmimMonitor<mwfsMonitor,dark2ShmimT>;
   friend class dev::frameGrabber<mwfsMonitor>;
   
   //The base shmimMonitor type
   typedef dev::shmimMonitor<mwfsMonitor> shmimMonitorT;
   
   //The dark shmimMonitor type
   typedef dev::shmimMonitor<mwfsMonitor, darkShmimT> darkMonitorT;
   
   //The dark shmimMonitor type for a 2nd dark
   typedef dev::shmimMonitor<mwfsMonitor, dark2ShmimT> dark2MonitorT;
   
   //The base frameGrabber type
   typedef dev::frameGrabber<mwfsMonitor> frameGrabberT;
   
   ///Floating point type in which to do all calculations.
   typedef float realT;
   
protected:

   /** \name Configurable Parameters
     *@{
     */

   std::vector<std::string> m_elNames;
   std::vector<realT> m_mSignals;
   std::vector<realT> m_Peaks;
   std::vector<realT> m_xSpot;
   std::vector<realT> m_ySpot;
   int m_nSpots {18};
   int m_boxSize {30};

   mx::improc::eigenImage<realT> m_im; // vAPP image

   ///@}
   
   size_t m_currImage {0};
   bool m_updated {false};

   sem_t m_smSemaphore; ///< Semaphore used to synchronize the fg thread and the sm thread.
   
   realT (*pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   
   mx::improc::eigenImage<realT> m_darkImage;
   bool m_darkSet {false};
   realT (*dark_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   
   mx::improc::eigenImage<realT> m_dark2Image;
   bool m_dark2Set {false};
   realT (*dark2_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   
public:
   /// Default c'tor.
   mwfsMonitor();

   /// D'tor, declared and defined for noexcept.
   ~mwfsMonitor() noexcept
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

   /// Implementation of the FSM for mwfsMonitor.
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

   int allocate( const dev::shmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const dev::shmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );
   
   int allocate( const darkShmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const darkShmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );
   
   int allocate( const dark2ShmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const dark2ShmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );
protected:

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
   
   ///@}
   
   pcf::IndiProperty m_indiP_nSpots;
  //pcf::IndiProperty m_indiP_mSignals;
   pcf::IndiProperty m_indiP_boxSize;
   pcf::IndiProperty m_indiP_xSpot;
   pcf::IndiProperty m_indiP_ySpot;

   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_nSpots);
   //INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_mSignals);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_boxSize);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_xSpot);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_ySpot);
};

inline
mwfsMonitor::mwfsMonitor() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   darkMonitorT::m_getExistingFirst = true;
   return;
}

inline
void mwfsMonitor::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   darkMonitorT::setupConfig(config);
   dark2MonitorT::setupConfig(config);
   
   frameGrabberT::setupConfig(config);

   config.add("mwfs.nSpots", "", "mwfs.nSpots", argType::Required, "mwfs", "nSpots", false, "int", "Number of spots to monitor.");
   config.add("mwfs.boxSize", "", "mwfs.boxSize", argType::Required, "mwfs", "boxSize", false, "int", "Size of box (length of a side) around mwfs spot in which to compute peak");

}

inline
int mwfsMonitor::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(config);
   darkMonitorT::loadConfig(config);
   dark2MonitorT::loadConfig(config);
   
   frameGrabberT::loadConfig(config);
   
   _config(m_nSpots, "mwfs.nSpots");
   _config(m_boxSize, "mwfs.boxSize");
   
   return 0;
}

inline
void mwfsMonitor::loadConfig()
{
   loadConfigImpl(config);
}

inline
int mwfsMonitor::appStartup()
{
   
   // nSpots
   createStandardIndiNumber<unsigned>( m_indiP_nSpots, "nSpots", 1, std::numeric_limits<unsigned>::max(), 1, "%u");
   m_indiP_nSpots["current"] = m_nSpots;

   if( registerIndiPropertyNew( m_indiP_nSpots, INDI_NEWCALLBACK(m_indiP_nSpots)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   // mSignals
   //REG_INDI_NEWPROP(m_indiP_mSignals, "mSignals", pcf::IndiProperty::Number);
   m_mSignals.resize(m_nSpots / 2, 0);
   m_Peaks.resize(m_nSpots, 0);
   m_elNames.resize(m_nSpots);
   for(size_t n=0; n < m_nSpots; ++n)
   {
      //std::string el = std::to_string(n);
      m_elNames[n] = mx::ioutils::convertToString<size_t, 2, '0'>(n);
      
      //m_indiP_mSignals.add( pcf::IndiElement(m_elNames[n]) );
      //m_indiP_mSignals[m_elNames[n]].set(0);
   }

   // boxSize
   createStandardIndiNumber<unsigned>( m_indiP_boxSize, "boxSize", 1, std::numeric_limits<unsigned>::max(), 1, "%u");
   m_indiP_boxSize["current"] = m_boxSize;

   if( registerIndiPropertyNew( m_indiP_boxSize, INDI_NEWCALLBACK(m_indiP_boxSize)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   // xSpot
   REG_INDI_NEWPROP(m_indiP_xSpot, "xSpot", pcf::IndiProperty::Number);
   m_xSpot.resize(m_nSpots, 0);
   for(size_t n=0; n < m_xSpot.size(); ++n)
   {
      //std::string el = std::to_string(n);
      m_elNames[n] = mx::ioutils::convertToString<size_t, 2, '0'>(n);
      
      m_indiP_xSpot.add( pcf::IndiElement(m_elNames[n]) );
      m_indiP_xSpot[m_elNames[n]].set(0);
   }

   // ySpot
   REG_INDI_NEWPROP(m_indiP_ySpot, "ySpot", pcf::IndiProperty::Number);
   m_ySpot.resize(m_nSpots, 0);
   for(size_t n=0; n < m_ySpot.size(); ++n)
   {
      //std::string el = std::to_string(n);
      m_elNames[n] = mx::ioutils::convertToString<size_t, 2, '0'>(n);
      
      m_indiP_ySpot.add( pcf::IndiElement(m_elNames[n]) );
      m_indiP_ySpot[m_elNames[n]].set(0);
   }
   
   if(sem_init(&m_smSemaphore, 0,0) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno,0, "Initializing S.M. semaphore"});
      return -1;
   }
   
   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   
   if(darkMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   
   if(dark2MonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   
   if(frameGrabberT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   
   state(stateCodes::OPERATING);
   //std::cout << "Done with startup, I guess!\n";
   return 0;
}

inline
int mwfsMonitor::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   if( darkMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   if( dark2MonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   if( frameGrabberT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if(shmimMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   if(darkMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   if(dark2MonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   
   if(frameGrabberT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
      
   return 0;
}

inline
int mwfsMonitor::appShutdown()
{
   shmimMonitorT::appShutdown();
   
   darkMonitorT::appShutdown();
   
   frameGrabberT::appShutdown();
   
   return 0;
}

inline
int mwfsMonitor::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy); //be unused
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   m_currImage = 0;

   m_im.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
   
   pixget = getPixPointer<realT>(shmimMonitorT::m_dataType);

   if(pixget == nullptr)
   {
      log<software_error>({__FILE__, __LINE__, "bad data type"});
      return -1;
   }
   
   //updateIfChanged(m_indiP_nSpots, "current", m_nSpots, INDI_IDLE);
   //updateIfChanged(m_indiP_nSpots, "target", m_nSpots, INDI_IDLE);

   updateIfChanged(m_indiP_boxSize, "current", m_boxSize, INDI_IDLE);
   updateIfChanged(m_indiP_boxSize, "target", m_boxSize, INDI_IDLE);
   
   m_reconfig = true;
   
   return 0;
}

inline
int mwfsMonitor::processImage( void * curr_src, 
                                   const dev::shmimT & dummy 
                                 )
{
   static_cast<void>(dummy); //be unused

   // extract the image
   for(size_t i=0; i < shmimMonitorT::m_width; ++i)
   {
      for(size_t j=0; j < shmimMonitorT::m_height; ++j)
      {
         m_im(i, j) = ((uint16_t*)curr_src)[i*shmimMonitorT::m_width + j];
      }
   }

   //background subtract
   m_im -= (m_im.coeff(0, 0) + m_im.coeff(0,shmimMonitorT::m_height-1) + m_im.coeff(shmimMonitorT::m_width-1, 0) + m_im.coeff(shmimMonitorT::m_width-1,shmimMonitorT::m_height-1)) / 4.;

   // dark subtract
   /*
   if(m_darkSet && !m_dark2Set) m_avgImage -= m_darkImage;
   else if(!m_darkSet && m_dark2Set) m_avgImage -= m_dark2Image;
   else if(m_darkSet && m_dark2Set) m_avgImage -= m_darkImage + m_dark2Image;
   */


   // grab peak values
   for(size_t n=0; n < m_nSpots; ++n)
   {
      /* get the estimated spot centroid and offset by
      half the box size */
      int x = static_cast<int>(m_xSpot[n] - m_boxSize/2.);
      int y = static_cast<int>(m_ySpot[n] - m_boxSize/2.);

      if( x > 0  && y > 0)
      {
         // just take maximum within psf cutout for now
         //m_Peaks[n] = m_im.block(x, y, m_boxSize, m_boxSize).maxCoeff();

         // actually, try a sum in a (smaller) instead?
         m_Peaks[n] = m_im.block(x, y, m_boxSize, m_boxSize).sum();
         //std::cout << "Peak " << std::to_string(n) << ": " << std::to_string(m_Peaks[n]) << "\n";
      }
      else
      {
         return 0;
      }
      
   }

   for(size_t n=0; n < m_mSignals.size(); ++n)
   {
      realT sum = (m_Peaks[n] + m_Peaks[n+m_nSpots/2]);
      realT diff = (m_Peaks[n] - m_Peaks[n+m_nSpots/2]);
      if(sum != 0)
      {
         m_mSignals[n] = diff / sum;
         //std::cout << "Signal " << std::to_string(n) << ": " << std::to_string(diff/sum) << "\n";
      }
      else // avoid dividing by 0
      {
         m_mSignals[n] = 0;
      }
   }
   

   m_updated = true;

   //"Now tell the f.g. to get going" - Jared
   if(sem_post(&m_smSemaphore) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
      return -1;
   }
   return 0;
}

inline
int mwfsMonitor::allocate(const darkShmimT & dummy)
{
   static_cast<void>(dummy); //be unused
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if(darkMonitorT::m_width != shmimMonitorT::m_width || darkMonitorT::m_height != shmimMonitorT::m_height)
   {
      m_darkSet = false;
      darkMonitorT::m_restart = true;
   }
   
   m_darkImage.resize(darkMonitorT::m_width, darkMonitorT::m_height);
   
   dark_pixget = getPixPointer<realT>(darkMonitorT::m_dataType);
   
   if(dark_pixget == nullptr)
   {
      log<software_error>({__FILE__, __LINE__, "bad data type"});
      return -1;
   }
   
   return 0;
}

inline
int mwfsMonitor::processImage( void * curr_src, 
                                   const darkShmimT & dummy 
                                 )
{
   static_cast<void>(dummy); //be unused
   
   realT * data = m_darkImage.data();
   
   for(unsigned nn=0; nn < darkMonitorT::m_width*darkMonitorT::m_height; ++nn)
   {
      //data[nn] = *( (int16_t * ) (curr_src + nn*shmimMonitorT::m_typeSize));
      data[nn] = dark_pixget(curr_src, nn);
   }
   
   m_darkSet = true;
   
   return 0;
}

inline
int mwfsMonitor::allocate(const dark2ShmimT & dummy)
{
   static_cast<void>(dummy); //be unused
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if(dark2MonitorT::m_width != shmimMonitorT::m_width || dark2MonitorT::m_height != shmimMonitorT::m_height)
   {
      m_dark2Set = false;
      dark2MonitorT::m_restart = true;
   }
   
   m_dark2Image.resize(dark2MonitorT::m_width, dark2MonitorT::m_height);
   
   dark2_pixget = getPixPointer<realT>(dark2MonitorT::m_dataType);
   
   if(dark2_pixget == nullptr)
   {
      log<software_error>({__FILE__, __LINE__, "bad data type"});
      return -1;
   }
   
   return 0;
}

inline
int mwfsMonitor::processImage( void * curr_src, 
                                   const dark2ShmimT & dummy 
                                 )
{
   static_cast<void>(dummy); //be unused
   
   realT * data = m_dark2Image.data();
   
   for(unsigned nn=0; nn < dark2MonitorT::m_width*dark2MonitorT::m_height; ++nn)
   {
      //data[nn] = *( (int16_t * ) (curr_src + nn*shmimMonitorT::m_typeSize));
      data[nn] = dark2_pixget(curr_src, nn);
   }
   
   m_dark2Set = true;
   
   return 0;
}


inline
int mwfsMonitor::configureAcquisition()
{
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if(shmimMonitorT::m_width==0 || shmimMonitorT::m_height==0 || shmimMonitorT::m_dataType == 0)
   {
      //This means we haven't connected to the stream to monitor
      sleep(1);
      return -1;
   }
   
   // This frameGrabberT guy is the vector of control inputs for cacao
   
   frameGrabberT::m_width = (int) m_nSpots/2;
   frameGrabberT::m_height = 1;
   frameGrabberT::m_dataType = _DATATYPE_FLOAT;

   std::cerr << "shmimMonitorT::m_dataType: " << (int) shmimMonitorT::m_dataType << "\n";
   std::cerr << "frameGrabberT::m_dataType: " << (int) frameGrabberT::m_dataType << "\n";
   
   return 0;
}

inline
int mwfsMonitor::startAcquisition()
{
   return 0;
}

inline
int mwfsMonitor::acquireAndCheckValid()
{
   timespec ts;
         
   if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
   {
      log<software_critical>({__FILE__,__LINE__,errno,0,"clock_gettime"}); 
      return -1;
   }
         
   ts.tv_sec += 1;
        
   if(sem_timedwait(&m_smSemaphore, &ts) == 0)
   {
      if( m_updated )
      {
         clock_gettime(CLOCK_REALTIME, &m_currImageTimestamp);
         return 0;
      }
      else
      {
         return 1;
      }
   }
   else
   {
      return 1;
   }
}

inline
int mwfsMonitor::loadImageIntoStream(void * dest)
{
   // this should copy hMWFS coefficients into a shmim, I think?
   //std::cout << __FILE__ << ":" << __LINE__ << " weewooweewoo\n";
   memcpy(dest, m_mSignals.data(), frameGrabberT::m_width*frameGrabberT::m_height*frameGrabberT::m_typeSize  ); 
   m_updated = false;
   return 0;
}

inline
int mwfsMonitor::reconfig()
{
   return 0;
}

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_nSpots)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_nSpots.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_nSpots, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_nSpots = target;

   //updateIfChanged(m_indiP_nSpots, "current", m_nSpots);
   //log<text_log>("set nSpots to " + std::to_string(m_nSpots), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_boxSize)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_boxSize.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_boxSize, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_boxSize = target;

   updateIfChanged(m_indiP_boxSize, "current", m_boxSize);
   updateIfChanged(m_indiP_boxSize, "target", m_boxSize);
   
   log<text_log>("set boxSize to " + std::to_string(m_boxSize) + " pixels", logPrio::LOG_NOTICE);
   
   return 0;
}

/*
INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_mSignals)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_mSignals.getName())
   {
      for(size_t n=0; n < m_mSignals.size(); ++n)
      {
         if(ipRecv.find(m_elNames[n]))
         {
            realT mSignal = ipRecv[m_elNames[n]].get<realT>();
            m_mSignals[n] = mSignal;
         }
      }
      return 0;
   }
   
   return log<software_error,-1>({__FILE__,__LINE__, "invalid indi property name"});
}
*/

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_xSpot)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_xSpot.getName())
   {
      for(size_t n=0; n < m_xSpot.size(); ++n)
      {
         if(ipRecv.find(m_elNames[n]))
         {
            realT x = ipRecv[m_elNames[n]].get<realT>();
            m_xSpot[n] = x;
            updateIfChanged(m_indiP_xSpot, m_elNames[n], x);
         }
      }
      return 0;
   }
   
   return log<software_error,-1>({__FILE__,__LINE__, "invalid indi property name"});
}

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_ySpot)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ySpot.getName())
   {
      for(size_t n=0; n < m_ySpot.size(); ++n)
      {
         if(ipRecv.find(m_elNames[n]))
         {
            realT y = ipRecv[m_elNames[n]].get<realT>();
            m_ySpot[n] = y;
            updateIfChanged(m_indiP_ySpot, m_elNames[n], y);
         }
      }
      return 0;
   }
   
   return log<software_error,-1>({__FILE__,__LINE__, "invalid indi property name"});
}

} //namespace app
} //namespace MagAOX

#endif //mwfsMonitor_hpp
