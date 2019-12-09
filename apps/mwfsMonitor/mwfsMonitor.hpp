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
#include <mx/improc/imageTransforms.hpp>
#include <mx/math/fit/fitGaussian.hpp>

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
   int m_leakBox {30};
   int m_mwfsBox {180};
   int m_xLeak {255};
   int m_yLeak {248};
   int m_xmwfs {120};
   int m_ymwfs {140};
   bool m_mwfs_toggle {false};

   // Declare the gauss fitter
   mx::math::fit::fitGaussian2D<mx::math::fit::gaussian2D_gen_fitter<float>> m_fit;

   mx::improc::eigenImage<realT> m_im; // full vAPP image
   mx::improc::eigenImage<realT> m_im_leak; // cut out around leak
   mx::improc::eigenImage<realT> m_im_centered; // vAPP image centered on leak
   mx::improc::eigenImage<realT> m_im_rot180; // rotated vAPP for subtraction
   mx::improc::eigenImage<realT> m_im_delta; // delta image
   mx::improc::eigenImage<realT> m_im_mwfs; // delta MWFS image

   ///@}
   
   size_t m_currImage {0};
   bool m_updated {false};

   sem_t m_smSemaphore; ///< Semaphore used to synchronize the fg thread and the sm thread.
   
   realT (*pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   realT (*mwfs_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   
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
   
   pcf::IndiProperty m_indiP_leakBox;
   pcf::IndiProperty m_indiP_mwfsBox;
   pcf::IndiProperty m_indiP_xLeak;
   pcf::IndiProperty m_indiP_yLeak;
   pcf::IndiProperty m_indiP_xmwfs;
   pcf::IndiProperty m_indiP_ymwfs;
   pcf::IndiProperty m_indiP_mwfs_toggle;


   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_leakBox);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_mwfsBox);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_xLeak);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_yLeak);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_xmwfs);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_ymwfs);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_mwfs_toggle);
};

inline
mwfsMonitor::mwfsMonitor() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   darkMonitorT::m_getExistingFirst = true;
   //shmimMonitorT::m_semaphoreNumber = 5;
   return;
}

inline
void mwfsMonitor::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   darkMonitorT::setupConfig(config);
   dark2MonitorT::setupConfig(config);
   
   frameGrabberT::setupConfig(config);

   config.add("mwfs.leakBox", "", "mwfs.leakBox", argType::Required, "mwfs", "leakBox", false, "int", "Size of box (length of a side) around leak term in which to centroid");
   config.add("mwfs.mwfsBox", "", "mwfs.mwfsBox", argType::Required, "mwfs", "mwfsBox", false, "int", "Size of box (length of a side) around mwfs spots");

}

inline
int mwfsMonitor::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(config);
   darkMonitorT::loadConfig(config);
   dark2MonitorT::loadConfig(config);
   
   frameGrabberT::loadConfig(config);
   
   _config(m_leakBox, "mwfs.leakBox");
   _config(m_mwfsBox, "mwfs.mwfsBox");
   
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
   
   // leakBox
   createStandardIndiNumber<unsigned>( m_indiP_leakBox, "leakBox", 1, std::numeric_limits<unsigned>::max(), 1, "%u");
   m_indiP_leakBox["current"] = m_leakBox;

   if( registerIndiPropertyNew( m_indiP_leakBox, INDI_NEWCALLBACK(m_indiP_leakBox)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   // mwfsBox
   createStandardIndiNumber<unsigned>( m_indiP_mwfsBox, "mwfsBox", 1, std::numeric_limits<unsigned>::max(), 1, "%u");
   m_indiP_mwfsBox["current"] = m_mwfsBox;

   if( registerIndiPropertyNew( m_indiP_mwfsBox, INDI_NEWCALLBACK(m_indiP_mwfsBox)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   // leak and mwfs positions
   createStandardIndiNumber<unsigned>( m_indiP_xLeak, "xLeak", 1, std::numeric_limits<unsigned>::max(), 1, "%u");
   m_indiP_xLeak["current"] = m_xLeak;
   if( registerIndiPropertyNew( m_indiP_xLeak, INDI_NEWCALLBACK(m_indiP_xLeak)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   createStandardIndiNumber<realT>( m_indiP_yLeak, "yLeak", 1, std::numeric_limits<realT>::max(), 1, "%u");
   m_indiP_yLeak["current"] = m_yLeak;
   if( registerIndiPropertyNew( m_indiP_yLeak, INDI_NEWCALLBACK(m_indiP_yLeak)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   createStandardIndiNumber<realT>( m_indiP_xmwfs, "xmwfs", 1, std::numeric_limits<realT>::max(), 1, "%u");
   m_indiP_xmwfs["current"] = m_xmwfs;
   if( registerIndiPropertyNew( m_indiP_xmwfs, INDI_NEWCALLBACK(m_indiP_xmwfs)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   createStandardIndiNumber<realT>( m_indiP_ymwfs, "ymwfs", 1, std::numeric_limits<realT>::max(), 1, "%u");
   m_indiP_ymwfs["current"] = m_ymwfs;
   if( registerIndiPropertyNew( m_indiP_ymwfs, INDI_NEWCALLBACK(m_indiP_ymwfs)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   createStandardIndiToggleSw( m_indiP_mwfs_toggle, "toggle");
   if( registerIndiPropertyNew( m_indiP_mwfs_toggle, INDI_NEWCALLBACK(m_indiP_mwfs_toggle)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
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
   m_im.setZero();
   m_im_leak.resize(m_leakBox, m_leakBox);
   m_im_leak.setZero();
   m_im_centered.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
   m_im_centered.setZero();
   m_im_rot180.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
   m_im_rot180.setZero();
   m_im_delta.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
   m_im_delta.setZero();
   m_im_mwfs.resize(m_mwfsBox, m_mwfsBox);
   m_im_mwfs.setZero();
   
   pixget = getPixPointer<realT>(shmimMonitorT::m_dataType);

   if(pixget == nullptr)
   {
      log<software_error>({__FILE__, __LINE__, "bad data type"});
      return -1;
   }
   
   updateIfChanged(m_indiP_leakBox, "current", m_leakBox, INDI_IDLE);
   updateIfChanged(m_indiP_leakBox, "target", m_leakBox, INDI_IDLE);
   updateIfChanged(m_indiP_mwfsBox, "current", m_mwfsBox, INDI_IDLE);
   updateIfChanged(m_indiP_mwfsBox, "target", m_mwfsBox, INDI_IDLE);
   
   //m_reconfig = true;
   
   return 0;
}

inline
int mwfsMonitor::processImage( void * curr_src, 
                                   const dev::shmimT & dummy 
                                 )
{

   // need to add checks for each block and centroid
   //if( leak_block_x > 0  && (leak_block_x + m_leakBox) < shmimMonitorT::m_width && leak_block_y > 0 && (leak_block_y + m_leakBox) < shmimMonitorT::m_height)

   static_cast<void>(dummy); //be unused

   // only process if toggled
   if(!m_mwfs_toggle)
   {
      m_updated = false;
      return 0;
   }

   // extract the full image
   for(size_t i=0; i < shmimMonitorT::m_width; ++i)
   {
      for(size_t j=0; j < shmimMonitorT::m_height; ++j)
      {
         m_im(j, i) = ((uint16_t*)curr_src)[i*shmimMonitorT::m_width + j];
      }
   }

   //background subtract
   m_im -= (m_im.coeff(0, 0) + m_im.coeff(0,shmimMonitorT::m_height-1) + m_im.coeff(shmimMonitorT::m_width-1, 0) + m_im.coeff(shmimMonitorT::m_width-1,shmimMonitorT::m_height-1)) / 4.;

   // dark subtract

   //if(m_darkSet && !m_dark2Set) m_avgImage -= m_darkImage;
   //else if(!m_darkSet && m_dark2Set) m_avgImage -= m_dark2Image;
   //else if(m_darkSet && m_dark2Set) m_avgImage -= m_darkImage + m_dark2Image;

   // run gauss fitter on leak block
   int leak_block_x = static_cast<int>(m_xLeak - m_leakBox/2.);
   int leak_block_y = static_cast<int>(m_yLeak - m_leakBox/2.);
   m_im_leak = m_im.block(leak_block_x, leak_block_y, m_leakBox, m_leakBox);

   m_fit.setArray(m_im_leak.data(), m_im_leak.rows(), m_im_leak.cols());
   m_fit.setGuess(0, m_im_leak.maxCoeff(),  m_leakBox/2.,  m_leakBox/2., 3, 3, 0); 
   m_fit.fit();

   // get fit centroid
   realT leak_x = m_fit.x0();
   realT leak_y = m_fit.y0();

   // convert from subarray to full image coordinates
   realT shift_x = leak_x + leak_block_x;
   realT shift_y = leak_y + leak_block_y;

   // shift image to center on leak term
   realT ycen = (shmimMonitorT::m_height - 1) / 2.;
   realT xcen = (shmimMonitorT::m_width - 1) / 2.;
   imageShift(m_im_centered, m_im, xcen - shift_x, ycen - shift_y, mx::improc::cubicConvolTransform<realT>());

   // rotate image and subtract
   imageRotate(m_im_rot180, m_im_centered, M_PI, mx::improc::cubicConvolTransform<realT>());
   m_im_delta = m_im_centered - m_im_rot180;

   // cut down to mwfs block
   //int mwfs_block_x = static_cast<int>(m_xmwfs - m_mwfsBox/2. + m_xLeak);
   //int mwfs_block_y = static_cast<int>(m_ymwfs - m_mwfsBox/2. + m_yLeak);
   int mwfs_block_x = static_cast<int>(m_xmwfs - m_mwfsBox/2. + shift_x);
   int mwfs_block_y = static_cast<int>(m_ymwfs - m_mwfsBox/2. + shift_y);
   if( mwfs_block_x > 0  && mwfs_block_y > 0)
   {
      m_im_mwfs = m_im_delta.block(mwfs_block_x, mwfs_block_y, m_mwfsBox, m_mwfsBox);
   }
   else
   {
      return 0;
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

   // mwfs image   
   frameGrabberT::m_width = m_mwfsBox;
   frameGrabberT::m_height = m_mwfsBox;
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
   memcpy(dest, m_im_mwfs.data(), m_mwfsBox*m_mwfsBox*frameGrabberT::m_typeSize); 
   m_updated = false;
   return 0;
   
}

inline
int mwfsMonitor::reconfig()
{
   m_im_mwfs.resize(m_mwfsBox, m_mwfsBox);

   frameGrabberT::m_width = m_mwfsBox;
   frameGrabberT::m_height = m_mwfsBox;

   return 0;
}

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_leakBox)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_leakBox.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_leakBox, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_leakBox = target;

   updateIfChanged(m_indiP_leakBox, "current", m_leakBox);
   updateIfChanged(m_indiP_leakBox, "target", m_leakBox);
   
   log<text_log>("set leakBox to " + std::to_string(m_leakBox) + " pixels", logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_mwfsBox)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_mwfsBox.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_mwfsBox, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_mwfsBox = target;

   updateIfChanged(m_indiP_mwfsBox, "current", m_mwfsBox);
   updateIfChanged(m_indiP_mwfsBox, "target", m_mwfsBox);

   m_reconfig = true;
   
   log<text_log>("set mwfsBox to " + std::to_string(m_mwfsBox) + " pixels", logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_xLeak)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_xLeak.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_xLeak, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_xLeak = target;

   updateIfChanged(m_indiP_xLeak, "current", m_xLeak);
   updateIfChanged(m_indiP_xLeak, "target", m_xLeak);
   
   log<text_log>("set xLeak to pixel " + std::to_string(m_xLeak), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_yLeak)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_yLeak.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_yLeak, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_yLeak = target;

   updateIfChanged(m_indiP_yLeak, "current", m_yLeak);
   updateIfChanged(m_indiP_yLeak, "target", m_yLeak);
   
   log<text_log>("set yLeak to pixel " + std::to_string(m_yLeak), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_xmwfs)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_xmwfs.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_xmwfs, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_xmwfs = target;

   updateIfChanged(m_indiP_xmwfs, "current", m_xmwfs);
   updateIfChanged(m_indiP_xmwfs, "target", m_xmwfs);
   
   log<text_log>("set xmwfs to pixel " + std::to_string(m_xmwfs), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_ymwfs)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_ymwfs.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_ymwfs, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_ymwfs = target;

   updateIfChanged(m_indiP_ymwfs, "current", m_ymwfs);
   updateIfChanged(m_indiP_ymwfs, "target", m_ymwfs);
   
   log<text_log>("set ymwfs to pixel " + std::to_string(m_ymwfs), logPrio::LOG_NOTICE);
   
   return 0;
}


INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_mwfs_toggle)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_mwfs_toggle.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   if(!ipRecv.find("toggle")) return 0;
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      updateSwitchIfChanged(m_indiP_mwfs_toggle, "toggle", pcf::IndiElement::On, INDI_BUSY);
      
      m_mwfs_toggle = true;
      
      log<text_log>("started mwfs stream");
   }
   else
   {
      updateSwitchIfChanged(m_indiP_mwfs_toggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      
      m_mwfs_toggle = false;
      
      log<text_log>("stopped mwfs stream");
   }
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //mwfsMonitor_hpp
