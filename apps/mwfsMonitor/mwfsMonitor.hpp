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
class mwfsMonitor : public MagAOXApp<true>, public dev::shmimMonitor<mwfsMonitor>, public dev::frameGrabber<mwfsMonitor>
{

   //Give the test harness access.
   friend class mwfsMonitor_test;

   friend class dev::shmimMonitor<mwfsMonitor>;
   friend class dev::frameGrabber<mwfsMonitor>;
   
   //The base shmimMonitor type
   typedef dev::shmimMonitor<mwfsMonitor> shmimMonitorT;
   
   //The base frameGrabber type
   typedef dev::frameGrabber<mwfsMonitor> frameGrabberT;
   
   ///Floating point type in which to do all calculations.
   typedef float realT;
   
protected:

   /** \name Configurable Parameters
     *@{
     */

   std::vector<std::string> m_elNames;
   int m_PSFBox {30};
   int m_mwfsBox {180};
   int m_xPSF1 {245};
   int m_yPSF1 {600};
   int m_xPSF2 {400};
   int m_yPSF2 {740};
   int m_xmwfs {-130};
   int m_ymwfs {140};
   bool m_mwfs_toggle {false};

   // Declare the gauss fitter
   mx::math::fit::fitGaussian2D<mx::math::fit::gaussian2D_gen_fitter<float>> m_fit;

   mx::improc::eigenImage<realT> m_im; // full vAPP image
   mx::improc::eigenImage<realT> m_im_psf1; // cut out around psf1
   mx::improc::eigenImage<realT> m_im_psf2; // cut out around psf2
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

   int centroid(mx::improc::eigenImage<realT> & image, mx::improc::eigenImage<realT> & im_block,
                int x, int y, int boxsize, realT & x_psf, realT & y_psf);
   
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
   
   pcf::IndiProperty m_indiP_PSFBox;
   pcf::IndiProperty m_indiP_mwfsBox;
   pcf::IndiProperty m_indiP_xPSF1;
   pcf::IndiProperty m_indiP_yPSF1;
   pcf::IndiProperty m_indiP_xPSF2;
   pcf::IndiProperty m_indiP_yPSF2;
   pcf::IndiProperty m_indiP_xmwfs;
   pcf::IndiProperty m_indiP_ymwfs;
   pcf::IndiProperty m_indiP_mwfs_toggle;


   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_PSFBox);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_mwfsBox);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_xPSF1);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_yPSF1);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_xPSF2);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_yPSF2);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_xmwfs);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_ymwfs);
   INDI_NEWCALLBACK_DECL(mwfsMonitor, m_indiP_mwfs_toggle);
};

inline
mwfsMonitor::mwfsMonitor() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   shmimMonitorT::m_semaphoreNumber = 5;
   return;
}

inline
void mwfsMonitor::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   
   frameGrabberT::setupConfig(config);

   config.add("mwfs.PSFBox", "", "mwfs.PSFBox", argType::Required, "mwfs", "PSFBox", false, "int", "Size of box (length of a side) around PSFs in which to centroid");
   config.add("mwfs.mwfsBox", "", "mwfs.mwfsBox", argType::Required, "mwfs", "mwfsBox", false, "int", "Size of box (length of a side) around mwfs spots");

}

inline
int mwfsMonitor::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(config);
   
   frameGrabberT::loadConfig(config);
   
   _config(m_PSFBox, "mwfs.PSFBox");
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
   createStandardIndiNumber<unsigned>( m_indiP_PSFBox, "PSFBox", 1, std::numeric_limits<unsigned>::max(), 1, "%u");
   m_indiP_PSFBox["current"] = m_PSFBox;

   if( registerIndiPropertyNew( m_indiP_PSFBox, INDI_NEWCALLBACK(m_indiP_PSFBox)) < 0)
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

   // psf and mwfs positions
   createStandardIndiNumber<unsigned>( m_indiP_xPSF1, "xPSF1", 1, std::numeric_limits<unsigned>::max(), 1, "%u");
   m_indiP_xPSF1["current"] = m_xPSF1;
   if( registerIndiPropertyNew( m_indiP_xPSF1, INDI_NEWCALLBACK(m_indiP_xPSF1)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   createStandardIndiNumber<realT>( m_indiP_yPSF1, "yPSF1", 1, std::numeric_limits<realT>::max(), 1, "%u");
   m_indiP_yPSF1["current"] = m_yPSF1;
   if( registerIndiPropertyNew( m_indiP_yPSF1, INDI_NEWCALLBACK(m_indiP_yPSF1)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   createStandardIndiNumber<unsigned>( m_indiP_xPSF2, "xPSF2", 1, std::numeric_limits<unsigned>::max(), 1, "%u");
   m_indiP_xPSF2["current"] = m_xPSF2;
   if( registerIndiPropertyNew( m_indiP_xPSF2, INDI_NEWCALLBACK(m_indiP_xPSF2)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   createStandardIndiNumber<realT>( m_indiP_yPSF2, "yPSF2", 1, std::numeric_limits<realT>::max(), 1, "%u");
   m_indiP_yPSF2["current"] = m_yPSF2;
   if( registerIndiPropertyNew( m_indiP_yPSF2, INDI_NEWCALLBACK(m_indiP_yPSF2)) < 0)
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
   
   if( frameGrabberT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if(shmimMonitorT::updateINDI() < 0)
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
   m_im_psf1.resize(m_PSFBox, m_PSFBox);
   m_im_psf1.setZero();
   m_im_psf2.resize(m_PSFBox, m_PSFBox);
   m_im_psf2.setZero();
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
   
   updateIfChanged(m_indiP_PSFBox, "current", m_PSFBox, INDI_IDLE);
   updateIfChanged(m_indiP_PSFBox, "target", m_PSFBox, INDI_IDLE);
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

   // get PSF centroids
   realT shift_x1, shift_y1, shift_x2, shift_y2;
   centroid(m_im, m_im_psf1, m_xPSF1, m_yPSF1, m_PSFBox, shift_x1, shift_y1);
   centroid(m_im, m_im_psf2, m_xPSF2, m_yPSF2, m_PSFBox, shift_x2, shift_y2);

   // take the mean centroid
   realT shift_x = (shift_x1 + shift_x2) / 2.;
   realT shift_y = (shift_y1 + shift_y2) / 2.;
   
   // shift image to center on leak term
   realT ycen = (shmimMonitorT::m_height - 1) / 2.;
   realT xcen = (shmimMonitorT::m_width - 1) / 2.;
   imageShift(m_im_centered, m_im, xcen - shift_x, ycen - shift_y, mx::improc::cubicConvolTransform<realT>());

   // rotate image and subtract
   imageRotate(m_im_rot180, m_im_centered, M_PI, mx::improc::cubicConvolTransform<realT>());
   m_im_delta = m_im_centered - m_im_rot180;

   // cut down to mwfs block
   int mwfs_block_x = static_cast<int>(m_xmwfs - m_mwfsBox/2. + xcen);
   int mwfs_block_y = static_cast<int>(m_ymwfs - m_mwfsBox/2. + ycen);
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
int mwfsMonitor::centroid(mx::improc::eigenImage<realT> & image, mx::improc::eigenImage<realT> & im_block, int x, int y, int boxsize, realT & x_psf, realT & y_psf)
{
   // run gauss fitter on image block
   int block_x = static_cast<int>(x - boxsize/2.);
   int block_y = static_cast<int>(y - boxsize/2.);
   im_block = image.block(block_x, block_y, boxsize, boxsize);

   m_fit.setArray(im_block.data(), im_block.rows(), im_block.cols());
   m_fit.setGuess(0, im_block.maxCoeff(),  boxsize/2.,  boxsize/2., 3, 3, 0); 
   m_fit.fit();

   // get fit centroid
   realT fit_x = m_fit.x0();
   realT fit_y = m_fit.y0();

   // convert from subarray to full image coordinates
   x_psf = fit_x + block_x;
   y_psf = fit_y + block_y;

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

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_PSFBox)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_PSFBox.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_PSFBox, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_PSFBox = target;

   updateIfChanged(m_indiP_PSFBox, "current", m_PSFBox);
   updateIfChanged(m_indiP_PSFBox, "target", m_PSFBox);
   
   log<text_log>("set PSFBox to " + std::to_string(m_PSFBox) + " pixels", logPrio::LOG_NOTICE);
   
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

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_xPSF1)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_xPSF1.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_xPSF1, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_xPSF1 = target;

   updateIfChanged(m_indiP_xPSF1, "current", m_xPSF1);
   updateIfChanged(m_indiP_xPSF1, "target", m_xPSF1);
   
   log<text_log>("set xPSF1 to pixel " + std::to_string(m_xPSF1), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_yPSF1)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_yPSF1.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_yPSF1, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_yPSF1 = target;

   updateIfChanged(m_indiP_yPSF1, "current", m_yPSF1);
   updateIfChanged(m_indiP_yPSF1, "target", m_yPSF1);
   
   log<text_log>("set yPSF1 to pixel " + std::to_string(m_yPSF1), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_xPSF2)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_xPSF2.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_xPSF2, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_xPSF2 = target;

   updateIfChanged(m_indiP_xPSF2, "current", m_xPSF2);
   updateIfChanged(m_indiP_xPSF2, "target", m_xPSF2);
   
   log<text_log>("set xPSF2 to pixel " + std::to_string(m_xPSF2), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(mwfsMonitor, m_indiP_yPSF2)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_yPSF2.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_yPSF2, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_yPSF2 = target;

   updateIfChanged(m_indiP_yPSF2, "current", m_yPSF2);
   updateIfChanged(m_indiP_yPSF2, "target", m_yPSF2);
   
   log<text_log>("set yPSF2 to pixel " + std::to_string(m_yPSF2), logPrio::LOG_NOTICE);
   
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
