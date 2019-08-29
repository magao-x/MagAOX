/** \file pupilFit.hpp
  * \brief The MagAO-X Pyramid Pupil Fitter application header
  *
  * \ingroup pupilFit_files
  */

#ifndef pupilFit_hpp
#define pupilFit_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "pupilFitter.hpp"

/** \defgroup pupilFit 
  * \brief The MagAO-X pyramid pupil fitter.
  *
  * <a href="../handbook/apps/pupilFit.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup pupilFit_files
  * \ingroup pupilFit
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X Pyramid Pupil Fitter
/** 
  * \ingroup pupilFit
  */
class pupilFit : public MagAOXApp<true>, public dev::shmimMonitor<pupilFit>
{

   //Give the test harness access.
   friend class pupilFit_test;

   friend class dev::shmimMonitor<pupilFit>;
   
   //The base shmimMonitor type
   typedef dev::shmimMonitor<pupilFit> shmimMonitorT;
   
   ///Floating point type in which to do all calculations.
   typedef float realT;
   
protected:

   /** \name Configurable Parameters
     *@{
     */
   
   std::string m_threshShmimName {"camwfs_thresh"}; ///<The name of the image stream for the thresholded images.  Default is camwfs_thresh.
   std::string m_edgeShmimName {"camwfs_edge"}; ///<The name of the image stream for the edge images.  Default is camwfs_edge.
   
   float m_threshold {0.5};
   
   ///@}

   mx::improc::eigenImage<float> m_fitIm;
   mx::improc::eigenImage<float> m_edgeIm;

   pupilFitter<realT> m_fitter;
   
   IMAGE m_threshShmim;
   bool m_threshShmimConnected {false};
   
   IMAGE m_edgeShmim;
   bool m_edgeShmimConnected {false};
   
public:
   /// Default c'tor.
   pupilFit();

   /// D'tor, declared and defined for noexcept.
   ~pupilFit() noexcept;
   
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

   /// Implementation of the FSM for pupilFit.
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

   int allocate( const dev::shmimT &);
   
   int processImage( void* curr_src,
                     const dev::shmimT &
                    );
   
protected:

    
   /** \name INDI
     * @{
     */ 
   
   pcf::IndiProperty m_indiP_thresh;
   
   INDI_NEWCALLBACK_DECL(pupilFit, m_indiP_thresh);
   
   pcf::IndiProperty m_indiP_quad1;
   pcf::IndiProperty m_indiP_quad2;
   pcf::IndiProperty m_indiP_quad3;
   pcf::IndiProperty m_indiP_quad4;
   
   ///@}
};

inline
pupilFit::pupilFit() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

inline
pupilFit::~pupilFit() noexcept
{
   if(m_threshShmimConnected)
   {
      ImageStreamIO_destroyIm( &m_threshShmim );
   }
   
   if(m_edgeShmimConnected)
   {
      ImageStreamIO_destroyIm( &m_edgeShmim );
   }
}

inline
void pupilFit::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   
   config.add("shmimMonitor.shmimName", "", "shmimMonitor.shmimName", argType::Required, "shmimMonitor", "shmimName", false, "string", "The name of the ImageStreamIO shared memory image. Will be used as /tmp/<shmimName>.im.shm. Default is camwfs_avg");
   
   config.add("fit.threshold", "", "fit.threshold", argType::Required, "fit", "threshold", false, "float", "The pupil finding threshold. 0 < threshold < 1");
   config.add("fit.threshShmimName", "", "fit.threshShmimName", argType::Required, "fit", "threshShmimName", false, "float", "The name of the image stream for the thresholded images.  Default is camwfs_thresh.");
   config.add("fit.edgeShmimName", "", "fit.edgeShmimName", argType::Required, "fit", "edgeShmimName", false, "float", "The name of the image stream for the edge images.  Default is camwfs_edge.");
}

inline
int pupilFit::loadConfigImpl( mx::app::appConfigurator & _config )
{
   m_shmimName = "camwfs_avg";
   shmimMonitorT::loadConfig(_config);
      
   _config(m_threshold, "fit.threshold");
   _config(m_threshShmimName, "fit.threshShmimName");
   _config(m_edgeShmimName, "fit.edgeShmimName");
   
   return 0;
}

inline
void pupilFit::loadConfig()
{
   loadConfigImpl(config);
}

inline
int pupilFit::appStartup()
{
   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   
   createStandardIndiNumber<float>( m_indiP_thresh, "threshold", 0, 1 ,0, "%0.2f", "Threshold");
   m_indiP_thresh["current"].set(m_threshold);
   m_indiP_thresh["target"].set(m_threshold);
   registerIndiPropertyNew(m_indiP_thresh, INDI_NEWCALLBACK(m_indiP_thresh));
   
   createROIndiNumber( m_indiP_quad1, "quadrant1", "Quadrant 1");
   indi::addNumberElement<float>( m_indiP_quad1, "x", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_quad1, "y", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_quad1, "D", 0, 59, 0, "%0.2f", "diameter");
   indi::addNumberElement<float>( m_indiP_quad1, "med", 0, std::numeric_limits<uint16_t>::max(), 0, "%0.1f", "flux");
   registerIndiPropertyReadOnly(m_indiP_quad1);
   
   createROIndiNumber( m_indiP_quad2, "quadrant2", "Quadrant 2");
   indi::addNumberElement<float>( m_indiP_quad2, "x", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_quad2, "y", 0, 59, 0, "%0.2f", "center y");
   indi::addNumberElement<float>( m_indiP_quad2, "D", 0, 59, 0, "%0.2f", "diameter");
   indi::addNumberElement<float>( m_indiP_quad2, "med", 0, std::numeric_limits<uint16_t>::max(), 0, "%0.1f", "flux");
   registerIndiPropertyReadOnly(m_indiP_quad2);
   
   createROIndiNumber( m_indiP_quad3, "quadrant3", "Quadrant 3");
   indi::addNumberElement<float>( m_indiP_quad3, "x", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_quad3, "y", 0, 59, 0, "%0.2f", "center y");
   indi::addNumberElement<float>( m_indiP_quad3, "D", 0, 59, 0, "%0.2f", "diameter");
   indi::addNumberElement<float>( m_indiP_quad3, "med", 0, std::numeric_limits<uint16_t>::max(), 0, "%0.1f", "flux");
   registerIndiPropertyReadOnly(m_indiP_quad3);
   
   createROIndiNumber( m_indiP_quad4, "quadrant4", "Quadrant 4");
   indi::addNumberElement<float>( m_indiP_quad4, "x", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_quad4, "y", 0, 59, 0, "%0.2f", "center y");
   indi::addNumberElement<float>( m_indiP_quad4, "D", 0, 59, 0, "%0.2f", "diameter");
   indi::addNumberElement<float>( m_indiP_quad4, "med", 0, std::numeric_limits<uint16_t>::max(), 0, "%0.1f", "flux");
   registerIndiPropertyReadOnly(m_indiP_quad4);

   state(stateCodes::OPERATING);
   
   return 0;
}

inline
int pupilFit::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   std::lock_guard<std::mutex> guard(m_indiMutex);
   updateIfChanged(m_indiP_thresh, "current", m_threshold, INDI_IDLE);
   updateIfChanged(m_indiP_thresh, "target", m_threshold, INDI_IDLE);
   
   shmimMonitorT::updateINDI();
   
   return 0;
}

inline
int pupilFit::appShutdown()
{
   shmimMonitorT::appShutdown();
   
   return 0;
}

inline
int pupilFit::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy);
   
   m_fitIm.resize(m_width, m_height);
   m_edgeIm.resize(m_width, m_height);
   
   m_fitter.setSize(0.5*m_width, 0.5*m_height);
   
   
   uint32_t imsize[3];
   imsize[0] = m_width;
   imsize[1] = m_height;
   imsize[2] = 1;
   
   if(m_threshShmimConnected)
   {
      ImageStreamIO_destroyIm( &m_threshShmim );
      m_threshShmimConnected = false;
   }
   
   if(m_edgeShmimConnected)
   {
      ImageStreamIO_destroyIm( &m_edgeShmim );
      m_edgeShmimConnected = false;
   }
   
   ImageStreamIO_createIm_gpu(&m_threshShmim , m_threshShmimName .c_str(), 3, imsize, m_dataType, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL);
   m_threshShmimConnected = true;
   
   ImageStreamIO_createIm_gpu(&m_edgeShmim , m_edgeShmimName .c_str(), 3, imsize, m_dataType, -1, 1, IMAGE_NB_SEMAPHORE, 0, CIRCULAR_BUFFER | ZAXIS_TEMPORAL);
   m_edgeShmimConnected = true;
   
   if(m_edgeShmimConnected)
   {
   }
   
   return 0;
}
   
inline
int pupilFit::processImage( void* curr_src,
                            const dev::shmimT & dummy
                          )
{
   static_cast<void>(dummy);
   
   for(unsigned nn=0; nn < m_width*m_height; ++nn)
   {
      m_fitIm.data()[nn] += ((float*)curr_src) [nn];
   }
   
   m_fitter.m_thresh = m_threshold;
   
   m_fitter.fit(m_fitIm, m_edgeIm);
   
   {//mutex scope
      
      std::lock_guard<std::mutex> guard(m_indiMutex);
      m_indiP_quad1["x"].set(m_fitter.m_avgx[0]);
      m_indiP_quad1["y"].set(m_fitter.m_avgy[0]);
      m_indiP_quad1["D"].set(2*m_fitter.m_avgr[0]);
      m_indiP_quad1["med"].set(m_fitter.m_med[0]);
      m_indiP_quad1.setState (INDI_BUSY);
      m_indiDriver->sendSetProperty (m_indiP_quad1);
   
      m_indiP_quad2["x"].set(m_fitter.m_avgx[1]);
      m_indiP_quad2["y"].set(m_fitter.m_avgy[1]);
      m_indiP_quad2["D"].set(2*m_fitter.m_avgr[1]);
      m_indiP_quad2["med"].set(m_fitter.m_med[1]);
      m_indiP_quad2.setState (INDI_BUSY);
      m_indiDriver->sendSetProperty (m_indiP_quad2);
   
      m_indiP_quad3["x"].set(m_fitter.m_avgx[2]);
      m_indiP_quad3["y"].set(m_fitter.m_avgy[2]);
      m_indiP_quad3["D"].set(2*m_fitter.m_avgr[2]);
      m_indiP_quad3["med"].set(m_fitter.m_med[2]);
      m_indiP_quad3.setState (INDI_BUSY);
      m_indiDriver->sendSetProperty (m_indiP_quad3);
   
      m_indiP_quad4["x"].set(m_fitter.m_avgx[3]);
      m_indiP_quad4["y"].set(m_fitter.m_avgy[3]);
      m_indiP_quad4["D"].set(2*m_fitter.m_avgr[3]);
      m_indiP_quad4["med"].set(m_fitter.m_med[3]);
      m_indiP_quad4.setState (INDI_BUSY);
      m_indiDriver->sendSetProperty (m_indiP_quad4);
   }
   
   m_threshShmim.md->write=1;
   m_edgeShmim.md->write=1;
   
   clock_gettime(CLOCK_REALTIME, &m_threshShmim.md->writetime);
   m_edgeShmim.md->writetime = m_threshShmim.md->writetime;
   
   m_threshShmim.md->atime = m_threshShmim.md->writetime;
   m_edgeShmim.md->atime = m_threshShmim.md->writetime;
   
   m_threshShmim.md->cnt0++;
   m_edgeShmim.md->cnt0++;
   
   memcpy(m_threshShmim.array.raw, m_fitIm.data(), m_fitIm.rows()*m_fitIm.cols()*sizeof(float));
   memcpy(m_edgeShmim.array.raw, m_edgeIm.data(), m_edgeIm.rows()*m_edgeIm.cols()*sizeof(float));

   m_threshShmim.md->write=0;
   m_edgeShmim.md->write=0;
   
   ImageStreamIO_sempost(&m_threshShmim,-1);
   ImageStreamIO_sempost(&m_edgeShmim,-1);
   
   return 0;
}
   
   
INDI_NEWCALLBACK_DEFN(pupilFit, m_indiP_thresh)(const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getName() != m_indiP_thresh.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_thresh, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_threshold = target;
   
   log<text_log>("set threshold = " + std::to_string(m_threshold), logPrio::LOG_NOTICE);
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //pupilFit_hpp
