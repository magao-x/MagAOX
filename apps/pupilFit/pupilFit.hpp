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
   
   double m_setx1 {29.5};
   double m_sety1 {29.5};
   double m_setD1 {56.0};
   
   double m_setx2 {89.5};
   double m_sety2 {29.5};
   double m_setD2 {56.0};
   
   double m_setx3 {29.5};
   double m_sety3 {89.5};
   double m_setD3 {56.0};
   
   double m_setx4 {89.5};
   double m_sety4 {89.5};
   double m_setD4 {56.0};
   
   bool m_averaging {false};
   size_t m_navg {0};
   
   double m_avgx1_accum {0};
   double m_avgx1sq_accum {0};
      
   double m_avgy1_accum {0};
   double m_avgy1sq_accum {0};
      
   double m_avgD1_accum {0};
   double m_avgD1sq_accum {0};
      
   double m_avgmed1_accum {0};
   double m_avgmed1sq_accum {0};
      
   double m_avgx1 {0};
   double m_varx1 {0};
      
   double m_avgy1 {0};
   double m_vary1 {0};
      
   double m_avgD1 {0};
   double m_varD1 {0};
      
   double m_avgmed1 {0};
   double m_varmed1 {0};
   
   double m_avgx2_accum {0};
   double m_avgx2sq_accum {0};
      
   double m_avgy2_accum {0};
   double m_avgy2sq_accum {0};
      
   double m_avgD2_accum {0};
   double m_avgD2sq_accum {0};
      
   double m_avgmed2_accum {0};
   double m_avgmed2sq_accum {0};
      
   double m_avgx2 {0};
   double m_varx2 {0};
      
   double m_avgy2 {0};
   double m_vary2 {0};
      
   double m_avgD2 {0};
   double m_varD2 {0};
      
   double m_avgmed2 {0};
   double m_varmed2 {0};
   
   
   double m_avgx3_accum {0};
   double m_avgx3sq_accum {0};
      
   double m_avgy3_accum {0};
   double m_avgy3sq_accum {0};
      
   double m_avgD3_accum {0};
   double m_avgD3sq_accum {0};
      
   double m_avgmed3_accum {0};
   double m_avgmed3sq_accum {0};
      
   double m_avgx3 {0};
   double m_varx3 {0};
      
   double m_avgy3 {0};
   double m_vary3 {0};
      
   double m_avgD3 {0};
   double m_varD3 {0};
      
   double m_avgmed3 {0};
   double m_varmed3 {0};
   
   double m_avgx4_accum {0};
   double m_avgx4sq_accum {0};
      
   double m_avgy4_accum {0};
   double m_avgy4sq_accum {0};
      
   double m_avgD4_accum {0};
   double m_avgD4sq_accum {0};
      
   double m_avgmed4_accum {0};
   double m_avgmed4sq_accum {0};
      
   double m_avgx4 {0};
   double m_varx4 {0};
      
   double m_avgy4 {0};
   double m_vary4 {0};
      
   double m_avgD4 {0};
   double m_varD4 {0};
      
   double m_avgmed4 {0};
   double m_varmed4 {0};
   
   double m_avgxAll_accum {0};
   double m_avgxAllsq_accum {0};
      
   double m_avgyAll_accum {0};
   double m_avgyAllsq_accum {0};
      
   double m_avgDAll_accum {0};
   double m_avgDAllsq_accum {0};
      
   double m_avgmedAll_accum {0};
   double m_avgmedAllsq_accum {0};
      
   double m_avgxAll {0};
   double m_varxAll {0};
      
   double m_avgyAll {0};
   double m_varyAll {0};
      
   double m_avgDAll {0};
   double m_varDAll {0};
      
   double m_avgmedAll {0};
   double m_varmedAll {0};
   
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
   
   pcf::IndiProperty m_indiP_averaging;
   INDI_NEWCALLBACK_DECL(pupilFit, m_indiP_averaging);
   
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
   
   config.add("setpoints.x1", "" , "setpoints.x1", argType::Required, "setpoints", "x1", false, "float", "The x-center of quadrant 1 (ll) set point");
   config.add("setpoints.y1", "" , "setpoints.y1", argType::Required, "setpoints", "y1", false, "float", "The y-center of quadrant 1 (ll) set point");
   config.add("setpoints.D1", "" , "setpoints.D1", argType::Required, "setpoints", "D1", false, "float", "The diameter of quadrant 1 (ll) set point");
   
   config.add("setpoints.x2", "" , "setpoints.x2", argType::Required, "setpoints", "x2", false, "float", "The x-center of quadrant 2 (ll) set point");
   config.add("setpoints.y2", "" , "setpoints.y2", argType::Required, "setpoints", "y2", false, "float", "The y-center of quadrant 2 (ll) set point");
   config.add("setpoints.D2", "" , "setpoints.D2", argType::Required, "setpoints", "D2", false, "float", "The diameter of quadrant 2 (ll) set point");
   
   config.add("setpoints.x3", "" , "setpoints.x3", argType::Required, "setpoints", "x3", false, "float", "The x-center of quadrant 3 (ll) set point");
   config.add("setpoints.y3", "" , "setpoints.y3", argType::Required, "setpoints", "y3", false, "float", "The y-center of quadrant 3 (ll) set point");
   config.add("setpoints.D3", "" , "setpoints.D3", argType::Required, "setpoints", "D3", false, "float", "The diameter of quadrant 3 (ll) set point");
   
   config.add("setpoints.x4", "" , "setpoints.x4", argType::Required, "setpoints", "x4", false, "float", "The x-center of quadrant 4 (ll) set point");
   config.add("setpoints.y4", "" , "setpoints.y4", argType::Required, "setpoints", "y4", false, "float", "The y-center of quadrant 4 (ll) set point");
   config.add("setpoints.D4", "" , "setpoints.D4", argType::Required, "setpoints", "D4", false, "float", "The diameter of quadrant 4 (ll) set point");
}


inline
int pupilFit::loadConfigImpl( mx::app::appConfigurator & _config )
{
   m_shmimName = "camwfs_avg";
   shmimMonitorT::loadConfig(_config);
      
   _config(m_threshold, "fit.threshold");
   _config(m_threshShmimName, "fit.threshShmimName");
   _config(m_edgeShmimName, "fit.edgeShmimName");
   
   _config(m_setx1, "setpoints.x1");
   _config(m_sety1, "setpoints.y1");
   _config(m_setD1, "setpoints.D1");
   
   _config(m_setx2, "setpoints.x2");
   _config(m_sety2, "setpoints.y2");
   _config(m_setD2, "setpoints.D2");
   
   _config(m_setx3, "setpoints.x3");
   _config(m_sety3, "setpoints.y3");
   _config(m_setD3, "setpoints.D3");
   
   _config(m_setx4, "setpoints.x4");
   _config(m_sety4, "setpoints.y4");
   _config(m_setD4, "setpoints.D4");
   
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
   
   createStandardIndiToggleSw( m_indiP_averaging, "averaging", "Averaging", "Start/Stop Averaging");
   m_indiP_averaging["toggle"].set(pcf::IndiElement::Off);
   if( registerIndiPropertyNew( m_indiP_averaging, INDI_NEWCALLBACK(m_indiP_averaging)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
      
   createROIndiNumber( m_indiP_quad1, "quadrant1", "Quadrant 1");
   indi::addNumberElement<float>( m_indiP_quad1, "x", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_quad1, "y", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_quad1, "D", 0, 59, 0, "%0.2f", "diameter");
   indi::addNumberElement<float>( m_indiP_quad1, "med", 0, std::numeric_limits<uint16_t>::max(), 0, "%0.1f", "flux");
   indi::addNumberElement<float>( m_indiP_quad1, "set-x", 0, 59, 0, "%0.2f", "set pt. center x");
   m_indiP_quad1["set-x"] = m_setx1;
   indi::addNumberElement<float>( m_indiP_quad1, "set-y", 0, 59, 0, "%0.2f", "set pt. center x");
   m_indiP_quad1["set-y"] = m_sety1;
   indi::addNumberElement<float>( m_indiP_quad1, "set-D", 0, 59, 0, "%0.2f", "set pt. diameter");
   m_indiP_quad1["set-D"] = m_setD1;
   
   registerIndiPropertyReadOnly(m_indiP_quad1);
   
   createROIndiNumber( m_indiP_quad2, "quadrant2", "Quadrant 2");
   indi::addNumberElement<float>( m_indiP_quad2, "x", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_quad2, "y", 0, 59, 0, "%0.2f", "center y");
   indi::addNumberElement<float>( m_indiP_quad2, "D", 0, 59, 0, "%0.2f", "diameter");
   indi::addNumberElement<float>( m_indiP_quad2, "med", 0, std::numeric_limits<uint16_t>::max(), 0, "%0.1f", "flux");
   indi::addNumberElement<float>( m_indiP_quad2, "set-x", 0, 59, 0, "%0.2f", "set pt. center x");
   m_indiP_quad2["set-x"] = m_setx2;
   indi::addNumberElement<float>( m_indiP_quad2, "set-y", 0, 59, 0, "%0.2f", "set pt. center x");
   m_indiP_quad2["set-y"] = m_sety2;
   indi::addNumberElement<float>( m_indiP_quad2, "set-D", 0, 59, 0, "%0.2f", "set pt. diameter");
   m_indiP_quad2["set-D"] = m_setD2;
   registerIndiPropertyReadOnly(m_indiP_quad2);
   
   createROIndiNumber( m_indiP_quad3, "quadrant3", "Quadrant 3");
   indi::addNumberElement<float>( m_indiP_quad3, "x", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_quad3, "y", 0, 59, 0, "%0.2f", "center y");
   indi::addNumberElement<float>( m_indiP_quad3, "D", 0, 59, 0, "%0.2f", "diameter");
   indi::addNumberElement<float>( m_indiP_quad3, "med", 0, std::numeric_limits<uint16_t>::max(), 0, "%0.1f", "flux");
   indi::addNumberElement<float>( m_indiP_quad3, "set-x", 0, 59, 0, "%0.2f", "set pt. center x");
   m_indiP_quad3["set-x"] = m_setx3;
   indi::addNumberElement<float>( m_indiP_quad3, "set-y", 0, 59, 0, "%0.2f", "set pt. center x");
   m_indiP_quad3["set-y"] = m_sety3;
   indi::addNumberElement<float>( m_indiP_quad3, "set-D", 0, 59, 0, "%0.2f", "set pt. diameter");
   m_indiP_quad3["set-D"] = m_setD3;
   registerIndiPropertyReadOnly(m_indiP_quad3);
   
   createROIndiNumber( m_indiP_quad4, "quadrant4", "Quadrant 4");
   indi::addNumberElement<float>( m_indiP_quad4, "x", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_quad4, "y", 0, 59, 0, "%0.2f", "center y");
   indi::addNumberElement<float>( m_indiP_quad4, "D", 0, 59, 0, "%0.2f", "diameter");
   indi::addNumberElement<float>( m_indiP_quad4, "med", 0, std::numeric_limits<uint16_t>::max(), 0, "%0.1f", "flux");
   indi::addNumberElement<float>( m_indiP_quad4, "set-x", 0, 59, 0, "%0.2f", "set pt. center x");
   m_indiP_quad4["set-x"] = m_setx4;
   indi::addNumberElement<float>( m_indiP_quad4, "set-y", 0, 59, 0, "%0.2f", "set pt. center x");
   m_indiP_quad4["set-y"] = m_sety4;
   indi::addNumberElement<float>( m_indiP_quad4, "set-D", 0, 59, 0, "%0.2f", "set pt. diameter");
   m_indiP_quad4["set-D"] = m_setD4;
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
   
   ///\todo need a more robust corner averaging system here.
   m_fitIm -= 0.25*( m_fitIm(0,0) + m_fitIm(0,119) + m_fitIm(119,119) + m_fitIm(119,0));
   
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
   
   
   if(m_averaging)
   {
      ++m_navg;
      
      m_avgx1_accum += m_fitter.m_avgx[0];
      m_avgx1sq_accum += m_fitter.m_avgx[0]*m_fitter.m_avgx[0];
      
      m_avgy1_accum += m_fitter.m_avgy[0];
      m_avgy1sq_accum += m_fitter.m_avgy[0]*m_fitter.m_avgy[0];
      
      m_avgD1_accum += 2*m_fitter.m_avgr[0];
      m_avgD1sq_accum += 4*m_fitter.m_avgr[0]*m_fitter.m_avgr[0];
      
      m_avgmed1_accum += m_fitter.m_med[0];
      m_avgmed1sq_accum += m_fitter.m_med[0]*m_fitter.m_med[0];
      
      m_avgx1 = m_avgx1_accum / m_navg;
      m_varx1 = m_avgx1sq_accum / m_navg - m_avgx1*m_avgx1;
      
      m_avgy1 = m_avgy1_accum / m_navg;
      m_vary1 = m_avgy1sq_accum / m_navg - m_avgy1*m_avgy1;
      
      m_avgD1 = m_avgD1_accum / m_navg;
      m_varD1 = m_avgD1sq_accum / m_navg - m_avgD1*m_avgD1;
      
      m_avgmed1 = m_avgmed1_accum / m_navg;
      m_varmed1 = m_avgmed1sq_accum / m_navg - m_avgmed1*m_avgmed1;
      
      m_avgx2_accum += m_fitter.m_avgx[1];
      m_avgx2sq_accum += m_fitter.m_avgx[1]*m_fitter.m_avgx[1];
      
      m_avgy2_accum += m_fitter.m_avgy[1];
      m_avgy2sq_accum += m_fitter.m_avgy[1]*m_fitter.m_avgy[1];
      
      m_avgD2_accum += 2*m_fitter.m_avgr[1];
      m_avgD2sq_accum += 4*m_fitter.m_avgr[1]*m_fitter.m_avgr[1];
      
      m_avgmed2_accum += m_fitter.m_med[1];
      m_avgmed2sq_accum += m_fitter.m_med[1]*m_fitter.m_med[1];
      
      m_avgx2 = m_avgx2_accum / m_navg;
      m_varx2 = m_avgx2sq_accum / m_navg - m_avgx2*m_avgx2;
      
      m_avgy2 = m_avgy2_accum / m_navg;
      m_vary2 = m_avgy2sq_accum / m_navg - m_avgy2*m_avgy2;
      
      m_avgD2 = m_avgD2_accum / m_navg;
      m_varD2 = m_avgD2sq_accum / m_navg - m_avgD2*m_avgD2;
      
      m_avgmed2 = m_avgmed2_accum / m_navg;
      m_varmed2 = m_avgmed2sq_accum / m_navg - m_avgmed2*m_avgmed2;
      
      
      
      m_avgx3_accum += m_fitter.m_avgx[2];
      m_avgx3sq_accum += m_fitter.m_avgx[2]*m_fitter.m_avgx[2];
      
      m_avgy3_accum += m_fitter.m_avgy[2];
      m_avgy3sq_accum += m_fitter.m_avgy[2]*m_fitter.m_avgy[2];
      
      m_avgD3_accum += 2*m_fitter.m_avgr[2];
      m_avgD3sq_accum += 4*m_fitter.m_avgr[2]*m_fitter.m_avgr[2];
      
      m_avgmed3_accum += m_fitter.m_med[2];
      m_avgmed3sq_accum += m_fitter.m_med[2]*m_fitter.m_med[2];
      
      m_avgx3 = m_avgx3_accum / m_navg;
      m_varx3 = m_avgx3sq_accum / m_navg - m_avgx3*m_avgx3;
      
      m_avgy3 = m_avgy3_accum / m_navg;
      m_vary3 = m_avgy3sq_accum / m_navg - m_avgy3*m_avgy3;
      
      m_avgD3 = m_avgD3_accum / m_navg;
      m_varD3 = m_avgD3sq_accum / m_navg - m_avgD3*m_avgD3;
      
      m_avgmed3 = m_avgmed3_accum / m_navg;
      m_varmed3 = m_avgmed3sq_accum / m_navg - m_avgmed3*m_avgmed3;
      
      
      m_avgx4_accum += m_fitter.m_avgx[3];
      m_avgx4sq_accum += m_fitter.m_avgx[3]*m_fitter.m_avgx[3];
      
      m_avgy4_accum += m_fitter.m_avgy[3];
      m_avgy4sq_accum += m_fitter.m_avgy[3]*m_fitter.m_avgy[3];
      
      m_avgD4_accum += 2*m_fitter.m_avgr[3];
      m_avgD4sq_accum += 4*m_fitter.m_avgr[3]*m_fitter.m_avgr[3];
      
      m_avgmed4_accum += m_fitter.m_med[3];
      m_avgmed4sq_accum += m_fitter.m_med[3]*m_fitter.m_med[3];
      
      m_avgx4 = m_avgx4_accum / m_navg;
      m_varx4 = m_avgx4sq_accum / m_navg - m_avgx4*m_avgx4;
      
      m_avgy4 = m_avgy4_accum / m_navg;
      m_vary4 = m_avgy4sq_accum / m_navg - m_avgy4*m_avgy4;
      
      m_avgD4 = m_avgD4_accum / m_navg;
      m_varD4 = m_avgD4sq_accum / m_navg - m_avgD4*m_avgD4;
      
      m_avgmed4 = m_avgmed4_accum / m_navg;
      m_varmed4 = m_avgmed4sq_accum / m_navg - m_avgmed4*m_avgmed4;
      
      
      double tmp = 0.25*(m_fitter.m_avgx[0]+m_fitter.m_avgx[1]+m_fitter.m_avgx[2]+m_fitter.m_avgx[3]);
      m_avgxAll_accum += tmp;
      m_avgxAllsq_accum += tmp*tmp;
      
      tmp = 0.25*(m_fitter.m_avgy[0]+m_fitter.m_avgy[1]+m_fitter.m_avgy[2]+m_fitter.m_avgy[3]);
      m_avgyAll_accum += tmp;
      m_avgyAllsq_accum += tmp*tmp;
      
      tmp = 2*0.25*(m_fitter.m_avgr[0]+m_fitter.m_avgr[1]+m_fitter.m_avgr[2]+m_fitter.m_avgr[3]);
      m_avgDAll_accum += tmp;
      m_avgDAllsq_accum += tmp*tmp;
      
      tmp = 0.25*(m_fitter.m_med[0]+m_fitter.m_med[1]+m_fitter.m_med[2]+m_fitter.m_med[3]);
      m_avgmedAll_accum += tmp;
      m_avgmedAllsq_accum += tmp*tmp;
      
      m_avgxAll = m_avgxAll_accum / m_navg;
      m_varxAll = m_avgxAllsq_accum / m_navg - m_avgxAll*m_avgxAll;
      
      m_avgyAll = m_avgyAll_accum / m_navg;
      m_varyAll = m_avgyAllsq_accum / m_navg - m_avgyAll*m_avgyAll;
      
      m_avgDAll = m_avgDAll_accum / m_navg;
      m_varDAll = m_avgDAllsq_accum / m_navg - m_avgDAll*m_avgDAll;
      
      m_avgmedAll = m_avgmedAll_accum / m_navg;
      m_varmedAll = m_avgmedAllsq_accum / m_navg - m_avgmedAll*m_avgmedAll;
      
      
      std::cerr << "****************************************************************\n";
      std::cerr << "Averaged: " << m_navg << "\n";
      std::cerr << "Average x1: " << m_avgx1 << " +/- " << sqrt(m_varx1) << "\n";
      std::cerr << "Average y1: " << m_avgy1 << " +/- " << sqrt(m_vary1) << "\n";
      std::cerr << "Average D1: " << m_avgD1 << " +/- " << sqrt(m_varD1) << "\n";
      std::cerr << "Average med1: " << m_avgmed1 << " +/- " << sqrt(m_varmed1) << "\n\n";
      std::cerr << "Average x2: " << m_avgx2 << " +/- " << sqrt(m_varx2) << "\n";
      std::cerr << "Average y2: " << m_avgy2 << " +/- " << sqrt(m_vary2) << "\n";
      std::cerr << "Average D2: " << m_avgD2 << " +/- " << sqrt(m_varD2) << "\n";
      std::cerr << "Average med2: " << m_avgmed2 << " +/- " << sqrt(m_varmed2) << "\n\n";
      std::cerr << "Average x3: " << m_avgx3 << " +/- " << sqrt(m_varx3) << "\n";
      std::cerr << "Average y3: " << m_avgy3 << " +/- " << sqrt(m_vary3) << "\n";
      std::cerr << "Average D3: " << m_avgD3 << " +/- " << sqrt(m_varD3) << "\n";
      std::cerr << "Average med3: " << m_avgmed3 << " +/- " << sqrt(m_varmed3) << "\n\n";
      std::cerr << "Average x4: " << m_avgx4 << " +/- " << sqrt(m_varx4) << "\n";
      std::cerr << "Average y4: " << m_avgy4 << " +/- " << sqrt(m_vary4) << "\n";
      std::cerr << "Average D4: " << m_avgD4 << " +/- " << sqrt(m_varD4) << "\n";
      std::cerr << "Average med4: " << m_avgmed4 << " +/- " << sqrt(m_varmed4) << "\n\n";
      std::cerr << "Average xAll: " << m_avgxAll << " +/- " << sqrt(m_varxAll) << "\n";
      std::cerr << "Average yAll: " << m_avgyAll << " +/- " << sqrt(m_varyAll) << "\n";
      std::cerr << "Average DAll: " << m_avgDAll << " +/- " << sqrt(m_varDAll) << "\n";
      std::cerr << "Average medAll: " << m_avgmedAll << " +/- " << sqrt(m_varmedAll) << "\n\n";
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

INDI_NEWCALLBACK_DEFN(pupilFit, m_indiP_averaging)(const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getName() != m_indiP_averaging.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      m_navg = 0;
      m_averaging = true;   
      
      m_avgx1_accum = 0;
      m_avgx1sq_accum = 0;
      m_avgy1_accum = 0;
      m_avgy1sq_accum = 0;
      m_avgD1_accum = 0;
      m_avgD1sq_accum = 0;
      m_avgmed1_accum = 0;
      m_avgmed1sq_accum = 0;
      
      m_avgx2_accum = 0;
      m_avgx2sq_accum = 0;
      m_avgy2_accum = 0;
      m_avgy2sq_accum = 0;
      m_avgD2_accum = 0;
      m_avgD2sq_accum = 0;
      m_avgmed2_accum = 0;
      m_avgmed2sq_accum = 0;
      
      m_avgx3_accum = 0;
      m_avgx3sq_accum = 0;
      m_avgy3_accum = 0;
      m_avgy3sq_accum = 0;
      m_avgD3_accum = 0;
      m_avgD3sq_accum = 0;
      m_avgmed3_accum = 0;
      m_avgmed3sq_accum = 0;
      
      m_avgx4_accum = 0;
      m_avgx4sq_accum = 0;
      m_avgy4_accum = 0;
      m_avgy4sq_accum = 0;
      m_avgD4_accum = 0;
      m_avgD4sq_accum = 0;
      m_avgmed4_accum = 0;
      m_avgmed4sq_accum = 0;
      
      m_avgxAll_accum = 0;
      m_avgxAllsq_accum = 0;
      m_avgyAll_accum = 0;
      m_avgyAllsq_accum = 0;
      m_avgDAll_accum = 0;
      m_avgDAllsq_accum = 0;
      m_avgmedAll_accum = 0;
      m_avgmedAllsq_accum = 0;
      
      updateSwitchIfChanged(m_indiP_averaging, "toggle", pcf::IndiElement::On, INDI_BUSY);

      log<text_log>("began averaging");
      
   }   
   else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      m_averaging = false;
      updateSwitchIfChanged(m_indiP_averaging, "toggle", pcf::IndiElement::Off, INDI_IDLE);

      log<text_log>("stopped averaging");
   }
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //pupilFit_hpp
