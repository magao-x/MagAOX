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
  * <a href="../handbook/operating/software/apps/pupilFit.html">Application Documentation</a>
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
   
   std::string m_wfsrefPath {"/opt/MagAOX/cacao/tweeter/conf"};
   std::string m_wfsrefName {"wfsref0.fits"};
   
   int m_numPupils {4}; ///< The number of pupils.  Default is 4.  3 is also supported.
   ///@}

   mx::improc::eigenImage<float> m_refIm;
   mx::improc::eigenImage<float> m_fitIm;
   mx::improc::eigenImage<float> m_edgeIm;

   pupilFitter<realT> m_fitter;
   
   IMAGE m_threshShmim;
   bool m_threshShmimConnected {false};
   
   IMAGE m_edgeShmim;
   bool m_edgeShmimConnected {false};
   
   bool m_useRefIm {false};
   
   bool m_refUpdated {false}; ///< Flag set if the online reference update is used.
   
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
   
   pcf::IndiProperty m_indiP_numPupils;
   
   pcf::IndiProperty m_indiP_quad1;
   pcf::IndiProperty m_indiP_quad2;
   pcf::IndiProperty m_indiP_quad3;
   pcf::IndiProperty m_indiP_quad4;
   
   pcf::IndiProperty m_indiP_avg;
   
   pcf::IndiProperty m_indiP_reload;
   INDI_NEWCALLBACK_DECL(pupilFit, m_indiP_reload);
   
   pcf::IndiProperty m_indiP_update;
   INDI_NEWCALLBACK_DECL(pupilFit, m_indiP_update);
   
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
      
   config.add("fit.numPupils", "", "fit.numPupils", argType::Required, "fit", "numPupils", false, "int", "The number of pupils.  Default is 4.  3 is also supported.");
   config.add("fit.pupMedIndex", "", "fit.pupMedIndex", argType::Required, "fit", "pupMedIndex", false, "float", "The index of the pupil median in a sorted quadrant.");
   
   config.add("wfsref.path", "" , "wfsref.path", argType::Required, "wfsref", "path", false, "float", "The path to the WFS reference image.  Default is /opt/MagAOX/cacao/tweeter");
   config.add("wfsref.name", "" , "wfsref.name", argType::Required, "wfsref", "name", false, "float", "The name the WFS reference image. Default is wfsref0.fits");
}


inline
int pupilFit::loadConfigImpl( mx::app::appConfigurator & _config )
{
   m_shmimName = "camwfs_avg";
   shmimMonitorT::loadConfig(_config);
      
   _config(m_threshold, "fit.threshold");
   _config(m_threshShmimName, "fit.threshShmimName");
   _config(m_edgeShmimName, "fit.edgeShmimName");
   _config(m_numPupils, "fit.numPupils");
   _config(m_fitter.m_pupMedIndex, "fit.pupMedIndex");
   
   _config(m_wfsrefPath, "wfsref.path");
   _config(m_wfsrefName, "wfsref.name");
   
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
   
   createStandardIndiToggleSw( m_indiP_averaging, "averaging", "Start/Stop Averaging");
   m_indiP_averaging["toggle"].set(pcf::IndiElement::Off);
   if( registerIndiPropertyNew( m_indiP_averaging, INDI_NEWCALLBACK(m_indiP_averaging)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
      
   createROIndiNumber( m_indiP_numPupils, "numPupils", "Number of Pupils");
   indi::addNumberElement<int>( m_indiP_numPupils, "value", 3, 4, 1, "%d", "");
   m_indiP_numPupils["value"].set(m_numPupils);
   registerIndiPropertyReadOnly(m_indiP_numPupils);
   
   createROIndiNumber( m_indiP_quad1, "quadrant1", "Quadrant 1");
   indi::addNumberElement<float>( m_indiP_quad1, "x", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_quad1, "dx", 0, 59, 0, "%0.2f", "delta-x");
   indi::addNumberElement<float>( m_indiP_quad1, "y", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_quad1, "dy", 0, 59, 0, "%0.2f", "delta-y");
   indi::addNumberElement<float>( m_indiP_quad1, "D", 0, 59, 0, "%0.2f", "diameter");
   indi::addNumberElement<float>( m_indiP_quad1, "dD", 0, 59, 0, "%0.2f", "delta-D");
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
   indi::addNumberElement<float>( m_indiP_quad2, "dx", 0, 59, 0, "%0.2f", "delta-x");
   indi::addNumberElement<float>( m_indiP_quad2, "y", 0, 59, 0, "%0.2f", "center y");
   indi::addNumberElement<float>( m_indiP_quad2, "dy", 0, 59, 0, "%0.2f", "delta-y");
   indi::addNumberElement<float>( m_indiP_quad2, "D", 0, 59, 0, "%0.2f", "diameter");
   indi::addNumberElement<float>( m_indiP_quad2, "dD", 0, 59, 0, "%0.2f", "delta-D");
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
   indi::addNumberElement<float>( m_indiP_quad3, "dx", 0, 59, 0, "%0.2f", "delta-x");
   indi::addNumberElement<float>( m_indiP_quad3, "y", 0, 59, 0, "%0.2f", "center y");
   indi::addNumberElement<float>( m_indiP_quad3, "dy", 0, 59, 0, "%0.2f", "delta-y");
   indi::addNumberElement<float>( m_indiP_quad3, "D", 0, 59, 0, "%0.2f", "diameter");
   indi::addNumberElement<float>( m_indiP_quad3, "dD", 0, 59, 0, "%0.2f", "delta-D");
   indi::addNumberElement<float>( m_indiP_quad3, "med", 0, std::numeric_limits<uint16_t>::max(), 0, "%0.1f", "flux");
   indi::addNumberElement<float>( m_indiP_quad3, "set-x", 0, 59, 0, "%0.2f", "set pt. center x");
   m_indiP_quad3["set-x"] = m_setx3;
   indi::addNumberElement<float>( m_indiP_quad3, "set-y", 0, 59, 0, "%0.2f", "set pt. center x");
   m_indiP_quad3["set-y"] = m_sety3;
   indi::addNumberElement<float>( m_indiP_quad3, "set-D", 0, 59, 0, "%0.2f", "set pt. diameter");
   m_indiP_quad3["set-D"] = m_setD3;
   registerIndiPropertyReadOnly(m_indiP_quad3);
   
   if(m_numPupils != 3)
   {
      createROIndiNumber( m_indiP_quad4, "quadrant4", "Quadrant 4");
      indi::addNumberElement<float>( m_indiP_quad4, "x", 0, 59, 0, "%0.2f", "center x");
      indi::addNumberElement<float>( m_indiP_quad4, "dx", 0, 59, 0, "%0.2f", "delta-x");
      indi::addNumberElement<float>( m_indiP_quad4, "y", 0, 59, 0, "%0.2f", "center y");
      indi::addNumberElement<float>( m_indiP_quad4, "dy", 0, 59, 0, "%0.2f", "delta-y");
      indi::addNumberElement<float>( m_indiP_quad4, "D", 0, 59, 0, "%0.2f", "diameter");
      indi::addNumberElement<float>( m_indiP_quad4, "dD", 0, 59, 0, "%0.2f", "delta-D");
      indi::addNumberElement<float>( m_indiP_quad4, "med", 0, std::numeric_limits<uint16_t>::max(), 0, "%0.1f", "flux");
      indi::addNumberElement<float>( m_indiP_quad4, "set-x", 0, 59, 0, "%0.2f", "set pt. center x");
      m_indiP_quad4["set-x"] = m_setx4;
      indi::addNumberElement<float>( m_indiP_quad4, "set-y", 0, 59, 0, "%0.2f", "set pt. center x");
      m_indiP_quad4["set-y"] = m_sety4;
      indi::addNumberElement<float>( m_indiP_quad4, "set-D", 0, 59, 0, "%0.2f", "set pt. diameter");
      m_indiP_quad4["set-D"] = m_setD4;
      registerIndiPropertyReadOnly(m_indiP_quad4);
   }
   
   createROIndiNumber( m_indiP_avg, "average", "Average");
   indi::addNumberElement<float>( m_indiP_avg, "x", 0, 59, 0, "%0.2f", "center x");
   indi::addNumberElement<float>( m_indiP_avg, "dx", 0, 59, 0, "%0.2f", "delta-x");
   indi::addNumberElement<float>( m_indiP_avg, "y", 0, 59, 0, "%0.2f", "center y");
   indi::addNumberElement<float>( m_indiP_avg, "dy", 0, 59, 0, "%0.2f", "delta-y");
   indi::addNumberElement<float>( m_indiP_avg, "D", 0, 59, 0, "%0.2f", "diameter");
   indi::addNumberElement<float>( m_indiP_avg, "dD", 0, 59, 0, "%0.2f", "delta-D");
   registerIndiPropertyReadOnly(m_indiP_avg);
   
   createStandardIndiRequestSw( m_indiP_reload, "reload", "Reload Calibration");
   m_indiP_reload["request"].set(pcf::IndiElement::Off);
   if( registerIndiPropertyNew( m_indiP_reload, INDI_NEWCALLBACK(m_indiP_reload)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   createStandardIndiRequestSw( m_indiP_update, "update_ref", "Update Reference");
   m_indiP_update["request"].set(pcf::IndiElement::Off);
   if( registerIndiPropertyNew( m_indiP_update, INDI_NEWCALLBACK(m_indiP_update)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
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
   
   m_fitter.m_numPupils = m_numPupils;
   m_fitter.setSize(0.5*m_width, 0.5*m_height);
   m_fitter.m_thresh = m_threshold;
   
   //Load and fit the reference image
   std::string reffits = m_wfsrefPath + "/" + m_wfsrefName;
  
   mx::improc::fitsFile<float> ff;
   mx::improc::eigenImage<float> refedge;
   
   ff.read(m_refIm, reffits);
   
   if(m_useRefIm)
   {
      if(m_refIm.rows() == m_width && m_refIm.cols() == m_height)
      {
         if(m_fitter.fit(m_refIm, refedge) < 0)
         {
            log<software_error>({__FILE__, __LINE__, "error from fitter"});
         }
         else
         {
            m_setx1 = m_fitter.m_avgx[0];
            m_sety1 = m_fitter.m_avgy[0];
            m_setD1 = 2*m_fitter.m_avgr[0];
            
            m_setx2 = m_fitter.m_avgx[1];
            m_sety2 = m_fitter.m_avgy[1];
            m_setD2 = 2*m_fitter.m_avgr[1];
            
            m_setx3 = m_fitter.m_avgx[2];
            m_sety3 = m_fitter.m_avgy[2];
            m_setD3 = 2*m_fitter.m_avgr[2];
            
            m_setx4 = m_fitter.m_avgx[3];
            m_sety4 = m_fitter.m_avgy[3];
            m_setD4 = 2*m_fitter.m_avgr[3];
            
            log<text_log>("Read reference image: " + reffits);
            log<text_log>("Quad 1 set points: " + std::to_string(m_setx1) + " " +  std::to_string(m_sety1) + " " + std::to_string(m_setD1));
            log<text_log>("Quad 2 set points: " + std::to_string(m_setx2) + " " +  std::to_string(m_sety2) + " " + std::to_string(m_setD2));
            log<text_log>("Quad 3 set points: " + std::to_string(m_setx3) + " " +  std::to_string(m_sety3) + " " + std::to_string(m_setD3));
            log<text_log>("Quad 4 set points: " + std::to_string(m_setx4) + " " +  std::to_string(m_sety4) + " " + std::to_string(m_setD4));
         }
      }
      else
      {
         log<text_log>("Reference image " + reffits + " size does not match shmim stream.", logPrio::LOG_ERROR);
      }
   }
   else
   {
      if(m_numPupils == 4 && !m_refUpdated)
      {
         m_setx1 = 29.5;
         m_sety1 = 29.5;
         m_setD1 = 56.0;
         
         m_setx2 = 89.5;
         m_sety2 = 29.5;
         m_setD2 = 56.0;
         
         m_setx3 = 29.5;
         m_sety3 = 89.5;
         m_setD3 = 56.0;
         
         m_setx4 = 89.5;
         m_sety4 = 89.5;
         m_setD4 = 56.0;
      }
      else if(!m_refUpdated)
      {
         m_setx1 = 30.0;
         m_sety1 = 38.0;
         m_setD1 = 14.0;
         
         m_setx2 = 96.0;
         m_sety2 = 38.0;
         m_setD2 = 14.0;
         
         m_setx3 = 65.0;
         m_sety3 = 95.0;
         m_setD3 = 14.0;
      }
   }
   
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
   m_fitIm -= 0.25*( m_fitIm(0,0) + m_fitIm(0,m_height-1) + m_fitIm(m_width-1,m_height-1) + m_fitIm(m_width-1,0));
   
   m_fitter.m_thresh = m_threshold;
   
   m_fitter.fit(m_fitIm, m_edgeIm);
   
   {//mutex scope
      
      std::lock_guard<std::mutex> guard(m_indiMutex);
      m_indiP_quad1["set-x"].set(m_setx1);
      m_indiP_quad1["x"].set(m_fitter.m_avgx[0]);
      m_indiP_quad1["dx"].set(m_fitter.m_avgx[0]-m_setx1);
      m_indiP_quad1["set-y"].set(m_sety1);
      m_indiP_quad1["y"].set(m_fitter.m_avgy[0]);
      m_indiP_quad1["dy"].set(m_fitter.m_avgy[0]-m_sety1);
      m_indiP_quad1["set-D"].set(m_setD1);
      m_indiP_quad1["D"].set(2*m_fitter.m_avgr[0]);
      m_indiP_quad1["dD"].set(2*m_fitter.m_avgr[0]-m_setD1);
      m_indiP_quad1["med"].set(m_fitter.m_med[0]);
      m_indiP_quad1.setState (INDI_BUSY);
      m_indiDriver->sendSetProperty (m_indiP_quad1);
   
      m_indiP_quad2["set-x"].set(m_setx2);
      m_indiP_quad2["x"].set(m_fitter.m_avgx[1]);
      m_indiP_quad2["dx"].set(m_fitter.m_avgx[1]-m_setx2);
      m_indiP_quad2["set-y"].set(m_sety2);
      m_indiP_quad2["y"].set(m_fitter.m_avgy[1]);
      m_indiP_quad2["dy"].set(m_fitter.m_avgy[1]-m_sety2);
      m_indiP_quad2["set-D"].set(m_setD2);
      m_indiP_quad2["D"].set(2*m_fitter.m_avgr[1]);
      m_indiP_quad2["dD"].set(2*m_fitter.m_avgr[1]-m_setD2);
      m_indiP_quad2["med"].set(m_fitter.m_med[1]);
      m_indiP_quad2.setState (INDI_BUSY);
      m_indiDriver->sendSetProperty (m_indiP_quad2);
   
      m_indiP_quad3["set-x"].set(m_setx3);
      m_indiP_quad3["x"].set(m_fitter.m_avgx[2]);
      m_indiP_quad3["dx"].set(m_fitter.m_avgx[2]-m_setx3);
      m_indiP_quad3["set-y"].set(m_sety3);
      m_indiP_quad3["y"].set(m_fitter.m_avgy[2]);
      m_indiP_quad3["dy"].set(m_fitter.m_avgy[2]-m_sety3);
      m_indiP_quad3["set-D"].set(m_setD3);
      m_indiP_quad3["D"].set(2*m_fitter.m_avgr[2]);
      m_indiP_quad3["dD"].set(2*m_fitter.m_avgr[2]-m_setD3);
      m_indiP_quad3["med"].set(m_fitter.m_med[2]);
      m_indiP_quad3.setState (INDI_BUSY);
      m_indiDriver->sendSetProperty (m_indiP_quad3);
   
      if(m_numPupils == 3)
      {
         m_indiP_avg["x"].set(.333*(m_fitter.m_avgx[0] + m_fitter.m_avgx[1] + m_fitter.m_avgx[2]));
         m_indiP_avg["y"].set(.333*(m_fitter.m_avgy[0] + m_fitter.m_avgy[1] + m_fitter.m_avgy[2]));
         m_indiP_avg["D"].set(.667*(m_fitter.m_avgr[0] + m_fitter.m_avgr[1] + m_fitter.m_avgr[2]));
         
         m_indiP_avg["dx"].set(.333*(m_fitter.m_avgx[0] + m_fitter.m_avgx[1] + m_fitter.m_avgx[2]) - 0.333*(m_setx1 + m_setx2 + m_setx3));
         m_indiP_avg["dy"].set(.333*(m_fitter.m_avgy[0] + m_fitter.m_avgy[1] + m_fitter.m_avgy[2]) - 0.333*(m_sety1 + m_sety2 + m_sety3));
         m_indiP_avg["dD"].set(.667*(m_fitter.m_avgr[0] + m_fitter.m_avgr[1] + m_fitter.m_avgr[2]) - 0.333*(m_setD1 + m_setD2 + m_setD3));
      }
      else
      {
         m_indiP_quad4["set-x"].set(m_setx4);
         m_indiP_quad4["x"].set(m_fitter.m_avgx[3]);
         m_indiP_quad4["dx"].set(m_fitter.m_avgx[3]-m_setx4);
         m_indiP_quad4["set-y"].set(m_sety4);
         m_indiP_quad4["y"].set(m_fitter.m_avgy[3]);
         m_indiP_quad4["dy"].set(m_fitter.m_avgy[3]-m_sety4);
         m_indiP_quad4["set-D"].set(m_setD4);
         m_indiP_quad4["D"].set(2*m_fitter.m_avgr[3]);
         m_indiP_quad4["dD"].set(2*m_fitter.m_avgr[3]-m_setD4);
         m_indiP_quad4["med"].set(m_fitter.m_med[3]);
         m_indiP_quad4.setState (INDI_BUSY);
         m_indiDriver->sendSetProperty (m_indiP_quad4);
         
         m_indiP_avg["x"].set(.25*(m_fitter.m_avgx[0] + m_fitter.m_avgx[1] + m_fitter.m_avgx[2] + m_fitter.m_avgx[3]));
         m_indiP_avg["y"].set(.25*(m_fitter.m_avgy[0] + m_fitter.m_avgy[1] + m_fitter.m_avgy[2] + m_fitter.m_avgy[3]));
         m_indiP_avg["D"].set(.5*(m_fitter.m_avgr[0] + m_fitter.m_avgr[1] + m_fitter.m_avgr[2] + m_fitter.m_avgr[3]));
         
         m_indiP_avg["dx"].set(.25*(m_fitter.m_avgx[0] + m_fitter.m_avgx[1] + m_fitter.m_avgx[2] + m_fitter.m_avgx[3]) - 0.25*(m_setx1 + m_setx2 + m_setx3 + m_setx4));
         m_indiP_avg["dy"].set(.25*(m_fitter.m_avgy[0] + m_fitter.m_avgy[1] + m_fitter.m_avgy[2] + m_fitter.m_avgy[3]) - 0.25*(m_sety1 + m_sety2 + m_sety3 + m_sety4));
         m_indiP_avg["dD"].set(.5*(m_fitter.m_avgr[0] + m_fitter.m_avgr[1] + m_fitter.m_avgr[2] + m_fitter.m_avgr[3]) - 0.25*(m_setD1 + m_setD2 + m_setD3 + m_setD4));
         m_indiDriver->sendSetProperty (m_indiP_avg);
         
      }
      
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
      
      if(m_numPupils == 3)
      {
         double tmp = 0.333*(m_fitter.m_avgx[0]+m_fitter.m_avgx[1]+m_fitter.m_avgx[2]+m_fitter.m_avgx[3]);
         m_avgxAll_accum += tmp;
         m_avgxAllsq_accum += tmp*tmp;
         
         tmp = 0.333*(m_fitter.m_avgy[0]+m_fitter.m_avgy[1]+m_fitter.m_avgy[2]);
         m_avgyAll_accum += tmp;
         m_avgyAllsq_accum += tmp*tmp;
         
         tmp = 2*0.333*(m_fitter.m_avgr[0]+m_fitter.m_avgr[1]+m_fitter.m_avgr[2]);
         m_avgDAll_accum += tmp;
         m_avgDAllsq_accum += tmp*tmp;
         
         tmp = 0.333*(m_fitter.m_med[0]+m_fitter.m_med[1]+m_fitter.m_med[2]);
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
         std::cerr << "Average xAll: " << m_avgxAll << " +/- " << sqrt(m_varxAll) << "\n";
         std::cerr << "Average yAll: " << m_avgyAll << " +/- " << sqrt(m_varyAll) << "\n";
         std::cerr << "Average DAll: " << m_avgDAll << " +/- " << sqrt(m_varDAll) << "\n";
         std::cerr << "Average medAll: " << m_avgmedAll << " +/- " << sqrt(m_varmedAll) << "\n\n";
      }
      else
      {
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
   
   m_restart = true; //need to re-process the reference
   
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

INDI_NEWCALLBACK_DEFN(pupilFit, m_indiP_reload)(const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getName() != m_indiP_reload.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   
   
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      log<text_log>("reloading");
      m_restart = 1;
   }
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(pupilFit, m_indiP_update)(const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getName() != m_indiP_update.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   
   
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      std::lock_guard<std::mutex> guard(m_indiMutex);
      
      log<text_log>("updating cal");
      m_setx1 =  m_indiP_quad1["x"].get<float>();
      m_sety1 =  m_indiP_quad1["y"].get<float>();
      m_setD1 =  m_indiP_quad1["D"].get<float>();
      
      m_setx2 =  m_indiP_quad2["x"].get<float>();
      m_sety2 =  m_indiP_quad2["y"].get<float>();
      m_setD2 =  m_indiP_quad2["D"].get<float>();
      
      m_setx3 =  m_indiP_quad3["x"].get<float>();
      m_sety3 =  m_indiP_quad3["y"].get<float>();
      m_setD3 =  m_indiP_quad3["D"].get<float>();
      
      if(m_numPupils == 4)
      {
         m_setx4 =  m_indiP_quad4["x"].get<float>();
         m_sety4 =  m_indiP_quad4["y"].get<float>();
         m_setD4 =  m_indiP_quad4["D"].get<float>();
      }
      
      m_refUpdated = true;
   }
   
   return 0;
}
      
} //namespace app
} //namespace MagAOX

#endif //pupilFit_hpp
