/** \file t2wOffloader.hpp
  * \brief The MagAO-X tweeter to woofer offloading manager
  *
  * \ingroup app_files
  */

#ifndef t2wOffloader_hpp
#define t2wOffloader_hpp

#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/sigproc/gramSchmidt.hpp>
#include <mx/math/templateBLAS.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

/** \defgroup t2wOffloader Tweeter to Woofer Offloading
  * \brief Monitors the averaged tweeter shape, and sends it to the woofer.
  *
  * <a href="../handbook/operating/software/apps/t2wOffloader.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup t2wOffloader_files Tweeter to Woofer Offloading
  * \ingroup t2wOffloader
  */

/** MagAO-X application to control offloading the tweeter to the woofer.
  *
  * \ingroup t2wOffloader
  * 
  */
class t2wOffloader : public MagAOXApp<true>, public dev::shmimMonitor<t2wOffloader>
{

   //Give the test harness access.
   friend class t2wOffloader_test;

   friend class dev::shmimMonitor<t2wOffloader>;
   
   //The base shmimMonitor type
   typedef dev::shmimMonitor<t2wOffloader> shmimMonitorT;
      
   ///Floating point type in which to do all calculations.
   typedef float realT;
   
protected:

   /** \name Configurable Parameters
     *@{
     */
   
   std::string m_twRespMPath;
   
   std::string m_dmChannel ;
   
   std::string m_fpsSource {"camwfs"};
   std::string m_navgSource {"dmtweeter-avg"};
   float m_gain {0.1};
   float m_leak {0.0};
   
   float m_actLim {7.0}; ///< the upper limit on woofer actuator commands.  default is 7.0.
   
   std::string m_tweeterModeFile; ///< File containing the tweeter modes to use for offloading
   std::string m_tweeterMaskFile;

   int m_maxModes {50};

   int m_numModes {0};
   ///@}

   mx::improc::eigenImage<realT> m_twRespM;
   mx::improc::eigenImage<realT> m_tweeter;
   mx::improc::eigenImage<realT> m_woofer;
   mx::improc::eigenImage<realT> m_wooferDelta;
   mx::improc::eigenImage<realT> m_modeAmps;


   mx::improc::eigenImage<realT> m_tweeterMask;

   mx::improc::eigenCube<float> m_tModesOrtho;

   mx::improc::eigenCube<float> m_wModes;

   float m_fps {0}; ///< Current FPS from the FPS source.
   int m_navg {0}; ///< Current navg from the averager

   float m_effFPS {0};

   IMAGE m_dmStream; 
   uint32_t m_dmWidth {0}; ///< The width of the image
   uint32_t m_dmHeight {0}; ///< The height of the image.
   
   uint8_t m_dmDataType{0}; ///< The ImageStreamIO type code.
   size_t m_dmTypeSize {0}; ///< The size of the type, in bytes.  
   
   bool m_dmOpened {false};
   bool m_dmRestart {false};
   
   bool m_offloading {false};
   
public:
   /// Default c'tor.
   t2wOffloader();

   /// D'tor, declared and defined for noexcept.
   ~t2wOffloader() noexcept
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

   /// Implementation of the FSM for t2wOffloader.
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

   int updateFPS();
   
   int allocate( const dev::shmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const dev::shmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );
   

   int zero();
   
   int prepareModes();

protected:

  
   
   pcf::IndiProperty m_indiP_gain;
   pcf::IndiProperty m_indiP_leak;
   pcf::IndiProperty m_indiP_actLim;
   
   pcf::IndiProperty m_indiP_zero;
   
   pcf::IndiProperty m_indiP_numModes;

   pcf::IndiProperty m_indiP_offloadToggle;
   
   INDI_NEWCALLBACK_DECL(t2wOffloader, m_indiP_gain);
   INDI_NEWCALLBACK_DECL(t2wOffloader, m_indiP_leak);
   INDI_NEWCALLBACK_DECL(t2wOffloader, m_indiP_actLim);
   
   INDI_NEWCALLBACK_DECL(t2wOffloader, m_indiP_zero);
   
   INDI_NEWCALLBACK_DECL(t2wOffloader, m_indiP_numModes);

   INDI_NEWCALLBACK_DECL(t2wOffloader, m_indiP_offloadToggle);

   pcf::IndiProperty m_indiP_fpsSource;
   INDI_SETCALLBACK_DECL(t2wOffloader, m_indiP_fpsSource);

   pcf::IndiProperty m_indiP_navgSource;
   INDI_SETCALLBACK_DECL(t2wOffloader, m_indiP_navgSource);

   pcf::IndiProperty m_indiP_fps;
};

inline
t2wOffloader::t2wOffloader() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

inline
void t2wOffloader::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   
   config.add("integrator.fpsSource", "", "integrator.fpsSource", argType::Required, "integrator", "fpsSource", false, "string", "Device name for getting fps of the loop.  This device should have *.fps.current.  Default is camwfs");
   config.add("integrator.navgSource", "", "integrator.navgSource", argType::Required, "integrator", "navgSource", false, "string", "Device name for getting navg of tweeter-ave.  This device should have *.fps.current. Default is dmtweeter-avg.");

   config.add("offload.respMPath", "", "offload.respMPath", argType::Required, "offload", "respMPath", false, "string", "The path to the response matrix.");
   config.add("offload.channel", "", "offload.channel", argType::Required, "offload", "channel", false, "string", "The DM channel to offload to.");
   
   config.add("offload.gain", "", "offload.gain", argType::Required, "offload", "gain", false, "float", "The starting offload gain.  Default is 0.1.");
   config.add("offload.leak", "", "offload.leak", argType::Required, "offload", "leak", false, "float", "The starting offload leak.  Default is 0.0.");
   config.add("offload.startupOffloading", "", "offload.startupOffloading", argType::Required, "offload", "startupOffloading", false, "bool", "Flag controlling whether offloading is on at startup.  Default is false.");
   config.add("offload.actLim", "", "offload.actLim", argType::Required, "offload", "actLim", false, "float", "The woofer actuator command limit.  Default is 7.0.");

   config.add("offload.tweeterModes", "", "offload.tweeterModes", argType::Required, "offload", "tweeterModes", false, "string", "File containing the tweeter modes to use for offloading");
   config.add("offload.tweeterMask", "", "offload.tweeterMask", argType::Required, "offload", "tweeterMask", false, "string", "File containing the tweeter mask.");
   config.add("offload.maxModes", "", "offload.maxModes", argType::Required, "offload", "maxModes", false, "string", "Maximum number of modes for modal offloading.");
   config.add("offload.numModes", "", "offload.numModes", argType::Required, "offload", "numModes", false, "string", "Number of modes to offload. 0 means use actuator offloading.");
}

inline
int t2wOffloader::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(_config);
   
   _config(m_fpsSource, "integrator.fpsSource");
   _config(m_navgSource, "integrator.navgSource");

   _config(m_twRespMPath, "offload.respMPath");
   _config(m_dmChannel, "offload.channel");
   _config(m_gain, "offload.gain");
   _config(m_leak, "offload.leak");
   _config(m_actLim, "offload.actLim");
   _config(m_tweeterModeFile, "offload.tweeterModes");
   _config(m_tweeterMaskFile, "offload.tweeterMask");
   _config(m_maxModes, "offload.maxModes");
   _config(m_numModes, "offload.numModes");

   bool startupOffloading = false;
   
   if(_config.isSet("offload.startupOffloading"))
   {
      _config(startupOffloading, "offload.startupOffloading");
   }
   m_offloading = startupOffloading;
   
   return 0;
}

inline
void t2wOffloader::loadConfig()
{
   loadConfigImpl(config);
}

inline
int t2wOffloader::appStartup()
{

   createStandardIndiNumber<float>( m_indiP_gain, "gain", 0, 1, 0, "%0.2f");
   m_indiP_gain["current"] = m_gain;
   m_indiP_gain["target"] = m_gain;
   
   if( registerIndiPropertyNew( m_indiP_gain, INDI_NEWCALLBACK(m_indiP_gain)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   
   createStandardIndiNumber<float>( m_indiP_leak, "leak", 0, 1, 0, "%0.2f");
   m_indiP_leak["current"] = m_leak;
   m_indiP_leak["target"] = m_leak;
   
   if( registerIndiPropertyNew( m_indiP_leak, INDI_NEWCALLBACK(m_indiP_leak)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   createStandardIndiNumber<float>( m_indiP_actLim, "actLim", 0, 8, 0, "%0.2f");
   m_indiP_actLim["current"] = m_actLim;
   m_indiP_actLim["target"] = m_actLim;
   
   if( registerIndiPropertyNew( m_indiP_actLim, INDI_NEWCALLBACK(m_indiP_actLim)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   if(prepareModes() < 0 )
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   
   createStandardIndiRequestSw( m_indiP_zero, "zero", "zero loop");
   if( registerIndiPropertyNew( m_indiP_zero, INDI_NEWCALLBACK(m_indiP_zero)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   createStandardIndiNumber<int>( m_indiP_numModes, "numModes", 0, 97, 0, "%d");
   m_indiP_numModes["current"] = m_numModes;
   m_indiP_numModes["target"] = m_numModes;
   
   if( registerIndiPropertyNew( m_indiP_numModes, INDI_NEWCALLBACK(m_indiP_numModes)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   createStandardIndiToggleSw( m_indiP_offloadToggle, "offload");  
   if( registerIndiPropertyNew( m_indiP_offloadToggle, INDI_NEWCALLBACK(m_indiP_offloadToggle)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   REG_INDI_SETPROP(m_indiP_fpsSource, m_fpsSource, std::string("fps"));
   REG_INDI_SETPROP(m_indiP_navgSource, m_navgSource, std::string("nAverage"));

   createROIndiNumber(m_indiP_fps, "fps");
   m_indiP_fps.add(pcf::IndiElement("current"));
   if( registerIndiPropertyReadOnly( m_indiP_fps ) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   state(stateCodes::OPERATING);
    
   return 0;
}

inline
int t2wOffloader::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if(shmimMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   
   return 0;
}

inline
int t2wOffloader::appShutdown()
{
   shmimMonitorT::appShutdown();
   
   
   return 0;
}

inline
int t2wOffloader::updateFPS()
{
   if(m_navg < 1)
   {
      m_effFPS = 0;
   }
   else m_effFPS = m_fps/m_navg;
   updateIfChanged(m_indiP_fps, "current", m_effFPS);

   std::cerr << "Effective FPS: " << m_effFPS << "\n";

   return 0;
}

inline
int t2wOffloader::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy); //be unused
   
   //std::unique_lock<std::mutex> lock(m_indiMutex);
      
   m_tweeter.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);

   /*mx::fits::fitsFile<float> ff;
   ff.read(m_twRespM, m_twRespMPath);
   
   std::cerr << "Read a " << m_twRespM.rows() << " x " << m_twRespM.cols() << " matrix.\n";
   */
   
   if(m_dmOpened)
   {
      ImageStreamIO_closeIm(&m_dmStream);
   }
   
   m_dmOpened = false;
   m_dmRestart = false; //Set this up front, since we're about to restart.
      
   if( ImageStreamIO_openIm(&m_dmStream, m_dmChannel.c_str()) == 0)
   {
      if(m_dmStream.md[0].sem < 10) 
      {
            ImageStreamIO_closeIm(&m_dmStream);
      }
      else
      {
         m_dmOpened = true;
      }
   }
      
   if(!m_dmOpened)
   {
      log<software_error>({__FILE__, __LINE__, m_dmChannel + " not opened."});
      return -1;
   }
   else
   {
      m_dmWidth = m_dmStream.md->size[0]; 
      m_dmHeight = m_dmStream.md->size[1]; 
   
      m_dmDataType = m_dmStream.md->datatype;
      m_dmTypeSize = ImageStreamIO_typesize(m_dataType);
      
      log<text_log>( "Opened " + m_dmChannel + " " + std::to_string(m_dmWidth) + " x " + std::to_string(m_dmHeight) + " with data type: " + std::to_string(m_dmDataType)); 
   
      m_woofer.resize(m_dmWidth, m_dmHeight);
      m_woofer.setZero();
   }
   
   m_modeAmps.resize(1,  m_tModesOrtho.planes());

   ///\todo size checks here.
   
   //state(stateCodes::OPERATING);
   
   return 0;
}

inline
int t2wOffloader::processImage( void * curr_src, 
                                const dev::shmimT & dummy 
                              )
{
   static_cast<void>(dummy); //be unused
   
   if(!m_offloading) return 0;
   
   if(m_numModes == 0)
   {
      m_wooferDelta = m_twRespM.matrix() * Eigen::Map<Eigen::Matrix<float,-1,-1>>((float *)curr_src,  m_width*m_height,1);
   }
   else
   {
      m_modeAmps = Eigen::Map<Eigen::Matrix<float,-1,-1>>((float *) curr_src, 1, m_width*m_height) * Eigen::Map<Eigen::Matrix<float,-1,-1>>(m_tModesOrtho.data(), m_tModesOrtho.rows()*m_tModesOrtho.cols(), m_tModesOrtho.planes()); 

      m_wooferDelta = m_modeAmps(0,0) * m_wModes.image(0);
      for(int p=1; p < m_numModes && p < m_maxModes; ++p)
      {
         m_wooferDelta += m_modeAmps(0,p)*m_wModes.image(p);
      }

      
   }
   
   while(m_dmStream.md[0].write == 1); //Check if zero() is running
   
   
   m_woofer = m_gain* Eigen::Map<Eigen::Array<float,-1,-1>>( m_wooferDelta.data(), m_dmWidth, m_dmHeight) + (1.0-m_leak)*m_woofer;
      
   for(int ii = 0; ii < m_woofer.rows(); ++ii)
   {
      for(int jj = 0; jj < m_woofer.cols(); ++jj)
      {
         if( fabs(m_woofer(ii,jj)) > m_actLim)
         {
            if(m_woofer(ii,jj) > 0) m_woofer(ii,jj) = m_actLim;
            else m_woofer(ii,jj) = -m_actLim;
         }
      }
   }
   
   m_dmStream.md[0].write = 1;
   
   memcpy(m_dmStream.array.raw, m_woofer.data(),  m_woofer.rows()*m_woofer.cols()*m_typeSize);
   
   m_dmStream.md[0].cnt0++;
   
   m_dmStream.md->write=0;
   ImageStreamIO_sempost(&m_dmStream,-1);
   
   return 0;
}

inline
int t2wOffloader::zero()
{
   
   //Check if processImage is running
   while(m_dmStream.md[0].write == 1);
   
   m_dmStream.md[0].write = 1;
   
   m_woofer.setZero();

   memcpy(m_dmStream.array.raw, m_woofer.data(),  m_woofer.rows()*m_woofer.cols()*m_typeSize);
   
   m_dmStream.md[0].cnt0++;
   
   m_dmStream.md->write=0;
   ImageStreamIO_sempost(&m_dmStream,-1);
   
   log<text_log>("zeroed", logPrio::LOG_NOTICE);
   
   return 0;
      
}

int t2wOffloader::prepareModes()
{
   mx::improc::eigenCube<float> tmodes;

   mx::fits::fitsFile<float> ff;

   ff.read(tmodes, m_tweeterModeFile);
   std::cerr << "Tweeter modes: " << tmodes.rows() << " x " << tmodes.cols() << " x " << tmodes.planes() << "\n";

   ff.read(m_tweeterMask, m_tweeterMaskFile);
   std::cerr << "Tweeter mask: " << m_tweeterMask.rows() << " x " << m_tweeterMask.cols() << "\n";

   ff.read(m_twRespM, m_twRespMPath);
   
   std::cerr << "t2w Response matrix: " << m_twRespM.rows() << " x " << m_twRespM.cols() << " matrix.\n";
   

   for(int p=0; p < tmodes.planes(); ++p)
   {
      tmodes.image(p) *= m_tweeterMask;
      float norm = (tmodes.image(p)).square().sum();
      tmodes.image(p) /= sqrt(norm);
   }

   m_tModesOrtho.resize(tmodes.rows(), tmodes.cols(), m_maxModes);

   for(int p=0;p<m_tModesOrtho.planes();++p)
   {
      m_tModesOrtho.image(p) = tmodes.image(p);
   }

   ff.write("/tmp/tModesOrtho.fits", m_tModesOrtho);

   m_wModes.resize(11,11,m_tModesOrtho.planes());  
   mx::improc::eigenImage<realT> win, wout;

   win.resize(11,11); 
   wout.resize(11,11);

   for(int p=0; p < m_tModesOrtho.planes(); ++p)
   {
      win = m_tModesOrtho.image(p);
      Eigen::Map<Eigen::Matrix<float,-1,-1>>(wout.data(), wout.rows()*wout.cols(),1) = m_twRespM.matrix() * Eigen::Map<Eigen::Matrix<float,-1,-1>>(win.data(), win.rows()*win.cols(),1);
      m_wModes.image(p) = wout;
   }

   ff.write("/tmp/wModes.fits", m_wModes);

   return 0;

}

INDI_NEWCALLBACK_DEFN(t2wOffloader, m_indiP_gain)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_gain.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_gain, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_gain = target;
   
   updateIfChanged(m_indiP_gain, "current", m_gain);
   updateIfChanged(m_indiP_gain, "target", m_gain);
   
   log<text_log>("set gain to " + std::to_string(m_gain), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(t2wOffloader, m_indiP_leak)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_leak.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_leak, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_leak = target;
   
   updateIfChanged(m_indiP_leak, "current", m_leak);
   updateIfChanged(m_indiP_leak, "target", m_leak);
   
   log<text_log>("set leak to " + std::to_string(m_leak), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(t2wOffloader, m_indiP_actLim)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_actLim.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_actLim, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_actLim = target;
   
   updateIfChanged(m_indiP_actLim, "current", m_actLim);
   updateIfChanged(m_indiP_actLim, "target", m_actLim);
   
   log<text_log>("set actuator limit to " + std::to_string(m_actLim), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(t2wOffloader, m_indiP_zero)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_zero.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      return zero();
   
   }
   return 0;
}

INDI_NEWCALLBACK_DEFN(t2wOffloader, m_indiP_numModes)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_numModes.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_numModes, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_numModes = target;
   
   updateIfChanged(m_indiP_numModes, "current", m_numModes);
   updateIfChanged(m_indiP_numModes, "target", m_numModes);
   
   log<text_log>("set number of modes to " + std::to_string(m_numModes), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(t2wOffloader, m_indiP_offloadToggle )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_offloadToggle.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   //switch is toggled to on
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      if(!m_offloading) //not offloading so change
      {
         m_woofer.setZero(); //always zero when offloading starts
         log<text_log>("zeroed", logPrio::LOG_NOTICE);
      
         m_offloading = true;
         log<text_log>("started offloading", logPrio::LOG_NOTICE);
         updateSwitchIfChanged(m_indiP_offloadToggle, "toggle", pcf::IndiElement::On, INDI_BUSY);
      }
      return 0;
   }

   //switch is toggle to off
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      if(m_offloading) //offloading so change it
      {
         m_offloading = false;
         log<text_log>("stopped offloading", logPrio::LOG_NOTICE);
         updateSwitchIfChanged(m_indiP_offloadToggle, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      }
      return 0;
   }
   
   return 0;
}

INDI_SETCALLBACK_DEFN( t2wOffloader, m_indiP_fpsSource )(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_fpsSource.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if( ipRecv.find("current") != true ) //this isn't valie
   {
      return 0;
   }
   
   std::lock_guard<std::mutex> guard(m_indiMutex);

   realT fps = ipRecv["current"].get<float>();
   
   if(fps != m_fps)
   {
      m_fps = fps;
      std::cout << "Got fps: " << m_fps << "\n";   
      updateFPS();
   }

   return 0;
}

INDI_SETCALLBACK_DEFN( t2wOffloader, m_indiP_navgSource )(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_navgSource.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if( ipRecv.find("current") != true ) //this isn't valie
   {
      return 0;
   }
   
   std::lock_guard<std::mutex> guard(m_indiMutex);

   realT navg = ipRecv["current"].get<float>();
   
   if(navg != m_navg)
   {
      m_navg = navg;
      std::cout << "Got navg: " << m_navg << "\n";   
      updateFPS();
   }

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //t2wOffloader_hpp
