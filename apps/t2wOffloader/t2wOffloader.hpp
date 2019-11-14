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

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{


   
/** \defgroup t2wOffloader Tweeter to Woofer Offloading
  * \brief Monitors the averaged tweeter shape, and sends it to the woofer.
  *
  * <a href="../handbook/apps/t2wOffloader.html">Application Documentation</a>
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
   
   std::string m_dmChannel;
   
   float m_gain {0.1};
   float m_leak {0.0};
   
   
   ///@}

   
   
   mx::improc::eigenImage<realT> m_twRespM;
   mx::improc::eigenImage<realT> m_tweeter;
   mx::improc::eigenImage<realT> m_woofer, m_wooferDelta;
   
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

   
   
   
   int allocate( const dev::shmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const dev::shmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );
   

protected:

  
   
   pcf::IndiProperty m_indiP_gain;
   pcf::IndiProperty m_indiP_leak;
   
   pcf::IndiProperty m_indiP_zero;
   
   pcf::IndiProperty m_indiP_offloadToggle;
   
   INDI_NEWCALLBACK_DECL(t2wOffloader, m_indiP_gain);
   INDI_NEWCALLBACK_DECL(t2wOffloader, m_indiP_leak);
   INDI_NEWCALLBACK_DECL(t2wOffloader, m_indiP_zero);
   INDI_NEWCALLBACK_DECL(t2wOffloader, m_indiP_offloadToggle);
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
   
   
   config.add("offload.respMPath", "", "offload.respMPath", argType::Required, "offload", "respMPath", false, "string", "The path to the response matrix.");
   config.add("offload.channel", "", "offload.channel", argType::Required, "offload", "channel", false, "string", "The DM channel to offload to.");
   
   config.add("offload.gain", "", "offload.gain", argType::Required, "offload", "gain", false, "float", "The starting offload gain.  Default is 0.1.");
   config.add("offload.leak", "", "offload.leak", argType::Required, "offload", "leak", false, "float", "The starting offload leak.  Default is 0.0.");
   config.add("offload.startupOffloading", "", "offload.startupOffloading", argType::Required, "offload", "startupOffloading", false, "bool", "Flag controlling whether offloading is on at startup.  Default is false.");
}

inline
int t2wOffloader::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(_config);
   
   _config(m_twRespMPath, "offload.respMPath");
   _config(m_dmChannel, "offload.channel");
   _config(m_gain, "offload.gain");
   _config(m_leak, "offload.leak");
   
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
   
   createStandardIndiNumber<unsigned>( m_indiP_gain, "gain", 0, 1, 0, "%0.2f");
   m_indiP_gain["current"] = m_gain;
   m_indiP_gain["target"] = m_gain;
   
   if( registerIndiPropertyNew( m_indiP_gain, INDI_NEWCALLBACK(m_indiP_gain)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   
   createStandardIndiNumber<unsigned>( m_indiP_leak, "leak", 0, 1, 0, "%0.2f");
   m_indiP_leak["current"] = m_leak;
   m_indiP_leak["target"] = m_leak;
   
   if( registerIndiPropertyNew( m_indiP_leak, INDI_NEWCALLBACK(m_indiP_leak)) < 0)
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
   
   createStandardIndiToggleSw( m_indiP_offloadToggle, "offload");  
   if( registerIndiPropertyNew( m_indiP_offloadToggle, INDI_NEWCALLBACK(m_indiP_offloadToggle)) < 0)
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
int t2wOffloader::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy); //be unused
   
   //std::unique_lock<std::mutex> lock(m_indiMutex);
      
   m_tweeter.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);

   mx::improc::fitsFile<float> ff;
   ff.read(m_twRespM, m_twRespMPath);
   
   std::cerr << "Read a " << m_twRespM.rows() << " x " << m_twRespM.cols() << " matrix.\n";
   
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
   
   m_wooferDelta = m_twRespM.matrix() * Eigen::Map<Matrix<float,-1,-1>>((float *)curr_src,  m_width*m_height,1);
   
   m_woofer = m_gain* Eigen::Map<Array<float,-1,-1>>( m_wooferDelta.data(), m_dmWidth, m_dmHeight) + (1.0-m_leak)*m_woofer;
      
   m_dmStream.md[0].write = 1;
   
   memcpy(m_dmStream.array.raw, m_woofer.data(),  m_woofer.rows()*m_woofer.cols()*m_typeSize);
   
   m_dmStream.md[0].cnt0++;
   
   m_dmStream.md->write=0;
   ImageStreamIO_sempost(&m_dmStream,-1);
   
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

INDI_NEWCALLBACK_DEFN(t2wOffloader, m_indiP_zero)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_zero.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      m_woofer.setZero();
      log<text_log>("zeroed", logPrio::LOG_NOTICE);
   
   }
   return 0;
}

INDI_NEWCALLBACK_DEFN(t2wOffloader, m_indiP_offloadToggle )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_zero.getName())
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

} //namespace app
} //namespace MagAOX

#endif //t2wOffloader_hpp
