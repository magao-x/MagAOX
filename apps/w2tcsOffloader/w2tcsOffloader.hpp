/** \file w2tcsOffloader.hpp
  * \brief The MagAO-X Woofer To Telescope Control System (TCS) offloading manager
  *
  * \ingroup app_files
  */

#ifndef w2tcsOffloader_hpp
#define w2tcsOffloader_hpp

#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{
   
/** \defgroup w2tcsOffloader Woofer to TCS Offloading
  * \brief Monitors the averaged woofer shape, fits Zernikes, and sends it to INDI.
  *
  * <a href="../handbook/operating/software/apps/w2tcsOffloader.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup w2tcsOffloader_files Woofer to TCS Offloading
  * \ingroup w2tcsOffloader
  */

/** MagAO-X application to control offloading the woofer to the TCS.
  *
  * \ingroup w2tcsOffloader
  * 
  */
class w2tcsOffloader : public MagAOXApp<true>, public dev::shmimMonitor<w2tcsOffloader>
{

   //Give the test harness access.
   friend class w2tcsOffloader_test;

   friend class dev::shmimMonitor<w2tcsOffloader>;
   
   //The base shmimMonitor type
   typedef dev::shmimMonitor<w2tcsOffloader> shmimMonitorT;
      
   ///Floating point type in which to do all calculations.
   typedef float realT;
   
protected:

   /** \name Configurable Parameters
     *@{
     */
   
   std::string m_wZModesPath;
   std::string m_wMaskPath;
   std::vector<std::string> m_elNames;
   std::vector<realT> m_zCoeffs;
   float m_gain {0.1};
   int m_nModes {2};
   float m_norm {1.0};
   
   ///@}

   mx::improc::eigenCube<realT> m_wZModes;
   mx::improc::eigenImage<realT> m_woofer;
   mx::improc::eigenImage<realT> m_wMask;
   
public:
   /// Default c'tor.
   w2tcsOffloader();

   /// D'tor, declared and defined for noexcept.
   ~w2tcsOffloader() noexcept
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

   /// Implementation of the FSM for w2tcsOffloader.
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
   pcf::IndiProperty m_indiP_nModes;
   pcf::IndiProperty m_indiP_zCoeffs;
   
   pcf::IndiProperty m_indiP_zero;
   
   INDI_NEWCALLBACK_DECL(w2tcsOffloader, m_indiP_gain);
   INDI_NEWCALLBACK_DECL(w2tcsOffloader, m_indiP_nModes);
   INDI_NEWCALLBACK_DECL(w2tcsOffloader, m_indiP_zero);
   INDI_NEWCALLBACK_DECL(w2tcsOffloader, m_indiP_zCoeffs);
};

inline
w2tcsOffloader::w2tcsOffloader() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

inline
void w2tcsOffloader::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   
   config.add("offload.wZModesPath", "", "offload.wZModesPath", argType::Required, "offload", "wZModesPath", false, "string", "The path to the woofer Zernike modes.");
   config.add("offload.wMaskPath", "", "offload.wMaskPath", argType::Required, "offload", "wMaskPath", false, "string", "Path to the woofer Zernike mode mask.");
   config.add("offload.gain", "", "offload.gain", argType::Required, "offload", "gain", false, "float", "The starting offload gain.  Default is 0.1.");
   config.add("offload.nModes", "", "offload.nModes", argType::Required, "offload", "nModes", false, "int", "Number of modes to offload to the TCS.");
}

inline
int w2tcsOffloader::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(_config);
   
   _config(m_wZModesPath, "offload.wZModesPath");
   _config(m_wMaskPath, "offload.wMaskPath");
   _config(m_gain, "offload.gain");
   _config(m_nModes, "offload.nModes");
   
   return 0;
}

inline
void w2tcsOffloader::loadConfig()
{
   loadConfigImpl(config);
}

inline
int w2tcsOffloader::appStartup(){

   mx::improc::fitsFile<float> ff;
   if(ff.read(m_wZModes, m_wZModesPath) < 0) 
   {
      return log<text_log,-1>("Could not open mode cube file", logPrio::LOG_ERROR);
   }
   
   m_zCoeffs.resize(m_wZModes.planes(), 0);

   if(ff.read(m_wMask, m_wMaskPath) < 0) 
   {
     return log<text_log,-1>("Could not open mode mask file", logPrio::LOG_ERROR);
   }

   m_norm = m_wMask.sum();

   createStandardIndiNumber<unsigned>( m_indiP_gain, "gain", 0, 1, 0, "%0.2f");
   m_indiP_gain["current"] = m_gain;
   m_indiP_gain["target"] = m_gain;
   
   if( registerIndiPropertyNew( m_indiP_gain, INDI_NEWCALLBACK(m_indiP_gain)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }


   createStandardIndiNumber<unsigned>( m_indiP_nModes, "nModes", 1, std::numeric_limits<unsigned>::max(), 1, "%u");
   m_indiP_nModes["current"] = m_nModes;

   if( registerIndiPropertyNew( m_indiP_nModes, INDI_NEWCALLBACK(m_indiP_nModes)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   REG_INDI_NEWPROP(m_indiP_zCoeffs, "zCoeffs", pcf::IndiProperty::Number);


   m_elNames.resize(m_zCoeffs.size());
   for(size_t n=0; n < m_zCoeffs.size(); ++n)
   {
      //std::string el = std::to_string(n);
      m_elNames[n] = mx::ioutils::convertToString<size_t, 2, '0'>(n);
      
      m_indiP_zCoeffs.add( pcf::IndiElement(m_elNames[n]) );
      m_indiP_zCoeffs[m_elNames[n]].set(0);
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
   
   state(stateCodes::OPERATING);
    
   return 0;
}

inline
int w2tcsOffloader::appLogic()
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
int w2tcsOffloader::appShutdown()
{
   shmimMonitorT::appShutdown();
   
   
   return 0;
}

inline
int w2tcsOffloader::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy); //be unused
   
   //std::unique_lock<std::mutex> lock(m_indiMutex);
      
   m_woofer.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);

   //state(stateCodes::OPERATING);
   
   return 0;
}

inline
int w2tcsOffloader::processImage( void * curr_src, 
                                const dev::shmimT & dummy 
                              )
{
   static_cast<void>(dummy); //be unused (what is this?)
   
   // Replace this:
   // project zernikes onto avg image
   // update INDI properties with coeffs

   for(size_t i=0; i < m_zCoeffs.size(); ++i)
   {
    /* update requested nModes and explicitly zero out any
    modes that shouldn't be offloaded (but might have been
    previously set)
    */
    if(i < m_nModes)
    {
       float coeff;
       coeff = ( Eigen::Map<mx::improc::eigenImage<realT>>((float *)curr_src, shmimMonitorT::m_width, shmimMonitorT::m_height) * m_wZModes.image(i) * m_wMask).sum() / m_norm;
       m_indiP_zCoeffs[m_elNames[i]] = m_gain * coeff;
    }
    else
    {
      m_indiP_zCoeffs[m_elNames[i]] = 0.;
    }
   }

   m_indiP_zCoeffs.setState (pcf::IndiProperty::Ok);
   m_indiDriver->sendSetProperty (m_indiP_zCoeffs);


   // loop over something like this
   //z0 = (im * basis.image(0)*mask).sum()/norm;


   return 0;
}

// update this: mode coefficients (maybe they shouldn't be settable. How to handle?)

INDI_NEWCALLBACK_DEFN(w2tcsOffloader, m_indiP_gain)(const pcf::IndiProperty &ipRecv)
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

INDI_NEWCALLBACK_DEFN(w2tcsOffloader, m_indiP_nModes)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_nModes.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   unsigned target;
   
   if( indiTargetUpdate( m_indiP_nModes, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_nModes = target;

   updateIfChanged(m_indiP_nModes, "current", m_nModes);
   updateIfChanged(m_indiP_nModes, "target", m_nModes);
   
   log<text_log>("set nModes to " + std::to_string(m_nModes), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(w2tcsOffloader, m_indiP_zCoeffs)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_zCoeffs.getName())
   {
      for(size_t n=0; n < m_zCoeffs.size(); ++n)
      {
         if(ipRecv.find(m_elNames[n]))
         {
            realT zcoeff = ipRecv[m_elNames[n]].get<realT>();
            m_zCoeffs[n] = zcoeff;
         }
      }
      return 0;
   }
   
   return log<software_error,-1>({__FILE__,__LINE__, "invalid indi property name"});
}

INDI_NEWCALLBACK_DEFN(w2tcsOffloader, m_indiP_zero)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_zero.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   float target;
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      m_woofer.setZero();
      log<text_log>("set zero", logPrio::LOG_NOTICE);
   
   }
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //w2tcsOffloader_hpp
  