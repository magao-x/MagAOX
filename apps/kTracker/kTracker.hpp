/** \file kTracker.hpp
  * \brief The MagAO-X K-mirror rotation tracker header file
  *
  * \ingroup kTracker_files
  */

#ifndef kTracker_hpp
#define kTracker_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <mx/gslInterpolation.hpp>
#include <mx/ioutils/readColumns.hpp>

/** \defgroup kTracker
  * \brief The MagAO-X application to track pupil rotation with the k-mirror.
  *
  * <a href="../handbook/operating/software/apps/kTracker.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup kTracker_files
  * \ingroup kTracker
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X ADC Tracker
/** 
  * \ingroup kTracker
  */
class kTracker : public MagAOXApp<true>
{

   //Give the test harness access.
   friend class kTracker_test;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   
   float m_zero {0}; ///< The starting point for the K-mirorr at zd=0.

   int m_sign {1}; ///< The sign to apply to the zd to rotate the k-mirror

   std::string m_devName {"stagek"}; ///< The device name of the K-mirror stage.  Default is 'stagek'
   std::string m_tcsDevName {"tcsi"}; ///< The device name of the TCS Interface providing 'teldata.zd'.  Default is 'tcsi'
   
   float m_updateInterval {10};
   
   ///@}

   bool m_tracking {false}; ///< The interval at which to update positions, in seconds.  Default is 10 secs.
   
   float m_zd {0};
   
public:
   /// Default c'tor.
   kTracker();

   /// D'tor, declared and defined for noexcept.
   ~kTracker() noexcept
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

   /// Implementation of the FSM for kTracker.
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


   /** @name INDI
     *
     * @{
     */
protected:
   
   pcf::IndiProperty m_indiP_tracking;
   
   pcf::IndiProperty m_indiP_teldata;
   

   pcf::IndiProperty m_indiP_kpos;
   
public:
   INDI_NEWCALLBACK_DECL(kTracker, m_indiP_tracking);
   
   
   INDI_SETCALLBACK_DECL(kTracker, m_indiP_teldata);
   
   
   
   ///@}
};

kTracker::kTracker() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void kTracker::setupConfig()
{
   config.add("k.zero", "", "k.zero", argType::Required, "k", "zero", false, "float", "The k-mirror zero position.  Default is -40.0.");
   
   config.add("k.sign", "", "k.sign", argType::Required, "k", "sign", false, "int", "The k-mirror rotation sign. Default is +1.");
   
   
   
   config.add("k.devName", "", "k.devName", argType::Required, "k", "devName", false, "string", "The device name of the k-mirrorstage.  Default is 'stagek'");
   
   config.add("tcs.devName", "", "tcs.devName", argType::Required, "tcs", "devName", false, "string", "The device name of the TCS Interface providing 'teldata.zd'.  Default is 'tcsi'");
   
   config.add("tracking.updateInterval", "", "tracking.updateInterval", argType::Required, "tracking", "updateInterval", false, "float", "The interval at which to update positions, in seconds.  Default is 10 secs.");
}

int kTracker::loadConfigImpl( mx::app::appConfigurator & _config )
{
   _config(m_zero, "k.zero");
   _config(m_sign, "k.sign");
   _config(m_devName, "k.devName");
   
   _config(m_tcsDevName, "tcs.devName");
   
   _config(m_updateInterval, "tracking.updateInterval");

   return 0;
}

void kTracker::loadConfig()
{
   loadConfigImpl(config);
}

int kTracker::appStartup()
{
   
   
   createStandardIndiToggleSw( m_indiP_tracking, "tracking");
   registerIndiPropertyNew( m_indiP_tracking, INDI_NEWCALLBACK(m_indiP_tracking));
   
   
   REG_INDI_SETPROP(m_indiP_teldata, m_tcsDevName, "teldata");
   
   m_indiP_kpos = pcf::IndiProperty(pcf::IndiProperty::Number);
   m_indiP_kpos.setDevice(m_devName);
   m_indiP_kpos.setName("position");
   m_indiP_kpos.add(pcf::IndiElement("target"));
      
   state(stateCodes::READY);
   
   return 0;
}

int kTracker::appLogic()
{
   
   static double lastupdate = 0;
   
   if(m_tracking && mx::get_curr_time() - lastupdate > m_updateInterval)
   {
      float k = m_zero + m_sign*0.5*m_zd;
      
      std::cerr << "Sending k-mirror to: " << k << "\n";
      
      m_indiP_kpos["target"] = k;
      sendNewProperty (m_indiP_kpos); 
      
      lastupdate = mx::get_curr_time();
      
      
   }
   else if(!m_tracking) lastupdate = 0;
      
   return 0;
}

int kTracker::appShutdown()
{
   return 0;
}

INDI_NEWCALLBACK_DEFN(kTracker, m_indiP_tracking)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_tracking.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   if(!ipRecv.find("toggle")) return 0;
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      updateSwitchIfChanged(m_indiP_tracking, "toggle", pcf::IndiElement::On, INDI_BUSY);
      
      m_tracking = true;
      
      log<text_log>("started ADC rotation tracking");
   }
   else
   {
      updateSwitchIfChanged(m_indiP_tracking, "toggle", pcf::IndiElement::Off, INDI_IDLE);
      
      m_tracking = false;
      
      log<text_log>("stopped ADC rotation tracking");
   }
   
   return 0;
}


INDI_SETCALLBACK_DEFN(kTracker, m_indiP_teldata)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_teldata.getName())
   {
      log<software_error>({__FILE__,__LINE__,"wrong INDI property received"});
      
      return -1;
   }
   
   if(!ipRecv.find("zd")) return 0;
   
   m_zd = ipRecv["zd"].get<float>();
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //kTracker_hpp

