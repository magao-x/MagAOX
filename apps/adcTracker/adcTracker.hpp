/** \file adcTracker.hpp
  * \brief The MagAO-X ADC Tracker header file
  *
  * \ingroup adcTracker_files
  */

#ifndef adcTracker_hpp
#define adcTracker_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <mx/gslInterpolation.hpp>
#include <mx/ioutils/readColumns.hpp>

/** \defgroup adcTracker
  * \brief The MagAO-X application to track sky rotation with the atmospheric dispersion corrector.
  *
  * <a href="../handbook/operating/software/apps/adcTracker.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup adcTracker_files
  * \ingroup adcTracker
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X ADC Tracker
/** 
  * \ingroup adcTracker
  */
class adcTracker : public MagAOXApp<true>
{

   //Give the test harness access.
   friend class adcTracker_test;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   //here add parameters which will be config-able at runtime
   std::string m_lookupFile {"adc_lookup_table.txt"}; ///< The name of the file, in the calib directory, containing the adc lookup table.  Default is 'adc_lookup_table.txt'.
   
   
   float m_adc1zero {0}; ///< The starting point for ADC 1. Default is 0.

   int m_adc1lupsign {1}; ///< The sign to apply to the lookup table value for ADC 1

   float m_adc2zero {0}; ///< The starting point for ADC 2. Default is 0.
   
   int m_adc2lupsign {1}; ///< The sign to apply to the lookup table value for ADC 2
   
   float m_deltaAngle {0}; ///< The offset angle to apply to the looked-up values, applied to both.  Default is 0.
   
   float m_adc1delta {0}; ///< The offset angle to apply to the looked-up value for ADC 1, applied in addition to deltaAngle.  Default is 0.
   
   float m_adc2delta {0}; ///< The offset angle to apply to the looked-up value for ADC 2, applied in addition to deltaAngle.  Default is 0.
   
   float m_minZD {5.1}; ///< "The minimum zenith distance at which to interpolate and move the ADCs.  Default is 0.
   
   
   std::string m_adc1DevName {"stageadc1"}; ///< The device name of the ADC 1 stage.  Default is 'stageadc1'
   std::string m_adc2DevName {"stageadc2"}; ///< The device name of the ADC 2 stage.  Default is 'stageadc2'
   
   std::string m_tcsDevName {"tcsi"}; ///< The device name of the TCS Interface providing 'teldata.zd'.  Default is 'tcsi'
   
   float m_updateInterval {10};
   
   ///@}


   std::vector<double> m_lupZD;
   std::vector<double> m_lupADC1;
   std::vector<double> m_lupADC2;

   mx::gslInterpolator<double> m_terpADC1;
   mx::gslInterpolator<double> m_terpADC2;
   
   bool m_tracking {false};
   
   float m_zd {0};
   
public:
   /// Default c'tor.
   adcTracker();

   /// D'tor, declared and defined for noexcept.
   ~adcTracker() noexcept
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

   /// Implementation of the FSM for adcTracker.
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
   
   pcf::IndiProperty m_indiP_deltaAngle;
   pcf::IndiProperty m_indiP_deltaADC1;
   pcf::IndiProperty m_indiP_deltaADC2;
   
   pcf::IndiProperty m_indiP_minZD;
   
   
   pcf::IndiProperty m_indiP_teldata;
   

   pcf::IndiProperty m_indiP_adc1pos;
   pcf::IndiProperty m_indiP_adc2pos;
   
public:
   INDI_NEWCALLBACK_DECL(adcTracker, m_indiP_tracking);
   
   INDI_NEWCALLBACK_DECL(adcTracker, m_indiP_deltaAngle);
   INDI_NEWCALLBACK_DECL(adcTracker, m_indiP_deltaADC1);
   INDI_NEWCALLBACK_DECL(adcTracker, m_indiP_deltaADC2);
   
   INDI_NEWCALLBACK_DECL(adcTracker, m_indiP_minZD);
   
   INDI_SETCALLBACK_DECL(adcTracker, m_indiP_teldata);
   
   
   
   ///@}
};

adcTracker::adcTracker() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void adcTracker::setupConfig()
{
   config.add("adcs.lookupFile", "", "adcs.lookupFile", argType::Required, "adcs", "lookupFile", false, "string", "The name of the file, in the calib directory, containing the adc lookup table.  Default is 'adc_lookup_table.txt'.");
   
   config.add("adcs.adc1zero", "", "adcs.adc1zero", argType::Required, "adcs", "adc1zero", false, "float", "The starting point for ADC 1. Default is 0.");
   
   config.add("adcs.adc1lupsign", "", "adcs.adc1lupsign", argType::Required, "adcs", "adc1lupsign", false, "int", "The sign to apply for the LUP values for ADC 1. Default is +1.");
   
   config.add("adcs.adc2zero", "", "adcs.adc2zero", argType::Required, "adcs", "adc2zero", false, "float", "The starting point for ADC 2. Default is 0.");
   
   config.add("adcs.adc2lupsign", "", "adcs.adc2lupsign", argType::Required, "adcs", "adc2lupsign", false, "int", "The sign to apply for the LUP values for ADC 2. Default is +1.");
   
   config.add("adcs.deltaAngle", "", "adcs.deltaAngle", argType::Required, "adcs", "deltaAngle", false, "float", "The offset angle to apply to the looked-up values, applied to both.  Default is 0.");
   
   config.add("adcs.adc1delta", "", "adcs.adc1delta", argType::Required, "adcs", "adc1delta", false, "float", "The offset angle to apply to the looked-up value for ADC 1, applied in addition to deltaAngle.  Default is 0.");
   
   config.add("adcs.adc2delta", "", "adcs.adc2delta", argType::Required, "adcs", "adc2delta", false, "float", "The offset angle to apply to the looked-up value for ADC 2, applied in addition to deltaAngle.  Default is 0.");
   
   config.add("adcs.minZD", "", "adcs.minZD", argType::Required, "adcs", "minZD", false, "float", "The minimum zenith distance at which to interpolate and move the ADCs.  Default is 5.1");
   
   config.add("adcs.adc1DevName", "", "adcs.adc1devName", argType::Required, "adcs", "adc1DevName", false, "string", "The device name of the ADC 1 stage.  Default is 'stageadc1'");
   
   config.add("adcs.adc2DevName", "", "adcs.adc2devName", argType::Required, "adcs", "adc2DevName", false, "string", "The device name of the ADC 2 stage.  Default is 'stageadc2'");
   
   config.add("tcs.devName", "", "tcs.devName", argType::Required, "tcs", "devName", false, "string", "The device name of the TCS Interface providing 'teldata.zd'.  Default is 'tcsi'");

   config.add("tracking.updateInterval", "", "tracking.updateInterval", argType::Required, "tracking", "updateInterval", false, "float", "The interval at which to update positions, in seconds.  Default is 10 secs.");
}

int adcTracker::loadConfigImpl( mx::app::appConfigurator & _config )
{
   _config(m_lookupFile, "adcs.lookupFile");
   _config(m_adc1zero, "adcs.adc1zero");
   _config(m_adc1lupsign, "adcs.adc1lupsign");
   _config(m_adc2zero, "adcs.adc2zero");
   _config(m_adc2lupsign, "adcs.adc2lupsign");
   _config(m_deltaAngle, "adcs.deltaAngle");
   _config(m_adc1delta, "adcs.adc1delta");
   _config(m_adc2delta, "adcs.adc2delta");
   _config(m_minZD, "adcs.minZD");
   _config(m_adc1DevName, "adcs.adc1DevName");
   _config(m_adc2DevName, "adcs.adc2DevName");
   
   _config(m_tcsDevName, "tcs.devName");
   
   _config(m_updateInterval, "tracking.updateInterval");
   
   return 0;
}

void adcTracker::loadConfig()
{
   loadConfigImpl(config);
}

int adcTracker::appStartup()
{
   
   std::string luppath = m_calibDir + "/" + m_lookupFile;
   
   std::cerr << "Reading " << luppath << "\n";
   
   if(mx::ioutils::readColumns<','>(luppath, m_lupZD, m_lupADC1, m_lupADC2) < 0)
   {
      log<software_critical>({__FILE__,__LINE__, "error reading lookup table from " + luppath});
      return -1;
   }
   
   if(m_lupZD.size() != m_lupADC1.size() || m_lupZD.size()!= m_lupADC2.size())
   {
      log<software_critical>({__FILE__,__LINE__, "inconsistent sizes in " + luppath});
      return -1;
   }
   
   log<text_log>("Read " + std::to_string(m_lupZD.size()) + " points from " + m_lookupFile);
   
   m_terpADC1.setup(gsl_interp_linear, m_lupZD, m_lupADC1);
   m_terpADC2.setup(gsl_interp_linear, m_lupZD, m_lupADC2);
   
   createStandardIndiToggleSw( m_indiP_tracking, "tracking");
   registerIndiPropertyNew( m_indiP_tracking, INDI_NEWCALLBACK(m_indiP_tracking));
   
   createStandardIndiNumber<float>( m_indiP_deltaAngle, "deltaAngle", 0.0, 180.0, 0, "%0.2f");
   m_indiP_deltaAngle["target"].set(m_deltaAngle);
   m_indiP_deltaAngle["current"].set(m_deltaAngle);
   registerIndiPropertyNew( m_indiP_deltaAngle, INDI_NEWCALLBACK(m_indiP_deltaAngle));
   
   createStandardIndiNumber<float>( m_indiP_deltaADC1, "deltaADC1", 0.0, 180.0, 0, "%0.2f");
   m_indiP_deltaADC1["target"].set(m_adc1delta);
   m_indiP_deltaADC1["current"].set(m_adc1delta);
   registerIndiPropertyNew( m_indiP_deltaADC1, INDI_NEWCALLBACK(m_indiP_deltaADC1));
   
   createStandardIndiNumber<float>( m_indiP_deltaADC2, "deltaADC2", 0.0, 180.0, 0, "%0.2f");
   m_indiP_deltaADC2["target"].set(m_adc2delta);
   m_indiP_deltaADC2["current"].set(m_adc2delta);
   registerIndiPropertyNew( m_indiP_deltaADC2, INDI_NEWCALLBACK(m_indiP_deltaADC2));
   
   createStandardIndiNumber<float>( m_indiP_minZD, "minZD", 0.0, 90.0, 0, "%0.2f");
   m_indiP_minZD["target"].set(m_minZD);
   m_indiP_minZD["current"].set(m_minZD);
   registerIndiPropertyNew( m_indiP_minZD, INDI_NEWCALLBACK(m_indiP_minZD));
   
   REG_INDI_SETPROP(m_indiP_teldata, m_tcsDevName, "teldata");
   
   m_indiP_adc1pos = pcf::IndiProperty(pcf::IndiProperty::Number);
   m_indiP_adc1pos.setDevice(m_adc1DevName);
   m_indiP_adc1pos.setName("position");
   m_indiP_adc1pos.add(pcf::IndiElement("target"));
      
   m_indiP_adc2pos = pcf::IndiProperty(pcf::IndiProperty::Number);
   m_indiP_adc2pos.setDevice(m_adc2DevName);
   m_indiP_adc2pos.setName("position");
   m_indiP_adc2pos.add(pcf::IndiElement("target"));
   
   state(stateCodes::READY);
   
   return 0;
}

int adcTracker::appLogic()
{
   
   static double lastupdate = 0;
   
   if(m_tracking && mx::get_curr_time() - lastupdate > m_updateInterval)
   {
      float dadc1 = 0.0;
      float dadc2 = 0.0;
      
      if(m_zd >= m_minZD)
      {
         dadc1 = fabs(m_terpADC1(m_zd)); 
         dadc2 = fabs(m_terpADC2(m_zd));
      }
      
      float adc1 = m_adc1zero + m_adc1lupsign*(dadc1 + m_adc1delta + m_deltaAngle);
      float adc2 = m_adc2zero + m_adc2lupsign*(dadc2 + m_adc2delta + m_deltaAngle);
      
      std::cerr << "Sending adcs to: " << adc1 << " " << adc2 << "\n";
      
      
      m_indiP_adc1pos["target"] = adc1;
      sendNewProperty (m_indiP_adc1pos); 
      
      m_indiP_adc2pos["target"] = adc2;
      sendNewProperty (m_indiP_adc2pos); 
      
      lastupdate = mx::get_curr_time();
   }
   else if(!m_tracking) lastupdate = 0;
      
   return 0;
}

int adcTracker::appShutdown()
{
   return 0;
}

INDI_NEWCALLBACK_DEFN(adcTracker, m_indiP_tracking)(const pcf::IndiProperty &ipRecv)
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

INDI_NEWCALLBACK_DEFN(adcTracker, m_indiP_deltaAngle)(const pcf::IndiProperty &ipRecv)
{
   float target;
   
   if( indiTargetUpdate( m_indiP_deltaAngle, target, ipRecv) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_deltaAngle = target;
   
   std::lock_guard<std::mutex> guard(m_indiMutex);
   
   updateIfChanged(m_indiP_deltaAngle, "current", m_deltaAngle);
   
   log<text_log>("set deltaAngle to " + std::to_string(m_deltaAngle));
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(adcTracker, m_indiP_deltaADC1)(const pcf::IndiProperty &ipRecv)
{
   float target;
   
   if( indiTargetUpdate( m_indiP_deltaADC1, target, ipRecv) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_adc1delta = target;
   
   std::lock_guard<std::mutex> guard(m_indiMutex);
   
   updateIfChanged(m_indiP_deltaADC1, "current", m_adc1delta);
   
   log<text_log>("set deltaADC1 to " + std::to_string(m_adc1delta));
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(adcTracker, m_indiP_deltaADC2)(const pcf::IndiProperty &ipRecv)
{
   float target;
   
   if( indiTargetUpdate( m_indiP_deltaADC2, target, ipRecv) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_adc2delta = target;
   
   std::lock_guard<std::mutex> guard(m_indiMutex);
   
   updateIfChanged(m_indiP_deltaADC2, "current", m_adc2delta);
   
   log<text_log>("set deltaADC2 to " + std::to_string(m_adc2delta));
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(adcTracker, m_indiP_minZD)(const pcf::IndiProperty &ipRecv)
{
   float target;
   
   if( indiTargetUpdate( m_indiP_minZD, target, ipRecv) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_minZD = target;
   
   std::lock_guard<std::mutex> guard(m_indiMutex);
   
   updateIfChanged(m_indiP_minZD, "current", m_minZD);
   
   log<text_log>("set minZD to " + std::to_string(m_minZD));
   
   return 0;
}

INDI_SETCALLBACK_DEFN(adcTracker, m_indiP_teldata)(const pcf::IndiProperty &ipRecv)
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

#endif //adcTracker_hpp

