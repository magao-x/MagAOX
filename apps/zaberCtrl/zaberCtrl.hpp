/** \file zaberCtrl.hpp
  * \brief The MagAO-X Zaber Stage Controller header file
  *
  * \ingroup zaberCtrl_files
  */

#ifndef zaberCtrl_hpp
#define zaberCtrl_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup zaberCtrl 
  * \brief The MagAO-X application to control a single Zaber stage
  *
  * <a href="..//apps_html/page_module_zaberCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup zaberCtrl_files
  * \ingroup zaberCtrl
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X Zaber Stage Controller
/** Interacts with the stage through a zaberLowLevel process via INDI.
  * 
  * \ingroup zaberCtrl
  */
class zaberCtrl : public MagAOXApp<>
{

   //Give the test harness access.
   friend class zaberCtrl_test;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   std::string m_lowLevelName {"zaberLowLevel"};
   
   std::string m_stageName;
   
   double m_countsPerMillimeter {10078.740157480315};
   
   ///@}

   double m_pos {0};


public:
   /// Default c'tor.
   zaberCtrl();

   /// D'tor, declared and defined for noexcept.
   ~zaberCtrl() noexcept
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

   /// Implementation of the FSM for zaberCtrl.
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

protected:

   //INDI properties for user interaction:   
   pcf::IndiProperty m_indiP_pos;
   
   pcf::IndiProperty m_indiP_home;
   
   //INDI_NEWCALLBACK_DECL(zaberCtrl, m_indiP_pos);
   //INDI_NEWCALLBACK_DECL(zaberCtrl, m_indiP_home);
   
   //INDI properties for interacting with Low-Level
   pcf::IndiProperty m_indiP_stageState;
   
   pcf::IndiProperty m_indiP_stageRawPos;
   
   INDI_SETCALLBACK_DECL(zaberCtrl, m_indiP_stageState);
   INDI_SETCALLBACK_DECL(zaberCtrl, m_indiP_stageRawPos);
   
};

zaberCtrl::zaberCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void zaberCtrl::setupConfig()
{
   config.add("stage.lowLevelName", "", "stage.lowLevelName", argType::Required, "stage", "lowLevelName", false, "string", "The name of the low-level control process actually controlling this stage.  Default is zaberLowLevel.");
   
   config.add("stage.stageName", "", "stage.stageName", argType::Required, "stage", "stageName", false, "string", "the name of this stage in the low-level process INDI properties.  Default is the configuration name.");
   
   config.add("stage.countsPerMillimeter", "", "stage.countsPerMillimeter", argType::Required, "stage", "countsPerMillimeter", false, "float", "The counts per mm calibration of the stage.  Default is 10078.74.");
}


int zaberCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   _config(m_lowLevelName, "stage.lowLevelName");
   
   m_stageName = m_configName;
   _config(m_stageName, "stage.stageName");
   
   _config(m_countsPerMillimeter, "stage.countsPerMillimeter");
   
   REG_INDI_SETPROP(m_indiP_stageState, m_lowLevelName, "curr_state");
   REG_INDI_SETPROP(m_indiP_stageRawPos, m_lowLevelName, std::string("curr_pos"));
   
   return 0;
}

void zaberCtrl::loadConfig()
{
   loadConfigImpl(config);
}

int zaberCtrl::appStartup()
{
   REG_INDI_NEWPROP_NOCB(m_indiP_pos, "position", pcf::IndiProperty::Number);
   m_indiP_pos.add (pcf::IndiElement("current"));
   m_indiP_pos.add (pcf::IndiElement("target"));
   m_indiP_pos.add (pcf::IndiElement("raw"));
//    REG_INDI_NEWPROP(m_indiP_home, "home", pcf::IndiProperty::Text);
//    m_indiP_home.add (pcf::IndiElement("request"));
   
   return 0;
}

int zaberCtrl::appLogic()
{
   //Check low level state.
   // -- if power off, we are power off
   // -- if not ready, we are notconnected
   // -- if ready, then check stage state
   // ---- if idle, then ready 
   // ---- if busy, then operating
   
   //Echo position
   
   return 0;
}

int zaberCtrl::appShutdown()
{
   return 0;
}

INDI_SETCALLBACK_DEFN( zaberCtrl, m_indiP_stageState)(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_stageState.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if( ipRecv.find(m_stageName) != true ) //Just not our stage.
   {
      return 0;
   }

   std::string sstr = ipRecv[m_stageName].get<std::string>();
   
   if(sstr == "READY") state(stateCodes::READY);
   if(sstr == "OPERATING") state(stateCodes::OPERATING);
   

   return 0;
}

INDI_SETCALLBACK_DEFN( zaberCtrl, m_indiP_stageRawPos )(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_stageRawPos.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if( ipRecv.find(m_stageName) != true ) //Just not our stage.
   {
      return 0;
   }

   double rawPos = ipRecv[m_stageName].get<double>();
   
   m_pos = rawPos / m_countsPerMillimeter;
   
   updateIfChanged(m_indiP_pos, "current", m_pos);
   updateIfChanged(m_indiP_pos, "raw", rawPos);
   

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //zaberCtrl_hpp
