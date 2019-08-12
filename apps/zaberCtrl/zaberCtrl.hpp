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

   double m_maxRawPos {0};
   double m_rawPos {0};
   double m_tgtRawPos {0};
   double m_stageTemp{0};
   
   double m_maxPos {0};
   double m_pos {0};
   double m_tgtPos {0};

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
   pcf::IndiProperty m_indiP_rawpos;
   pcf::IndiProperty m_indiP_temp;
   
   pcf::IndiProperty m_indiP_home;
   
   INDI_NEWCALLBACK_DECL(zaberCtrl, m_indiP_pos);
   INDI_NEWCALLBACK_DECL(zaberCtrl, m_indiP_rawpos);
   INDI_NEWCALLBACK_DECL(zaberCtrl, m_indiP_home);
   
   //INDI properties for interacting with Low-Level
   pcf::IndiProperty m_indiP_stageState;
   pcf::IndiProperty m_indiP_stageMaxRawPos;
   pcf::IndiProperty m_indiP_stageRawPos;
   pcf::IndiProperty m_indiP_stageTgtPos;
   pcf::IndiProperty m_indiP_stageTemp;
   
   
   INDI_SETCALLBACK_DECL(zaberCtrl, m_indiP_stageState);
   INDI_SETCALLBACK_DECL(zaberCtrl, m_indiP_stageMaxRawPos);
   INDI_SETCALLBACK_DECL(zaberCtrl, m_indiP_stageRawPos);
   INDI_SETCALLBACK_DECL(zaberCtrl, m_indiP_stageTgtPos);
   INDI_SETCALLBACK_DECL(zaberCtrl, m_indiP_stageTemp);
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
   
   
   
   return 0;
}

void zaberCtrl::loadConfig()
{
   loadConfigImpl(config);
}

int zaberCtrl::appStartup()
{
   REG_INDI_NEWPROP(m_indiP_pos, "position", pcf::IndiProperty::Number);
   m_indiP_pos.add (pcf::IndiElement("current"));
   m_indiP_pos["current"] = -1;
   m_indiP_pos.add (pcf::IndiElement("target"));
   m_indiP_pos.add (pcf::IndiElement("max"));
   
   REG_INDI_NEWPROP(m_indiP_rawpos, "rawpos", pcf::IndiProperty::Number);
   m_indiP_rawpos.add (pcf::IndiElement("current"));
   m_indiP_rawpos["current"] = -1;
   m_indiP_rawpos.add (pcf::IndiElement("target"));
   m_indiP_rawpos.add (pcf::IndiElement("max"));
   
   REG_INDI_NEWPROP(m_indiP_home, "home", pcf::IndiProperty::Text);
   m_indiP_home.add (pcf::IndiElement("request"));
   
   REG_INDI_NEWPROP_NOCB(m_indiP_temp, "temp", pcf::IndiProperty::Number);
   m_indiP_temp.add (pcf::IndiElement("current"));
   
   REG_INDI_SETPROP(m_indiP_stageState, m_lowLevelName, "curr_state");
   REG_INDI_SETPROP(m_indiP_stageMaxRawPos, m_lowLevelName, std::string("max_pos"));
   REG_INDI_SETPROP(m_indiP_stageRawPos, m_lowLevelName, std::string("curr_pos"));
   REG_INDI_SETPROP(m_indiP_stageTgtPos, m_lowLevelName, std::string("tgt_pos"));
   REG_INDI_SETPROP(m_indiP_stageTemp, m_lowLevelName, std::string("temp"));
   
   return 0;
}

int zaberCtrl::appLogic()
{
   if(state() == stateCodes::INITIALIZED)
   {
      state(stateCodes::NOTCONNECTED);
   }
   
   if(state() == stateCodes::NOTCONNECTED || state() == stateCodes::POWEROFF || state() == stateCodes::POWERON)
   {
      //Here do poweroff update
      return 0;
   }
   

   //Otherwise we don't do anything differently
   
   return 0;
}

int zaberCtrl::appShutdown()
{
   return 0;
}

INDI_NEWCALLBACK_DEFN( zaberCtrl, m_indiP_pos)(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_pos.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   double target =-1;
   if( ipRecv.find("target") ) //Just not our stage.
   {
      target = ipRecv["target"].get<double>();
   }

   if(target < 0)
   {
      if( ipRecv.find("current") ) //Just not our stage.
      {
         target = ipRecv["current"].get<double>();
      }
      
      if(target < 0 )
      {
         return log<text_log,-1>("no valid target position provided", logPrio::LOG_ERROR);
      }
   }
      
   long tgt = (target*m_countsPerMillimeter + 0.5);
   
   pcf::IndiProperty indiP_stageTgtPos = pcf::IndiProperty(pcf::IndiProperty::Text);
   indiP_stageTgtPos.setDevice(m_lowLevelName);
   indiP_stageTgtPos.setName("tgt_pos");
   indiP_stageTgtPos.setPerm(pcf::IndiProperty::ReadWrite); 
   indiP_stageTgtPos.setState(pcf::IndiProperty::Idle);
   indiP_stageTgtPos.add(pcf::IndiElement(m_stageName));
   
   if( sendNewProperty(indiP_stageTgtPos, m_stageName, tgt) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   

   return 0;
}

INDI_NEWCALLBACK_DEFN( zaberCtrl, m_indiP_rawpos)(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_rawpos.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   double target =-1;
   if( ipRecv.find("target") ) //Just not our stage.
   {
      target = ipRecv["target"].get<double>();
   }

   if(target < 0)
   {
      if( ipRecv.find("current") ) //Just not our stage.
      {
         target = ipRecv["current"].get<double>();
      }
      
      if(target < 0 )
      {
         return log<text_log,-1>("no valid target position provided", logPrio::LOG_ERROR);
      }
   }
      
   pcf::IndiProperty indiP_stageTgtPos = pcf::IndiProperty(pcf::IndiProperty::Text);
   indiP_stageTgtPos.setDevice(m_lowLevelName);
   indiP_stageTgtPos.setName("tgt_pos");
   indiP_stageTgtPos.setPerm(pcf::IndiProperty::ReadWrite); 
   indiP_stageTgtPos.setState(pcf::IndiProperty::Idle);
   indiP_stageTgtPos.add(pcf::IndiElement(m_stageName));
   
   if( sendNewProperty(indiP_stageTgtPos, m_stageName, target) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   

   return 0;
}

INDI_NEWCALLBACK_DEFN( zaberCtrl, m_indiP_home)(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_home.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if( !ipRecv.find("request") )
   {
      return 0;
   }
   
   pcf::IndiProperty indiP_stageHome = pcf::IndiProperty(pcf::IndiProperty::Text);
   indiP_stageHome.setDevice(m_lowLevelName);
   indiP_stageHome.setName("req_home");
   indiP_stageHome.setPerm(pcf::IndiProperty::ReadWrite); 
   indiP_stageHome.setState(pcf::IndiProperty::Idle);
   indiP_stageHome.add(pcf::IndiElement(m_stageName));
   
   if( sendNewProperty(indiP_stageHome, m_stageName, "1") < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   
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

   m_indiP_stageState = ipRecv;
   
   std::string sstr = ipRecv[m_stageName].get<std::string>();
   
   if(sstr == "POWEROFF") state(stateCodes::POWEROFF);
   if(sstr == "POWERON") state(stateCodes::POWERON);
   if(sstr == "NOTHOMED") state(stateCodes::NOTHOMED);
   if(sstr == "HOMING") state(stateCodes::HOMING);
   if(sstr == "READY") state(stateCodes::READY);
   if(sstr == "OPERATING") state(stateCodes::OPERATING);
   if(sstr == "SHUTDOWN") state(stateCodes::NOTCONNECTED);
   

   return 0;
}

INDI_SETCALLBACK_DEFN( zaberCtrl, m_indiP_stageMaxRawPos )(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_stageMaxRawPos.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if( ipRecv.find(m_stageName) != true ) //Just not our stage.
   {
      return 0;
   }
   
   m_maxRawPos = ipRecv[m_stageName].get<double>();
   
   m_maxPos = m_maxRawPos / m_countsPerMillimeter;

   updateIfChanged(m_indiP_rawpos, "max", m_maxRawPos);   
   updateIfChanged(m_indiP_pos, "max", m_maxPos);
   
   

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
   
   m_rawPos = ipRecv[m_stageName].get<double>();
   
   m_pos = m_rawPos / m_countsPerMillimeter;

   updateIfChanged(m_indiP_rawpos, "current", m_rawPos);   
   updateIfChanged(m_indiP_pos, "current", m_pos);
   
   

   return 0;
}

INDI_SETCALLBACK_DEFN( zaberCtrl, m_indiP_stageTgtPos )(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_stageTgtPos.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if( ipRecv.find(m_stageName) != true ) //Just not our stage.
   {
      return 0;
   }

   //Test for empty property to see if target reached
   std::string test = ipRecv[m_stageName].get<std::string>();
   
   if(test == "")
   {
      updateIfChanged(m_indiP_rawpos, "target", std::string(""));   
      updateIfChanged(m_indiP_pos, "target", std::string(""));
      return 0;
   }
   
   m_tgtRawPos = ipRecv[m_stageName].get<double>();
   m_tgtPos = m_tgtRawPos / m_countsPerMillimeter;
   
   updateIfChanged(m_indiP_rawpos, "target", m_tgtRawPos);   
   updateIfChanged(m_indiP_pos, "target", m_tgtPos);

   return 0;
}

INDI_SETCALLBACK_DEFN( zaberCtrl, m_indiP_stageTemp )(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_stageTemp.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if( ipRecv.find(m_stageName) != true ) //Just not our stage.
   {
      return 0;
   }
   
   m_stageTemp = ipRecv[m_stageName].get<double>();
   
   updateIfChanged(m_indiP_temp, "current", m_stageTemp);   
   
   

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //zaberCtrl_hpp
