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
  * <a href="../handbook/apps/zaberCtrl.html">Application Documentation</a>
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
class zaberCtrl : public MagAOXApp<>, public dev::stdMotionStage<zaberCtrl>
{

   //Give the test harness access.
   friend class zaberCtrl_test;
   
   friend class dev::stdMotionStage<zaberCtrl>;

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

   /// Start a high-level homing sequence.
   /** 
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int startHoming();

   /// Stop the stage motion immediately.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int stop();
   
   double presetNumber();
   
   /// Move to a new position in mm.
   /**  \todo this actually should move to a preset position by nearest integet..  A moveToMM should be used for the mm command.
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int moveTo( const double & pos /**< [in] The new position in mm*/ );
   
protected:

   //INDI properties for user interaction:   
   pcf::IndiProperty m_indiP_pos;
   pcf::IndiProperty m_indiP_rawpos;
   pcf::IndiProperty m_indiP_temp;
   
   INDI_NEWCALLBACK_DECL(zaberCtrl, m_indiP_pos);
   INDI_NEWCALLBACK_DECL(zaberCtrl, m_indiP_rawpos);
   
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
   m_defaultPositions = false;
   
   return;
}

void zaberCtrl::setupConfig()
{
   config.add("stage.lowLevelName", "", "stage.lowLevelName", argType::Required, "stage", "lowLevelName", false, "string", "The name of the low-level control process actually controlling this stage.  Default is zaberLowLevel.");
   
   config.add("stage.stageName", "", "stage.stageName", argType::Required, "stage", "stageName", false, "string", "the name of this stage in the low-level process INDI properties.  Default is the configuration name.");
   
   config.add("stage.countsPerMillimeter", "", "stage.countsPerMillimeter", argType::Required, "stage", "countsPerMillimeter", false, "float", "The counts per mm calibration of the stage.  Default is 10078.74.");
   
   dev::stdMotionStage<zaberCtrl>::setupConfig(config);
}


int zaberCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   _config(m_lowLevelName, "stage.lowLevelName");
   
   m_stageName = m_configName;
   _config(m_stageName, "stage.stageName");
   
   _config(m_countsPerMillimeter, "stage.countsPerMillimeter");
   
   
   dev::stdMotionStage<zaberCtrl>::loadConfig(_config);
   
   return 0;
}

void zaberCtrl::loadConfig()
{
   loadConfigImpl(config);
}

int zaberCtrl::appStartup()
{

   createStandardIndiNumber<float>( m_indiP_pos, "position", 0.0, std::numeric_limits<float>::max(), 0.0, "%.4f");  // units of mm, stage has ~ 1/10 um of precision
   m_indiP_pos["current"].set(0);
   m_indiP_pos["target"].set(0);
   if( registerIndiPropertyNew( m_indiP_pos, INDI_NEWCALLBACK(m_indiP_pos)) < 0)
   {
      #ifndef ZABERCTRL_TEST_NOLOG
      log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
      
   createStandardIndiNumber<unsigned long>( m_indiP_rawpos, "rawpos", 0.0, std::numeric_limits<unsigned long>::max(), 0.0, "%lu");
   m_indiP_rawpos["current"].set(0);
   m_indiP_rawpos["target"].set(0);
   if( registerIndiPropertyNew( m_indiP_rawpos, INDI_NEWCALLBACK(m_indiP_rawpos)) < 0)
   {
      #ifndef ZABERCTRL_TEST_NOLOG
      log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
      
   createROIndiNumber( m_indiP_temp, "temp", "Stage Temperature [C]");    
   indi::addNumberElement( m_indiP_temp, "current", -50,100, 0, "%0.1f");
   registerIndiPropertyReadOnly(m_indiP_temp);
   
   //Register for the low level status reports
   REG_INDI_SETPROP(m_indiP_stageState, m_lowLevelName, "curr_state");
   REG_INDI_SETPROP(m_indiP_stageMaxRawPos, m_lowLevelName, std::string("max_pos"));
   REG_INDI_SETPROP(m_indiP_stageRawPos, m_lowLevelName, std::string("curr_pos"));
   REG_INDI_SETPROP(m_indiP_stageTgtPos, m_lowLevelName, std::string("tgt_pos"));
   REG_INDI_SETPROP(m_indiP_stageTemp, m_lowLevelName, std::string("temp"));
   
   
   if(m_presetNames.size() != m_presetPositions.size())
   {
      return log<text_log,-1>("must set a position for each preset", logPrio::LOG_CRITICAL);
   }
   
   m_presetNames.insert(m_presetNames.begin(), "none");
   m_presetPositions.insert(m_presetPositions.begin(), -1);
   
   dev::stdMotionStage<zaberCtrl>::appStartup();
   
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
   
   static int last_moving = -1;
   
   bool changed = false;
   if(last_moving != m_moving)
   {
      changed = true;
      last_moving = m_moving;
   }
   
   if(changed)
   {
      if(m_moving)
      {
         m_indiP_pos.setState(INDI_BUSY);
         m_indiP_rawpos.setState(INDI_BUSY);
      }
      else
      {
         m_indiP_pos.setState(INDI_IDLE);
         m_indiP_pos["target"] = m_pos;
         m_indiP_rawpos.setState(INDI_IDLE);
         m_indiP_rawpos["target"] = m_rawPos;
      }
      m_indiDriver->sendSetProperty(m_indiP_pos);
      m_indiDriver->sendSetProperty(m_indiP_rawpos);
   }
   
   int n = presetNumber();
   if(n == -1)
   {
      m_preset = 0;
      m_preset_target = 0;
   }
   else
   {
      m_preset = n;
      m_preset_target = n;
   }

   dev::stdMotionStage<zaberCtrl>::updateINDI();
   

      
   return 0;
}

int zaberCtrl::appShutdown()
{
   return 0;
}

int zaberCtrl::startHoming()
{
   updateSwitchIfChanged(m_indiP_home, "request", pcf::IndiElement::Off, INDI_IDLE);
   
   pcf::IndiProperty indiP_stageHome = pcf::IndiProperty(pcf::IndiProperty::Text);
   indiP_stageHome.setDevice(m_lowLevelName);
   indiP_stageHome.setName("req_home");
   indiP_stageHome.setPerm(pcf::IndiProperty::ReadWrite); 
   indiP_stageHome.setState(pcf::IndiProperty::Idle);
   indiP_stageHome.add(pcf::IndiElement(m_stageName));
   
   if( sendNewProperty(indiP_stageHome, m_stageName, "1") < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   
   return 0;
}

int zaberCtrl::stop()
{
   updateSwitchIfChanged(m_indiP_stop, "request", pcf::IndiElement::Off, INDI_IDLE);
   
   pcf::IndiProperty indiP_stageHalt = pcf::IndiProperty(pcf::IndiProperty::Text);
   indiP_stageHalt.setDevice(m_lowLevelName);
   indiP_stageHalt.setName("req_halt");
   indiP_stageHalt.setPerm(pcf::IndiProperty::ReadWrite); 
   indiP_stageHalt.setState(pcf::IndiProperty::Idle);
   indiP_stageHalt.add(pcf::IndiElement(m_stageName));
   
   if( sendNewProperty(indiP_stageHalt, m_stageName, "1") < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   
   return 0;
}

double zaberCtrl::presetNumber()
{
   for( size_t n=1; n < m_presetPositions.size(); ++n)
   {
      if( fabs(m_pos-m_presetPositions[n]) < 1./m_countsPerMillimeter) return n;
   }
   
   return 0;
}

int zaberCtrl::moveTo( const double & target)
{
   if(target < 0) return 0;
   
   long tgt = (target*m_countsPerMillimeter + 0.5);
   
   pcf::IndiProperty indiP_stageTgtPos = pcf::IndiProperty(pcf::IndiProperty::Number);
   indiP_stageTgtPos.setDevice(m_lowLevelName);
   indiP_stageTgtPos.setName("tgt_pos");
   indiP_stageTgtPos.setPerm(pcf::IndiProperty::ReadWrite); 
   indiP_stageTgtPos.setState(pcf::IndiProperty::Idle);
   indiP_stageTgtPos.add(pcf::IndiElement(m_stageName));
   
   if( sendNewProperty(indiP_stageTgtPos, m_stageName, tgt) < 0 ) return log<software_error,-1>({__FILE__,__LINE__});
   
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
      
   return moveTo(target);
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
   
   if(sstr == "POWEROFF") 
   {
      state(stateCodes::POWEROFF);
      m_moving = 0;
   }
   else if(sstr == "POWERON") 
   {
      state(stateCodes::POWERON);
      m_moving = 0;
   }
   else if(sstr == "NOTHOMED") 
   {
      state(stateCodes::NOTHOMED);
      m_moving = 0;
   }
   else if(sstr == "HOMING") 
   {
      state(stateCodes::HOMING);
      m_moving = 1;
   }
   else if(sstr == "READY") 
   {
      state(stateCodes::READY);
      m_moving = 0;
   }
   else if(sstr == "OPERATING") 
   {
      state(stateCodes::OPERATING);
      m_moving = 1;
   }
   else if(sstr == "SHUTDOWN") 
   {
      state(stateCodes::NOTCONNECTED);
      m_moving = 0;
   }

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

   //updateIfChanged(m_indiP_rawpos, "max", m_maxRawPos);   
   //updateIfChanged(m_indiP_pos, "max", m_maxPos);
   
   

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

   if(m_moving)
   {
      updateIfChanged(m_indiP_rawpos, "current", m_rawPos, INDI_BUSY);   
      updateIfChanged(m_indiP_pos, "current", m_pos, INDI_BUSY);
   }
   else
   {
      updateIfChanged(m_indiP_rawpos, "current", m_rawPos, INDI_IDLE);   
      updateIfChanged(m_indiP_pos, "current", m_pos, INDI_IDLE);
   }
   

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
//       updateIfChanged(m_indiP_rawpos, "target", std::string(""));   
//       updateIfChanged(m_indiP_pos, "target", std::string(""));
      return 0;
   }
   
   m_tgtRawPos = ipRecv[m_stageName].get<double>();
   m_tgtPos = m_tgtRawPos / m_countsPerMillimeter;
   
   if(m_moving)
   {
      updateIfChanged(m_indiP_rawpos, "target", m_tgtRawPos, INDI_BUSY);   
      updateIfChanged(m_indiP_pos, "target", m_tgtPos, INDI_BUSY);
   }
   else
   {
      updateIfChanged(m_indiP_rawpos, "target", m_tgtRawPos, INDI_IDLE);   
      updateIfChanged(m_indiP_pos, "target", m_tgtPos, INDI_IDLE);
   }
   
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
