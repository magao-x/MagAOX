/** \file xt1121DCDU.hpp
  * \brief The MagAO-X xt1121-based D.C. Distribution Unit controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup xt1121DCDU_files
  */

#ifndef xt1121DCDU_hpp
#define xt1121DCDU_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup xt1121DCDU xt1121-based DC distribution unit
  * \brief Control of MagAO-X D.C. power distrubition via the xt1121 DIO module.
  *
  * <a href="../apps_html/page_module_xt1121DCDU.html">Application Documentation</a>
  *
  * \ingroup apps
  * 
  */

/** \defgroup xt1121DCDU_files xt1121 DCDU Files
  * \ingroup xt1121DCDU
  */

namespace MagAOX
{
namespace app
{

/// MagAO-X application to control D.C. distribution via an xt1121 DIO unit.
/** The device outlets are organized into channels.  See \ref dev::outletController for details of configuring the channels.
  *
  * 
  * \ingroup xt1121DCDU
  */
class xt1121DCDU : public MagAOXApp<>, public dev::outletController<xt1121DCDU>
{

protected:

   std::string m_deviceName; ///< The device address

   std::vector<int> m_channelNumbers; ///< Vector of outlet numbers, used to construct the channel names to monitor as outlets 0-7.
   
   int m_outletStateDelay {5000}; ///< The maximum time to wait for an outlet to change state [msec].


   pcf::IndiProperty ip_ch0;
   pcf::IndiProperty ip_ch1;
   pcf::IndiProperty ip_ch2;
   pcf::IndiProperty ip_ch3;
   pcf::IndiProperty ip_ch4;
   pcf::IndiProperty ip_ch5;
   pcf::IndiProperty ip_ch6;
   pcf::IndiProperty ip_ch7;
   
   
   
public:

   /// Default c'tor.
   xt1121DCDU();

   /// D'tor, declared and defined for noexcept.
   ~xt1121DCDU() noexcept
   {}

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Startup functions
   /** Setsup the INDI vars.
     * Checks if the device was found during loadConfig.
     */
   virtual int appStartup();

   /// Implementation of the FSM for the xt1121 DCDU.
   virtual int appLogic();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();

   /// Update a single outlet state
   /**
     *
     * \returns 0 on success
     * \returns -1 on error
     */
   virtual int updateOutletState( int outletNum /**< [in] the outlet number to update */);
   
   /// Turn on an outlet.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   virtual int turnOutletOn( int outletNum /**< [in] the outlet number to turn on */);

   /// Turn off an outlet.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   virtual int turnOutletOff( int outletNum /**< [in] the outlet number to turn off */);

protected:
   
   ///Helper function to get the xt1121Ctrl channel name for the given channel number.
   /**
     * \returns chXX where XX is 00 to 15, set by chno.
     * \returns empty string if chno is not valid.
     */
   std::string xtChannelName( int chno);
   
   ///Helper function to get a pointer to the right INDI property for an outlet number.
   /**
     * \returns a pointer to one of the INDI properties if outletNum is valid 
     * \returns nullptr if outletNum is not valid.
     */ 
   pcf::IndiProperty * xtChannelProperty( int outletNum /**< [in] the outlet number */);
     
   INDI_SETCALLBACK_DECL(xt1121DCDU, ip_ch0);
   INDI_SETCALLBACK_DECL(xt1121DCDU, ip_ch1);
   INDI_SETCALLBACK_DECL(xt1121DCDU, ip_ch2);
   INDI_SETCALLBACK_DECL(xt1121DCDU, ip_ch3);
   INDI_SETCALLBACK_DECL(xt1121DCDU, ip_ch4);
   INDI_SETCALLBACK_DECL(xt1121DCDU, ip_ch5);
   INDI_SETCALLBACK_DECL(xt1121DCDU, ip_ch6);
   INDI_SETCALLBACK_DECL(xt1121DCDU, ip_ch7);
};

xt1121DCDU::xt1121DCDU() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_firstOne = true;
   m_powerMgtEnabled = true;
  
   setNumberOfOutlets(8);
   
   return;
}

void xt1121DCDU::setupConfig()
{
   config.add("device.name", "", "device.name", argType::Required, "device", "name", false, "string", "The device INDI name.");

   config.add("device.channelNumbers", "", "device.channelNumbers", argType::Required, "device", "channelNumbers", false, "vector<int>", "The channel numbers to use for the outlets, in order.");
   

   dev::outletController<xt1121DCDU>::setupConfig(config);
   
}


void xt1121DCDU::loadConfig()
{
   config(m_deviceName, "device.name");
   
   m_channelNumbers = {0,1,2,3,4,5,6,7};
   config(m_channelNumbers, "device.channelNumbers");
   
   dev::outletController<xt1121DCDU>::loadConfig(config);

}



int xt1121DCDU::appStartup()
{
   if(m_channelNumbers.size() != 8)
   {
      return log<text_log,-1>("Something other than 8 channel numbers specified.", logPrio::LOG_CRITICAL);
   }
   
   REG_INDI_SETPROP(ip_ch0, m_deviceName, xtChannelName(m_channelNumbers[0]));
   REG_INDI_SETPROP(ip_ch1, m_deviceName, xtChannelName(m_channelNumbers[1]));
   REG_INDI_SETPROP(ip_ch2, m_deviceName, xtChannelName(m_channelNumbers[2]));
   REG_INDI_SETPROP(ip_ch3, m_deviceName, xtChannelName(m_channelNumbers[3]));
   REG_INDI_SETPROP(ip_ch4, m_deviceName, xtChannelName(m_channelNumbers[4]));
   REG_INDI_SETPROP(ip_ch5, m_deviceName, xtChannelName(m_channelNumbers[5]));
   REG_INDI_SETPROP(ip_ch6, m_deviceName, xtChannelName(m_channelNumbers[6]));
   REG_INDI_SETPROP(ip_ch7, m_deviceName, xtChannelName(m_channelNumbers[7]));
   
   if(dev::outletController<xt1121DCDU>::setupINDI() < 0)
   {
      return log<text_log,-1>("Error setting up INDI for outlet control.", logPrio::LOG_CRITICAL);
   }

   state(stateCodes::NOTCONNECTED);

   return 0;
}

int xt1121DCDU::appLogic()
{
   if( state() == stateCodes::POWERON )
   {
      state(stateCodes::READY);
   }


   if(state() == stateCodes::READY)
   {
      std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

      int rv = updateOutletStates();

      if(rv < 0) return log<software_error,-1>({__FILE__, __LINE__});

      dev::outletController<xt1121DCDU>::updateINDI();
      
      return 0;
   }

   state(stateCodes::FAILURE);
   log<text_log>("appLogic fell through", logPrio::LOG_CRITICAL);
   return -1;

}

int xt1121DCDU::appShutdown()
{
   //don't bother
   return 0;
}

int xt1121DCDU::updateOutletState( int outletNum )
{
   pcf::IndiProperty * ip = xtChannelProperty(outletNum);
   
   if(ip == nullptr)
   {
      return log<software_error, -1>({__FILE__, __LINE__, "bad outlet number"});
   }
   
   int os = OUTLET_STATE_UNKNOWN;
   if(ip->find("current"))
   {
      if((*ip)["current"].get<int>() == 0) os = OUTLET_STATE_OFF;
      else if((*ip)["current"].get<int>() == 1) os = OUTLET_STATE_ON;
   }
   
   m_outletStates[outletNum] = os;
   
   return 0;
   
}

int xt1121DCDU::turnOutletOn( int outletNum )
{
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before doing anything

   pcf::IndiProperty * ip = xtChannelProperty(outletNum);
   
   if(ip == nullptr)
   {
      return log<software_error, -1>({__FILE__, __LINE__, "bad outlet number"});
   }
      
   return sendNewProperty(*ip, "target", 1);
   
}

int xt1121DCDU::turnOutletOff( int outletNum )
{
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before doing anything

   pcf::IndiProperty * ip = xtChannelProperty(outletNum);
   
   if(ip == nullptr)
   {
      return log<software_error, -1>({__FILE__, __LINE__, "bad outlet number"});
   }
      
   return sendNewProperty(*ip, "target", 0);

}

std::string xt1121DCDU::xtChannelName( int chno)
{
   switch(chno)
   {
      case 0:
         return "ch00";
      case 1:
         return "ch01";
      case 2:
         return "ch02";
      case 3:
         return "ch03";
      case 4:
         return "ch04";
      case 5:
         return "ch05";
      case 6:
         return "ch06";
      case 7:
         return "ch07";
      case 8:
         return "ch08";
      case 9:
         return "ch09";
      case 10:
         return "ch10";
      case 11:
         return "ch11";
      case 12:
         return "ch12";
      case 13:
         return "ch13";
      case 14:
         return "ch14";
      case 15:
         return "ch15";
      case 16:
         return "ch16";
      default:
         return "";
   }
}
pcf::IndiProperty * xt1121DCDU::xtChannelProperty( int outletNum )
{
   switch(outletNum)
   {
      case 0:
         return &ip_ch0;
      case 1:
         return &ip_ch1;
      case 2:
         return &ip_ch2;
      case 3:
         return &ip_ch3;
      case 4:
         return &ip_ch4;
      case 5:
         return &ip_ch5;
      case 6:
         return &ip_ch6;
      case 7:
         return &ip_ch7;
      default:
         return nullptr;
   }
}
   
INDI_SETCALLBACK_DEFN(xt1121DCDU, ip_ch0)(const pcf::IndiProperty &ipRecv)
{
   ip_ch0 = ipRecv;
   
   return updateOutletState(0);
}

INDI_SETCALLBACK_DEFN(xt1121DCDU, ip_ch1)(const pcf::IndiProperty &ipRecv)
{
   ip_ch1 = ipRecv;
   
   return updateOutletState(1);
}

INDI_SETCALLBACK_DEFN(xt1121DCDU, ip_ch2)(const pcf::IndiProperty &ipRecv)
{
   ip_ch2 = ipRecv;
   
   return updateOutletState(2);
}

INDI_SETCALLBACK_DEFN(xt1121DCDU, ip_ch3)(const pcf::IndiProperty &ipRecv)
{
   ip_ch3 = ipRecv;
   
   return updateOutletState(3);
}

INDI_SETCALLBACK_DEFN(xt1121DCDU, ip_ch4)(const pcf::IndiProperty &ipRecv)
{
   ip_ch4 = ipRecv;
   
   return updateOutletState(4);
}

INDI_SETCALLBACK_DEFN(xt1121DCDU, ip_ch5)(const pcf::IndiProperty &ipRecv)
{
   ip_ch5 = ipRecv;
   
   return updateOutletState(5);
}

INDI_SETCALLBACK_DEFN(xt1121DCDU, ip_ch6)(const pcf::IndiProperty &ipRecv)
{
   ip_ch6 = ipRecv;
   
   return updateOutletState(6);
}

INDI_SETCALLBACK_DEFN(xt1121DCDU, ip_ch7)(const pcf::IndiProperty &ipRecv)
{
   ip_ch7 = ipRecv;
   
   return updateOutletState(7);
}

} //namespace app
} //namespace MagAOX

#endif //xt1121DCDU_hpp
