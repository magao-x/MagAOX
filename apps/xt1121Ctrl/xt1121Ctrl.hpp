/** \file xt1121Ctrl.hpp
  * \brief The MagAO-X Acromag XT 1121digital I/O controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup xt1121Ctrl_files
  */

#ifndef xt1121Ctrl_hpp
#define xt1121Ctrl_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "xtChannels.hpp"

namespace MagAOX
{
namespace app
{


/** \defgroup xt1121Ctrl Acromag xt1121Controller
  * \brief Control of an Acromag xt1121digital I/O module
  *
  *  <a href="../apps_html/page_module_xt1121Ctrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup xt1121Ctrl_files Acromag xt1121Controller Files
  * \ingroup xt1121Ctrl
  */

/** MagAO-X application to control an Acromag xt1121digital i/o module
  *
  * \ingroup xt1121Ctrl
  * 
  */
class xt1121Ctrl : public MagAOXApp<>, public xt1121Channels
{

protected:

   /** \name configurable parameters 
     *@{
     */ 

   std::string m_address; ///< The I.P. address of the device
   
   uint16_t m_port {502}; ///< The port to use.  Default is 502 for modbus.
   
   unsigned long m_powerOnWait {2}; ///< Time in sec to wait for device to boot after power on.
   
   ///@}
   
   int m_powerOnCounter {0}; ///< Counts numer of loops after power on, implements delay for camera bootup.
    
   modbus * m_mb {nullptr}; ///< The modbus protocol communication object
   
public:

   ///Default c'tor
   xt1121Ctrl();

   ///Destructor
   ~xt1121Ctrl() noexcept;

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Startup functions
   /** Sets up the INDI vars.
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for the Siglent SDG
   virtual int appLogic();

   /// Implementation of the on-power-off FSM logic
   virtual int onPowerOff();

   /// Implementation of the while-powered-off FSM
   virtual int whilePowerOff();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();
   
   /// Get the current state of the outlets.
   /**
     * \returns 0 on success
     * \returns -1 on error 
     */
   int getState();
         
   //INDI:
protected:
   //declare our properties
   pcf::IndiProperty m_indiP_ch00;
   pcf::IndiProperty m_indiP_ch01;
   pcf::IndiProperty m_indiP_ch02;
   pcf::IndiProperty m_indiP_ch03;
   pcf::IndiProperty m_indiP_ch04;
   pcf::IndiProperty m_indiP_ch05;
   pcf::IndiProperty m_indiP_ch06;
   pcf::IndiProperty m_indiP_ch07;
   pcf::IndiProperty m_indiP_ch08;
   pcf::IndiProperty m_indiP_ch09;
   pcf::IndiProperty m_indiP_ch10;
   pcf::IndiProperty m_indiP_ch11;
   pcf::IndiProperty m_indiP_ch12;
   pcf::IndiProperty m_indiP_ch13;
   pcf::IndiProperty m_indiP_ch14;
   pcf::IndiProperty m_indiP_ch15;

public:
   
   /// Callback worker to actually set or clear a channel and send it to the device 
   /** Contains the target/current logic, and calls the xtChannels::setRegisters
     * function, and then the modbus write_registers.
     * 
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int channelSetCallback( size_t chNo,                     ///< [in] The channel number to set                  
                           pcf::IndiProperty & ipToSet,     ///< [in] The corresponding local INDI property
                           const pcf::IndiProperty & ipRecv ///< [in] The received INDI property
                         );
   
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch00);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch01);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch02);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch03);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch04);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch05);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch06);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch07);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch08);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch09);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch10);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch11);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch12);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch13);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch14);
   INDI_NEWCALLBACK_DECL(xt1121Ctrl, m_indiP_ch15);
   


};

inline
xt1121Ctrl::xt1121Ctrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   
   return;
}

inline
xt1121Ctrl::~xt1121Ctrl() noexcept
{
   if(m_mb)
   {
      delete m_mb;
   };

   return;
}

inline
void xt1121Ctrl::setupConfig()
{
   config.add("device.address", "", "device.address", argType::Required, "device", "address", true, "string", "The device I.P. address.");
   config.add("device.port", "", "device.port", argType::Required, "device", "port", true, "int", "The device port.  Default is 502.");
   config.add("device.powerOnWait", "", "device.powerOnWait", argType::Required, "device", "powerOnWait", false, "int", "Time after power-on to begin attempting connections [sec].  Default is 2 sec.");
   config.add("device.inputOnly", "", "device.inputOnly", argType::Required, "device", "inputOnly", false, "vector<int>", "List of channels which are input-only.");

}


///\todo mxlib loadConfig needs to return int to propagate errors!

inline
void xt1121Ctrl::loadConfig()
{
   config(m_address, "device.address");
   config(m_port, "device.port");
   config(m_powerOnWait, "device.powerOnWait");
  
   std::vector<int> ino;
   config(ino, "device.inputOnly");
   
   for(size_t i=0; i< ino.size();++i) 
   {
      if(setInputOnly(ino[i]) != 0)
      {
         log<text_log>("Error setting channel " + std::to_string(i) + " to input only.", logPrio::LOG_ERROR);
      }
   }
}



inline
int xt1121Ctrl::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP(m_indiP_ch00, "ch00", pcf::IndiProperty::Number);
   m_indiP_ch00.add (pcf::IndiElement("current"));
   m_indiP_ch00["current"].set(-1);
   m_indiP_ch00.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch01, "ch01", pcf::IndiProperty::Number);
   m_indiP_ch01.add (pcf::IndiElement("current"));
   m_indiP_ch01["current"].set(-1);
   m_indiP_ch01.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch02, "ch02", pcf::IndiProperty::Number);
   m_indiP_ch02.add (pcf::IndiElement("current"));
   m_indiP_ch02["current"].set(-1);
   m_indiP_ch02.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch03, "ch03", pcf::IndiProperty::Number);
   m_indiP_ch03.add (pcf::IndiElement("current"));
   m_indiP_ch03["current"].set(-1);
   m_indiP_ch03.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch04, "ch04", pcf::IndiProperty::Number);
   m_indiP_ch04.add (pcf::IndiElement("current"));
   m_indiP_ch04["current"].set(-1);
   m_indiP_ch04.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch05, "ch05", pcf::IndiProperty::Number);
   m_indiP_ch05.add (pcf::IndiElement("current"));
   m_indiP_ch05["current"].set(-1);
   m_indiP_ch05.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch06, "ch06", pcf::IndiProperty::Number);
   m_indiP_ch06.add (pcf::IndiElement("current"));
   m_indiP_ch06["current"].set(-1);
   m_indiP_ch06.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch07, "ch07", pcf::IndiProperty::Number);
   m_indiP_ch07.add (pcf::IndiElement("current"));
   m_indiP_ch07["current"].set(-1);
   m_indiP_ch07.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch08, "ch08", pcf::IndiProperty::Number);
   m_indiP_ch08.add (pcf::IndiElement("current"));
   m_indiP_ch08["current"].set(-1);
   m_indiP_ch08.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch09, "ch09", pcf::IndiProperty::Number);
   m_indiP_ch09.add (pcf::IndiElement("current"));
   m_indiP_ch09["current"].set(-1);
   m_indiP_ch09.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch10, "ch10", pcf::IndiProperty::Number);
   m_indiP_ch10.add (pcf::IndiElement("current"));
   m_indiP_ch10["current"].set(-1);
   m_indiP_ch10.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch11, "ch11", pcf::IndiProperty::Number);
   m_indiP_ch11.add (pcf::IndiElement("current"));
   m_indiP_ch11["current"].set(-1);
   m_indiP_ch11.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch12, "ch12", pcf::IndiProperty::Number);
   m_indiP_ch12.add (pcf::IndiElement("current"));
   m_indiP_ch12["current"].set(-1);
   m_indiP_ch12.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch13, "ch13", pcf::IndiProperty::Number);
   m_indiP_ch13.add (pcf::IndiElement("current"));
   m_indiP_ch13["current"].set(-1);
   m_indiP_ch13.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch14, "ch14", pcf::IndiProperty::Number);
   m_indiP_ch14.add (pcf::IndiElement("current"));
   m_indiP_ch14["current"].set(-1);
   m_indiP_ch14.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_ch15, "ch15", pcf::IndiProperty::Number);
   m_indiP_ch15.add (pcf::IndiElement("current"));
   m_indiP_ch15["current"].set(-1);
   m_indiP_ch15.add (pcf::IndiElement("target"));

   
   return 0;

}



inline
int xt1121Ctrl::appLogic()
{
   if( state() == stateCodes::POWERON )
   {
      if(m_powerOnCounter*m_loopPause > ((double) m_powerOnWait)*1e9)
      {
         state(stateCodes::NOTCONNECTED);
         m_powerOnCounter = 0;
      }
      else
      {
         ++m_powerOnCounter;
         return 0;
      }
   }

   if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::ERROR)
   {
      std::string response;

      //Might have gotten here because of a power off.
      if(m_powerState == 0) return 0;
      
      m_mb = new(std::nothrow) modbus(m_address, m_port);
   
      if(m_mb == nullptr)
      {
         return log<software_critical, -1>({__FILE__, __LINE__, "allocation failure"});
      }
   
      m_mb->modbus_set_slave_id(1);
   
      if( m_mb->modbus_connect() == false)
      {
         if(!stateLogged())
         {
            log<text_log>("connect failed at " + m_address + ":" + std::to_string(m_port));
         }
         delete m_mb;
         m_mb = nullptr;
         return 0;
      }
   
      state(stateCodes::CONNECTED);
      log<text_log>("connected to " + m_address + ":" + std::to_string(m_port));
   }

   if( state() == stateCodes::CONNECTED )
   {
      //Get a lock
      std::unique_lock<std::mutex> lock(m_indiMutex);
      
      if( getState() == 0 )
      {
         state(stateCodes::READY);
         return 0;
      }
      else
      {
         state(stateCodes::ERROR);
         return log<software_error,0>({__FILE__,__LINE__});
      }
   }

   if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
   {
      //Get a lock if we can
      std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

      //but don't wait for it, just go back around.
      if(!lock.owns_lock()) return 0;
      
      if(getState() < 0)
      {
         if(m_powerState == 0) return 0;
         
         state(stateCodes::ERROR);
         return 0;
      }

     return 0;
   }

   //Fall through check?

   return 0;

}

inline
int xt1121Ctrl::onPowerOff()
{
   m_powerOnCounter = 0;
   
   std::lock_guard<std::mutex> lock(m_indiMutex);
   
   updateIfChanged(m_indiP_ch00, "current", -1);
   updateIfChanged(m_indiP_ch00, "target", -1);
   
   updateIfChanged(m_indiP_ch01, "current", -1);
   updateIfChanged(m_indiP_ch01, "target", -1);
   
   updateIfChanged(m_indiP_ch02, "current", -1);
   updateIfChanged(m_indiP_ch02, "target", -1);
   
   updateIfChanged(m_indiP_ch03, "current", -1);
   updateIfChanged(m_indiP_ch03, "target", -1);
   
   updateIfChanged(m_indiP_ch04, "current", -1);
   updateIfChanged(m_indiP_ch04, "target", -1);
   
   updateIfChanged(m_indiP_ch05, "current", -1);
   updateIfChanged(m_indiP_ch05, "target", -1);
   
   updateIfChanged(m_indiP_ch06, "current", -1);
   updateIfChanged(m_indiP_ch06, "target", -1);
   
   updateIfChanged(m_indiP_ch07, "current", -1);
   updateIfChanged(m_indiP_ch07, "target", -1);
   
   updateIfChanged(m_indiP_ch08, "current", -1);
   updateIfChanged(m_indiP_ch08, "target", -1);
   
   updateIfChanged(m_indiP_ch09, "current", -1);
   updateIfChanged(m_indiP_ch09, "target", -1);
   
   updateIfChanged(m_indiP_ch10, "current", -1);
   updateIfChanged(m_indiP_ch10, "target", -1);
   
   updateIfChanged(m_indiP_ch11, "current", -1);
   updateIfChanged(m_indiP_ch11, "target", -1);
   
   updateIfChanged(m_indiP_ch12, "current", -1);
   updateIfChanged(m_indiP_ch12, "target", -1);
   
   updateIfChanged(m_indiP_ch13, "current", -1);
   updateIfChanged(m_indiP_ch13, "target", -1);
   
   updateIfChanged(m_indiP_ch14, "current", -1);
   updateIfChanged(m_indiP_ch14, "target", -1);
   
   updateIfChanged(m_indiP_ch15, "current", -1);
   updateIfChanged(m_indiP_ch15, "target", -1);
   
   
   return 0;
}

inline
int xt1121Ctrl::whilePowerOff()
{
   return 0;
}

inline
int xt1121Ctrl::appShutdown()
{
   if(m_mb) m_mb->modbus_close();
   return 0;
}



inline
int xt1121Ctrl::getState()
{
   uint16_t input_regs[numRegisters];
   
   try 
   {
      ///\todo this hangs if power goes off during call
      m_mb->modbus_read_input_registers(0,numRegisters,input_regs);
   }
   catch(std::exception & e)
   {
      if(m_powerState == 0) return 0; //due to power off
      
      return log<software_error,-1>({__FILE__, __LINE__, std::string("Exception caught: ") + e.what()});   
   }
   
   if( readRegisters(input_regs) !=0 )
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   
   updateIfChanged(m_indiP_ch00, "current", channel(0));
   updateIfChanged(m_indiP_ch01, "current", channel(1));
   updateIfChanged(m_indiP_ch02, "current", channel(2));
   updateIfChanged(m_indiP_ch03, "current", channel(3));
   updateIfChanged(m_indiP_ch04, "current", channel(4));
   updateIfChanged(m_indiP_ch05, "current", channel(5));
   updateIfChanged(m_indiP_ch06, "current", channel(6));
   updateIfChanged(m_indiP_ch07, "current", channel(7));
   updateIfChanged(m_indiP_ch08, "current", channel(8));
   updateIfChanged(m_indiP_ch09, "current", channel(9));
   updateIfChanged(m_indiP_ch10, "current", channel(10));
   updateIfChanged(m_indiP_ch11, "current", channel(11));
   updateIfChanged(m_indiP_ch12, "current", channel(12));
   updateIfChanged(m_indiP_ch13, "current", channel(13));
   updateIfChanged(m_indiP_ch14, "current", channel(14));
   updateIfChanged(m_indiP_ch15, "current", channel(15));
   
   return 0;

   
}


int xt1121Ctrl::channelSetCallback( size_t chNo,
                                    pcf::IndiProperty & ipToSet,
                                    const pcf::IndiProperty & ipRecv
                                  )
{
   int current = -1, target = -1;

   if(ipRecv.find("current"))
   {
      current = ipRecv["current"].get<unsigned>();
   }

   if(ipRecv.find("target"))
   {
      target = ipRecv["target"].get<unsigned>();
   }
   
   if(target == -1) target = current;
   
   if(target < 0) return 0;
   
   //Lock the mutex, waiting if necessary
   std::unique_lock<std::mutex> lock(m_indiMutex);

   if(target == 0) clearChannel(chNo);
   else setChannel(chNo);
   
   target = channel(chNo); //This checks for inputOnly
   
   updateIfChanged(ipToSet, "target", target);
   
   uint16_t input_regs[numRegisters];
   
   if( setRegisters(input_regs) !=0 )
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   
   try 
   {
      m_mb->modbus_write_registers(0,numRegisters,input_regs);
   }
   catch(std::exception & e)
   {
      if(m_powerState == 0) return 0; //due to power off
      
      return log<software_error,-1>({__FILE__, __LINE__, std::string("Exception caught: ") + e.what()});   
   }
   
   
   log<text_log>("Set channel " + std::to_string(chNo) + " to " + std::to_string(target));
   
   return 0;
  
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch00)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch00.getName())
   {      
      return channelSetCallback(0, m_indiP_ch00, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch01)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch01.getName())
   {      
      return channelSetCallback(1, m_indiP_ch01, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch02)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch02.getName())
   {      
      return channelSetCallback(2, m_indiP_ch02, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch03)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch03.getName())
   {      
      return channelSetCallback(3, m_indiP_ch03, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch04)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch04.getName())
   {      
      return channelSetCallback(4, m_indiP_ch04, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch05)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch05.getName())
   {      
      return channelSetCallback(5, m_indiP_ch05, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch06)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch06.getName())
   {      
      return channelSetCallback(6, m_indiP_ch06, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch07)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch07.getName())
   {    
      return channelSetCallback(7, m_indiP_ch07, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch08)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch08.getName())
   {      
      return channelSetCallback(8, m_indiP_ch08, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch09)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch09.getName())
   {      
      return channelSetCallback(9, m_indiP_ch09, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch10)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch10.getName())
   {      
      return channelSetCallback(10, m_indiP_ch10, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch11)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch11.getName())
   {      
      return channelSetCallback(11, m_indiP_ch11, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch12)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch12.getName())
   {      
      return channelSetCallback(12, m_indiP_ch12, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch13)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch13.getName())
   {      
      return channelSetCallback(13, m_indiP_ch13, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch14)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch14.getName())
   {      
      return channelSetCallback(14, m_indiP_ch14, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(xt1121Ctrl, m_indiP_ch15)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ch15.getName())
   {      
      return channelSetCallback(15, m_indiP_ch15, ipRecv);
   }
   return -1;
}


}//namespace app
} //namespace MagAOX
#endif
