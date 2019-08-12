/** \file zaberLowLevel.hpp
  * \brief The MagAO-X Low-Level Zaber Controller
  *
  * \ingroup zaberLowLevel_files
  */

#ifndef zaberLowLevel_hpp
#define zaberLowLevel_hpp

#include <iostream>


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

typedef MagAOX::app::MagAOXApp<true> MagAOXAppT; //This needs to be before zaberStage.hpp for logging to work.

#include "zaberUtils.hpp"
#include "zaberStage.hpp"
#include "za_serial.h"

#define ZC_CONNECTED (0)
#define ZC_ERROR (-1)
#define ZC_NOT_CONNECTED (10)

/** \defgroup zaberLowLevel low-level zaber controller
  * \brief The low-level interface to a set of chained Zaber stages
  *
  * <a href="..//apps_html/page_module_zaberLowLevel.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup zaberLowLevel_files zaber low-level files
  * \ingroup zaberLowLevel
  */

namespace MagAOX
{
namespace app
{

class zaberLowLevel : public MagAOXAppT, public tty::usbDevice
{

   //Give the test harness access.
   friend class zaberLowLevel_test;


protected:

   int m_numStages {0};

   z_port m_port {0};

   std::vector<zaberStage> m_stages;
   
   std::unordered_map<int, size_t> m_stageAddress;
   std::unordered_map<std::string, size_t> m_stageSerial;
   std::unordered_map<std::string, size_t> m_stageName;

public:
   /// Default c'tor.
   zaberLowLevel();

   /// D'tor, declared and defined for noexcept.
   ~zaberLowLevel() noexcept
   {}

   virtual void setupConfig();

   virtual void loadConfig();

   int connect();
   
   int loadStages( std:: string & serialRes );

   /// Startup functions
   /** Sets up the INDI vars.
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for zaberLowLevel.
   virtual int appLogic();

   /// Implementation of the on-power-off FSM logic
   virtual int onPowerOff();

   /// Implementation of the while-powered-off FSM
   virtual int whilePowerOff();
   
   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();

protected:
   ///Current state of the stage.  
   pcf::IndiProperty m_indiP_curr_state;
   
   ///Maximum raw position of the stage.  
   pcf::IndiProperty m_indiP_max_pos;
   
   ///Current raw position of the stage.  
   pcf::IndiProperty m_indiP_curr_pos;
   
   ///Current temperature of the stage.  
   pcf::IndiProperty m_indiP_temp;
   
   ///Target raw position of the stage.  
   pcf::IndiProperty m_indiP_tgt_pos;
      
   ///Target relative position of the stage.  
   pcf::IndiProperty m_indiP_tgt_relpos;
   
   ///Command a stage to home.  
   pcf::IndiProperty m_indiP_req_home;
   
   ///Command a stage to safely halt. 
   pcf::IndiProperty m_indiP_req_halt;
   
   ///Command a stage to safely immediately halt. 
   pcf::IndiProperty m_indiP_req_ehalt;
   
public:
   INDI_NEWCALLBACK_DECL(zaberLowLevel, m_indiP_tgt_pos);
   INDI_NEWCALLBACK_DECL(zaberLowLevel, m_indiP_tgt_relpos);
   INDI_NEWCALLBACK_DECL(zaberLowLevel, m_indiP_req_home);
   INDI_NEWCALLBACK_DECL(zaberLowLevel, m_indiP_req_halt);
   INDI_NEWCALLBACK_DECL(zaberLowLevel, m_indiP_req_ehalt);
   
};

zaberLowLevel::zaberLowLevel() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   
   return;
}

void zaberLowLevel::setupConfig()
{
   tty::usbDevice::setupConfig(config);


}

void zaberLowLevel::loadConfig()
{

   this->m_baudRate = B115200; //default for Zaber stages.  Will be overridden by any config setting.

   int rv = tty::usbDevice::loadConfig(config);

   if(rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND) //Ignore error if not plugged in
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }

   std::vector<std::string> sections;

   config.unusedSections(sections);
   
   if(sections.size() == 0)
   {
      log<software_error>({__FILE__, __LINE__, "No stages found"});
      return;
   }
   
   for(size_t n=0; n<sections.size(); ++n)
   {
      if(config.isSetUnused(mx::app::iniFile::makeKey(sections[n], "serial" )))
      {
         m_stages.push_back(zaberStage());
         
         size_t idx = m_stages.size()-1;
         
         m_stages[idx].name(sections[n]);
         
         
         //Get serial number from config.
         std::string tmp = m_stages[idx].serial(); //get default
         config.configUnused( tmp , mx::app::iniFile::makeKey(sections[n], "serial" ) );
         m_stages[idx].serial(tmp);
         
         m_stageName.insert( {m_stages[idx].name(), idx});
         m_stageSerial.insert( {m_stages[idx].serial(), idx});
      }
   }
}

int zaberLowLevel::connect()
{
   if(m_port <= 0)
   {
      int rv = euidCalled();

      if(rv < 0)
      {
         log<software_critical>({__FILE__, __LINE__});
         state(stateCodes::FAILURE);
         return ZC_NOT_CONNECTED;
      }

      int zrv = za_connect(&m_port, m_deviceName.c_str());

      rv = euidReal();
      if(rv < 0)
      {
         log<software_critical>({__FILE__, __LINE__});
         state(stateCodes::FAILURE);
         return ZC_NOT_CONNECTED;
      }


      if(zrv != Z_SUCCESS)
      {
         if(m_port > 0)
         {
            za_disconnect(m_port);
            m_port = 0;
         }
         state(stateCodes::ERROR); //Should not get this here.  Probably means no device.
         return ZC_NOT_CONNECTED; //We aren't connected.
      }
   }

   if(m_port <= 0)
   {
      state(stateCodes::ERROR); //Should not get this here.  Probably means no device.
      return ZC_NOT_CONNECTED; //We aren't connected.
   }
   
   log<text_log>("DRAINING", logPrio::LOG_DEBUG);
      
   int rv = za_drain(m_port);
   
   if(rv != Z_SUCCESS)
   {
      log<software_error>({__FILE__,__LINE__, rv, "error from za_drain"});
      state(stateCodes::ERROR);
      return ZC_ERROR;
   }
   
   char buffer[256];
      
   log<text_log>("Sending: / get system.serial", logPrio::LOG_DEBUG);
   int nwr = za_send(m_port, "/ get system.serial");

   if(nwr == Z_ERROR_SYSTEM_ERROR)
   {
      log<text_log>("Error sending system.serial query to stages", logPrio::LOG_ERROR);
      state(stateCodes::ERROR);
      return ZC_ERROR;
   }

   std::string serialRes;
   while(1)
   {
      int nrd = za_receive(m_port, buffer, sizeof(buffer));
      if(nrd >= 0 )
      {
         buffer[nrd] = '\0';
         log<text_log>(std::string("Received: ")+buffer, logPrio::LOG_DEBUG);
         serialRes += buffer;
      }
      else if( nrd != Z_ERROR_TIMEOUT)
      {
         log<text_log>("Error receiving from stages", logPrio::LOG_ERROR);
         state(stateCodes::ERROR);
         return ZC_ERROR;
      }
      else
      {
         log<text_log>("TIMEOUT", logPrio::LOG_DEBUG);
         break; //Timeout ok.
      }  
   }
      
   return loadStages( serialRes );
}

int zaberLowLevel::loadStages( std::string & serialRes )
{
   std::vector<int> addresses;
   std::vector<std::string> serials;
         
   int rv = parseSystemSerial( addresses, serials, serialRes ); 
   if( rv < 0)
   {
      log<software_error>({__FILE__, __LINE__, errno, rv, "error in parseSystemSerial"});
      state(stateCodes::ERROR);
      return ZC_ERROR;
   }
   else
   {
      log<text_log>("Found " + std::to_string(addresses.size()) + " stages.");
      m_stageAddress.clear(); //We clear this map before re-populating.
      for(size_t n=0;n<addresses.size(); ++n)
      {
         if( m_stageSerial.count( serials[n] ) == 1)
         {
            m_stages[m_stageSerial[serials[n]]].deviceAddress(addresses[n]);
            
            m_stageAddress.insert({ addresses[n], m_stageSerial[serials[n]]});
            log<text_log>("stage @" + std::to_string(addresses[n]) + " with s/n " + serials[n] + " corresponds to " + m_stages[m_stageSerial[serials[n]]].name());
         }
         else
         {
            log<text_log>("Unkown stage @" + std::to_string(addresses[n]) + " with s/n " + serials[n], logPrio::LOG_WARNING);
         }
      }
   }

   return ZC_CONNECTED;
}


int zaberLowLevel::appStartup()
{
   if( state() == stateCodes::UNINITIALIZED )
   {
      log<text_log>( "In appStartup but in state UNINITIALIZED.", logPrio::LOG_CRITICAL );
      return -1;
   }

   if( m_stages.size() == 0 )
   {
      log<text_log>( "No stages configured.", logPrio::LOG_CRITICAL);
      return -1;
   }
   
   REG_INDI_NEWPROP_NOCB(m_indiP_curr_state, "curr_state", pcf::IndiProperty::Text);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_curr_state.add (pcf::IndiElement(m_stages[n].name()));
   }
   
   REG_INDI_NEWPROP_NOCB(m_indiP_max_pos, "max_pos", pcf::IndiProperty::Text);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_max_pos.add (pcf::IndiElement(m_stages[n].name()));
   }
   
   REG_INDI_NEWPROP_NOCB(m_indiP_curr_pos, "curr_pos", pcf::IndiProperty::Number);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_curr_pos.add (pcf::IndiElement(m_stages[n].name()));
   }
   
   REG_INDI_NEWPROP_NOCB(m_indiP_temp, "temp", pcf::IndiProperty::Number);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_temp.add (pcf::IndiElement(m_stages[n].name()));
   }
   
   REG_INDI_NEWPROP(m_indiP_tgt_pos, "tgt_pos", pcf::IndiProperty::Number);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_tgt_pos.add (pcf::IndiElement(m_stages[n].name()));
   }
   
   REG_INDI_NEWPROP(m_indiP_tgt_relpos, "tgt_relpos", pcf::IndiProperty::Number);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_tgt_relpos.add (pcf::IndiElement(m_stages[n].name()));
   }
   
   REG_INDI_NEWPROP(m_indiP_req_home, "req_home", pcf::IndiProperty::Number);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_req_home.add (pcf::IndiElement(m_stages[n].name()));
   }
   
   REG_INDI_NEWPROP(m_indiP_req_halt, "req_halt", pcf::IndiProperty::Number);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_req_halt.add (pcf::IndiElement(m_stages[n].name()));
   }
   
   REG_INDI_NEWPROP(m_indiP_req_ehalt, "req_ehalt", pcf::IndiProperty::Number);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_req_ehalt.add (pcf::IndiElement(m_stages[n].name()));
   }
   
   //Get the USB device if it's in udev
   if(m_deviceName == "") state(stateCodes::NODEVICE);
   else
   {
      state(stateCodes::NOTCONNECTED);
      std::stringstream logs;
      logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as " << m_deviceName;
      log<text_log>(logs.str());
   }


   return 0;
}

int zaberLowLevel::appLogic()
{
   if( state() == stateCodes::INITIALIZED )
   {
      log<text_log>( "In appLogic but in state INITIALIZED.", logPrio::LOG_CRITICAL );
      return -1;
   }

   if( state() == stateCodes::NODEVICE )
   {
      int rv = tty::usbDevice::getDeviceName();
      if(rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES)
      {
         state(stateCodes::FAILURE);
         if(!stateLogged())
         {
            log<software_critical>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
         }
         return -1;
      }

      if(rv == TTY_E_DEVNOTFOUND || rv == TTY_E_NODEVNAMES)
      {
         state(stateCodes::NODEVICE);

         if(!stateLogged())
         {
            std::stringstream logs;
            logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " not found in udev";
            log<text_log>(logs.str());
         }
         return 0;
      }
      else
      {
         state(stateCodes::NOTCONNECTED);
         if(!stateLogged())
         {
            std::stringstream logs;
            logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as " << m_deviceName;
            log<text_log>(logs.str());
         }
      }

   }

   if( state() == stateCodes::POWERON)
   {
      state(stateCodes::NOTCONNECTED);
   }
   
   if( state() == stateCodes::NOTCONNECTED )
   {
      std::lock_guard<std::mutex> guard(m_indiMutex);
      
      int rv = connect();
      
      if( rv == ZC_CONNECTED) 
      {
         state(stateCodes::CONNECTED);

         if(!stateLogged())
         {
            log<text_log>("Connected to stage(s) on " + m_deviceName);
         }
      }
   }

   if( state() == stateCodes::CONNECTED )
   {
      for(size_t i=0; i < m_stages.size();++i)
      {
         std::lock_guard<std::mutex> guard(m_indiMutex); //Inside loop so INDI requests can steal it

         m_stages[i].getMaxPos(m_port);
         updateIfChanged(m_indiP_max_pos, m_stages[i].name(), m_stages[i].maxPos());
      }
      
      state(stateCodes::READY);
      
   }

   if( state() == stateCodes::READY )
   {
      
      //Here we check complete stage state.
      for(size_t i=0; i < m_stages.size();++i)
      {
         std::lock_guard<std::mutex> guard(m_indiMutex); //Inside loop so INDI requests can steal it

         m_stages[i].updatePos(m_port);
         
         updateIfChanged(m_indiP_curr_pos, m_stages[i].name(), m_stages[i].rawPos());
         
         if(m_stages[i].rawPos() == m_stages[i].tgtPos())
         {
            updateIfChanged(m_indiP_tgt_pos, m_stages[i].name(), std::string(""));
         }
         else
         {
            updateIfChanged(m_indiP_tgt_pos, m_stages[i].name(), m_stages[i].tgtPos());
         }
         
         
         if(m_stages[i].deviceStatus() == 'B') 
         {
            if(m_stages[i].homing())
            {
               updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("HOMING"));
            }
            else
            {
               updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("OPERATING"));
            }
         }
         else if(m_stages[i].deviceStatus() == 'I') 
         {
            if(m_stages[i].warnWR())
            {
               updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("NOTHOMED"));
            }
            else
            {
               updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("READY"));
            }
         }
         else updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("NODEVICE"));
         
         m_stages[i].updateTemp(m_port);
         updateIfChanged(m_indiP_temp, m_stages[i].name(), m_stages[i].temp());
         
         m_stages[i].getWarnings(m_port);
            
      }
   }

   if( state() == stateCodes::ERROR )
   {
      int rv = tty::usbDevice::getDeviceName();
      if(rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES)
      {
         state(stateCodes::FAILURE);
         if(!stateLogged())
         {
            log<software_critical>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
         }
         return rv;
      }

      if(rv == TTY_E_DEVNOTFOUND || rv == TTY_E_NODEVNAMES)
      {
         state(stateCodes::NODEVICE);

         if(!stateLogged())
         {
            std::stringstream logs;
            logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " not found in udev";
            log<text_log>(logs.str());
         }
         return 0;
      }

      state(stateCodes::FAILURE);
      if(!stateLogged())
      {
         log<text_log>("Error NOT due to loss of USB connection.  I can't fix it myself.", logPrio::LOG_CRITICAL);
      }
   }




   if( state() == stateCodes::FAILURE )
   {
      return -1;
   }

   return 0;
}

inline
int zaberLowLevel::onPowerOff()
{
   std::lock_guard<std::mutex> lock(m_indiMutex);
   
   for(size_t i=0; i < m_stages.size();++i)
   {

      updateIfChanged(m_indiP_curr_pos, m_stages[i].name(), -1);
      updateIfChanged(m_indiP_tgt_pos, m_stages[i].name(), std::string(""));
      updateIfChanged(m_indiP_tgt_relpos, m_stages[i].name(), std::string(""));
      
      updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("POWEROFF"));
      
         
   }
   return 0;
}

inline
int zaberLowLevel::whilePowerOff()
{

   return 0;
}

inline
int zaberLowLevel::appShutdown()
{
   return 0;
}

INDI_NEWCALLBACK_DEFN(zaberLowLevel, m_indiP_tgt_pos)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_tgt_pos.getName())
   {  
      for(size_t n=0; n < m_stages.size(); ++n)
      {
         if( ipRecv.find(m_stages[n].name()) )
         {
            long tgt = ipRecv[m_stages[n].name()].get<long>();
            if(tgt >= 0)
            {
               std::lock_guard<std::mutex> guard(m_indiMutex);
               std::cerr << "moving " << m_stages[n].name() << " to " << std::to_string(tgt) << "\n";
               
               return m_stages[n].moveAbs(m_port, tgt);
            }
         }
      }
   }
   return 0;
}

INDI_NEWCALLBACK_DEFN(zaberLowLevel, m_indiP_tgt_relpos)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_tgt_relpos.getName())
   {  
      for(size_t n=0; n < m_stages.size(); ++n)
      {
         if( ipRecv.find(m_stages[n].name()) )
         {
            long tgt = ipRecv[m_stages[n].name()].get<long>();
            tgt += m_stages[n].rawPos();
            if(tgt >= 0)
            {
               std::lock_guard<std::mutex> guard(m_indiMutex);
               std::cerr << "moving " << m_stages[n].name() << " to " << std::to_string(tgt) << "\n";
               return m_stages[n].moveAbs(m_port, tgt);
            }
         }
      }
   }
   return 0;
}

INDI_NEWCALLBACK_DEFN(zaberLowLevel, m_indiP_req_home)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_req_home.getName())
   {  
      for(size_t n=0; n < m_stages.size(); ++n)
      {
         if( ipRecv.find(m_stages[n].name()) )
         {
            int tgt = ipRecv[m_stages[n].name()].get<int>();
            if(tgt > 0)
            {
               std::lock_guard<std::mutex> guard(m_indiMutex);
               std::cerr << "homing " << m_stages[n].name() << "\n";
               
               return m_stages[n].home(m_port);
            }
         }
      }
   }
   return 0;
}

INDI_NEWCALLBACK_DEFN(zaberLowLevel, m_indiP_req_halt)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_req_halt.getName())
   {  
      for(size_t n=0; n < m_stages.size(); ++n)
      {
         if( ipRecv.find(m_stages[n].name()) )
         {
            int tgt = ipRecv[m_stages[n].name()].get<int>();
            if(tgt > 0)
            {
               std::lock_guard<std::mutex> guard(m_indiMutex);
               std::cerr << "halting " << m_stages[n].name() << "\n";
               
               return m_stages[n].stop(m_port);
            }
         }
      }
   }
   return 0;
}

INDI_NEWCALLBACK_DEFN(zaberLowLevel, m_indiP_req_ehalt)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_req_ehalt.getName())
   {  
      for(size_t n=0; n < m_stages.size(); ++n)
      {
         if( ipRecv.find(m_stages[n].name()) )
         {
            int tgt = ipRecv[m_stages[n].name()].get<int>();
            if(tgt > 0)
            {
               std::lock_guard<std::mutex> guard(m_indiMutex);
               std::cerr << "e-halting " << m_stages[n].name() << "\n";
               
               return m_stages[n].estop(m_port);
            }
         }
      }
   }
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //zaberLowLevel_hpp
