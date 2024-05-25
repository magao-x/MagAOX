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
  * <a href="../handbook/operating/software/apps/zaberLowLevel.html">Application Documentation</a>
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

   std::vector<zaberStage<zaberLowLevel>> m_stages;
   
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
         m_stages.push_back(zaberStage<zaberLowLevel>(this));
         
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
   if(m_port > 0)
   {
      int rv = za_disconnect(m_port);
      if(rv < 0)
      {
         log<text_log>("Error disconnecting from zaber system.", logPrio::LOG_ERROR);
      }
      m_port = 0;
   }
   
   
   if(m_port <= 0)
   {
      
      int zrv;
      
      {//scope for elPriv
         elevatedPrivileges elPriv(this);
         zrv = za_connect(&m_port, m_deviceName.c_str());
      }

      if(zrv != Z_SUCCESS)
      {
         if(m_port > 0)
         {
            za_disconnect(m_port);
            m_port = 0;
         }
         
         if(!stateLogged())
         {
            log<software_error>({__FILE__, __LINE__, "can not connect to zaber stage(s)"});
         }
         
         return ZC_NOT_CONNECTED; //We aren't connected.
      }
   }

   if(m_port <= 0)
   {
      //state(stateCodes::ERROR); //Should not get this here.  Probably means no device.
      log<text_log>("can not connect to zaber stage(s): no port", logPrio::LOG_WARNING);
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
   
   //===== First renumber so they are unique.   
   log<text_log>("Sending: / renumber", logPrio::LOG_DEBUG);
   std::string renum = "/ renumber";
   int nwr = za_send(m_port, renum.c_str(), renum.size());
   
   if(nwr == Z_ERROR_SYSTEM_ERROR)
   {
      log<text_log>("Error sending renumber query to stages", logPrio::LOG_ERROR);
      state(stateCodes::ERROR);
      return ZC_ERROR;
   }
   
   //===== Drain the result
   log<text_log>("DRAINING", logPrio::LOG_DEBUG);
      
   rv = za_drain(m_port);
   
   if(rv != Z_SUCCESS)
   {
      log<software_error>({__FILE__,__LINE__, rv, "error from za_drain"});
      state(stateCodes::ERROR);
      return ZC_ERROR;
   }
   
   //======= Now find the stages
   log<text_log>("Sending: / get system.serial", logPrio::LOG_DEBUG);
   std::string gss = "/ get system.serial";
   nwr = za_send(m_port, gss.c_str(), gss.size() );

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

      for(size_t n=0; n < m_stages.size(); ++n)
      {
         if(m_stages[n].deviceAddress() < 1)
         {
            log<text_log>("stage " + m_stages[n].name() + " with with s/n " + serials[n] + " not found in system.", logPrio::LOG_ERROR);
            state(state(), true);
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
      m_indiP_max_pos[m_stages[n].name()] = -1;
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
   
   /*--> Kill this */
   REG_INDI_NEWPROP(m_indiP_tgt_relpos, "tgt_relpos", pcf::IndiProperty::Number);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_tgt_relpos.add (pcf::IndiElement(m_stages[n].name()));
   }
   
   /*--> Make a switch */
   REG_INDI_NEWPROP(m_indiP_req_home, "req_home", pcf::IndiProperty::Number);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_req_home.add (pcf::IndiElement(m_stages[n].name()));
   }
   
   /*--> Make a switch */
   REG_INDI_NEWPROP(m_indiP_req_halt, "req_halt", pcf::IndiProperty::Number);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_req_halt.add (pcf::IndiElement(m_stages[n].name()));
   }
   
   /*--> Make a switch */
   REG_INDI_NEWPROP(m_indiP_req_ehalt, "req_ehalt", pcf::IndiProperty::Number);
   for(size_t n=0; n< m_stages.size(); ++n)
   {
      m_indiP_req_ehalt.add (pcf::IndiElement(m_stages[n].name()));
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

   if( state() == stateCodes::POWERON)
   {
      state(stateCodes::NODEVICE);
      for(size_t i=0; i < m_stages.size();++i)
      {
         updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("NODEVICE"));
      }         
   }  

   if( state() == stateCodes::NODEVICE )
   {
      int rv = tty::usbDevice::getDeviceName();
      if(rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES)
      {
         if( powerState() != 1 || powerStateTarget() != 1 ) return 0; //means we're powering off

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
         std::stringstream logs;
         logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as " << m_deviceName;
         log<text_log>(logs.str());
         
         state(stateCodes::NOTCONNECTED);
         
         for(size_t i=0; i < m_stages.size();++i)
         {
            if(m_stages[i].deviceAddress() < 1) continue;
            updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("NOTCONNECTED"));
         }

         return 0; //we return to give the stage time to initialize the connection if this is a USB-FTDI power on/plug-in event.
      }

   }

   if( state() == stateCodes::NOTCONNECTED )
   {
      std::lock_guard<std::mutex> guard(m_indiMutex);
      
      int rv = connect();
      
      if( rv == ZC_CONNECTED) 
      {
         state(stateCodes::CONNECTED);
         for(size_t i=0; i < m_stages.size();++i)
         {
            if(m_stages[i].deviceAddress() < 1) continue;
            updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("CONNECTED"));
         }

         if(!stateLogged())
         {
            log<text_log>("Connected to stage(s) on " + m_deviceName);
         }
      }
      else if(rv == ZC_NOT_CONNECTED)
      {
         return 0;
      }
      else
      {
         
      }
   }

   if( state() == stateCodes::CONNECTED )
   {
      for(size_t i=0; i < m_stages.size();++i)
      {
         if(m_stages[i].deviceAddress() < 1) 
         {
            updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("NODEVICE"));
            continue; //Skip configured but not found stage
         }
         std::lock_guard<std::mutex> guard(m_indiMutex); //Inside loop so INDI requests can steal it

         m_stages[i].getMaxPos(m_port);
         std::cerr << i << " " << m_stages[i].name() << " " <<  m_stages[i].maxPos() << "\n";
         updateIfChanged(m_indiP_max_pos, m_stages[i].name(), m_stages[i].maxPos());

         //Get warnings so first pass through has correct state for home/not-homed
         if(m_stages[i].getWarnings(m_port) < 0)
         {
            if( powerState() != 1 || powerStateTarget() != 1 ) return 0; //means we're powering off
            log<software_error>({__FILE__, __LINE__});
            state(stateCodes::ERROR);
            return 0;
         }
         
      }
      state(stateCodes::READY);

      return 0;
   }

   if( state() == stateCodes::READY )
   {
      
      //Here we check complete stage state.
      for(size_t i=0; i < m_stages.size();++i)
      {
         if(m_stages[i].deviceAddress() < 1) continue; //Skip configured but not found stage

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
            if(m_stages[i].homing())
            {
               std::cerr << __FILE__ << " " << __LINE__ << "\n";
               return 0;
            }
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
         
         if(m_stages[i].getWarnings(m_port) < 0)
         {
            if( powerState() != 1 || powerStateTarget() != 1 ) return 0; //means we're powering off
            log<software_error>({__FILE__, __LINE__});
            state(stateCodes::ERROR);
            return 0;
         }
            
      }
   }

   if( state() == stateCodes::ERROR )
   {
      int rv = tty::usbDevice::getDeviceName();
      if(rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES)
      {
         if( powerState() != 1 || powerStateTarget() != 1 ) return 0; //means we're powering off
         state(stateCodes::FAILURE);
         for(size_t i=0; i < m_stages.size();++i)
         {
            updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("FAILURE"));
         }
         if(!stateLogged())
         {
            log<software_critical>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
         }
         return rv;
      }

      if(rv == TTY_E_DEVNOTFOUND || rv == TTY_E_NODEVNAMES)
      {
         if( powerState() != 1 || powerStateTarget() != 1 ) return 0; //means we're powering off
         state(stateCodes::NODEVICE);
         for(size_t i=0; i < m_stages.size();++i)
         {
            updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("NODEVICE"));
         }

         if(!stateLogged())
         {
            std::stringstream logs;
            logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " not found in udev";
            log<text_log>(logs.str());
         }
         return 0;
      }

      if( powerState() != 1 || powerStateTarget() != 1 ) return 0; //means we're powering off
      state(stateCodes::FAILURE);
      for(size_t i=0; i < m_stages.size();++i)
      {
         updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("FAILURE"));
      }

      log<software_critical>({__FILE__, __LINE__});
      log<text_log>("Error NOT due to loss of USB connection.  I can't fix it myself.", logPrio::LOG_CRITICAL);
   }



   if( powerState() != 1 || powerStateTarget() != 1 ) return 0; //means we're powering off

   if( state() == stateCodes::FAILURE )
   {
      return -1;
   }

   return 0;
}

inline
int zaberLowLevel::onPowerOff()
{
   int rv = za_disconnect(m_port);
   if(rv < 0)
   {
      log<text_log>("Error disconnecting from zaber system.", logPrio::LOG_ERROR);
   }
   
   
   m_port = 0;
   
   std::lock_guard<std::mutex> lock(m_indiMutex);
   
   for(size_t i=0; i < m_stages.size();++i)
   {
      updateIfChanged(m_indiP_tgt_pos, m_stages[i].name(), std::string(""));
      updateIfChanged(m_indiP_tgt_relpos, m_stages[i].name(), std::string(""));
      updateIfChanged(m_indiP_temp, m_stages[i].name(), std::string(""));
      
      m_stages[i].onPowerOff();
      
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
   for(size_t i=0; i < m_stages.size();++i)
   {
      if(m_stages[i].deviceAddress() < 1) continue;
      updateIfChanged(m_indiP_curr_state, m_stages[i].name(), std::string("NODEVICE"));
   }

   return 0;
}

INDI_NEWCALLBACK_DEFN(zaberLowLevel, m_indiP_tgt_pos)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_tgt_pos, ipRecv);

    for(size_t n=0; n < m_stages.size(); ++n)
    {
       if( ipRecv.find(m_stages[n].name()) )
       {
          long tgt = ipRecv[m_stages[n].name()].get<long>();
          if(tgt >= 0)
          {
             if(m_stages[n].deviceAddress() < 1)
             {
                return log<software_error,-1>({__FILE__, __LINE__, "stage " + m_stages[n].name() + " with with s/n " + m_stages[n].serial() + " not found in system."});
             }

             std::lock_guard<std::mutex> guard(m_indiMutex);
             updateIfChanged(m_indiP_curr_state, m_stages[n].name(), std::string("OPERATING"));
             return m_stages[n].moveAbs(m_port, tgt);
          }
       }
    }
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(zaberLowLevel, m_indiP_tgt_relpos)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_tgt_relpos, ipRecv);

    for(size_t n=0; n < m_stages.size(); ++n)
    {
       if( ipRecv.find(m_stages[n].name()) )
       {
          long tgt = ipRecv[m_stages[n].name()].get<long>();
          tgt += m_stages[n].rawPos();
          if(tgt >= 0)
          {
             if(m_stages[n].deviceAddress() < 1)
             {
                return log<software_error,-1>({__FILE__, __LINE__, "stage " + m_stages[n].name() + " with with s/n " + m_stages[n].serial() + " not found in system."});
             }
             std::lock_guard<std::mutex> guard(m_indiMutex);
    
             updateIfChanged(m_indiP_curr_state, m_stages[n].name(), std::string("OPERATING"));
             return m_stages[n].moveAbs(m_port, tgt);
          }
       }
    }
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(zaberLowLevel, m_indiP_req_home)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_req_home, ipRecv);

    for(size_t n=0; n < m_stages.size(); ++n)
    {
       if( ipRecv.find(m_stages[n].name()) )
       {
          int tgt = ipRecv[m_stages[n].name()].get<int>();
          if(tgt > 0)
          {
             if(m_stages[n].deviceAddress() < 1)
             {
                return log<software_error,-1>({__FILE__, __LINE__, "stage " + m_stages[n].name() + " with with s/n " + m_stages[n].serial() + " not found in system."});
             }
             std::lock_guard<std::mutex> guard(m_indiMutex);
             return m_stages[n].home(m_port);
          }
       }
    }
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(zaberLowLevel, m_indiP_req_halt)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_req_halt, ipRecv);


    for(size_t n=0; n < m_stages.size(); ++n)
    {
        if( ipRecv.find(m_stages[n].name()) )
        {
           int tgt = ipRecv[m_stages[n].name()].get<int>();
           if(tgt > 0)
           {
              if(m_stages[n].deviceAddress() < 1)
              {
                 return log<software_error,-1>({__FILE__, __LINE__, "stage " + m_stages[n].name() + " with with s/n " + m_stages[n].serial() + " not found in system."});
              }
 
              std::lock_guard<std::mutex> guard(m_indiMutex);
              return m_stages[n].stop(m_port);
           }
        }
    }
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(zaberLowLevel, m_indiP_req_ehalt)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_req_ehalt, ipRecv);

    for(size_t n=0; n < m_stages.size(); ++n)
    {
        if( ipRecv.find(m_stages[n].name()) )
        {
            int tgt = ipRecv[m_stages[n].name()].get<int>();
            if(tgt > 0)
            {
                if(m_stages[n].deviceAddress() < 1)
                {
                    return log<software_error,-1>({__FILE__, __LINE__, "stage " + m_stages[n].name() + " with with s/n " + m_stages[n].serial() + " not found in system."});
                }
  
                std::lock_guard<std::mutex> guard(m_indiMutex);
                return m_stages[n].estop(m_port);
            }
        }
    }
    
    return 0;
}

} //namespace app
} //namespace MagAOX

#endif //zaberLowLevel_hpp
