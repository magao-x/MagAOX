/** \file flipperCtrl.hpp
  * \brief The MagAO-X XXXXXX header file
  *
  * \ingroup flipperCtrl_files
  */

#ifndef flipperCtrl_hpp
#define flipperCtrl_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup flipperCtrl
  * \brief The XXXXXX application to do YYYYYYY
  *
  * <a href="../handbook/apps/XXXXXX.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup flipperCtrl_files
  * \ingroup flipperCtrl
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X xxxxxxxx
/** 
  * \ingroup flipperCtrl
  */
class flipperCtrl : public MagAOXApp<true>, public tty::usbDevice, public dev::ioDevice, public dev::telemeter<flipperCtrl>
{

   //Give the test harness access.
   friend class flipperCtrl_test;

   friend class dev::telemeter<flipperCtrl>;
protected:

   /** \name Configurable Parameters
     *@{
     */
   
   //here add parameters which will be config-able at runtime
   int m_inPos {1};
   int m_outPos {2};
   
   ///@}


   int m_pos {1};
   int m_tgt {0};

public:
   /// Default c'tor.
   flipperCtrl();

   /// D'tor, declared and defined for noexcept.
   ~flipperCtrl() noexcept
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

   /// Implementation of the FSM for flipperCtrl.
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

   
   int getPos();
   
   int moveTo(int pos);
   
   pcf::IndiProperty m_indiP_position;
   
   INDI_NEWCALLBACK_DECL(flipperCtrl, m_indiP_position);

   
   /* Telemetry */
   int checkRecordTimes();
   
   int recordTelem( const telem_stage *);
   
   int recordStage( bool force = false);
   
};

flipperCtrl::flipperCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   return;
}

void flipperCtrl::setupConfig()
{
   tty::usbDevice::setupConfig(config);
   dev::ioDevice::setupConfig(config);
   
   config.add("flipper.reverse", "", "flipper.reverse", argType::Required, "flipper", "reverse", false, "bool", "If true, reverse the positions for in and out.");
   
   dev::telemeter<flipperCtrl>::setupConfig(config);
}

int flipperCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   this->m_baudRate = B115200; //default for MCBL controller.  Will be overridden by any config setting.

   int rv = tty::usbDevice::loadConfig(_config);
   
   if(rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND) //Ignore error if not plugged in
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
      
   dev::ioDevice::loadConfig(_config);
   
   bool rev = false;
   _config(rev, "flipper.reverse");
   
   if(rev)
   {
      m_inPos = 2;
      m_outPos = 1;
   }
   
   dev::telemeter<flipperCtrl>::loadConfig(_config);
   
   return 0;
}

void flipperCtrl::loadConfig()
{
   if(loadConfigImpl(config)<0)
   {
      log<software_critical>({__FILE__, __LINE__});
      m_shutdown = 1;
      return;
   }
}

int flipperCtrl::appStartup()
{
/*
   if(m_deviceName == "") state(stateCodes::NODEVICE);
   else
   {
      state(stateCodes::NOTCONNECTED);
      std::stringstream logs;
      logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as " << m_deviceName;
      log<text_log>(logs.str());
   }*/

   createStandardIndiSelectionSw( m_indiP_position, "position", {"in", "out"});
   
   if( registerIndiPropertyNew( m_indiP_position, INDI_NEWCALLBACK(m_indiP_position)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
      
   
   if(dev::telemeter<flipperCtrl>::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   return 0;
}

int flipperCtrl::appLogic()
{
   if(state() == stateCodes::POWERON)
   {
      state(stateCodes::NODEVICE);
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

   if( state() == stateCodes::NOTCONNECTED )
   {
      elevatedPrivileges ep(this);
      int rv = connect();
      ep.restore();
      
      if(rv < 0)
      {
         int nrv = tty::usbDevice::getDeviceName();
         if(nrv < 0 && nrv != TTY_E_DEVNOTFOUND && nrv != TTY_E_NODEVNAMES)
         {
            state(stateCodes::FAILURE);
            if(!stateLogged()) log<software_critical>({__FILE__, __LINE__, nrv, tty::ttyErrorString(nrv)});
            return -1;
         }

         if(nrv == TTY_E_DEVNOTFOUND || nrv == TTY_E_NODEVNAMES)
         {
            state(stateCodes::NODEVICE);

            if(!stateLogged())
            {
               std::stringstream logs;
               logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " no longer found in udev";
               log<text_log>(logs.str());
            }
            return 0;
         }
      }
      
      state(stateCodes::CONNECTED);
   }
        
   if( state() == stateCodes::CONNECTED )
   {
      std::unique_lock<std::mutex> lock(m_indiMutex);
      getPos();
      m_tgt = m_pos;
      
      state(stateCodes::READY);
   }
   
   if( state() == stateCodes::READY || state() == stateCodes::OPERATING)
   {
      std::unique_lock<std::mutex> lock(m_indiMutex);
      
      getPos();
      
      if(m_pos == m_inPos)
      {
         if(m_pos == m_tgt)
         {
            updateSwitchIfChanged(m_indiP_position, "in", pcf::IndiElement::On, INDI_IDLE);
            updateSwitchIfChanged(m_indiP_position, "out", pcf::IndiElement::Off, INDI_IDLE);
            state(stateCodes::READY);
         }
         else
         {
            updateSwitchIfChanged(m_indiP_position, "in", pcf::IndiElement::On, INDI_BUSY);
            updateSwitchIfChanged(m_indiP_position, "out", pcf::IndiElement::Off, INDI_BUSY);
            state(stateCodes::OPERATING);
         }
      }
      else
      {
         if(m_pos == m_tgt)
         {
            updateSwitchIfChanged(m_indiP_position, "in", pcf::IndiElement::Off, INDI_IDLE);
            updateSwitchIfChanged(m_indiP_position, "out", pcf::IndiElement::On, INDI_IDLE);
            state(stateCodes::READY);
         }
         else
         {
            updateSwitchIfChanged(m_indiP_position, "in", pcf::IndiElement::Off, INDI_BUSY);
            updateSwitchIfChanged(m_indiP_position, "out", pcf::IndiElement::On, INDI_BUSY);
            state(stateCodes::OPERATING);
         }
      }
      
      recordStage();
      
      if(telemeter<flipperCtrl>::appLogic() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         return 0;
      }
    /*  */
      
      //sleep(2);
   }
   
   return 0;
}

int flipperCtrl::appShutdown()
{
   return 0;
}


int flipperCtrl::getPos()
{
   std::string header(6,'\0');
      
   header[0] = 0x80;
   header[1] = 0x04;
   header[2] = 0x00;
   header[3] = 0x00;
   header[4] = 0x50;
   header[5] = 0x01;
      
   tty::ttyWrite( header, m_fileDescrip, m_writeTimeout);
      
   std::string response;
   if(tty::ttyRead(response, 20, m_fileDescrip, m_readTimeout) < 0)
   {
      log<software_error>({__FILE__,__LINE__, "error getting response from flipper"});
   }
   
   if(response[16] == 1)
   {
      m_pos = 1;
   }
   else
   {
      m_pos = 2;
   }
   
   return 0;
}

int flipperCtrl::moveTo(int pos)
{
   std::string header(6,'\0');
      
   header[0] = 0x6A;
   header[1] = 0x04;
   header[2] = 0x00;
   if(pos == 1)
   {
      header[3] = 0x01;
   }
   else if(pos == 2)
   {
      header[3] = 0x02;
   }
   else
   {
      return log<software_error,-1>({__FILE__,__LINE__, "invalid position"});
   }
   header[4] = 0x50;
   header[5] = 0x01;
     
   tty::ttyWrite( header, m_fileDescrip, m_writeTimeout);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(flipperCtrl, m_indiP_position )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_position.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   

   int newpos = 0;
   
   if(ipRecv.find("in"))
   {
      if(ipRecv["in"].getSwitchState() == pcf::IndiElement::On)
      {
         newpos = m_inPos;
      }
   }
   
   if(ipRecv.find("out"))
   {
      if(ipRecv["out"].getSwitchState() == pcf::IndiElement::On)
      {
         if(newpos)
         {
            log<text_log>("can not set position to both in and out", logPrio::LOG_ERROR);
         }
         else newpos = m_outPos;
      }
   }
   
   if(newpos)
   {
      m_tgt = newpos;
      
      std::unique_lock<std::mutex> lock(m_indiMutex);
      
      m_indiP_position.setState (INDI_BUSY);
      m_indiDriver->sendSetProperty (m_indiP_position);
      
      state(stateCodes::OPERATING);
      
      if(moveTo(m_tgt) < 0)
      {
         return log<software_error,-1>({__FILE__, __LINE__});
      }
      
      recordStage();
         
      return 0;
   }
   

   
   return 0;
}


int flipperCtrl::checkRecordTimes()
{
   return telemeter<flipperCtrl>::checkRecordTimes(telem_stage());
}
   
int flipperCtrl::recordTelem( const telem_stage * )
{
   return recordStage(true);
}

inline
int flipperCtrl::recordStage( bool force )
{
   static int last_pos = -1;
   static int last_moving = -1;
   
   int moving = (m_tgt != m_pos);
   
   if(last_pos != m_pos || last_moving != moving || force)
   {
      std::string ps = "in";
      if(m_pos == m_outPos) ps = "out";
      
      telem<telem_stage>({moving, (double) m_pos, ps});
      
      last_pos = m_pos;
      last_moving = moving;
   }
   

   return 0;
}


} //namespace app
} //namespace MagAOX

#endif //flipperCtrl_hpp
