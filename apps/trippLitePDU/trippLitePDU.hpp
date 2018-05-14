
#ifndef trippLitePDU_hpp
#define trippLitePDU_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "magaox_git_version.h"

namespace MagAOX
{
namespace app 
{

/** MagAO-X application to control a Tripp Lite PDU
  *
  * \todo need config for timeouts
  * \todo handle timeouts gracefully -- maybe go to error, flush, disconnect, reconnect, etc.
  * \todo need username and secure password handling
  * \todo need to robustify login logic
  * \todo parse outlets
  * \todo control outlets
  * \todo need to recognize signals in tty polls and not return errors, etc.
  */
class trippLitePDU : public MagAOXApp, public tty::usbDevice
{
   
public:
   
   /// Default c'tor.
   trippLitePDU();

   /// Setup the configuration system (called by MagAOXApp::setup())   
   virtual void setupConfig();
   
   /// Load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();
   
   /// Checks if the device was found during loadConfig.
   virtual int appStartup();
   
   /// Implementation of the FSM for the tripp lite PDU.
   virtual int appLogic();
   
   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();
   
   /// Parse the PDU devstatus response.
   /** 
     * \returns 0 on success
     * \returns \<0 on error, with value indicating location of error.
     */ 
   int parsePDUStatus( std::string & statStr,
                       float & voltage,
                       float & frequency,
                       float & current,
                       std::string & strRead 
                     );
};

trippLitePDU::trippLitePDU() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void trippLitePDU::setupConfig()
{
   tty::usbDevice::setupConfig(config);
}

void trippLitePDU::loadConfig()
{
   
   this->m_speed = B9600; //default for trippLite PDUs.  Will be overridden by any config setting.
   
   int rv = tty::usbDevice::loadConfig(config);
   
   if(rv != 0 && rv != TTY_E_NODEVNAMES) //Ignore error if nothing plugged in
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
}

int trippLitePDU::appStartup()
{
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

int trippLitePDU::appLogic()
{

   if( state() == stateCodes::UNINITIALIZED )
   {
      log<text_log>( "In appLogic but in state UNINITIALIZED.", logLevels::FATAL );
      return -1;
   }
   if( state() == stateCodes::INITIALIZED )
   {
      log<text_log>( "In appLogic but in state INITIALIZED.", logLevels::FATAL );
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
            log<software_fatal>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
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
      euidCalled();
      int rv = tty::usbDevice::connect();
      euidReal();
      
      if(rv == 0 && m_fileDescrip > 0) 
      {
         state(stateCodes::CONNECTED);
         
         if(!stateLogged())
         {
            std::stringstream logs;
            logs << "Connected to " << m_deviceName;
            log<text_log>(logs.str());
         }
         
      }
      else
      {
         state(stateCodes::FAILURE);
         log<text_log>("Error connecting to USB device.", logLevels::FATAL);
         return -1;
      }
   }
   
   if( state() == stateCodes::CONNECTED )
   {
      std::string strRead;  
      int rv = MagAOX::tty::ttyWriteRead( strRead, "\r", "$> ", true, m_fileDescrip, 1000, 1000); 
   
      if( rv == TTY_E_TIMEOUTONREADPOLL || rv == TTY_E_TIMEOUTONREAD )
      {
         std::cerr << "Read timeout  . . . \n";
         if( strRead.size() > 0 )
         {
            std::cerr << strRead << "\n";
         
            if(strRead.find("Username:") != std::string::npos)
            {
               MagAOX::tty::ttyWriteRead( strRead, "localadmin\r", "$> ", true, m_fileDescrip, 1000, 2000);
               MagAOX::tty::ttyWriteRead( strRead, "localadmin\r", "$> ", true, m_fileDescrip, 1000, 2000);
               rv = MagAOX::tty::ttyWriteRead( strRead, "\r", "$> ", true, m_fileDescrip, 1000, 2000); 
               
               if( rv == TTY_E_NOERROR )
               {
                  state(stateCodes::LOGGEDIN);
               }
               else
               {
                  state(stateCodes::FAILURE);
                  log<text_log>("Login failed.", logLevels::FATAL);
                  return -1;
               }
            }
            else return 0; //We keep trying until we get Username:
         }
         else
         {
            state(stateCodes::FAILURE);
            log<text_log>("No response from device. Can not connect.", logLevels::FATAL);
            return -1;
         }
      }
      else if (rv < 0)
      {
         state(stateCodes::FAILURE);
         log<text_log>(tty::ttyErrorString(rv), logLevels::FATAL);
         return -1;
      }
      
      state(stateCodes::LOGGEDIN);
   }
   
   if(state() == stateCodes::LOGGEDIN)
   {
      std::string strRead;
      int rv = MagAOX::tty::ttyWriteRead( strRead, "devstatus\r", "$> ", true, m_fileDescrip, 1000, 5000);
      
      if(rv < 0)
      {
         if(rv == TTY_E_TIMEOUTONREAD || rv == TTY_E_TIMEOUTONREADPOLL)
         {
            log<text_log>(tty::ttyErrorString(rv), logLevels::ERROR);
            return 0;
         }
         else
         {
            state(stateCodes::NOTCONNECTED);
            log<text_log>(tty::ttyErrorString(rv), logLevels::ERROR);
            
            return 0;
         }
      }

      std::string statStr;
      float voltage, frequency, current;
   
      rv = parsePDUStatus( statStr, voltage, frequency, current, strRead);
   
      if(rv == 0)
      {
         time::timespecX ts;
         ts.gettime();
         
         std::cerr << ts.time_s << " " << ts.time_ns << " " << statStr << " " << voltage << " " << frequency << " " << current << "\n";
         std::cout << ts.time_s << " " << ts.time_ns << " " << statStr << " " << voltage << " " << frequency << " " << current << std::endl;
      }
      else
      {
         std::cerr << "Parse Error: " << rv << "\n";
      }
      return 0;
   }
   
   state(stateCodes::FAILURE);
   log<text_log>("appLogic fell through", logLevels::FATAL);
   return -1;
   
}

int trippLitePDU::appShutdown()
{
   //don't bother
   return 0;
}

int trippLitePDU::parsePDUStatus( std::string & statStr,
                                  float & voltage,
                                  float & frequency,
                                  float & current,
                                  std::string & strRead 
                                )
{
   std::string pstr = mx::removeWhiteSpace(strRead);

   size_t st = pstr.find("Status:", 0);
   if( st == std::string::npos ) return -1;
   
   st = pstr.find(':', st) + 1;
   if( st == std::string::npos ) return -2;
   
   size_t ed = pstr.find('I', st);
   if( ed == std::string::npos ) return -3;

   statStr = pstr.substr(st, ed-st);
   
   st = pstr.find(':', ed) + 1;
   if( st == std::string::npos ) return -4;
   
   ed = pstr.find('V', st);
   if( ed == std::string::npos ) return -5;
   
   voltage = mx::convertFromString<float>( pstr.substr(st, ed-st) );
   
   st = pstr.find(':', ed) + 1;
   if( st == std::string::npos ) return -6;
   
   ed = pstr.find('H', st);
   if( ed == std::string::npos ) return -7;
   
   frequency = mx::convertFromString<float>( pstr.substr(st, ed-st) );
   
   st = pstr.find(':', ed) + 1;
   if( st == std::string::npos ) return -8;
   st = pstr.find(':', st) + 1;
   if( st == std::string::npos ) return -9;
   ed = pstr.find('A', st);
   if( ed == std::string::npos ) return -10;
   
   current = mx::convertFromString<float>( pstr.substr(st, ed-st) );
   
   return 0;
   
}

} //namespace app
} //namespace MagAOX

#endif //trippLitePDU_hpp
