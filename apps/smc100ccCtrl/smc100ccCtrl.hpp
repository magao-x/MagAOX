/** \file smc100ccCtrl.hpp
  * \brief The smc controller communicator
  * \author Chris Bohlman (cbohlman@pm.me)
  *
  * \ingroup smc100ccCtrl_files
  *
  * History:
  * - 2019-01-10 created by CJB
  *
  * To compile:
  * - make clean (recommended)
  * - make CACAO=false
  * - sudo make install
  * - /opt/MagAOX/bin/smc100ccCtrl 
  *
  *
  * To run with cursesIndi
  * 1. /opt/MagAOX/bin/xindiserver -n xindiserverMaths
  * 2. /opt/MagAOX/bin/smc100ccCtrl -n ssmc100ccCtrl
  * 3. /opt/MagAOX/bin/cursesINDI 
  */
#ifndef smc100ccCtrl_hpp
#define smc100ccCtrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <bitset>


namespace MagAOX
{
namespace app
{

/** TS command: Checks if there were any errors during initialization
  * Solid orange LED: everything is okay, TS should return 1TS00000A
  * PW command: change all stage and motor configuration parameters
  * OR command: gets controller to ready state (must go through homing first)
  * In ready state, can move relative and move absolute
  * RS command: TO get from ready to not referenced

  Change to stateCodes::OPERATING and stateCodes::READY

  */
class smc100ccCtrl : public MagAOXApp<>, public tty::usbDevice, public dev::ioDevice, public dev::stdMotionStage<smc100ccCtrl>
{

   friend class dev::stdMotionStage<smc100ccCtrl>;
protected:   
   
   /** \name Configurable Parameters 
     *
     *@{
     */
   double m_homingOffset {0};
   
   ///@}
   
   
   pcf::IndiProperty m_indiP_position;   ///< Indi variable for reporting the stage position.
   
   std::vector<std::string> validStateCodes{};
   
   double m_position {0};
   
   double m_target {0};

   bool m_wasHoming {0};
   
   
public:

   INDI_NEWCALLBACK_DECL(smc100ccCtrl, m_indiP_position);


   /// Default c'tor.
   smc100ccCtrl();

   ~smc100ccCtrl() noexcept
   {
   }

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// Load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Checks if the device was found during loadConfig.
   virtual int appStartup();

   /// Changes device state based on testing connection and device status
   virtual int appLogic();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();

   int makeCom( std::string &str, 
                const std::string & com
              );
   
   int splitResponse( int &axis, 
                      std::string &com, 
                      std::string &val, 
                      std::string &resp
                    );


   int getCtrlState( std::string &state );
   
   /// Tests if device is cabale of recieving/executing IO commands
   /** Sends command for device to return serial number, and compares to device serial number indi property
    * 
    * \returns -1 on serial numbers being different, thus ensuring connection test was sucsessful
    * \returns 0 on serial numbers being equal
    */
   int testConnection();

   /// Verifies current status of controller
   /** Checks if controller is moving or has moved to correct position
    * 
    * \returns 0 if controller is currently moving or has moved correctly.
    * \returns -1 on error with sending commands or if current position does not match target position.
    */
   int getPosition( double & pos  /**< [out] on output, the current position*/);

   /// Returns any error controller has
   /** Called after every command is sent
    * 
    * \returns 0 if no error is reported
    * \returns -1 if an error is reported and error string is set in reference
    */
   int getLastError( std::string& errStr /** [out] the last error string */);

   /** \name Standard Motion Stage Interface
     * @{
     * 
     */
   
   int stop();
   
   int startHoming();
   
   double presetNumber();
   
   int moveTo(double position);
   
   
   ///@}
};

inline smc100ccCtrl::smc100ccCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   m_powerOnWait = 5; // default to 5 seconds for controller boot up.
   
   m_defaultPositions = false;
   
   return;
}

void smc100ccCtrl::setupConfig()
{
   config.add("stage.homingOffset", "", "stage.homingOffset", argType::Required, "stage", "homingOffset", false, "float", "Homing offset, a.k.a. default starting position.");
   
   tty::usbDevice::setupConfig(config);
   dev::ioDevice::setupConfig(config);
   dev::stdMotionStage<smc100ccCtrl>::setupConfig(config);
   
}

void smc100ccCtrl::loadConfig()
{
   
   config(m_homingOffset, "stage.homingOffset");
   
   this->m_baudRate = B57600; //default for SMC100CC controller.  Will be overridden by any config setting.

   int rv = tty::usbDevice::loadConfig(config);

   if(rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND) //Ignore error if not plugged in
   {
      log<software_error>({ __FILE__, __LINE__, "error loading USB device configs"});
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
      m_shutdown = 1;
   }
   
   rv = dev::ioDevice::loadConfig(config);
   if(rv != 0)
   {
      log<software_error>({ __FILE__, __LINE__, "error loading io device configs"});
      m_shutdown = 1;
   }
   
   dev::stdMotionStage<smc100ccCtrl>::loadConfig(config);
   if(rv != 0)
   {
      log<software_error>({ __FILE__, __LINE__, "error loading io device configs"});
      m_shutdown = 1;
   }
}

int smc100ccCtrl::appStartup()
{
   
    REG_INDI_NEWPROP(m_indiP_position, "position", pcf::IndiProperty::Number);
    m_indiP_position.add (pcf::IndiElement("current"));
    m_indiP_position["current"].set(0);
    m_indiP_position.add (pcf::IndiElement("target"));
   

   if( state() == stateCodes::UNINITIALIZED )
   {
      log<text_log>( "In appStartup but in state UNINITIALIZED.", logPrio::LOG_CRITICAL );
      return -1;
   }
   
   if(m_presetNames.size() != m_presetPositions.size())
   {
      return log<text_log,-1>("must set a position for each preset", logPrio::LOG_CRITICAL);
   }
   
   m_presetNames.insert(m_presetNames.begin(), "none");
   m_presetPositions.insert(m_presetPositions.begin(), -1);
   
   int rv = dev::stdMotionStage<smc100ccCtrl>::appStartup();
   if(rv != 0)
   {
      log<software_error>({ __FILE__, __LINE__, "error loading io device configs"});
      m_shutdown = 1;
   }
   
   return 0;
}

int smc100ccCtrl::appLogic()
{   
   if( state() == stateCodes::INITIALIZED )
   {
      log<text_log>( "In appLogic but in state INITIALIZED.", logPrio::LOG_CRITICAL );
      return -1;
   }

   if( state() == stateCodes::POWERON)
   {
      if(!powerOnWaitElapsed()) return 0;
      
      if(m_deviceName == "")
      {
         state(stateCodes::NODEVICE);
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
      euidCalled();
      int rv = connect();
      euidReal();

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
         
         //if connect failed, and there is a device, then we have some other problem.
         state(stateCodes::FAILURE);
         if(!stateLogged()) log<software_error>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
         return -1;
                           
      }

      if( testConnection() == 0 ) 
      {
         state(stateCodes::CONNECTED);
      }
      else 
      {
         std::string errorString;
         if (getLastError(errorString) != 0) 
         {
              log<software_error>({__FILE__, __LINE__,errorString});
         }
         
         return 0;
      }
      

      if(state() == stateCodes::CONNECTED && !stateLogged())
      {
         std::stringstream logs;
         logs << "Connected to stage(s) on " << m_deviceName;
         log<text_log>(logs.str());
      }
   }

   //If we're here, we can get state from controller...
   std::string axState;
   getCtrlState(axState); /// \todo error check
   
   if(axState[0] == '0') 
   {
      std::cerr << "Axis state: " << axState[0] << " " << axState[1] << "\n";
      state(stateCodes::NOTHOMED); //This always means this.
   }
   else if (axState[0] == '1' && axState[1] == '0')
   {
      //Need to download stage info
      log<text_log>("getting stage information");
      std::string com;
      if(makeCom(com, "PW1") < 0)
      {
         log<software_error>({__FILE__, __LINE__,"Error making command PW1" });
         return 0;
      }
   
      int rv = MagAOX::tty::ttyWrite( com, m_fileDescrip, m_writeTimeout); 
      if (rv != TTY_E_NOERROR)
      {
         if(m_powerTargetState == 0) return -1;
         log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
         return -1;
      } 
      
      sleep(5);
      if(makeCom(com, "ZX2") < 0)
      {
         log<software_error>({__FILE__, __LINE__,"Error making command ZX2" });
         return 0;
      }
   
      rv = MagAOX::tty::ttyWrite( com, m_fileDescrip, m_writeTimeout); 
      if (rv != TTY_E_NOERROR)
      {
         if(m_powerTargetState == 0) return -1;
         log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
         return -1;
      }
      
      sleep(5);
      if(makeCom(com, "PW0") < 0)
      {
         log<software_error>({__FILE__, __LINE__,"Error making command PW0" });
         return 0;
      }
   
      rv = MagAOX::tty::ttyWrite( com, m_fileDescrip, m_writeTimeout); 
      if (rv != TTY_E_NOERROR)
      {
         if(m_powerTargetState == 0) return -1;
         log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
         return -1;
      }
      
      sleep(5);
      log<text_log>("stage information loaded");
      return 0;
      
   }
   
   else if (axState[0] == '1' && (axState[1] == 'E' || axState[1] == 'F'))
   {
      state(stateCodes::HOMING);
      m_moving = 1;
      m_wasHoming = 1;
   }
   else if (axState[0] == '2') 
   {
      m_moving = 1;
      state(stateCodes::OPERATING);
   }
   else if (axState[0] == '3' && isdigit(axState[1]))
   {
      if(m_wasHoming)
      {
         std::unique_lock<std::mutex> lock(m_indiMutex);
         moveTo(m_homingOffset);
         m_wasHoming = 0;
      }
      else
      {
         m_moving = 0;
         state(stateCodes::READY);
      }
   }
   else if (axState[0] == '3')
   {
      log<text_log>("Stage disabled.  Enabling");
      std::string com;
      if(makeCom(com, "MM1") < 0)
      {
         log<software_error>({__FILE__, __LINE__,"Error making command PW1" });
         return 0;
      }
      int rv = MagAOX::tty::ttyWrite( com, m_fileDescrip, m_writeTimeout); 
      if (rv != TTY_E_NOERROR)
      {
         if(m_powerTargetState == 0) return -1;
         log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
         return -1;
      } 
   }
   else
   {
      sleep(1);
      if(m_powerState == 0) return 0;
      
      log<software_error>({__FILE__,__LINE__, "Invalid state: " + axState});
      
      state(stateCodes::ERROR);
   }
      
   if( state() == stateCodes::NOTHOMED)
   {
      if(m_powerOnHome)
      {
         std::unique_lock<std::mutex> lock(m_indiMutex);
         startHoming(); 
      }
      return 0;
   }
   
   if( state() == stateCodes::READY || state() == stateCodes::OPERATING)
   {
      std::unique_lock<std::mutex> lock(m_indiMutex);
   

      int rv = getPosition(m_position);
            
      if(rv < 0)
      {
         sleep(1);
         if(m_powerState == 0) return 0;
         
         std::string errorString;
      
         if (getLastError(errorString) != 0 && errorString.size() != 0) 
         {
            log<software_error>({__FILE__, __LINE__,errorString});
         }
      
         log<software_error>({__FILE__, __LINE__,"There's been an error with getting current controller position."});
      }

      updateIfChanged(m_indiP_position, "current", m_position);
      
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
            m_indiP_position.setState(INDI_BUSY);
         }
         else
         {
            m_indiP_position.setState(INDI_IDLE);
            m_indiP_position["target"] = m_position;
         }
         m_indiDriver->sendSetProperty(m_indiP_position);
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

      dev::stdMotionStage<smc100ccCtrl>::updateINDI();
   
         
      return 0;
   }

   if( state() == stateCodes::ERROR )
   {
      if(m_powerTargetState == 0) return 0;
      sleep(1);
      if(m_powerState == 0) return 0;
      
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

      sleep(1);
      if(m_powerState == 0) return 0;
      
      state(stateCodes::FAILURE);
      if(!stateLogged())
      {
         log<text_log>("Error NOT due to loss of USB connection.  I can't fix it myself.", logPrio::LOG_CRITICAL);
      }
      return -1;
   }

   return 0;
}

int smc100ccCtrl::testConnection() 
{
   std::string buffer{"1TS\r\n"};
   std::string output;
   
   int rv = MagAOX::tty::ttyWriteRead( output, buffer, "\r\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout); 

   if (rv != TTY_E_NOERROR)
   {
      if(m_powerTargetState == 0) return -1;
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
      return -1;
   } 
   
   int axis;
   std::string com;
   std::string val;
   
   splitResponse( axis, com, val, output);

   return 0;
}

int smc100ccCtrl::appShutdown()
{
   return 0;
}

int smc100ccCtrl::makeCom( std::string &str, 
                           const std::string & com
                         )
{
   char tmp[10];

   int axis = 1;
   
   snprintf(tmp, 10, "%i", axis);
   
   str = tmp;
   
   str += com;
   
   str += "\r\n";

   return 0;
}
int smc100ccCtrl::splitResponse(int &axis, std::string &com, std::string &val, std::string &resp)
{
   if(resp.length() < 3)
   {
      log<software_error>({__FILE__,__LINE__, "Invalid response"});
      return -1;
   }
   
   if(isalpha(resp[0]))
   {
      log<software_error>({__FILE__,__LINE__, "Invalid response"});
      axis = 0;
      com = "";
      val = resp;
      return 0;
   }

   if(isalpha(resp[1]))
   {
      axis = resp[0] - '0';
   }
   else
   {
      axis = atoi(resp.substr(0,2).c_str());
   }
   
   if(axis < 10)
   {
      
      com = resp.substr(1,2);
      if(resp.length() < 4 ) val = "";
      else val = resp.substr(3, resp.length()-3);
       if(val.size() > 1)
      {
         while(val[val.size()-1] == '\r' || val[val.size()-1] == '\n') 
         {
            val.erase(val.size()-1);
            if(val.size() < 1) break;
         }
      }
   }
   else
   {
      if(resp.length() < 4)
      {
         log<software_error>({__FILE__,__LINE__, "Invalid response"});
         com = "";
         val = "";
         return -1;
      }
      com = resp.substr(2,2);
      if(resp.length() < 5) val = "";
      else val = resp.substr(4, resp.length()-4);
      
      if(val.size() > 1)
      {
         while(val[val.size()-1] == '\r' || val[val.size()-1] == '\n') 
         {
            val.erase(val.size()-1);
            if(val.size() < 1) break;
         }
      }
   }

   return 0;
}

int smc100ccCtrl::getCtrlState( std::string &state )
{
   std::string com, resp;
   
   if(makeCom(com, "TS") < 0)
   {
      log<software_error>({__FILE__, __LINE__,"Error making command TS" });
      return 0;
   }
   
   int rv = MagAOX::tty::ttyWriteRead( resp, com, "\r\n", false, m_fileDescrip, m_readTimeout, m_writeTimeout); 
   if (rv != TTY_E_NOERROR)
   {
      if(m_powerTargetState == 0) return -1;
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
      return -1;
   } 
   
   std::cerr << "TS Response: " << resp << "\n";
   int raxis;
   std::string rcom, rval;
   
   splitResponse(raxis, rcom, rval, resp);
   
   if(rcom == "")
   {
      log<software_error>({__FILE__, __LINE__, "An Error occurred"});
      return -1;
   }
      
   if(raxis != 1)
   {
      log<software_error>({__FILE__, __LINE__, "Wrong axis returned"});
      return -1;
   }

   if(rcom != "TS")
   {
      log<software_error>({__FILE__, __LINE__, "Wrong command returned"});
      return -1;
   }
   
   if(rval.length() != 6)
   {
      log<software_error>({__FILE__, __LINE__,"Incorrect response length" });
      return -1;
   }

   state = rval.substr(4, 2);

   return 0;
}


int smc100ccCtrl::getPosition(double& current) 
{
   std::string buffer{"1TP\r\n"};
   std::string output;
   int rv = MagAOX::tty::ttyWriteRead( output, buffer, "\r\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout); 

   if (rv != TTY_E_NOERROR)
   {
      if(m_powerTargetState == 0) return -1;
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
      return -1;
   } 

   // Parse current and place into argument
   try 
   {
      current = std::stod(output.substr(3));
   }
   catch (...) 
   {
      log<software_error>({__FILE__, __LINE__,"Error occured: Unexpected output in getPosition()"});
      return -1;
   }
   return 0;
}

int smc100ccCtrl::getLastError( std::string& errorString) 
{
   std::string buffer{"1TE\r\n"};
   std::string output;
   int rv = MagAOX::tty::ttyWriteRead( output, buffer, "\r\n",false, m_fileDescrip, m_writeTimeout, m_readTimeout);

   if (rv != TTY_E_NOERROR)
   {
      if(m_powerTargetState == 0) return -1;
      log<software_error>({__FILE__, __LINE__});
      errorString = MagAOX::tty::ttyErrorString(rv);
      return -1;
   } 

   char status;
   try 
   {
      status = output.at(3);
   }
   catch (const std::out_of_range& oor) 
   {
      log<software_error>({__FILE__, __LINE__});
      errorString = "Unknown output; controller not responding correctly.";
      return -1;
   }

   if (status == '@') 
   {
      return 0;
   }
   else 
   {
      switch(status) 
      {
         case 'A': 
            errorString = "Unknown message code or floating point controller address.";
            break;
         case 'B': 
            errorString = "Controller address not correct.";
            break;
         case 'C': 
            errorString = "Parameter missing or out of range.";
            break;
         case 'D': 
            errorString = "Command not allowed.";
            break;
         case 'E': 
            errorString = "Home sequence already started.";
            break;
         case 'F': 
            errorString = "ESP stage name unknown.";
            break;
         case 'G': 
            errorString = "Displacement out of limits.";
            break;
         case 'H': 
            errorString = "Command not allowed in NOT REFERENCED state.";
            break;
         case 'I': 
            errorString = "Command not allowed in CONFIGURATION state.";
            break;
         case 'J': 
            errorString = "Command not allowed in DISABLE state.";
            break;
         case 'K': 
            errorString = "Command not allowed in READY state.";
            break;
         case 'L': 
            errorString = "Command not allowed in HOMING state.";
            break;
         case 'M': 
            errorString = "UCommand not allowed in MOVING state.";
            break;
         case 'N': 
            errorString = "Current position out of software limit.";
            break;
         case 'S': 
            errorString = "Communication Time Out.";
            break;
         case 'U': 
            errorString = "Error during EEPROM access.";
            break;
         case 'V': 
            errorString = "Error during command execution.";
            break;
         case 'W': 
            errorString = "Command not allowed for PP version.";
            break;
         case 'X': 
            errorString = "Command not allowed for CC version.";
            break;
         default:
            errorString = "unknown status";
      }
      
      log<software_error>({__FILE__, __LINE__});
      return -1;
   }
}

INDI_NEWCALLBACK_DEFN(smc100ccCtrl, m_indiP_position)(const pcf::IndiProperty &ipRecv)
{
   if(!( state() == stateCodes::READY || state() == stateCodes::OPERATING))
   {
      log<text_log>("can not command position in current state");
      return 0;
   }
   
   if (ipRecv.getName() == m_indiP_position.getName())
   {
      float current = -1e55, target = -1e55;

      try
      {
         current = ipRecv["current"].get<float>();
      }
      catch(...){}
      
      try
      {
         target = ipRecv["target"].get<float>();
      }
      catch(...){}
      
      if(target == -1e55) target = current;
      
      if(target == -1e55) return 0;
      
      //Lock the mutex, waiting if necessary
      std::unique_lock<std::mutex> lock(m_indiMutex);

      updateIfChanged(m_indiP_position, "target", target, INDI_BUSY);
      m_target = target;

      
      return moveTo(target);
   }
   return -1;
}

int smc100ccCtrl::stop()
{
   std::unique_lock<std::mutex> lock(m_indiMutex);
   std::string buffer{"1ST\r\n"};
   int rv = MagAOX::tty::ttyWrite(buffer, m_fileDescrip, m_writeTimeout); 

   updateSwitchIfChanged(m_indiP_stop, "request", pcf::IndiElement::Off, INDI_IDLE);
   
   if (rv != TTY_E_NOERROR)
   {
      if(m_powerTargetState == 0) return -1;
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
      return -1;
   } 
   return 0;
}
   
int smc100ccCtrl::startHoming()
{
   updateSwitchIfChanged(m_indiP_home, "request", pcf::IndiElement::Off, INDI_IDLE);
   
   std::string buffer{"1OR\r\n"};
   int rv = MagAOX::tty::ttyWrite(buffer, m_fileDescrip, m_writeTimeout); 

   if (rv != TTY_E_NOERROR)
   {
      if(m_powerTargetState == 0) return -1;
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
      return -1;
   } 
   return 0;
}

double smc100ccCtrl::presetNumber()
{
   for( size_t n=1; n < m_presetPositions.size(); ++n)
   {
      if( fabs(m_position-m_presetPositions[n]) < 1e-3) return n;
   }
   
   return 0;
}

int smc100ccCtrl::moveTo(double position)
{      
   std::string buffer{"1PA"};
   buffer = buffer + std::to_string(position) + "\r\n";
   
   int rv = MagAOX::tty::ttyWrite( buffer, m_fileDescrip, m_writeTimeout);

   if (rv != TTY_E_NOERROR)
   {
      if(m_powerTargetState == 0) return -1;
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
      return -1;
   }

   std::string errorString;
   if (getLastError(errorString) == 0) 
   {
      state(stateCodes::OPERATING);
      updateIfChanged(m_indiP_position, "target", position);
      return 0;
   }
   else 
   {
      log<software_error>({__FILE__, __LINE__,errorString});
      return -1;
   }
}
   
} //namespace app
} //namespace MagAOX

#endif //smc100ccCtrl_hpp
