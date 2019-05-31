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
  */
#ifndef smc100ccCtrl_hpp
#define smc100ccCtrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"
#include "../../libMagAOX/tty/ttyUSB.hpp"

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
class smc100ccCtrl : public MagAOXApp<>, public tty::usbDevice
{

protected:	
   pcf::IndiProperty m_indiP_position;   ///< Indi variable for reporting CPU core loads
   std::vector<std::string> validStateCodes{};


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

   // Purges and resets device. Currently nothing in this app.
   virtual int callCommand();


   /// Tests if device is cabale of recieving/executing IO commands
   /** Sends command for device to return serial number, and compares to device serial number indi property
    * 
    * \returns -1 on serial numbers being different, thus ensuring connection test was sucsessful
    * \returns 0 on serial numbers being equal
    */
   int testConnection();


   /// Changes device status to READY
   /** Sets device to start to home in order to move controller
    * 
    * \returns -1 on error with sending command
    * \returns 0 on cpmmand sending success
    */
   int setUpMoving();


   /// Moves stage to the specified position
   /** Sends command for device to move to a position.
    * 
    * \returns -1 on error with sending command
    * \returns 0 on command sending success
    */
   int moveToPosition(
      float pos   /**< [in] the desired position*/
   );


   /// Verifies current status of controller
   /** Checks if controller is moving or has moved to correct position
    * 
    * \returns 0 if controller is currently moving or has moved correctly.
    * \returns -1 on error with sending commands or if current position does not match target position.
    */
   int checkPosition();


   /// Verifies current status of controller
   /** Checks if controller is moving or has moved to correct position
    * 
    * \returns 0 if controller is currently moving or has moved correctly.
    * \returns -1 on error with sending commands or if current position does not match target position.
    */
   int getPosition(
      float&
   );

   /// Returns any error controller has
   /** Called after every command is sent
    * 
    * \returns 0 if no error is reported
    * \returns -1 if an error is reported and error string is set in reference
    */
   int getLastError(
   	std::string&
   );

};

inline smc100ccCtrl::smc100ccCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void smc100ccCtrl::setupConfig()
{
   tty::usbDevice::setupConfig(config);
}

void smc100ccCtrl::loadConfig()
{
   this->m_speed = B115200; //default for Zaber stages.  Will be overridden by any config setting.

   int rv = tty::usbDevice::loadConfig(config);

   if(rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND) //Ignore error if not plugged in
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
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

	//Get the USB device if it's in udev
   if(m_deviceName == "")
   {
      state(stateCodes::NODEVICE);
   }
   else
   {
     	state(stateCodes::NOTCONNECTED);
     	std::stringstream logs;
     	logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as " << m_deviceName;
     	log<text_log>(logs.str());
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
            callCommand();
     	   }
      }
   }

   if( state() == stateCodes::CONNECTED )
   {
      // Only test connection before a command goes through
      // position, moving status, errors
      if( testConnection() != 0)
      {
          state(stateCodes::NOTCONNECTED);
      }
      else {
         // Still connected
         // Update current
         float current = -99;
         int rv = getPosition(current);
         if (rv == 0) {
            updateIfChanged(m_indiP_position, "curent", current);
         }
         else {
         	log<software_error>({__FILE__, __LINE__,"There's been an error with getting current controller position."});
         }
         // Check target and position
         if (checkPosition() != 0) {
           log<software_error>({__FILE__, __LINE__,"There's been an error with movement."});
         }
         // TODO: Log errors: use TE after every command?
         std::string errorString;
         if (getLastError(errorString) != 0) {
         	log<software_error>({__FILE__, __LINE__,errorString});
         }
      }
   }

   if( state() == stateCodes::NOTCONNECTED )
   {
      int rv = testConnection();
      if( rv == 0) 
      {
         // Connection successful
         state(stateCodes::CONNECTED);
         // Update current
         float current = -99;
         int rv = getPosition(current);
         if (rv == 0) {
            updateIfChanged(m_indiP_position, "curent", current);
         }
         else {
         	log<software_error>({__FILE__, __LINE__,"There's been an error with getting current controller position."});
         }
         // Check target and position
         if (checkPosition() != 0) {
           log<software_error>({__FILE__, __LINE__,"There's been an error with movement."});
         }
         // TODO: Log errors: use TE after every command?
         std::string errorString;
         if (getLastError(errorString) != 0) {
         	log<software_error>({__FILE__, __LINE__,errorString});
         }
      }
      else if (rv == TTY_E_TCGETATTR) 
      {
         state(stateCodes::NODEVICE);
      }

      if(state() == stateCodes::CONNECTED && !stateLogged())
      {
         std::stringstream logs;
         logs << "Connected to stage(s) on " << m_deviceName;
         log<text_log>(logs.str());
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

   return 0;
}

int smc100ccCtrl::testConnection() 
{
   // Testing connection
   int uid_rv = euidCalled();
   if(uid_rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      state(stateCodes::FAILURE);
      return -1;
   }

   int fileDescrip = 0;
   int rv = MagAOX::tty::ttyOpenRaw(
      fileDescrip,         ///< [out] the file descriptor.  Set to 0 on an error.
      m_deviceName,        ///< [in] the device path name, e.g. /dev/ttyUSB0
      B57600              ///< [in] indicates the baud rate (see http://pubs.opengroup.org/onlinepubs/7908799/xsh/termios.h.html)
   );

   uid_rv = euidReal();

   if(uid_rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      state(stateCodes::FAILURE);
      return -1;
   }

   if (rv != 0) 
   {
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
      return rv;
   }

   //std::cout << m_deviceName << "   " << fileDescrip << std::endl;

   std::string buffer{"1TS\r\n"};
   std::string output;
   output.resize(11);
   rv = MagAOX::tty::ttyWriteRead( 
      output,        		///< [out] The string in which to store the output.
      buffer, 				   ///< [in] The characters to write to the tty.
      "\r\n",      			///< [in] A sequence of characters which indicates the end of transmission.
      false,             	///< [in] If true, strWrite.size() characters are read after the write
      fileDescrip,         ///< [in] The file descriptor of the open tty.
      2000,             	///< [in] The write timeout in milliseconds.
      2000               	///< [in] The read timeout in milliseconds.
   );

   if (rv != TTY_E_NOERROR)
   {
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
   } 
   
   if (output.size() != 11)
   {
      std::string errorString = "Wrongly sized output in testConnection(): "+output+" = size "+std::to_string(output.size())+".";
      log<software_error>({__FILE__, __LINE__,errorString});
      return -1;
   }
   //Compare output minus controller state (all are fine)
   if (output.substr(0, 7) == "1TS0000") {
   	//Test successful
      // Set up moving if controller is not homed
      if (state() == stateCodes::CONNECTED) {
         setUpMoving();
      }  
   	return 0;
   }
   else {
      //Error, offending output is printed and diagnosis occurs in parent function
      log<software_error>({__FILE__, __LINE__,"Error occured: "+output});
   	return -1;
   }
}

int smc100ccCtrl::appShutdown()
{
	return 0;
}

int smc100ccCtrl::callCommand()
{
	/*
	// Set baud rate to 115200.
	ftStatus = FT_SetBaudRate(m_hFTDevice, (ULONG)uBaudRate);
	// 8 data bits, 1 stop bit, no parity
	ftStatus = FT_SetDataCharacteristics(m_hFTDevice, FT_BITS_8, FT_STOP_BITS_1,
	FT_PARITY_NONE);
	// Pre purge dwell 50ms.
	Sleep(uPrePurgeDwell);
	// Purge the device.
	ftStatus = FT_Purge(m_hFTDevice, FT_PURGE_RX | FT_PURGE_TX);
	// Post purge dwell 50ms.
	Sleep(uPostPurgeDwell);
	Page 27 of 367Thorlabs APT Controllers
	Host-Controller Communications Protocol
	Issue 23
	// Reset device.
	ftStatus = FT_ResetDevice(m_hFTDevice);
	// Set flow control to RTS/CTS.
	ftStatus = FT_SetFlowControl(m_hFTDevice, FT_FLOW_RTS_CTS, 0, 0);
	// Set RTS.
	ftStatus = FT_SetRts(m_hFTDevice);
	*/
	return 0;
}

int smc100ccCtrl::setUpMoving() 
{
   // Execute OR command
   int uid_rv = euidCalled();
   if(uid_rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      state(stateCodes::FAILURE);
      return -1;
   }

   int fileDescrip = 0;
   int rv = MagAOX::tty::ttyOpenRaw(
      fileDescrip,         ///< [out] the file descriptor.  Set to 0 on an error.
      m_deviceName,        ///< [in] the device path name, e.g. /dev/ttyUSB0
      B57600              ///< [in] indicates the baud rate (see http://pubs.opengroup.org/onlinepubs/7908799/xsh/termios.h.html)
   );

   uid_rv = euidReal();

   if(uid_rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      state(stateCodes::FAILURE);
      return -1;
   }

   if (rv != 0) 
   {
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
      return rv;
   }

   //std::cout << m_deviceName << "   " << fileDescrip << std::endl;

   std::string buffer{"1OR\r\n"};
   rv = MagAOX::tty::ttyWrite( 
      buffer,              ///< [in] The characters to write to the tty.
      fileDescrip,         ///< [in] The file descriptor of the open tty.
      2000                ///< [in] The write timeout in milliseconds.
   );

   if (rv != TTY_E_NOERROR)
   {
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
   } 
   state(stateCodes::READY);
   return 0;
}

int smc100ccCtrl::moveToPosition(float pos) 
{
   int uid_rv = euidCalled();
   if(uid_rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      state(stateCodes::FAILURE);
      return -1;
   }

   int fileDescrip = 0;
   int rv = MagAOX::tty::ttyOpenRaw(
      fileDescrip,         ///< [out] the file descriptor.  Set to 0 on an error.
      m_deviceName,        ///< [in] the device path name, e.g. /dev/ttyUSB0
      B57600              ///< [in] indicates the baud rate (see http://pubs.opengroup.org/onlinepubs/7908799/xsh/termios.h.html)
   );

   uid_rv = euidReal();

   if(uid_rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      state(stateCodes::FAILURE);
      return -1;
   }

   if (rv != 0) 
   {
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
      return rv;
   }


   // TODO: Check if controller is in correct state
   std::string buffer{"1PA"};
   buffer = buffer + std::to_string(pos) + "\r\n";
   rv = MagAOX::tty::ttyWrite( 
      buffer,              ///< [in] The characters to write to the tty.
      fileDescrip,         ///< [in] The file descriptor of the open tty.
      2000                ///< [in] The write timeout in milliseconds.
   );

   if (rv != TTY_E_NOERROR)
   {
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
   }

   state(stateCodes::OPERATING);

   std::string errorString;
   if (getLastError(errorString) == 0) {
   	return 0;
   }
   else {
   	log<software_error>({__FILE__, __LINE__,errorString});
   	return -1;
   }
}

int smc100ccCtrl::checkPosition() {
   int uid_rv = euidCalled();
   if(uid_rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      state(stateCodes::FAILURE);
      return -1;
   }

   int fileDescrip = 0;
   int rv = MagAOX::tty::ttyOpenRaw(
      fileDescrip,         ///< [out] the file descriptor.  Set to 0 on an error.
      m_deviceName,        ///< [in] the device path name, e.g. /dev/ttyUSB0
      B57600              ///< [in] indicates the baud rate (see http://pubs.opengroup.org/onlinepubs/7908799/xsh/termios.h.html)
   );

   uid_rv = euidReal();

   if(uid_rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      state(stateCodes::FAILURE);
      return -1;
   }

   if (rv != 0) 
   {
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
      return rv;
   }

   //std::cout << m_deviceName << "   " << fileDescrip << std::endl;

   std::string buffer{"1TS\r\n"};
   std::string output;
   output.resize(11);
   rv = MagAOX::tty::ttyWriteRead( 
      output,              ///< [out] The string in which to store the output.
      buffer,              ///< [in] The characters to write to the tty.
      "\r\n",              ///< [in] A sequence of characters which indicates the end of transmission.
      false,               ///< [in] If true, strWrite.size() characters are read after the write
      fileDescrip,         ///< [in] The file descriptor of the open tty.
      2000,                ///< [in] The write timeout in milliseconds.
      2000                 ///< [in] The read timeout in milliseconds.
   );

   if (rv != TTY_E_NOERROR)
   {
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
   } 
   
   if (output.size() != 11)
   {
      std::string errorString = "Wrongly sized output in checkPosition(): "+output+" = size "+std::to_string(output.size())+".";
      log<software_error>({__FILE__, __LINE__,errorString});
      return -1;
   }

   if (output.substr(0, 9) == "1TS000028") {
      // Controller is moving.
      return 0;
   }
   else {
      state(stateCodes::READY);

      // TODO: Get target position and current position from INDI (This code doesn't compile)

      float current = -99, target = -99;

      // try
      // {
      //    current = ipRecv["current"].get<float>();
      // }
      // catch(...){}
      
      // try
      // {
      //    target = ipRecv["target"].get<float>();
      // }
      // catch(...){}

      // TODO: Check if target position is equal to current position with error band

      if (target != current) {
         log<software_error>({__FILE__, __LINE__,"Current and target don't match when controller is not moving."});
         return -1;
      }

      return 0;
   }
}

int smc100ccCtrl::getPosition(float& current) 
{
   int uid_rv = euidCalled();
   if(uid_rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      state(stateCodes::FAILURE);
      return -1;
   }

   int fileDescrip = 0;
   int rv = MagAOX::tty::ttyOpenRaw(
      fileDescrip,         ///< [out] the file descriptor.  Set to 0 on an error.
      m_deviceName,        ///< [in] the device path name, e.g. /dev/ttyUSB0
      B57600              ///< [in] indicates the baud rate (see http://pubs.opengroup.org/onlinepubs/7908799/xsh/termios.h.html)
   );

   uid_rv = euidReal();

   if(uid_rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      state(stateCodes::FAILURE);
      return -1;
   }

   if (rv != 0) 
   {
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
      return rv;
   }

   //std::cout << m_deviceName << "   " << fileDescrip << std::endl;

   std::string buffer{"1TP\r\n"};
   std::string output;
   output.resize(13);
   rv = MagAOX::tty::ttyWriteRead( 
      output,              ///< [out] The string in which to store the output.
      buffer,              ///< [in] The characters to write to the tty.
      "\r\n",              ///< [in] A sequence of characters which indicates the end of transmission.
      false,               ///< [in] If true, strWrite.size() characters are read after the write
      fileDescrip,         ///< [in] The file descriptor of the open tty.
      2000,                ///< [in] The write timeout in milliseconds.
      2000                 ///< [in] The read timeout in milliseconds.
   );

   if (rv != TTY_E_NOERROR)
   {
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
   } 
   
   // TODO: What sizes should positive and negative returns be?
   // if (output.size() != 13)
   // {
   //    std::cerr << "Wrongly sized output: " << output << " = size " << output.size() << std::endl;
   //    return -1;
   // }

   // parse current and place into argument
   try {
      current = std::stof(output.substr(3));
   }
   catch (...) {
   	return -1;
   }
   return 0;
}

int smc100ccCtrl::getLastError( std::string& errorString) {
	int uid_rv = euidCalled();
   if(uid_rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      state(stateCodes::FAILURE);
      return -1;
   }

   int fileDescrip = 0;
   int rv = MagAOX::tty::ttyOpenRaw(
      fileDescrip,         ///< [out] the file descriptor.  Set to 0 on an error.
      m_deviceName,        ///< [in] the device path name, e.g. /dev/ttyUSB0
      B57600              ///< [in] indicates the baud rate (see http://pubs.opengroup.org/onlinepubs/7908799/xsh/termios.h.html)
   );

   uid_rv = euidReal();

   if(uid_rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      state(stateCodes::FAILURE);
      return -1;
   }

   if (rv != 0) 
   {
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
      return rv;
   }

   //std::cout << m_deviceName << "   " << fileDescrip << std::endl;

   std::string buffer{"1TE\r\n"};
   std::string output;
   output.resize(6);
   rv = MagAOX::tty::ttyWriteRead( 
      output,              ///< [out] The string in which to store the output.
      buffer,              ///< [in] The characters to write to the tty.
      "\r\n",              ///< [in] A sequence of characters which indicates the end of transmission.
      false,               ///< [in] If true, strWrite.size() characters are read after the write
      fileDescrip,         ///< [in] The file descriptor of the open tty.
      2000,                ///< [in] The write timeout in milliseconds.
      2000                 ///< [in] The read timeout in milliseconds.
   );

   if (rv != TTY_E_NOERROR)
   {
      log<software_error>({__FILE__, __LINE__,MagAOX::tty::ttyErrorString(rv)});
   } 

   if (output.at(3) == '@') {
   	return 0;
   }
   else {
   	switch(output.at(3)) {
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
   	}
   	return -1;
   }
}

INDI_NEWCALLBACK_DEFN(smc100ccCtrl, m_indiP_position)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_position.getName())
   {
      float current = -99, target = -99;

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
      
      if(target == -99) target = current;
      
      if(target <= 0) return 0;
      
      //Lock the mutex, waiting if necessary
      std::unique_lock<std::mutex> lock(m_indiMutex);

      updateIfChanged(m_indiP_position, "target", target);

      // Check for state in here?
      // Busy wait until we have a non moving state
      while (state() == stateCodes::OPERATING);
      
      return moveToPosition(target);
   }
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //smc100ccCtrl_hpp
