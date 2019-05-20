/** \file ttmTalker.hpp
  * \brief The ttm Talker for generic usb devices
  * \author Chris Bohlman (cbohlmanaz@gmail.com)
  *
  * \ingroup ttmTalker_files
  *
  * History:
  * - 2018-08-10 created by CJB
  */
#ifndef ttmTalker_hpp
#define ttmTalker_hpp

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

  */
class ttmTalker : public MagAOXApp<>, public tty::usbDevice
{

protected:	
   pcf::IndiProperty m_indiP_position;
   std::vector<std::string> validStateCodes{};


public:

   INDI_NEWCALLBACK_DECL(ttmTalker, m_indiP_position);


   /// Default c'tor.
   ttmTalker();

   ~ttmTalker() noexcept
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

   int setUpMoving();

   int moveToPosition(float pos);

   int checkPosition();

};

inline ttmTalker::ttmTalker() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void ttmTalker::setupConfig()
{
   tty::usbDevice::setupConfig(config);
}

void ttmTalker::loadConfig()
{
   this->m_speed = B115200; //default for Zaber stages.  Will be overridden by any config setting.

   int rv = tty::usbDevice::loadConfig(config);

   if(rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND) //Ignore error if not plugged in
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
}

int ttmTalker::appStartup()
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

int ttmTalker::appLogic()
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
      
      if( testConnection() != 0)
      {
          state(stateCodes::NOTCONNECTED);
      }
      else {
         std::cout << "Connected!" << std::endl;
      }
   }

   if( state() == stateCodes::NOTCONNECTED )
   {
      int rv = testConnection();
      if( rv == 0) 
      {
         state(stateCodes::CONNECTED);
         std::cout << "Connection successful." << std::endl;
         if (checkPosition() != 0) {
            std::cout << "There's been an error with movement." << std::endl;
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

int ttmTalker::testConnection() 
{
   std::cout << "Testing connection..." << std::endl;
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
      std::cout << MagAOX::tty::ttyErrorString(rv) << std::endl;
      return rv;
   }

   std::cout << m_deviceName << "   " << fileDescrip << std::endl;

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

   //std::cout << output << std::endl;
   if (rv != TTY_E_NOERROR)
   {
      std::cerr << MagAOX::tty::ttyErrorString(rv) << std::endl;
   } 
   
   if (output.size() != 11)
   {
      std::cerr << "Wrongly sized output: " << output << " = size " << output.size() << std::endl;
      return -1;
   }
   //Compare output minus controller state (all are fine)
   if (output.substr(0, 7) == "1TS0000") {
   	//Test successful
      std::cout << "Test successful." << std::endl;
      // Set up moving if controller is not homed

      setUpMoving();
   	return 0;
   }
   else {
      //Diagnose error
      std::cerr << "Error occured: " << output << std::endl;
   	return -1;
   }
}

int ttmTalker::appShutdown()
{
	return 0;
}

int ttmTalker::callCommand()
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

int ttmTalker::setUpMoving() 
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
      std::cout << MagAOX::tty::ttyErrorString(rv) << std::endl;
      return rv;
   }

   //std::cout << m_deviceName << "   " << fileDescrip << std::endl;

   std::string buffer{"1OR\r\n"};
   rv = MagAOX::tty::ttyWrite( 
      buffer,              ///< [in] The characters to write to the tty.
      fileDescrip,         ///< [in] The file descriptor of the open tty.
      2000                ///< [in] The write timeout in milliseconds.
   );

   //std::cout << output << std::endl;
   if (rv != TTY_E_NOERROR)
   {
      std::cerr << MagAOX::tty::ttyErrorString(rv) << std::endl;
   } 

   return 0;
}

int ttmTalker::moveToPosition(float pos) 
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
      std::cout << MagAOX::tty::ttyErrorString(rv) << std::endl;
      return rv;
   }

   
   std::string moveAmt = std::to_string(pos);

   std::string buffer{"1PA"};
   buffer  = buffer + moveAmt + "\r\n";
   std::string output = "";
   output.resize(11);
   rv = MagAOX::tty::ttyWrite( 
      buffer,              ///< [in] The characters to write to the tty.
      fileDescrip,         ///< [in] The file descriptor of the open tty.
      2000                ///< [in] The write timeout in milliseconds.
   );

   //std::cout << output << std::endl;
   if (rv != TTY_E_NOERROR)
   {
      std::cerr << MagAOX::tty::ttyErrorString(rv) << std::endl;
   }

   return 0;
}

int ttmTalker::checkPosition() {
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
      std::cout << MagAOX::tty::ttyErrorString(rv) << std::endl;
      return rv;
   }

   std::cout << m_deviceName << "   " << fileDescrip << std::endl;

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
      std::cerr << MagAOX::tty::ttyErrorString(rv) << std::endl;
   } 
   
   if (output.size() != 11)
   {
      std::cerr << "Wrongly sized output: " << output << " = size " << output.size() << std::endl;
      return -1;
   }

   if (output == "1TS000028\r\n") {
      // Controller is moving.
      // Do I check something here?
      return 0;
   }
   else {
      // TODO: Check if target position is equal to current position
      return 0;
   }
}

INDI_NEWCALLBACK_DEFN(ttmTalker, m_indiP_position)(const pcf::IndiProperty &ipRecv)
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
      
      return moveToPosition(target);
      
   }
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //ttmTalker_hpp
