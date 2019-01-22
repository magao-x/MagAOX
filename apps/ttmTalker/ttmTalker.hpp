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
#include "magaox_git_version.h"
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

/** MagAO-X application to do math on some numbers
  *
  */
class ttmTalker : public MagAOXApp<>, public tty::usbDevice
{

protected:		

public:

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
			//if(!stateLogged())
			//{
         std::stringstream logs;
         logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as " << m_deviceName;
         log<text_log>(logs.str());
         callCommand();
     	   //}
      }
   }

   if( state() == stateCodes::CONNECTED )
   {
      if( testConnection() != 0)
      {
         state(stateCodes::NOTCONNECTED);
      }
   }

   if( state() == stateCodes::NOTCONNECTED )
   {
      int rv = testConnection();
      if( rv == 0) 
      {
         state(stateCodes::CONNECTED);
         std::cout << "Connection successful." << std::endl;
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
      B115200              ///< [in] indicates the baud rate (see http://pubs.opengroup.org/onlinepubs/7908799/xsh/termios.h.html)
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

   std::string buffer;
   buffer.resize(6);
   buffer[0] = 0x05;
   buffer[1] = 0x00;
   buffer[2] = 0x00;
   buffer[3] = 0x00;
   buffer[4] = 0x50;
   buffer[5] = 0x01;
   std::string output;
   output.resize(90);
   rv = MagAOX::tty::ttyWriteRead( 
      output,        		///< [out] The string in which to store the output.
      buffer, 				   ///< [in] The characters to write to the tty.
      "",      				///< [in] A sequence of characters which indicates the end of transmission.
      false,             	///< [in] If true, strWrite.size() characters are read after the write
      fileDescrip,         ///< [in] The file descriptor of the open tty.
      2000,             	///< [in] The write timeout in milliseconds.
      2000               	///< [in] The read timeout in milliseconds.
   );


   std::cout << MagAOX::tty::ttyErrorString(rv) << std::endl;
   if (rv == TTY_E_NOERROR)
   {
      std::cout << *((uint32_t *) (  output.data() + 6)) << "   " << output.substr(10, 8) << std::endl;
   } 
   else
   {
      return -1;
   }

   if (*((uint32_t *) (  output.data() + 6)) == stoi(m_serial)) 
   {
      std::cout << "Serial number test successful" << std::endl;
      return 0;
   }
   else 
   {
      std::cout << "Serial number test unsuccessful" << std::endl;
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

} //namespace app
} //namespace MagAOX

#endif //ttmTalker_hpp
