// inter process communications system: INDI in magaoxmaths
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
		class ttmTalker : public MagAOXApp<>, public tty::usbDevice {

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

   			/// Implementation of the FSM for the maths.
			virtual int appLogic();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
			virtual int appShutdown();

			virtual int callCommand();

		};

		ttmTalker::ttmTalker() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
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

			if( state() == stateCodes::NOTCONNECTED )
		   	{
		      	//if( testConnection() == ZC_CONNECTED) state(stateCodes::CONNECTED);
		      	int rv = euidCalled();
		      	if(rv < 0)
    			{
    				log<software_critical>({__FILE__, __LINE__});
    				state(stateCodes::FAILURE);
    				return -1;
				}
				rv = euidReal();
      			if(rv < 0)
      			{
         			log<software_critical>({__FILE__, __LINE__});
         			state(stateCodes::FAILURE);
         			return -1;
				}
		   		int fileDescrip;
            	MagAOX::tty::ttyOpenRaw(
            		fileDescrip,      	///< [out] the file descriptor.  Set to 0 on an error.
               		m_deviceName, 		///< [in] the device path name, e.g. /dev/ttyUSB0
                	B115200             ///< [in] indicates the baud rate (see http://pubs.opengroup.org/onlinepubs/7908799/xsh/termios.h.html)
              	);
            	std::cout << m_deviceName << "    " << fileDescrip << std::endl;
            	std::string buffer;
            	buffer.resize(6);
            	buffer[0] = 0x05;
            	buffer[1] = 0x00;
            	buffer[2] = 0x00;
            	buffer[3] = 0x00;
            	buffer[4] = 0x50;
            	buffer[5] = 0x01;

     //        	MagAOX::tty::ttyWrite(
     //        		buffer, 			///< [in] The characters to write to the tty.
     //          	fileDescrip,        ///< [in] The file descriptor of the open tty.
	 // 			2000				///< [in] The timeout in milliseconds.
     //        	);

				std::string output;
            	output.resize(90);
            	rv = MagAOX::tty::ttyWriteRead( 
            	  output,        		///< [out] The string in which to store the output.
                  buffer, 				///< [in] The characters to write to the tty.
                  "",      				///< [in] A sequence of characters which indicates the end of transmission.
                  false,             	///< [in] If true, strWrite.size() characters are read after the write
                  fileDescrip,          ///< [in] The file descriptor of the open tty.
                  2000,             	///< [in] The write timeout in milliseconds.
                  2000               	///< [in] The read timeout in milliseconds.
                );
            	long serial;
            	char temp[4];
            	int iterator = 0;
            	switch(rv) {
            		case TTY_E_NOERROR:
            			std::cout << "No error with read or write." << std::endl;
            			std::cout << *((uint32_t *) (  output.data() + 6)) << '\t' << output.substr(10, 8) << std::endl;

            			break;
            		case TTY_E_TIMEOUTONWRITEPOLL: 
            			std::cout << "Error with write poll timeout." << std::endl;
            			break;
 					case TTY_E_ERRORONWRITEPOLL:
            			std::cout << "Error with write poll." << std::endl;
            			break;
					case TTY_E_TIMEOUTONWRITE:
            			std::cout << "Error with write timeout." << std::endl;
            			break;
					case TTY_E_ERRORONWRITE:
            			std::cout << "Error with writing to file." << std::endl;
            			break;
  					case TTY_E_TIMEOUTONREADPOLL:
            			std::cout << "Error with read poll timeout." << std::endl;
            			break;
  					case TTY_E_ERRORONREADPOLL:
            			std::cout << "Error with read poll." << std::endl;
            			break;
  					case TTY_E_TIMEOUTONREAD:
            			std::cout << "Error with read timeout." << std::endl;
            			break;
					case TTY_E_ERRORONREAD:
            			std::cout << "Error with reading from file." << std::endl;
            			break;
     				default:
     					std::cout << "Something happened, but I don't know what." << std::endl;
     					break;
            	}

				return -1;

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
