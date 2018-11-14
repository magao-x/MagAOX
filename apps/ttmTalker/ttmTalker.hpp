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
			int rv = tty::usbDevice::loadConfig(config);

   			if(rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND) //Ignore error if not plugged in
   			{
      			log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
			}
		}

		int ttmTalker::appStartup()
		{
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
         			}
      			}

   			}

			if( state() == stateCodes::NOTCONNECTED )
		   	{
		      	//if( testConnection() == ZC_CONNECTED) state(stateCodes::CONNECTED);

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

   } //namespace app
} //namespace MagAOX

#endif //ttmTalker_hpp
