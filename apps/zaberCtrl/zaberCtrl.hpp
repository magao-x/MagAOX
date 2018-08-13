/** \file zaberCtrl.hpp
  * \brief The MagAO-X Zaber Controller
  *
  * \ingroup zaberCtrl_files
  */

#ifndef zaberCtrl_hpp
#define zaberCtrl_hpp

#include <iostream>


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "magaox_git_version.h"

#include "za_serial.h"

#define ZC_CONNECTED (0)
#define ZC_NOT_CONNECTED (10)

namespace MagAOX
{
namespace app
{

class zaberStage
{
protected:
   int m_deviceAddress;
   
   int m_axisNumber{0};
   
   bool m_commandStatus {true};
   
   char m_deviceStatus {'U'};
   
   bool m_warningState {false};
   
   double m_rawPos;
   
public:
   
   int deviceAddress()
   {
      return m_deviceAddress;
   }
   
   int deviceAddress( const int & da )
   {
      m_deviceAddress = da;
      return 0;
   }
   
   int axisNumber()
   {
      return m_axisNumber;
   }
   
   int axisNumber( const int & an )
   {
      m_axisNumber = an;
      return 0;
   }
   
   bool commandStatus()
   {
      return m_commandStatus;
   }
   
   char deviceStatus()
   {
      return m_deviceStatus;
   }
   
   bool warningState()
   {
      return m_warningState;
   }
   
   int getResponse(std::string & response, const za_reply & rep)
   {
      if(rep.device_address == m_deviceAddress)
      { 
         if(rep.reply_flags[0] == 'O') m_commandStatus = true;
         else m_commandStatus = false;
               
         m_deviceStatus = rep.device_status[0];
               
         if(rep.warning_flags[0] == '-') m_warningState = false;
         else m_warningState = true;
         
         response = rep.response_data;
               
         return 0;
      }
      else return -1;
   }   
   
   int sendCommand(std::string & response, z_port port, const std::string & command)
   {
      za_send(port, command.c_str());
      
      char buff[256];
   
      while(1)
      {
         int rv = za_receive(port, buff, sizeof(buff));
         if(rv == Z_ERROR_TIMEOUT) 
         {
            //log timeout
            break; //error but just get out.
         }
         else if(rv != Z_SUCCESS)
         {
            //log error
            break;
         }
         za_reply rep;
      
         rv = za_decode(&rep, buff);
         if(rv != Z_SUCCESS)
         {
            //log error
            break;
         }
         
         if(rep.device_address == m_deviceAddress) return getResponse(response, rep);
      }
      
      response = "";
      
      return -1;
   }
      
   int updatePos(z_port port)
   {
      std::string com = "/" + mx::ioutils::convertToString(m_deviceAddress) + " ";
      com += "get pos\n";
      
      std::string response;
      
      int rv = sendCommand(response, port, com);
      
      if(rv == 0)
      {
         if( m_commandStatus )
         {
            m_rawPos = mx::ioutils::convertFromString<double>(response);
            return 0;
         }
         else
         {
            //Log rejected command

            return -1;
         }
      }
      else
      {
         //trace
         return -1;
      }
   }
};

   
class zaberCtrl : public MagAOXApp<>, public tty::usbDevice
{

   //Give the test harness access.
   friend class zaberCtrl_test;


protected:

   int m_numStages {0};

   z_port m_port;

   ///Mutex for locking device communications.
   std::mutex m_devMutex;
   
public:
   /// Default c'tor.
   zaberCtrl();

   /// D'tor, declared and defined for noexcept.
   ~zaberCtrl() noexcept
   {}

   virtual void setupConfig();

   virtual void loadConfig();


   int testConnection();

   /// Startup functions
   /** Sets up the INDI vars.
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for zaberCtrl.
   virtual int appLogic();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();


};

zaberCtrl::zaberCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void zaberCtrl::setupConfig()
{
   tty::usbDevice::setupConfig(config);

   config.add("stages.numStages", "N", "numStages", mx::argType::Required, "stages", "numStages", false,  "int", "number of stages being controlled by this connection");

}

void zaberCtrl::loadConfig()
{

   this->m_speed = B115200; //default for Zaber stages.  Will be overridden by any config setting.

   int rv = tty::usbDevice::loadConfig(config);

   if(rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND) //Ignore error if not plugged in
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }


   config(m_numStages, "stages.numStages");

   for(int i=0; i< m_numStages; ++i)
   {
      std::string newtarg = "stage" + mx::ioutils::convertToString(i);
      config.add(newtarg+".name", "", "", mx::argType::Required, newtarg, "name", false,  "", "");
      config.add(newtarg+".deviceNum", "", "", mx::argType::Required, newtarg, "deviceNum", false,  "", "");
   }

   reReadConfig();

   for(int i=0; i< m_numStages; ++i)
   {
      std::string newtarg = "stage" + mx::ioutils::convertToString(i);
      std::string name;
      int dnum;

      config(name, newtarg+".name");
      config(dnum, newtarg+".deviceNum");

      std::cerr << name << " " << dnum << "\n";
   }


}

int zaberCtrl::testConnection()
{
   std::lock_guard<std::mutex> guard(m_devMutex);
   
   if(m_port <= 0)
   {
      int rv = euidCalled();
      
      if(rv < 0)
      {
         log<software_trace_fatal>({__FILE__, __LINE__});
         state(stateCodes::FAILURE);
         return ZC_NOT_CONNECTED;
      }

      int zrv = za_connect(&m_port, m_deviceName.c_str());

      rv = euidReal();
      if(rv < 0)
      {
         log<software_trace_fatal>({__FILE__, __LINE__});
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

   int rv = za_drain(m_port);
   if(rv != Z_SUCCESS)
   {
      za_disconnect(m_port);
      m_port = 0;
      state(stateCodes::NOTCONNECTED);
      return ZC_NOT_CONNECTED; //Not an error, just no device talking.
   }

   int nwr = za_send(m_port, "/");
   if(nwr == Z_ERROR_SYSTEM_ERROR)
   {
      za_disconnect(m_port);
      m_port = 0;

      log<text_log>("Error sending test com to stages", logLevels::ERROR);
      state(stateCodes::ERROR);
      return ZC_NOT_CONNECTED;
   }

   char buffer[256];
   int stageCnt = 0;
   while(1) //We have to read all responses until timeout in case an !alert comes in
   {
      int nrd = za_receive(m_port, buffer, sizeof(buffer));
      if(nrd >= 0)
      {
         buffer[nrd] = '\0';
         std::cerr << buffer << "\n";
         //Here: process these messages -- have to be careful because we have the mutex and don't want to give it up!
         ++stageCnt;
      }
      else if (nrd != Z_ERROR_TIMEOUT)
      {
         log<text_log>("Error receiving from stages", logLevels::ERROR);
         state(stateCodes::ERROR);
         return ZC_NOT_CONNECTED;
      }
      else break; //timeout
   }
   if(stageCnt == 0)
   {
      state(stateCodes::NOTCONNECTED);
      return ZC_NOT_CONNECTED; //We aren't connected.
   }

   return ZC_CONNECTED;
}

int zaberCtrl::appStartup()
{
   if( state() == stateCodes::UNINITIALIZED )
   {
      log<text_log>( "In appStartup but in state UNINITIALIZED.", logLevels::FATAL );
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

int zaberCtrl::appLogic()
{
   
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
      if( testConnection() == ZC_CONNECTED) state(stateCodes::CONNECTED);

      if(state() == stateCodes::CONNECTED && !stateLogged())
      {
         std::stringstream logs;
         logs << "Connected to stage(s) on " << m_deviceName;
         log<text_log>(logs.str());
      }

   }

   //If we get here already more than CONNECTED, see if we're still actually connected
   if( state() > stateCodes::CONNECTED )
   {
      testConnection();
   }
   else if( state() == stateCodes::CONNECTED )
   {
      state(stateCodes::CONFIGURING);
      
      std::lock_guard<std::mutex> guard(m_devMutex);
      
      int nwr = za_send(m_port, "renumber");
      
      if(nwr == Z_ERROR_SYSTEM_ERROR)
      {
         log<text_log>("Error sending renumber to stages", logLevels::ERROR);
         state(stateCodes::ERROR);
      }

      char buffer[256];
      int stageCnt = 0;
      while(1)
      {
         int nrd = za_receive(m_port, buffer, sizeof(buffer));
         if(nrd >= 0 ) 
         {
            buffer[nrd] = '\0';
            std::cerr << buffer << "\n";
            //Need to track these? -- YES, what if something else comes in?
            ++stageCnt;
         }
         else if( nrd != Z_ERROR_TIMEOUT)
         {
            log<text_log>("Error receiving from stages", logLevels::ERROR);
            state(stateCodes::ERROR);
         }
         else break; //Timeout ok.
      }
      
      if(state() == stateCodes::CONFIGURING) //Check that nothing changed.
      {
         std::cerr << "stageCnt: " << stageCnt << "\n";
         state(stateCodes::LOGGEDIN);
      }
   } //mutex is given up at this point

   if( state() == stateCodes::ERROR )
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

      state(stateCodes::FAILURE);
      if(!stateLogged())
      {
         log<text_log>("Error NOT due to loss of USB connection.  I can't fix it myself.", logLevels::FATAL);
      }
   }




   if( state() == stateCodes::FAILURE )
   {
      return -1;
   }
   
   return 0;
}

int zaberCtrl::appShutdown()
{
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //zaberCtrl_hpp
