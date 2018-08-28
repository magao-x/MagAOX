

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
  * \todo handle timeouts gracefully -- maybe go to error, flush, disconnect, reconnect, etc.
  * \todo need username and secure password handling
  * \todo need to robustify login logic
  * \todo need to recognize signals in tty polls and not return errors, etc.
  * \todo should check if values changed and do a sendSetProperty if so (pub/sub?)
  */
class trippLitePDU : public MagAOXApp<>
{

protected:

   std::string m_deviceAddr; ///< The device address
   std::string m_devicePort; ///< The device port
   std::string m_deviceUsername;
   std::string m_devicePassFile;

   tty::telnetConn m_telnetConn; ///< The telnet connection manager

   int m_writeTimeOut {1000};  ///< The timeout for writing to the device [msec].
   int m_readTimeOut {2000}; ///< The timeout for reading from the device [msec].
   int m_outletStateDelay {5000}; ///< The maximum time to wait for an outlet to change state [msec].

   std::string m_status; ///< The device status
   float m_frequency {0}; ///< The line frequency reported by the device.
   float m_voltage {0}; ///< The line voltage reported by the device.
   float m_current {0}; ///< The current being reported by the device.
   std::vector<bool> m_outletStates; ///< The outlet states, false = off, true = on.

   ///Mutex for locking device communications.
   std::mutex m_devMutex;
public:

   /// Default c'tor.
   trippLitePDU();

   /// D'tor, declared and defined for noexcept.
   ~trippLitePDU() noexcept
   {}

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Startup functions
   /** Setsup the INDI vars.
     * Checks if the device was found during loadConfig.
     */
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
   int parsePDUStatus( std::string & strRead );

protected:

   //declare our properties
   pcf::IndiProperty m_indiStatus;
   pcf::IndiProperty m_indiFrequency;
   pcf::IndiProperty m_indiVoltage;
   pcf::IndiProperty m_indiCurrent;

   pcf::IndiProperty m_indiOutlet1;
   pcf::IndiProperty m_indiOutlet2;
   pcf::IndiProperty m_indiOutlet3;
   pcf::IndiProperty m_indiOutlet4;
   pcf::IndiProperty m_indiOutlet5;
   pcf::IndiProperty m_indiOutlet6;
   pcf::IndiProperty m_indiOutlet7;
   pcf::IndiProperty m_indiOutlet8;

   ///Common function called by the individual outlet callbacks.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeOutletState( const pcf::IndiProperty &ipRecv, ///< [in] the received INDI property
                          pcf::IndiProperty & indiOutlet,  ///< [in] the INDI property corresponding to the outlet.
                          uint8_t onum ///< [in] the number of the outlet (0-7)
                        );

public:
   INDI_NEWCALLBACK_DECL(trippLitePDU, m_indiOutlet1);
   INDI_NEWCALLBACK_DECL(trippLitePDU, m_indiOutlet2);
   INDI_NEWCALLBACK_DECL(trippLitePDU, m_indiOutlet3);
   INDI_NEWCALLBACK_DECL(trippLitePDU, m_indiOutlet4);
   INDI_NEWCALLBACK_DECL(trippLitePDU, m_indiOutlet5);
   INDI_NEWCALLBACK_DECL(trippLitePDU, m_indiOutlet6);
   INDI_NEWCALLBACK_DECL(trippLitePDU, m_indiOutlet7);
   INDI_NEWCALLBACK_DECL(trippLitePDU, m_indiOutlet8);

};

trippLitePDU::trippLitePDU() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_outletStates.resize(8,0);
   return;
}

void trippLitePDU::setupConfig()
{
   config.add("device.address", "a", "device.address", mx::argType::Required, "device", "address", false, "string", "The device address.");
   config.add("device.port", "p", "device.port", mx::argType::Required, "device", "port", false, "string", "The device port.");
   config.add("device.username", "u", "device.username", mx::argType::Required, "device", "username", false, "string", "The device login username.");
   config.add("device.passfile", "", "device.passfile", mx::argType::Required, "device", "passfile", false, "string", "The device login password file (relative to secrets dir).");

   config.add("timeouts.write", "", "timeouts.write", mx::argType::Required, "timeouts", "write", false, "int", "The timeout for writing to the device [msec]. Default = 1000");
   config.add("timeouts.read", "", "timeouts.read", mx::argType::Required, "timeouts", "read", false, "int", "The timeout for reading the device [msec]. Default = 2000");
   config.add("timeouts.outletStateDelay", "", "timeouts.outletStateDelay", mx::argType::Required, "timeouts", "outletStateDelay", false, "int", "The maximum time to wait for an outlet to change state [msec]. Default = 5000");

}

void trippLitePDU::loadConfig()
{
   config(m_deviceAddr, "device.address");
   config(m_devicePort, "device.port");
   config(m_deviceUsername, "device.username");
   config(m_devicePassFile, "device.passfile");

   config(m_writeTimeOut, "timeouts.write");
   config(m_readTimeOut, "timeouts.read");
   config(m_outletStateDelay, "timeouts.outletStateDelay");




}

int trippLitePDU::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP_NOCB(m_indiStatus, "status", pcf::IndiProperty::Text, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiStatus.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiFrequency, "frequency", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiFrequency.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiVoltage, "voltage", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiVoltage.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiCurrent, "current", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiCurrent.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP(m_indiOutlet1, "outlet1", pcf::IndiProperty::Text, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiOutlet1.add (pcf::IndiElement("state"));

   REG_INDI_NEWPROP(m_indiOutlet2, "outlet2", pcf::IndiProperty::Text, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiOutlet2.add (pcf::IndiElement("state"));

   REG_INDI_NEWPROP(m_indiOutlet3, "outlet3", pcf::IndiProperty::Text, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiOutlet3.add (pcf::IndiElement("state"));

   REG_INDI_NEWPROP(m_indiOutlet4, "outlet4", pcf::IndiProperty::Text, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiOutlet4.add (pcf::IndiElement("state"));

   REG_INDI_NEWPROP(m_indiOutlet5, "outlet5", pcf::IndiProperty::Text, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiOutlet5.add (pcf::IndiElement("state"));

   REG_INDI_NEWPROP(m_indiOutlet6, "outlet6", pcf::IndiProperty::Text, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiOutlet6.add (pcf::IndiElement("state"));

   REG_INDI_NEWPROP(m_indiOutlet7, "outlet7", pcf::IndiProperty::Text, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiOutlet7.add (pcf::IndiElement("state"));

   REG_INDI_NEWPROP(m_indiOutlet8, "outlet8", pcf::IndiProperty::Text, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiOutlet8.add (pcf::IndiElement("state"));

   state(stateCodes::NOTCONNECTED);

   return 0;
}

int trippLitePDU::appLogic()
{

   if( state() == stateCodes::UNINITIALIZED )
   {
      log<text_log>( "In appLogic but in state UNINITIALIZED.", logPrio::LOG_CRITICAL );
      return -1;
   }
   if( state() == stateCodes::INITIALIZED )
   {
      log<text_log>( "In appLogic but in state INITIALIZED.", logPrio::LOG_CRITICAL );
      return -1;
   }


   if( state() == stateCodes::NOTCONNECTED )
   {
      std::cerr << m_deviceAddr << " " << m_devicePort << "\n";
      int rv = m_telnetConn.connect(m_deviceAddr, m_devicePort);

      if(rv == 0)
      {
         state(stateCodes::CONNECTED);

         if(!stateLogged())
         {
            std::stringstream logs;
            logs << "Connected to " << m_deviceAddr << ":" << m_devicePort;
            log<text_log>(logs.str());
         }
      }
      else
      {
         if(!stateLogged())
         {
            std::stringstream logs;
            logs << "Failed to connect to " << m_deviceAddr << ":" << m_devicePort;
            log<text_log>(logs.str());
         }
      }
   }

   if( state() == stateCodes::CONNECTED )
   {
      int rv = m_telnetConn.login("localadmin", "localadmin");

      if(rv == 0)
      {
         state(stateCodes::LOGGEDIN);
      }
      else
      {
         std::cerr << rv << "\n";
         state(stateCodes::FAILURE);
         log<text_log>("login failure", logPrio::LOG_CRITICAL);
         return -1;
      }
   }

   if(state() == stateCodes::LOGGEDIN)
   {
      int rv;
      std::string strRead;
      //Scoping the mutex
      {
         std::lock_guard<std::mutex> guard(m_devMutex);

         rv = m_telnetConn.writeRead("devstatus\n", true, 1000,1000);

         strRead = m_telnetConn.m_strRead;
      }

      if(rv < 0)
      {
         if(rv == TTY_E_TIMEOUTONREAD || rv == TTY_E_TIMEOUTONREADPOLL)
         {
            log<text_log>(tty::ttyErrorString(rv), logPrio::LOG_ERROR);
            return 0;
         }
         else
         {
            state(stateCodes::NOTCONNECTED);
            log<text_log>(tty::ttyErrorString(rv), logPrio::LOG_ERROR);
            return 0;
         }
      }

      rv = parsePDUStatus( strRead);

      if(rv == 0)
      {
         std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting INDI communications.

         m_indiStatus["value"] = m_status;
         m_indiStatus.setState (pcf::IndiProperty::Ok);

         m_indiFrequency["value"] = m_frequency;
         m_indiFrequency.setState (pcf::IndiProperty::Ok);

         m_indiVoltage["value"] = m_voltage;
         m_indiVoltage.setState (pcf::IndiProperty::Ok);

         m_indiCurrent["value"] = m_current;
         m_indiCurrent.setState (pcf::IndiProperty::Ok);

         if(m_outletStates[0] == 0) m_indiOutlet1["state"] = "Off";
         else m_indiOutlet1["state"] = "On";
         m_indiOutlet1.setState(pcf::IndiProperty::Ok);

         if(m_outletStates[1] == 0) m_indiOutlet2["state"] = "Off";
         else m_indiOutlet2["state"] = "On";
         m_indiOutlet2.setState(pcf::IndiProperty::Ok);

         if(m_outletStates[2] == 0) m_indiOutlet3["state"] = "Off";
         else m_indiOutlet3["state"] = "On";
         m_indiOutlet3.setState(pcf::IndiProperty::Ok);

         if(m_outletStates[3] == 0) m_indiOutlet4["state"] = "Off";
         else m_indiOutlet4["state"] = "On";
         m_indiOutlet4.setState(pcf::IndiProperty::Ok);

         if(m_outletStates[4] == 0) m_indiOutlet5["state"] = "Off";
         else m_indiOutlet5["state"] = "On";
         m_indiOutlet5.setState(pcf::IndiProperty::Ok);

         if(m_outletStates[5] == 0) m_indiOutlet6["state"] = "Off";
         else m_indiOutlet6["state"] = "On";
         m_indiOutlet6.setState(pcf::IndiProperty::Ok);


         if(m_outletStates[6] == 0) m_indiOutlet7["state"] = "Off";
         else m_indiOutlet7["state"] = "On";
         m_indiOutlet7.setState(pcf::IndiProperty::Ok);

         std::string currSt = m_indiOutlet7["state"].get();
         if(m_outletStates[7] == 0) m_indiOutlet8["state"] = "Off";
         else m_indiOutlet8["state"] = "On";
         m_indiOutlet8.setState(pcf::IndiProperty::Ok);

         if(currSt != m_indiOutlet7["state"].get()) m_indiDriver->sendSetProperty (m_indiOutlet8);

      }
      else
      {
         std::cerr << "Parse Error: " << rv << "\n";
      }

      return 0;
   }

   state(stateCodes::FAILURE);
   log<text_log>("appLogic fell through", logPrio::LOG_CRITICAL);
   return -1;

}

int trippLitePDU::appShutdown()
{
   //don't bother
   return 0;
}

int trippLitePDU::parsePDUStatus( std::string & strRead )
{
   std::string status;
   float frequency, current, voltage;
   std::vector<bool> outletStates(8,0);

   std::string pstr = mx::ioutils::removeWhiteSpace(strRead);

   size_t st = pstr.find("Status:", 0);
   if( st == std::string::npos ) return -1;

   st = pstr.find(':', st) + 1;
   if( st == std::string::npos ) return -2;

   size_t ed = pstr.find('I', st);
   if( ed == std::string::npos ) return -3;

   status = pstr.substr(st, ed-st);

   st = pstr.find(':', ed) + 1;
   if( st == std::string::npos ) return -4;

   ed = pstr.find('V', st);
   if( ed == std::string::npos ) return -5;

   voltage = mx::ioutils::convertFromString<float>( pstr.substr(st, ed-st) );

   st = pstr.find(':', ed) + 1;
   if( st == std::string::npos ) return -6;

   ed = pstr.find('H', st);
   if( ed == std::string::npos ) return -7;

   frequency = mx::ioutils::convertFromString<float>( pstr.substr(st, ed-st) );

   st = pstr.find(':', ed) + 1;
   if( st == std::string::npos ) return -8;
   st = pstr.find(':', st) + 1;
   if( st == std::string::npos ) return -9;
   ed = pstr.find('A', st);
   if( ed == std::string::npos ) return -10;

   current = mx::ioutils::convertFromString<float>( pstr.substr(st, ed-st) );

   st = pstr.find("On:", ed) + 3;
   if( st != std::string::npos )
   {
      char ch = pstr[st];
      while(isdigit(ch) && st < pstr.size())
      {
         int onum = ch - '0';
         if(onum > 0 && onum < 9) outletStates[onum-1] = 1; //this outlet is on.

         ++st;
         if(st > pstr.size()-1) break;
         ch = pstr[st];
      }
   }

   //Ok, we're here with no errors.  Now update members.
   m_status = status;
   m_frequency = frequency;
   m_voltage = voltage;
   m_current = current;
   for(int i=0;i<8;++i) m_outletStates[i]=outletStates[i];

   return 0;

}

int trippLitePDU::changeOutletState( const pcf::IndiProperty &ipRecv,
                                     pcf::IndiProperty & indiOutlet,
                                     uint8_t onum
                                   )
{
   std::string oreq = ipRecv["state"].get<std::string>();

   if( oreq == "On" && m_outletStates[onum] == 0)
   {
      std::lock_guard<std::mutex> guard(m_devMutex);

      std::string cmd = "loadctl on -o ";
      cmd += mx::ioutils::convertToString<int>(onum+1);
      cmd += " --force\r";

      std::cerr << "Sending " << cmd << "\n";

      std::string strRead;
      int rv = m_telnetConn.writeRead( cmd, true, m_writeTimeOut, m_readTimeOut);

      strRead = m_telnetConn.m_strRead;

      std::cerr << "Received " << strRead << " (" << rv << ")\n";

      m_outletStates[onum] = 1;

      uint8_t lonum = onum + 1;   //Do this without narrowing
      log<pdu_outlet_state>({ lonum, 1});
   }
   if( oreq == "Off" && m_outletStates[onum] == 1)
   {
      std::lock_guard<std::mutex> guard(m_devMutex);

      std::cerr << "Request to turn outlet " << indiOutlet.getName() << " " << onum << " Off\n";

      std::string cmd = "loadctl off -o ";
      cmd += mx::ioutils::convertToString<int>(onum+1);
      cmd += " --force\r";

      std::cerr << "Sending " << cmd << "\n";

      std::string strRead;
      int rv = m_telnetConn.writeRead( cmd, true, m_writeTimeOut, m_readTimeOut);
      strRead = m_telnetConn.m_strRead;

      std::cerr << "Received " << strRead << " (" << rv << ")\n";

      m_outletStates[onum] = 0;

      uint8_t lonum = onum + 1; //Do this without narrowing
      log<pdu_outlet_state>({ lonum, 0});
   }

   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting INDI communications.

   if(m_outletStates[onum] == 0) indiOutlet["state"] = "Off";
   else indiOutlet["state"] = "On";

   indiOutlet.setState(pcf::IndiProperty::Ok);
   m_indiDriver->sendSetProperty (indiOutlet);

   return 0;

}

INDI_NEWCALLBACK_DEFN(trippLitePDU, m_indiOutlet1)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiOutlet1.getName())
   {
      return changeOutletState(ipRecv, m_indiOutlet1, 0);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(trippLitePDU, m_indiOutlet2)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiOutlet2.getName())
   {
      return changeOutletState(ipRecv, m_indiOutlet2, 1);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(trippLitePDU, m_indiOutlet3)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiOutlet3.getName())
   {
      return changeOutletState(ipRecv, m_indiOutlet3, 2);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(trippLitePDU, m_indiOutlet4)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiOutlet4.getName())
   {
      return changeOutletState(ipRecv, m_indiOutlet4, 3);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(trippLitePDU, m_indiOutlet5)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiOutlet5.getName())
   {
      return changeOutletState(ipRecv, m_indiOutlet5, 4);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(trippLitePDU, m_indiOutlet6)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiOutlet6.getName())
   {
      return changeOutletState(ipRecv, m_indiOutlet6, 5);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(trippLitePDU, m_indiOutlet7)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiOutlet7.getName())
   {
      return changeOutletState(ipRecv, m_indiOutlet7, 6);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(trippLitePDU, m_indiOutlet8)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiOutlet8.getName())
   {
      return changeOutletState(ipRecv, m_indiOutlet8, 7);
   }
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //trippLitePDU_hpp
