

#ifndef siglentSDG_hpp
#define siglentSDG_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "magaox_git_version.h"

namespace MagAOX
{
namespace app
{

/** MagAO-X application to control a Siglent SDG series function generator
  *
  * \todo handle timeouts gracefully -- maybe go to error, flush, disconnect, reconnect, etc.
  * \todo need username and secure password handling
  * \todo need to robustify login logic
  * \todo need to recognize signals in tty polls and not return errors, etc.
  * \todo should check if values changed and do a sendSetProperty if so (pub/sub?)
  */
class siglentSDG : public MagAOXApp<>
{

protected:

   std::string m_deviceAddr; ///< The device address
   std::string m_devicePort; ///< The device port

   tty::telnetConn m_telnetConn; ///< The telnet connection manager

   int m_writeTimeOut {1000};  ///< The timeout for writing to the device [msec].
   int m_readTimeOut {1000}; ///< The timeout for reading from the device [msec].

   std::string m_status; ///< The device status

   int m_C1outp {0}; ///< The output status channel 1
   double m_C1frequency {0}; ///< The output frequency of channel 1
   double m_C1amp {0}; ///< The peak-2-peak voltage of channel 1


   int m_C2outp {0}; ///<  The output status channel 2
   double m_C2frequency {0}; ///< The output frequency of channel 2
   double m_C2vpp {0}; ///< The peak-2-peak voltage of channel 2

   ///Mutex for locking device communications.
   std::mutex m_devMutex;
public:

   /// Default c'tor.
   siglentSDG();

   /// D'tor, declared and defined for noexcept.
   ~siglentSDG() noexcept
   {}

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Startup functions
   /** Setsup the INDI vars.
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for the Siglent SDG
   virtual int appLogic();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();

   /// Parse the SDG response to the OUTP query
   /**
     * Example: C1:OUTP OFF,LOAD,HZ,PLRT,NOR
     *
     * \returns 0 on success
     * \returns \<0 on error, with value indicating location of error.
     */
   int parseOUTP( int & channel,
                  int & output,
                  const std::string & strRead 
                );

   /// Parse the SDG response to the BSWV query
   /**
     * Example: C1:BSWV WVTP,SINE,FRQ,10HZ,PERI,0.1S,AMP,2V,AMPVRMS,0.707Vrms,OFST,0V,HLEV,1V,LLEV,-1V,PHSE,0
     *
     * \returns 0 on success
     * \returns \<0 on error, with value indicating location of error.
     */
   int parseBSWV( int & channel,
                  std::string & wvtp,
                  double & freq,
                  double & peri,
                  double & amp,
                  double & amprs,
                  double & ofst,
                  double & hlev,
                  double & llev,
                  double & phse,
                  const std::string & strRead 
                );



protected:

   //declare our properties
   pcf::IndiProperty m_indiP_status;

   pcf::IndiProperty m_indiP_C1outp;
   pcf::IndiProperty m_indiP_C1freq;
   pcf::IndiProperty m_indiP_C1amp;

   pcf::IndiProperty m_indiP_C2outp;
   pcf::IndiProperty m_indiP_C2freq;
   pcf::IndiProperty m_indiP_C2amp;



public:
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1outp);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1freq);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1amp);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2outp);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2freq);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2amp);

};

siglentSDG::siglentSDG() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_telnetConn.m_prompt = "\n";
   return;
}

void siglentSDG::setupConfig()
{
   config.add("device.address", "a", "device.address", mx::argType::Required, "device", "address", false, "string", "The device address.");
   config.add("device.port", "p", "device.port", mx::argType::Required, "device", "port", false, "string", "The device port.");


   config.add("timeouts.write", "", "timeouts.write", mx::argType::Required, "timeouts", "write", false, "int", "The timeout for writing to the device [msec]. Default = 1000");
   config.add("timeouts.read", "", "timeouts.read", mx::argType::Required, "timeouts", "read", false, "int", "The timeout for reading the device [msec]. Default = 2000");


}

void siglentSDG::loadConfig()
{
   config(m_deviceAddr, "device.address");
   config(m_devicePort, "device.port");

   config(m_writeTimeOut, "timeouts.write");
   config(m_readTimeOut, "timeouts.read");


}

int siglentSDG::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP_NOCB(m_indiP_status, "status", pcf::IndiProperty::Text, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_status.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP(m_indiP_C1outp, "C1outp", pcf::IndiProperty::Text, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiP_C1outp.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP(m_indiP_C1freq, "C1freq", pcf::IndiProperty::Number, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiP_C1freq.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP(m_indiP_C1amp, "C1amp", pcf::IndiProperty::Number, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiP_C1amp.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP(m_indiP_C2outp, "C2outp", pcf::IndiProperty::Text, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiP_C2outp.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP(m_indiP_C2freq, "C2freq", pcf::IndiProperty::Number, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiP_C2freq.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP(m_indiP_C2amp, "C2amp", pcf::IndiProperty::Number, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiP_C2amp.add (pcf::IndiElement("value"));

   state(stateCodes::NOTCONNECTED);

   return 0;
}

int siglentSDG::appLogic()
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
         m_telnetConn.noLogin();
         m_telnetConn.write("\n", m_writeTimeOut);
         m_telnetConn.read(">>", m_readTimeOut);

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
         return 0;
      }
   }

   if(state() == stateCodes::CONNECTED)
   {
      int rv;
      std::string strRead;
      //Scoping the mutex
      {
         std::lock_guard<std::mutex> guard(m_devMutex);

         std::cerr << "Connected, sending OUTP\n";
         rv = m_telnetConn.writeRead("C1:OUTP?\r\n", false, 1000,1000);
         strRead = m_telnetConn.m_strRead;
         m_telnetConn.write("\n", m_writeTimeOut);
         m_telnetConn.read(">>", m_readTimeOut);
      }

      if(rv < 0)
      {
         state(stateCodes::NOTCONNECTED);
         log<text_log>(tty::ttyErrorString(rv), logPrio::LOG_ERROR);
         return 0;
      }

      std::cerr << "C1:OUTP: " << strRead << "\n";
      rv = 0;
      //rv = parsePDUStatus( strRead);


      if(rv == 0)
      {
         std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting INDI communications.

         //m_indiStatus["value"] = m_status;
         //m_indiStatus.setState (pcf::IndiProperty::Ok);



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

int siglentSDG::appShutdown()
{
   //don't bother
   return 0;
}

int siglentSDG::parseOUTP( int & channel,
                           int & output,
                           const std::string & strRead 
                         )
{
   std::vector<std::string> v;
   
   mx::ioutils::parseStringVector(v, strRead);
      
   if(v[0][0] != 'C') return -1;
   size_t tok = v[0].find(':',1);
   
   if(tok < 2 || tok == std::string::npos) return -1;
   
   channel = mx::ioutils::convertFromString<int>(v[0].substr(1, tok-1));
   
   size_t tok2 = v[0].find(' ', tok);
   
   if(tok2 == std::string::npos) return -1;
   
   std::string o = v[0].substr(tok2+1, v[0].size()-tok2-1);
   
   std::cerr << o << "\n";
   if(o == "OFF") output = 0;
   else if(o == "ON") output = 1;
   else 
   {
      return -1;
   }
   return 0;
}
   


INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1outp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C1outp.getName())
   {
      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1freq)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C1freq.getName())
   {
      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1amp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C1amp.getName())
   {
      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2outp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C2outp.getName())
   {
      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2freq)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C2freq.getName())
   {
      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2amp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C2amp.getName())
   {
      return 0;
   }
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //siglentSDG_hpp
