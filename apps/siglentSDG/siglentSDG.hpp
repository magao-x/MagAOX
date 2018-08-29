

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

   constexpr static double cs_MaxAmp = 10.0;
   constexpr static double cs_MaxFreq = 3700.0;
   
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

   /// Write a command to the device and get the response.  Not mutex-ed.
   /** We assume this is called after the m_indiMutex is locked.
     *
     * \returns 0 on success
     * \returns -1 on an error.  May set DISCONNECTED.
     */
   int writeRead( std::string & strRead,  ///< [out] The string responseread in
                  const std::string & command ///< [in] The command to send.
                 );
   
   /// Send the OUTP? query for a channel.
   /** This can set state to DISCONNECTED.
     * \returns 0 on success
     * \returns -1 on an error.
     */ 
   int queryOUTP( int channel /**< [in] the channel to query */);
   
   /// Send the BSWV? query for a channel.
   /** This can set state to DISCONNECTED.
     * \returns 0 on success
     * \returns -1 on an error.
     */ 
   int queryBSWV( int channel /**< [in] the channel to query */);
   
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


   /// Write a command to the device.  This locks the mutex.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int writeCommand( const std::string & commmand /**< [in] the complete command string to send to the device */);
   
   /// Send a change frequency command to the device in response to an INDI property.  This locks the mutex.
   /** 
     * The mutex is locked in the call to writeCommand.
     * 
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeFreq( int channel, ///< [in] the channel to send the command to.
                   const pcf::IndiProperty &ipRecv ///< INDI property containing the requested new frequency [Hz]
                 );
   
   /// Send a change amplitude command to the device in response to an INDI property.  This locks the mutex.
   /** 
     * The mutex is locked in the call to writeCommand.
     * 
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeAmp( int channel, ///< [in] the channel to send the command to.
                  const pcf::IndiProperty &ipRecv ///< INDI property containing the requested new amplitude [V p2p]
                );
   
protected:

   //declare our properties
   pcf::IndiProperty m_indiP_status;

   pcf::IndiProperty m_indiP_C1outp;
   pcf::IndiProperty m_indiP_C1wvtp;
   pcf::IndiProperty m_indiP_C1freq;
   pcf::IndiProperty m_indiP_C1peri;
   pcf::IndiProperty m_indiP_C1amp;
   pcf::IndiProperty m_indiP_C1ampvrms;
   pcf::IndiProperty m_indiP_C1ofst;
   pcf::IndiProperty m_indiP_C1hlev;
   pcf::IndiProperty m_indiP_C1llev;
   pcf::IndiProperty m_indiP_C1phse;

   pcf::IndiProperty m_indiP_C2outp;
   pcf::IndiProperty m_indiP_C2wvtp;
   pcf::IndiProperty m_indiP_C2freq;
   pcf::IndiProperty m_indiP_C2peri;
   pcf::IndiProperty m_indiP_C2amp;
   pcf::IndiProperty m_indiP_C2ampvrms;
   pcf::IndiProperty m_indiP_C2ofst;
   pcf::IndiProperty m_indiP_C2hlev;
   pcf::IndiProperty m_indiP_C2llev;
   pcf::IndiProperty m_indiP_C2phse;



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

   REG_INDI_NEWPROP_NOCB(m_indiP_C1wvtp, "C1wvtp", pcf::IndiProperty::Text, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C1wvtp.add (pcf::IndiElement("value"));
   
   
   REG_INDI_NEWPROP_NOCB(m_indiP_C1peri, "C1peri", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C1peri.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiP_C1ampvrms, "C1ampvrms", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C1ampvrms.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiP_C1ofst, "C1ofst", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C1ofst.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiP_C1hlev, "C1hlev", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C1hlev.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiP_C1llev, "C1llev", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C1llev.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiP_C1phse, "C1phse", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C1phse.add (pcf::IndiElement("value"));
   
   
   REG_INDI_NEWPROP(m_indiP_C2outp, "C2outp", pcf::IndiProperty::Text, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiP_C2outp.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP(m_indiP_C2freq, "C2freq", pcf::IndiProperty::Number, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiP_C2freq.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP(m_indiP_C2amp, "C2amp", pcf::IndiProperty::Number, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle);
   m_indiP_C2amp.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiP_C2wvtp, "C2wvtp", pcf::IndiProperty::Text, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C2wvtp.add (pcf::IndiElement("value"));
      
   REG_INDI_NEWPROP_NOCB(m_indiP_C2peri, "C2peri", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C2peri.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiP_C2ampvrms, "C2ampvrms", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C2ampvrms.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiP_C2ofst, "C2ofst", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C2ofst.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiP_C2hlev, "C2hlev", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C2hlev.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiP_C2llev, "C2llev", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C2llev.add (pcf::IndiElement("value"));

   REG_INDI_NEWPROP_NOCB(m_indiP_C2phse, "C2phse", pcf::IndiProperty::Number, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle);
   m_indiP_C2phse.add (pcf::IndiElement("value"));
   

   state(stateCodes::NOTCONNECTED);

   return 0;
}

int siglentSDG::appLogic()
{

 


   if( state() == stateCodes::NOTCONNECTED )
   {
      int rv = m_telnetConn.connect(m_deviceAddr, m_devicePort);

      if(rv == 0)
      {
         state(stateCodes::CONNECTED);
         m_telnetConn.noLogin();
         sleep(1);//Wait for the connection to take.

         m_telnetConn.read(">>", m_readTimeOut);

         m_telnetConn.write("\n", m_writeTimeOut);
         m_telnetConn.read(">>", m_readTimeOut);

         if(!stateLogged())
         {
            std::stringstream logs;
            logs << "Connected to " << m_deviceAddr << ":" << m_devicePort;
            log<text_log>(logs.str());
         }
         return 0;//We cycle out to give connection time to settle.
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
      if( queryOUTP(1) < 0 ) return 0; //Might be disconnected, might just need to start over.
      if( queryOUTP(2) < 0 ) return 0;
      if( queryBSWV(1) < 0 ) return 0;
      if( queryBSWV(2) < 0 ) return 0;
      
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

int siglentSDG::writeRead( std::string & strRead,
                           const std::string & command
                         )
{
   int rv;
   //Scoping the mutex
   
   rv = m_telnetConn.writeRead(command, false, m_writeTimeOut, m_readTimeOut);
   strRead = m_telnetConn.m_strRead;

   if(rv < 0)
   {
      log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      state(stateCodes::NOTCONNECTED);
      return -1;
   }

   //Clear the newline
   rv = m_telnetConn.write("\n", m_writeTimeOut);
   if(rv < 0)
   {
      log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }
   
   rv = m_telnetConn.read(">>", m_readTimeOut);
   if(rv < 0)
   {
      log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }
   return 0;
   
}

std::string makeCommand( int channel,
                         const std::string afterColon
                       )
{
   std::string command = "C";
   command += mx::ioutils::convertToString<int>(channel);
   command += ":";
   command += afterColon;
   command += "\r\n";
   
   return command;
}

int siglentSDG::queryOUTP( int channel )
{
   int rv;

   if(channel < 1 || channel > 2) return -1;
   
   std::string strRead;

   std::string com = makeCommand(channel, "OUTP?");
   
   // Do this right away to avoid a different thread updating something after we get it.
   // Note that it's dangerous to have this before writeRead because there's another mutex in there.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting INDI communications.

   rv = writeRead( strRead, com);
   
   if(rv < 0)
   {
      log<text_log>("Error on OUTP? for channel " + mx::ioutils::convertToString<int>(channel), logPrio::LOG_ERROR);
      return -1;
   }
   
   int resp_channel;
   int resp_output;
      
   rv = parseOUTP(resp_channel, resp_output, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
      }

      std::string ro;
      if(resp_output > 0) ro = "On";
      else if(resp_output == 0 ) ro = "Off";
      else ro = "UNK";
      
      if(channel == 1) updateIfChanged(m_indiP_C1outp, "value", ro);
      else if(channel == 2) updateIfChanged(m_indiP_C2outp, "value", ro);
   }
   else
   {
      log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
      return -1;
   }

   return 0;
}

int siglentSDG::queryBSWV( int channel )
{
   int rv;

   if(channel < 1 || channel > 2) return -1;
   
   std::string strRead;

   std::string com = makeCommand(channel, "BSWV?");

   // Do this right away to avoid a different thread updating something after we get it.
   // Note that it's dangerous to have this before writeRead because there's another mutex in there.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting INDI communications.
   
   rv = writeRead( strRead, com);
   
   
   if(rv < 0)
   {
      log<text_log>("Error on BSWV? for channel " + mx::ioutils::convertToString<int>(channel), logPrio::LOG_ERROR);
      return -1;
   }
   
   int resp_channel;
   std::string resp_wvtp;
   double resp_freq, resp_peri, resp_amp, resp_ampvrms, resp_ofst, resp_hlev, resp_llev, resp_phse;
      
   rv = parseBSWV(resp_channel, resp_wvtp, resp_freq, resp_peri, resp_amp, resp_ampvrms, resp_ofst, resp_hlev, resp_llev, resp_phse, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
      }

      if(channel == 1) 
      {
         updateIfChanged(m_indiP_C1wvtp, "value", resp_wvtp);
         updateIfChanged(m_indiP_C1freq, "value", resp_freq);
         updateIfChanged(m_indiP_C1peri, "value", resp_peri);
         updateIfChanged(m_indiP_C1amp, "value", resp_amp);
         updateIfChanged(m_indiP_C1ampvrms, "value", resp_ampvrms);
         updateIfChanged(m_indiP_C1ofst, "value", resp_ofst);
         updateIfChanged(m_indiP_C1hlev, "value", resp_hlev);
         updateIfChanged(m_indiP_C1llev, "value", resp_llev);
         updateIfChanged(m_indiP_C1phse, "value", resp_phse);
      }
      else if(channel == 2)
      {
         updateIfChanged(m_indiP_C2wvtp, "value", resp_wvtp);
         updateIfChanged(m_indiP_C2freq, "value", resp_freq);
         updateIfChanged(m_indiP_C2peri, "value", resp_peri);
         updateIfChanged(m_indiP_C2amp, "value", resp_amp);
         updateIfChanged(m_indiP_C2ampvrms, "value", resp_ampvrms);
         updateIfChanged(m_indiP_C2ofst, "value", resp_ofst);
         updateIfChanged(m_indiP_C2hlev, "value", resp_hlev);
         updateIfChanged(m_indiP_C2llev, "value", resp_llev);
         updateIfChanged(m_indiP_C2phse, "value", resp_phse);
      }
   }
   else
   {
      log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
      return -1;
   }

   return 0;
}

int siglentSDG::parseOUTP( int & channel,
                           int & output,
                           const std::string & strRead
                         )
{
   std::vector<std::string> v;

   mx::ioutils::parseStringVector(v, strRead, ":, ");

   if(v[1] != "OUTP") return -1;

   if(v[0][0] != 'C') return -2;
   if(v[0].size() < 2) return -3;
   channel = mx::ioutils::convertFromString<int>(v[0].substr(1, v[0].size()-1));

   if(v[2] == "OFF") output = 0;
   else if(v[2] == "ON") output = 1;
   else
   {
      return -4;
   }
   return 0;
}



int siglentSDG::parseBSWV( int & channel,
                           std::string & wvtp,
                           double & freq,
                           double & peri,
                           double & amp,
                           double & ampvrms,
                           double & ofst,
                           double & hlev,
                           double & llev,
                           double & phse,
                           const std::string & strRead
                         )
{
   std::vector<std::string> v;

   mx::ioutils::parseStringVector(v, strRead, ":, ");

   //for(size_t i=0; i<v.size();++i) std::cerr << v[i] << "\n";

   if(v.size()!= 20) return -1;

   if(v[1] != "BSWV") return -2;

   if(v[0][0] != 'C') return -3;
   if(v[0].size() < 2) return -4;
   channel = mx::ioutils::convertFromString<int>(v[0].substr(1, v[0].size()-1));

   if(v[2] != "WVTP") return -5;
   wvtp = v[3];

   if(wvtp != "SINE") return -6;
   
   if(v[4] != "FRQ") return -7;
   freq = mx::ioutils::convertFromString<double>(v[5]);

   if(v[6] != "PERI") return -8;
   peri = mx::ioutils::convertFromString<double>(v[7]);

   if(v[8] != "AMP") return -9;
   amp = mx::ioutils::convertFromString<double>(v[9]);

   if(v[10] != "AMPVRMS") return -10;
   ampvrms = mx::ioutils::convertFromString<double>(v[11]);

   if(v[12] != "OFST") return -11;
   ofst = mx::ioutils::convertFromString<double>(v[13]);

   if(v[14] != "HLEV") return -12;
   hlev = mx::ioutils::convertFromString<double>(v[15]);

   if(v[16] != "LLEV") return -13;
   llev = mx::ioutils::convertFromString<double>(v[17]);

   if(v[18] != "PHSE") return -14;
   phse = mx::ioutils::convertFromString<double>(v[19]);

   return 0;
}

int siglentSDG::writeCommand( const std::string & command )
{
   //Make sure we don't change things while other things are being updated.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.
    
   int rv = m_telnetConn.write(command, m_writeTimeOut);
   if(rv < 0)
   {
      log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }
      
   //Clear the newline
   rv = m_telnetConn.write("\n", m_writeTimeOut);
   if(rv < 0)
   {
      log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }
   
   rv = m_telnetConn.read(">>", m_readTimeOut);
   if(rv < 0)
   {
      log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }
   
   return 0;
}
   
int siglentSDG::changeFreq( int channel,
                            const pcf::IndiProperty &ipRecv
                          )
{
   if(channel < 1 || channel > 2) return -1;
   
   double newFreq;
   try
   {
      newFreq = ipRecv["value"].get<double>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }
      
   ///\todo logs here
   if(newFreq > cs_MaxFreq)
   {
      newFreq = cs_MaxFreq;
   }
      
   std::string afterColon = "BSWV FRQ," + mx::ioutils::convertToString<double>(newFreq);
   std::string command = makeCommand(channel, afterColon);
      
   int rv = writeCommand(command);
   
   if(rv < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return -1;
   }
   
   return 0;
}

int siglentSDG::changeAmp( int channel,
                           const pcf::IndiProperty &ipRecv
                         )
{
   if(channel < 1 || channel > 2) return -1;
   
   double newAmp;
   try
   {
      newAmp = ipRecv["value"].get<double>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }
      
      
   ///\todo logs here
   
   if(newAmp > cs_MaxAmp)
   {
      newAmp = cs_MaxAmp;
   }
   
   std::string afterColon = "BSWV AMP," + mx::ioutils::convertToString<double>(newAmp);
   std::string command = makeCommand(channel, afterColon);
      
   int rv = writeCommand(command);
   
   if(rv < 0)
   {
      log<software_error>({__FILE__, __LINE__});
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
      return changeFreq(1, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1amp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C1amp.getName())
   {
      return changeAmp(1, ipRecv);
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
      return changeFreq(2, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2amp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C2amp.getName())
   {
      return changeAmp(2, ipRecv);
   }
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //siglentSDG_hpp
