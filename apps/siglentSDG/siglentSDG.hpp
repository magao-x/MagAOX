

#ifndef siglentSDG_hpp
#define siglentSDG_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "siglentSDG_parsers.hpp"

namespace MagAOX
{
namespace app
{

/** MagAO-X application to control a Siglent SDG series function generator
  *
  * \todo need to recognize signals in tty polls and not return errors, etc.
  * \todo need to implement an onDisconnect() to update values to unknown indicators.
  * \todo need a frequency-dependent max amp facility.
  *
  */
class siglentSDG : public MagAOXApp<>
{

   constexpr static double cs_MaxAmp = 0.87;
   constexpr static double cs_MaxOfst = 10.0;
   constexpr static double cs_MaxFreq = 1000.0;

protected:

   /** \name Configurable Parameters
     * @{
     */

   std::string m_deviceAddr; ///< The device address
   std::string m_devicePort; ///< The device port

   double m_bootDelay {10}; ///< Time in seconds it takes the device to boot.

   int m_writeTimeOut {1000};  ///< The timeout for writing to the device [msec].
   int m_readTimeOut {1000}; ///< The timeout for reading from the device [msec].

   double m_C1setVoltage {5.0}; ///< the set position voltage of Ch. 1.
   double m_C2setVoltage {5.0}; ///< the set position voltage of Ch. 2.

   ///@}

   tty::telnetConn m_telnetConn; ///< The telnet connection manager

   uint8_t m_C1outp {0}; ///< The output status channel 1
   double m_C1frequency {0}; ///< The output frequency of channel 1
   double m_C1vpp {0}; ///< The peak-2-peak voltage of channel 1
   double m_C1ofst {0}; ///< The offset voltage of channel 1
   double m_C1phse {0}; ///< The phase of channel 1
   std::string m_C1wvtp; ///< The wave type of channel 1

   uint8_t m_C2outp {0}; ///<  The output status channel 2
   double m_C2frequency {0}; ///< The output frequency of channel 2
   double m_C2vpp {0}; ///< The peak-2-peak voltage of channel 2
   double m_C2ofst {0}; ///< The offset voltage of channel 2
   double m_C2phse {0}; ///< The phase of channel 2
   std::string m_C2wvtp; ///< The wave type of channel 2

   int m_changeToLog {1}; ///< Flag telling the main loop that a log entry should be made

private:

   bool m_poweredOn {false};

   double m_powerOnCounter {0}; ///< Counts the number of loops since power-on, used to control logging of connect failures.

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

   /// Implementation of the on-power-off FSM logic
   virtual int onPowerOff();

   /// Implementation of the while-powered-off FSM
   virtual int whilePowerOff();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();

   /// Write a command to the device and get the response.  Not mutex-ed.
   /** We assume this is called after the m_indiMutex is locked.
     *
     * \returns 0 on success
     * \returns -1 on an error.  May set DISCONNECTED.
     */
   int writeRead( std::string & strRead,  ///< [out] The string responseread in
                  const std::string & command ///< [in] The command to send.
                );

    /// Write a command to the device.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int writeCommand( const std::string & commmand /**< [in] the complete command string to send to the device */);

   /// Send the MDWV? query and get the response state.
   /** This does not update internal state.
     *
     * \returns 0 on success
     * \returns -1 on an error.
     */
   int queryMDWV( std::string & state, ///< [out] the MDWV state, ON or OFF
                  int channel ///< [in] the channel to query
                );

   /// Send the SWWV? query and get the response state.
   /** This does not update internal state.
     *
     * \returns 0 on success
     * \returns -1 on an error.
     */
   int querySWWV( std::string & state, ///< [out] the SWWV state, ON or OFF
                  int channel ///< [in] the channel to query
                );

   /// Send the BTWV? query and get the response state.
   /** This does not update internal state.
     *
     * \returns 0 on success
     * \returns -1 on an error.
     */
   int queryBTWV( std::string & state,  ///< [out] the BTWV state, ON or OFF
                  int channel ///< [in] the channel to query
                );

   /// Send the ARWV? query and get the response index.
   /** This does not update internal state.
     *
     * \returns 0 on success
     * \returns -1 on an error.
     */
   int queryARWV( int & index,  ///< [out] the ARWV index
                  int channel ///< [in] the channel to query
                );

   /// Send the BSWV? query for a channel.
   /** This updates member variables and INDI.
     *
     * \returns 0 on success
     * \returns -1 on an error.
     */
   int queryBSWV( int channel  /** < [in] the channel to query */ );


   /// Check the setup is correct and safe for PI TTM control.
   /**
     * \returns 0 if the fxn gen is setup for safe operation
     * \returns 1 if a non-normal setup is detected.
     * \returns -1 on an error, e.g. comms or parsing.
     */
   int checkSetup();

   /// Normalize the setup, called during connection if checkSetup shows a problem, or on power-up.
   int normalizeSetup();

   /// Send the OUTP? query for a channel.
   /**
     * \returns 0 on success
     * \returns -1 on an error.
     */
   int queryOUTP( int channel /**< [in] the channel to query */);

   /// Change the output status (on/off) of one channel.
   /**
     * \returns 0 on success
     * \returns -1 on error.
     */
   int changeOutp( int channel,                ///< [in] the channel to send the command to.
                   const std::string & newOutp ///< [in] The requested output state [On/Off]
                 );

   /// Change the output status (on/off) of one channel in response to an INDI property. This locks the mutex.
   /**
     * \returns 0 on success
     * \returns -1 on error.
     */
   int changeOutp( int channel,                    ///< [in] the channel to send the command to.
                   const pcf::IndiProperty &ipRecv ///< [in] INDI property containing the requested output state [On/Off]
                 );

   /// Send a change frequency command to the device.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeFreq( int channel,   ///< [in] the channel to send the command to.
                   double newFreq ///< [in] The requested new frequency [Hz]
                 );

   /// Send a change frequency command to the device in response to an INDI property.  This locks the mutex.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeFreq( int channel,                    ///< [in] the channel to send the command to.
                   const pcf::IndiProperty &ipRecv ///< [in] INDI property containing the requested new frequency [Hz]
                 );

   /// Send a change amplitude command to the device.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeAmp( int channel,  ///< [in] the channel to send the command to.
                  double newAmp ///< [in] The requested new amplitude [V p2p]
                );

   /// Send a change amplitude command to the device in response to an INDI property.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeAmp( int channel,                    ///< [in] the channel to send the command to.
                  const pcf::IndiProperty &ipRecv ///< [in] INDI property containing the requested new amplitude [V p2p]
                );

   /// Send a change offset command to the device.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeOfst( int channel,  ///< [in] the channel to send the command to.
                  double newOfst ///< [in] The requested new offset [V p2p]
                );

   /// Send a change offset command to the device in response to an INDI property.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeOfst( int channel,                    ///< [in] the channel to send the command to.
                   const pcf::IndiProperty &ipRecv ///< [in] INDI property containing the requested new offset [V p2p]
                 );

   /// Send a change phase command to the device.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changePhse( int channel,  ///< [in] the channel to send the command to.
                   double newPhse ///< [in] The requested new phase [deg]
                 );

   /// Send a change phase command to the device in response to an INDI property.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changePhse( int channel,                    ///< [in] the channel to send the command to.
                   const pcf::IndiProperty &ipRecv ///< [in] INDI property containing the requested new phase [deg]
                 );

   /// Send a change wavetype command to the device.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeWvtp( int channel,  ///< [in] the channel to send the command to.
                   const std::string & newWvtp ///< [in] The requested new wavetype
                 );

   /// Send a change wavetype command to the device in response to an INDI property.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeWvtp( int channel,                    ///< [in] the channel to send the command to.
                   const pcf::IndiProperty &ipRecv ///< [in] INDI property containing the requested new wavetype
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
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1ofst);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1phse);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1wvtp);

   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2outp);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2freq);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2amp);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2ofst);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2phse);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2wvtp);

};

inline
siglentSDG::siglentSDG() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   m_telnetConn.m_prompt = "\n";
   return;
}

inline
void siglentSDG::setupConfig()
{
   config.add("device.address", "a", "device.address", argType::Required, "device", "address", false, "string", "The device address.");
   config.add("device.port", "p", "device.port", argType::Required, "device", "port", false, "string", "The device port.");


   config.add("timeouts.write", "", "timeouts.write", argType::Required, "timeouts", "write", false, "int", "The timeout for writing to the device [msec]. Default = 1000");
   config.add("timeouts.read", "", "timeouts.read", argType::Required, "timeouts", "read", false, "int", "The timeout for reading the device [msec]. Default = 2000");
}

inline
void siglentSDG::loadConfig()
{
   config(m_deviceAddr, "device.address");
   config(m_devicePort, "device.port");

   config(m_writeTimeOut, "timeouts.write");
   config(m_readTimeOut, "timeouts.read");
}

inline
int siglentSDG::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP_NOCB(m_indiP_status, "status", pcf::IndiProperty::Text);
   m_indiP_status.add (pcf::IndiElement("value"));
   m_indiP_status["value"].set(0);

   REG_INDI_NEWPROP(m_indiP_C1outp, "C1outp", pcf::IndiProperty::Text);
   m_indiP_C1outp.add (pcf::IndiElement("value"));
   m_indiP_C1outp["value"].set("");

   REG_INDI_NEWPROP(m_indiP_C1freq, "C1freq", pcf::IndiProperty::Number);
   m_indiP_C1freq.add (pcf::IndiElement("value"));
   m_indiP_C1freq["value"].set(0);

   REG_INDI_NEWPROP(m_indiP_C1amp, "C1amp", pcf::IndiProperty::Number);
   m_indiP_C1amp.add (pcf::IndiElement("value"));
   m_indiP_C1amp["value"].set(0);

   REG_INDI_NEWPROP(m_indiP_C1ofst, "C1ofst", pcf::IndiProperty::Number);
   m_indiP_C1ofst.add (pcf::IndiElement("value"));
   m_indiP_C1ofst["value"].set(0);

   REG_INDI_NEWPROP(m_indiP_C1phse, "C1phse", pcf::IndiProperty::Number);
   m_indiP_C1phse.add (pcf::IndiElement("value"));
   m_indiP_C1phse["value"].set(0);

   REG_INDI_NEWPROP(m_indiP_C1wvtp, "C1wvtp", pcf::IndiProperty::Text);
   m_indiP_C1wvtp.add (pcf::IndiElement("value"));
   m_indiP_C1wvtp["value"].set("");

   REG_INDI_NEWPROP_NOCB(m_indiP_C1peri, "C1peri", pcf::IndiProperty::Number);
   m_indiP_C1peri.add (pcf::IndiElement("value"));
   m_indiP_C1peri["value"].set(0);

   REG_INDI_NEWPROP_NOCB(m_indiP_C1ampvrms, "C1ampvrms", pcf::IndiProperty::Number);
   m_indiP_C1ampvrms.add (pcf::IndiElement("value"));
   m_indiP_C1ampvrms["value"].set(0);

   REG_INDI_NEWPROP_NOCB(m_indiP_C1hlev, "C1hlev", pcf::IndiProperty::Number);
   m_indiP_C1hlev.add (pcf::IndiElement("value"));
   m_indiP_C1hlev["value"].set(0);

   REG_INDI_NEWPROP_NOCB(m_indiP_C1llev, "C1llev", pcf::IndiProperty::Number);
   m_indiP_C1llev.add (pcf::IndiElement("value"));
   m_indiP_C1llev["value"].set(0);

   REG_INDI_NEWPROP_NOCB(m_indiP_C1phse, "C1phse", pcf::IndiProperty::Number);
   m_indiP_C1phse.add (pcf::IndiElement("value"));
   m_indiP_C1phse["value"].set(0);

   REG_INDI_NEWPROP(m_indiP_C2outp, "C2outp", pcf::IndiProperty::Text);
   m_indiP_C2outp.add (pcf::IndiElement("value"));
   m_indiP_C2outp["value"].set("");

   REG_INDI_NEWPROP(m_indiP_C2freq, "C2freq", pcf::IndiProperty::Number);
   m_indiP_C2freq.add (pcf::IndiElement("value"));
   m_indiP_C2freq["value"].set(0);

   REG_INDI_NEWPROP(m_indiP_C2amp, "C2amp", pcf::IndiProperty::Number);
   m_indiP_C2amp.add (pcf::IndiElement("value"));
   m_indiP_C2amp["value"].set(0);

   REG_INDI_NEWPROP(m_indiP_C2ofst, "C2ofst", pcf::IndiProperty::Number);
   m_indiP_C2ofst.add (pcf::IndiElement("value"));
   m_indiP_C2ofst["value"].set(0);

   REG_INDI_NEWPROP(m_indiP_C2phse, "C2phse", pcf::IndiProperty::Number);
   m_indiP_C2phse.add (pcf::IndiElement("value"));
   m_indiP_C2phse["value"].set(0);

   REG_INDI_NEWPROP(m_indiP_C2wvtp, "C2wvtp", pcf::IndiProperty::Text);
   m_indiP_C2wvtp.add (pcf::IndiElement("value"));
   m_indiP_C2wvtp["value"].set("");

   REG_INDI_NEWPROP_NOCB(m_indiP_C2peri, "C2peri", pcf::IndiProperty::Number);
   m_indiP_C2peri.add (pcf::IndiElement("value"));
   m_indiP_C2peri["value"].set(0);

   REG_INDI_NEWPROP_NOCB(m_indiP_C2ampvrms, "C2ampvrms", pcf::IndiProperty::Number);
   m_indiP_C2ampvrms.add (pcf::IndiElement("value"));
   m_indiP_C2ampvrms["value"].set(0);

   REG_INDI_NEWPROP_NOCB(m_indiP_C2hlev, "C2hlev", pcf::IndiProperty::Number);
   m_indiP_C2hlev.add (pcf::IndiElement("value"));
   m_indiP_C2hlev["value"].set(0);

   REG_INDI_NEWPROP_NOCB(m_indiP_C2llev, "C2llev", pcf::IndiProperty::Number);
   m_indiP_C2llev.add (pcf::IndiElement("value"));
   m_indiP_C2llev["value"].set(0);

   REG_INDI_NEWPROP_NOCB(m_indiP_C2phse, "C2phse", pcf::IndiProperty::Number);
   m_indiP_C2phse.add (pcf::IndiElement("value"));
   m_indiP_C2phse["value"].set(0);

   return 0;
}

inline
int siglentSDG::appLogic()
{

   if( state() == stateCodes::POWERON )
   {
      m_poweredOn = true; //So we reset the device.

      state(stateCodes::NOTCONNECTED);
      m_powerOnCounter = 0;
   }

   //If we enter this loop in state ERROR, we wait 1 sec and then check power state.
   if( state() == stateCodes::ERROR )
   {
      sleep(1);

      //This allows for the case where the device powers off causing a comm error
      //But we haven't gotten the update from the power controller before going through
      //the main loop after the error.
      if(m_powerState < 1)
      {
         return 0;
      }
   }
   
   if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::ERROR )
   {
      int rv = m_telnetConn.connect(m_deviceAddr, m_devicePort);

      if(rv == 0)
      {
         ///\todo the connection process in siglentSDG is a total hack.  Figure out why this is needed to clear the channel, especially on a post-poweroff/on reconnect.

         //The sleeps here seem to be necessary to make sure there is a good
         //comm with device.  Probably a more graceful way.
         state(stateCodes::CONNECTED);
         m_telnetConn.noLogin();
         //sleep(1);//Wait for the connection to take.

         m_telnetConn.read(">>", m_readTimeOut);

         m_telnetConn.m_strRead.clear();
         m_telnetConn.write("\n", m_writeTimeOut);

         m_telnetConn.read(">>", m_readTimeOut);

         int n = 0;
         while( m_telnetConn.m_strRead != ">>")
         {
            if(n>9)
            {
               log<software_critical>({__FILE__, __LINE__});
               return -1;
            }
            m_telnetConn.write("\n", m_writeTimeOut);
            sleep(1);
            m_telnetConn.read(">>", m_readTimeOut);
            ++n;
         }

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

         if(m_powerOnCounter > m_bootDelay && !stateLogged())
         {
            std::stringstream logs;
            logs << "Failed to connect to " << m_deviceAddr << ":" << m_devicePort;
            log<text_log>(logs.str());
         }

         m_powerOnCounter += 1 + m_loopPause/1e9;

         return 0;
      }
   }

   if(state() == stateCodes::CONNECTED )
   {
      //Do Initial Checks Here.
      std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);
      if(lock.owns_lock())
      {

         if(m_poweredOn)
         {
            //This means we need to do the power-on setup.
            if(normalizeSetup() < 0 )
            {
               log<software_critical>({__FILE__, __LINE__});
               return -1;
            }

            m_poweredOn = false;
         }


         int cs = checkSetup();

         if(cs < 0) return 0; //This means we aren't really connected yet.

         int rv;

         rv = queryBSWV(1);

         if( rv < 0 )
         {
            if(rv != SDG_PARSEERR_WVTP ) return 0; //This means we aren't really connected yet.

            cs = 1; //Trigger normalizeSetup
         }

         rv = queryBSWV(2);

         if( rv < 0 )
         {
            if(rv != SDG_PARSEERR_WVTP ) return 0; //This means we aren't really connected yet.

            cs = 1; //Trigger normalizeSetup
         }

         if(cs > 0)
         {
            log<text_log>("Failed setup check, normalizing setup.", logPrio::LOG_NOTICE);
            if(normalizeSetup() < 0)
            {
               log<software_critical>({__FILE__, __LINE__});
               return -1;
            }

            return 0;
         }

         if( queryOUTP(1) < 0 ) return 0; //This means we aren't really connected yet.
         if( queryOUTP(2) < 0 ) return 0; //This means we aren't really connected yet.



         if( m_C1outp == 1 || m_C2outp == 1)
         {
            state(stateCodes::OPERATING);
         }
         else
         {
            state(stateCodes::READY);
         }

         int start = m_changeToLog;
         log<fxngen_params>({m_C1outp, m_C1frequency, m_C1vpp, m_C1ofst, m_C1phse, m_C1wvtp,
                             m_C2outp, m_C2frequency, m_C2vpp, m_C2ofst, m_C2phse, m_C2wvtp});

         m_changeToLog -= start;
         if(m_changeToLog < 0) m_changeToLog = 0;

      }
      else
      {
         log<text_log>("Could not get mutex after connecting.", logPrio::LOG_CRITICAL);
         return -1;
      }
   }

   if(state() == stateCodes::READY || state() == stateCodes::OPERATING)
   {
      // Do this right away to avoid a different thread updating something after we get it.
      std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);
      if(lock.owns_lock())
      {
         int cs = checkSetup();

         if(cs < 0)
         {
            if(m_powerState > 0 && !m_shutdown)
            {
               log<software_error>({__FILE__, __LINE__});
               state(stateCodes::ERROR);
            }
            return 0;
         }

         int rv;

         rv = queryBSWV(1);

         if( rv < 0 )
         {
            if(rv != SDG_PARSEERR_WVTP )
            {
               if(m_powerState > 0 && !m_shutdown)
               {
                  log<software_error>({__FILE__, __LINE__});
                  state(stateCodes::ERROR);
               }
               return 0;
            }

            cs = 1; //Trigger normalizeSetup
         }

         rv = queryBSWV(2);

         if( rv < 0 )
         {
            if(rv != SDG_PARSEERR_WVTP )
            {
               if(m_powerState > 0 && !m_shutdown)
               {
                  log<software_error>({__FILE__, __LINE__});
                  state(stateCodes::ERROR);
               }
               return 0;
            }

            cs = 1; //Trigger normalizeSetup
         }


         if(cs > 0)
         {
            log<text_log>("Failed setup check, normalizing setup.", logPrio::LOG_NOTICE);
            normalizeSetup();

            return 0;
         }


         if( queryOUTP(1) < 0 )
         {
            if(m_powerState > 0 && !m_shutdown)
            {
               log<software_error>({__FILE__, __LINE__});
               state(stateCodes::ERROR);
            }
            return 0;
         }

         if( queryOUTP(2) < 0 )
         {
            if(m_powerState > 0 && !m_shutdown)
            {
               log<software_error>({__FILE__, __LINE__});
               state(stateCodes::ERROR);
            }
            return 0;
         }

         if( m_C1outp == 1 || m_C2outp == 1)
         {
            state(stateCodes::OPERATING);
         }
         else
         {
            state(stateCodes::READY);
         }
      }

      if( m_changeToLog )
      {
         int start = m_changeToLog;
         log<fxngen_params>({m_C1outp, m_C1frequency, m_C1vpp, m_C1ofst, m_C1phse, m_C1wvtp,
                             m_C2outp, m_C2frequency, m_C2vpp, m_C2ofst, m_C2phse, m_C2wvtp});

         m_changeToLog -= start;
         if(m_changeToLog < 0) m_changeToLog = 0;

      }

      return 0;

   }

   if( state() == stateCodes::CONFIGURING )
   {
      return 0;
   }

   //It's possible to get here because other threads are changing states.
   //These are the only valid states for this APP at this point.  Anything else and we'll log it.
   if( state() == stateCodes::READY || state() == stateCodes::OPERATING || state() == stateCodes::CONFIGURING )
   {
      return 0;
   }


   log<software_error>({__FILE__, __LINE__, "appLogic fell through in state " + stateCodes::codeText(state())});
   return 0;

}

inline
int siglentSDG::onPowerOff()
{
   std::lock_guard<std::mutex> lock(m_indiMutex);

   m_C1wvtp = "NONE";
   m_C1frequency = 0.0;
   m_C1vpp = 0.0;
   m_C1ofst = 0.0;
   m_C1outp = 0;

   updateIfChanged(m_indiP_C1wvtp, "value", m_C1wvtp);
   updateIfChanged(m_indiP_C1freq, "value", 0.0);
   updateIfChanged(m_indiP_C1peri, "value", 0.0);
   updateIfChanged(m_indiP_C1amp, "value", 0.0);
   updateIfChanged(m_indiP_C1ampvrms, "value", 0.0);
   updateIfChanged(m_indiP_C1ofst, "value", 0.0);
   updateIfChanged(m_indiP_C1hlev, "value", 0.0);
   updateIfChanged(m_indiP_C1llev, "value", 0.0);
   updateIfChanged(m_indiP_C1phse, "value", 0.0);
   updateIfChanged(m_indiP_C1outp, "value", std::string("Off"));

   m_C2wvtp = "NONE";
   m_C2frequency = 0.0;
   m_C2vpp = 0.0;
   m_C2ofst = 0.0;
   m_C2outp = 0;

   updateIfChanged(m_indiP_C2wvtp, "value", m_C2wvtp);
   updateIfChanged(m_indiP_C2freq, "value", 0.0);
   updateIfChanged(m_indiP_C2peri, "value", 0.0);
   updateIfChanged(m_indiP_C2amp, "value", 0.0);
   updateIfChanged(m_indiP_C2ampvrms, "value", 0.0);
   updateIfChanged(m_indiP_C2ofst, "value", 0.0);
   updateIfChanged(m_indiP_C2hlev, "value", 0.0);
   updateIfChanged(m_indiP_C2llev, "value", 0.0);
   updateIfChanged(m_indiP_C2phse, "value", 0.0);
   updateIfChanged(m_indiP_C1outp, "value", std::string("Off"));

   return 0;
}

inline
int siglentSDG::whilePowerOff()
{
   return onPowerOff();
}

inline
int siglentSDG::appShutdown()
{
   //don't bother
   return 0;
}

inline
int siglentSDG::writeRead( std::string & strRead,
                           const std::string & command
                         )
{
   int rv;

   rv = m_telnetConn.writeRead(command, false, m_writeTimeOut, m_readTimeOut);
   strRead = m_telnetConn.m_strRead;

   if(rv < 0)
   {
      if(m_powerState && !m_shutdown) log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      state(stateCodes::NOTCONNECTED);
      return -1;
   }

   //Clear the newline
   rv = m_telnetConn.write("\n", m_writeTimeOut);
   if(rv < 0)
   {
      if(m_powerState && !m_shutdown) log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }

   rv = m_telnetConn.read(">>", m_readTimeOut);
   if(rv < 0)
   {
      if(m_powerState && !m_shutdown) log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }
   return 0;

}

inline
int siglentSDG::writeCommand( const std::string & command )
{

   int rv = m_telnetConn.write(command, m_writeTimeOut);
   if(rv < 0)
   {
      if(m_powerState && !m_shutdown) log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }

   //Clear the newline
   rv = m_telnetConn.write("\n", m_writeTimeOut);
   if(rv < 0)
   {
      if(m_powerState && !m_shutdown) log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }

   rv = m_telnetConn.read(">>", m_readTimeOut);
   if(rv < 0)
   {
      if(m_powerState && !m_shutdown) log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }

   return 0;
}

inline
std::string makeCommand( int channel,
                         const std::string & afterColon
                       )
{
   std::string command = "C";
   command += mx::ioutils::convertToString<int>(channel);
   command += ":";
   command += afterColon;
   command += "\r\n";

   return command;
}

inline
int siglentSDG::queryMDWV( std::string & state,
                           int channel
                         )
{
   int rv;

   if(channel < 1 || channel > 2) return -1;

   std::string strRead;

   std::string com = makeCommand(channel, "MDWV?");

   rv = writeRead( strRead, com);

   if(rv < 0)
   {
      if(m_powerState && !m_shutdown) log<text_log>("Error on MDWV? for channel " + mx::ioutils::convertToString<int>(channel), logPrio::LOG_ERROR);
      return -1;
   }

   int resp_channel;
   std::string resp_state;

   rv = parseMDWV(resp_channel, resp_state, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         if(m_powerState && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      state = resp_state;
   }
   else
   {
      if(m_powerState && !m_shutdown) log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::querySWWV( std::string & state,
                           int channel
                         )
{
   int rv;

   if(channel < 1 || channel > 2) return -1;

   std::string strRead;

   std::string com = makeCommand(channel, "SWWV?");

   rv = writeRead( strRead, com);

   if(rv < 0)
   {
      if(m_powerState && !m_shutdown) log<text_log>("Error on SWWV? for channel " + mx::ioutils::convertToString<int>(channel), logPrio::LOG_ERROR);
      return -1;
   }

   int resp_channel;
   std::string resp_state;

   rv = parseSWWV(resp_channel, resp_state, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         if(m_powerState && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      state = resp_state;
   }
   else
   {
      if(m_powerState && !m_shutdown) log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::queryBTWV( std::string & state,
                           int channel
                         )
{
   int rv;

   if(channel < 1 || channel > 2) return -1;

   std::string strRead;

   std::string com = makeCommand(channel, "BTWV?");

   rv = writeRead( strRead, com);

   if(rv < 0)
   {
      if(m_powerState && !m_shutdown) log<text_log>("Error on BTWV? for channel " + mx::ioutils::convertToString<int>(channel), logPrio::LOG_ERROR);
      return -1;
   }

   int resp_channel;
   std::string resp_state;

   rv = parseBTWV(resp_channel, resp_state, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         if(m_powerState && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      state = resp_state;
   }
   else
   {
      if(m_powerState && !m_shutdown) log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::queryARWV( int & index,
                           int channel
                         )
{
   int rv;

   if(channel < 1 || channel > 2) return -1;

   std::string strRead;

   std::string com = makeCommand(channel, "ARWV?");

   rv = writeRead( strRead, com);

   if(rv < 0)
   {
      if(m_powerState && !m_shutdown) log<text_log>("Error on ARWV? for channel " + mx::ioutils::convertToString<int>(channel), logPrio::LOG_ERROR);
      return -1;
   }

   int resp_channel;
   int resp_index;

   rv = parseARWV(resp_channel, resp_index, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         if(m_powerState && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      index = resp_index;
   }
   else
   {
      if(m_powerState && !m_shutdown) log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::queryBSWV( int channel)
{
   int rv;

   if(channel < 1 || channel > 2) return -1;

   std::string strRead;

   std::string com = makeCommand(channel, "BSWV?");

   rv = writeRead( strRead, com);

   if(rv < 0)
   {
      if(m_powerState && !m_shutdown) log<text_log>("Error on BSWV? for channel " + mx::ioutils::convertToString<int>(channel), logPrio::LOG_ERROR);
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
         if(m_powerState && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      if(channel == 1)
      {
         m_C1wvtp = resp_wvtp;
         m_C1frequency = resp_freq;
         m_C1vpp = resp_amp;
         m_C1ofst = resp_ofst;
         m_C1phse = resp_phse;

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
         m_C2wvtp = resp_wvtp;
         m_C2frequency = resp_freq;
         m_C2vpp = resp_amp;
         m_C2ofst = resp_ofst;
         m_C2phse = resp_phse;

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

inline
int siglentSDG::queryOUTP( int channel )
{
   int rv;

   if(channel < 1 || channel > 2) return -1;

   std::string strRead;

   std::string com = makeCommand(channel, "OUTP?");

   rv = writeRead( strRead, com);

   if(rv < 0)
   {
      if(m_powerState && !m_shutdown) log<text_log>("Error on OUTP? for channel " + mx::ioutils::convertToString<int>(channel), logPrio::LOG_ERROR);
      return -1;
   }

   int resp_channel;
   int resp_output;

   rv = parseOUTP(resp_channel, resp_output, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         if(m_powerState && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      std::string ro;
      if(resp_output > 0) ro = "On";
      else if(resp_output == 0 ) ro = "Off";
      else ro = "UNK";

      if(channel == 1)
      {
         m_C1outp = resp_output;
         updateIfChanged(m_indiP_C1outp, "value", ro);
      }

      else if(channel == 2)
      {
         m_C2outp = resp_output;
         updateIfChanged(m_indiP_C2outp, "value", ro);
      }
   }
   else
   {
      if(m_powerState && !m_shutdown) log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::checkSetup()
{
   std::string state;
   int index;
   int rv;

   rv = queryMDWV(state, 1);

   if(rv < 0)
   {
      if(m_powerState > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(state != "OFF")
   {
      if(m_powerState > 0 && !m_shutdown) log<text_log>("Channel 1 MDWV not OFF");
      return 1;
   }

   rv = queryMDWV(state, 2);

   if(rv < 0)
   {
      if(m_powerState > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(state != "OFF")
   {
      if(m_powerState > 0 && !m_shutdown) log<text_log>("Channel 2 MDWV not OFF");
      return 1;
   }

   rv = querySWWV(state, 1);

   if(rv < 0)
   {
      if(m_powerState > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(state != "OFF")
   {
      if(m_powerState > 0 && !m_shutdown) log<text_log>("Channel 1 SWWV not OFF");
      return 1;
   }

   rv = querySWWV(state, 2);

   if(rv < 0)
   {
      if(m_powerState > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(state != "OFF")
   {
      if(m_powerState > 0 && !m_shutdown) log<text_log>("Channel 2 SWWV no OFF");
      return 1;
   }

   rv = queryBTWV(state, 1);

   if(rv < 0)
   {
      if(m_powerState > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(state != "OFF")
   {
      if(m_powerState > 0 && !m_shutdown) log<text_log>("Channel 1 BTWV not OFF");
      return 1;
   }

   rv = queryBTWV(state, 2);

   if(rv < 0)
   {
      if(m_powerState > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(state != "OFF")
   {
      if(m_powerState > 0 && !m_shutdown) log<text_log>("Channel 2 BTWV not OFF");
      return 1;
   }

   rv = queryARWV(index, 1);

   if(rv < 0)
   {
      if(m_powerState > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(index != 0)
   {
      if(m_powerState > 0 && !m_shutdown) log<text_log>("Channel 1 ARWV not 1");
      return 1;
   }

   rv = queryARWV(index, 2);

   if(rv < 0)
   {
      if(m_powerState > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(index != 0)
   {
      if(m_powerState > 0 && !m_shutdown) log<text_log>("Channel 2 ARWV not 1");
      return 1;
   }



   return 0;
}

inline
int siglentSDG::normalizeSetup()
{

   std::cerr << "Normalizing . . .";
   changeOutp(1, "OFF");
   changeOutp(2, "OFF");



   std::string afterColon;
   std::string command;

   afterColon = "MDWV STATE,OFF";
   command = makeCommand(1, afterColon);
   writeCommand(command);

   command = makeCommand(2, afterColon);
   writeCommand(command);

   afterColon = "SWWV STATE,OFF";
   command = makeCommand(1, afterColon);
   writeCommand(command);

   command = makeCommand(2, afterColon);
   writeCommand(command);

   afterColon = "BTWV STATE,OFF";
   command = makeCommand(1, afterColon);
   writeCommand(command);

   command = makeCommand(2, afterColon);
   writeCommand(command);

   afterColon = "ARWV INDEX,0";
   command = makeCommand(1, afterColon);
   writeCommand(command);

   command = makeCommand(2, afterColon);
   writeCommand(command);

   changeWvtp(1, "SINE");
   changeWvtp(2, "SINE");

   changeFreq(1, 0);
   changeFreq(2, 0);

   changeAmp(1, 0);
   changeAmp(2, 0);

   changePhse(1, 0);
   changePhse(2, 0);

   changeOfst(1, 0.0);
   changeOfst(2, 0.0);

   changeWvtp(1, "DC");
   changeWvtp(2, "DC");

   changeOfst(1, 0.0);
   changeOfst(2, 0.0);

   changeOutp(1, "OFF");
   changeOutp(2, "OFF");

   changeWvtp(1, "SINE");
   changeWvtp(2, "SINE");

   std::cerr << "Done\n";
   return 0;
}

inline
int siglentSDG::changeOutp( int channel,
                            const std::string & newOutp
                          )
{
   if(channel < 1 || channel > 2) return -1;

   std::string no;

   if(newOutp == "Off" || newOutp == "OFF" || newOutp == "off") no = "OFF";
   else if(newOutp == "On" || newOutp == "ON" || newOutp == "on") no = "ON";
   else
   {
      log<software_error>({__FILE__, __LINE__, "Invalid OUTP spec: " + newOutp});
      return -1;
   }

   std::string afterColon = "OUTP " + no;
   std::string command = makeCommand(channel, afterColon);

   log<text_log>("Ch. " + std::to_string(channel) + " OUTP to " + newOutp, logPrio::LOG_NOTICE);

   int rv = writeCommand(command);

   ++m_changeToLog;

   if(rv < 0)
   {
      if(m_powerState) log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::changeOutp( int channel,
                            const pcf::IndiProperty &ipRecv
                          )
{
   if(channel < 1 || channel > 2) return -1;

   if(state() != stateCodes::READY && state() != stateCodes::OPERATING) return 0;

   std::string newOutp;
   try
   {
      newOutp = ipRecv["value"].get<std::string>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }

   //Make sure we don't change things while other things are being updated.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.

   stateCodes::stateCodeT enterState = state();
   state(stateCodes::CONFIGURING);

   int rv = changeOutp(channel, newOutp);
   if(rv < 0) log<software_error>({__FILE__, __LINE__});

   state(enterState);

   return rv;
}

inline
int siglentSDG::changeFreq( int channel,
                            double newFreq
                          )
{
   if(channel < 1 || channel > 2) return -1;

   if(newFreq > cs_MaxFreq)
   {
      newFreq = cs_MaxFreq;
   }

   std::string afterColon = "BSWV FRQ," + mx::ioutils::convertToString<double>(newFreq);
   std::string command = makeCommand(channel, afterColon);

   log<text_log>("Ch. " + std::to_string(channel) + " FREQ to " + std::to_string(newFreq), logPrio::LOG_NOTICE);

   int rv = writeCommand(command);

   ++m_changeToLog;

   if(rv < 0)
   {
      if(m_powerState) log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::changeFreq( int channel,
                            const pcf::IndiProperty &ipRecv
                          )
{
   if(channel < 1 || channel > 2) return -1;

   if(state() != stateCodes::READY && state() != stateCodes::OPERATING) return 0;

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

   //Make sure we don't change things while other things are being updated.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.
   stateCodes::stateCodeT enterState = state();
   state(stateCodes::CONFIGURING);

   int rv = changeFreq(channel,newFreq);
   if(rv < 0) log<software_error>({__FILE__, __LINE__});

   state(enterState);

   return rv;
}

inline
int siglentSDG::changeAmp( int channel,
                           double newAmp
                         )
{
   if(channel < 1 || channel > 2) return -1;

   if(newAmp > cs_MaxAmp)
   {
      newAmp = cs_MaxAmp;
   }

   std::string afterColon = "BSWV AMP," + mx::ioutils::convertToString<double>(newAmp);
   std::string command = makeCommand(channel, afterColon);

   log<text_log>("Ch. " + std::to_string(channel) + " AMP to " + std::to_string(newAmp), logPrio::LOG_NOTICE);

   int rv = writeCommand(command);

   ++m_changeToLog;

   if(rv < 0)
   {
      if(m_powerState) log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::changeAmp( int channel,
                           const pcf::IndiProperty &ipRecv
                         )
{
   if(channel < 1 || channel > 2) return -1;

   if(state() != stateCodes::READY && state() != stateCodes::OPERATING) return 0;

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

   //Make sure we don't change things while other things are being updated.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.

   stateCodes::stateCodeT enterState = state();
   state(stateCodes::CONFIGURING);

   int rv = changeAmp(channel, newAmp);
   if(rv < 0) log<software_error>({__FILE__, __LINE__});

   state(enterState);

   return rv;
}

inline
int siglentSDG::changeOfst( int channel,
                            double newOfst
                          )
{
   if(channel < 1 || channel > 2) return -1;

   if(newOfst > cs_MaxOfst)
   {
      newOfst = cs_MaxOfst;
   }

   if(newOfst < 0.0)
   {
      newOfst = 0.0;
   }

   std::string afterColon = "BSWV OFST," + mx::ioutils::convertToString<double>(newOfst);
   std::string command = makeCommand(channel, afterColon);

   log<text_log>("Ch. " + std::to_string(channel) + " OFST to " + std::to_string(newOfst), logPrio::LOG_NOTICE);

   int rv = writeCommand(command);

   ++m_changeToLog;

   if(rv < 0)
   {
      if(m_powerState) log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::changeOfst( int channel,
                            const pcf::IndiProperty &ipRecv
                          )
{
   if(channel < 1 || channel > 2) return -1;

   if(state() != stateCodes::READY && state() != stateCodes::OPERATING) return 0;

   double newOfst;
   try
   {
      newOfst = ipRecv["value"].get<double>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }

   //Make sure we don't change things while other things are being updated.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.

   stateCodes::stateCodeT enterState = state();
   state(stateCodes::CONFIGURING);

   int rv = changeOfst(channel, newOfst);
   if(rv < 0) log<software_error>({__FILE__, __LINE__});

   state(enterState);

   return rv;
}

inline
int siglentSDG::changePhse( int channel,
                            double newPhse
                          )
{
   if(channel < 1 || channel > 2) return -1;

   std::string afterColon = "BSWV PHSE," + mx::ioutils::convertToString<double>(newPhse);
   std::string command = makeCommand(channel, afterColon);

   log<text_log>("Ch. " + std::to_string(channel) + " PHSE to " + std::to_string(newPhse), logPrio::LOG_NOTICE);

   int rv = writeCommand(command);

   ++m_changeToLog;

   if(rv < 0)
   {
      if(m_powerState) log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::changePhse( int channel,
                            const pcf::IndiProperty &ipRecv
                          )
{
   if(channel < 1 || channel > 2) return -1;

   if(state() != stateCodes::READY && state() != stateCodes::OPERATING) return 0;

   double newPhse;
   try
   {
      newPhse = ipRecv["value"].get<double>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }

   //Make sure we don't change things while other things are being updated.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.

   stateCodes::stateCodeT enterState = state();
   state(stateCodes::CONFIGURING);

   int rv = changePhse(channel, newPhse);
   if(rv < 0) log<software_error>({__FILE__, __LINE__});

   state(enterState);

   return rv;
}

inline
int siglentSDG::changeWvtp( int channel,
                            const std::string & newWvtp
                          )
{
   if(channel < 1 || channel > 2) return -1;

   std::string afterColon = "BSWV WVTP," + newWvtp;
   std::string command = makeCommand(channel, afterColon);

   log<text_log>("Ch. " + std::to_string(channel) + " WVTP to " + newWvtp, logPrio::LOG_NOTICE);

   int rv = writeCommand(command);

   ++m_changeToLog;

   if(rv < 0)
   {
      if(m_powerState) log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::changeWvtp( int channel,
                            const pcf::IndiProperty &ipRecv
                          )
{
   if(channel < 1 || channel > 2) return -1;

   if(state() != stateCodes::READY && state() != stateCodes::OPERATING) return 0;

   std::string newWvtp;
   try
   {
      newWvtp = ipRecv["value"].get<std::string>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }

   //Make sure we don't change things while other things are being updated.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.

   stateCodes::stateCodeT enterState = state();
   state(stateCodes::CONFIGURING);

   int rv = changeWvtp(channel, newWvtp);
   if(rv < 0) log<software_error>({__FILE__, __LINE__});

   state(enterState);

   return rv;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1outp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C1outp.getName())
   {
      return changeOutp(1, ipRecv);
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

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1ofst)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C1ofst.getName())
   {
      return changeOfst(1, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1phse)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C1phse.getName())
   {
      return changePhse(1, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1wvtp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C1wvtp.getName())
   {
      return changeWvtp(1, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2outp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C2outp.getName())
   {
      return changeOutp(2, ipRecv);
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

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2ofst)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C2ofst.getName())
   {
      return changeOfst(2, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2phse)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C2phse.getName())
   {
      return changePhse(2, ipRecv);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2wvtp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_C2wvtp.getName())
   {
      return changeWvtp(2, ipRecv);
   }
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //siglentSDG_hpp
