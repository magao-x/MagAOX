

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
  * \todo convert to ioDevice
  * \todo need telnet device, with optional username/password.
  *
  */
class siglentSDG : public MagAOXApp<>, public dev::telemeter<siglentSDG>
{

   friend class siglentSDG_test;

   friend class dev::telemeter<siglentSDG>;

   typedef dev::telemeter<siglentSDG> telemeterT;

   //constexpr static double cs_MaxAmp = 0.87;//2.1;//0.87;
   constexpr static double cs_MaxOfst = 10.0;
   constexpr static double cs_MaxVolts = 10.0;
   //constexpr static double cs_MaxFreq = 3622.0;//101;//3622.0;

private:
   std::vector<double> m_ampMax = {1.2801,   1.2801,  1.0201};//0.71,  0.83, 0.88, 1.05, 1.15, 3.45}; //1.5,     1.2,     1.1     };
   std::vector<double> m_maxFreq = {0.0,   2000,      3000};//100.0,   150,  200,  250,  300, 1000}; //2999.99, 3499.99, 3500.01};
   //todo: do we need to add max and min pulse variables?
protected:

   /** \name Configurable Parameters
     * @{
     */

   std::string m_deviceAddr; ///< The device address
   std::string m_devicePort; ///< The device port

   double m_bootDelay {10}; ///< Time in seconds it takes the device to boot.

   int m_writeTimeOut {10000};  ///< The timeout for writing to the device [msec].
   int m_readTimeOut {10000}; ///< The timeout for reading from the device [msec].

   double m_C1setVoltage {5.0}; ///< the set position voltage of Ch. 1.
   double m_C2setVoltage {5.0}; ///< the set position voltage of Ch. 2.

   bool m_C1outpOn {false}; /**< Flag controlling if C1 output is on after normalization. */
   bool m_C2outpOn {false}; /**< Flag controlling if C2 output is on after normalization.
                                 This will only have an effect if m_C1wvtp is "pulse" */

   ///@}

   tty::telnetConn m_telnetConn; ///< The telnet connection manager

   std::string m_waveform; ///< The chosen funciton to generate
   /// std::string m_clock; ///<INTernal or EXTernal

   uint8_t m_C1outp {0}; ///< The output status channel 1
   double m_C1frequency {0}; ///< The output frequency of channel 1
   double m_C1vpp {0}; ///< The peak-2-peak voltage of channel 1
   double m_C1vppDefault {0}; ///< default value for vpp of channel 1
   double m_C1ofst {0}; ///< The offset voltage of channel 1
   double m_C1phse {0}; ///< The phase of channel 1 (SINE only)
   double m_C1wdth {0}; ///< The width of channel 1 (PULSE only)
   std::string m_C1wvtp; ///< The wave type of channel 1
   double m_C1ampMax {10.0}; ///< The maximum voltage output for channel 1

   uint8_t m_C2outp {0}; ///<  The output status channel 2
   double m_C2frequency {0}; ///< The output frequency of channel 2
   double m_C2vpp {0}; ///< The peak-2-peak voltage of channel 2
   double m_C2vppDefault {0}; ///< default value for vpp of channel 2
   double m_C2ofst {0}; ///< The offset voltage of channel 2
   double m_C2phse {0}; ///< The phase of channel 2 (SINE only)
   double m_C2wdth {0}; ///< The width of channel 2 (PULSE only)
   std::string m_C2wvtp; ///< The wave type of channel 2
   double m_C2ampMax {10.0}; ///< The maximum voltage output for channel 2

   double m_C1frequency_tgt {-1};
   double m_C1vpp_tgt {-1};

   double m_C2frequency_tgt {-1};
   double m_C2vpp_tgt {-1};

   bool m_C1sync {false};
   bool m_C2sync {false};

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

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

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

   /// Send the SYNC? query for a channel.
   /** This updates member variables and INDI.
     *
     * \returns 0 on success
     * \returns -1 on an error.
     */
   int querySYNC( bool & sync, /// < [in] the sync state for this channel
                  int channel  /// < [in] the channel to query
                );

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
                   bool newOutp ///< [in] The requested output state [On/Off]
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

   /// Send a width command to the device.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeWdth( int channel,  ///< [in] the channel to send the command to.
                   double newWdth ///< [in] The requested new width [s]
                 );

   /// Send a change phase command to the device in response to an INDI property.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeWdth( int channel,                    ///< [in] the channel to send the command to.
                   const pcf::IndiProperty &ipRecv ///< [in] INDI property containing the requested new width [s]
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

   /// Send a change sync command to the device.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeSync( int channel,  ///< [in] the channel to send the command to.
                   bool newSync ///< [in] The requested new sync state
                 );

   /// Send a change sync command to the device in response to an INDI property.
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int changeSync( int channel,                    ///< [in] the channel to send the command to.
                   const pcf::IndiProperty &ipRecv ///< [in] INDI property containing the requested new sync state
                 );

   /** \name INDI
     * @{
     */
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
   pcf::IndiProperty m_indiP_C1wdth;
   pcf::IndiProperty m_indiP_C1sync;

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
   pcf::IndiProperty m_indiP_C2wdth;
   pcf::IndiProperty m_indiP_C2sync;

public:
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1outp);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1freq);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1amp);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1ofst);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1phse);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1wdth);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1wvtp);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C1sync);

   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2outp);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2freq);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2amp);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2ofst);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2phse);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2wdth);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2wvtp);
   INDI_NEWCALLBACK_DECL(siglentSDG, m_indiP_C2sync);
   ///@}


   /** \name Telemeter Interface
     *
     * @{
     */

   int checkRecordTimes();

   int recordTelem( const telem_fxngen * );

   int recordParams(bool force = false);

   /// @}

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

   config.add("fxngen.waveform", "w", "fxngen.waveform", argType::Required, "fxngen", "waveform", false, "string", "The waveform to populate function.");

   config.add("fxngen.C1outpOn", "", "fxngen.C1outpOn", argType::Required, "fxngen", "C1outpOn", false, "bool", "Whether (true) or not (false) C1 output is enabled at startup. Only effective wavefrom is pulse. Default is false.");
   config.add("fxngen.C2outpOn", "", "fxngen.C2outpOn", argType::Required, "fxngen", "C2outpOn", false, "bool", "Whether (true) or not (false) C2 output is enabled at startup. Only effective wavefrom is pulse. Default is false.");

   config.add("fxngen.C1ampDefault", "", "fxngen.C1ampDefault", argType::Required, "fxngen", "C1ampDefault", false, "float", "C1 Default P2V Amplitude of waveform. Default = 0.0");
   config.add("fxngen.C2ampDefault", "", "fxngen.C2ampDefault", argType::Required, "fxngen", "C2ampDefault", false, "float", "C2 Default P2V Amplitude of waveform. Default = 0.0");

   config.add("fxngen.C1ofstDefault", "", "fxngen.C1ofstDefault", argType::Required, "fxngen", "C1ofstDefault", false, "float", "C1 Default Offset Amplitude of waveform. Default = 0.0");
   config.add("fxngen.C2ofstDefault", "", "fxngen.C2ofstDefault", argType::Required, "fxngen", "C2ofstDefault", false, "float", "C2 Default Offset Amplitude of waveform. Default = 0.0");

   config.add("fxngen.C1ampMax", "", "fxngen.C1ampMax", argType::Required, "fxngen", "C1ampMax", false, "float", "C1 Maximum amplitude");
   config.add("fxngen.C2ampMax", "", "fxngen.C2ampMax", argType::Required, "fxngen", "C2ampMax", false, "float", "C2 Maximum amplitude");

   TELEMETER_SETUP_CONFIG(config);
}

inline
int siglentSDG::loadConfigImpl(mx::app::appConfigurator &_config )
{
   _config(m_deviceAddr, "device.address");
   _config(m_devicePort, "device.port");

   _config(m_writeTimeOut, "timeouts.write");
   _config(m_readTimeOut, "timeouts.read");

   _config(m_waveform, "fxngen.waveform"); // todo: check if this is a valid waveform?
   _config(m_C1outpOn, "fxngen.C1outpOn");
   _config(m_C2outpOn, "fxngen.C2outpOn");

   _config(m_C1vppDefault, "fxngen.C1ampDefault");
   _config(m_C2vppDefault, "fxngen.C2ampDefault");

   _config(m_C1ofst, "fxngen.C1ofstDefault");
   _config(m_C2ofst, "fxngen.C2ofstDefault");

   _config(m_C1ampMax, "fxngen.C1ampMax");
   _config(m_C2ampMax, "fxngen.C2ampMax");

   TELEMETER_LOAD_CONFIG(_config);

   return 0;
}

inline
void siglentSDG::loadConfig()
{
   loadConfigImpl( config );
}


inline
int siglentSDG::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP_NOCB(m_indiP_status, "status", pcf::IndiProperty::Text);
   m_indiP_status.add (pcf::IndiElement("value"));
   m_indiP_status["value"].set(0);

   CREATE_REG_INDI_NEW_TOGGLESWITCH(m_indiP_C1outp, "C1outp");

   CREATE_REG_INDI_NEW_NUMBERF( m_indiP_C1freq, "C1freq", -1e15, 1e15, 1, "%g", "C1freq", "C1freq");
   m_indiP_C1freq["current"].set(0);
   m_indiP_C1freq["target"].set(0);

   CREATE_REG_INDI_NEW_NUMBERF( m_indiP_C1amp, "C1amp", -1e15, 1e15, 1, "%g", "C1amp", "C1amp");
   m_indiP_C1amp["current"].set(0);
   m_indiP_C1amp["target"].set(0);

   REG_INDI_NEWPROP(m_indiP_C1ofst, "C1ofst", pcf::IndiProperty::Number);
   m_indiP_C1ofst.add (pcf::IndiElement("value"));
   m_indiP_C1ofst["value"].set(0);

   if(m_waveform == "SINE"){
      REG_INDI_NEWPROP(m_indiP_C1phse, "C1phse", pcf::IndiProperty::Number);
      m_indiP_C1phse.add (pcf::IndiElement("value"));
      m_indiP_C1phse["value"].set(0);
   }
   else if(m_waveform == "PULSE")
   {
      REG_INDI_NEWPROP(m_indiP_C1wdth, "C1wdth", pcf::IndiProperty::Number);
      m_indiP_C1wdth.add (pcf::IndiElement("value"));
      m_indiP_C1wdth["value"].set(0);
   }

   CREATE_REG_INDI_NEW_TEXT(m_indiP_C1wvtp, "C1wvtp", "C1wvtp", "C1wvtp");
   m_indiP_C1wvtp["current"].set("");
   m_indiP_C1wvtp["target"].set("");

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

   CREATE_REG_INDI_NEW_TOGGLESWITCH(m_indiP_C1sync, "C1synchro");

   /* Channel 2 */
   CREATE_REG_INDI_NEW_TOGGLESWITCH(m_indiP_C2outp, "C2outp");

   CREATE_REG_INDI_NEW_NUMBERF( m_indiP_C2freq, "C2freq", -1e15, 1e15, 1, "%g", "C2freq", "C2freq");
   m_indiP_C2freq["current"].set(0);
   m_indiP_C2freq["target"].set(0);

   CREATE_REG_INDI_NEW_NUMBERF( m_indiP_C2amp, "C2amp", -1e15, 1e15, 1, "%g", "C2amp", "C2amp");
   m_indiP_C2amp["current"].set(0);
   m_indiP_C2amp["target"].set(0);

   REG_INDI_NEWPROP(m_indiP_C2ofst, "C2ofst", pcf::IndiProperty::Number);
   m_indiP_C2ofst.add (pcf::IndiElement("value"));
   m_indiP_C2ofst["value"].set(0);

   if(m_waveform == "SINE"){
      REG_INDI_NEWPROP(m_indiP_C2phse, "C2phse", pcf::IndiProperty::Number);
      m_indiP_C2phse.add (pcf::IndiElement("value"));
      m_indiP_C2phse["value"].set(0);
   }
   else if(m_waveform == "PULSE")
   {
      REG_INDI_NEWPROP(m_indiP_C2wdth, "C2wdth", pcf::IndiProperty::Number);
      m_indiP_C2wdth.add (pcf::IndiElement("value"));
      m_indiP_C2wdth["value"].set(0);
   }

   CREATE_REG_INDI_NEW_TEXT(m_indiP_C2wvtp, "C2wvtp", "C2wvtp", "C2wvtp");
   m_indiP_C2wvtp["current"].set("");
   m_indiP_C2wvtp["target"].set("");

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

   CREATE_REG_INDI_NEW_TOGGLESWITCH(m_indiP_C2sync, "C2synchro");


   TELEMETER_APP_STARTUP;

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
      if( (m_powerState != 1 || m_powerTargetState != 1) == true)
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
               log<software_critical>({__FILE__, __LINE__, "No response from device.  Time to power cycle."});
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

         recordParams(true);

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
            if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown)
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
               if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown)
               {
                  log<software_error>({__FILE__, __LINE__});
                  state(stateCodes::ERROR);
               }
               return 0;
            }

            cs = 1; //Trigger normalizeSetup
         }

         if(m_C1sync)
         {
            updateSwitchIfChanged(m_indiP_C1sync, "toggle", pcf::IndiElement::On, INDI_OK);
         }
         else
         {
            updateSwitchIfChanged(m_indiP_C1sync, "toggle", pcf::IndiElement::Off, INDI_IDLE);
         }

         rv = queryBSWV(2);

         if( rv < 0 )
         {
            if(rv != SDG_PARSEERR_WVTP )
            {
               if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown)
               {
                  log<software_error>({__FILE__, __LINE__});
                  state(stateCodes::ERROR);
               }
               return 0;
            }

            cs = 1; //Trigger normalizeSetup
         }

         if(m_C2sync)
         {
            updateSwitchIfChanged(m_indiP_C2sync, "toggle", pcf::IndiElement::On, INDI_OK);
         }
         else
         {
            updateSwitchIfChanged(m_indiP_C2sync, "toggle", pcf::IndiElement::Off, INDI_IDLE);
         }

         if(cs > 0)
         {
            log<text_log>("Failed setup check, normalizing setup.", logPrio::LOG_NOTICE);
            normalizeSetup();

            return 0;
         }


         if( queryOUTP(1) < 0 )
         {
            if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown)
            {
               log<software_error>({__FILE__, __LINE__});
               state(stateCodes::ERROR);
            }
            return 0;
         }

         if( queryOUTP(2) < 0 )
         {
            if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown)
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

         recordParams(); //This will check if anything changed.
      }

      TELEMETER_APP_LOGIC;

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

   m_C1frequency_tgt = -1;
   m_C1vpp_tgt = -1;

   updatesIfChanged<std::string>(m_indiP_C1wvtp, {"current", "target"}, {m_C1wvtp, m_C1wvtp});
   updatesIfChanged<double>(m_indiP_C1freq, {"current", "target"}, {0.0, 0.0});
   updatesIfChanged<double>(m_indiP_C1peri, {"current", "target"}, {0.0, 0.0});
   updatesIfChanged<double>(m_indiP_C1amp, {"current", "target"}, {0.0, 0.0});
   updatesIfChanged<double>(m_indiP_C1ofst, {"current", "target"}, {0.0, 0.0});
   updateIfChanged(m_indiP_C1ampvrms, "value", 0.0);
   updateIfChanged(m_indiP_C1hlev, "value", 0.0);
   updateIfChanged(m_indiP_C1llev, "value", 0.0);
   if(m_waveform == "SINE")
   {
      updatesIfChanged<double>(m_indiP_C1phse, {"current", "target"}, {0.0, 0.0});
   }
   else if(m_waveform == "PULSE")
   {
      updatesIfChanged<double>(m_indiP_C1wdth, {"current", "target"}, {0.0, 0.0});
   }
   updateSwitchIfChanged(m_indiP_C1outp, "toggle", pcf::IndiElement::Off, INDI_IDLE);
   updateSwitchIfChanged(m_indiP_C1sync, "toggle", pcf::IndiElement::Off, INDI_IDLE);


   m_C2wvtp = "NONE";
   m_C2frequency = 0.0;
   m_C2vpp = 0.0;
   m_C2ofst = 0.0;
   m_C2outp = 0;

   m_C2frequency_tgt = -1;
   m_C2vpp_tgt = -1;

   updatesIfChanged<std::string>(m_indiP_C2wvtp,{"current", "target"}, {m_C2wvtp, m_C2wvtp});
   updatesIfChanged<double>(m_indiP_C2freq, {"current", "target"}, {0.0, 0.0});
   updatesIfChanged<double>(m_indiP_C2peri, {"current", "target"}, {0.0, 0.0});
   updatesIfChanged<double>(m_indiP_C2amp, {"current", "target"}, {0.0, 0.0});
   updatesIfChanged<double>(m_indiP_C2ofst, {"current", "target"}, {0.0, 0.0});
   updateIfChanged(m_indiP_C2ampvrms, "value", 0.0);
   updateIfChanged(m_indiP_C2hlev, "value", 0.0);
   updateIfChanged(m_indiP_C2llev, "value", 0.0);
   if(m_waveform == "SINE")
   {
      updatesIfChanged<double>(m_indiP_C2phse, {"current", "target"}, {0.0, 0.0});
   }
   else if(m_waveform == "PULSE")
   {
      updatesIfChanged<double>(m_indiP_C2wdth, {"current", "target"}, {0.0, 0.0});
   }
   updateSwitchIfChanged(m_indiP_C2outp, "toggle", pcf::IndiElement::Off, INDI_IDLE);
   updateSwitchIfChanged(m_indiP_C2sync, "toggle", pcf::IndiElement::Off, INDI_IDLE);

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
   TELEMETER_APP_SHUTDOWN;

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
      std::cout << command << "\n";
      std::cout << "writeRead return val was " << rv << "\n";
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      state(stateCodes::NOTCONNECTED);
      return -1;
   }

   //Clear the newline
   rv = m_telnetConn.write("\n", m_writeTimeOut);
   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }

   rv = m_telnetConn.read(">>", m_readTimeOut);
   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
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
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }

   //Clear the newline
   rv = m_telnetConn.write("\n", m_writeTimeOut);
   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }

   rv = m_telnetConn.read(">>", m_readTimeOut);
   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
      return -1;
   }

   return 0;
}

inline
std::string makeCommand( int channel,
                         const std::string & afterColon
                       )
{
   std::string command = std::format("C{}:{}\r\n", channel, afterColon);
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
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<text_log>(std::format("Error on MDWV? for channel {}", channel), logPrio::LOG_ERROR);
      return -1;
   }

   int resp_channel;
   std::string resp_state;

   rv = parseMDWV(resp_channel, resp_state, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      state = resp_state;
   }
   else
   {
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
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
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<text_log>(std::format("Error on SWWV? for channel {}", channel), logPrio::LOG_ERROR);
      return -1;
   }

   int resp_channel;
   std::string resp_state;

   rv = parseSWWV(resp_channel, resp_state, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      state = resp_state;
   }
   else
   {
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
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
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<text_log>(std::format("Error on BTWV? for channel {}", channel), logPrio::LOG_ERROR);
      return -1;
   }

   int resp_channel;
   std::string resp_state;

   rv = parseBTWV(resp_channel, resp_state, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      state = resp_state;
   }
   else
   {
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
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
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<text_log>(std::format("Error on ARWV? for channel {}", channel), logPrio::LOG_ERROR);
      return -1;
   }

   int resp_channel;
   int resp_index;

   rv = parseARWV(resp_channel, resp_index, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      index = resp_index;
   }
   else
   {
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
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
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<text_log>(std::format("Error on BSWV? for channel {}", channel), logPrio::LOG_ERROR);
      return -1;
   }

   int resp_channel;
   std::string resp_wvtp;
   double resp_freq, resp_peri, resp_amp, resp_ampvrms, resp_ofst, resp_hlev, resp_llev, resp_phse, resp_wdth;

   rv = parseBSWV(resp_channel, resp_wvtp, resp_freq, resp_peri, resp_amp, resp_ampvrms, resp_ofst, resp_hlev, resp_llev, resp_phse, resp_wdth, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      if(channel == 1)
      {
         m_C1wvtp = resp_wvtp;
         m_C1frequency = resp_freq;
         m_C1vpp = resp_amp;
         m_C1ofst = resp_ofst;
         m_C1phse = resp_phse;
         m_C1wdth = resp_wdth;

         if(m_C1frequency_tgt == -1) m_C1frequency_tgt = m_C1frequency;
         if(m_C1vpp_tgt == -1) m_C1vpp_tgt = m_C1vpp;

         recordParams();

         updateIfChanged(m_indiP_C1wvtp, "current", resp_wvtp);
         updateIfChanged(m_indiP_C1freq, "current", resp_freq);
         updateIfChanged(m_indiP_C1peri, "current", resp_peri);
         updateIfChanged(m_indiP_C1amp, "current", resp_amp);
         updateIfChanged(m_indiP_C1ofst, "current", resp_ofst);
         updateIfChanged(m_indiP_C1ampvrms, "value", resp_ampvrms);
         updateIfChanged(m_indiP_C1hlev, "value", resp_hlev);
         updateIfChanged(m_indiP_C1llev, "value", resp_llev);
         if(m_waveform == "SINE"){updateIfChanged(m_indiP_C1phse, "current", resp_phse);}
         else if(m_waveform == "PULSE"){updateIfChanged(m_indiP_C1wdth, "current", resp_wdth);}
      }
      else if(channel == 2)
      {
         m_C2wvtp = resp_wvtp;
         m_C2frequency = resp_freq;
         m_C2vpp = resp_amp;
         m_C2ofst = resp_ofst;
         m_C2phse = resp_phse;
         m_C2wdth = resp_wdth;

         if(m_C2frequency_tgt == -1) m_C2frequency_tgt = m_C2frequency;
         if(m_C2vpp_tgt == -1) m_C2vpp_tgt = m_C2vpp;

         recordParams();

         updateIfChanged(m_indiP_C2wvtp, "current", resp_wvtp);
         updateIfChanged(m_indiP_C2freq, "current", resp_freq);
         updateIfChanged(m_indiP_C2peri, "current", resp_peri);
         updateIfChanged(m_indiP_C2amp, "current", resp_amp);
         updateIfChanged(m_indiP_C2ofst, "current", resp_ofst);
         updateIfChanged(m_indiP_C2ampvrms, "value", resp_ampvrms);
         updateIfChanged(m_indiP_C2hlev, "value", resp_hlev);
         updateIfChanged(m_indiP_C2llev, "value", resp_llev);
         if(m_waveform == "SINE"){updateIfChanged(m_indiP_C2phse, "current", resp_phse);}
         else if(m_waveform == "PULSE"){updateIfChanged(m_indiP_C2wdth, "current", resp_wdth);}
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
int siglentSDG::querySYNC( bool & sync,
                           int channel
                         )
{
   int rv;

   if(channel < 1 || channel > 2) return -1;

   std::string strRead;

   std::string com = makeCommand(channel, "SYNC?");

   rv = writeRead( strRead, com);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<text_log>(std::format("Error on SYNC? for channel {}", channel), logPrio::LOG_ERROR);
      return -1;
   }

   int resp_channel;
   bool resp_sync;

   rv = parseSYNC(resp_channel, resp_sync, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      sync = resp_sync;
   }
   else
   {
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
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
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<text_log>(std::format("Error on OUTP? for channel {}", channel), logPrio::LOG_ERROR);
      return -1;
   }

   int resp_channel;
   int resp_output;

   rv = parseOUTP(resp_channel, resp_output, strRead );

   if(rv == 0)
   {
      if(resp_channel != channel)
      {
         if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, "wrong channel returned"});
         return -1;
      }

      std::string ro;
      pcf::IndiElement ro_indi;
      if(resp_output > 0)
      {
         ro = "On";
         ro_indi = pcf::IndiElement::On;
      }
      else if(resp_output == 0 )
      {
         ro = "Off";
         ro_indi = pcf::IndiElement::Off;
      }
      else
      {
         ro = "UNK";
      }

      if(channel == 1)
      {
         m_C1outp = resp_output;
         recordParams();
         updateSwitchIfChanged(m_indiP_C1outp, "toggle", ro_indi, INDI_IDLE);
      }

      else if(channel == 2)
      {
         m_C2outp = resp_output;
         recordParams();
         updateSwitchIfChanged(m_indiP_C2outp, "toggle", ro_indi, INDI_IDLE);
      }
   }
   else
   {
      if((m_powerState != 1 || m_powerTargetState != 1) && !m_shutdown) log<software_error>({__FILE__,__LINE__, 0, rv, "parse error"});
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
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(state != "OFF")
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<text_log>("Channel 1 MDWV not OFF");
      return 1;
   }

   rv = queryMDWV(state, 2);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(state != "OFF")
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<text_log>("Channel 2 MDWV not OFF");
      return 1;
   }

   rv = querySWWV(state, 1);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(state != "OFF")
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<text_log>("Channel 1 SWWV not OFF");
      return 1;
   }

   rv = querySWWV(state, 2);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(state != "OFF")
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<text_log>("Channel 2 SWWV no OFF");
      return 1;
   }

   rv = queryBTWV(state, 1);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(state != "OFF")
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<text_log>("Channel 1 BTWV not OFF");
      return 1;
   }

   rv = queryBTWV(state, 2);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(state != "OFF")
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<text_log>("Channel 2 BTWV not OFF");
      return 1;
   }

   rv = queryARWV(index, 1);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(index != 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<text_log>("Channel 1 ARWV not 1");
      return 1;
   }

   rv = queryARWV(index, 2);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   if(index != 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<text_log>("Channel 2 ARWV not 1");
      return 1;
   }

   rv = querySYNC(m_C1sync, 1);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }

   rv = querySYNC(m_C2sync, 2);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1) > 0 && !m_shutdown) log<software_error>({__FILE__,__LINE__});
      return rv;
   }


   return 0;
}

inline
int siglentSDG::normalizeSetup()
{

   std::cerr << "Normalizing . . .";

   recordParams(true);

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

   changeWvtp(1, m_waveform);
   changeWvtp(2, m_waveform);

   changeFreq(1, 0);
   changeFreq(2, 0);

   changeAmp(1, m_C1vppDefault);
   changeAmp(2, m_C2vppDefault);

   if(m_waveform == "SINE")
   {
      changePhse(1, 0);
      changePhse(2, 0);
   }
   else if(m_waveform == "PULSE")
   {
      changeWdth(1, 0);
      changeWdth(2, 0);
   }

   changeOfst(1, m_C1ofst);
   changeOfst(2, m_C2ofst);

   changeWvtp(1, "DC");
   changeWvtp(2, "DC");


   if(m_C1outpOn && m_waveform == "PULSE")
   {
      changeOutp(1, "ON");
   }
   else
   {
      changeOutp(1, "OFF");
   }
   if(m_C2outpOn && m_waveform == "PULSE")
   {
      changeOutp(2, "ON");
   }
   else
   {
      changeOutp(2, "OFF");
   }

   changeWvtp(1, m_waveform);
   changeWvtp(2, m_waveform);

   recordParams(true);

   std::cerr << "Done\n";
   return 0;
}

inline
int siglentSDG::changeOutp( int channel,
                            bool newOutp
                          )
{
   if(channel < 1 || channel > 2) return -1;

   std::string no = newOutp ? "ON" : "OFF";

   std::string afterColon = "OUTP " + no;
   std::string command = makeCommand(channel, afterColon);

   log<text_log>("Ch. " + std::to_string(channel) + " OUTP to " + no, logPrio::LOG_NOTICE);

   recordParams(true);
   int rv = writeCommand(command);
   recordParams(true);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1)) log<software_error>({__FILE__, __LINE__});
      return -1;
   }


   if(channel == 1 && no == "ON")
   {
      if(changeSync(1, true) < 0)
      {
         return log<software_error,-1>({__FILE__, __LINE__});
      }
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

   bool output = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;

   //Make sure we don't change things while other things are being updated.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.

   stateCodes::stateCodeT enterState = state();
   state(stateCodes::CONFIGURING);

   int rv = changeOutp(channel, output);
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

   newFreq = std::clamp(newFreq, 0.0, m_maxFreq.back());

   if(m_waveform == "SINE"){
      // Limit amp for SINE waves

      double amp = (channel == 1) ? m_C1vpp_tgt : m_C2vpp_tgt;

      size_t i = 0;
      while( i < m_ampMax.size())
      {
         if(m_maxFreq[i] >= newFreq) break;
         ++i;
      }

      std::cerr << "Max Amp @ " << amp << " = " << m_ampMax[i] << " (freq)\n";

      if( amp > m_ampMax[i] )
      {
         log<text_log>("Ch. " + std::to_string(channel) + " FREQ not set due to amplitude exceeding limit for " + std::to_string(newFreq), logPrio::LOG_WARNING);
         return 0;
      }

   }

   //Now we update target
   if(channel==1)
   {
      m_C1frequency_tgt = newFreq;
   }
   else
   {
      m_C2frequency_tgt = newFreq;
   }


   std::string afterColon = std::format("BSWV FRQ,{}", newFreq);
   std::string command = makeCommand(channel, afterColon);

   log<text_log>("Ch. " + std::to_string(channel) + " FREQ to " + std::to_string(newFreq), logPrio::LOG_NOTICE);

   recordParams(true);
   int rv = writeCommand(command);
   recordParams(true);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1)) log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   // we want to automatically set the pulse width when setting a new frequency
   if(m_waveform == "PULSE"){
      // we want to auto change the pulse duration, want either 0.000250 or 0.5%
      double wdthLim = 0.5 / newFreq ;         // this is the limit if we don't have long enough frequencies
      double wdth250 = 0.000250;  // this is the ideal length of low dip WHACK THINGS.. it's doubling, want to be 0.00025
      double newWdth = wdthLim;

      if(wdthLim > wdth250){
         newWdth = wdth250;
         log<text_log>("Ch. " + std::to_string(channel) + " WDTH auto-changing to duty cycle limit: " + std::to_string(newWdth), logPrio::LOG_NOTICE);
      }else{
         log<text_log>("Ch. " + std::to_string(channel) + " WDTH auto-changing to 250us ideal case: " + std::to_string(newWdth), logPrio::LOG_NOTICE);
      }

      //changing pulse width
      changeWdth(channel, newWdth);
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
      newFreq = ipRecv["target"].get<double>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }

   if (channel==1) updateIfChanged(m_indiP_C1freq, "target", newFreq);
   else updateIfChanged(m_indiP_C2freq, "target", newFreq);

   //Make sure we don't change things while other things are being updated.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.
   stateCodes::stateCodeT enterState = state();
   state(stateCodes::CONFIGURING);

   int rv = changeFreq(channel, newFreq);
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

   double offst = m_C1ofst;
   if(channel == 2) offst = m_C2ofst;

   double confAmpMax = m_C1ampMax;
   if(channel == 2) confAmpMax = m_C2ampMax;

   if (0.5 * newAmp + offst > confAmpMax)
   {
      newAmp = 2 * (confAmpMax - offst);
      log<text_log>("Ch. " + std::to_string(channel) + " AMP max-limited by config value to " + std::to_string(newAmp), logPrio::LOG_WARNING);

   }

   // Do not limit freq if a PULSE wave
   if(m_waveform != "PULSE")
   {

      //Ensure we won't excede the 0-10V range for SINE
      if(offst + 0.5*newAmp > 10)
      {
         newAmp = 2.*(10.0 - offst);
         log<text_log>("Ch. " + std::to_string(channel) + " AMP limited at 10 V by OFST to " + std::to_string(newAmp), logPrio::LOG_WARNING);
      }

      if(offst - 0.5*newAmp < 0)
      {
         newAmp = 2*(offst);
         log<text_log>("Ch. " + std::to_string(channel) + " AMP limited at 0 V by OFST to " + std::to_string(newAmp), logPrio::LOG_WARNING);
      }

      double freq = m_C1frequency_tgt;
      if(channel == 2) freq = m_C2frequency_tgt;

      double ampMax;
      size_t i=0;
      while(i < m_ampMax.size())
      {
         if( m_maxFreq[i] >= freq ) break;
         ++i;
      }

      std::cerr << "Max Amp @ " << freq << " = " << ampMax << "\n";

      //Ensure we don't exced safe ranges for device
      if(newAmp > ampMax)
      {
         newAmp = ampMax;
         log<text_log>("Ch. " + std::to_string(channel) + " AMP max-limited to " + std::to_string(newAmp), logPrio::LOG_WARNING);
      }

      if(newAmp < 0)
      {
         newAmp = 0;
         log<text_log>("Ch. " + std::to_string(channel) + " AMP min-limited to " + std::to_string(newAmp), logPrio::LOG_WARNING);
      }
   }

   //Now update target
   if(channel==1)
   {
      m_C1vpp_tgt = newAmp;
   }
   else
   {
      m_C2vpp_tgt = newAmp;
   }


   std::string afterColon = std::format("BSWV AMP,{}", newAmp);
   std::string command = makeCommand(channel, afterColon);

   log<text_log>("Ch. " + std::to_string(channel) + " AMP set to " + std::to_string(newAmp), logPrio::LOG_NOTICE);

   recordParams(true);
   int rv = writeCommand(command);
   recordParams(true);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1)) log<software_error>({__FILE__, __LINE__});
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
      newAmp = ipRecv["target"].get<double>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }

   if (channel==1) updateIfChanged(m_indiP_C1amp, "target", newAmp);
   else updateIfChanged(m_indiP_C2amp, "target", newAmp);

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

   double amp = m_C1vpp;
   if(channel == 2) amp = m_C2vpp;

   double ampMax = m_C1ampMax;
   if(channel == 2) ampMax = m_C2ampMax;


   if(newOfst + 0.5*amp > ampMax)
   {
      newOfst = ampMax - 0.5*amp;
      log<text_log>("Ch. " + std::to_string(channel) + " OFST limited at " + std::to_string(ampMax) + " V by AMP to " + std::to_string(newOfst), logPrio::LOG_WARNING);
   }

   if(newOfst - 0.5*amp < 0)
   {
      newOfst = 0.5*amp;
      log<text_log>("Ch. " + std::to_string(channel) + " OFST limited at 0 V by AMP to " + std::to_string(newOfst), logPrio::LOG_WARNING);
   }

   if(newOfst > cs_MaxOfst)
   {
      newOfst = cs_MaxOfst;
      log<text_log>("Ch. " + std::to_string(channel) + " OFST max-limited to " + std::to_string(newOfst), logPrio::LOG_WARNING);
   }

   if(newOfst < 0.0)
   {
      newOfst = 0.0;
      log<text_log>("Ch. " + std::to_string(channel) + " OFST min-limited to " + std::to_string(newOfst), logPrio::LOG_WARNING);
   }

   std::string afterColon = std::format("BSWV OFST,{}", newOfst);
   std::string command = makeCommand(channel, afterColon);

   log<text_log>("Ch. " + std::to_string(channel) + " OFST set to " + std::to_string(newOfst), logPrio::LOG_NOTICE);

   int rv = writeCommand(command);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1)) log<software_error>({__FILE__, __LINE__});
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
      newOfst = ipRecv["target"].get<double>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }

   if (channel==1) updateIfChanged(m_indiP_C1ofst, "target", newOfst);
   else updateIfChanged(m_indiP_C2ofst, "target", newOfst);

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

   if(m_waveform == "PULSE"){
      log<text_log>("Ch. " + std::to_string(channel) + " PHSE not set for PULSE waveform.", logPrio::LOG_WARNING);
      return 0;
   }

   std::string afterColon = std::format("BSWV PHSE,{}", newPhse);
   std::string command = makeCommand(channel, afterColon);

   log<text_log>("Ch. " + std::to_string(channel) + " PHSE to " + std::to_string(newPhse), logPrio::LOG_NOTICE);

   int rv = writeCommand(command);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1)) log<software_error>({__FILE__, __LINE__});
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
      newPhse = ipRecv["target"].get<double>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }

   if (channel==1) updateIfChanged(m_indiP_C1phse, "target", newPhse);
   else updateIfChanged(m_indiP_C2phse, "target", newPhse);

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
int siglentSDG::changeWdth( int channel,
                            double newWdth
                          )
{
   if(channel < 1 || channel > 2) return -1;

   if(m_waveform != "PULSE"){
      log<text_log>("Ch. " + std::to_string(channel) + " WDTH can not be set, waveforem not PULSE.", logPrio::LOG_WARNING);
      return 0;
   }

   std::string afterColon = std::format("BSWV WIDTH,{}", newWdth);
   std::string command = makeCommand(channel, afterColon);

   log<text_log>("Ch. " + std::to_string(channel) + " WDTH to " + std::to_string(newWdth), logPrio::LOG_NOTICE);

   int rv = writeCommand(command);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1)) log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::changeWdth( int channel,
                            const pcf::IndiProperty &ipRecv
                          )
{
   if(channel < 1 || channel > 2) return -1;

   if(state() != stateCodes::READY && state() != stateCodes::OPERATING) return 0;

   double newWdth;
   try
   {
      newWdth = ipRecv["target"].get<double>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }

   if (channel==1) updateIfChanged(m_indiP_C1wdth, "target", newWdth);
   else updateIfChanged(m_indiP_C2wdth, "target", newWdth);

   //Make sure we don't change things while other things are being updated.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.

   stateCodes::stateCodeT enterState = state();
   state(stateCodes::CONFIGURING);

   int rv = changeWdth(channel, newWdth);
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

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1)) log<software_error>({__FILE__, __LINE__});
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
      newWvtp = ipRecv["target"].get<std::string>();
   }
   catch(...)
   {
      log<software_error>({__FILE__, __LINE__, "Exception caught."});
      return -1;
   }


   if (channel==1) updateIfChanged(m_indiP_C1wvtp, "target", newWvtp);
   else updateIfChanged(m_indiP_C2wvtp, "target", newWvtp);

   //Make sure we don't change things while other things are being updated.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.

   stateCodes::stateCodeT enterState = state();
   state(stateCodes::CONFIGURING);

   int rv = changeWvtp(channel, newWvtp);
   if(rv < 0) log<software_error>({__FILE__, __LINE__});

   state(enterState);

   return rv;
}

inline
int siglentSDG::changeSync( int channel,
                            const bool newSync)
{
   if(channel < 1 || channel > 2) return -1;

   std::string afterColon = "SYNC ";
   if(newSync) afterColon += "ON";
   else afterColon += "OFF";

   std::string command = makeCommand(channel, afterColon);

   if(newSync) log<text_log>("Ch. " + std::to_string(channel) + " SYNC to ON", logPrio::LOG_NOTICE);
   else log<text_log>("Ch. " + std::to_string(channel) + " SYNC to OFF", logPrio::LOG_NOTICE);

   recordParams(true);
   int rv = writeCommand(command);
   recordParams(true);

   if(rv < 0)
   {
      if((m_powerState != 1 || m_powerTargetState != 1)) log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   return 0;
}

inline
int siglentSDG::changeSync( int channel,
                            const pcf::IndiProperty &ipRecv
                          )
{
   if(channel < 1 || channel > 2) return -1;

   if(state() != stateCodes::READY && state() != stateCodes::OPERATING) return 0;

   bool newSync;

   if(!ipRecv.find("toggle")) return 0;

   newSync = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;

   //Make sure we don't change things while other things are being updated.
   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.

   stateCodes::stateCodeT enterState = state();
   state(stateCodes::CONFIGURING);

   int rv = changeSync(channel, newSync);
   if(rv < 0) log<software_error>({__FILE__, __LINE__});

   state(enterState);

   return rv;
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1outp)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C1outp, ipRecv);

    return changeOutp(1, ipRecv);

}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1freq)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C1freq, ipRecv);

    return changeFreq(1, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1amp)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C1amp, ipRecv);

    return changeAmp(1, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1ofst)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C1ofst, ipRecv);

    return changeOfst(1, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1phse)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C1phse, ipRecv);

    return changePhse(1, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1wdth)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C1wdth, ipRecv);

    return changeWdth(1, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1wvtp)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C1wvtp, ipRecv);

    return changeWvtp(1, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C1sync)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C1sync, ipRecv);

    return changeSync(1, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2outp)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C2outp, ipRecv);

    return changeOutp(2, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2freq)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C2freq, ipRecv);

    return changeFreq(2, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2amp)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C2amp, ipRecv);

    return changeAmp(2, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2ofst)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C2ofst, ipRecv);

    return changeOfst(2, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2phse)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C2phse, ipRecv);

    return changePhse(2, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2wdth)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C2wdth, ipRecv);

    return changeWdth(2, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2wvtp)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C2wvtp, ipRecv);

    return changeWvtp(2, ipRecv);
}

INDI_NEWCALLBACK_DEFN(siglentSDG, m_indiP_C2sync)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_C2sync, ipRecv);

    return changeSync(2, ipRecv);
}

// todo: add change width INDI

// todo: add change edge INDI

inline
int siglentSDG::checkRecordTimes()
{
   return telemeter<siglentSDG>::checkRecordTimes(telem_fxngen());
}

inline
int siglentSDG::recordTelem( const telem_fxngen * )
{
   return recordParams(true);
}

inline
int siglentSDG::recordParams(bool force)
{
   static double old_C1outp = -1e30; //Ensure first time writes
   static double old_C1frequency = m_C1frequency;
   static double old_C1vpp = m_C1vpp;
   static double old_C1ofst = m_C1ofst;
   static double old_C1phse = m_C1phse;
   static double old_C1wdth = m_C1wdth;
   static std::string old_C1wvtp = m_C1wvtp;
   static bool old_C1sync = m_C1sync;
   static double old_C2outp = m_C2outp;
   static double old_C2frequency = m_C2frequency;
   static double old_C2vpp = m_C2vpp;
   static double old_C2ofst = m_C2ofst;
   static double old_C2phse = m_C2phse;
   static double old_C2wdth = m_C2wdth;
   static std::string old_C2wvtp = m_C2wvtp;
   static bool old_C2sync = m_C2sync;

   bool write = false;

   if(!force)
   {
      if( old_C1outp != m_C1outp ) write = true;
      else if( old_C1frequency != m_C1frequency ) write = true;
      else if( old_C1vpp != m_C1vpp ) write = true;
      else if( old_C1ofst != m_C1ofst ) write = true;
      else if( old_C1phse != m_C1phse ) write = true;
      else if( old_C1wdth != m_C1wdth ) write = true;
      else if( old_C1wvtp != m_C1wvtp ) write = true;
      else if( old_C1sync != m_C1sync ) write = true;
      else if( old_C2outp != m_C2outp ) write = true;
      else if( old_C2frequency != m_C2frequency ) write = true;
      else if( old_C2vpp != m_C2vpp ) write = true;
      else if( old_C2ofst != m_C2ofst ) write = true;
      else if( old_C2phse != m_C2phse ) write = true;
      else if( old_C2wdth != m_C2wdth ) write = true;
      else if( old_C2wvtp != m_C2wvtp ) write = true;
      else if( old_C2sync != m_C2sync ) write = true;
   }

   // todo: add if statement for all of the write??

   if(force || write)
   {
      uint8_t C1wvtp = 3;
      if(m_C1wvtp == "DC") C1wvtp = TELEM_FXNGEN_WVTP_DC;
      else if(m_C1wvtp == "SINE") C1wvtp = TELEM_FXNGEN_WVTP_SINE;
      else if(m_C1wvtp == "PULSE") C1wvtp = TELEM_FXNGEN_WVTP_PULSE;

      uint8_t C2wvtp = 3;
      if(m_C2wvtp == "DC") C2wvtp = TELEM_FXNGEN_WVTP_DC;
      else if(m_C2wvtp == "SINE") C2wvtp = TELEM_FXNGEN_WVTP_SINE;
      else if(m_C2wvtp == "PULSE") C2wvtp = TELEM_FXNGEN_WVTP_PULSE;

      telem<telem_fxngen>({m_C1outp, m_C1frequency, m_C1vpp, m_C1ofst, m_C1phse, C1wvtp,
                             m_C2outp, m_C2frequency, m_C2vpp, m_C2ofst, m_C2phse, C2wvtp,
                                m_C1sync, m_C2sync, m_C1wdth, m_C2wdth});

      old_C1outp = m_C1outp;
      old_C1frequency = m_C1frequency;
      old_C1vpp = m_C1vpp;
      old_C1ofst = m_C1ofst;
      old_C1phse = m_C1phse;
      old_C1wdth = m_C1wdth;
      old_C1wvtp = m_C1wvtp;
      old_C1sync = m_C1sync;

      old_C2outp = m_C2outp;
      old_C2frequency = m_C2frequency;
      old_C2vpp = m_C2vpp;
      old_C2ofst = m_C2ofst;
      old_C2phse = m_C2phse;
      old_C2wdth = m_C2wdth;
      old_C2wvtp = m_C2wvtp;
      old_C2sync = m_C2sync;
   }

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //siglentSDG_hpp