/** \file trippLitePDU.hpp
  * \brief The MagAO-X Tripp Lite Power Distribution Unit controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup trippLitePDU_files
  */

#ifndef trippLitePDU_hpp
#define trippLitePDU_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#ifdef XWC_SIM_MODE

struct trippLitePDU_simulator
{
    float m_voltage {120};
    float m_frequency {60};
    float m_lowTransferVoltage {70};

    float m_current {4};

    std::vector<int> m_outlets;

    trippLitePDU_simulator()
    {
        m_outlets.resize(8,false);
    }

    int connect( const std::string & ipAddr,
                 const std::string & port 
               )
    {
        static_cast<void>(ipAddr);
        static_cast<void>(port);

        return 0;
    }

    int login( const std::string & user,
               const std::string & pass 
             )
    {
        static_cast<void>(user);
        static_cast<void>(pass);

        return 0;
    }


    void postLogin()
    {
    }

    int turnOutletOn( uint16_t outletNum )
    {
        if(outletNum >= m_outlets.size())
        {
            return -1;
        }

        m_outlets[outletNum] = 1;

        return 0;
    }

    int turnOutletOff( uint16_t outletNum )
    {
        if(outletNum >= m_outlets.size())
        {
            return -1;
        }

        m_outlets[outletNum] = 0;

        return 0;
    }

    int devStatus(std::string & strRead)
    {
        char vstr[64];
        snprintf(vstr, sizeof(vstr), "%0.1f", m_voltage);

        char fstr[64];
        snprintf(fstr, sizeof(fstr), "%0.1f", m_frequency);

        char tvstr[64];
        snprintf(tvstr, sizeof(tvstr), "%0.1f", m_lowTransferVoltage);

        char cstr[64];
        snprintf(cstr, sizeof(cstr), "%0.2f", m_current);

        strRead =  "-------------------------------------------------------------------------------\n";
        strRead += "01: PDUMH20NET2LX 'Device0062'\n";
        strRead += "--------------------------------------------------------------------------------\n";
        strRead += "Device Type:                    PDU\n";
        strRead += "Device Status:                  WARNING        !\n";
        strRead += "\n";
        strRead += "Input Voltage:                  " + std::string(vstr) + " V    \n"     ;
        strRead += "Input Frequency:                " + std::string(fstr) + " Hz       \n"  ;
        strRead += "Low Transfer Voltage:           " + std::string(tvstr) + " V          \n";
        strRead += "\n";
        strRead += "Output Current:                 " + std::string(cstr) + " A - Total  \n";
        strRead += "\n";
        strRead += "Outlets On:                     ";

        //Print outlet numbers of on-outlets, with no space at beginning or end.
        bool prev = false;
        bool none = true;
        for(size_t n=0; n < m_outlets.size(); ++n)
        {
            if(m_outlets[n])
            {
                if(prev) strRead += " ";
                strRead += std::to_string(n+1);
                prev = true;
                none = false;
            }
        }
        if(none)
        {
            strRead += "NONE";
        }

        strRead += "\n";
        
        return 0;
    }
};

#endif

/** \defgroup trippLitePDU Tripp Lite PDU
  * \brief Control of MagAO-X Tripp Lite PDUs.
  *
  * <a href="../handbook/operating/software/apps/trippLitePDU.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup trippLitePDU_files Tripp Lite PDU Files
  * \ingroup trippLitePDU
  */

namespace MagAOX
{
namespace app
{

/// MagAO-X application to control a Tripp Lite power distribution unit (PDU).
/** The device outlets are organized into channels.  See \ref dev::outletController for details of configuring the channels.
  *
  * The line frequency and voltage, and the total load on the PDU, are monitored.
  *
  * \todo need username and secure password handling
  * \todo need to recognize signals in tty polls and not return errors, etc.
  * \todo begin logging freq/volt/amps telemetry
  * \todo research load warnings
  * \todo tests for parser
  * \todo test for load warnings
  * \todo load warnings/crit values can be logged on parse errors -- make this an issue
  * \todo segfaults if device can not be reached on network -- make this an issue
  * 
  * \ingroup trippLitePDU
  */
class trippLitePDU : public MagAOXApp<>, public dev::outletController<trippLitePDU>, public dev::ioDevice
{

protected:

   std::string m_deviceAddr; ///< The device address
   std::string m_devicePort; ///< The device port
   std::string m_deviceUsername; ///< The login username for this device
   std::string m_devicePassFile; ///< The login password for this device
   int m_deviceVersion {0}; ///< Version 0 = the old PDUs, version 1 = new PDUMH15NET2LX, which is a new login procedure to get to the CLI.

   float m_freqLowWarn {59};    ///< The low-frequency warning threshold
   float m_freqHighWarn {61};   ///< The high-frequency warning threshold

   float m_freqLowAlert {58};   ///< The low-frequency alert threshold
   float m_freqHighAlert {62};  ///< The high-frequency alert threshold

   float m_freqLowEmerg {57};   ///< The low-frequency emergency threshold
   float m_freqHighEmerg {63};  ///< The high-frequency emergency threshold

   float m_voltLowWarn {105};   ///< The low-voltage warning threshold
   float m_voltHighWarn {125};  ///< The high-voltage warning threshold

   float m_voltLowAlert {101};  ///< The low-voltage alert threshold
   float m_voltHighAlert {126}; ///< The high-voltage alert threshold

   float m_voltLowEmerg {99};   ///< The low-voltage emergency threshold
   float m_voltHighEmerg {128}; ///< The high-voltage emergency threshold

   float m_currWarn {15};  ///< The high-current warning threshold
   float m_currAlert {16}; ///< The high-current alert threshold
   float m_currEmerg {20}; ///< The high-current emergency threshold

   tty::telnetConn m_telnetConn; ///< The telnet connection manager

   std::string m_status; ///< The device status
   float m_frequency {0}; ///< The line frequency reported by the device.
   float m_voltage {0}; ///< The line voltage reported by the device.
   float m_current {0}; ///< The current being reported by the device.

public:

    /// Default c'tor.
    trippLitePDU();
 
    /// D'tor, declared and defined for noexcept.
    ~trippLitePDU() noexcept
    {}
 
    /** \name MagAOXApp Interface
      *
      * @{ 
      */

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
 
    ///@}

    /** \name outletController Interface
      *
      * @{ 
      */

    /// Update a single outlet state
    /** For the trippLitePDU this isn't possible for a single outlet, so this calls updateOutletStates.
      *
      * \returns 0 on success
      * \returns -1 on error
      */
    virtual int updateOutletState( int outletNum /**< [in] the outlet number to update */);
 
    /// Queries the device to find the state of each outlet, as well as other parameters.
    /** Sends `devstatus` to the device, and parses the result.
      *
      * \returns 0 on success
      * \returns -1 on error
      */
    virtual int updateOutletStates();
 
    /// Turn on an outlet.
    /**
      * \returns 0 on success
      * \returns -1 on error
      */
    virtual int turnOutletOn( int outletNum /**< [in] the outlet number to turn on */);
 
    /// Turn off an outlet.
    /**
      * \returns 0 on success
      * \returns -1 on error
      */
    virtual int turnOutletOff( int outletNum /**< [in] the outlet number to turn off */);
 
    ///@}

    /** \name Device Interface 
      *
      * These functions invoke the simulator code when enabled. 
      *
      * @{
      */
    int devConnect();
    
    int devLogin();

    void devPostLogin();

    int devStatus(std::string & strRead);

    /// Parse the PDU devstatus response.
    /**
      * \returns 0 on success
      * \returns \<0 on error, with value indicating location of error.
      */
    int parsePDUStatus( std::string & strRead );

    ///@}

    void updateAlarmsAndWarnings();

protected:

   //declare our properties
   pcf::IndiProperty m_indiP_status; ///< The device's status string
   pcf::IndiProperty m_indiP_load; ///< The line and load characteristics


#ifdef XWC_SIM_MODE

public:

    trippLitePDU_simulator m_simulator;

#endif

};

trippLitePDU::trippLitePDU() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_firstOne = true;
   setNumberOfOutlets(8);
   m_loopPause=2000000000;//Default to 2 sec loop pause to lessen the load on the PDUs.
   
   return;
}

void trippLitePDU::setupConfig()
{
   config.add("device.address", "a", "device.address", argType::Required, "device", "address", false, "string", "The device address.");
   config.add("device.port", "p", "device.port", argType::Required, "device", "port", false, "string", "The device port.");
   config.add("device.username", "u", "device.username", argType::Required, "device", "username", false, "string", "The device login username.");
   config.add("device.passfile", "", "device.passfile", argType::Required, "device", "passfile", false, "string", "The device login password file (relative to secrets dir).");
   config.add("device.powerAlertVersion", "", "device.powerAlertVersion", argType::Required, "device", "powerAlertVersion", false, "int", "The device network interface version.  0 is PDU..., 1 is newer LX platform.");

   dev::ioDevice::setupConfig(config);

   config.add("limits.freqLowWarn", "", "limits.freqLowWarn", argType::Required, "limits", "freqLowWarn", false, "int", "The low-frequency warning threshold");
   config.add("limits.freqHighWarn", "", "limits.freqHighWarn", argType::Required, "limits", "freqHighWarn", false, "int", "The high-frequency warning threshold");
   config.add("limits.freqLowAlert", "", "limits.freqLowAlert", argType::Required, "limits", "freqLowAlert", false, "int", "The low-frequency alert threshold");
   config.add("limits.freqHighAlert", "", "limits.freqHighAlert", argType::Required, "limits", "freqHighAlert", false, "int", "The high-frequency alert threshold");
   config.add("limits.freqLowEmerg", "", "limits.freqLowEmerg", argType::Required, "limits", "freqLowEmerg", false, "int", "The low-frequency emergency threshold");
   config.add("limits.freqHighEmerg", "", "limits.freqHighEmerg", argType::Required, "limits", "freqHighEmerg", false, "int", "The high-frequency emergency threshold");

   config.add("limits.voltLowWarn", "", "limits.voltLowWarn", argType::Required, "limits", "voltLowWarn", false, "int", "The low-voltage warning threshold");
   config.add("limits.voltHighWarn", "", "limits.voltHighWarn", argType::Required, "limits", "voltHighWarn", false, "int", "The high-voltage warning threshold");
   config.add("limits.voltLowAlert", "", "limits.voltLowAlert", argType::Required, "limits", "voltLowAlert", false, "int", "The low-voltage alert threshold");
   config.add("limits.voltHighAlert", "", "limits.voltHighAlert", argType::Required, "limits", "voltHighAlert", false, "int", "The high-voltage alert threshold");
   config.add("limits.voltLowEmerg", "", "limits.voltLowEmerg", argType::Required, "limits", "voltLowEmerg", false, "int", "The low-voltage emergency threshold");
   config.add("limits.voltHighEmerg", "", "limits.voltHighEmerg", argType::Required, "limits", "voltHighEmerg", false, "int", "The high-voltage emergency threshold");

   config.add("limits.currWarn", "", "limits.currWarn", argType::Required, "limits", "currWarn", false, "int", "The high-current warning threshold");
   config.add("limits.currAlert", "", "limits.currAlert", argType::Required, "limits", "currAlert", false, "int", "The high-current alert threshold");
   config.add("limits.currEmerg", "", "limits.currEmerg", argType::Required, "limits", "currEmerg", false, "int", "The high-current emergency threshold");

   dev::outletController<trippLitePDU>::setupConfig(config);
   
}


void trippLitePDU::loadConfig()
{
   config(m_deviceAddr, "device.address");
   config(m_devicePort, "device.port");
   config(m_deviceUsername, "device.username");
   config(m_devicePassFile, "device.passfile");
   config(m_deviceVersion, "device.powerAlertVersion");
   
   dev::ioDevice::loadConfig(config);

   config(m_freqLowWarn, "limits.freqLowWarn");
   config(m_freqHighWarn, "limits.freqHighWarn");
   config(m_freqLowAlert, "limits.freqLowAlert");
   config(m_freqHighAlert, "limits.freqHighAlert");
   config(m_freqLowEmerg, "limits.freqLowEmerg");
   config(m_freqHighEmerg, "limits.freqHighEmerg");

   config(m_voltLowWarn, "limits.voltLowWarn");
   config(m_voltHighWarn, "limits.voltHighWarn");
   config(m_voltLowAlert, "limits.voltLowAlert");
   config(m_voltHighAlert, "limits.voltHighAlert");
   config(m_voltLowEmerg, "limits.voltLowEmerg");
   config(m_voltHighEmerg, "limits.voltHighEmerg");

   config(m_currWarn, "limits.currWarn");
   config(m_currAlert, "limits.currAlert");
   config(m_currEmerg, "limits.currEmerg");

   dev::outletController<trippLitePDU>::loadConfig(config);
   

}

int trippLitePDU::appStartup()
{
    #ifdef XWC_SIM_MODE
    log<text_log>("XWC_SIM_MODE active");
    #endif

    // set up the  INDI properties
    REG_INDI_NEWPROP_NOCB(m_indiP_status, "status", pcf::IndiProperty::Text);
    m_indiP_status.add (pcf::IndiElement("value"));

    REG_INDI_NEWPROP_NOCB(m_indiP_load, "load", pcf::IndiProperty::Number);
    m_indiP_load.add (pcf::IndiElement("frequency"));
    m_indiP_load.add (pcf::IndiElement("voltage"));
    m_indiP_load.add (pcf::IndiElement("current"));

    if(dev::outletController<trippLitePDU>::setupINDI() < 0)
    {
        return log<text_log,-1>("Error setting up INDI for outlet control.", logPrio::LOG_CRITICAL);
    }

    state(stateCodes::NOTCONNECTED);

    return 0;
}

int trippLitePDU::appLogic()
{
    if( state() == stateCodes::NOTCONNECTED )
    {
        static int lastrv = 0; //Used to handle a change in error within the same state.  Make general?
        static int lasterrno = 0;
         
        int rv = devConnect();

        if(rv == 0)
        {
            state(stateCodes::CONNECTED);

            if(!stateLogged())
            {
                std::string logs = "Connected to " + m_deviceAddr + ":" + m_devicePort;
                log<text_log>(logs);
            }
            lastrv = rv;
            lasterrno = errno;
        }
        else
        {
            if(!stateLogged())
            {
               log<text_log>({"Failed to connect to " + m_deviceAddr + ":" + m_devicePort}, logPrio::LOG_ERROR);
            }
            if( rv != lastrv )
            {
               log<software_error>( {__FILE__,__LINE__, 0, rv,  tty::ttyErrorString(rv)} );
               lastrv = rv;
            }
            if( errno != lasterrno )
            {
               log<software_error>( {__FILE__,__LINE__, errno});
               lasterrno = errno;
            }
            return 0;
        }
    }
 
    if( state() == stateCodes::CONNECTED )
    {
        int rv = devLogin();

        if(rv == 0)
        {
            state(stateCodes::LOGGEDIN);
        }
        else
        {
           if(rv == TELNET_E_LOGINTIMEOUT)
           {
                state(stateCodes::NOTCONNECTED);
                log<text_log>("login timedout", logPrio::LOG_ERROR);
                return 0;
           }

           state(stateCodes::FAILURE);
           log<text_log>("login failure", logPrio::LOG_CRITICAL);
           return -1;
        }
    }
 
    if(state() == stateCodes::LOGGEDIN)
    {
        devPostLogin();

        state(stateCodes::READY);
    }
 
    if(state() == stateCodes::READY)
    {
       std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);
 
       if( !lock.owns_lock())
       {
          return 0;
       }
 
       int rv = updateOutletStates();
 
       if(rv < 0) return log<software_error,-1>({__FILE__, __LINE__});
 
       
 
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

int trippLitePDU::updateOutletState( int outletNum )
{
   static_cast<void>(outletNum);

   return updateOutletStates(); //We can't do just one.
}


int trippLitePDU::updateOutletStates()
{
    int rv;
    std::string strRead;

    rv = devStatus(strRead);

    if(rv < 0 )
    {
        log<software_error>({__FILE__, __LINE__, "error getting device status"});
        state(stateCodes::NOTCONNECTED);
        return 0;
    }

    if(rv > 0)
    {
        return 0; //this means the re-read was successful, but we don't want to parse this time.
    }

    rv = parsePDUStatus( strRead);

    if(rv == 0)
    {
        updateIfChanged(m_indiP_status, "value", m_status);

        updateIfChanged(m_indiP_load, "frequency", m_frequency);

        updateIfChanged(m_indiP_load, "voltage", m_voltage);

        updateIfChanged(m_indiP_load, "current", m_current);

        dev::outletController<trippLitePDU>::updateINDI();
    }
    else
    {
        log<software_error>({__FILE__, __LINE__, 0, rv, "parse error"});
    }

    return 0;
}

int trippLitePDU::turnOutletOn( int outletNum )
{
    std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before doing anything

    #ifndef XWC_SIM_MODE
    std::string cmd = "loadctl on -o ";
    cmd += mx::ioutils::convertToString<int>(outletNum+1); //Internally 0 counted, device starts at 1.
    cmd += " --force\r";

    int rv = m_telnetConn.writeRead( cmd, true, m_writeTimeout, m_readTimeout);

    #else 

    int rv = m_simulator.turnOutletOn(outletNum);

    #endif

    if(rv < 0) return log<software_error, -1>({__FILE__, __LINE__, 0, rv, "telnet error"});

    return 0;
}

int trippLitePDU::turnOutletOff( int outletNum )
{
    std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before doing anything

    #ifndef XWC_SIM_MODE
    std::string cmd = "loadctl off -o ";
    cmd += mx::ioutils::convertToString<int>(outletNum+1); //Internally 0 counted, device starts at 1.
    cmd += " --force\r";

    int rv = m_telnetConn.writeRead( cmd, true, m_writeTimeout, m_readTimeout);

    #else

    int rv = m_simulator.turnOutletOff(outletNum);

    #endif

    if(rv < 0) return log<software_error, -1>({__FILE__, __LINE__, 0, rv, "telnet error"});

   return 0;
}

int trippLitePDU::devConnect()
{
    #ifndef XWC_SIM_MODE

    return m_telnetConn.connect(m_deviceAddr, m_devicePort);

    #else

    return m_simulator.connect(m_deviceAddr, m_devicePort);
        
    #endif
}
    
int trippLitePDU::devLogin()
{
    #ifndef XWC_SIM_MODE

    //Newer version of power alert changed login (at least the first one)
    if(m_deviceVersion > 0)
    {
        m_telnetConn.m_usernamePrompt = "login:";
        m_telnetConn.m_prompt = ">>";
    }

    return m_telnetConn.login("localadmin", "localadmin");    
        
    #else
        
    return m_simulator.login("localadmin", "localadmin");
        
    #endif
}

void trippLitePDU::devPostLogin()
{

    #ifndef XWC_SIM_MODE
  
    //For newer version of power alert we need to select C.L.I.
    if(m_deviceVersion > 0)
    {
        m_telnetConn.writeRead("E\n", false, m_writeTimeout, m_readTimeout);
        m_telnetConn.m_prompt = "$> ";
    }

    #else

    m_simulator.postLogin();

    #endif
}

int trippLitePDU::devStatus(std::string & strRead)
{
    #ifndef XWC_SIM_MODE

    int rv = m_telnetConn.writeRead("devstatus\n", true, m_writeTimeout, m_readTimeout);

    strRead = m_telnetConn.m_strRead;

    if(rv == TTY_E_TIMEOUTONREAD || rv == TTY_E_TIMEOUTONREADPOLL)
    {
        rv = m_telnetConn.read(m_readTimeout, false);

        if( rv < 0 )
        {
            log<software_error>({__FILE__, __LINE__, 0, rv, "devstatus timeout, timed out on re-read: " + tty::ttyErrorString(rv)});
            return -1;
        }

        log<text_log>("devstatus timeout, re-read successful");
      
        return 1;
    }
    else if(rv < 0 )
    {
        log<software_error>({__FILE__, __LINE__, 0, rv, tty::ttyErrorString(rv)});
        return -1;
    }

    return 0;

    #else

    return m_simulator.devStatus(strRead);

    #endif
}

int trippLitePDU::parsePDUStatus( std::string & strRead )
{  
    size_t curpos = 0;

    curpos = strRead.find_first_of("\r\n", curpos);

    std::string sstr;

    while(curpos < strRead.size())
    {
        size_t eol = strRead.find_first_of("\r\n", curpos);

        if(eol == std::string::npos) eol = strRead.size();

        if(eol == curpos)
        {
            curpos = eol + 1;
            continue;
        }

        sstr = strRead.substr(curpos, eol-curpos);
        curpos = eol + 1;

        if(sstr[0] == '-' || sstr[0] == '0' || sstr[0] == 'L' || sstr[0] == ' ' || sstr[0] == 'D' || sstr[0] == '$') continue;

        if(sstr[0] == 'I')
        {
            if(sstr[6] == 'V') 
            {
                size_t begin = sstr.find(' ',6);
                if(begin == std::string::npos) 
                {
                    return -1;
                }

                begin = sstr.find_first_not_of(' ', begin);
                if(begin == std::string::npos) 
                {
                    return -2;
                }

                size_t end = sstr.find('V', begin);
                if(end == std::string::npos) 
                {
                    return -3;
                }

                float V = mx::ioutils::convertFromString<float>( sstr.substr(begin, end-begin) );

                m_voltage = V;
            }

            else if(sstr[6] == 'F') 
            {
                size_t begin = sstr.find(' ',6);
                if(begin == std::string::npos) 
                {
                    return -4;
                }

                begin = sstr.find_first_not_of(' ', begin);
                if(begin == std::string::npos) 
                {
                    return -5;
                }

                size_t end = sstr.find('H', begin);
                if(end == std::string::npos) 
                {
                    return -6;
                }

                float F = mx::ioutils::convertFromString<float>( sstr.substr(begin, end-begin) );

                m_frequency = F;
            }
            else return -1;
        }
        else if(sstr[0] == 'O')
        {
            if(sstr[7] == 'C') 
            {
                size_t begin = sstr.find(' ',7);
                if(begin == std::string::npos) 
                {
                    return -7;
                }

                begin = sstr.find_first_not_of(' ', begin);
                if(begin == std::string::npos) 
                {
                    return -8;
                }

                size_t end = sstr.find('A', begin);
                if(end == std::string::npos) 
                {
                    return -9;
                }

                float C = mx::ioutils::convertFromString<float>( sstr.substr(begin, end-begin) );

                m_current = C;
            }
            else if(sstr[8] == 'O') 
            {
                std::vector<int> outletStates(m_outletStates.size(),OUTLET_STATE_OFF);

                size_t begin = sstr.find(' ',8);
                if(begin == std::string::npos) 
                {
                    return -10; 
                }

                begin = sstr.find_first_not_of(' ', begin);

                if(begin == std::string::npos) 
                {
                    return -11;
                }

                while(begin < sstr.size())
                {
                    size_t end = sstr.find(' ', begin);
                    if(end == std::string::npos) 
                    {
                        end = sstr.size();
                    }

                    int onum = atoi(sstr.substr(begin, end-begin).c_str());

                    if(onum > 0 && onum < 9)
                    {
                        outletStates[onum-1] = OUTLET_STATE_ON; //this outlet is on.
                    }
                    begin = sstr.find_first_not_of(' ', end+1);
                }

                for(size_t i=0;i<m_outletStates.size();++i) 
                {
                    m_outletStates[i]=outletStates[i];
                }
            }
            else if( sstr[7] == 'V' || sstr[7] == 'F') 
            {
                continue;
            }
            else 
            {
                return -12;
            }
        }
        else 
        {
            return -13;
        }
    }

    return 0;
}

void trippLitePDU::updateAlarmsAndWarnings()
{
    if (m_frequency <= m_freqLowEmerg)
    {
        log<text_log>("Frequency is " + std::to_string(m_frequency) + " Hz, below " + 
                              std::to_string(m_freqLowEmerg) + " Hz.",  logPrio::LOG_EMERGENCY);
    }
    else if (m_frequency >= m_freqHighEmerg)
    {
        log<text_log>("Frequency is " + std::to_string(m_frequency) + " Hz, above " + 
                             std::to_string(m_freqHighEmerg) + " Hz.",  logPrio::LOG_EMERGENCY);
    }
    else if (m_frequency <= m_freqLowAlert)
    {
        log<text_log>("Frequency is " + std::to_string(m_frequency) + " Hz, below " + 
                                  std::to_string(m_freqLowAlert) + " Hz.",  logPrio::LOG_ALERT);
    }
    else if (m_frequency >= m_freqHighAlert)
    {
        log<text_log>("Frequency is " + std::to_string(m_frequency) + " Hz, above " + 
                                  std::to_string(m_freqHighAlert) + " Hz.",  logPrio::LOG_ALERT);
    }
    else if(m_frequency <= m_freqLowWarn)
    {
        log<text_log>("Frequency is " + std::to_string(m_frequency) + " Hz, below " + 
                                    std::to_string(m_freqLowWarn) + " Hz.",  logPrio::LOG_WARNING);
    }
    else if (m_frequency >= m_freqHighWarn)
    {
        log<text_log>("Frequency is " + std::to_string(m_frequency) + " Hz, above " + 
                                            std::to_string(m_freqHighWarn) + " Hz.",  logPrio::LOG_WARNING);
    }
 
    if (m_voltage <= m_voltLowEmerg)
    {
        log<text_log>("Voltage is " + std::to_string(m_voltage) + " V, below " + 
                                            std::to_string(m_voltLowEmerg) + " V.",  logPrio::LOG_EMERGENCY);
    }
    else if (m_voltage >= m_voltHighEmerg)
    {
        log<text_log>("Voltage is " + std::to_string(m_voltage) + " V, above " + 
                                            std::to_string(m_voltHighEmerg) + " V.",  logPrio::LOG_EMERGENCY);
    }
    else if (m_voltage <= m_voltLowAlert)
    {
        log<text_log>("Voltage is " + std::to_string(m_voltage) + " V, below " + 
                                            std::to_string(m_voltLowAlert) + " V.",  logPrio::LOG_ALERT);
    }
    else if (m_voltage >= m_voltHighAlert)
    {
        log<text_log>("Voltage is " + std::to_string(m_voltage) + " V, above " + 
                                                        std::to_string(m_voltHighAlert) + " V.",  logPrio::LOG_ALERT);
    }
    else if(m_voltage <= m_voltLowWarn)
    {
        log<text_log>("Voltage is " + std::to_string(m_voltage) + " V, below " + 
                                                std::to_string(m_voltLowWarn) + " V.",  logPrio::LOG_WARNING);
    }
    else if (m_voltage >= m_voltHighWarn)
    {
        log<text_log>("Voltage is " + std::to_string(m_voltage) + " V, above " + 
                                                std::to_string(m_voltHighWarn) + " V.",  logPrio::LOG_WARNING);
    }
 
    if (m_current >= m_currEmerg)
    {
        log<text_log>("Current is " + std::to_string(m_current) + " A, above " + 
                                                    std::to_string(m_currEmerg) + " A.",  logPrio::LOG_EMERGENCY);
    }
    else if (m_current >= m_currAlert)
    {
        log<text_log>("Current is " + std::to_string(m_current) + " A, above " + 
                                                        std::to_string(m_currAlert) + " A.",  logPrio::LOG_ALERT);
    }
    else if (m_current >= m_currWarn)
    {
        log<text_log>("Current is " + std::to_string(m_current) + " A, above " + 
                                                            std::to_string(m_currWarn) + " A.",  logPrio::LOG_WARNING);
    }
}

} //namespace app
} //namespace MagAOX

#endif //trippLitePDU_hpp
