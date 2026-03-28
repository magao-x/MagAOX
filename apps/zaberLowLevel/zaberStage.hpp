/** \file zaberStage.hpp
 * \brief A class with details of a single zaber stage
 *
 * \ingroup zaberLowLevel_files
 */

#ifndef zaberStage_hpp
#define zaberStage_hpp

#include <iostream>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch

#include "za_serial.h"

namespace MagAOX
{
namespace app
{

/// A class to manage the details of one stage in a Zaber system.
/**
 * \ingroup zaberLowLevel
 */
template <class parentT>
class zaberStage
{
  protected:
    parentT *m_parent{ nullptr }; // The parent MagAOXApp

    std::string m_name; ///< The stage's name.

    std::string m_serial; ///< The stage's serial number.

    int m_deviceAddress{ -1 }; ///< The device's address, a.k.a. its order in the chain

    int m_axisNumber{ 0 }; ///< The axis number at the address (normally 0 in MagAO-X)

    bool m_commandStatus{ true }; ///< The status of the last command sent. true = OK, false = RJ (rejected)

    char m_deviceStatus{
        'U' }; ///< Current status.  Either 'I' for IDLE or 'B' for BUSY.  Intializes to 'U' for UNKOWN.

    bool m_homing{ false };

    timespec m_lastHomed{ 0, 0 }; ///< Time stamp of the last time the stage was homed

    bool m_parked{ false };

    long m_rawPos; ///< The raw position reported by the device, in microsteps.

    long m_tgtPos{ 0 }; ///< The tgt position last sent to the device, in microsteps.

    long m_maxPos; ///< The max position allowed for the device, set by config.  Will be set to no larger m_maxPosHW.

    float m_temp{ -999 }; ///< The driver temperature, in C.

    bool m_knobEnabled{ false };

    bool m_ledEnabled{ false };

    bool m_warn{ false };

    bool m_warnFD{ false };
    bool m_warnFDreported{ false };
    bool m_warnFQ{ false };
    bool m_warnFQreported{ false };
    bool m_warnFS{ false };
    bool m_warnFSreported{ false };
    bool m_warnFT{ false };
    bool m_warnFTreported{ false };
    bool m_warnFB{ false };
    bool m_warnFBreported{ false };
    bool m_warnFP{ false };
    bool m_warnFPreported{ false };
    bool m_warnFE{ false };
    bool m_warnFEreported{ false };
    bool m_warnWH{ false };
    bool m_warnWHreported{ false };
    bool m_warnWL{ false };
    bool m_warnWLreported{ false };
    bool m_warnWP{ false };
    bool m_warnWPreported{ false };
    bool m_warnWV{ false };
    bool m_warnWVreported{ false };
    bool m_warnWT{ false };
    bool m_warnWTreported{ false };
    bool m_warnWM{ false };
    bool m_warnWMreported{ false };
    bool m_warnWR{ false };
    bool m_warnWRreported{ false };
    bool m_warnNC{ false };
    bool m_warnNCreported{ false };
    bool m_warnNI{ false };
    bool m_warnNIreported{ false };
    bool m_warnND{ false };
    bool m_warnNDreported{ false };
    bool m_warnNU{ false };
    bool m_warnNUreported{ false };
    bool m_warnNJ{ false };
    bool m_warnNJreported{ false };
    bool m_warnUNK{ false };

  public:
    zaberStage() = delete;

    zaberStage( parentT *parent )
    {
        if( parent == nullptr )
        {
            throw mx::exception( mx::error_t::invalidarg, "parent was null on construction of zaberState" );
        }

        m_parent = parent;
    }

    /// Get the device name
    /**
     * \returns the current value of m_name
     */
    std::string name();

    /// Set the device name
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int name( const std::string &n /**< [in] the new device name*/ );

    /// Get the device serial  number
    /**
     * \returns the current value of m_serial
     */
    std::string serial();

    /// Set the device serial
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int serial( const std::string &s /**< [in] the new device serial*/ );

    /// Get the device address
    /**
     * \returns the current value of m_deviceAddress
     */
    int deviceAddress();

    /// Set the device address
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int deviceAddress( const int &da /**< [in] the new device address*/ );

    /// Get the axis number
    /**
     * \returns the current value of m_axisNumber
     */
    int axisNumber();

    /// Set the axis number
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int axisNumber( const int &an /**< [in] the new axis number */ );

    /// Get the command status
    /**
     * \returns the current value of m_commandStatus
     */
    bool commandStatus();

    /// Get the device status
    /**
     * \returns the current value of m_deviceStatus
     */
    char deviceStatus();

    /// Get the homing status
    /**
     * \returns the current value of m_homing
     */
    bool homing();

    /// Get the time of last homing
    /**
     * \returns the current value of m_lastHomed
     */
    time_t lastHomed();

    /// Get the parked status
    /**
     * \returns the current value of m_parked
     */
    int parked();

    /// Get the knob status
    /**
     * \returns the current value of m_knobEnabled
     */
    bool knobEnabled();

    /// Get the LED status
    /**
     * \returns the current value of m_ledEnabled
     */
    bool ledEnabled();

    /// Get the current raw position, in counts
    /**
     * \returns the current value of m_rawPos
     */
    long rawPos();

    /// Get the current tgt position, in counts
    /**
     * \returns the current value of m_tgtPos
     */
    long tgtPos();

    /// Get the max position, in counts
    /**
     * \returns the current value of m_maxPos
     */
    long maxPos();

    /// Get the status of the warning flag
    /**
     * \returns the current value of m_warn
     */
    bool warn();

    /// Get the temperature, in C
    /**
     * \returns the current value of m_temp
     */
    float temp();

    /// Get the warning state
    /**
     * \returns the true if any warning flags are set.
     */
    bool warningState();

    bool warnFD();
    bool warnFQ();
    bool warnFS();
    bool warnFT();
    bool warnFB();
    bool warnFP();
    bool warnFE();
    bool warnWH();
    bool warnWL();
    bool warnWP();
    bool warnWV();
    bool warnWT();
    bool warnWM();
    bool warnWR();
    bool warnNC();
    bool warnNI();
    bool warnND();
    bool warnNU();
    bool warnNJ();
    bool warnUNK();

    /// Get a response from the device, after a command has been sent.
    /** Parses the standard parts of the response in this stage's fields,
     * and extracts the response string.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int getResponse( std::string       &response, ///< [out]  the text response
                     const std::string &repBuff   ///< [in] the reply buffer, not decoded.
    );

    /// Get a response from the device, after a command has been sent.
    /** Parses the standard parts of the response in this stages fields,
     * and extracts the response string.
     *
     * \overload
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int getResponse( std::string    &response, ///< [out] the text response component
                     const za_reply &rep       ///< [in] the decodedstage reply
    );

    /// Determine whether a decoded message is the awaited command reply.
    bool isCommandReply( const za_reply &rep /**< [in] the decoded device message */ );

    /// Send a command and get the response
    int sendCommand( std::string       &response, ///< [out] the response received from the stage
                     z_port             port,     ///< [in]  the port with which to communicate
                     const std::string &command   ///< [in] the command to send
    );

    /// Send a command for which no response is expected
    int sendCommand( z_port             port,   /**< [in] the port with which to communicate */
                     const std::string &command /**< [in] the command to send */
    );

    /// Get a value for this device
    /** Sends the command specified and converts the response to the specified type.
     *
     * The command passed should include `get` if needed.
     *
     * \tparam valT is the value type to convert to.
     */
    template <typename valT>
    int getValue( valT             &val,    /**< [out] the value to update */
                  z_port            port,   /**< [in] the port with which to communicate */
                  const std::string command /**< [in] the command to send */
    );

    /// Get the maximum position from the stage
    int getMaxPos( z_port port /**< [in] the port with which to communicate */ );

    /// Get the parked state from the stage
    int getParked( z_port port /**< [in] the port with which to communicate */ );

    int getKnob( z_port port /**< [in] the port with which to communicate */ );

    int getLED( z_port port /**< [in] the port with which to communicate */ );

    /// Update the position of the stage
    int updatePos( z_port port /**< [in] the port with which to communicate */ );

    /// Update the stage temperature
    int updateTemp( z_port port /**< [in] the port with which to communicate */ );

    /// Enable/Disable the knob
    int enableKnob( z_port port, bool enable );

    // Enable/disable LED
    int enableLED( z_port port, bool enable );

    /// Stop the stage
    int stop( z_port port );

    /// Emergency stop the stage
    int estop( z_port port );

    /// Initiate homing
    int home( z_port port /**< [in] the port with which to communicate */ );

    /// Park the stage
    int park( z_port port /**< [in] the port with which to communicate */ );

    /// Unpark the stage
    int unpark( z_port port /**< [in] the port with which to communicate */ );

    /// Move to a new absolute position
    int moveAbs( z_port port,  ///< [in] the port with which to communicate
                 long   rawPos ///< [in] the position to move to, in counts
    );

    /// Sets all warning flags to false
    /** This is not the same as clearing warnings on the device, this is just used for
     * bookkeeping.
     *
     * \returns 0 on success (always)
     */
    int unsetWarnings();

    /// Process a single warning from the device, setting the appropriate flag.
    /** Warnings are two ASCII characeters, e.g. "WR".
     *
     * \returns 0 if the warning is processed, including if it's not recognized.
     * \returns -1 on an error, currently not possible.
     */
    int processWarning( std::string &warn /**< [in] the two-character warning flag */ );

    /// Parse the warning response from the device.
    /** Sends each warning flag to processWarning.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int parseWarnings( std::string &response /**< [in] the response from the warnings query*/ );

    /// Get warnings from the device
    /** Log entries will be made and flags will be set in this structure.
     *
     * \returns 0 on success
     * \returns -1 on error.
     */
    int getWarnings( z_port port /**< [in] the port with which to communicate */ );

    /// Clear all state so that when the system is powered back on we get the correct new state.
    int onPowerOff();

    int writeStateFile( std::ofstream &fout /**< [in] an open ofstream to write to */ );

    int readStateFile( std::ifstream &fin /**< [in] an open ofstream to write to */ );
};

template <class parentT>
std::string zaberStage<parentT>::name()
{
    return m_name;
}

template <class parentT>
int zaberStage<parentT>::name( const std::string &n )
{
    m_name = n;
    return 0;
}

template <class parentT>
std::string zaberStage<parentT>::serial()
{
    return m_serial;
}

template <class parentT>
int zaberStage<parentT>::serial( const std::string &s )
{
    m_serial = s;
    return 0;
}

template <class parentT>
int zaberStage<parentT>::deviceAddress()
{
    return m_deviceAddress;
}

template <class parentT>
int zaberStage<parentT>::deviceAddress( const int &da )
{
    m_deviceAddress = da;
    return 0;
}

template <class parentT>
int zaberStage<parentT>::axisNumber()
{
    return m_axisNumber;
}

template <class parentT>
int zaberStage<parentT>::axisNumber( const int &an )
{
    m_axisNumber = an;
    return 0;
}

template <class parentT>
bool zaberStage<parentT>::commandStatus()
{
    return m_commandStatus;
}

template <class parentT>
char zaberStage<parentT>::deviceStatus()
{
    return m_deviceStatus;
}

template <class parentT>
bool zaberStage<parentT>::homing()
{
    return m_homing;
}

template <class parentT>
time_t zaberStage<parentT>::lastHomed()
{
    return m_lastHomed.tv_sec;
}

template <class parentT>
int zaberStage<parentT>::parked()
{
    return m_parked;
}

template <class parentT>
bool zaberStage<parentT>::knobEnabled()
{
    return m_knobEnabled;
}

template <class parentT>
bool zaberStage<parentT>::ledEnabled()
{
    return m_ledEnabled;
}

template <class parentT>
long zaberStage<parentT>::rawPos()
{
    return m_rawPos;
}

template <class parentT>
long zaberStage<parentT>::tgtPos()
{
    return m_tgtPos;
}

template <class parentT>
long zaberStage<parentT>::maxPos()
{
    return m_maxPos;
}

template <class parentT>
bool zaberStage<parentT>::warn()
{
    return m_warn;
}

template <class parentT>
float zaberStage<parentT>::temp()
{
    return m_temp;
}

template <class parentT>
bool zaberStage<parentT>::warningState()
{
    return m_warn;
}

template <class parentT>
bool zaberStage<parentT>::warnFD()
{
    return m_warnFD;
}

template <class parentT>
bool zaberStage<parentT>::warnFQ()
{
    return m_warnFQ;
}

template <class parentT>
bool zaberStage<parentT>::warnFS()
{
    return m_warnFS;
}

template <class parentT>
bool zaberStage<parentT>::warnFT()
{
    return m_warnFT;
}

template <class parentT>
bool zaberStage<parentT>::warnFB()
{
    return m_warnFB;
}

template <class parentT>
bool zaberStage<parentT>::warnFP()
{
    return m_warnFP;
}

template <class parentT>
bool zaberStage<parentT>::warnFE()
{
    return m_warnFE;
}

template <class parentT>
bool zaberStage<parentT>::warnWH()
{
    return m_warnWH;
}

template <class parentT>
bool zaberStage<parentT>::warnWL()
{
    return m_warnWL;
}

template <class parentT>
bool zaberStage<parentT>::warnWP()
{
    return m_warnWP;
}

template <class parentT>
bool zaberStage<parentT>::warnWV()
{
    return m_warnWV;
}

template <class parentT>
bool zaberStage<parentT>::warnWT()
{
    return m_warnWT;
}

template <class parentT>
bool zaberStage<parentT>::warnWM()
{
    return m_warnWM;
}

template <class parentT>
bool zaberStage<parentT>::warnWR()
{
    return m_warnWR;
}

template <class parentT>
bool zaberStage<parentT>::warnNC()
{
    return m_warnNC;
}

template <class parentT>
bool zaberStage<parentT>::warnNI()
{
    return m_warnNI;
}

template <class parentT>
bool zaberStage<parentT>::warnND()
{
    return m_warnND;
}

template <class parentT>
bool zaberStage<parentT>::warnNU()
{
    return m_warnNU;
}

template <class parentT>
bool zaberStage<parentT>::warnNJ()
{
    return m_warnNJ;
}

template <class parentT>
bool zaberStage<parentT>::warnUNK()
{
    return m_warnUNK;
}

template <class parentT>
int zaberStage<parentT>::getResponse( std::string &response, const std::string &repBuff )
{
    za_reply rep;
    int      rv = za_decode( &rep, repBuff.c_str(), repBuff.size() );
    if( rv != Z_SUCCESS )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return rv; // don't log, but propagate error
        }

        MagAOXAppT::log<software_error>( { rv, "za_decode !=Z_SUCCESS" } );
        return rv;
    }

    return getResponse( response, rep );
}

template <class parentT>
int zaberStage<parentT>::getResponse( std::string &response, const za_reply &rep )
{
    if( rep.device_address == m_deviceAddress )
    {
        if( rep.reply_flags[0] == 'O' )
        {
            m_commandStatus = true;
        }
        else
        {
            m_commandStatus = false;
        }

        m_deviceStatus = rep.device_status[0];

        if( m_deviceStatus == 'I' && m_homing )
        {
            m_warnWR = false; // Clear preemptively
            m_homing = false;
            if( clock_gettime( CLOCK_ISIO, &m_lastHomed ) < 0 )
            {
                MagAOXAppT::log<software_error>( { errno, 0, "clock_gettime for last homed" } );
            }
        }

        if( rep.warning_flags[0] == '-' )
        {
            unsetWarnings();
        }
        else
        {
            m_warn = true;
        }

        response = rep.response_data;

        return 0;
    }
    else
    {
        MagAOXAppT::log<software_error>( "wrong device" );
        return -1;
    }
}

template <class parentT>
bool zaberStage<parentT>::isCommandReply( const za_reply &rep )
{
    return rep.message_type == '@' && rep.device_address == m_deviceAddress;
}

template <class parentT>
int zaberStage<parentT>::sendCommand( std::string &response, z_port port, const std::string &command )
{
    za_send( port, command.c_str(), command.size() );

    char buff[256];

    while( 1 )
    {
        int rv = za_receive( port, buff, sizeof( buff ) );

        if( rv == Z_ERROR_TIMEOUT )
        {
            if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
            {
                return rv; // don't log, but propagate error
            }

            MagAOXAppT::log<software_error>( "Z_ERROR_TIMEOUT" );
            break; // assume error and just get out.
        }
        else if( rv < 0 )
        {
            if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
            {
                return rv; // don't log, but propagate error
            }

            MagAOXAppT::log<software_error>( { 0, "za_receive !=Z_SUCCESS" } );
            break;
        }
        za_reply rep;

        rv = za_decode( &rep, buff, sizeof( buff ) );
        if( rv != Z_SUCCESS )
        {
            if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
            {
                return rv; // don't log, but propagate error
            }

            MagAOXAppT::log<software_error>( "za_decode !=Z_SUCCESS" );
            break;
        }

        if( isCommandReply( rep ) )
            return getResponse( response, rep );
    }

    response = "";

    return -1;
}

template <class parentT>
template <typename valT>
int zaberStage<parentT>::getValue( valT &val, z_port port, const std::string command )
{
    if( m_deviceAddress < 1 )
    {
        return MagAOXAppT::log<software_error, -1>(
            { std::format( "stage {} with s/n {} not found in system.", m_name, m_serial ) } );
    }

    std::string response;

    int rv = sendCommand( response, port, std::format( "/{} {}", m_deviceAddress, command ) );

    if( rv == 0 )
    {
        if( m_commandStatus )
        {
            mx::error_t errc;

            val = mx::ioutils::stoT<valT>( response, &errc );

            if( !!errc )
            {
                return MagAOXAppT::log<software_error, -1>(
                    std::format( "parsing [{}] response from {}: {}", command, m_name, mx::errorMessage( errc ) ) );
            }

            return 0;
        }
        else
        {
            if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
            {
                return -1; // don't log, but propagate error
            }

            MagAOXAppT::log<software_error>( { rv, command + "command Rejected" } );
            return -1;
        }
    }
    else
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        MagAOXAppT::log<software_error>();

        return -1;
    }
}

template <class parentT>
int zaberStage<parentT>::getMaxPos( z_port port )
{
    int rv = getValue( m_maxPos, port, "get limit.max" );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::getParked( z_port port )
{
    int rv = getValue( m_parked, port, "tools parking state" );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::getKnob( z_port port )
{
    int rv = getValue( m_knobEnabled, port, "get knob.enable" );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::getLED( z_port port )
{
    int rv = getValue( m_ledEnabled, port, "get system.led.enable" );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::updatePos( z_port port )
{
    int rv = getValue( m_rawPos, port, "get pos" );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::updateTemp( z_port port )
{
    int rv = getValue( m_temp, port, "get driver.temperature" );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::sendCommand( z_port port, const std::string &command )
{
    if( m_deviceAddress < 1 )
    {
        return MagAOXAppT::log<software_error, -1>(
            std::format( "stage {} with s/n {} not found in system.", m_name, m_serial ) );
    }

    std::string response;

    int rv = sendCommand( response, port, std::format( "/{} {}", m_deviceAddress, command ) );

    if( rv == 0 )
    {
        if( m_commandStatus )
        {
            return 0;
        }
        else
        {
            if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
            {
                return -1; // don't log, but propagate error
            }

            return MagAOXAppT::log<software_error, -1>(
                { rv, std::format( "{} {} commmand rejected", m_name, command ) } );
        }
    }
    else
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        MagAOXAppT::log<software_error>();
        return -1;
    }
}

template <class parentT>
int zaberStage<parentT>::enableKnob( z_port port, bool enable )
{
    std::string cmd = std::format( "set knob.enable {}", enable ? "1" : "0" );

    int rv = sendCommand( port, cmd );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::enableLED( z_port port, bool enable )
{
    std::string cmd = std::format( "set system.led.enable {}", enable ? "1" : "0" );

    int rv = sendCommand( port, cmd );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::stop( z_port port )
{
    int rv = sendCommand( port, "stop" );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::estop( z_port port )
{
    int rv = sendCommand( port, "estop" );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::home( z_port port )
{
    int rv = sendCommand( port, "home" );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    m_homing = true;
    m_tgtPos = 0;
    return 0;
}

template <class parentT>
int zaberStage<parentT>::park( z_port port )
{
    int rv = sendCommand( port, "tools parking park" );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    m_parked = true;

    return 0;
}

template <class parentT>
int zaberStage<parentT>::unpark( z_port port )
{
    int rv = sendCommand( port, "tools parking unpark" );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    m_parked = false;
    return 0;
}

template <class parentT>
int zaberStage<parentT>::moveAbs( z_port port, long rawPos )
{
    m_tgtPos = rawPos;

    if( m_parked )
    {
        if( unpark( port ) < 0 )
        {
            if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
            {
                return -1; // don't log, but propagate error
            }

            return MagAOXAppT::log<software_error, -1>();
        }
    }

    int rv = sendCommand( port, std::format( "move abs {}", rawPos ) );

    if( rv < 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        return MagAOXAppT::log<software_error, -1>();
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::unsetWarnings()
{
    m_warn = false;

    m_warnFD  = false;
    m_warnFQ  = false;
    m_warnFS  = false;
    m_warnFT  = false;
    m_warnFB  = false;
    m_warnFP  = false;
    m_warnFE  = false;
    m_warnWH  = false;
    m_warnWL  = false;
    m_warnWP  = false;
    m_warnWV  = false;
    m_warnWT  = false;
    m_warnWM  = false;
    m_warnWR  = false;
    m_warnNC  = false;
    m_warnNI  = false;
    m_warnND  = false;
    m_warnNU  = false;
    m_warnNJ  = false;
    m_warnUNK = false;

    return 0;
}

template <class parentT>
int zaberStage<parentT>::processWarning( std::string &warn )
{
    if( warn == "FD" )
    {
        if( !m_warnFDreported )
        {
            MagAOXAppT::log<text_log>( m_name +
                                           " Driver Disabled (FD): The driver has disabled itself due to overheating.",
                                       logPrio::LOG_EMERGENCY );
            m_warnFDreported = true;
        }

        m_warnFD = true;
        return 0;
    }
    else if( warn == "FQ" )
    {
        if( !m_warnFQreported )
        {
            MagAOXAppT::log<text_log>( m_name + " warning FQ: you should probably check.", logPrio::LOG_EMERGENCY );
            m_warnFQreported = true;
        }

        m_warnFQ = true;
        return 0;
    }
    else if( warn == "FS" )
    {
        if( !m_warnFSreported )
        {
            MagAOXAppT::log<text_log>(
                m_name + " Stalled and Stopped (FS): Stalling was detected and the axis has stopped itself. ",
                logPrio::LOG_WARNING );
            m_warnFSreported = true;
        }
        m_warnFS = true;
        return 0;
    }
    else if( warn == "FT" )
    {
        if( !m_warnFTreported )
        {
            MagAOXAppT::log<text_log>(
                m_name + " Excessive Twist (FT): The lockstep group has exceeded allowable twist and has stopped. ",
                logPrio::LOG_WARNING );
            m_warnFTreported = true;
        }

        m_warnFT = true;
        return 0;
    }
    else if( warn == "FB" )
    {
        if( !m_warnFBreported )
        {
            MagAOXAppT::log<text_log>( m_name + " Stream Bounds Error (FB): A previous streamed motion could not be "
                                                "executed because it failed a precondition",
                                       logPrio::LOG_WARNING );
            m_warnFBreported = true;
        }

        m_warnFB = true;
        return 0;
    }
    else if( warn == "FP" )
    {
        if( !m_warnFPreported )
        {
            MagAOXAppT::log<text_log>(
                m_name + " Interpolated Path Deviation (FP): Streamed or sinusoidal motion was terminated because an "
                         "axis slipped and thus the device deviated from the requested path. ",
                logPrio::LOG_WARNING );
            m_warnFPreported = true;
        }

        m_warnFP = true;
        return 0;
    }
    else if( warn == "FE" )
    {
        if( !m_warnFEreported )
        {
            MagAOXAppT::log<text_log>(
                m_name + " Limit Error (FE): The target limit sensor cannot be reached or is faulty. ",
                logPrio::LOG_WARNING );
            m_warnFEreported = true;
        }

        m_warnFE = true;
        return 0;
    }
    else if( warn == "WH" )
    {
        if( m_warnWHreported == false )
        {
            MagAOXAppT::log<text_log>(
                m_name + " Device not homed (WH): The device has a position reference, but has not been homed.",
                logPrio::LOG_WARNING );
            m_warnWHreported = true;
        }

        m_warnWH = true;
        return 0;
    }
    else if( warn == "WL" )
    {
        if( !m_warnWLreported )
        {
            MagAOXAppT::log<text_log>( m_name + " Unexpected Limit Trigger warning (WL): A movement operation did not "
                                                "complete due to a triggered limit sensor.",
                                       logPrio::LOG_WARNING );
            m_warnWLreported = true;
        }

        m_warnWL = true;
        return 0;
    }
    else if( warn == "WP" )
    {
        if( !m_warnWPreported )
        {
            MagAOXAppT::log<text_log>(
                m_name + " Invalid calibration type (WP): The saved calibration data type is unsupported",
                logPrio::LOG_WARNING );
            m_warnWPreported = true;
        }

        m_warnWP = true;
        return 0;
    }
    else if( warn == "WV" )
    {
        if( !m_warnWVreported )
        {
            MagAOXAppT::log<text_log>( m_name + " Voltage Out of Range (WV): The supply voltage is outside the "
                                                "recommended operating range of the device",
                                       logPrio::LOG_WARNING );
            m_warnWVreported = true;
        }

        m_warnWV = true;
        return 0;
    }
    else if( warn == "WT" )
    {
        if( !m_warnWTreported )
        {
            MagAOXAppT::log<text_log>( m_name + " Controller Temperature High (WT): The internal temperature of the "
                                                "controller has exceeded the recommended limit for the device.",
                                       logPrio::LOG_WARNING );
            m_warnWTreported = true;
        }

        m_warnWT = true;
        return 0;
    }
    else if( warn == "WM" )
    {
        if( m_warnWMreported == false )
        {
            MagAOXAppT::log<text_log>( m_name + " Displaced when Stationary (WM): While not in motion, the axis has "
                                                "been forced out of its position.",
                                       logPrio::LOG_WARNING );
            m_warnWMreported = true;
        }

        m_warnWM = true;
        return 0;
    }
    else if( warn == "WR" )
    {
        if( m_warnWRreported == false )
        {
            MagAOXAppT::log<text_log>(
                m_name +
                    " No Reference Position (WR): Axis has not had a reference position established. [homing required]",
                logPrio::LOG_WARNING );
            m_warnWRreported = true;
        }

        m_warnWR = true;
        return 0;
    }
    else if( warn == "NC" )
    {
        if( !m_warnNCreported )
        {
            MagAOXAppT::log<text_log>( m_name +
                                           " Manual Control (NC): Axis is busy due to manual control via the knob.",
                                       logPrio::LOG_WARNING );
            m_warnNCreported = true;
        }

        m_warnNC = true;
        return 0;
    }
    else if( warn == "NI" )
    {
        if( m_homing == true || warnWR() )
        {
            return 0; // ignore this during homing
        }

        if( !m_warnNIreported )
        {
            MagAOXAppT::log<text_log>( m_name + " Command Interrupted (NI): A movement operation "
                                                "(command or manual control) was requested "
                                                "while the axis was executing another movement command.",
                                       logPrio::LOG_WARNING );
            m_warnNIreported = true;
        }

        m_warnNI = true;
        return 0;
    }
    else if( warn == "ND" )
    {
        if( !m_warnNDreported )
        {
            MagAOXAppT::log<text_log>( m_name + " Stream Discontinuity (ND): The device has slowed down while "
                                                "following a "
                                                "streamed motion path because it has run out of queued motions.",
                                       logPrio::LOG_WARNING );
            m_warnNDreported = true;
        }

        m_warnND = true;
        return 0;
    }
    else if( warn == "NU" )
    {
        if( !m_warnNUreported )
        {
            MagAOXAppT::log<text_log>( m_name + " Setting Update Pending (NU): A setting is pending to be "
                                                "updated or a reset is pending.",
                                       logPrio::LOG_WARNING );
            m_warnNUreported = true;
        }

        m_warnNU = true;
        return 0;
    }
    else if( warn == "NJ" )
    {
        if( !m_warnNJreported )
        {
            MagAOXAppT::log<text_log>( m_name + " Joystick Calibrating (NJ): Joystick calibration is "
                                                "in progress.",
                                       logPrio::LOG_WARNING );
            m_warnNJreported = true;
        }

        m_warnNJ = true;
        return 0;
    }
    else
    {
        MagAOXAppT::log<software_warning>( std::format( "{} unknown stage warning: {}", m_name, warn ) );

        m_warnUNK = true;

        return 0;
    }

    return -1;
}

template <class parentT>
int zaberStage<parentT>::parseWarnings( std::string &response )
{

    size_t nwarn;

    try
    {
        nwarn = std::stoi( response.substr( 0, 2 ) );
    }
    catch( ... )
    {
        return MagAOXAppT::log<software_error, -1>( { "exception while parsing warning" } );
    }

    if( nwarn > 0 )
    {
        m_warn = true;
    }

    for( size_t n = 0; n < nwarn; ++n )
    {
        if( response.size() < 5 + n * 3 )
        {
            if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
            {
                return -1; // don't log, but propagate error
            }

            return MagAOXAppT::log<software_error, -1>( { "parsing incomplete warning response" } );
        }

        std::string warn = response.substr( 3 + n * 3, 2 );

        int rv = processWarning( warn );
        if( rv < 0 )
        {
            if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
            {
                return -1; // don't log, but propagate error
            }

            MagAOXAppT::log<software_error>();
            return -1;
        }
    }

    if( m_warnFDreported )
    {
        if( !m_warnFD )
        {
            m_warnFDreported = false;
        }
    }

    if( m_warnFQreported )
    {
        if( !m_warnFQ )
        {
            m_warnFQreported = false;
        }
    }

    if( m_warnFSreported )
    {
        if( !m_warnFS )
        {
            m_warnFSreported = false;
        }
    }

    if( m_warnFTreported )
    {
        if( !m_warnFT )
        {
            m_warnFTreported = false;
        }
    }

    if( m_warnFBreported )
    {
        if( !m_warnFB )
        {
            m_warnFBreported = false;
        }
    }

    if( m_warnFPreported )
    {
        if( !m_warnFP )
        {
            m_warnFPreported = false;
        }
    }

    if( m_warnFEreported )
    {
        if( !m_warnFE )
        {
            m_warnFEreported = false;
        }
    }

    if( m_warnWHreported )
    {
        if( !m_warnWH )
        {
            m_warnWHreported = false;
        }
    }

    if( m_warnWLreported )
    {
        if( !m_warnWL )
        {
            m_warnWLreported = false;
        }
    }

    if( m_warnWPreported )
    {
        if( !m_warnWP )
        {
            m_warnWPreported = false;
        }
    }

    if( m_warnWVreported )
    {
        if( !m_warnWV )
        {
            m_warnWVreported = false;
        }
    }

    if( m_warnWTreported )
    {
        if( !m_warnWT )
        {
            m_warnWTreported = false;
        }
    }

    if( m_warnWMreported )
    {
        if( !m_warnWM )
        {
            m_warnWMreported = false;
        }
    }

    if( m_warnWRreported )
    {
        if( !m_warnWR )
        {
            m_warnWRreported = false;
        }
    }

    if( m_warnNCreported )
    {
        if( !m_warnNC )
        {
            m_warnNCreported = false;
        }
    }

    if( m_warnNIreported )
    {
        if( !m_warnNI )
        {
            m_warnNIreported = false;
        }
    }

    if( m_warnNDreported )
    {
        if( !m_warnND )
        {
            m_warnNDreported = false;
        }
    }

    if( m_warnNUreported )
    {
        if( !m_warnNU )
        {
            m_warnNUreported = false;
        }
    }

    if( m_warnNJreported )
    {
        if( !m_warnNJ )
        {
            m_warnNJreported = false;
        }
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::getWarnings( z_port port )
{
    if( m_deviceAddress < 1 )
    {
        return MagAOXAppT::log<software_error, -1>(
            { "stage " + m_name + " with with s/n " + m_serial + " not found in system." } );
    }

    std::string response;

    int rv = sendCommand( response, port, std::format( "/{} warnings", m_deviceAddress ) );

    if( rv == 0 )
    {
        if( m_commandStatus )
        {
            unsetWarnings(); // Clear all the flags before setting them to stay current.
            return parseWarnings( response );
        }
        else
        {
            if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
            {
                return -1; // don't log, but propagate error
            }

            MagAOXAppT::log<software_error>( { rv, "warnings Command Rejected" } );
            return -1;
        }
    }
    else
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1; // don't log, but propagate error
        }

        MagAOXAppT::log<software_error>();
        return -1;
    }
}

template <class parentT>
int zaberStage<parentT>::onPowerOff()
{
    m_commandStatus = true; ///< The status of the last command sent. true = OK, false = RJ (rejected)

    m_deviceStatus = 'U'; ///< Current status.  Either 'I' for IDLE or 'B' for BUSY.  Intializes to 'U' for UNKOWN.

    m_homing = false;

    // We don't 0 rawPos so it is retained

    m_temp = -999; ///< The driver temperature, in C.

    unsetWarnings();
    m_warnWRreported = false;

    return 0;
}

template <class parentT>
int zaberStage<parentT>::writeStateFile( std::ofstream &fout )
{
    fout << m_rawPos << '\n';

    if( !fout )
    {
        return MagAOXAppT::log<software_error, -1>( { "error writing raw position" } );
    }

    fout << m_parked << '\n';

    if( !fout )
    {
        return MagAOXAppT::log<software_error, -1>( { "error writing parked state" } );
    }

    fout << m_maxPos << '\n';

    if( !fout )
    {
        return MagAOXAppT::log<software_error, -1>( { "error writing max position" } );
    }

    fout << m_lastHomed.tv_sec << '\n';

    if( !fout )
    {
        return MagAOXAppT::log<software_error, -1>( { "error writing last home time" } );
    }

    return 0;
}

template <class parentT>
int zaberStage<parentT>::readStateFile( std::ifstream &fin )
{
    long   rawPos;
    bool   parked;
    long   maxPos;
    time_t lastHomed;

    fin >> rawPos;

    if( !fin )
    {
        return MagAOXAppT::log<software_error, -1>( { "error reading raw position" } );
    }

    fin >> parked;

    if( !fin )
    {
        return MagAOXAppT::log<software_error, -1>( { "error reading parked state" } );
    }

    fin >> maxPos;

    if( !fin )
    {
        return MagAOXAppT::log<software_error, -1>( { "error reading max position" } );
    }

    fin >> lastHomed;

    if( !fin )
    {
        return MagAOXAppT::log<software_error, -1>( { "error reading last home time" } );
    }

    m_rawPos            = rawPos;
    m_tgtPos            = rawPos;
    m_parked            = parked;
    m_maxPos            = maxPos;
    m_lastHomed.tv_sec  = lastHomed;
    m_lastHomed.tv_nsec = 0;

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // zaberStage_hpp
