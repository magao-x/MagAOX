/** \file zaberBinaryStage.hpp
 * \brief A class with details of a single binary-protocol Zaber stage.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup zaberLowLevelBinary_files
 */

#ifndef zaberBinaryStage_hpp
#define zaberBinaryStage_hpp

#include <ctime>
#include <fstream>
#include <limits>
#include <string>

#include "../../libMagAOX/libMagAOX.hpp" // Note this is included on command line to trigger pch

#include "zb_serial.h"

namespace MagAOX
{
namespace app
{

/// A class to manage the details of one binary-protocol stage in a Zaber system.
/**
 * \ingroup zaberLowLevelBinary
 */
template <class parentT>
class zaberBinaryStage
{
  public:
    /// Binary protocol command numbers used by the T-LSM firmware 5.xx implementation.
    enum commandCodes : uint8_t
    {
        cmdHome                  = 1,
        cmdRenumber              = 2,
        cmdStoreCurrentPosition  = 16,
        cmdReturnStoredPosition  = 17,
        cmdMoveToStoredPosition  = 18,
        cmdMoveAbsolute          = 20,
        cmdMoveRelative          = 21,
        cmdStop                  = 23,
        cmdSetHoldCurrent        = 39,
        cmdSetDeviceMode         = 40,
        cmdSetTargetSpeed        = 42,
        cmdSetMaximumPosition    = 44,
        cmdSetCurrentPosition    = 45,
        cmdReturnFirmwareVersion = 51,
        cmdReturnSetting         = 53,
        cmdReturnStatus          = 54,
        cmdReturnCurrentPosition = 60,
        cmdReturnSerialNumber    = 63
    };

    /// Device-mode bits used by the implementation.
    enum modeBits : int32_t
    {
        modeDisableAutoReply     = ( 1 << 0 ),
        modeDisablePotentiometer = ( 1 << 3 ),
        modeHomeStatus           = ( 1 << 7 )
    };

    /// Stored-position register used to persist the MagAO-X parked position.
    static constexpr int32_t parkPositionRegister = 0;

  protected:
    /// Parent application used for logging and power-state checks.
    parentT *m_parent{ nullptr };

    /// Configured stage name used in INDI properties.
    std::string m_name;

    /// Configured stage serial number.
    std::string m_serial;

    /// Current binary device address.
    int m_deviceAddress{ -1 };

    /// Axis number placeholder for future multi-axis support.
    int m_axisNumber{ 0 };

    /// Whether the most recent command was accepted.
    bool m_commandStatus{ true };

    /// Current device state mapped to ASCII-app semantics.
    char m_deviceStatus{ 'U' };

    /// Whether the stage is currently homing.
    bool m_homing{ false };

    /// Time stamp of the last successful home operation.
    timespec m_lastHomed{ 0, 0 };

    /// Whether MagAO-X currently considers this stage parked.
    bool m_parked{ false };

    /// Current raw position in microsteps.
    long m_rawPos{ 0 };

    /// Last target position sent to the device in microsteps.
    long m_tgtPos{ 0 };

    /// Configured target speed command value for this stage.
    int32_t m_targetSpeed{ 1000 };

    /// Maximum position in microsteps.
    long m_maxPos{ -1 };

    /// Emulated parked position in microsteps.
    long m_parkPos{ 0 };

    /// Whether a parked position has been recovered from device non-volatile memory.
    bool m_hasParkPos{ false };

    /// Whether a parked state file has been loaded from disk.
    bool m_hasStateFile{ false };

    /// Raw position loaded from the parked state file.
    long m_stateFileRawPos{ 0 };

    /// Parked flag loaded from the parked state file.
    bool m_stateFileParked{ false };

    /// Last reported stage temperature. Firmware 5.35 does not expose this directly.
    float m_temp{ -999.0 };

    /// whether potentiometer knob is enabled
    bool m_knobEnabled{ false };

    /// Whether any warning-equivalent condition is active.
    bool m_warn{ false };

    /// Driver disabled warning flag.
    bool m_warnFD{ false };

    /// Device-specific warning flag placeholders retained for API compatibility.
    bool m_warnFQ{ false };
    bool m_warnFS{ false };
    bool m_warnFT{ false };
    bool m_warnFB{ false };
    bool m_warnFP{ false };
    bool m_warnFE{ false };
    bool m_warnWH{ false };
    bool m_warnWL{ false };
    bool m_warnWP{ false };
    bool m_warnWV{ false };
    bool m_warnWT{ false };
    bool m_warnWM{ false };
    bool m_warnWR{ false };
    bool m_warnNC{ false };
    bool m_warnNI{ false };
    bool m_warnND{ false };
    bool m_warnNU{ false };
    bool m_warnNJ{ false };
    bool m_warnUNK{ false };

  public:
    /// Default constructor deleted because stages require a parent app.
    zaberBinaryStage() = delete;

    /// Construct the stage helper.
    zaberBinaryStage( parentT *parent /**< [in] the parent application */ )
    {
        if( parent == nullptr )
        {
            throw mx::exception( mx::error_t::invalidarg, "parent was null on construction of zaberBinaryStage" );
        }

        m_parent = parent;
    }

    /// Get the stage name.
    std::string name();

    /// Set the stage name.
    int name( const std::string &n /**< [in] the new stage name */ );

    /// Get the stage serial number.
    std::string serial();

    /// Set the stage serial number.
    int serial( const std::string &s /**< [in] the new stage serial number */ );

    /// Get the device address.
    int deviceAddress();

    /// Set the device address.
    int deviceAddress( const int &da /**< [in] the new device address */ );

    /// Get the axis number.
    int axisNumber();

    /// Set the axis number.
    int axisNumber( const int &an /**< [in] the new axis number */ );

    /// Get the status of the last command.
    bool commandStatus();

    /// Get the current device status.
    char deviceStatus();

    /// Get the homing state.
    bool homing();

    /// Get the last home time.
    time_t lastHomed();

    /// Get the parked state.
    int parked();

    /// Get the knob state.
    bool knobEnabled();

    /// Get the current raw position.
    long rawPos();

    /// Get the target raw position.
    long tgtPos();

    /// Get the maximum position.
    long maxPos();

    /// Get the configured target speed.
    int32_t targetSpeed();

    /// Set the configured target speed.
    int targetSpeed( const int32_t &speed /**< [in] the target speed command value */ );

    /// Get whether any warning-equivalent flag is set.
    bool warn();

    /// Get the driver temperature.
    float temp();

    /// Get whether any warning-equivalent flag is set.
    bool warningState();

    /// Get the driver-disabled warning flag.
    bool warnFD();
    /// Get the FQ warning flag.
    bool warnFQ();
    /// Get the FS warning flag.
    bool warnFS();
    /// Get the FT warning flag.
    bool warnFT();
    /// Get the FB warning flag.
    bool warnFB();
    /// Get the FP warning flag.
    bool warnFP();
    /// Get the FE warning flag.
    bool warnFE();
    /// Get the WH warning flag.
    bool warnWH();
    /// Get the WL warning flag.
    bool warnWL();
    /// Get the WP warning flag.
    bool warnWP();
    /// Get the WV warning flag.
    bool warnWV();
    /// Get the WT warning flag.
    bool warnWT();
    /// Get the WM warning flag.
    bool warnWM();
    /// Get the WR warning flag.
    bool warnWR();
    /// Get the NC warning flag.
    bool warnNC();
    /// Get the NI warning flag.
    bool warnNI();
    /// Get the ND warning flag.
    bool warnND();
    /// Get the NU warning flag.
    bool warnNU();
    /// Get the NJ warning flag.
    bool warnNJ();
    /// Get the unknown-warning flag.
    bool warnUNK();

    /// Send a command and wait for the corresponding binary reply.
    int queryCommand( int32_t &response,      /**< [out] decoded reply data */
                      z_port   port,          /**< [in] the port with which to communicate */
                      uint8_t  commandNumber, /**< [in] the command number to send */
                      int32_t  data,          /**< [in] the command data */
                      uint8_t  expectedReply  /**< [in] the expected reply command number */
    );

    /// Send a command for which no reply is expected.
    int sendCommandNoReply( z_port  port,          /**< [in] the port with which to communicate */
                            uint8_t commandNumber, /**< [in] the command number to send */
                            int32_t data           /**< [in] the command data */
    );

    /// Return a setting value from the device.
    int getSetting( int32_t &value,        /**< [out] the setting value */
                    z_port   port,         /**< [in] the port with which to communicate */
                    uint8_t  settingNumber /**< [in] the setting command number to query */
    );

    /// Get the maximum position from the device.
    int getMaxPos( z_port port /**< [in] the port with which to communicate */ );

    /// Get the parked state for MagAO-X compatibility.
    int getParked( z_port port /**< [in] the port with which to communicate */ );

    /// Get the knob enabled status
    int getKnob( z_port port /**< [in] the port with which to communicate */ );

    /// Update the current position and derived motion state.
    int updatePos( z_port port /**< [in] the port with which to communicate */ );

    /// Update the stage temperature, if supported.
    int updateTemp( z_port port /**< [in] the port with which to communicate */ );

    /// Disable the manual knob and asynchronous command replies.
    int enableKnob( z_port port, bool enable /**< [in] the port with which to communicate */ );

    /// Set the target speed used for absolute and relative moves.
    int setTargetSpeed( z_port  port, /**< [in] the port with which to communicate */
                        int32_t speed /**< [in] the target speed command value */
    );

    /// Set the hold current used while the stage is idle.
    int setHoldCurrent( z_port  port, /**< [in] the port with which to communicate */
                        int32_t value /**< [in] the hold current command value */
    );

    /// Stop the stage.
    int stop( z_port port /**< [in] the port with which to communicate */ );

    /// Emergency-stop the stage.
    int estop( z_port port /**< [in] the port with which to communicate */ );

    /// Home the stage.
    int home( z_port port /**< [in] the port with which to communicate */ );

    /// Mark the stage parked using MagAO-X bookkeeping semantics.
    int park( z_port port /**< [in] the port with which to communicate */ );

    /// Clear the parked bookkeeping state.
    int unpark( z_port port /**< [in] the port with which to communicate */ );

    /// Recall the MagAO-X parked position from device non-volatile memory.
    int recallParkPosition( z_port port /**< [in] the port with which to communicate */ );

    /// Restore parked state after a power cycle if device and disk state agree.
    int restoreParkedState( z_port port /**< [in] the port with which to communicate */ );

    /// Move to a new absolute position.
    int moveAbs( z_port port, /**< [in] the port with which to communicate */
                 long   rawPos /**< [in] the position to move to in microsteps */ );

    /// Clear all warning flags.
    int unsetWarnings();

    /// Refresh warning-equivalent state from firmware 5.xx information.
    int getWarnings( z_port port /**< [in] the port with which to communicate */ );

    /// Refresh the stored last-home timestamp when a home completion is detected.
    int updateLastHomed( bool wasHoming /**< [in] whether the stage was homing before the latest status refresh */ );

    /// Clear transient state on power-off.
    int onPowerOff();

    /// Write the state file used by the low-level app.
    int writeStateFile( std::ofstream &fout /**< [in] an open output stream */ );

    /// Read the state file used by the low-level app.
    int readStateFile( std::ifstream &fin /**< [in] an open input stream */ );
};

template <class parentT>
std::string zaberBinaryStage<parentT>::name()
{
    return m_name;
}

template <class parentT>
int zaberBinaryStage<parentT>::name( const std::string &n )
{
    m_name = n;
    return 0;
}

template <class parentT>
std::string zaberBinaryStage<parentT>::serial()
{
    return m_serial;
}

template <class parentT>
int zaberBinaryStage<parentT>::serial( const std::string &s )
{
    m_serial = s;
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::deviceAddress()
{
    return m_deviceAddress;
}

template <class parentT>
int zaberBinaryStage<parentT>::deviceAddress( const int &da )
{
    m_deviceAddress = da;
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::axisNumber()
{
    return m_axisNumber;
}

template <class parentT>
int zaberBinaryStage<parentT>::axisNumber( const int &an )
{
    m_axisNumber = an;
    return 0;
}

template <class parentT>
bool zaberBinaryStage<parentT>::commandStatus()
{
    return m_commandStatus;
}

template <class parentT>
char zaberBinaryStage<parentT>::deviceStatus()
{
    return m_deviceStatus;
}

template <class parentT>
bool zaberBinaryStage<parentT>::homing()
{
    return m_homing;
}

template <class parentT>
time_t zaberBinaryStage<parentT>::lastHomed()
{
    return m_lastHomed.tv_sec;
}

template <class parentT>
int zaberBinaryStage<parentT>::parked()
{
    return m_parked;
}

template <class parentT>
bool zaberBinaryStage<parentT>::knobEnabled()
{
    return m_knobEnabled;
}

template <class parentT>
long zaberBinaryStage<parentT>::rawPos()
{
    return m_rawPos;
}

template <class parentT>
long zaberBinaryStage<parentT>::tgtPos()
{
    return m_tgtPos;
}

template <class parentT>
long zaberBinaryStage<parentT>::maxPos()
{
    return m_maxPos;
}

template <class parentT>
int32_t zaberBinaryStage<parentT>::targetSpeed()
{
    return m_targetSpeed;
}

template <class parentT>
int zaberBinaryStage<parentT>::targetSpeed( const int32_t &speed )
{
    m_targetSpeed = speed;
    return 0;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warn()
{
    return m_warn;
}

template <class parentT>
float zaberBinaryStage<parentT>::temp()
{
    return m_temp;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warningState()
{
    return m_warn;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnFD()
{
    return m_warnFD;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnFQ()
{
    return m_warnFQ;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnFS()
{
    return m_warnFS;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnFT()
{
    return m_warnFT;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnFB()
{
    return m_warnFB;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnFP()
{
    return m_warnFP;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnFE()
{
    return m_warnFE;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnWH()
{
    return m_warnWH;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnWL()
{
    return m_warnWL;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnWP()
{
    return m_warnWP;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnWV()
{
    return m_warnWV;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnWT()
{
    return m_warnWT;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnWM()
{
    return m_warnWM;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnWR()
{
    return m_warnWR;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnNC()
{
    return m_warnNC;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnNI()
{
    return m_warnNI;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnND()
{
    return m_warnND;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnNU()
{
    return m_warnNU;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnNJ()
{
    return m_warnNJ;
}

template <class parentT>
bool zaberBinaryStage<parentT>::warnUNK()
{
    return m_warnUNK;
}

template <class parentT>
int zaberBinaryStage<parentT>::queryCommand(
    int32_t &response, z_port port, uint8_t commandNumber, int32_t data, uint8_t expectedReply )
{
    if( m_deviceAddress < 1 )
    {
        return MagAOXAppT::log<software_error, -1>(
            std::format( "stage {} with s/n {} not found in system.", m_name, m_serial ) );
    }

    if( port <= 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1;
        }

        return MagAOXAppT::log<software_error, -1>( { 0, "invalid zaber binary port" } );
    }

    uint8_t command[6];
    if( zb_encode( command, static_cast<uint8_t>( m_deviceAddress ), commandNumber, data ) != Z_SUCCESS )
    {
        return MagAOXAppT::log<software_error, -1>( "zb_encode failed" );
    }

    if( zb_send( port, command ) != 6 )
    {
        return MagAOXAppT::log<software_error, -1>( "zb_send failed" );
    }

    uint8_t reply[6];
    int     rv = zb_receive( port, reply );
    if( rv != 6 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1;
        }

        return MagAOXAppT::log<software_error, -1>( "zb_receive failed" );
    }

    if( reply[0] != static_cast<uint8_t>( m_deviceAddress ) )
    {
        return MagAOXAppT::log<software_error, -1>(
            std::format( "unexpected reply from device {} while querying {}", reply[0], m_name ) );
    }

    if( reply[1] == 255 )
    {
        int32_t errorCode;
        zb_decode( &errorCode, reply );
        return MagAOXAppT::log<software_error, -1>(
            std::format( "device {} command {} returned error {}", m_name, commandNumber, errorCode ) );
    }

    if( reply[1] != expectedReply )
    {
        return MagAOXAppT::log<software_error, -1>(
            std::format( "device {} returned reply {} while expecting {}", m_name, reply[1], expectedReply ) );
    }

    if( zb_decode( &response, reply ) != Z_SUCCESS )
    {
        return MagAOXAppT::log<software_error, -1>( "zb_decode failed" );
    }

    m_commandStatus = true;
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::sendCommandNoReply( z_port port, uint8_t commandNumber, int32_t data )
{
    if( m_deviceAddress < 1 )
    {
        return MagAOXAppT::log<software_error, -1>(
            std::format( "stage {} with s/n {} not found in system.", m_name, m_serial ) );
    }

    if( port <= 0 )
    {
        if( m_parent->powerState() != 1 || m_parent->powerStateTarget() != 1 )
        {
            return -1;
        }

        return MagAOXAppT::log<software_error, -1>( { 0, "invalid zaber binary port" } );
    }

    uint8_t command[6];
    if( zb_encode( command, static_cast<uint8_t>( m_deviceAddress ), commandNumber, data ) != Z_SUCCESS )
    {
        return MagAOXAppT::log<software_error, -1>( "zb_encode failed" );
    }

    if( zb_send( port, command ) != 6 )
    {
        return MagAOXAppT::log<software_error, -1>( "zb_send failed" );
    }

    m_commandStatus = true;
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::getSetting( int32_t &value, z_port port, uint8_t settingNumber )
{
    return queryCommand( value, port, cmdReturnSetting, settingNumber, settingNumber );
}

template <class parentT>
int zaberBinaryStage<parentT>::getMaxPos( z_port port )
{
    int32_t value;
    int     rv = getSetting( value, port, cmdSetMaximumPosition );
    if( rv < 0 )
    {
        return rv;
    }

    m_maxPos = value;
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::getParked( z_port )
{
    if( m_hasParkPos && m_rawPos == m_parkPos )
    {
        m_parked = true;
    }
    else
    {
        m_parked = false;
    }

    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::getKnob( z_port port )
{
    int32_t value;
    int     rv = getSetting( value, port, cmdSetDeviceMode );
    if( rv < 0 )
    {
        return rv;
    }
    m_knobEnabled = !( value & modeDisablePotentiometer );

    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::updatePos( z_port port )
{
    bool wasHoming = m_homing;

    int32_t status;
    int     rv = queryCommand( status, port, cmdReturnStatus, 0, cmdReturnStatus );
    if( rv < 0 )
    {
        return rv;
    }

    if( status == 0 )
    {
        m_deviceStatus = 'I';
    }
    else
    {
        m_deviceStatus = 'B';
    }

    m_homing = ( status == 1 );

    int32_t pos;
    rv = queryCommand( pos, port, cmdReturnCurrentPosition, 0, cmdReturnCurrentPosition );
    if( rv < 0 )
    {
        return rv;
    }

    m_rawPos = pos;

    rv = getWarnings( port );
    if( rv < 0 )
    {
        return rv;
    }

    return updateLastHomed( wasHoming );
}

template <class parentT>
int zaberBinaryStage<parentT>::updateTemp( z_port )
{
    m_temp = -999.0;
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::enableKnob( z_port port, bool enable )
{
    int32_t mode;
    int     rv = getSetting( mode, port, cmdSetDeviceMode );
    if( rv < 0 )
    {
        return rv;
    }

    mode |= modeDisableAutoReply;
    if( enable )
        mode &= ~modeDisablePotentiometer; // clear the bit to enable
    else
        mode |= modeDisablePotentiometer; // set the bit to disable

    rv = sendCommandNoReply( port, cmdSetDeviceMode, mode );
    if( rv < 0 )
    {
        return rv;
    }

    int32_t appliedMode;
    rv = getSetting( appliedMode, port, cmdSetDeviceMode );
    if( rv < 0 )
    {
        return rv;
    }

    bool knobOk = enable ? !( appliedMode & modeDisablePotentiometer ) // bit should be 0
                         : ( appliedMode & modeDisablePotentiometer ); // bit should be 1

    if( !knobOk || !( appliedMode & modeDisableAutoReply ) )
    {
        return MagAOXAppT::log<software_error, -1>(
            std::format( "device {} did not apply requested device mode {}, got {}", m_name, mode, appliedMode ) );
    }

    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::setTargetSpeed( z_port port, int32_t speed )
{
    int rv = sendCommandNoReply( port, cmdSetTargetSpeed, speed );
    if( rv < 0 )
    {
        return rv;
    }

    int32_t appliedSpeed;
    rv = getSetting( appliedSpeed, port, cmdSetTargetSpeed );
    if( rv < 0 )
    {
        return rv;
    }

    if( appliedSpeed != speed )
    {
        return MagAOXAppT::log<software_error, -1>(
            std::format( "device {} reported target speed {} after requesting {}", m_name, appliedSpeed, speed ) );
    }

    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::setHoldCurrent( z_port port, int32_t value )
{
    int rv = sendCommandNoReply( port, cmdSetHoldCurrent, value );
    if( rv < 0 )
    {
        return rv;
    }

    int32_t appliedValue;
    rv = getSetting( appliedValue, port, cmdSetHoldCurrent );
    if( rv < 0 )
    {
        return rv;
    }

    if( appliedValue != value )
    {
        return MagAOXAppT::log<software_error, -1>(
            std::format( "device {} reported hold current {} after requesting {}", m_name, appliedValue, value ) );
    }

    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::stop( z_port port )
{
    int rv = sendCommandNoReply( port, cmdStop, 0 );
    if( rv < 0 )
    {
        return rv;
    }

    m_homing = false;
    m_parked = false;
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::estop( z_port port )
{
    return stop( port );
}

template <class parentT>
int zaberBinaryStage<parentT>::home( z_port port )
{
    int rv = sendCommandNoReply( port, cmdHome, 0 );
    if( rv < 0 )
    {
        return rv;
    }

    m_homing = true;
    m_tgtPos = 0;
    m_parked = false;
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::park( z_port port )
{
    int rv = sendCommandNoReply( port, cmdStoreCurrentPosition, parkPositionRegister );
    if( rv < 0 )
    {
        return rv;
    }

    rv = recallParkPosition( port );
    if( rv < 0 )
    {
        return rv;
    }

    rv = setHoldCurrent( port, 0 );
    if( rv < 0 )
    {
        return rv;
    }

    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::unpark( z_port )
{
    m_parked = false;
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::recallParkPosition( z_port port )
{
    int32_t storedPosition;
    int     rv =
        queryCommand( storedPosition, port, cmdReturnStoredPosition, parkPositionRegister, cmdReturnStoredPosition );
    if( rv < 0 )
    {
        m_hasParkPos = false;
        return -1;
    }

    m_parkPos    = storedPosition;
    m_hasParkPos = true;
    m_parked     = ( m_rawPos == m_parkPos );
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::restoreParkedState( z_port port )
{
    if( !m_hasStateFile || !m_stateFileParked || !m_hasParkPos )
    {
        return 0;
    }

    if( m_stateFileRawPos != m_parkPos )
    {
        return MagAOXAppT::log<software_warning, 0>(
            std::format( "parked state mismatch for {}: disk {} device {}", m_name, m_stateFileRawPos, m_parkPos ) );
    }

    int rv = setHoldCurrent( port, 0 );
    if( rv < 0 )
    {
        return rv;
    }

    rv = sendCommandNoReply( port, cmdSetCurrentPosition, m_stateFileRawPos );
    if( rv < 0 )
    {
        return rv;
    }

    int32_t restoredPos;
    rv = queryCommand( restoredPos, port, cmdReturnCurrentPosition, 0, cmdReturnCurrentPosition );
    if( rv < 0 )
    {
        return rv;
    }

    if( restoredPos != m_stateFileRawPos )
    {
        return MagAOXAppT::log<software_error, -1>(
            std::format( "device {} restored position {} but expected {}", m_name, restoredPos, m_stateFileRawPos ) );
    }

    m_rawPos = restoredPos;
    m_tgtPos = restoredPos;
    m_parked = true;
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::moveAbs( z_port port, long rawPos )
{
    if( m_parked )
    {
        int rv = unpark( port );
        if( rv < 0 )
        {
            return rv;
        }
    }

    int rv = sendCommandNoReply( port, cmdMoveAbsolute, rawPos );
    if( rv < 0 )
    {
        return rv;
    }

    m_tgtPos = rawPos;
    m_homing = false;
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::unsetWarnings()
{
    m_warn    = false;
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
int zaberBinaryStage<parentT>::getWarnings( z_port port )
{
    unsetWarnings();

    int32_t mode;
    int     rv = getSetting( mode, port, cmdSetDeviceMode );
    if( rv < 0 )
    {
        return rv;
    }

    if( ( mode & modeHomeStatus ) == 0 )
    {
        m_warn   = true;
        m_warnWR = true;
    }

    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::updateLastHomed( bool wasHoming )
{
    if( !m_homing && !m_warnWR && m_tgtPos == 0 && m_rawPos == 0 && ( wasHoming || m_lastHomed.tv_sec == 0 ) )
    {
        if( clock_gettime( CLOCK_ISIO, &m_lastHomed ) < 0 )
        {
            return MagAOXAppT::log<software_error, -1>( { errno, 0, "clock_gettime for last homed" } );
        }
    }

    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::onPowerOff()
{
    m_commandStatus = true;
    m_deviceStatus  = 'U';
    m_homing        = false;
    m_temp          = -999.0;
    unsetWarnings();
    return 0;
}

template <class parentT>
int zaberBinaryStage<parentT>::writeStateFile( std::ofstream &fout )
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
int zaberBinaryStage<parentT>::readStateFile( std::ifstream &fin )
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
    m_parkPos           = rawPos;
    m_hasParkPos        = parked;
    m_hasStateFile      = true;
    m_stateFileRawPos   = rawPos;
    m_stateFileParked   = parked;
    m_maxPos            = maxPos;
    m_lastHomed.tv_sec  = lastHomed;
    m_lastHomed.tv_nsec = 0;
    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // zaberBinaryStage_hpp
