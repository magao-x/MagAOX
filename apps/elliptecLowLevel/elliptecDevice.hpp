/** \file elliptecDevice.hpp
 * \brief Per-address state for one Thorlabs Elliptec device on a shared serial bus.
 *
 * \ingroup elliptecLowLevel_files
 */

#ifndef elliptecDevice_hpp
#define elliptecDevice_hpp

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>

namespace MagAOX
{
namespace app
{

/// Protocol helpers shared by the Elliptec apps.
namespace elliptec
{

/// Human-readable meaning of an Elliptec `GS` status code.
std::string statusMeaning( uint8_t code /**< [in] status/error code from a `GS` frame */ );

/// Minimum acceptable velocity percentage for an Elliptec model, or 0 if unrestricted.
int minVelocityForModel( uint8_t model /**< [in] model number from the `in` reply, e.g. 21 */ );

/// Wrap an angle into [0, 360).
double wrapDeg( double deg /**< [in] angle in degrees */ );

/// Shortest angular distance between two angles on a circle, in [0, 180].
double circularDistanceDeg( double a, /**< [in] first angle in degrees */
                            double b  /**< [in] second angle in degrees */
);

/// Parse exactly `n` hex digits starting at `pos`.
/**
 * \returns true and sets `value` if all digits are hex, false otherwise
 */
bool parseHex( const std::string &s,    /**< [in] source text */
               size_t             pos,  /**< [in] first digit */
               size_t             n,    /**< [in] number of digits, at most 8 */
               uint32_t          &value /**< [out] parsed value */
);

} // namespace elliptec

/// State of one Elliptec device on a shared bus, identified by its address.
/** Holds no I/O. The bus owner sends commands, parses replies, and reports them here; this class decides what the
 * replies mean: identification from `in`, the pending operation and its target, completion by position, error
 * latching, missed-reply accounting, and the one motion request held while the bus is busy.
 *
 * Observed ELL21 behavior this accommodates: motion commands are not acknowledged, most queries go unanswered while
 * the stage moves, and `gs` can report idle (`GS00`) mid-motion. So an operation with a target completes only when a
 * fully answered poll reports idle and the position has held still at the target.
 *
 * \ingroup elliptecLowLevel
 */
class elliptecDevice
{
  public:
    /// Long-running operation tracked until a poll resolves it.
    enum class Pending
    {
        None,    ///< No operation in progress.
        Move,    ///< An absolute move.
        Home,    ///< A home command; the target is the home position, 0 pulses.
        Optimize ///< The motor optimization routine; no target position.
    };

    /// Outcome of resolving a poll or a missed reply, for the bus owner to log.
    enum class Event
    {
        None,      ///< Nothing to report.
        Completed, ///< The pending operation completed; see `eventText()`.
        Failed,    ///< The pending operation failed; see `eventText()`.
        Lost       ///< The device stopped answering and must be re-identified.
    };

    /// Result of applying an `in` reply.
    enum class InfoResult
    {
        Ok,            ///< Identified.
        Malformed,     ///< The reply was truncated or not hex; retry.
        SerialMismatch ///< The device serial differs from the configured one; never command this device.
    };

    /// Largest change in pulses between idle polls that still counts as stationary (encoder jitter).
    static constexpr int32_t s_stationaryPulses{ 2 };

    /// Minimum time after a command before idle-and-still can complete it, in seconds.
    static constexpr double s_minCompletionSec{ 0.5 };

    /** \name Configuration
     * @{
     */

    /// Stage name used for this device's elements in the shared INDI properties.
    const std::string &name() const;

    /// Set the stage name.
    void name( const std::string &n /**< [in] stage name */ );

    /// Device address character, '0'-'F'.
    char address() const;

    /// Set the device address; returns -1 if not a hex digit.
    int address( char a /**< [in] address character, case-insensitive */ );

    /// Configured device serial, or empty if not checked.
    const std::string &expectedSerial() const;

    /// Set the configured device serial to check against the `in` reply.
    void expectedSerial( const std::string &s /**< [in] 8-digit device serial, empty to skip the check */ );

    /// Requested velocity percentage.
    int velPercent() const;

    /// Set the requested velocity percentage, clamped to [model minimum, 100]; returns the value applied.
    int velPercent( int pct /**< [in] requested velocity percentage */ );

    ///@}

    /** \name Identity
     * @{
     */

    /// Whether the device has answered `in` with a valid reply on the current connection.
    bool identified() const;

    /// Whether the device answered `in` with a serial that does not match the configuration.
    bool wrongDevice() const;

    /// Model number from the `in` reply (21 for an ELL21), or 0.
    uint8_t model() const;

    /// Device serial from the `in` reply.
    const std::string &serial() const;

    /// Pulses per revolution from the `in` reply.
    uint32_t pulsesPerRev() const;

    /// Apply the payload of an `in` reply (the text after "IN").
    InfoResult applyInfo( const std::string &payload /**< [in] info payload, 30 hex digits */ );

    ///@}

    /** \name Position and Status
     * @{
     */

    /// Position in pulses as last reported.
    int32_t positionPulses() const;

    /// Position in degrees, wrapped into [0, 360).
    double positionDeg() const;

    /// Apply the payload of a `PO` frame; returns false if malformed.
    bool applyPosition( const std::string &payload /**< [in] 8 hex digit two's complement position */ );

    /// Apply a status code from a `GS` frame.
    void applyStatus( uint8_t code /**< [in] status/error code */ );

    /// Last status code received.
    uint8_t status() const;

    /// Whether the device has completed a home since power-on.
    bool homed() const;

    /// Convert degrees to pulses using the identified scale.
    int32_t degToPulses( double deg /**< [in] angle in degrees */ ) const;

    ///@}

    /** \name Operations
     * @{
     */

    /// Operation in progress.
    Pending pending() const;

    /// Target of the operation in progress in pulses (meaningful when `hasTarget()`).
    int32_t targetPulses() const;

    /// Whether the operation in progress has a target position.
    bool hasTarget() const;

    /// Whether a new motion may be commanded; sets `eventText()` to the reason when not.
    bool motionAllowed();

    /// Record that an absolute move to `target` pulses was sent.
    void beginMove( int32_t target /**< [in] target in pulses */ );

    /// Record that a home was sent.
    void beginHome();

    /// Record that an optimize was sent.
    void beginOptimize();

    /// Hold a motion request until the bus is free, replacing any request already held.
    void queue( Pending op,     /**< [in] the operation to hold, not `Pending::None` */
                double  deg = 0 /**< [in] target angle in degrees, for `Pending::Move` */
    );

    /// The motion request held for the bus, or `Pending::None`.
    Pending queued() const;

    /// Target angle in degrees of the held request (meaningful for `Pending::Move`).
    double queuedDeg() const;

    /// Drop the held motion request.
    void clearQueued();

    /// Record that a stop was requested and the device's reply code.
    /**
     * \returns true if the device reported the stop as unsupported (`GS03`)
     */
    bool stopRequested( int replyCode /**< [in] GS code replied to `st`, or -1 if none */ );

    /// Resolve the operation after a fully answered poll (position and status both received).
    Event pollAnswered( double toleranceDeg /**< [in] how close to the target counts as arrived */ );

    /// Account for a poll that missed a reply.
    Event pollMissed( double moveTimeoutSec,     /**< [in] motion timeout */
                      double optimizeTimeoutSec, /**< [in] optimize timeout */
                      int    maxMisses           /**< [in] consecutive misses tolerated when idle */
    );

    /// Whether the device is moving: 2 homing, 1 moving or optimizing (including motion not commanded here), 0 idle.
    int moving() const;

    /// Text for the last event: completion or failure reason.
    const std::string &eventText() const;

    /// Operator-facing reason the last operation failed, empty when OK; cleared by the next command.
    const std::string &lastError() const;

    /// State string published in `curr_state` while powered and the port is open.
    /** A held request reports as the motion it will become, HOMING or OPERATING, so clients wait for it. */
    std::string stateString() const;

    ///@}

    /** \name Lifecycle
     * @{
     */

    /// Forget identification, any operation in progress, and any held request, keeping the homed flag (communication
    /// loss).
    void disconnect();

    /// Reset everything that power-off invalidates, including the homed flag.
    void powerOff();

    ///@}

  protected:
    /** \name Configuration - Data
     * @{
     */

    /// Stage name used for this device's elements in the shared INDI properties.
    std::string m_name;

    /// Device address character, '0'-'F'.
    char m_address{ '0' };

    /// Configured device serial to check against the `in` reply, empty to skip.
    std::string m_expectedSerial;

    /// Requested velocity percentage.
    int m_velPercent{ 80 };

    ///@}

    /** \name Runtime State - Data
     * @{
     */

    /// Whether the device has answered `in` validly on the current connection.
    bool m_identified{ false };

    /// Whether the device answered with a serial that does not match the configuration.
    bool m_wrongDevice{ false };

    /// Model number from the `in` reply.
    uint8_t m_model{ 0 };

    /// Device serial from the `in` reply.
    std::string m_serial;

    /// Pulses per revolution from the `in` reply.
    uint32_t m_pulsesPerRev{ 0 };

    /// Position in pulses as last reported.
    int32_t m_posPulses{ 0 };

    /// Last status code received.
    uint8_t m_status{ 0 };

    /// Whether the device has completed a home since power-on.
    bool m_homed{ false };

    /// Operation in progress.
    Pending m_pending{ Pending::None };

    /// Motion request held until the bus is free.
    Pending m_queued{ Pending::None };

    /// Target angle in degrees of the held request.
    double m_queuedDeg{ 0 };

    /// Time the operation in progress was commanded.
    std::chrono::steady_clock::time_point m_pendingStart;

    /// Whether a busy status has been seen during the operation in progress.
    bool m_sawBusy{ false };

    /// First error status reported during the operation in progress, or 0.
    uint8_t m_latchedError{ 0 };

    /// Whether a stop was requested during the operation in progress.
    bool m_stopRequested{ false };

    /// Whether the operation in progress has a target position.
    bool m_hasTarget{ false };

    /// Target of the operation in progress in pulses.
    int32_t m_targetPulses{ 0 };

    /// Consecutive idle polls with an unchanged position during the operation in progress.
    int m_idleStreak{ 0 };

    /// Position at the previous idle poll.
    int32_t m_lastIdlePulses{ 0 };

    /// Whether the device reported busy with nothing commanded here (external motion).
    bool m_externalBusy{ false };

    /// Consecutive missed polls while idle.
    int m_misses{ 0 };

    /// Text for the last event.
    std::string m_eventText;

    /// Reason the last operation failed, cleared by the next command.
    std::string m_lastError;

    ///@}

    /// Record that an operation was sent.
    void begin_( Pending p /**< [in] the operation now in progress */ );

    /// Complete the operation in progress.
    Event complete_();

    /// Fail the operation in progress.
    Event fail_( const std::string &reason /**< [in] why it failed */ );

    /// Distance in pulses between two positions on the circle.
    double pulseDistance_( int32_t a, /**< [in] first position */
                           int32_t b  /**< [in] second position */
    ) const;

    /// Seconds since the operation in progress was commanded.
    double elapsed_() const;

    /// Operator-facing name of the operation in progress.
    std::string operationName_() const;
};

/* ========================= impl ========================= */

namespace elliptec
{

inline std::string statusMeaning( uint8_t code )
{
    switch( code )
    {
    case 0:
        return "OK";
    case 1:
        return "communication time out";
    case 2:
        return "mechanical time out";
    case 3:
        return "command error or not supported";
    case 4:
        return "value out of range";
    case 5:
        return "module isolated";
    case 6:
        return "module out of isolation";
    case 7:
        return "initializing error";
    case 8:
        return "thermal error";
    case 9:
        return "busy";
    case 10:
        return "sensor error";
    case 11:
        return "motor error";
    case 12:
        return "out of range";
    case 13:
        return "over current error";
    default:
        return "reserved status " + std::to_string( static_cast<int>( code ) );
    }
}

inline int minVelocityForModel( uint8_t model )
{
    // Protocol manual: "Lowest acceptable velocity to ELL16 and ELL21 is 50%". ELL22 shares the ELL21 drive.
    if( model == 16 || model == 21 || model == 22 )
    {
        return 50;
    }

    return 0;
}

inline double wrapDeg( double deg )
{
    double w = std::fmod( deg, 360.0 );
    if( w < 0 )
    {
        w += 360.0;
    }
    if( w >= 360.0 )
    {
        w -= 360.0;
    }
    return w;
}

inline double circularDistanceDeg( double a, double b )
{
    double d = std::abs( wrapDeg( a ) - wrapDeg( b ) );
    return ( d > 180.0 ) ? 360.0 - d : d;
}

inline bool parseHex( const std::string &s, size_t pos, size_t n, uint32_t &value )
{
    if( n == 0 || n > 8 || pos + n > s.size() )
    {
        return false;
    }

    uint32_t v = 0;
    for( size_t i = pos; i < pos + n; ++i )
    {
        char c = s[i];
        int  d;
        if( c >= '0' && c <= '9' )
        {
            d = c - '0';
        }
        else if( c >= 'A' && c <= 'F' )
        {
            d = 10 + c - 'A';
        }
        else if( c >= 'a' && c <= 'f' )
        {
            d = 10 + c - 'a';
        }
        else
        {
            return false;
        }
        v = ( v << 4 ) | static_cast<uint32_t>( d );
    }

    value = v;
    return true;
}

} // namespace elliptec

inline const std::string &elliptecDevice::name() const
{
    return m_name;
}

inline void elliptecDevice::name( const std::string &n )
{
    m_name = n;
}

inline char elliptecDevice::address() const
{
    return m_address;
}

inline int elliptecDevice::address( char a )
{
    char c = static_cast<char>( std::toupper( static_cast<unsigned char>( a ) ) );
    if( !std::isxdigit( static_cast<unsigned char>( c ) ) )
    {
        return -1;
    }

    m_address = c;
    return 0;
}

inline const std::string &elliptecDevice::expectedSerial() const
{
    return m_expectedSerial;
}

inline void elliptecDevice::expectedSerial( const std::string &s )
{
    m_expectedSerial = s;
}

inline int elliptecDevice::velPercent() const
{
    return m_velPercent;
}

inline int elliptecDevice::velPercent( int pct )
{
    pct          = std::max( 0, std::min( 100, pct ) );
    pct          = std::max( pct, elliptec::minVelocityForModel( m_model ) );
    m_velPercent = pct;
    return pct;
}

inline bool elliptecDevice::identified() const
{
    return m_identified;
}

inline bool elliptecDevice::wrongDevice() const
{
    return m_wrongDevice;
}

inline uint8_t elliptecDevice::model() const
{
    return m_model;
}

inline const std::string &elliptecDevice::serial() const
{
    return m_serial;
}

inline uint32_t elliptecDevice::pulsesPerRev() const
{
    return m_pulsesPerRev;
}

inline elliptecDevice::InfoResult elliptecDevice::applyInfo( const std::string &payload )
{
    // Fields: type(2) serial(8) year(4) firmware(2) hardware(2) travel(4) pulses per measurement unit(8).
    uint32_t model = 0;
    uint32_t ppr   = 0;
    uint32_t dummy = 0;
    if( payload.size() != 30 || !elliptec::parseHex( payload, 0, 2, model ) ||
        !elliptec::parseHex( payload, 2, 8, dummy ) || !elliptec::parseHex( payload, 10, 8, dummy ) ||
        !elliptec::parseHex( payload, 18, 4, dummy ) || !elliptec::parseHex( payload, 22, 8, ppr ) || ppr == 0 )
    {
        return InfoResult::Malformed;
    }

    const std::string serial = payload.substr( 2, 8 );
    if( !m_expectedSerial.empty() && serial != m_expectedSerial )
    {
        m_wrongDevice = true;
        m_identified  = false;
        m_serial      = serial;
        return InfoResult::SerialMismatch;
    }

    m_wrongDevice  = false;
    m_identified   = true;
    m_model        = static_cast<uint8_t>( model );
    m_serial       = serial;
    m_pulsesPerRev = ppr;
    m_misses       = 0;
    velPercent( m_velPercent ); // apply the model minimum
    return InfoResult::Ok;
}

inline int32_t elliptecDevice::positionPulses() const
{
    return m_posPulses;
}

inline double elliptecDevice::positionDeg() const
{
    if( m_pulsesPerRev == 0 )
    {
        return 0.0;
    }

    return elliptec::wrapDeg( static_cast<double>( m_posPulses ) * 360.0 / static_cast<double>( m_pulsesPerRev ) );
}

inline bool elliptecDevice::applyPosition( const std::string &payload )
{
    uint32_t v = 0;
    if( payload.size() < 8 || !elliptec::parseHex( payload, 0, 8, v ) )
    {
        return false;
    }

    m_posPulses = static_cast<int32_t>( v );
    return true;
}

inline void elliptecDevice::applyStatus( uint8_t code )
{
    m_status = code;

    if( code == 0x09 )
    {
        if( m_pending != Pending::None )
        {
            m_sawBusy = true;
        }
        return;
    }

    // A "not supported" reply to a stop request can arrive after the stop query has taken an earlier frame as its
    // reply; it says the stop was refused, not that the motion failed.
    if( code == 0x03 && m_stopRequested )
    {
        return;
    }

    if( code != 0x00 && m_pending != Pending::None && m_latchedError == 0 )
    {
        m_latchedError = code;
    }
}

inline uint8_t elliptecDevice::status() const
{
    return m_status;
}

inline bool elliptecDevice::homed() const
{
    return m_homed;
}

inline int32_t elliptecDevice::degToPulses( double deg ) const
{
    int32_t p = static_cast<int32_t>(
        std::llround( elliptec::wrapDeg( deg ) / 360.0 * static_cast<double>( m_pulsesPerRev ) ) );
    if( m_pulsesPerRev > 0 && p >= static_cast<int32_t>( m_pulsesPerRev ) )
    {
        p -= static_cast<int32_t>( m_pulsesPerRev );
    }
    return p;
}

inline elliptecDevice::Pending elliptecDevice::pending() const
{
    return m_pending;
}

inline int32_t elliptecDevice::targetPulses() const
{
    return m_targetPulses;
}

inline bool elliptecDevice::hasTarget() const
{
    return m_hasTarget;
}

inline bool elliptecDevice::motionAllowed()
{
    if( !m_identified )
    {
        m_eventText = "not connected";
        return false;
    }

    if( m_pending == Pending::Optimize )
    {
        m_eventText = "motor optimization in progress";
        return false;
    }

    if( m_pending == Pending::Home )
    {
        m_eventText = "homing in progress";
        return false;
    }

    return true;
}

inline void elliptecDevice::beginMove( int32_t target )
{
    begin_( Pending::Move );
    m_hasTarget    = true;
    m_targetPulses = target;
}

inline void elliptecDevice::beginHome()
{
    begin_( Pending::Home );
    m_hasTarget    = true;
    m_targetPulses = 0;
    m_homed        = false;
}

inline void elliptecDevice::beginOptimize()
{
    begin_( Pending::Optimize );
}

inline void elliptecDevice::queue( Pending op, double deg )
{
    m_queued    = op;
    m_queuedDeg = deg;
}

inline elliptecDevice::Pending elliptecDevice::queued() const
{
    return m_queued;
}

inline double elliptecDevice::queuedDeg() const
{
    return m_queuedDeg;
}

inline void elliptecDevice::clearQueued()
{
    m_queued = Pending::None;
}

inline bool elliptecDevice::stopRequested( int replyCode )
{
    if( m_pending != Pending::None )
    {
        m_stopRequested = true;
    }

    if( replyCode == 0x09 && m_pending != Pending::None )
    {
        m_sawBusy = true;
    }

    return replyCode == 0x03;
}

inline elliptecDevice::Event elliptecDevice::pollAnswered( double toleranceDeg )
{
    m_misses = 0;

    if( m_pending == Pending::None )
    {
        m_externalBusy = ( m_status == 0x09 );

        if( m_status != 0x00 && m_status != 0x09 )
        {
            const std::string text = "device reported " + elliptec::statusMeaning( m_status );
            if( m_lastError != text )
            {
                m_lastError = text;
                m_eventText = text;
                return Event::Failed;
            }
        }
        return Event::None;
    }

    if( m_latchedError != 0 )
    {
        return fail_( elliptec::statusMeaning( m_latchedError ) );
    }

    if( m_status == 0x09 )
    {
        m_idleStreak = 0;
        return Event::None;
    }

    if( m_status != 0x00 )
    {
        return Event::None;
    }

    if( !m_hasTarget )
    {
        // Operations without a target complete on idle status, guarded against a read taken before they started.
        if( !m_sawBusy && elapsed_() < 0.2 )
        {
            return Event::None;
        }
        return complete_();
    }

    // Idle status alone is not completion: require the position to hold still over two consecutive idle polls.
    if( m_idleStreak > 0 && pulseDistance_( m_posPulses, m_lastIdlePulses ) <= s_stationaryPulses )
    {
        ++m_idleStreak;
    }
    else
    {
        m_idleStreak = 1;
    }
    m_lastIdlePulses = m_posPulses;

    if( m_idleStreak < 2 || elapsed_() < s_minCompletionSec )
    {
        return Event::None;
    }

    const double tolPulses = toleranceDeg / 360.0 * static_cast<double>( m_pulsesPerRev );
    if( pulseDistance_( m_posPulses, m_targetPulses ) <= tolPulses )
    {
        return complete_();
    }

    if( m_stopRequested )
    {
        Event e     = complete_();
        m_eventText = operationName_() + " stopped by request before reaching its target";
        return e;
    }

    return fail_( "stopped away from target" );
}

inline elliptecDevice::Event
elliptecDevice::pollMissed( double moveTimeoutSec, double optimizeTimeoutSec, int maxMisses )
{
    if( m_pending != Pending::None )
    {
        // Silence is expected while moving or optimizing, until the operation times out.
        m_idleStreak = 0;

        const bool optimizing = ( m_pending == Pending::Optimize );
        if( elapsed_() < ( optimizing ? optimizeTimeoutSec : moveTimeoutSec ) )
        {
            return Event::None;
        }

        Event e  = fail_( optimizing ? "motor optimization timed out" : "no response from the device during motion" );
        m_misses = 1;
        return e;
    }

    if( ++m_misses <= maxMisses )
    {
        return Event::None;
    }

    disconnect();
    m_eventText = "device stopped answering";
    return Event::Lost;
}

inline int elliptecDevice::moving() const
{
    if( m_pending == Pending::Home )
    {
        return 2;
    }

    if( m_pending != Pending::None || m_externalBusy )
    {
        return 1;
    }

    return 0;
}

inline const std::string &elliptecDevice::eventText() const
{
    return m_eventText;
}

inline const std::string &elliptecDevice::lastError() const
{
    return m_lastError;
}

inline std::string elliptecDevice::stateString() const
{
    if( m_wrongDevice )
    {
        return "NODEVICE";
    }

    if( !m_identified )
    {
        return "NOTCONNECTED";
    }

    int mv = moving();
    if( mv == 2 || ( mv == 0 && m_queued == Pending::Home ) )
    {
        return "HOMING";
    }

    if( mv == 1 || m_queued != Pending::None )
    {
        return "OPERATING";
    }

    return m_homed ? "READY" : "NOTHOMED";
}

inline void elliptecDevice::disconnect()
{
    m_identified    = false;
    m_pending       = Pending::None;
    m_queued        = Pending::None;
    m_hasTarget     = false;
    m_latchedError  = 0;
    m_stopRequested = false;
    m_externalBusy  = false;
    m_idleStreak    = 0;
    m_misses        = 0;
}

inline void elliptecDevice::powerOff()
{
    disconnect();
    m_wrongDevice = false;
    m_homed       = false;
    m_lastError.clear();
}

inline void elliptecDevice::begin_( Pending p )
{
    m_pending       = p;
    m_pendingStart  = std::chrono::steady_clock::now();
    m_sawBusy       = false;
    m_latchedError  = 0;
    m_stopRequested = false;
    m_hasTarget     = false;
    m_idleStreak    = 0;
    m_externalBusy  = false;
    m_lastError.clear();
}

inline elliptecDevice::Event elliptecDevice::complete_()
{
    switch( m_pending )
    {
    case Pending::Home:
        m_homed     = true;
        m_eventText = "homing complete";
        break;
    case Pending::Optimize:
        m_homed     = false;
        m_eventText = "motor optimization complete; the stage must be re-homed";
        break;
    case Pending::Move:
    case Pending::None:
        m_eventText = "move complete";
        break;
    }

    m_pending       = Pending::None;
    m_hasTarget     = false;
    m_stopRequested = false;
    m_latchedError  = 0;
    return Event::Completed;
}

inline elliptecDevice::Event elliptecDevice::fail_( const std::string &reason )
{
    const std::string what = operationName_();
    if( m_pending == Pending::Home || m_pending == Pending::Optimize )
    {
        m_homed = false;
    }

    m_eventText = what + " failed: " + reason;
    m_lastError = what + " failed (" + reason + ")";

    m_pending       = Pending::None;
    m_hasTarget     = false;
    m_stopRequested = false;
    m_latchedError  = 0;
    return Event::Failed;
}

inline double elliptecDevice::pulseDistance_( int32_t a, int32_t b ) const
{
    double d = std::abs( static_cast<double>( a ) - static_cast<double>( b ) );
    if( m_pulsesPerRev > 0 )
    {
        d = std::fmod( d, static_cast<double>( m_pulsesPerRev ) );
        d = std::min( d, static_cast<double>( m_pulsesPerRev ) - d );
    }
    return d;
}

inline double elliptecDevice::elapsed_() const
{
    return std::chrono::duration<double>( std::chrono::steady_clock::now() - m_pendingStart ).count();
}

inline std::string elliptecDevice::operationName_() const
{
    switch( m_pending )
    {
    case Pending::Home:
        return "home";
    case Pending::Optimize:
        return "optimize";
    default:
        return "move";
    }
}

} // namespace app
} // namespace MagAOX

#endif // elliptecDevice_hpp
