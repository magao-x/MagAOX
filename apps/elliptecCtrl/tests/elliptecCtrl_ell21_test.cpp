/** \file elliptecCtrl_ell21_test.cpp
 * \brief Catch2 device-model tests for elliptecCtrl against a simulated ELL21 rotation mount.
 * \author Miles Lucas
 *
 * These tests drive the app through its normal lifecycle against a stateful PTY simulation of an
 * Elliptec ELL21 that follows the ELLx protocol manual (Issue 12): motion takes time, the device
 * answers `GS09` while busy, and a move reports `PO` when it completes.
 *
 * \ingroup elliptecCtrl_files
 */

#include "../../../tests/catch2/catch.hpp"
#include "../../../tests/testXWC.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <unistd.h>

#include "../elliptecCtrl.hpp"

namespace libXWCTest
{
namespace elliptecCtrlTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS

namespace
{

/// Build a per-test scratch directory under `/tmp`.
std::string ell21Root( const std::string &name /**< [in] unique scratch-directory name */ )
{
    std::string root = "/tmp/elliptecCtrl_ell21_test/" + name;
    std::filesystem::remove_all( root );
    std::filesystem::create_directories( root + "/telem" );
    std::filesystem::create_directories( root + "/calib" );
    return root;
}

/// Stateful simulation of an Elliptec ELL21 rotation mount behind a PTY.
/** Commands are parsed by their fixed Elliptec lengths. Motion commands acknowledge with `GS09`,
 * report `GS09` to `gs` while moving, and send an unsolicited `PO` (or an error `GS`) on completion.
 */
class fakeEll21
{
  public:
    /// Pulses per revolution reported by an ELL21.
    static constexpr int32_t pulsesPerRev = 65536;

    /// Construct the simulation and start its responder thread.
    fakeEll21()
    {
        m_masterFd = ::posix_openpt( O_RDWR | O_NOCTTY | O_NONBLOCK );
        if( m_masterFd < 0 )
        {
            throw std::runtime_error( "posix_openpt failed" );
        }

        if( ::grantpt( m_masterFd ) != 0 || ::unlockpt( m_masterFd ) != 0 )
        {
            throw std::runtime_error( "grantpt/unlockpt failed" );
        }

        char slaveName[256];
        if( ::ptsname_r( m_masterFd, slaveName, sizeof( slaveName ) ) != 0 )
        {
            throw std::runtime_error( "ptsname_r failed" );
        }

        m_slavePath = slaveName;
        m_thread    = std::thread(
            [this]()
            {
                run();
            } );
    }

    /// Stop the responder thread and close the PTY master.
    ~fakeEll21()
    {
        m_stop = true;

        if( m_thread.joinable() )
        {
            m_thread.join();
        }

        if( m_masterFd >= 0 )
        {
            ::close( m_masterFd );
        }
    }

    /// PTY slave path the app should open.
    const std::string &slavePath() const
    {
        return m_slavePath;
    }

    /// Set the duration of each move or home.
    void motionMs( int ms /**< [in] motion duration in milliseconds */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_motionMs = ms;
    }

    /// Set the duration of an optimize routine.
    void optimizeMs( int ms /**< [in] optimize duration in milliseconds */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_optimizeMs = ms;
    }

    /// Set the status code the device replies to `st` with.
    void stopReply( int code /**< [in] GS code replied to `st` */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_stopReply = code;
    }

    /// Make the next motion end with an error status instead of completing.
    void failNextMotion( int code /**< [in] GS error code to report */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_failNextCode = code;
    }

    /// Ignore the next `n` query commands (`gs`/`gp`) without replying.
    void dropQueries( int n /**< [in] number of queries to ignore */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_dropQueries = n;
    }

    /// Stop answering queries entirely while the device is busy.
    void silentWhileBusy( bool silent /**< [in] true to ignore queries during motion */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_silentWhileBusy = silent;
    }

    /// Send a raw frame to the app as soon as possible, unprompted.
    void inject( const std::string &raw /**< [in] raw bytes including CRLF */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_inject += raw;
    }

    /// Whether motion completion is reported with an unsolicited `PO` frame.
    void completionFrame( bool send /**< [in] true to send `PO` when motion completes */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_completionFrame = send;
    }

    /// Set the simulated position in pulses.
    void positionPulses( int32_t pulses /**< [in] new position */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_pos = pulses;
    }

    /// Simulated position in pulses.
    int32_t positionPulses() const
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        return m_pos;
    }

    /// Whether the simulated device is currently busy.
    bool busy() const
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        return m_busy;
    }

    /// Commands received so far, without the address character.
    std::vector<std::string> commands() const
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        return m_commands;
    }

    /// Number of received commands with the given two-letter mnemonic.
    size_t count( const std::string &mnemonic /**< [in] command mnemonic, e.g. "ho" */ ) const
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        size_t                      n = 0;
        for( const auto &c : m_commands )
        {
            if( c.compare( 0, 2, mnemonic ) == 0 )
            {
                ++n;
            }
        }
        return n;
    }

  private:
    typedef std::chrono::steady_clock clockT;

    int               m_masterFd{ -1 };
    std::string       m_slavePath;
    std::thread       m_thread;
    std::atomic<bool> m_stop{ false };

    mutable std::mutex       m_mutex;
    std::vector<std::string> m_commands;
    std::string              m_rx;

    int32_t            m_pos{ 0 };
    int32_t            m_target{ 0 };
    bool               m_busy{ false };
    bool               m_optimizing{ false };
    clockT::time_point m_doneAt;
    int                m_errorCode{ 0 };
    int                m_motionMs{ 150 };
    int                m_optimizeMs{ 400 };
    int                m_stopReply{ 3 };
    int                m_failNextCode{ 0 };
    int                m_dropQueries{ 0 };
    bool               m_silentWhileBusy{ false };
    bool               m_completionFrame{ true };
    std::string        m_inject;

    /// Format an 8-digit two's complement hex position.
    static std::string hex8( int32_t v )
    {
        char buf[9];
        std::snprintf( buf, sizeof( buf ), "%08X", static_cast<uint32_t>( v ) );
        return buf;
    }

    /// Format a GS reply.
    static std::string gs( int code )
    {
        char buf[16];
        std::snprintf( buf, sizeof( buf ), "0GS%02X\r\n", code & 0xFF );
        return buf;
    }

    /// Format a PO reply.
    static std::string po( int32_t v )
    {
        return "0PO" + hex8( v ) + "\r\n";
    }

    /// Parse 8 hex digits as a signed 32-bit value.
    static int32_t parseHex8( const std::string &s )
    {
        return static_cast<int32_t>( std::stoul( s, nullptr, 16 ) );
    }

    /// Write bytes to the PTY master.
    void writeRaw( const std::string &out )
    {
        size_t off = 0;
        while( off < out.size() )
        {
            ssize_t w = ::write( m_masterFd, out.data() + off, out.size() - off );
            if( w > 0 )
            {
                off += static_cast<size_t>( w );
                continue;
            }
            if( w < 0 && ( errno == EAGAIN || errno == EWOULDBLOCK ) )
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                continue;
            }
            return;
        }
    }

    /// Start a timed motion toward a target, caller holds m_mutex.
    void startMotion( int32_t target, int ms )
    {
        m_target = target;
        m_busy   = true;
        m_doneAt = clockT::now() + std::chrono::milliseconds( ms );
    }

    /// Complete motion if its time has elapsed, returning any unsolicited frame to send.
    std::string service()
    {
        if( !m_busy || clockT::now() < m_doneAt )
        {
            return "";
        }

        m_busy = false;

        if( m_optimizing )
        {
            m_optimizing = false;
            return gs( 0 );
        }

        if( m_failNextCode != 0 )
        {
            m_errorCode    = m_failNextCode;
            m_failNextCode = 0;
            return gs( m_errorCode );
        }

        m_pos = m_target;
        return m_completionFrame ? po( m_pos ) : "";
    }

    /// Length of a complete command starting at the head of the receive buffer, or 0 if incomplete.
    size_t commandLength() const
    {
        if( m_rx.size() < 3 )
        {
            return 0;
        }

        const std::string mn  = m_rx.substr( 1, 2 );
        size_t            len = 3;
        if( mn == "ho" )
        {
            len = 4;
        }
        else if( mn == "sv" )
        {
            len = 5;
        }
        else if( mn == "ma" || mn == "mr" )
        {
            len = 11;
        }

        return ( m_rx.size() >= len ) ? len : 0;
    }

    /// Handle one complete command, caller holds m_mutex; returns the reply bytes.
    std::string handle( const std::string &cmd )
    {
        const std::string mn  = cmd.substr( 1, 2 );
        const std::string arg = cmd.substr( 3 );
        m_commands.push_back( cmd.substr( 1 ) );

        if( mn == "gs" || mn == "gp" )
        {
            if( m_dropQueries > 0 )
            {
                --m_dropQueries;
                return "";
            }
            if( m_busy && m_silentWhileBusy )
            {
                return "";
            }
        }

        if( mn == "in" )
        {
            // type 0x15 (ELL21), SN, year, fw, hw, travel 0x0168 (360), pulses/rev 0x00010000
            return "0IN151234567820241701016800010000\r\n";
        }
        if( mn == "gs" )
        {
            if( m_busy )
            {
                return gs( 9 );
            }
            const int code = m_errorCode;
            m_errorCode    = 0; // reading status clears the error
            return gs( code );
        }
        if( mn == "gp" )
        {
            return po( m_pos );
        }
        if( mn == "ho" )
        {
            startMotion( 0, m_motionMs );
            return gs( 9 );
        }
        if( mn == "ma" )
        {
            startMotion( parseHex8( arg ), m_motionMs );
            return gs( 9 );
        }
        if( mn == "mr" )
        {
            startMotion( m_pos + parseHex8( arg ), m_motionMs );
            return gs( 9 );
        }
        if( mn == "st" )
        {
            return gs( m_stopReply );
        }
        if( mn == "sv" )
        {
            return gs( 0 );
        }
        if( mn == "om" )
        {
            m_optimizing = true;
            startMotion( m_pos, m_optimizeMs );
            return gs( 9 );
        }
        if( mn == "us" )
        {
            return gs( 0 );
        }

        return gs( 3 );
    }

    /// Responder loop: read commands, reply, and complete timed motion.
    void run()
    {
        while( !m_stop )
        {
            struct pollfd pfd = { m_masterFd, POLLIN, 0 };
            int           pr  = ::poll( &pfd, 1, 2 );

            std::string out;
            {
                std::lock_guard<std::mutex> lock( m_mutex );

                if( pr > 0 )
                {
                    char    buf[256];
                    ssize_t n;
                    while( ( n = ::read( m_masterFd, buf, sizeof( buf ) ) ) > 0 )
                    {
                        m_rx.append( buf, buf + n );
                    }
                }

                size_t len;
                while( ( len = commandLength() ) > 0 )
                {
                    out += handle( m_rx.substr( 0, len ) );
                    m_rx.erase( 0, len );
                }

                out += service();
                out += m_inject;
                m_inject.clear();
            }

            if( !out.empty() )
            {
                writeRaw( out );
            }
        }
    }
};

/// Test harness exposing what the device-model scenarios need from elliptecCtrl.
class ell21App : public MagAOX::app::elliptecCtrl
{
  public:
    /// Construct an app wired to the simulated device.
    ell21App( const std::string &name /**< [in] device name and scratch directory */,
              const std::string &port /**< [in] PTY slave path */ )
    {
        const std::string root = ell21Root( name );

        m_configName = name;
        m_basePath   = root;
        m_calibDir   = root + "/calib";
        m_sysPath    = root + "/sys";

        m_tel.logPath( root + "/telem" );
        m_tel.logExt( "bintel" );
        m_tel.logName( m_configName );
        m_tel.m_logLevel = logPrio::LOG_TELEM;
        m_maxInterval    = 3600.0;

        m_port              = port;
        m_startupDelayMs    = 0;
        m_readTimeoutMs     = 100;
        m_busyReadTimeoutMs = 100;
        m_powerState        = 1;
        m_powerTargetState  = 1;

        m_presetPositions = { 0.0F, 90.0F, 180.0F };
        m_presetNames     = { "zero", "ninety", "oneEighty" };

        m_indiP_home.setDevice( m_configName );
        m_indiP_home.setName( "home" );
        m_indiP_stop.setDevice( m_configName );
        m_indiP_stop.setName( "stop" );
        m_indiP_preset.setDevice( m_configName );
        m_indiP_preset.setName( "preset" );
    }

    /// Shut down telemetry and the serial port.
    ~ell21App()
    {
        m_tel.logShutdown( true );
        closePort_();
    }

    /// Run appStartup from INITIALIZED.
    int startup()
    {
        state( MagAOX::app::stateCodes::INITIALIZED );
        return appStartup();
    }

    /// Run one appLogic iteration and record the resulting FSM state.
    int logic()
    {
        int                         rv = appLogic();
        std::lock_guard<std::mutex> lock( m_statesMutex );
        if( m_states.empty() || m_states.back() != state() )
        {
            m_states.push_back( state() );
        }
        return rv;
    }

    /// Run appLogic repeatedly until a predicate holds or the timeout expires.
    bool runUntil( const std::function<bool()> &done /**< [in] completion predicate */,
                   int                          timeoutMs /**< [in] timeout in milliseconds */ )
    {
        auto start = std::chrono::steady_clock::now();
        while( std::chrono::steady_clock::now() - start < std::chrono::milliseconds( timeoutMs ) )
        {
            logic();
            if( done() )
            {
                return true;
            }
            std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
        }
        return false;
    }

    /// Press the standard `home` request through the stdMotionStage callback.
    int pressHome()
    {
        pcf::IndiProperty ip = m_indiP_home;
        ip["request"].setSwitchState( pcf::IndiElement::On );
        return newCallBack_m_indiP_home( ip );
    }

    /// Request a preset by number through the stdMotionStage callback.
    int requestPreset( float preset /**< [in] 1-based preset number */ )
    {
        pcf::IndiProperty ip = m_indiP_preset;
        ip["target"].set( preset );
        return newCallBack_m_indiP_preset( ip );
    }

    /// Configure the post-home offset in degrees.
    void homeOffset( double deg /**< [in] offset in degrees */ )
    {
        m_homeOffsetDeg = deg;
    }

    /// FSM states visited, with consecutive duplicates collapsed.
    std::vector<MagAOX::app::stateCodes::stateCodeT> states()
    {
        std::lock_guard<std::mutex> lock( m_statesMutex );
        return m_states;
    }

    /// Clear the recorded state history.
    void clearStates()
    {
        std::lock_guard<std::mutex> lock( m_statesMutex );
        m_states.clear();
    }

    /// Whether a given FSM state was visited.
    bool visited( MagAOX::app::stateCodes::stateCodeT s /**< [in] state to look for */ )
    {
        for( auto v : states() )
        {
            if( v == s )
            {
                return true;
            }
        }
        return false;
    }

    /// Whether the app considers the stage homed.
    bool homed() const
    {
        return m_homed;
    }

    /// Whether the serial link is open.
    bool connected() const
    {
        return m_connected;
    }

    /// Current position in degrees.
    double positionDeg() const
    {
        return m_posDeg;
    }

    /// Current stdMotionStage moving code.
    int8_t moving() const
    {
        return m_moving;
    }

  private:
    std::mutex                                       m_statesMutex;
    std::vector<MagAOX::app::stateCodes::stateCodeT> m_states;
};

/// Connect the app to the simulated device and wait for the first poll to settle.
void connectApp( ell21App &app )
{
    REQUIRE( app.startup() == 0 );
    REQUIRE( app.runUntil(
        [&]()
        {
            return app.connected() && app.state() == MagAOX::app::stateCodes::NOTHOMED;
        },
        2000 ) );
    app.clearStates();
}

} // namespace

/// \endcond

/// A home with a post-home offset completes cleanly on an ELL21.
/** Pressing `home` must take the stage HOMING -> READY, apply the offset, and never pass through
 * NOTCONNECTED. Regression for the hardware loop HOMING -> NOTCONNECTED -> NOTHOMED.
 *
 * \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "ELL21 home with offset completes without dropping the connection", "[elliptecCtrl][ell21]" )
{
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecCtrl::startHoming();
    MagAOX::app::elliptecCtrl::appLogic();
    // clang-format on
#endif

    fakeEll21 dev;
    dev.motionMs( 300 );

    ell21App app( "ell21-home", dev.slavePath() );
    app.homeOffset( 90.0 );
    connectApp( app );

    REQUIRE( app.pressHome() == 0 );

    REQUIRE( app.runUntil(
        [&]()
        {
            return app.state() == MagAOX::app::stateCodes::READY;
        },
        5000 ) );

    REQUIRE( app.homed() );
    REQUIRE_FALSE( app.visited( MagAOX::app::stateCodes::NOTCONNECTED ) );
    REQUIRE( app.visited( MagAOX::app::stateCodes::HOMING ) );
    REQUIRE( app.positionDeg() == Approx( 90.0 ).margin( 0.01 ) );
    REQUIRE( dev.count( "ho" ) == 1 );
}

/// Pressing `home` from the INDI thread while appLogic runs in the main thread completes cleanly.
/** Serial I/O from INDI callbacks and the main loop must be serialized so neither thread consumes
 * the other's reply frames.
 *
 * \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "ELL21 home from the INDI thread during polling completes without misses", "[elliptecCtrl][ell21]" )
{
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecCtrl::startHoming();
    MagAOX::app::elliptecCtrl::appLogic();
    // clang-format on
#endif

    fakeEll21 dev;
    dev.motionMs( 300 );

    ell21App app( "ell21-concurrent", dev.slavePath() );
    connectApp( app );

    std::atomic<bool> stopLoop{ false };
    std::thread       mainLoop(
        [&]()
        {
            while( !stopLoop )
            {
                app.logic();
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
            }
        } );

    std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
    int homeRv = app.pressHome();

    auto start = std::chrono::steady_clock::now();
    while( std::chrono::steady_clock::now() - start < std::chrono::milliseconds( 5000 ) )
    {
        if( app.homed() && app.state() == MagAOX::app::stateCodes::READY )
        {
            break;
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    }

    stopLoop = true;
    mainLoop.join();

    REQUIRE( homeRv == 0 );
    REQUIRE( app.homed() );
    REQUIRE( app.state() == MagAOX::app::stateCodes::READY );
    REQUIRE_FALSE( app.visited( MagAOX::app::stateCodes::NOTCONNECTED ) );
}

/// A stray `PO` frame during a move neither desynchronizes the poll nor completes the move early.
/** The device here reports completion only through `gs`, so a reader that takes "the next frame" as the
 * reply stays one frame out of step after the stray frame and never sees the move finish.
 *
 * \ingroup elliptecCtrl_unit_test
 */
TEST_CASE( "ELL21 stray PO frame during a move is handled", "[elliptecCtrl][ell21]" )
{
#ifdef ELLIPTECCTRL_TEST_DOXYGEN_REF
    // clang-format off
    MagAOX::app::elliptecCtrl::pollDevice_();
    // clang-format on
#endif

    fakeEll21 dev;
    dev.motionMs( 600 );
    dev.completionFrame( false );

    ell21App app( "ell21-stray", dev.slavePath() );
    connectApp( app );

    REQUIRE( app.requestPreset( 2 ) == 0 ); // 90 deg
    REQUIRE( app.runUntil(
        [&]()
        {
            return app.state() == MagAOX::app::stateCodes::OPERATING;
        },
        1000 ) );

    // A stale position frame arrives unprompted while the device is still busy.
    dev.inject( "0PO00000000\r\n" );

    // The move must not be declared complete while the device still reports busy.
    bool earlyCompletion = false;
    auto start           = std::chrono::steady_clock::now();
    while( dev.busy() && std::chrono::steady_clock::now() - start < std::chrono::milliseconds( 2000 ) )
    {
        app.logic();
        if( dev.busy() && app.moving() <= 0 )
        {
            earlyCompletion = true;
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    }

    REQUIRE_FALSE( earlyCompletion );
    REQUIRE( app.runUntil(
        [&]()
        {
            return app.moving() == 0 || app.moving() == -1;
        },
        2000 ) );

    REQUIRE( app.positionDeg() == Approx( 90.0 ).margin( 0.01 ) );
    REQUIRE_FALSE( app.visited( MagAOX::app::stateCodes::NOTCONNECTED ) );
}

} // namespace elliptecCtrlTest
} // namespace libXWCTest
