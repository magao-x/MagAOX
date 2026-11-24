/** \file fakeElliptecBus.hpp
 * \brief A simulated bus of Thorlabs Elliptec devices behind a PTY, for unit tests.
 * \author Miles Lucas
 *
 * \ingroup elliptecLowLevel_files
 */

#ifndef fakeElliptecBus_hpp
#define fakeElliptecBus_hpp

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <map>
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

namespace libXWCTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS

/// One simulated Elliptec device; all fields are guarded by the bus mutex.
struct simElliptec
{
    typedef std::chrono::steady_clock clockT; ///< Clock for motion timing.

    std::string serial{ "12345678" }; ///< Device serial reported by `in`.
    uint8_t     model{ 21 };          ///< Model number reported by `in`.

    int32_t            pos{ 0 };            ///< Position in pulses.
    int32_t            target{ 0 };         ///< Target of the motion in progress.
    int32_t            startPos{ 0 };       ///< Position when the motion started.
    bool               busy{ false };       ///< Whether a motion or optimize is in progress.
    bool               optimizing{ false }; ///< Whether the operation in progress is an optimize.
    clockT::time_point startAt;             ///< When the operation started.
    clockT::time_point doneAt;              ///< When the operation completes.
    int                errorCode{ 0 };      ///< Error reported to the next `gs`, then cleared.

    int     motionMs{ 150 };          ///< Duration of each move or home.
    int     optimizeMs{ 400 };        ///< Duration of an optimize.
    int     stopReply{ 3 };           ///< Status code replied to `st`.
    int     failNextCode{ 0 };        ///< Error code the next motion ends with.
    int32_t landingError{ 0 };        ///< Final position offset from the target, in pulses.
    int     bootSilent{ 0 };          ///< Commands of any kind to ignore (booting).
    int     dropQueries{ 0 };         ///< `gs`/`gp` queries to ignore.
    bool    ackMotion{ true };        ///< Whether motion commands are acknowledged with `GS09`.
    bool    completionFrame{ true };  ///< Whether completion is announced with `PO`.
    bool    silentWhileBusy{ false }; ///< Whether queries are ignored while busy.
    int     falseIdleEvery{ 0 };      ///< Answer one poll in this many while busy, with a false idle.
    int     busyPolls{ 0 };           ///< `gp` queries received while busy.
    bool    answerNextGs{ false };    ///< Whether the next busy `gs` gets a false idle.
    bool    truncateInfo{ false };    ///< Whether the next `in` reply is truncated.

    /// Behave like the ELL21 observed on hardware: no acknowledgement, mostly silent while moving.
    void hardwareQuirks( int falseIdle /**< [in] answer one poll in this many while busy, 0 for never */ )
    {
        ackMotion       = false;
        completionFrame = false;
        silentWhileBusy = true;
        falseIdleEvery  = falseIdle;
    }
};

/// A PTY-backed Elliptec bus with any number of simulated devices at distinct addresses.
/** Commands are parsed by their fixed Elliptec lengths and routed by address; commands to addresses with no device
 * get no reply, as on a real bus.
 */
class fakeElliptecBus
{
  public:
    /// Pulses per revolution of the simulated ELL21.
    static constexpr int32_t pulsesPerRev = 65536;

    /// Open the PTY and start the responder.
    fakeElliptecBus()
    {
        m_masterFd = ::posix_openpt( O_RDWR | O_NOCTTY | O_NONBLOCK );
        if( m_masterFd < 0 || ::grantpt( m_masterFd ) != 0 || ::unlockpt( m_masterFd ) != 0 )
        {
            throw std::runtime_error( "posix_openpt failed" );
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

    /// Stop the responder and close the PTY.
    ~fakeElliptecBus()
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

    /// PTY slave path to open.
    const std::string &slavePath() const
    {
        return m_slavePath;
    }

    /// Add a device at an address.
    void addDevice( char addr, /**< [in] address character */ const std::string &serial /**< [in] device serial */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        simElliptec                 d;
        d.serial        = serial;
        m_devices[addr] = d;
    }

    /// Change a device's simulation settings.
    void configure( char addr, /**< [in] address */ const std::function<void( simElliptec & )> &f /**< [in] edit */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        f( m_devices.at( addr ) );
    }

    /// Copy of a device's simulation state.
    simElliptec device( char addr /**< [in] address */ ) const
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        return m_devices.at( addr );
    }

    /// Whether a device is busy.
    bool busy( char addr /**< [in] address */ ) const
    {
        return device( addr ).busy;
    }

    /// Send raw bytes unprompted.
    void inject( const std::string &raw /**< [in] bytes including CRLF */ )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_inject += raw;
    }

    /// Number of commands with this mnemonic sent to an address.
    size_t count( char addr, /**< [in] address */ const std::string &mnemonic /**< [in] e.g. "ma" */ ) const
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        size_t                      n = 0;
        for( const auto &c : m_commands )
        {
            if( c.size() >= 3 && c[0] == addr && c.compare( 1, 2, mnemonic ) == 0 )
            {
                ++n;
            }
        }
        return n;
    }

  private:
    typedef simElliptec::clockT clockT;

    int               m_masterFd{ -1 }; ///< PTY master.
    std::string       m_slavePath;      ///< PTY slave path.
    std::thread       m_thread;         ///< Responder thread.
    std::atomic<bool> m_stop{ false };  ///< Signals the responder to exit.

    mutable std::mutex          m_mutex;    ///< Guards everything below.
    std::map<char, simElliptec> m_devices;  ///< Devices by address.
    std::vector<std::string>    m_commands; ///< Commands received, with address.
    std::string                 m_rx;       ///< Unparsed received bytes.
    std::string                 m_inject;   ///< Bytes queued to send unprompted.

    static std::string hex8( int32_t v )
    {
        char buf[9];
        std::snprintf( buf, sizeof( buf ), "%08X", static_cast<uint32_t>( v ) );
        return buf;
    }

    static std::string gs( char a, int code )
    {
        char buf[16];
        std::snprintf( buf, sizeof( buf ), "%cGS%02X\r\n", a, code & 0xFF );
        return buf;
    }

    static std::string po( char a, int32_t v )
    {
        return std::string( 1, a ) + "PO" + hex8( v ) + "\r\n";
    }

    static int32_t intermediate( const simElliptec &d )
    {
        double total = std::chrono::duration<double>( d.doneAt - d.startAt ).count();
        double done  = std::chrono::duration<double>( clockT::now() - d.startAt ).count();
        double f     = ( total > 0 ) ? std::min( 1.0, done / total ) : 1.0;
        return d.startPos + static_cast<int32_t>( ( d.target - d.startPos ) * f );
    }

    static void startMotion( simElliptec &d, int32_t target, int ms )
    {
        d.startPos = d.pos;
        d.startAt  = clockT::now();
        d.target   = target;
        d.busy     = true;
        d.doneAt   = d.startAt + std::chrono::milliseconds( ms );
    }

    std::string service( char a, simElliptec &d )
    {
        if( !d.busy || clockT::now() < d.doneAt )
        {
            return "";
        }

        d.busy = false;

        if( d.optimizing )
        {
            d.optimizing = false;
            return gs( a, 0 );
        }

        if( d.failNextCode != 0 )
        {
            d.errorCode    = d.failNextCode;
            d.failNextCode = 0;
            return gs( a, d.errorCode );
        }

        d.pos = d.target + d.landingError;
        return d.completionFrame ? po( a, d.pos ) : "";
    }

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

    std::string handle( const std::string &cmd )
    {
        m_commands.push_back( cmd );

        const char a  = cmd[0];
        auto       it = m_devices.find( a );
        if( it == m_devices.end() )
        {
            return ""; // nobody at this address
        }

        simElliptec      &d   = it->second;
        const std::string mn  = cmd.substr( 1, 2 );
        const std::string arg = cmd.substr( 3 );

        if( d.bootSilent > 0 )
        {
            --d.bootSilent;
            return "";
        }

        if( mn == "gs" || mn == "gp" )
        {
            if( d.dropQueries > 0 )
            {
                --d.dropQueries;
                return "";
            }
            if( d.busy && d.silentWhileBusy )
            {
                if( mn == "gp" && d.falseIdleEvery > 0 && ++d.busyPolls % d.falseIdleEvery == 0 )
                {
                    d.answerNextGs = true;
                    return po( a, intermediate( d ) );
                }
                if( mn == "gs" && d.answerNextGs )
                {
                    d.answerNextGs = false;
                    return gs( a, 0 );
                }
                return "";
            }
        }

        if( mn == "in" )
        {
            char buf[64];
            std::snprintf(
                buf, sizeof( buf ), "%cIN%02X%s2024170101680001000%c\r\n", a, d.model, d.serial.c_str(), '0' );
            std::string r = buf;
            if( d.truncateInfo )
            {
                d.truncateInfo = false;
                r              = r.substr( 0, 12 ) + "\r\n";
            }
            return r;
        }
        if( mn == "gs" )
        {
            if( d.busy )
            {
                return gs( a, 9 );
            }
            int code    = d.errorCode;
            d.errorCode = 0;
            return gs( a, code );
        }
        if( mn == "gp" )
        {
            return po( a, d.pos );
        }
        if( mn == "ho" )
        {
            startMotion( d, 0, d.motionMs );
            return d.ackMotion ? gs( a, 9 ) : "";
        }
        if( mn == "ma" )
        {
            startMotion( d, static_cast<int32_t>( std::stoul( arg, nullptr, 16 ) ), d.motionMs );
            return d.ackMotion ? gs( a, 9 ) : "";
        }
        if( mn == "st" )
        {
            return gs( a, d.stopReply );
        }
        if( mn == "sv" || mn == "us" )
        {
            return gs( a, 0 );
        }
        if( mn == "om" )
        {
            d.optimizing = true;
            startMotion( d, d.pos, d.optimizeMs );
            return d.ackMotion ? gs( a, 9 ) : "";
        }

        return gs( a, 3 );
    }

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
            if( w < 0 && ( errno == EAGAIN || errno == EWOULDBLOCK ) && !m_stop )
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                continue;
            }
            return;
        }
    }

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

                for( auto &kv : m_devices )
                {
                    out += service( kv.first, kv.second );
                }

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

/// \endcond

} // namespace libXWCTest

#endif // fakeElliptecBus_hpp
