/** \file AD7689_test.cpp
 * \brief Hardware-independent tests of the actual AD7689 GPIO/SPI implementation.
 * \ingroup ad7689Ctrl_unit_test
 */
#include "../../../tests/testXWC.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <gpiod.h>
#include <limits>
#include <linux/spi/spidev.h>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Limit access indirection to the driver declaration after its standard headers are included.
#define private public
#include "../dependencies/AD7689.h"
#undef private

/// \cond AD7689_DRIVER_FIXTURE
namespace
{
/// Independent converter state: pending serial CFG latches only at the next EOC.
struct Converter
{
    uint64_t m_time{ 1000000 }, m_rising{ 0 }, m_falling{ 0 }, m_serialEnd{ 0 }; ///< Independent wire event times.
    uint64_t m_highHold{ 10000 }, m_setup{ 1000 }, m_quiet{ 1000 }; ///< Configured guards checked at wire events.
    uint16_t m_active{ 0 }, m_pending{ 0 }, m_capturedCfg{ 0 }, m_capturedCode{ 0 };
    uint32_t m_mode{ 0 };
    uint8_t  m_bits{ 8 }, m_order{ 0 };
    unsigned m_baud{ 1000000 }, m_pulses{ 0 }, m_closes{ 0 }, m_releases{ 0 }, m_chips{ 0 }, m_allocations{ 0 };
    unsigned m_calls{ 0 }, m_failAt{ 0 };
    bool     m_high{ false }, m_requested{ false }, m_pendingUpdate{ false }, m_corrupt{ false }, m_shortRead{ false };
    bool     m_busy{ false }, m_openFailure{ false }, m_badMode{ false }, m_clockFailure{ false }, m_backwards{ false };
    bool     m_clockAfterHigh{ false }; ///< Inject a clock failure immediately after a successful rising edge.
    bool     m_lowFailure{ false }, m_highFailure{ false }, m_closeFailure{ false }, m_constant{ false };
    uint16_t m_constantCode{ 0xffff };
    std::vector<unsigned> m_channels; ///< Physical inputs independently captured at each rising edge.
    std::vector<unsigned> m_lengths;  ///< Byte counts observed on the fake serial bus.
    std::vector<uint64_t> m_edges;    ///< Independent monotonic edge times for timing checks.
    std::vector<uint16_t> m_written;  ///< Actual left-aligned CFG words decoded from MOSI.

    /// Inject a selected setup/transport operation failure.
    bool fails();
};
bool Converter::fails()
{
    ++m_calls;
    if( m_failAt && m_calls == m_failAt )
    {
        errno = EIO;
        return true;
    }
    return false;
}
Converter hardware;

/// Fake open cannot access a real endpoint.
int fakeOpen( const char *path /**< [in] requested endpoint */, int flags /**< [in] open flags */ )
{
    REQUIRE( std::string( path ) == "/dev/spidev0.0" );
    REQUIRE( flags == ( O_RDWR | O_CLOEXEC ) );
    if( hardware.m_openFailure || hardware.fails() )
    {
        errno = ENOENT;
        return -1;
    }
    return 42;
}

/// Close only the expected fake descriptor.
int fakeClose( int fd /**< [in] fake descriptor */ )
{
    REQUIRE( fd == 42 );
    ++hardware.m_closes;
    return hardware.m_closeFailure ? -1 : 0;
}

/// Advance a deterministic monotonic clock; realtime has a separate epoch.
int fakeClock( clockid_t clock /**< [in] clock domain */, timespec *ts /**< [out] synthetic timestamp */ )
{
    if( hardware.m_clockFailure )
    {
        errno = EIO;
        return -1;
    }
    hardware.m_time += 100;
    if( hardware.m_backwards )
        hardware.m_time -= 200;
    uint64_t value = hardware.m_time + ( clock == CLOCK_REALTIME ? 1700000000000000000ULL : 0 );
    ts->tv_sec     = value / 1000000000ULL;
    ts->tv_nsec    = value % 1000000000ULL;
    return 0;
}

/// Complete a relative cleanup hold independently of a failed clock read.
int fakeSleep( const timespec *requested /**< [in] relative hold */, timespec * /**< [out] unused remainder */ )
{
    hardware.m_time += requested->tv_sec * 1000000000ULL + requested->tv_nsec;
    return 0;
}

/// Model mode setup and the wire protocol independently of the driver's state.
int fakeIoctl( int           fd /**< [in] fake descriptor */,
               unsigned long operation /**< [in] SPI command */,
               void         *data /**< [in/out] SPI setup or transfer */ )
{
    REQUIRE( fd == 42 );
    if( hardware.fails() )
        return -1;
    if( operation == SPI_IOC_WR_MODE32 )
        hardware.m_mode = *static_cast<uint32_t *>( data );
    else if( operation == SPI_IOC_RD_MODE32 )
        *static_cast<uint32_t *>( data ) = hardware.m_badMode ? 0 : hardware.m_mode;
    else if( operation == SPI_IOC_WR_BITS_PER_WORD )
        hardware.m_bits = *static_cast<uint8_t *>( data );
    else if( operation == SPI_IOC_RD_BITS_PER_WORD )
        *static_cast<uint8_t *>( data ) = hardware.m_bits;
    else if( operation == SPI_IOC_WR_LSB_FIRST )
        hardware.m_order = *static_cast<uint8_t *>( data );
    else if( operation == SPI_IOC_RD_LSB_FIRST )
        *static_cast<uint8_t *>( data ) = hardware.m_order;
    else if( operation == SPI_IOC_WR_MAX_SPEED_HZ )
        hardware.m_baud = *static_cast<unsigned *>( data );
    else
    {
        REQUIRE( operation == SPI_IOC_MESSAGE( 1 ) );
        auto &xfer = *static_cast<spi_ioc_transfer *>( data );
        REQUIRE_FALSE( hardware.m_high );
        REQUIRE( hardware.m_time - hardware.m_falling >= hardware.m_setup );
        REQUIRE( hardware.m_requested );
        REQUIRE( hardware.m_mode == ( SPI_MODE_0 | SPI_NO_CS ) );
        REQUIRE( hardware.m_bits == 8 );
        REQUIRE( hardware.m_order == 0 );
        REQUIRE( xfer.bits_per_word == 8 );
        REQUIRE( xfer.cs_change == 0 );
        REQUIRE( ( xfer.len == 2 || xfer.len == 4 ) );
        if( !( hardware.m_capturedCfg & 1U ) )
            REQUIRE( xfer.len == 4 );
        const auto    *tx  = reinterpret_cast<const uint8_t *>( static_cast<uintptr_t>( xfer.tx_buf ) );
        auto          *rx  = reinterpret_cast<uint8_t *>( static_cast<uintptr_t>( xfer.rx_buf ) );
        const uint16_t cfg = static_cast<uint16_t>( ( ( tx[0] << 8 ) | tx[1] ) >> 2 );
        REQUIRE( ( tx[1] & 3 ) == 0 );
        hardware.m_written.push_back( cfg );
        hardware.m_lengths.push_back( xfer.len );
        if( cfg & ( 1U << 13 ) )
        {
            hardware.m_pending       = cfg;
            hardware.m_pendingUpdate = true;
        }
        else
            hardware.m_pendingUpdate = false;
        rx[0] = hardware.m_capturedCode >> 8;
        rx[1] = hardware.m_capturedCode & 0xff;
        if( xfer.len == 4 )
        {
            uint16_t rb = static_cast<uint16_t>( hardware.m_capturedCfg << 2 );
            if( hardware.m_corrupt )
                rb ^= 0x0200;
            rx[2] = rb >> 8;
            rx[3] = rb & 0xff;
        }
        hardware.m_time += xfer.len * 8ULL * 1000000000ULL / hardware.m_baud;
        hardware.m_serialEnd = hardware.m_time;
        return hardware.m_shortRead ? static_cast<int>( xfer.len ) - 1 : xfer.len;
    }
    return 0;
}
} // namespace

gpiod_chip *gpiod_chip_open( const char * )
{
    if( hardware.fails() )
        return nullptr;
    ++hardware.m_chips;
    return new gpiod_chip;
}
void gpiod_chip_close( gpiod_chip *chip )
{
    --hardware.m_chips;
    delete chip;
}
gpiod_line_settings *gpiod_line_settings_new()
{
    if( hardware.fails() )
        return nullptr;
    ++hardware.m_allocations;
    return new gpiod_line_settings;
}
void gpiod_line_settings_free( gpiod_line_settings *value )
{
    --hardware.m_allocations;
    delete value;
}
gpiod_line_config *gpiod_line_config_new()
{
    if( hardware.fails() )
        return nullptr;
    ++hardware.m_allocations;
    return new gpiod_line_config;
}
void gpiod_line_config_free( gpiod_line_config *value )
{
    --hardware.m_allocations;
    delete value;
}
gpiod_request_config *gpiod_request_config_new()
{
    if( hardware.fails() )
        return nullptr;
    ++hardware.m_allocations;
    return new gpiod_request_config;
}
void gpiod_request_config_free( gpiod_request_config *value )
{
    --hardware.m_allocations;
    delete value;
}
void gpiod_line_settings_set_active_low( gpiod_line_settings *, bool inverted )
{
    REQUIRE_FALSE( inverted );
}
int gpiod_line_settings_set_direction( gpiod_line_settings *, gpiod_line_direction dir )
{
    REQUIRE( dir == GPIOD_LINE_DIRECTION_OUTPUT );
    return hardware.fails() ? -1 : 0;
}
int gpiod_line_settings_set_drive( gpiod_line_settings *, gpiod_line_drive drive )
{
    REQUIRE( drive == GPIOD_LINE_DRIVE_PUSH_PULL );
    return hardware.fails() ? -1 : 0;
}
int gpiod_line_settings_set_output_value( gpiod_line_settings *, gpiod_line_value value )
{
    REQUIRE( value == GPIOD_LINE_VALUE_INACTIVE );
    return hardware.fails() ? -1 : 0;
}
int gpiod_line_config_add_line_settings( gpiod_line_config *, const unsigned *offsets, size_t n, gpiod_line_settings * )
{
    REQUIRE( n == 1 );
    REQUIRE( offsets[0] == 8 );
    return hardware.fails() ? -1 : 0;
}
void gpiod_request_config_set_consumer( gpiod_request_config *, const char *value )
{
    REQUIRE( std::string( value ) == "ad7689Ctrl-cnv" );
}
gpiod_line_request *gpiod_chip_request_lines( gpiod_chip *, gpiod_request_config *, gpiod_line_config * )
{
    if( hardware.m_busy || hardware.fails() )
    {
        errno = EBUSY;
        return nullptr;
    }
    REQUIRE_FALSE( hardware.m_requested );
    hardware.m_requested = true;
    return new gpiod_line_request;
}
void gpiod_line_request_release( gpiod_line_request *request )
{
    hardware.m_requested = false;
    ++hardware.m_releases;
    delete request;
}
int gpiod_line_request_set_value( gpiod_line_request *, unsigned offset, gpiod_line_value value )
{
    REQUIRE( offset == 8 );
    if( hardware.fails() )
        return -1;
    if( value == GPIOD_LINE_VALUE_ACTIVE )
    {
        if( hardware.m_highFailure )
        {
            errno = EIO;
            return -1;
        }
        REQUIRE_FALSE( hardware.m_high );
        if( hardware.m_rising )
            REQUIRE( hardware.m_time - hardware.m_rising >= 5000 );
        REQUIRE( hardware.m_time - hardware.m_serialEnd >= hardware.m_quiet );
        hardware.m_rising = hardware.m_time;
        hardware.m_edges.push_back( hardware.m_time );
        hardware.m_high = true;
        if( hardware.m_clockAfterHigh )
            hardware.m_clockFailure = true;
        hardware.m_capturedCfg = hardware.m_active;
        const unsigned channel = ( hardware.m_active >> 7 ) & 7;
        hardware.m_channels.push_back( channel );
        ++hardware.m_pulses;
        hardware.m_capturedCode = hardware.m_constant
                                      ? hardware.m_constantCode
                                      : static_cast<uint16_t>( channel * 8192 + ( hardware.m_pulses & 0xfff ) );
    }
    else if( hardware.m_high )
    {
        if( hardware.m_lowFailure )
        {
            errno = EIO;
            return -1;
        }
        REQUIRE( hardware.m_time - hardware.m_rising >= hardware.m_highHold );
        hardware.m_falling = hardware.m_time;
        hardware.m_high    = false;
        if( hardware.m_pendingUpdate )
            hardware.m_active = hardware.m_pending;
        hardware.m_pendingUpdate = false;
    }
    return 0;
}

// Test-local indirection exercises the unchanged production implementation body.
#define open fakeOpen
#define close fakeClose
#define ioctl fakeIoctl
#define clock_gettime fakeClock
#define nanosleep fakeSleep
#include "../dependencies/AD7689.cpp"
#undef clock_gettime
#undef nanosleep
#undef ioctl
#undef close
#undef open
/// \endcond

namespace libXWCTest
{
namespace ad7689CtrlTest
{
using AD7689Lib::AD7689;
using AD7689Lib::Settings;

/** Verify CFG fields, complete unsigned decoding, and the 65536 scaling denominator.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 configuration and full-width data", "[ad7689Ctrl][driver]" )
{
    for( unsigned i = 0; i < 8; ++i )
    {
        REQUIRE( AD7689::configuration( i ) == ( 0x3c71 | ( i << 7 ) ) );
        REQUIRE( AD7689::configuration( i, true ) == ( 0x3c70 | ( i << 7 ) ) );
    }
    REQUIRE_THROWS_AS( AD7689::configuration( 8 ), std::out_of_range );
    for( unsigned code : { 0U, 1U, 4095U, 4096U, 32767U, 32768U, 0xabcdU, 65535U } )
    {
        uint8_t bytes[]{ static_cast<uint8_t>( code >> 8 ), static_cast<uint8_t>( code ) };
        REQUIRE( AD7689::decode( bytes ) == code );
    }
    REQUIRE( AD7689::volts( 32768, 3.32 ) == Approx( 1.66 ) );
    REQUIRE( AD7689::volts( 65535, 3.32 ) < 3.32 );
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    MagAOX::app::ad7689Ctrl::readScan;
    AD7689Lib::AD7689::configuration;
    AD7689Lib::AD7689::decode;
    AD7689Lib::AD7689::volts;
#endif
    // clang-format on
}

/** Reject invalid physical inputs/settings and analytically impossible acquisition rates.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 settings and rate admission", "[ad7689Ctrl][driver]" )
{
    Settings settings;
    REQUIRE( AD7689::validSettings( settings ) );
    REQUIRE( AD7689::validChannels( { 7, 0, 4 } ) );
    REQUIRE_FALSE( AD7689::validChannels( {} ) );
    REQUIRE_FALSE( AD7689::validChannels( { 0, 0 } ) );
    REQUIRE_FALSE( AD7689::validChannels( { 8 } ) );
    REQUIRE( AD7689::scanTime_ns( settings, 2 ) == Approx( 56000 ) );
    REQUIRE_FALSE( AD7689::validRate( settings, 8, 8000 ) );
    for( double rate :
         { 0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
        REQUIRE_FALSE( AD7689::validRate( settings, 2, rate ) );
    for( double reference :
         { 0.0, -1.0, 5.6, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        Settings invalid;
        invalid.m_vref = reference;
        REQUIRE_FALSE( AD7689::validSettings( invalid ) );
    }
    for( unsigned clock : { 0U, 99999U, 8000001U } )
    {
        Settings invalid;
        invalid.m_baud = clock;
        REQUIRE_FALSE( AD7689::validSettings( invalid ) );
    }
    Settings tooShort;
    tooShort.m_spiSetup_ns = 29;
    REQUIRE_FALSE( AD7689::validSettings( tooShort ) );
    tooShort                = {};
    tooShort.m_quietTime_ns = 19;
    REQUIRE_FALSE( AD7689::validSettings( tooShort ) );
    settings.m_baud              = 4000000;
    settings.m_conversionTime_ns = 4000;
    REQUIRE( AD7689::scanTime_ns( settings, 8 ) == Approx( 80000 ) );
    REQUIRE( AD7689::validRate( settings, 8, 8000, 20000 ) );
    settings.m_conversionTime_ns = 3199;
    REQUIRE_FALSE( AD7689::validSettings( settings ) );
}

/** Verify startup readback and two-conversion look-ahead for repeated ordered frames.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 real driver preserves channel and frame association", "[ad7689Ctrl][driver]" )
{
    for( const std::vector<unsigned> &channels :
         { std::vector<unsigned>{ 1 }, { 0, 1 }, { 7, 2, 4 }, { 7, 0, 4, 1, 6, 2, 5, 3 } } )
    {
        hardware = {};
        AD7689 adc;
        REQUIRE( adc.connect( Settings{} ) == 0 );
        REQUIRE( adc.prepare( channels ) == 0 );
        REQUIRE( adc.ready() );
        REQUIRE( adc.statistics().m_discarded == 12 );
        const unsigned warmup = hardware.m_pulses;
        REQUIRE( std::find( hardware.m_lengths.begin(), hardware.m_lengths.end(), 4 ) != hardware.m_lengths.end() );
        for( unsigned frame = 0; frame < 5; ++frame )
        {
            hardware.m_time += 100000;
            timespec                timestamp{};
            std::array<uint16_t, 8> values{};
            const unsigned          firstPulse = hardware.m_pulses;
            REQUIRE( adc.readScan( values.data(), channels.size(), timestamp ) == 0 );
            REQUIRE( timestamp.tv_sec >= 1700000000 );
            const uint64_t scanStart = ( timestamp.tv_sec - 1700000000ULL ) * 1000000000ULL + timestamp.tv_nsec;
            REQUIRE( scanStart <= hardware.m_edges[firstPulse] );
            REQUIRE( hardware.m_pulses - firstPulse == channels.size() );
            for( size_t i = 0; i < channels.size(); ++i )
            {
                REQUIRE( hardware.m_channels[firstPulse + i] == channels[i] );
                REQUIRE( values[i] == channels[i] * 8192 + ( ( firstPulse + i + 1 ) & 0xfff ) );
            }
        }
        REQUIRE( hardware.m_pulses == warmup + 5 * channels.size() );
        REQUIRE( adc.disconnect() == 0 );
        REQUIRE_FALSE( adc.ready() );
        REQUIRE( hardware.m_closes == 1 );
        REQUIRE( hardware.m_releases == 1 );
        REQUIRE( hardware.m_chips == 0 );
        REQUIRE( hardware.m_allocations == 0 );
    }
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    AD7689Lib::AD7689::connect;
    AD7689Lib::AD7689::prepare;
    AD7689Lib::AD7689::readScan;
    AD7689Lib::AD7689::disconnect;
#endif
    // clang-format on
}

/** Roll back every partial connection stage and leave no real resource operation available.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 partial connection failures release resources", "[ad7689Ctrl][driver]" )
{
    for( unsigned failure = 1; failure <= 17; ++failure )
    {
        hardware          = {};
        hardware.m_failAt = failure;
        {
            AD7689 adc;
            adc.connect( Settings{} );
        }
        REQUIRE_FALSE( hardware.m_requested );
        REQUIRE( hardware.m_chips == 0 );
        REQUIRE( hardware.m_allocations == 0 );
    }
    hardware        = {};
    hardware.m_busy = true;
    AD7689 adc;
    REQUIRE( adc.connect( Settings{} ) == -1 );
    REQUIRE( adc.error().find( "GPIO8" ) != std::string::npos );
    REQUIRE_FALSE( hardware.m_requested );
}

/** Reject corrupted CFG, short/failed reads, and uncertain GPIO progress without committing data.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 readout failures never commit a partial frame", "[ad7689Ctrl][driver]" )
{
    for( int fault = 0; fault < 5; ++fault )
    {
        hardware = {};
        AD7689 adc;
        REQUIRE( adc.connect( Settings{} ) == 0 );
        REQUIRE( adc.prepare( { 0, 7 } ) == 0 );
        std::array<uint16_t, 2> values{ 123, 456 };
        timespec                timestamp{};
        if( fault == 0 )
            hardware.m_shortRead = true;
        if( fault == 1 )
            hardware.m_highFailure = true;
        if( fault == 2 )
            hardware.m_lowFailure = true;
        if( fault == 3 )
            hardware.m_clockFailure = true;
        if( fault == 4 )
            hardware.m_failAt = hardware.m_calls + 3;
        REQUIRE( adc.readScan( values.data(), 2, timestamp ) == -1 );
        REQUIRE( values == std::array<uint16_t, 2>{ 123, 456 } );
        REQUIRE_FALSE( adc.ready() );
        hardware.m_shortRead = hardware.m_highFailure = hardware.m_lowFailure = hardware.m_clockFailure = false;
        REQUIRE( adc.prepare( { 7, 0 } ) == 0 );
    }
    hardware = {};
    AD7689 adc;
    REQUIRE( adc.connect( Settings{} ) == 0 );
    hardware.m_corrupt = true;
    REQUIRE( adc.prepare( { 0 } ) == -1 );
    REQUIRE( adc.error().find( "readback" ) != std::string::npos );
}

/** Preserve legal constant rail samples and quantify the configured four-microsecond timing.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 rails and shortened timing remain explicit", "[ad7689Ctrl][driver]" )
{
    hardware = {};
    Settings settings;
    settings.m_baud              = 8000000;
    settings.m_conversionTime_ns = 4000;
    hardware.m_highHold          = 4000;
    AD7689 adc;
    REQUIRE( adc.connect( settings ) == 0 );
    REQUIRE( adc.prepare( { 0 } ) == 0 );
    for( size_t i = 1; i < hardware.m_edges.size(); ++i )
        REQUIRE( hardware.m_edges[i] - hardware.m_edges[i - 1] >= 20000 );
    hardware.m_constant = true;
    for( uint16_t code : { static_cast<uint16_t>( 0 ), static_cast<uint16_t>( 65535 ) } )
    {
        hardware.m_constantCode = code;
        uint16_t result         = 0;
        timespec timestamp{};
        REQUIRE( adc.readScan( &result, 1, timestamp ) == 0 );
        REQUIRE( result == code );
    }
}

/** Reject clock regression and hold CNV high safely if the post-edge clock read fails.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 timing failures invalidate the scan and permit safe cleanup", "[ad7689Ctrl][driver]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    AD7689Lib::AD7689::readScan;
    AD7689Lib::AD7689::disconnect;
#endif
    // clang-format on
    for( bool afterHigh : { false, true } )
    {
        hardware = {};
        AD7689 adc;
        REQUIRE( adc.connect( Settings{} ) == 0 );
        REQUIRE( adc.prepare( { 0 } ) == 0 );
        hardware.m_backwards      = !afterHigh;
        hardware.m_clockAfterHigh = afterHigh;
        uint16_t value            = 123;
        timespec timestamp{};
        REQUIRE( adc.readScan( &value, 1, timestamp ) == -1 );
        REQUIRE( value == 123 );
        REQUIRE_FALSE( adc.ready() );
        adc.disconnect();
        REQUIRE_FALSE( hardware.m_high );
        REQUIRE_FALSE( hardware.m_requested );
    }
}

/** A CFG[13]=0 serial word must retain a selected nonzero input rather than selecting IN0.
 * \ingroup ad7689Ctrl_unit_test
 */
TEST_CASE( "AD7689 zero configuration write retains the selected input", "[ad7689Ctrl][driver]" )
{
    // clang-format off
#ifdef AD7689CTRL_TEST_DOXYGEN_REF
    AD7689Lib::AD7689::exchange;
    AD7689Lib::AD7689::prepare;
#endif
    // clang-format on
    hardware = {};
    AD7689 adc;
    REQUIRE( adc.connect( Settings{} ) == 0 );
    REQUIRE( adc.prepare( { 7 } ) == 0 );
    for( unsigned i = 0; i < 4; ++i )
    {
        uint16_t sample = 0;
        REQUIRE( adc.exchange( 0, sample, false ) == 0 );
        REQUIRE( hardware.m_written.back() == 0 );
        REQUIRE( hardware.m_channels.back() == 7 );
        REQUIRE( sample >= 7 * 8192 );
    }
    uint16_t sample = 0;
    timespec timestamp{};
    REQUIRE( adc.readScan( &sample, 1, timestamp ) == 0 );
    REQUIRE( sample >= 7 * 8192 );
}

} // namespace ad7689CtrlTest
} // namespace libXWCTest
