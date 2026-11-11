/** \file AD7689.cpp
 * \brief Linux spidev/libgpiod v2 AD7689 conversions with explicit CNV timing.
 * \ingroup ad7689Ctrl_files
 */
#include "AD7689.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <gpiod.h>
#include <linux/spi/spidev.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace AD7689Lib
{

AD7689::AD7689() = default;
AD7689::~AD7689() noexcept
{
    disconnect();
}

int AD7689::fail( const char *operation )
{
    ++m_statistics.m_errors;
    m_ready = m_activeValid = m_pendingValid = false;
    m_error                                  = operation;
    if( errno )
        m_error += std::string( ": " ) + std::strerror( errno );
    return -1;
}

int AD7689::now( uint64_t &ns )
{
    timespec ts{};
    if( clock_gettime( CLOCK_MONOTONIC, &ts ) < 0 )
        return fail( "clock_gettime(CLOCK_MONOTONIC)" );
    ns = static_cast<uint64_t>( ts.tv_sec ) * 1000000000ULL + ts.tv_nsec;
    if( ns < m_lastClock_ns )
    {
        errno = EIO;
        return fail( "monotonic clock moved backwards" );
    }
    m_lastClock_ns = ns;
    return 0;
}

int AD7689::waitUntil( uint64_t deadline )
{
    uint64_t previous = 0, current = 0;
    for( unsigned spins = 0; spins < 100000000; ++spins )
    {
        if( now( current ) < 0 )
            return -1;
        if( current < previous )
        {
            errno = EIO;
            return fail( "monotonic clock moved backwards" );
        }
        if( current >= deadline )
            return 0;
        previous = current;
    }
    errno = ETIMEDOUT;
    return fail( "ADC timing wait did not advance" );
}

int AD7689::connect( const Settings &settings )
{
    if( disconnect() < 0 )
        return -1;
    errno = 0;
    if( !validSettings( settings ) )
        return fail( "invalid ADC settings" );
    m_settings = settings;
    m_chip     = gpiod_chip_open( m_settings.m_chip.c_str() );
    if( !m_chip )
        return fail( "open CNV GPIO chip" );

    auto *lineSettings  = gpiod_line_settings_new();
    auto *lineConfig    = gpiod_line_config_new();
    auto *requestConfig = gpiod_request_config_new();
    bool  valid         = lineSettings && lineConfig && requestConfig;
    if( valid )
    {
        gpiod_line_settings_set_active_low( lineSettings, false );
        valid = gpiod_line_settings_set_direction( lineSettings, GPIOD_LINE_DIRECTION_OUTPUT ) == 0 &&
                gpiod_line_settings_set_drive( lineSettings, GPIOD_LINE_DRIVE_PUSH_PULL ) == 0 &&
                gpiod_line_settings_set_output_value( lineSettings, GPIOD_LINE_VALUE_INACTIVE ) == 0 &&
                gpiod_line_config_add_line_settings( lineConfig, &m_settings.m_line, 1, lineSettings ) == 0;
        gpiod_request_config_set_consumer( requestConfig, "ad7689Ctrl-cnv" );
        if( valid )
            m_request = gpiod_chip_request_lines( m_chip, requestConfig, lineConfig );
    }
    if( requestConfig )
        gpiod_request_config_free( requestConfig );
    if( lineConfig )
        gpiod_line_config_free( lineConfig );
    if( lineSettings )
        gpiod_line_settings_free( lineSettings );
    if( !valid || !m_request )
    {
        fail( "claim CNV output (GPIO8 must be released from spi0 CS0)" );
        disconnect();
        return -1;
    }

    const std::string path =
        "/dev/spidev" + std::to_string( settings.m_bus ) + "." + std::to_string( settings.m_device );
    m_fd          = open( path.c_str(), O_RDWR | O_CLOEXEC );
    uint32_t mode = SPI_MODE_0, actual = 0;
    uint8_t  bits = 8, order = 0;
    if( m_fd < 0 || ioctl( m_fd, SPI_IOC_WR_MODE32, &mode ) < 0 || ioctl( m_fd, SPI_IOC_WR_BITS_PER_WORD, &bits ) < 0 ||
        ioctl( m_fd, SPI_IOC_WR_LSB_FIRST, &order ) < 0 ||
        ioctl( m_fd, SPI_IOC_WR_MAX_SPEED_HZ, &m_settings.m_baud ) < 0 ||
        ioctl( m_fd, SPI_IOC_RD_MODE32, &actual ) < 0 || actual != mode ||
        ioctl( m_fd, SPI_IOC_RD_BITS_PER_WORD, &bits ) < 0 || bits != 8 ||
        ioctl( m_fd, SPI_IOC_RD_LSB_FIRST, &order ) < 0 || order != 0 )
    {
        fail( "open/configure SPI mode 0, eight bits, MSB first" );
        disconnect();
        return -1;
    }
    if( now( m_lastSerial_ns ) < 0 )
    {
        disconnect();
        return -1;
    }
    m_lastHigh_ns = 0;
    return 0;
}

int AD7689::cleanupHold() const noexcept
{
    timespec hold{ static_cast<time_t>( m_settings.m_conversionTime_ns / 1000000000ULL ),
                   static_cast<long>( m_settings.m_conversionTime_ns % 1000000000ULL ) };
    while( nanosleep( &hold, &hold ) < 0 )
        if( errno != EINTR )
            return -1;
    return 0;
}

int AD7689::disconnect() noexcept
{
    int result = 0;
    m_ready = m_activeValid = m_pendingValid = false;
    if( m_request )
    {
        if( m_cnvHigh )
        {
            // Also covers uncertain GPIO-high progress and failure of the post-high clock query.
            if( cleanupHold() < 0 )
                result = -1;
        }
        if( gpiod_line_request_set_value( m_request, m_settings.m_line, GPIOD_LINE_VALUE_INACTIVE ) < 0 )
            result = -1;
    }
    if( m_fd >= 0 && close( m_fd ) < 0 )
        result = -1;
    m_fd = -1;
    if( m_request )
        gpiod_line_request_release( m_request );
    if( m_chip )
        gpiod_chip_close( m_chip );
    m_request = nullptr;
    m_chip    = nullptr;
    m_cnvHigh = false;
    return result;
}

int AD7689::exchange( uint16_t cfg, uint16_t &value, bool startup )
{
    uint64_t time = 0;
    if( now( time ) < 0 )
        return -1;
    const uint64_t cycle    = startup ? 20000 : 5000;
    const uint64_t acq      = startup ? 16800 : 1800;
    const uint64_t earliest = std::max( m_lastSerial_ns + m_settings.m_quietTime_ns,
                                        m_lastHigh_ns ? m_lastHigh_ns + std::max( cycle, 3200 + acq ) : 0 );
    if( waitUntil( earliest ) < 0 )
        return -1;
    const uint16_t expected = m_activeCfg;
    const bool     known    = m_activeValid;
    // A failed ioctl can leave hardware progress uncertain. Cleanup must hold before any low command.
    m_cnvHigh = true;
    if( gpiod_line_request_set_value( m_request, m_settings.m_line, GPIOD_LINE_VALUE_ACTIVE ) < 0 )
        return fail( "CNV high" );
    ++m_statistics.m_conversions;
    // Cleanup uses an independent relative hold if this post-high clock read fails.
    m_lastHigh_ns = time;
    if( now( m_lastHigh_ns ) < 0 || waitUntil( m_lastHigh_ns + m_settings.m_conversionTime_ns ) < 0 )
        return -1;
    if( gpiod_line_request_set_value( m_request, m_settings.m_line, GPIOD_LINE_VALUE_INACTIVE ) < 0 )
        return fail( "CNV low" );
    m_cnvHigh = false;
    if( now( time ) < 0 || waitUntil( time + m_settings.m_spiSetup_ns ) < 0 )
        return -1;

    const uint16_t   txWord = static_cast<uint16_t>( cfg << 2 );
    uint8_t          tx[4]{ static_cast<uint8_t>( txWord >> 8 ), static_cast<uint8_t>( txWord ), 0, 0 };
    uint8_t          rx[4]{};
    const unsigned   count = startup || ( known && !( expected & 1U ) ) ? 4 : 2;
    spi_ioc_transfer transfer{};
    transfer.tx_buf        = reinterpret_cast<uintptr_t>( tx );
    transfer.rx_buf        = reinterpret_cast<uintptr_t>( rx );
    transfer.len           = count;
    transfer.speed_hz      = m_settings.m_baud;
    transfer.bits_per_word = 8;
    const int transferred  = ioctl( m_fd, SPI_IOC_MESSAGE( 1 ), &transfer );
    // Even an error or short count can follow activity on the wire; never reuse its old quiet deadline.
    if( now( m_lastSerial_ns ) < 0 )
        return -1;
    if( transferred != static_cast<int>( count ) )
        return fail( "SPI conversion transfer incomplete" );
    if( known && !( expected & 1U ) && ( ( decode( rx + 2 ) >> 2 ) & 0x1fffU ) != ( expected & 0x1fffU ) )
    {
        errno = EIO;
        return fail( "ADC configuration readback mismatch" );
    }
    value = decode( rx );
    // The pending word was applied at this conversion's EOC, before this SPI write.
    if( m_pendingValid )
    {
        m_activeCfg   = m_pendingCfg;
        m_activeValid = true;
    }
    if( cfg & 0x2000U )
    {
        m_pendingCfg   = cfg;
        m_pendingValid = true;
    }
    else
        m_pendingValid = false;
    return 0;
}

int AD7689::prepare( const std::vector<unsigned> &channels )
{
    m_ready = m_activeValid = m_pendingValid = false;
    errno                                    = 0;
    if( m_fd < 0 || !m_request || !validChannels( channels ) )
        return fail( "invalid or disconnected scan" );
    if( m_cnvHigh )
    {
        if( cleanupHold() < 0 )
            return fail( "conversion hold before reinitialization" );
        if( gpiod_line_request_set_value( m_request, m_settings.m_line, GPIOD_LINE_VALUE_INACTIVE ) < 0 )
            return fail( "CNV low before reinitialization" );
        m_cnvHigh = false;
    }
    // Reinitialization follows potentially uncertain progress; start a fresh serial-quiet guard.
    if( now( m_lastSerial_ns ) < 0 )
        return -1;
    m_count = channels.size();
    std::copy( channels.begin(), channels.end(), m_channels.begin() );
    uint16_t       unused = 0;
    const uint16_t first  = configuration( channels[0] );
    // Repeated explicit writes establish both stages even from undefined power-up CFG/RB/BW.
    for( int i = 0; i < 4; ++i )
    {
        if( exchange( first, unused, true ) < 0 )
            return -1;
        ++m_statistics.m_discarded;
    }
    const uint16_t checked = configuration( channels[0], true );
    for( int i = 0; i < 4; ++i )
    {
        if( exchange( checked, unused, true ) < 0 )
            return -1;
        ++m_statistics.m_discarded;
    }
    // Restore normal framing; response length still follows each historical RB bit.
    for( int i = 0; i < 3; ++i )
    {
        if( exchange( first, unused, true ) < 0 )
            return -1;
        ++m_statistics.m_discarded;
    }
    if( exchange( configuration( channels[m_count == 1 ? 0 : 1] ), unused, true ) < 0 )
        return -1;
    ++m_statistics.m_discarded;
    m_ready = m_activeValid && m_pendingValid && m_activeCfg == first;
    return m_ready ? 0 : fail( "failed to prime ADC channel history" );
}

int AD7689::readScan( uint16_t *dest, size_t count, timespec &timestamp )
{
    errno = 0;
    if( !m_ready || !dest || count != m_count )
        return fail( "unprepared ADC scan" );
    if( clock_gettime( CLOCK_REALTIME, &timestamp ) < 0 )
        return fail( "scan-start realtime timestamp" );
    std::array<uint16_t, 8> staging{};
    for( size_t i = 0; i < count; ++i )
    {
        if( m_activeCfg != configuration( m_channels[i] ) )
            return fail( "ADC channel association mismatch" );
        if( exchange( configuration( m_channels[( i + 2 ) % count] ), staging[i], false ) < 0 )
            return -1;
    }
    std::copy_n( staging.begin(), count, dest );
    return 0;
}

} // namespace AD7689Lib
