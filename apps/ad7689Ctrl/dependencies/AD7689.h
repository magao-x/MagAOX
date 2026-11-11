/** \file AD7689.h
 * \brief AD7689 manual-CNV SPI acquisition and portable configuration checks.
 * \ingroup ad7689Ctrl_files
 */
#ifndef ad7689Ctrl_AD7689_h
#define ad7689Ctrl_AD7689_h

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

struct gpiod_chip;
struct gpiod_line_request;

namespace AD7689Lib
{

constexpr uint16_t c_cfgUpdate        = 1U << 13; ///< CFG writes update the configuration only when this bit is set.
constexpr uint16_t c_cfgUnipolarGnd   = 7U << 10; ///< INCC=111: selected input relative to GND.
constexpr uint16_t c_cfgFullBandwidth = 1U << 6;  ///< BW=1: full bandwidth, with 1.8 us minimum acquisition.
constexpr uint16_t c_cfgExternalRef   = 6U << 3;  ///< REF=110: external REF, internal buffer/reference disabled.
constexpr uint16_t c_cfgNoReadback    = 1U;       ///< RB=1: normal 16-clock result without appended CFG.

/// Startup-only transport, timing, and ADC-pin calibration settings.
struct Settings
{
    unsigned    m_bus{ 0 };                   ///< Linux SPI controller index.
    unsigned    m_device{ 0 };                ///< Spidev endpoint, independent of the analog input number.
    unsigned    m_baud{ 1000000 };            ///< Requested serial clock; qualify the actual clock on the board.
    std::string m_chip{ "/dev/gpiochip0" };   ///< GPIO chip containing the physical CNV line.
    unsigned    m_line{ 8 };                  ///< Chip-local offset of the independent CNV output.
    uint64_t    m_conversionTime_ns{ 10000 }; ///< High hold measured after the successful GPIO write returns.
    uint64_t    m_spiSetup_ns{ 1000 };        ///< Minimum low-to-first-SPI setup guard.
    uint64_t    m_quietTime_ns{ 1000 };       ///< Minimum serial quiet interval before a rising CNV edge.
    double      m_vref{ 3.32 };               ///< Measured external REF voltage; does not scale raw stream data.
};

/// Single-acquisition-owner counters; the application publishes synchronized snapshots.
struct Statistics
{
    uint64_t m_conversions{ 0 }; ///< Successful CNV-high commands, including warmup and failed scans.
    uint64_t m_discarded{ 0 };   ///< Intentionally discarded initialization/verification conversions.
    uint64_t m_errors{ 0 };      ///< Failed transport, clock, state, or readback operations.
};

/// Own one SPI endpoint and persistent GPIO request; never share this object across acquisition threads.
/** RAC call k reads conversion k, but its CFG write controls conversion k+2.
 * prepare() establishes the two-entry history, then readScan() performs N useful
 * conversions for N ordered inputs without a recurring dummy conversion.
 */
class AD7689
{
  private:
    Settings                m_settings;              ///< Immutable settings while connected.
    Statistics              m_statistics;            ///< Lifetime hardware operation counters.
    int                     m_fd{ -1 };              ///< Owned SPI descriptor; negative when disconnected.
    gpiod_chip             *m_chip{ nullptr };       ///< Owned GPIO chip handle.
    gpiod_line_request     *m_request{ nullptr };    ///< Owned persistent active-high CNV output request.
    std::array<unsigned, 8> m_channels{};            ///< Physical analog inputs in stream order.
    size_t                  m_count{ 0 };            ///< Number of initialized scan inputs.
    uint16_t                m_activeCfg{ 0 };        ///< CFG for the acquisition preceding the next CNV edge.
    uint16_t                m_pendingCfg{ 0 };       ///< CFG to apply at the next EOC.
    bool                    m_activeValid{ false };  ///< Whether the acquired channel/configuration is known.
    bool                    m_pendingValid{ false }; ///< Whether a complete known CFG is pending at EOC.
    bool                    m_ready{ false }; ///< Both pipeline stages match the beginning of an initialized scan.
    bool m_cnvHigh{ false }; ///< A high command may have taken effect; cleanup must complete a hold before driving low.
    uint64_t    m_lastHigh_ns{ 0 };   ///< Conservative monotonic reference after a GPIO-high completion.
    uint64_t    m_lastSerial_ns{ 0 }; ///< Monotonic reference after the most recent SPI operation.
    uint64_t    m_lastClock_ns{ 0 };  ///< Last observed clock value, used to reject cross-operation regressions.
    std::string m_error;              ///< Most recent failure context for application logging.

    /// Set an error, invalidate the conversion history, and return -1.
    int fail( const char *operation /**< [in] failed operation context */ );

    /// Read monotonic nanoseconds, reporting a clock error.
    int now( uint64_t &ns /**< [out] monotonic time */ );

    /// Wait until a monotonic deadline, rejecting backwards or stalled clocks.
    int waitUntil( uint64_t deadline /**< [in] minimum allowed completion time */ );

    /// Hold for a full conversion independently of clock-query failures during recovery/cleanup.
    int cleanupHold() const noexcept;

    /// Perform one controlled conversion and read its result/historical CFG.
    int exchange( uint16_t  cfg /**< [in] future configuration; zero retains configuration */,
                  uint16_t &value /**< [out] current conversion's unsigned result */,
                  bool      startup /**< [in] conservative unknown-state timing/framing */ );

  public:
    /// Construct a disconnected device without touching hardware.
    AD7689();

    /// Release all resources without throwing.
    ~AD7689() noexcept;

    /// Disable copying of owned descriptors and conversion history.
    AD7689( const AD7689 &other /**< [in] noncopyable device */ ) = delete;

    /// Disable copy assignment of owned descriptors and conversion history.
    AD7689 &operator=( const AD7689 &other /**< [in] noncopyable device */ ) = delete;

    /// Validate startup settings without hardware access.
    static bool validSettings( const Settings &settings /**< [in] proposed transport and timing */ );

    /// Validate an ordered, unique, nonempty physical input list.
    static bool validChannels( const std::vector<unsigned> &channels /**< [in] physical input list */ );

    /// Calculate a conservative configured scan budget, excluding syscall/framework overhead.
    static double scanTime_ns( const Settings &settings /**< [in] proposed timing */,
                               size_t          count /**< [in] number of useful inputs */ );

    /// Check a positive finite requested frame rate against its configured timing budget.
    static bool validRate( const Settings &settings /**< [in] proposed timing */,
                           size_t          count /**< [in] number of enabled inputs */,
                           double          fps /**< [in] requested per-input/frame rate */,
                           double          margin_ns = 0 /**< [in] reserved non-ADC time */ );

    /// Encode the supported full-bandwidth, GND-unipolar, external-direct-reference CFG.
    /** \throws std::out_of_range for an input outside 0..7. */
    static uint16_t configuration( unsigned channel /**< [in] validated physical input 0..7 */,
                                   bool     readback = false /**< [in] append CFG to conversion data */ );

    /// Decode a complete MSB-first unsigned sample without truncation.
    static uint16_t decode( const uint8_t *bytes /**< [in] at least two received bytes */ );

    /// Convert ADC-pin counts to volts with the 2^16 transfer-function denominator.
    static double volts( uint16_t code /**< [in] unsigned raw count */,
                         double   vref /**< [in] measured external REF in volts */ );

    /// Claim CNV and open/configure SPI_NO_CS; unwind partial failures and return -1.
    int connect( const Settings &settings /**< [in] validated immutable connection settings */ );

    /// Safely close SPI/release CNV and invalidate the pipeline; return -1 on cleanup failure.
    int disconnect() noexcept;

    /// Warm up, verify CFG, and initialize a complete ordered scan; return -1 on error.
    int prepare( const std::vector<unsigned> &channels /**< [in] ordered enabled inputs */ );

    /// Read one complete fresh scan, committing output only on success; return -1 on error.
    int readScan( uint16_t *dest /**< [out] caller storage for count samples */,
                  size_t    count /**< [in] must match the prepared input list */,
                  timespec &timestamp /**< [out] software scan start before the first useful high request */ );

    /// Report whether the connection has an initialized scan history.
    bool ready() const;

    /// Return acquisition-owner statistics; callers must synchronize cross-thread snapshots.
    const Statistics &statistics() const;

    /// Return the most recent error text for a failed operation.
    const std::string &error() const;
};

inline bool AD7689::validSettings( const Settings &s )
{
    return s.m_bus <= 255 && s.m_device <= 255 && s.m_baud >= 100000 && s.m_baud <= 8000000 && !s.m_chip.empty() &&
           s.m_line < 1024 && s.m_conversionTime_ns >= 3200 && s.m_conversionTime_ns <= 1000000000 &&
           s.m_spiSetup_ns >= 30 && s.m_spiSetup_ns <= 1000000000 && s.m_quietTime_ns >= 20 &&
           s.m_quietTime_ns <= 1000000000 && std::isfinite( s.m_vref ) && s.m_vref >= 0.5 && s.m_vref <= 5.5;
}

inline bool AD7689::validChannels( const std::vector<unsigned> &channels )
{
    if( channels.empty() || channels.size() > 8 )
        return false;
    unsigned mask = 0;
    for( unsigned channel : channels )
    {
        if( channel > 7 || ( mask & ( 1U << channel ) ) )
            return false;
        mask |= 1U << channel;
    }
    return true;
}

inline double AD7689::scanTime_ns( const Settings &s, size_t count )
{
    if( !validSettings( s ) || count < 1 || count > 8 )
        return INFINITY;
    const double cycle =
        static_cast<double>( s.m_conversionTime_ns + s.m_spiSetup_ns + s.m_quietTime_ns ) + 16e9 / s.m_baud;
    return count * std::max( cycle, 5000.0 );
}

inline bool AD7689::validRate( const Settings &s, size_t count, double fps, double margin_ns )
{
    return std::isfinite( fps ) && fps > 0 && fps <= 10000 && 1e9 / fps <= std::numeric_limits<float>::max() &&
           std::isfinite( margin_ns ) && margin_ns >= 0 && scanTime_ns( s, count ) + margin_ns < 1e9 / fps;
}

inline uint16_t AD7689::configuration( unsigned channel, bool readback )
{
    if( channel > 7 )
        throw std::out_of_range( "AD7689 input must be in 0..7" );
    return c_cfgUpdate | c_cfgUnipolarGnd | ( channel << 7 ) | c_cfgFullBandwidth | c_cfgExternalRef |
           ( readback ? 0 : c_cfgNoReadback ); // SEQ=00: explicit input selection.
}

inline uint16_t AD7689::decode( const uint8_t *bytes )
{
    return static_cast<uint16_t>( ( static_cast<uint16_t>( bytes[0] ) << 8 ) | bytes[1] );
}

inline double AD7689::volts( uint16_t code, double vref )
{
    return static_cast<double>( code ) * vref / 65536.0;
}

inline bool AD7689::ready() const
{
    return m_ready;
}
inline const Statistics &AD7689::statistics() const
{
    return m_statistics;
}
inline const std::string &AD7689::error() const
{
    return m_error;
}

} // namespace AD7689Lib
#endif // ad7689Ctrl_AD7689_h
