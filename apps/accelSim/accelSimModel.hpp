/** \file accelSimModel.hpp
 * \brief Streaming accelerometer physics and acquisition timing primitives.
 * \ingroup accelSim_files
 */
#ifndef accelSimModel_hpp
#define accelSimModel_hpp

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace MagAOX
{
namespace app
{
namespace accelSimDetail
{

constexpr double   pi          = 3.14159265358979323846;
constexpr double   arcsecToRad = pi / ( 180.0 * 3600.0 );
constexpr uint64_t maxCatchUp  = 8; ///< Maximum skipped ticks advanced before a stationary restart.

/// Parameters for one independent pair of pointing oscillators.
struct modeSpec
{
    double m_frequency; ///< Mode frequency in Hz.
    double m_accelRms;  ///< Requested per-axis acceleration RMS in m/s^2.
    double m_r;         ///< Pole radius, specified directly as in the reference notebook.
    double m_couplingX; ///< Fraction of X pointing sensed by the WFS.
    double m_couplingY; ///< Fraction of Y pointing sensed by the WFS.
};

/// Configurable physical model and empirical ADC calibration.
struct parameters
{
    double   m_fps{ 6000 };              ///< Nominal discrete sample rate in Hz.
    uint64_t m_seed{ 42 };               ///< Seed for initialization, innovations, and sensor noise.
    uint64_t m_modeSeed{ 123 };          ///< Independent seed for the fixed high-frequency population.
    double   m_leverArm{ 10 };           ///< Effective lever arm in metres.
    double   m_noisePSD{ 1e-8 };         ///< One-sided acceleration noise density.
    double   m_zero{ 2048 };             ///< ADC zero point in counts.
    double   m_voltsPerBit{ 0.0083799 }; ///< Empirical volts-per-count calibration.
    double   m_sensitivity{ 1.02 };      ///< Empirical acceleration sensitivity.
};

/// Stable stationary scaling and a two-sample oscillator history.
struct oscillator
{
    double m_phi1{ 0 };      ///< First AR coefficient.
    double m_phi2{ 0 };      ///< Second AR coefficient.
    double m_sigma{ 0 };     ///< Innovation standard deviation.
    double m_rho{ 0 };       ///< Stationary correlation of adjacent samples.
    double m_eta{ 1 };       ///< Stable square root of one minus squared correlation.
    double m_rms{ 0 };       ///< Requested stationary pointing RMS in arcsec.
    double m_previous1{ 0 }; ///< Most recent pointing sample.
    double m_previous2{ 0 }; ///< Previous pointing sample.

    /// Calculate coefficients and stationary innovation scaling.
    void configure( double frequency /**< [in] frequency in Hz */,
                    double r /**< [in] pole radius */,
                    double fps /**< [in] sampling frequency */,
                    double rms /**< [in] stationary pointing RMS */ );

    /// Draw a stationary joint history from two independent standard normals.
    void initialize( double z1 /**< [in] first standard normal */, double z2 /**< [in] independent standard normal */ );

    /// Advance one sample using an independent standard normal innovation.
    double step( double z /**< [in] standard normal innovation */ );
};

/// Physical and digitized quantities for one simultaneous X/Y sample.
struct sample
{
    std::array<double, 2>   m_pointing{};     ///< Physical pointing in arcsec.
    std::array<double, 2>   m_wfsPointing{};  ///< Coupled pointing in arcsec, without sensor noise.
    std::array<double, 2>   m_acceleration{}; ///< Acceleration including sensor noise in m/s^2.
    std::array<uint16_t, 2> m_adc{};          ///< Clipped raw X/Y ADC counts.
};

/// One resolved mode with independent X/Y histories.
struct mode
{
    modeSpec                  m_spec;           ///< Fixed reference-model parameters.
    double                    m_accelPerArcsec; ///< Negative frequency-squared acceleration conversion.
    std::array<oscillator, 2> m_axes;           ///< Independent X and Y oscillators.
};

/// Construct the notebook's 14 prominent and 200 weak modes once.
std::vector<modeSpec> referenceModes( uint64_t seed /**< [in] mode-parameter seed */ );

/// Convert one arcsecond of pointing to acceleration for a mode.
double accelerationFactor( double frequency /**< [in] frequency in Hz */,
                           double leverArm /**< [in] effective length in metres */ );

/// Round and clamp the calibrated ADC value before integer conversion.
uint16_t adcCounts( double            acceleration /**< [in] acceleration including noise */,
                    const parameters &config /**< [in] empirical calibration */,
                    uint64_t         &low /**< [in,out] lower clamp count */,
                    uint64_t         &high /**< [in,out] upper clamp count */ );

/// Sample-by-sample model, independent of the MagAO-X runtime.
class model
{
  public:
    parameters              m_config;          ///< Active configuration; only the acquisition thread changes it.
    std::vector<mode>       m_modes;           ///< Resolved population; storage is fixed during stepping.
    sample                  m_sample;          ///< Latest generated physical and ADC values.
    std::array<uint64_t, 2> m_clipLow{};       ///< Actual low-rail clamps, including skipped numerical ticks.
    std::array<uint64_t, 2> m_clipHigh{};      ///< Actual high-rail clamps, including skipped numerical ticks.
    uint64_t                m_ticks{ 0 };      ///< Generated numerical ticks, independent of publications.
    double                  m_noiseSigma{ 0 }; ///< Noise standard deviation for the nominal sample rate.

    /// Initialize configuration, resolved modes, random generator, and stationary histories.
    void configure( const parameters            &config /**< [in] physical configuration */,
                    const std::vector<modeSpec> &specs /**< [in] resolved vibration modes */ );

    /// Check whether a nominal rate supports all retained modes.
    bool validRate( double fps /**< [in] candidate rate */ ) const;

    /// Apply a new nominal rate and draw new stationary histories without reseeding.
    void setRate( double fps /**< [in] validated nominal rate */ );

    /// Redraw histories in stationarity, retaining the population and RNG progression.
    void restart();

    /// Generate one pair without allocating or accessing external interfaces.
    const sample &step();

    /// Advance skipped ticks, or restart for a gap exceeding maxCatchUp, then generate one current pair.
    bool advance( uint64_t skipped /**< [in] skipped nominal publication opportunities */ );

  private:
    std::mt19937_64                  m_rng;            ///< Reproducible generator owned by the model.
    std::normal_distribution<double> m_normal{ 0, 1 }; ///< Independent standard-normal draws.
};

/// Absolute monotonic pacing with fractional-nanosecond period accumulation.
struct schedule
{
    int64_t  m_epoch{ 0 };  ///< Monotonic epoch in nanoseconds.
    uint64_t m_next{ 1 };   ///< Next nominal tick relative to epoch.
    double   m_fps{ 6000 }; ///< Nominal sampling frequency.

    /// Establish a new pacing epoch.
    void reset( int64_t now /**< [in] monotonic nanoseconds */, double fps /**< [in] nominal sampling frequency */ );

    /// Return the next absolute deadline, rounding only the accumulated interval.
    int64_t deadline() const;

    /// Consume the latest due tick and return the number of older ticks skipped.
    uint64_t consume( int64_t now /**< [in] monotonic acquisition time, at or after deadline */ );
};

/// Tentative source notification; committed only after acquisition or explicit cancellation.
struct syncTick
{
    uint64_t m_producer{ 0 }, m_delta{ 0 };  ///< Producer counter and gap from the previous consumed notification.
    uint64_t m_sourceTick{ 0 }, m_tick{ 0 }; ///< Independent source position and latest nominal opportunity.
    uint64_t m_originTick{ 0 };              ///< Nominal position at the monotonic epoch.
    int64_t  m_epoch{ 0 };                   ///< Monotonic epoch retained across arrival jitter.
    uint64_t m_advance{ 0 }, m_skipped{ 0 }; ///< Model catch-up and newly omitted publications, respectively.
    bool m_newFrame{ false }, m_due{ false }, m_reset{ false }; ///< Unique counter, new opportunity, producer reset.
};

/// Reconcile elapsed nominal time and producer progress without counting the same gap twice.
struct syncClock
{
    uint64_t m_producer{ 0 }, m_sourceTick{ 0 };     ///< Last consumed producer counter and cumulative source progress.
    uint64_t m_nextAccounted{ 0 }, m_nextModel{ 0 }; ///< First unaccounted opportunity and ungenerated model tick.
    uint64_t m_originTick{ 0 };                      ///< Nominal position at the current timing epoch.
    int64_t  m_epoch{ 0 };                           ///< Monotonic epoch, unaffected by per-trigger jitter.
    bool     m_haveSource{ false }, m_haveTime{ false }; ///< Valid counter history and current-rate time anchor.

    /// Inspect a notification without modifying committed counters or simulation time.
    syncTick prepare( uint64_t producer /**< [in] current source counter */,
                      int64_t  now /**< [in] monotonic arrival nanoseconds */,
                      double   fps /**< [in] nominal model rate */ ) const;

    /// Commit acquisition or cancellation and return newly omitted publication opportunities.
    uint64_t finish( const syncTick &tick /**< [in] prepared notification */,
                     bool            acquired /**< [in] whether its current model sample was generated */ );

    /// Start a new rate epoch while retaining consumed source-counter history.
    void resetTime();
};

/// Indexed form of the unchanged mcp3208Ctrl timingDiag interface.
enum timingIndex
{
    readLatency,
    delay,
    delayTarget,
    delayApplied,
    delayModel,
    phaseError,
    lock,
    delayBudget,
    service,
    capped,
    latencyError,
    semaphorePeriod,
    channelReadout,
    triggerInterval,
    triggerTime,
    frameSequence,
    received,
    written,
    dropped,
    gapCount,
    producerId,
    producerDelta,
    modeCode,
    timingCount
};

/// Worker-owned diagnostic values and a copy transferred to the app loop.
struct diagnostics
{
    std::array<double, timingCount> m_timing{};     ///< Compatible timingDiag values, in microseconds where named.
    uint64_t                        m_skipped{ 0 }; ///< Nominal opportunities omitted from publication.
    uint64_t                m_discontinuities{ 0 }; ///< Stationary restarts after gaps, rate changes, or source resets.
    uint64_t                m_late{ 0 };            ///< Acquisitions later than their intended deadlines.
    double                  m_maxLateUs{ 0 };       ///< Largest measured monotonic lateness.
    double                  m_cadenceRatio{ 0 };    ///< Measured source period divided by the nominal model period.
    std::array<uint64_t, 2> m_low{}, m_high{};      ///< ADC rail clamps in generated numerical samples.

    /// Clear synchronization measurements on outage without erasing lifetime accounting.
    void invalidateSync( double postDelay /**< [in] configured phase offset in microseconds */ );

    /// Finalize a consumed trigger exactly once, including canceled acquisitions.
    void finishSync( syncClock      &clock /**< [in,out] committed nominal/source timeline */,
                     const syncTick &tick /**< [in] tentative trigger */,
                     bool            acquired /**< [in] whether a fresh model sample was generated */ );
};

inline syncTick syncClock::prepare( uint64_t producer, int64_t now, double fps ) const
{
    syncTick tick;
    tick.m_producer = producer;
    tick.m_newFrame = !m_haveSource || producer != m_producer;
    if( !tick.m_newFrame )
        return tick;
    tick.m_reset = m_haveSource && producer < m_producer;
    tick.m_delta = m_haveSource && !tick.m_reset ? producer - m_producer : 0;
    if( !m_haveTime || tick.m_reset )
    {
        tick.m_epoch      = now;
        tick.m_originTick = tick.m_delta ? tick.m_delta - 1 : 0;
        tick.m_sourceTick = tick.m_tick = tick.m_originTick;
        tick.m_due                      = true;
        tick.m_advance = tick.m_skipped = tick.m_tick;
    }
    else
    {
        tick.m_epoch       = m_epoch;
        tick.m_originTick  = m_originTick;
        tick.m_sourceTick  = m_sourceTick + tick.m_delta;
        const auto elapsed = static_cast<uint64_t>( std::floor( ( now - m_epoch + .5 ) * fps / 1e9 ) );
        tick.m_tick        = std::max( tick.m_sourceTick, m_originTick + elapsed );
        tick.m_due         = tick.m_tick >= m_nextAccounted;
        if( tick.m_due )
        {
            tick.m_advance = tick.m_tick - m_nextModel;
            tick.m_skipped = tick.m_tick - m_nextAccounted;
        }
    }
    return tick;
}

inline uint64_t syncClock::finish( const syncTick &tick, bool acquired )
{
    if( !tick.m_newFrame )
        return 0;
    if( !m_haveTime || tick.m_reset )
        m_nextAccounted = m_nextModel = 0;
    m_producer   = tick.m_producer;
    m_sourceTick = tick.m_sourceTick;
    m_epoch      = tick.m_epoch;
    m_originTick = tick.m_originTick;
    m_haveSource = m_haveTime = true;
    if( !tick.m_due )
        return 0; // This opportunity was already covered by elapsed-time recovery.
    m_nextAccounted = tick.m_tick + 1;
    if( acquired )
        m_nextModel = tick.m_tick + 1;
    return tick.m_skipped + ( acquired ? 0 : 1 );
}

inline void syncClock::resetTime()
{
    m_haveTime = false;
}

inline void diagnostics::invalidateSync( double postDelay )
{
    std::fill( m_timing.begin(), m_timing.begin() + frameSequence, 0 );
    m_timing[delayTarget]   = postDelay;
    m_timing[producerDelta] = 0;
    m_cadenceRatio          = 0;
}

inline void diagnostics::finishSync( syncClock &clock, const syncTick &tick, bool acquired )
{
    if( !tick.m_newFrame )
        return;
    m_skipped += clock.finish( tick, acquired );
    m_timing[producerId]    = tick.m_producer;
    m_timing[producerDelta] = tick.m_delta;
    if( tick.m_delta > 1 )
    {
        m_timing[dropped] += tick.m_delta - 1;
        ++m_timing[gapCount];
    }
}

inline void oscillator::configure( double frequency, double r, double fps, double rms )
{
    if( !std::isfinite( frequency ) || !std::isfinite( r ) || !std::isfinite( fps ) || !std::isfinite( rms ) ||
        frequency <= 0 || fps <= 0 || frequency >= fps / 2 || r < 0 || r >= 1 || rms < 0 )
        throw std::invalid_argument( "invalid AR(2) parameters" );
    const double omega = 2 * pi * frequency / fps;
    const double d     = ( 1 - r ) * ( 1 + r );
    const double sine  = std::sin( omega );
    const double b     = d * d + 4 * r * r * sine * sine;
    m_phi1             = 2 * r * std::cos( omega );
    m_phi2             = -r * r;
    m_sigma            = rms * std::sqrt( d * b / ( 1 + r * r ) );
    m_rho              = m_phi1 / ( 1 + r * r );
    m_eta              = std::sqrt( b ) / ( 1 + r * r );
    m_rms              = rms;
    if( !std::isfinite( m_sigma ) )
        throw std::invalid_argument( "AR(2) scaling overflow" );
}

inline void oscillator::initialize( double z1, double z2 )
{
    m_previous1 = m_rms * z1;
    m_previous2 = m_rms * ( m_rho * z1 + m_eta * z2 );
}

inline double oscillator::step( double z )
{
    const double value = m_phi1 * m_previous1 + m_phi2 * m_previous2 + m_sigma * z;
    m_previous2        = m_previous1;
    m_previous1        = value;
    return value;
}

inline double accelerationFactor( double frequency, double leverArm )
{
    return -( 2 * pi * frequency ) * ( 2 * pi * frequency ) * leverArm * arcsecToRad;
}

inline std::vector<modeSpec> referenceModes( uint64_t seed )
{
    std::vector<modeSpec>                  specs{ { 11, .00025, .9999999, .05, .05 },
                                                  { 14, .00025, .9999999, .05, .05 },
                                                  { 17, .00025, .9999999, .05, .05 },
                                                  { 20, .00025, .9999999, .05, .05 },
                                                  { 25, .00025, .9999999, .05, .05 },
                                                  { 28, .002, .99999, 1, 1 },
                                                  { 54, .002, .99995, 1, 1 },
                                                  { 56, .001, .99995, 1, 1 },
                                                  { 58, .002, .99995, 1, 1 },
                                                  { 75, .0008, .9998, 1, 1 },
                                                  { 85, .0008, .9998, 1, 1 },
                                                  { 120, .008, .998, 1, 1 },
                                                  { 220, .015, .996, 1, 1 },
                                                  { 300, .018, .995, 1, 1 } };
    std::mt19937_64                        rng( seed );
    std::uniform_real_distribution<double> frequency( 100, 900 ), amplitude( .0005, .002 );
    std::array<double, 200>                frequencies;
    for( auto &f : frequencies )
        f = frequency( rng );
    for( double f : frequencies )
    {
        const double envelope = std::exp( -.5 * std::pow( ( f - 250 ) / 250, 2 ) );
        specs.push_back( { f, amplitude( rng ) * ( .5 + 2 * envelope ), .998 - ( f - 100 ) / 800 * .006, .01, .01 } );
    }
    std::sort( specs.begin(),
               specs.end(),
               []( const modeSpec &a, const modeSpec &b ) { return a.m_frequency < b.m_frequency; } );
    return specs;
}

inline uint16_t adcCounts( double acceleration, const parameters &config, uint64_t &low, uint64_t &high )
{
    const double value = config.m_zero + acceleration * config.m_sensitivity / config.m_voltsPerBit;
    if( !std::isfinite( value ) )
        throw std::runtime_error( "nonfinite simulated ADC value" );
    const double rounded = std::round( value );
    if( rounded < 0 )
        ++low;
    if( rounded > 4095 )
        ++high;
    return static_cast<uint16_t>( std::clamp( rounded, 0.0, 4095.0 ) );
}

inline void model::configure( const parameters &config, const std::vector<modeSpec> &specs )
{
    if( !std::isfinite( config.m_leverArm ) || config.m_leverArm <= 0 || !std::isfinite( config.m_noisePSD ) ||
        config.m_noisePSD < 0 || !std::isfinite( config.m_zero ) || !std::isfinite( config.m_voltsPerBit ) ||
        config.m_voltsPerBit <= 0 || !std::isfinite( config.m_sensitivity ) || config.m_sensitivity <= 0 )
        throw std::invalid_argument( "invalid physical/ADC configuration" );
    m_config = config;
    m_modes.clear();
    for( const auto &spec : specs )
    {
        const double factor = accelerationFactor( spec.m_frequency, config.m_leverArm );
        if( !std::isfinite( factor ) || factor >= 0 || !std::isfinite( spec.m_accelRms ) || spec.m_accelRms < 0 ||
            !std::isfinite( spec.m_couplingX ) || !std::isfinite( spec.m_couplingY ) )
            throw std::invalid_argument( "invalid vibration mode" );
        m_modes.push_back( { spec, factor, {} } );
    }
    m_rng.seed( config.m_seed );
    m_normal.reset();
    m_clipLow  = {};
    m_clipHigh = {};
    m_ticks    = 0;
    m_sample   = {};
    setRate( config.m_fps );
}

inline bool model::validRate( double fps ) const
{
    if( !std::isfinite( fps ) || fps <= 0 || fps > 10000 )
        return false;
    for( const auto &mode : m_modes )
        if( mode.m_spec.m_frequency >= fps / 2 )
            return false;
    return true;
}

inline void model::setRate( double fps )
{
    if( !validRate( fps ) )
        throw std::invalid_argument( "sample rate must be positive, <=10000 and above mode Nyquist" );
    const double sigma = std::sqrt( m_config.m_noisePSD * fps / 2 );
    if( !std::isfinite( sigma ) )
        throw std::invalid_argument( "sensor noise overflow" );
    for( auto &mode : m_modes )
        for( auto &axis : mode.m_axes )
            axis.configure(
                mode.m_spec.m_frequency, mode.m_spec.m_r, fps, mode.m_spec.m_accelRms / -mode.m_accelPerArcsec );
    m_config.m_fps = fps;
    m_noiseSigma   = sigma;
    restart();
}

inline void model::restart()
{
    for( auto &mode : m_modes )
        for( auto &axis : mode.m_axes )
        {
            const double z1 = m_normal( m_rng );
            const double z2 = m_normal( m_rng );
            axis.initialize( z1, z2 );
        }
}

inline const sample &model::step()
{
    m_sample = {};
    for( auto &mode : m_modes )
    {
        for( size_t axis = 0; axis < 2; ++axis )
        {
            const double pointing = mode.m_axes[axis].step( m_normal( m_rng ) );
            m_sample.m_pointing[axis] += pointing;
            m_sample.m_wfsPointing[axis] +=
                pointing * ( axis == 0 ? mode.m_spec.m_couplingX : mode.m_spec.m_couplingY );
            m_sample.m_acceleration[axis] += mode.m_accelPerArcsec * pointing;
        }
    }
    for( size_t axis = 0; axis < 2; ++axis )
    {
        m_sample.m_acceleration[axis] += m_noiseSigma * m_normal( m_rng );
        if( !std::isfinite( m_sample.m_pointing[axis] ) || !std::isfinite( m_sample.m_wfsPointing[axis] ) )
            throw std::runtime_error( "nonfinite simulated pointing" );
        m_sample.m_adc[axis] = adcCounts( m_sample.m_acceleration[axis], m_config, m_clipLow[axis], m_clipHigh[axis] );
    }
    ++m_ticks;
    return m_sample;
}

inline bool model::advance( uint64_t skipped )
{
    const bool discontinuity = skipped > maxCatchUp;
    if( discontinuity )
        restart();
    else
        for( uint64_t tick = 0; tick < skipped; ++tick )
            step();
    step();
    return discontinuity;
}

inline void schedule::reset( int64_t now, double fps )
{
    if( !std::isfinite( fps ) || fps <= 0 || fps > 10000 )
        throw std::invalid_argument( "invalid schedule rate" );
    m_epoch = now;
    m_next  = 1;
    m_fps   = fps;
}

inline int64_t schedule::deadline() const
{
    return m_epoch + static_cast<int64_t>( std::llround( m_next * ( 1e9 / m_fps ) ) );
}

inline uint64_t schedule::consume( int64_t now )
{
    if( now < deadline() )
        throw std::logic_error( "sample before deadline" );
    const uint64_t latest =
        std::max( m_next, static_cast<uint64_t>( std::floor( ( now - m_epoch + .5 ) * m_fps / 1e9 ) ) );
    const uint64_t skipped = latest - m_next;
    m_next                 = latest + 1;
    return skipped;
}

} // namespace accelSimDetail
} // namespace app
} // namespace MagAOX
#endif
