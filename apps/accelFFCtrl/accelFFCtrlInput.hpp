/** \file accelFFCtrlInput.hpp
 * \brief Bounded retained-frame ingestion and clock translation for non-actuating observation.
 * \ingroup accelFFCtrl_files
 */
#ifndef accelFFCtrlInput_hpp
#define accelFFCtrlInput_hpp

#include <ctime>
#include <string>
#include <vector>

#include "accelFFCtrlModel.hpp"

namespace MagAOX
{
namespace app
{
namespace accelFFCtrlDetail
{

/// Public scalar controls, with milliseconds used for the prediction horizon only.
inline constexpr std::array<const char *, 11> parameterNames{ "frequency",
                                                              "memory_time",
                                                              "lever_arm",
                                                              "rotation",
                                                              "prediction_horizon",
                                                              "adc_zero_x",
                                                              "adc_zero_y",
                                                              "adc_volts_per_bit_x",
                                                              "adc_volts_per_bit_y",
                                                              "adc_sensitivity_x",
                                                              "adc_sensitivity_y" };

/// Read a scalar control in its INDI units; index must belong to parameterNames.
inline double parameterValue( const parameters &config /**< [in] configuration */,
                              size_t            index /**< [in] scalar index */ )
{
    if( index >= parameterNames.size() )
        return std::numeric_limits<double>::quiet_NaN();
    switch( index )
    {
    case 0:
        return config.m_frequency;
    case 1:
        return config.m_memoryTime;
    case 2:
        return config.m_leverArm;
    case 3:
        return config.m_rotationDegrees;
    case 4:
        return config.m_predictionNs * 1e-6;
    default:
        if( index < 7 )
            return config.m_zero[index - 5];
        if( index < 9 )
            return config.m_voltsPerBit[index - 7];
        return config.m_sensitivity[index - 9];
    }
}

/// Validate a complete candidate before replacing settings; used outside the real-time loop.
inline bool parameterUpdate( parameters &config /**< [in,out] accepted settings */,
                             size_t      index /**< [in] scalar index */,
                             double      value /**< [in] requested value in INDI units */ )
{
    if( index >= parameterNames.size() || !std::isfinite( value ) )
        return false;
    parameters candidate = config;
    switch( index )
    {
    case 0:
        candidate.m_frequency = value;
        break;
    case 1:
        candidate.m_memoryTime = value;
        break;
    case 2:
        candidate.m_leverArm = value;
        break;
    case 3:
        candidate.m_rotationDegrees = value;
        break;
    case 4:
        if( value < 0 || value > candidate.m_maxPredictionNs * 1e-6 )
            return false;
        candidate.m_predictionNs = static_cast<int64_t>( std::llround( value * 1e6 ) );
        break;
    default:
        if( index < 7 )
            candidate.m_zero[index - 5] = value;
        else if( index < 9 )
            candidate.m_voltsPerBit[index - 7] = value;
        else
            candidate.m_sensitivity[index - 9] = value;
    }
    try
    {
        harmonicEstimator validated( candidate );
    }
    catch( const std::exception & )
    {
        return false;
    }
    config = candidate;
    return true;
}

/// Parse a complete finite scalar control outside the real-time loop; reject malformed/trailing text.
inline bool parameterUpdate( parameters        &config /**< [in,out] accepted settings */,
                             size_t             index /**< [in] scalar index */,
                             const std::string &text /**< [in] requested value in INDI units */ )
{
    try
    {
        size_t       consumed;
        const double value = std::stod( text, &consumed );
        if( text.find_first_not_of( " \t\r\n", consumed ) != std::string::npos )
            return false;
        return parameterUpdate( config, index, value );
    }
    catch( const std::exception & )
    {
        return false;
    }
}

/// Reject actuator stream names before any native attachment; this app is only an ADC consumer.
inline bool inputNameValid( const std::string &name /**< [in] proposed stream name */ )
{
    if( name.empty() || name.starts_with( "dm" ) )
        return false;
    for( unsigned char c : name )
        if( !( ( c >= 'a' && c <= 'z' ) || ( c >= 'A' && c <= 'Z' ) || ( c >= '0' && c <= '9' ) || c == '_' ||
               c == '-' ) )
            return false;
    return true;
}

/// Parse the startup opt-in explicitly; malformed text preserves the previous value.
inline bool singleFrameGapSetting( bool              &enabled /**< [in,out] accepted startup choice */,
                                   const std::string &text /**< [in] true/false or 1/0 */ )
{
    if( text != "true" && text != "false" && text != "1" && text != "0" )
        return false;
    enabled = text == "true" || text == "1";
    return true;
}

/// Snapshot of native metadata; equality checks publication identity, geometry and write state.
struct inputHeader
{
    uint32_t m_axes{ 0 }, m_width{ 0 }, m_height{ 0 }, m_depth{ 0 }; ///< Attached immutable geometry.
    uint64_t m_counter{ 0 }, m_index{ 0 }; ///< Latest completed publication and temporal plane.
    bool m_valid{ false }, m_uint16{ false }, m_cpu{ false }, m_write{ false }; ///< Mapping/type/publication validity.

    /// Compare complete snapshots surrounding a local copy.
    bool operator==( const inputHeader &other /**< [in] second snapshot */ ) const = default;
};

/// Owned copy of one temporal plane; never a queued pointer into shared memory.
struct inputFrame
{
    std::array<uint16_t, 2> m_adc{};                   ///< Raw X/Y codes, without floating-point reinterpretation.
    timespec                m_acquired{}, m_written{}; ///< The same plane's realtime acquisition and write timestamps.
    uint64_t                m_counter{ 0 }; ///< Per-plane cntarray, or associated global counter for depth one.
};

/// A closely bracketed sample of the local realtime/monotonic relationship.
struct inputClock
{
    int64_t m_real{ 0 }, m_mono{ 0 }, m_uncertainty{ 0 }; ///< Epoch ns, midpoint monotonic ns and half-bracket span.
};

/// Input health codes are independent of fit warmup/readiness; lastEvent retains the latest reset or observed overrun.
enum class inputState
{
    disconnected,
    waiting,
    streaming,
    incompatible,
    overrun,
    invalidTime,
    clockFault,
    stale,
    numerical
};

/// Fixed snapshot copied to supervision without making the worker wait on INDI.
struct inputDiagnostics
{
    prediction m_prediction; ///< Cleared unless the latest numerical query is valid.
    parameters m_config;     ///< Settings actually applied by the owner.
    inputState m_state{ inputState::disconnected },
        m_lastEvent{ inputState::disconnected }; ///< Current status and most recent reset/overrun event.
    uint64_t m_revision{ 0 }, m_generation{ 0 }; ///< Applied control revision and source identity generation.
    uint64_t m_processed{ 0 }, m_dropped{ 0 },
        m_invalid{ 0 }; ///< Lifetime accepted, lost/discarded and invalid records.
    uint64_t m_restarts{ 0 }, m_resets{ 0 }, m_rollbacks{ 0 }; ///< Lifetime source/fit discontinuity counters.
    uint64_t m_admittedGaps{ 0 }, m_rejectedGaps{ 0 }; ///< Observed counter gaps retained or given a full reset.
    bool     m_allowSingleFrameGap{ false };           ///< Startup-only opt-in; false preserves reset-on-loss behavior.
    uint64_t m_duplicates{ 0 }, m_busy{ 0 }, m_races{ 0 }; ///< Empty polls, busy writes and rejected snapshot attempts.
    uint64_t m_counter{ 0 }; ///< Last examined source publication; may differ from last accepted prediction identity.
    uint32_t m_depth{ 0 };   ///< Retention depth; one explicitly means overwritten samples cannot be recovered.
    double   m_rate{ 0 }, m_sampleRate{ 0 }, m_ageUs{ -1 }, m_horizonUs{ 0 }, m_writeDelayUs{ 0 },
        m_serviceUs{ 0 }; ///< Rates/timing in Hz/us.
};

/// Single-owner runtime: bounded source snapshots, strict time mapping and unchanged numerical estimator.
/** Opt-in retention uses private accepted ordinals only after physical counter/clock validation.
 * Public predictions retain physical counter identity; a retained gap requires a new prediction query.
 */
class inputProcessor
{
    harmonicEstimator       m_estimator; ///< Sole numerical owner; no simulator access.
    inputDiagnostics        m_stats;     ///< Cumulative diagnostics survive resets and configuration changes.
    std::vector<inputFrame> m_batch;     ///< Startup-allocated bounded local copy of retained source frames.
    inputClock              m_anchor;    ///< Fixed epoch translation until a source restart or detected clock step.
    int64_t  m_clockTolerance, m_lastTime{ -1 }; ///< Clock uncertainty/step allowance and last accepted acquisition.
    double   m_period{ 0 };      ///< Smoothed accepted acquisition interval; never derived from semaphore count.
    int64_t  m_rateTime{ -1 };   ///< Monotonic start of a 250-ms accepted-throughput reporting window.
    uint64_t m_rateCount{ 0 };   ///< Accepted records in that window; fit resets do not hide lost throughput.
    uint64_t m_cursor{ 0 };      ///< Last examined counter in this source generation.
    uint64_t m_seenCounter{ 0 }; ///< Latest stable snapshot counter, including an unconsumed bounded-batch tail.
    uint64_t m_sequence{ 0 }, m_lastCounter{ 0 }; ///< Optional kernel ordinal and last accepted physical counter.
    int64_t  m_lastGap{ -1 }; ///< Acquisition time of the last admitted gap; cleared by every full reset.
    bool     m_connected{ false }, m_bound{ false }, m_haveAnchor{ false }; ///< Source/counter/clock binding state.

    /// Clear the fit and estimates while preserving lifetime counters and source cursor.
    void invalidate( inputState reason /**< [in] diagnostic reset cause */ );

    /// Convert a validated source epoch timestamp into the anchored monotonic domain without overflow.
    bool translate( int64_t real /**< [in] realtime ns */, int64_t &mono /**< [out] mapped monotonic ns */ ) const;

    /// Admit only a mature, fresh, isolated single-frame gap with consistent physical timing at depth one.
    bool singleGapValid( uint64_t counter /**< [in] new physical counter */,
                         int64_t  mapped /**< [in] new acquisition ns */,
                         int64_t  now /**< [in] actual receipt ns */ ) const;

  public:
    /// Validate settings and allocate only the fixed maximum batch at startup.
    inputProcessor( const parameters &config         = parameters{} /**< [in] numerical settings */,
                    size_t            capacity       = 256 /**< [in] bounded copy capacity */,
                    int64_t           clockTolerance = 1000000 /**< [in] maximum clock uncertainty/step [ns] */,
                    bool allowSingleFrameGap = false /**< [in] retain qualified single gaps; disabled by default */ );

    /// Validate a timespec and convert to epoch nanoseconds, rejecting absent or overflowing timestamps.
    static bool timeNs( const timespec &value /**< [in] timestamp */, int64_t &result /**< [out] nanoseconds */ );

    /// Apply a validated coherent control revision and reset the fit without replaying previous records.
    void configure( const parameters &config /**< [in] numerical settings */,
                    uint64_t          revision /**< [in] applied revision */ );

    /// Clear an observation fit explicitly; current source cursor and lifetime counters remain intact.
    void reset();

    /// Bind a newly attached stream; the first poll starts at its newest committed sample.
    void attached();

    /// Invalidate and clear output immediately on mapping loss.
    void disconnected();

    /// Drain one bounded stable snapshot in order; source provides header() and frame(index) value copies.
    template <typename sourceT, typename clockT>
    size_t drain( sourceT &source /**< [in] mapped source adapter */,
                  clockT   clockNow /**< [in] allocation-free clock callback, invoked after the stable copy */ );

    /// Update prediction/freshness without requiring a new source callback.
    void predict( int64_t now /**< [in] current monotonic ns */ );

    /// Return the owner's complete value snapshot.
    const inputDiagnostics &diagnostics() const;
};

inline inputProcessor::inputProcessor( const parameters &config,
                                       size_t            capacity,
                                       int64_t           clockTolerance,
                                       bool              allowSingleFrameGap )
    : m_estimator( config ), m_clockTolerance( clockTolerance )
{
    if( capacity == 0 || capacity > 65536 || clockTolerance <= 0 )
        throw std::invalid_argument( "invalid input batch/clock settings" );
    m_batch.resize( capacity );
    m_stats.m_config              = config;
    m_stats.m_allowSingleFrameGap = allowSingleFrameGap;
}

inline bool inputProcessor::timeNs( const timespec &value, int64_t &result )
{
    if( value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000 ||
        value.tv_sec > ( std::numeric_limits<int64_t>::max() - value.tv_nsec ) / 1000000000 )
        return false;
    result = int64_t( value.tv_sec ) * 1000000000 + value.tv_nsec;
    return result > 0;
}

inline void inputProcessor::invalidate( inputState reason )
{
    m_estimator.reset();
    m_stats.m_prediction = {};
    m_stats.m_state = m_stats.m_lastEvent = reason;
    m_stats.m_sampleRate = m_period = 0;
    m_stats.m_ageUs                 = -1;
    m_stats.m_horizonUs             = 0;
    m_lastTime                      = -1;
    m_sequence = m_lastCounter = 0;
    m_lastGap                  = -1;
    ++m_stats.m_resets;
}

inline void inputProcessor::configure( const parameters &config, uint64_t revision )
{
    m_estimator.configure( config );
    m_stats.m_config   = config;
    m_stats.m_revision = revision;
    invalidate( m_connected ? inputState::waiting : inputState::disconnected );
}

inline void inputProcessor::reset()
{
    invalidate( m_connected ? inputState::waiting : inputState::disconnected );
}

inline void inputProcessor::attached()
{
    if( m_bound && m_seenCounter > m_cursor )
        m_stats.m_dropped += m_seenCounter - m_cursor;
    if( m_stats.m_generation )
        ++m_stats.m_restarts;
    ++m_stats.m_generation;
    m_connected = true;
    m_bound = m_haveAnchor = false;
    m_cursor = m_seenCounter = m_stats.m_counter = 0;
    m_stats.m_rate                               = 0;
    m_rateTime                                   = -1;
    m_rateCount                                  = 0;
    invalidate( inputState::waiting );
}

inline void inputProcessor::disconnected()
{
    if( m_bound && m_seenCounter > m_cursor )
        m_stats.m_dropped += m_seenCounter - m_cursor;
    if( m_connected )
        invalidate( inputState::disconnected );
    m_connected = m_bound = m_haveAnchor = false;
    m_stats.m_depth                      = 0;
    m_stats.m_rate                       = 0;
    m_rateTime                           = -1;
    m_rateCount                          = 0;
}

inline bool inputProcessor::translate( int64_t real, int64_t &mono ) const
{
    const int64_t delta = real - m_anchor.m_real;
    if( ( delta > 0 && m_anchor.m_mono > std::numeric_limits<int64_t>::max() - delta ) ||
        ( delta < 0 && m_anchor.m_mono < -delta ) )
        return false;
    mono = m_anchor.m_mono + delta;
    return mono >= 0;
}

inline bool inputProcessor::singleGapValid( uint64_t counter, int64_t mapped, int64_t now ) const
{
    if( m_stats.m_depth != 1 || m_lastTime < 0 || counter <= m_lastCounter || counter - m_lastCounter != 2 ||
        m_estimator.state() != estimatorState::ready || m_stats.m_prediction.m_state != estimatorState::ready ||
        now - m_lastTime > m_stats.m_config.m_maxAgeNs || now - mapped > m_stats.m_config.m_maxAgeNs )
        return false;
    const int64_t elapsed = mapped - m_lastTime;
    const double  period  = 1e9 / m_stats.m_config.m_sampleRate;
    return elapsed > 0 && elapsed <= 1000000 && elapsed <= m_stats.m_config.m_maxIntervalNs &&
           std::abs( elapsed - 2 * period ) <= .05 * period &&
           ( m_lastGap < 0 || mapped - m_lastGap >= std::llround( m_stats.m_config.m_memoryTime * 1e9 ) );
}

template <typename sourceT, typename clockT>
size_t inputProcessor::drain( sourceT &source, clockT clockNow )
{
    if( !m_connected )
        return 0;
    const auto before = source.header();
    if( !before.m_valid || !before.m_uint16 || !before.m_cpu || before.m_axes < 2 || before.m_axes > 3 ||
        before.m_width != 2 || before.m_height != 1 || before.m_depth == 0 || before.m_index >= before.m_depth )
    {
        if( m_stats.m_state != inputState::incompatible )
            invalidate( inputState::incompatible );
        return 0;
    }
    m_stats.m_depth = before.m_depth;
    if( before.m_write )
    {
        ++m_stats.m_busy;
        return 0;
    }
    if( m_bound && before.m_counter < std::max( m_cursor, m_seenCounter ) )
    {
        ++m_stats.m_rollbacks;
        attached();
    }
    if( !m_bound )
    {
        m_cursor = before.m_counter ? before.m_counter - 1 : 0;
        m_bound  = true;
    }
    if( before.m_counter == m_cursor )
    {
        ++m_stats.m_duplicates;
        return 0;
    }
    const uint64_t retained = before.m_counter >= before.m_depth ? before.m_counter - before.m_depth + 1 : 1;
    const uint64_t first    = std::max( m_cursor + 1, retained );
    const size_t   count    = std::min<uint64_t>( before.m_counter - first + 1, m_batch.size() );
    for( size_t n = 0; n < count; ++n )
    {
        const size_t plane =
            ( before.m_index + before.m_depth - ( ( before.m_counter - first - n ) % before.m_depth ) ) %
            before.m_depth;
        m_batch[n] = source.frame( plane );
        if( m_batch[n].m_counter != first + n )
        {
            ++m_stats.m_races;
            return 0;
        }
    }
    if( !( source.header() == before ) )
    {
        ++m_stats.m_races;
        return 0;
    }
    m_seenCounter          = before.m_counter;
    const inputClock clock = clockNow();
    if( clock.m_real <= 0 || clock.m_mono < 0 || clock.m_uncertainty < 0 || clock.m_uncertainty > m_clockTolerance ||
        ( m_haveAnchor &&
          ( clock.m_mono < m_anchor.m_mono ||
            std::abs( ( static_cast<long double>( clock.m_real ) - m_anchor.m_real ) -
                      ( static_cast<long double>( clock.m_mono ) - m_anchor.m_mono ) ) > m_clockTolerance ) ) )
    {
        if( m_stats.m_state != inputState::clockFault )
            invalidate( inputState::clockFault );
        m_haveAnchor = false;
        return 0;
    }
    if( !m_haveAnchor )
    {
        m_anchor     = clock;
        m_haveAnchor = true;
    }
    const uint64_t missing    = first - m_cursor - 1;
    bool           pendingGap = missing && m_stats.m_allowSingleFrameGap;
    if( missing )
    {
        m_stats.m_dropped += missing;
        if( !pendingGap )
        {
            ++m_stats.m_rejectedGaps;
            invalidate( inputState::overrun );
        }
    }
    for( size_t n = 0; n < count; ++n )
    {
        const auto &frame = m_batch[n];
        m_cursor = m_stats.m_counter = frame.m_counter;
        int64_t acquired, written, mapped;
        if( !timeNs( frame.m_acquired, acquired ) || !timeNs( frame.m_written, written ) || written < acquired ||
            written > clock.m_real || !translate( acquired, mapped ) || mapped > clock.m_mono ||
            ( m_lastTime >= 0 && mapped <= m_lastTime ) )
        {
            if( pendingGap )
            {
                ++m_stats.m_rejectedGaps;
                pendingGap = false;
            }
            ++m_stats.m_dropped;
            ++m_stats.m_invalid;
            invalidate( inputState::invalidTime );
            continue;
        }
        bool admitted = false;
        if( pendingGap )
        {
            admitted = missing == 1 && frame.m_adc[0] <= 4095 && frame.m_adc[1] <= 4095 &&
                       singleGapValid( frame.m_counter, mapped, clock.m_mono );
            if( !admitted )
            {
                ++m_stats.m_rejectedGaps;
                invalidate( inputState::overrun );
            }
            pendingGap = false;
        }
        if( m_estimator.state() != estimatorState::warming && m_estimator.state() != estimatorState::ready )
            invalidate( m_estimator.state() == estimatorState::stale ? inputState::stale : inputState::numerical );
        const uint64_t sequence = m_stats.m_allowSingleFrameGap ? m_sequence + 1 : frame.m_counter;
        if( !m_estimator.consume( adcSample{ frame.m_adc, mapped, sequence, m_stats.m_generation }, clock.m_mono ) )
        {
            if( admitted )
                ++m_stats.m_rejectedGaps;
            const auto reason =
                m_estimator.state() == estimatorState::stale ? inputState::stale : inputState::numerical;
            ++m_stats.m_dropped;
            ++m_stats.m_invalid;
            invalidate( reason );
            continue;
        }
        m_sequence = sequence;
        if( admitted )
        {
            ++m_stats.m_admittedGaps;
            m_lastGap            = mapped;
            m_stats.m_lastEvent  = inputState::overrun;
            m_stats.m_prediction = {}; // Re-solve before publishing any post-gap prediction.
            m_stats.m_ageUs      = -1;
            m_stats.m_horizonUs  = 0;
        }
        if( m_lastTime >= 0 )
        {
            const double period  = admitted ? ( mapped - m_lastTime ) / 2. : mapped - m_lastTime;
            m_period             = m_period > 0 ? .01 * period + .99 * m_period : period;
            m_stats.m_sampleRate = 1e9 / m_period;
        }
        m_lastTime             = mapped;
        m_lastCounter          = frame.m_counter;
        m_stats.m_writeDelayUs = ( written - acquired ) * 1e-3;
        m_stats.m_state        = inputState::streaming;
        ++m_stats.m_processed;
        ++m_rateCount;
    }
    return count;
}

inline void inputProcessor::predict( int64_t now )
{
    if( m_rateTime < 0 )
        m_rateTime = now;
    else if( now - m_rateTime >= 250000000 )
    {
        m_stats.m_rate = 1e9 * m_rateCount / ( now - m_rateTime );
        m_rateCount    = 0;
        m_rateTime     = now;
    }
    m_stats.m_prediction = m_estimator.predict( now );
    if( m_stats.m_allowSingleFrameGap )
        m_stats.m_prediction.m_sequence = m_lastCounter; // Public identity always remains the physical counter.
    if( m_lastTime >= 0 )
        m_stats.m_ageUs = ( now - m_lastTime ) * 1e-3;
    if( m_stats.m_prediction.m_state == estimatorState::stale && m_connected )
        m_stats.m_state = m_stats.m_lastEvent = inputState::stale;
    if( m_stats.m_prediction.m_state == estimatorState::numericalFault )
        m_stats.m_state = m_stats.m_lastEvent = inputState::numerical;
    m_stats.m_horizonUs = m_stats.m_prediction.m_state == estimatorState::ready
                              ? ( m_stats.m_prediction.m_targetTimeNs - m_stats.m_prediction.m_sampleTimeNs ) * 1e-3
                              : 0;
}

inline const inputDiagnostics &inputProcessor::diagnostics() const
{
    return m_stats;
}

} // namespace accelFFCtrlDetail
} // namespace app
} // namespace MagAOX
#endif
