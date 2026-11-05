/** \file accelFFCtrlModel.hpp
 * \brief Timestamped raw-ADC harmonic acceleration and physical pointing estimation, without stream access.
 * \ingroup accelFFCtrl_files
 */
#ifndef accelFFCtrlModel_hpp
#define accelFFCtrlModel_hpp

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>

/** \defgroup accelFFCtrl Accelerometer Feedforward Numerics
 * \brief Independent acceleration-to-pointing estimation without actuator publication.
 * \ingroup apps
 */
/** \defgroup accelFFCtrl_files Accelerometer Feedforward Numerical Files
 * \ingroup accelFFCtrl
 */

namespace MagAOX
{
namespace app
{
namespace accelFFCtrlDetail
{

/// Startup-only physical calibration, window, and numerical time-validity settings.
struct parameters
{
    std::array<double, 2> m_zero{ 2048, 2048 }; ///< ADC offsets in counts, before channel polarity is applied.
    std::array<double, 2> m_voltsPerBit{ .0083799, .0083799 }; ///< Empirical per-channel volts per ADC count.
    std::array<double, 2> m_sensitivity{ 1.02, 1.02 }; ///< Effective volts per m/s^2, matching the source calibration.
    std::array<unsigned, 2> m_channel{ 0, 1 };         ///< Raw-pair indices for the two reported sensor axes.
    std::array<int, 2>      m_polarity{ 1, 1 };        ///< Sensor-axis signs, each exactly -1 or +1.
    double  m_frequency{ 28.5 };    ///< Configured harmonic frequency [Hz], not an automatically tracked peak.
    double  m_memoryTime{ .1 };     ///< Approximate total mean memory [s]; each of two stages uses half this value.
    double  m_sampleRate{ 8000 };   ///< Nominal rate [Hz], used for validation and the first sample's window weight.
    double  m_leverArm{ 10 };       ///< Effective length [m]; a simulation starting value, not a universal calibration.
    double  m_rotationDegrees{ 0 }; ///< Positive rotation from sensor X/Y into the requested pointing axes.
    int64_t m_maxAgeNs{ 2000000 };  ///< Maximum accepted acquisition age, also checked without new input.
    int64_t m_maxIntervalNs{ 2000000 };    ///< Maximum consecutive acquisition interval before fit invalidation.
    int64_t m_maxPredictionNs{ 50000000 }; ///< Maximum target time beyond the newest sample; model-validity bound.
    int64_t m_predictionNs{ 0 };           ///< Default future horizon measured from the query's current time.
};

/// One owned raw pair with source identity; time is already mapped into the caller's monotonic domain.
struct adcSample
{
    std::array<uint16_t, 2> m_adc{}; ///< Two contiguous UINT16 codes; values above 4095 are invalid.
    int64_t  m_timeNs{ 0 };     ///< Acquisition nanoseconds in the same domain as consume()/predict() current time.
    uint64_t m_sequence{ 0 };   ///< Unique source publication counter, consecutive within one generation.
    uint64_t m_generation{ 0 }; ///< Source identity; changes require an explicit reset before ingestion.
};

/// Input faults latch until reset; invalidRequest applies only to the rejected prediction query.
enum class estimatorState
{
    warming,
    ready,
    invalidSample,
    timingFault,
    sequenceFault,
    stale,
    numericalFault,
    invalidRequest
};

/// Complete value result; every invalid result has zero numerical estimates, never a held prediction.
struct prediction
{
    estimatorState        m_state{ estimatorState::warming }; ///< Ready means a fresh, conditioned numerical fit only.
    std::array<double, 2> m_amplitude{}; ///< Peak harmonic acceleration [m/s^2] in each decoded sensor axis.
    std::array<double, 2> m_phase{};     ///< Acceleration phase [rad] at newest sample: amplitude*cos(omega*dt+phase).
    std::array<double, 2> m_offset{};    ///< Fitted sensor-axis DC acceleration [m/s^2], excluded from vibration.
    std::array<double, 2> m_acceleration{}; ///< Isolated sensor-axis acceleration [m/s^2] at the requested target.
    std::array<double, 2> m_pointing{}; ///< Rotated physical disturbance [rad] at target, without correction sign/gain.
    double                m_rcond{ 0 }; ///< Reciprocal condition estimate of the shared three-regressor Gram matrix.
    uint64_t m_samples{ 0 }; ///< Unique samples consumed since reset; duplicate notifications do not increment it.
    uint64_t m_sequence{ 0 }, m_generation{ 0 }; ///< Last accepted source identity anchoring this fit/result.
    int64_t  m_sampleTimeNs{ 0 };                ///< Acquisition time anchoring the reported sensor phases.
    int64_t  m_targetTimeNs{ 0 };                ///< Requested prediction time; zero for invalid results.
};

/// Single-owner, fixed-storage harmonic estimator; no model, RNG, clock, I/O, or actuator dependency.
/** All available ordered records must be passed to consume(), independently of prediction/query cadence.
 * Two cascaded filters apply the same positive window to Gram and data moments. A matched sinusoid plus
 * DC is exactly represented before quantization/noise; detuning/envelope tracking retains window phase.
 * Configure/reset may throw only for invalid startup settings. consume()/predict() allocate no storage.
 */
class harmonicEstimator
{
    /** \name Estimator - Data
     * @{
     */
    parameters                     m_config; ///< Immutable between explicit configure() calls, which reset the fit.
    std::array<Eigen::Matrix3d, 2> m_gram;   ///< Shared offset/cosine/sine statistics at each window stage.
    std::array<Eigen::Matrix<double, 3, 2>, 2> m_moments; ///< Corresponding acceleration moments for both axes.
    std::array<double, 2>                      m_scale{}; ///< Precomputed signed physical acceleration per count.
    double    m_omega{ 0 }, m_anglePerAccel{ 0 }; ///< Carrier rate [rad/s] and negative harmonic inverse [rad/(m/s^2)].
    double    m_cosRotation{ 1 }, m_sinRotation{ 0 }; ///< Prepared sensor-to-pointing rotation.
    double    m_phase{ 0 }; ///< Wrapped carrier phase at the last accepted acquisition, initially zero.
    int64_t   m_warmupNs{ 0 }, m_firstTimeNs{ 0 }, m_lastTimeNs{ 0 }; ///< Coverage and acquisition history.
    int64_t   m_lastReceiptNs{ 0 }; ///< Last accepted current-time argument, checked for clock rollback.
    adcSample m_last;         ///< Last accepted source identity and raw values, used to recognize true duplicates.
    uint64_t  m_samples{ 0 }; ///< Number of accepted unique samples; zero also indicates no bound source yet.
    estimatorState m_state{ estimatorState::warming }; ///< Latched validity; only reset/reconfigure clears faults.
    ///@}

    /// Return a cleared invalid result, optionally latching the reason until reset.
    prediction fail( estimatorState reason /**< [in] failure/readiness status */,
                     bool           latch = true /**< [in] update owner state, except for a bad query */ );

  public:
    /** \name Estimator
     * @{
     */
    /// Validate configuration and initialize fixed storage; invalid settings throw before consuming any sample.
    explicit harmonicEstimator( const parameters &config = parameters{} /**< [in] startup numerical configuration */ );

    /// Validate new settings, preserving the old configuration if invalid; a successful change resets all history.
    void configure( const parameters &config /**< [in] replacement startup settings */ );

    /// Clear source binding, window, warmup, and fault state, retaining the validated configuration.
    void reset();

    /// Consume one unique pair; false means duplicate or latched fault, distinguishable through state()/count.
    bool consume( const adcSample &value /**< [in] owned raw measurement and acquisition identity */,
                  int64_t          nowNs /**< [in] current monotonic ns, not a synthesized sample time */ );

    /// Predict at now plus the configured horizon; sample age is included exactly once.
    prediction predict( int64_t nowNs /**< [in] current monotonic ns in the acquisition clock domain */ );

    /// Predict physical disturbance at an explicit future target without extrapolating beyond configured validity.
    prediction predict( int64_t nowNs /**< [in] current monotonic ns, used for freshness */,
                        int64_t targetNs /**< [in] target ns in that domain, at or after now */ );

    /// Return owner state; valid output also requires a successful predict() freshness/conditioning check.
    estimatorState state() const;

    /// Return the unique-sample count since the last reset.
    uint64_t samples() const;
    ///@}
};

inline harmonicEstimator::harmonicEstimator( const parameters &config )
{
    configure( config );
}

inline void harmonicEstimator::configure( const parameters &config )
{
    constexpr double pi      = 3.14159265358979323846;
    const double     omega   = 2 * pi * config.m_frequency;
    const double     inverse = -1 / ( omega * omega * config.m_leverArm );
    const double     warmup  = std::max( 3 / config.m_frequency, 3 * config.m_memoryTime ) * 1e9;
    if( !std::isfinite( config.m_frequency ) || config.m_frequency <= 0 || !std::isfinite( omega ) ||
        !std::isfinite( config.m_memoryTime ) || config.m_memoryTime <= 0 || !std::isfinite( config.m_sampleRate ) ||
        config.m_sampleRate <= 0 || config.m_frequency >= config.m_sampleRate / 2 ||
        !std::isfinite( config.m_leverArm ) || config.m_leverArm <= 0 || !std::isfinite( inverse ) || inverse == 0 ||
        !std::isfinite( config.m_rotationDegrees ) || config.m_maxAgeNs <= 0 || config.m_maxIntervalNs <= 0 ||
        config.m_frequency * ( config.m_maxIntervalNs * 1e-9 ) >= .5 || config.m_maxPredictionNs <= 0 ||
        config.m_predictionNs < 0 || config.m_predictionNs > config.m_maxPredictionNs || !std::isfinite( warmup ) ||
        warmup >= double( std::numeric_limits<int64_t>::max() ) )
        throw std::invalid_argument( "invalid harmonic estimator configuration" );
    std::array<double, 2> scale;
    for( size_t axis = 0; axis < 2; ++axis )
    {
        scale[axis] = config.m_polarity[axis] * config.m_voltsPerBit[axis] / config.m_sensitivity[axis];
        if( !std::isfinite( config.m_zero[axis] ) || !std::isfinite( config.m_voltsPerBit[axis] ) ||
            config.m_voltsPerBit[axis] <= 0 || !std::isfinite( config.m_sensitivity[axis] ) ||
            config.m_sensitivity[axis] <= 0 || config.m_channel[axis] >= 2 ||
            ( config.m_polarity[axis] != -1 && config.m_polarity[axis] != 1 ) || !std::isfinite( scale[axis] ) ||
            scale[axis] == 0 )
            throw std::invalid_argument( "invalid accelerometer ADC calibration" );
    }
    m_config              = config;
    m_scale               = scale;
    m_omega               = omega;
    m_anglePerAccel       = inverse;
    const double rotation = std::remainder( config.m_rotationDegrees, 360. ) * pi / 180;
    m_cosRotation         = std::cos( rotation );
    m_sinRotation         = std::sin( rotation );
    m_warmupNs            = static_cast<int64_t>( std::ceil( warmup ) );
    reset();
}

inline void harmonicEstimator::reset()
{
    for( size_t stage = 0; stage < 2; ++stage )
    {
        m_gram[stage].setZero();
        m_moments[stage].setZero();
    }
    m_phase       = 0;
    m_firstTimeNs = m_lastTimeNs = m_lastReceiptNs = 0;
    m_last                                         = {};
    m_samples                                      = 0;
    m_state                                        = estimatorState::warming;
}

inline prediction harmonicEstimator::fail( estimatorState reason, bool latch )
{
    if( latch )
        m_state = reason;
    prediction result;
    result.m_state        = reason;
    result.m_samples      = m_samples;
    result.m_sequence     = m_last.m_sequence;
    result.m_generation   = m_last.m_generation;
    result.m_sampleTimeNs = m_lastTimeNs;
    return result;
}

inline bool harmonicEstimator::consume( const adcSample &value, int64_t nowNs )
{
    if( m_state != estimatorState::warming && m_state != estimatorState::ready )
        return false;
    if( value.m_timeNs < 0 || nowNs < value.m_timeNs || ( m_samples && nowNs < m_lastReceiptNs ) )
    {
        m_state = estimatorState::timingFault;
        return false;
    }
    if( m_samples && value.m_generation == m_last.m_generation && value.m_sequence == m_last.m_sequence &&
        value.m_timeNs == m_last.m_timeNs && value.m_adc == m_last.m_adc )
        return false; // Repeated notifications never advance the fit or its freshness time.
    if( value.m_adc[0] > 4095 || value.m_adc[1] > 4095 )
    {
        m_state = estimatorState::invalidSample;
        return false;
    }
    if( nowNs - value.m_timeNs > m_config.m_maxAgeNs )
    {
        m_state = estimatorState::stale;
        return false;
    }
    if( m_samples &&
        ( value.m_generation != m_last.m_generation || m_last.m_sequence == std::numeric_limits<uint64_t>::max() ||
          value.m_sequence != m_last.m_sequence + 1 ) )
    {
        m_state = estimatorState::sequenceFault;
        return false;
    }
    if( m_samples && ( value.m_timeNs <= m_lastTimeNs || value.m_timeNs - m_lastTimeNs > m_config.m_maxIntervalNs ) )
    {
        m_state = estimatorState::timingFault;
        return false;
    }
    const double     dt    = m_samples ? ( value.m_timeNs - m_lastTimeNs ) * 1e-9 : 1 / m_config.m_sampleRate;
    constexpr double twoPi = 6.28318530717958647692;
    if( m_samples )
        m_phase = std::remainder( m_phase + m_omega * dt, twoPi );
    const double          weight = -std::expm1( -2 * dt / m_config.m_memoryTime ), lambda = 1 - weight;
    const Eigen::Vector3d u( 1, std::cos( m_phase ), std::sin( m_phase ) );
    Eigen::Vector2d       acceleration;
    for( size_t axis = 0; axis < 2; ++axis )
        acceleration[axis] =
            ( double( value.m_adc[m_config.m_channel[axis]] ) - m_config.m_zero[axis] ) * m_scale[axis];
    if( !acceleration.allFinite() )
    {
        m_state = estimatorState::numericalFault;
        return false;
    }
    m_gram[0]    = lambda * m_gram[0] + weight * u * u.transpose();
    m_moments[0] = lambda * m_moments[0] + weight * u * acceleration.transpose();
    m_gram[1]    = lambda * m_gram[1] + weight * m_gram[0];
    m_moments[1] = lambda * m_moments[1] + weight * m_moments[0];
    if( !m_gram[0].allFinite() || !m_gram[1].allFinite() || !m_moments[0].allFinite() || !m_moments[1].allFinite() )
    {
        m_state = estimatorState::numericalFault;
        return false;
    }
    if( !m_samples )
        m_firstTimeNs = value.m_timeNs;
    m_last          = value;
    m_lastTimeNs    = value.m_timeNs;
    m_lastReceiptNs = nowNs;
    ++m_samples;
    return true;
}

inline prediction harmonicEstimator::predict( int64_t nowNs )
{
    if( nowNs < 0 || nowNs > std::numeric_limits<int64_t>::max() - m_config.m_predictionNs )
        return fail( estimatorState::invalidRequest, false );
    return predict( nowNs, nowNs + m_config.m_predictionNs );
}

inline prediction harmonicEstimator::predict( int64_t nowNs, int64_t targetNs )
{
    if( m_state != estimatorState::warming && m_state != estimatorState::ready )
        return fail( m_state );
    if( nowNs < 0 || targetNs < nowNs || ( m_samples && nowNs < m_lastReceiptNs ) )
        return fail( estimatorState::invalidRequest, false );
    if( !m_samples )
        return fail( estimatorState::warming );
    if( nowNs - m_lastTimeNs > m_config.m_maxAgeNs )
        return fail( estimatorState::stale );
    if( targetNs - m_lastTimeNs > m_config.m_maxPredictionNs )
        return fail( estimatorState::invalidRequest, false );
    if( m_lastTimeNs - m_firstTimeNs < m_warmupNs )
        return fail( estimatorState::warming );
    const Eigen::LDLT<Eigen::Matrix3d> solver( m_gram[1] );
    const double                       rcond = solver.rcond();
    if( solver.info() != Eigen::Success || !solver.isPositive() || solver.vectorD().minCoeff() <= 0 ||
        !std::isfinite( rcond ) || rcond < 1e-8 )
        return fail( estimatorState::numericalFault );
    const Eigen::Matrix<double, 3, 2> coefficients = solver.solve( m_moments[1] );
    if( !coefficients.allFinite() ||
        ( m_gram[1] * coefficients - m_moments[1] ).cwiseAbs().maxCoeff() > 1e-10 * m_moments[1].cwiseAbs().maxCoeff() )
        return fail( estimatorState::numericalFault );
    const double phase  = m_phase + m_omega * ( ( targetNs - m_lastTimeNs ) * 1e-9 );
    const double cosine = std::cos( phase ), sine = std::sin( phase );
    prediction   result;
    result.m_state        = estimatorState::ready;
    result.m_rcond        = rcond;
    result.m_samples      = m_samples;
    result.m_sequence     = m_last.m_sequence;
    result.m_generation   = m_last.m_generation;
    result.m_sampleTimeNs = m_lastTimeNs;
    result.m_targetTimeNs = targetNs;
    for( size_t axis = 0; axis < 2; ++axis )
    {
        const double a = coefficients( 1, axis ), b = coefficients( 2, axis );
        result.m_amplitude[axis] = std::hypot( a, b );
        result.m_phase[axis] =
            result.m_amplitude[axis] == 0 ? 0 : std::remainder( m_phase - std::atan2( b, a ), 6.28318530717958647692 );
        result.m_offset[axis]       = coefficients( 0, axis );
        result.m_acceleration[axis] = a * cosine + b * sine;
    }
    const double x = m_anglePerAccel * result.m_acceleration[0], y = m_anglePerAccel * result.m_acceleration[1];
    result.m_pointing = { m_cosRotation * x - m_sinRotation * y, m_sinRotation * x + m_cosRotation * y };
    for( size_t axis = 0; axis < 2; ++axis )
        if( !std::isfinite( result.m_amplitude[axis] ) || !std::isfinite( result.m_phase[axis] ) ||
            !std::isfinite( result.m_offset[axis] ) || !std::isfinite( result.m_acceleration[axis] ) ||
            !std::isfinite( result.m_pointing[axis] ) )
            return fail( estimatorState::numericalFault );
    m_state = estimatorState::ready;
    return result;
}

inline estimatorState harmonicEstimator::state() const
{
    return m_state;
}

inline uint64_t harmonicEstimator::samples() const
{
    return m_samples;
}

} // namespace accelFFCtrlDetail
} // namespace app
} // namespace MagAOX
#endif
