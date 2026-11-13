/** \file offline_ddspc.cpp
 * \brief Memory-only event timing, actual ADC/DDSPC processing and bounded fallback in a private synthetic plant.
 */
#include "offline_ddspc.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <stdexcept>

/// \cond OFFLINE_TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
namespace
{
using DDSPC::Matrix;
using DDSPC::ShadowMode;
constexpr std::int64_t originNs = 1000000000, wfsPeriodNs = 500000;

/// Owned offline snapshot delivered in publication order; no pointer to a producer or runtime exists.
struct AdcEvent : DDSPC::AdcSource
{
    DDSPC::AdcHeader      m_header;     ///< Stable one-plane publication metadata.
    DDSPC::AdcFrame       m_frame;      ///< Raw ADC values and original acquisition/publication timestamps.
    DDSPC::AdcClock       m_clock;      ///< Virtual receipt, with known REALTIME/common-clock mapping.
    std::array<double, 2> m_physical{}; ///< Latent plant input; never returned by the ADC transport interface.
    bool m_observed{ false }; ///< An unread depth-one sample is lost, while the physical disturbance persists.

    /// Copy the immutable simulated publication metadata.
    DDSPC::AdcHeader header() const override;

    /// Copy the sole raw plane, with the same bounds contract as the original synthetic source.
    DDSPC::AdcFrame frame( std::size_t index /**< [in] Sole plane zero. */ ) const override;

    /// Observe only this event's receipt time, not a later decision clock.
    DDSPC::AdcClock clock() const override;
};

DDSPC::AdcHeader AdcEvent::header() const
{
    return m_header;
}

DDSPC::AdcFrame AdcEvent::frame( std::size_t index ) const
{
    if( index )
        throw std::out_of_range( "Offline source has one plane" );
    return m_frame;
}

DDSPC::AdcClock AdcEvent::clock() const
{
    return m_clock;
}

/// Shared immutable exogenous experiment; each policy gets a separate plant, ADC capture and two-model learner.
struct Experiment
{
    std::vector<AdcEvent> m_adc; ///< Complete physical schedule, independently masked/delayed for sensor delivery.
    std::vector<std::array<double, 2>> m_noise, m_probe; ///< Common WFS noise and independent excitation draws.
};

/// Generate all exogenous events once; the decision loop exposes only events received by its deadline.
Experiment experiment( const AdcDdspcConfig &c /**< [in] Deterministic offline scenario. */ )
{
    Experiment         e;
    SyntheticAdcSource source( c.m_adc );
    auto               independentConfig = c.m_adc;
    independentConfig.m_seed += 1000;
    independentConfig.m_lowAmplitude = independentConfig.m_highAmplitude = { 0, 0 };
    independentConfig.m_forceNoise                                       = .15;
    SyntheticAdcSource independent( independentConfig );
    std::int64_t       previousReceipt = 0;
    const auto         end             = originNs + std::int64_t( c.m_train + c.m_evaluate ) * wfsPeriodNs;
    while( true )
    {
        AdcEvent event;
        event.m_observed    = source.advance();
        event.m_header      = source.header();
        event.m_frame       = source.frame( 0 );
        event.m_clock       = source.clock();
        event.m_physical    = source.physical();
        const auto acquired = event.m_frame.m_acquiredRealNs - ( event.m_clock.m_realNs - event.m_clock.m_monoNs );
        if( acquired >= end )
            break;
        if( c.m_uncorrelatedAdc )
        {
            independent.advance();
            event.m_frame.m_raw = independent.frame( 0 ).m_raw;
        }
        // FIFO transport: variable latency may cause head-of-line delay, never a false producer restart.
        const auto offset      = event.m_clock.m_realNs - event.m_clock.m_monoNs;
        event.m_clock.m_monoNs = std::max( previousReceipt + 1, event.m_clock.m_monoNs );
        event.m_clock.m_realNs = offset + event.m_clock.m_monoNs;
        previousReceipt        = event.m_clock.m_monoNs;
        e.m_adc.push_back( event );
    }
    std::mt19937_64                  wfs( c.m_adc.m_seed + 10 ), probe( c.m_adc.m_seed + 11 );
    std::normal_distribution<double> wfsNormal, probeNormal;
    for( int i = 0; i < c.m_train + c.m_evaluate; ++i )
    {
        std::array<double, 2> noise{}, excitation{};
        for( int axis = 0; axis < 2; ++axis )
        {
            noise[axis]      = c.m_wfsNoise * wfsNormal( wfs );
            excitation[axis] = probeNormal( probe );
        }
        e.m_noise.push_back( noise );
        e.m_probe.push_back( excitation );
    }
    return e;
}

/// Assemble the actual canonical state from this trajectory's available histories; never inspect future records.
Matrix state( const DDSPC::ShadowConfig &c /**< [in] Fixed dimensions. */,
              const std::deque<Matrix>  &y /**< [in] Latest WFS first. */,
              const std::deque<Matrix>  &u /**< [in] Latest consumed input first. */,
              const std::deque<Matrix>  &a /**< [in] Latest decoded feature first. */,
              bool                       assisted /**< [in] Include ADC feature history. */ )
{
    Matrix result = Matrix::Zero( ( 2 * c.m_history - 1 ) * 2 + ( assisted ? 2 * c.m_accelHistory : 0 ), 1 );
    int    row    = 0;
    for( int i = 0; i < c.m_history; ++i, row += 2 )
    {
        if( std::size_t( i ) < y.size() )
        {
            result.middleRows( row, 2 ) = y[i];
        }
    }
    for( int i = 0; i < c.m_history - 1; ++i, row += 2 )
    {
        if( std::size_t( i ) < u.size() )
        {
            result.middleRows( row, 2 ) = u[i];
        }
    }
    if( assisted )
    {
        for( int i = 0; i < c.m_accelHistory; ++i, row += 2 )
        {
            if( std::size_t( i ) < a.size() )
            {
                result.middleRows( row, 2 ) = a[i];
            }
        }
    }
    return result;
}

/// Evaluate a fixed requested policy against exactly the shared physical/noise/delivery schedule.
AdcDdspcTrajectory run( const AdcDdspcConfig &c /**< [in] Fixed synthetic configuration. */,
                        const Experiment     &e /**< [in] Shared immutable exogenous schedule. */,
                        int selected /**< [in] -1 open loop, 0 standard requested, 1 assisted requested. */ )
{
    DDSPC::AdcCapture capture( c.m_adc.m_calibration );
    capture.attached( 1, 1 );
    DDSPC::ShadowController controller( c.m_controller );
    controller.training( ShadowMode::Standard, true );
    controller.training( ShadowMode::AccelAssisted, true );
    const auto         mode = selected == 1 ? ShadowMode::AccelAssisted : ShadowMode::Standard;
    OfflinePolicy      policy( selected == 1 ? OfflineMode::Assisted : OfflineMode::Standard, c.m_policy );
    std::deque<Matrix> ys, us, as;
    std::deque<bool>   validHistory;
    Matrix             disturbance = Matrix::Zero( 2, 1 ), held = Matrix::Zero( 2, 1 ), physical = Matrix::Zero( 2, 1 );
    std::size_t        delivered = 0, forced = 0;
    std::uint64_t      previousPredictions = 0;
    double             previousSse = 0, totalSse = 0;
    std::uint64_t      fallbackRun = 0;
    AdcDdspcTrajectory result;
    for( int i = 0; i < c.m_train + c.m_evaluate; ++i )
    {
        const auto decision = originNs + std::int64_t( i + 1 ) * wfsPeriodNs;
        while( delivered < e.m_adc.size() && e.m_adc[delivered].m_clock.m_monoNs <= decision )
        {
            const auto &event = e.m_adc[delivered++];
            if( event.m_observed )
                capture.drain( event );
        }
        capture.tick( decision );
        const auto feature = acceleration_window( capture.samples(), decision, c.m_features );
        const bool valid   = feature.m_reason == FeatureReason::Ready;
        const bool common  = !( i >= c.m_wfsInvalidStart && i - c.m_wfsInvalidStart < c.m_wfsInvalidCount );
        // y_i consumes only u_(i-1); the physical window below will affect d_(i+1).
        const Matrix residual = disturbance + float( c.m_dmGain ) * held;
        Matrix       measured = residual;
        Matrix       probe( 2, 1 );
        for( int axis = 0; axis < 2; ++axis )
        {
            measured( axis ) += float( e.m_noise[i][axis] );
            probe( axis ) = float( e.m_probe[i][axis] * ( i < c.m_train ? c.m_trainProbe : c.m_evalProbe ) );
        }
        ys.push_front( measured );
        as.push_front( feature.m_value );
        validHistory.push_front( valid && common );
        if( ys.size() > std::size_t( c.m_controller.m_history ) )
            ys.pop_back();
        if( as.size() > std::size_t( c.m_controller.m_accelHistory ) )
        {
            as.pop_back();
            validHistory.pop_back();
        }
        const bool featuresValid = validHistory.size() == std::size_t( c.m_controller.m_accelHistory ) &&
                                   std::all_of( validHistory.begin(), validHistory.end(), []( bool v ) { return v; } );
        ControlEvidence evidence;
        evidence.m_commonValid   = common;
        evidence.m_featuresValid = featuresValid;
        evidence.m_featureReason = feature.m_reason;
        if( i >= c.m_train && selected >= 0 && common )
        {
            evidence.m_standardReady = controller.predict_cached(
                ShadowMode::Standard, i, 1, state( c.m_controller, ys, us, as, false ), evidence.m_standard );
            if( featuresValid )
                evidence.m_assistedReady = controller.predict_cached(
                    ShadowMode::AccelAssisted, i, 1, state( c.m_controller, ys, us, as, true ), evidence.m_assisted );
        }
        ControlDecision control;
        control.m_requested = selected == 1 ? OfflineMode::Assisted : OfflineMode::Standard;
        if( i < c.m_train )
            control.m_command = bounded_command( probe, held, c.m_policy );
        else if( selected < 0 )
            control.m_command = bounded_command( Matrix::Zero( 2, 1 ), held, c.m_policy );
        else
            control = policy.decide( evidence, held, probe );
        if( !residual.allFinite() || !control.m_command.allFinite() )
            throw std::runtime_error( "Nonfinite synthetic plant" );
        held = control.m_command; // Both learners receive precisely this consumed input, after all constraints.
        DDSPC::ShadowRecord record;
        record.m_index          = i;
        record.m_residual       = measured;
        record.m_appliedCommand = held;
        record.m_acceleration   = feature.m_value;
        record.m_wfsValid       = common;
        record.m_appliedValid = record.m_timingValid = true;
        record.m_accelValid                          = valid;
        if( selected >= 0 )
            controller.process( record );
        us.push_front( held );
        if( us.size() > std::size_t( c.m_controller.m_history - 1 ) )
            us.pop_back();
        if( i >= c.m_train )
        {
            for( int axis = 0; axis < 2; ++axis )
            {
                result.m_residual[axis].push_back( residual( axis ) );
                result.m_commandRms += double( held( axis ) ) * held( axis ) / ( 2 * c.m_evaluate );
            }
            result.m_maxCommand = std::max( result.m_maxCommand, double( held.cwiseAbs().maxCoeff() ) );
            result.m_commands += selected >= 0 && control.m_effective != OfflineMode::Safe;
            result.m_blocked += !valid;
            result.m_fallbackFrames += selected == 1 && control.m_effective == OfflineMode::Standard;
            result.m_safeFrames += selected >= 0 && control.m_effective == OfflineMode::Safe;
            fallbackRun = selected == 1 && control.m_effective != OfflineMode::Assisted ? fallbackRun + 1 : 0;
            result.m_longestFallback = std::max( result.m_longestFallback, fallbackRun );
            if( control.m_switched )
                result.m_maxSwitchStep = std::max( result.m_maxSwitchStep, control.m_step );
            result.m_gateViolations += control.m_effective == OfflineMode::Assisted &&
                                       ( !featuresValid || !common || !evidence.m_assistedReady );
            if( selected >= 0 )
                result.m_decisions.push_back( { i,
                                                decision,
                                                feature,
                                                control,
                                                common,
                                                featuresValid,
                                                evidence.m_standardReady,
                                                evidence.m_assistedReady } );
        }
        if( selected >= 0 )
        {
            const auto status     = controller.status( mode );
            const bool firstReady = status.m_ready && result.m_firstReady < 0;
            result.m_updates      = status.m_updates;
            // Accumulate the actual conditional forecast score across invalidated/recovered segments.
            const auto   n   = status.m_predictions;
            const double sse = n ? status.m_rmse * status.m_rmse * n : 0;
            if( n < previousPredictions )
            {
                previousPredictions = 0;
                previousSse         = 0;
            }
            if( n > previousPredictions )
            {
                totalSse += std::max( 0., sse - previousSse );
                result.m_predictions += n - previousPredictions;
                result.m_predictionRmse = std::sqrt( totalSse / result.m_predictions );
            }
            previousPredictions = n;
            previousSse         = sse;
            result.m_faults += status.m_reason == DDSPC::ShadowReason::NumericalFault;
            if( status.m_ready )
            {
                if( firstReady )
                    result.m_firstReady = i;
                result.m_lastReady = i;
            }
            if( i % 100 == 0 || firstReady || i + 1 == c.m_train + c.m_evaluate )
                result.m_learning.push_back(
                    { i, status, controller.prediction_matrix( mode ).cast<double>().norm() } );
        }
        // Integrate the common latent physical input by acquisition time, independent of sensor loss/delivery.
        Matrix force = Matrix::Zero( 2, 1 );
        auto   at    = decision - wfsPeriodNs;
        while( forced < e.m_adc.size() )
        {
            const auto &event    = e.m_adc[forced];
            const auto  acquired = event.m_frame.m_acquiredRealNs - ( event.m_clock.m_realNs - event.m_clock.m_monoNs );
            if( acquired >= decision )
                break;
            const auto until = std::max( at, acquired );
            force += float( double( until - at ) / wfsPeriodNs ) * physical;
            for( int axis = 0; axis < 2; ++axis )
                physical( axis ) = float( event.m_physical[axis] );
            at = until;
            ++forced;
        }
        force += float( double( decision - at ) / wfsPeriodNs ) * physical;
        Matrix coupled( 2, 1 );
        coupled << force( 0 ) + .15f * force( 1 ), -.1f * force( 0 ) + force( 1 );
        disturbance = float( c.m_pole ) * disturbance + float( c.m_coupling ) * coupled;
    }
    result.m_commandRms = std::sqrt( result.m_commandRms );
    for( const auto &axis : result.m_residual )
    {
        const double mean = std::accumulate( axis.begin(), axis.end(), 0. ) / axis.size();
        for( double value : axis )
            result.m_variance += ( value - mean ) * ( value - mean ) / ( 2 * axis.size() );
        const auto   spectrum = periodogram( axis, 2000, 60 );
        const double df       = 2000. / axis.size();
        for( std::size_t k = 0; k < spectrum.size(); ++k )
        {
            if( k * df >= 28 && k * df <= 29 )
                result.m_power28 += spectrum[k] * df / 2;
            if( k * df >= 54 && k * df <= 58 )
                result.m_power56 += spectrum[k] * df / 2;
        }
    }
    if( selected >= 0 )
    {
        result.m_thetaNorm = controller.prediction_matrix( mode ).cast<double>().norm();
        if( !std::isfinite( result.m_thetaNorm ) )
            throw std::runtime_error( "Nonfinite actual DDSPC coefficients" );
    }
    result.m_adcStatus = capture.status();
    return result;
}
} // namespace

AdcDdspcConfig::AdcDdspcConfig()
{
    m_controller.m_modes           = 2;
    m_controller.m_history         = 4;
    m_controller.m_future          = 3;
    m_controller.m_accelHistory    = 3;
    m_controller.m_lambda          = .1f;
    m_controller.m_prior           = .001f;
    m_controller.m_gamma           = .9995f;
    m_controller.m_useQrd          = true;
    m_adc.m_calibration.m_capacity = 64;
}

AdcDdspcResult adc_ddspc_run( const AdcDdspcConfig &c )
{
    if( c.m_train < 100 || c.m_evaluate < 256 || c.m_train > 100000 || c.m_evaluate > 100000 ||
        c.m_controller.m_modes != 2 || c.m_controller.m_accelChannels != 2 || c.m_controller.m_history < 2 ||
        c.m_controller.m_history > 32 || c.m_controller.m_future < 1 || c.m_controller.m_future > 16 ||
        c.m_controller.m_accelHistory < 1 || c.m_controller.m_accelHistory > 32 || !std::isfinite( c.m_coupling ) ||
        std::abs( c.m_coupling ) > 100 || !std::isfinite( c.m_pole ) || std::abs( c.m_pole ) >= 1 ||
        !std::isfinite( c.m_dmGain ) || c.m_dmGain == 0 || !std::isfinite( c.m_wfsNoise ) || c.m_wfsNoise < 0 ||
        !std::isfinite( c.m_trainProbe ) || c.m_trainProbe < 0 || !std::isfinite( c.m_evalProbe ) ||
        c.m_evalProbe < 0 || c.m_wfsInvalidStart < 0 || c.m_wfsInvalidCount < 0 )
        throw std::invalid_argument( "Invalid offline ADC-DDSPC scenario" );
    // Validate before allocating the shared experiment; neither validator has external effects.
    OfflinePolicy validatePolicy( OfflineMode::Assisted, c.m_policy );
    acceleration_window( {}, originNs, c.m_features );
    const auto     shared = experiment( c );
    AdcDdspcResult result;
    for( int selected = -1; selected <= 1; ++selected )
        result.m_runs[selected + 1] = run( c, shared, selected );
    return result;
}
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
