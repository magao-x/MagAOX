/** \file vibration.cpp
 * \brief Causal synthetic modal plant and existing Legacy/reference/worker controllers, without application linkage.
 */
#include "vibration.hpp"
#include "stability.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <random>
#include <stdexcept>

namespace libXWCTest::loPredCtrlAccTest
{
namespace
{
using namespace DDSPC;
constexpr std::uint64_t period = 500000; ///< Virtual nanoseconds per WFS period; no wall-clock pacing.
constexpr double        pi     = 3.14159265358979323846;

/// Portable centered uniform innovations, with unit variance in the continuous limit.
double noise( std::mt19937 &rng /**< [in,out] Independent deterministic generator. */ )
{
    return ( int( rng() % 2000001 ) - 1000000 ) * ( std::sqrt( 3. ) / 1000000 );
}
/// Require explicit finite/stable synthetic parameters before allocating or simulating.
void validate( const VibrationConfig &c /**< [in] Immutable settings. */ )
{
    if( c.m_train < 100 || c.m_evaluate < 100 || c.m_history < 2 || c.m_future < 1 || c.m_accelHistory < 1 ||
        c.m_dmExtraFrames < 0 || c.m_wfsDelayFrames < 0 || c.m_sensorDelayFrames < 0 || c.m_publicationFrames < 0 ||
        c.m_dmExtraFrames > 20 || c.m_wfsDelayFrames > 20 || c.m_sensorDelayFrames > 20 ||
        !std::isfinite( c.m_dmScale ) || c.m_dmScale == 0 || !std::isfinite( c.m_controlGain ) ||
        !( c.m_dmPole >= 0 && c.m_dmPole < 1 ) || !( c.m_coherence >= 0 && c.m_coherence <= 1 ) ||
        !( c.m_limit > 0 && std::isfinite( c.m_limit ) ) || c.m_jitterNs >= period / 4 )
        throw std::invalid_argument( "Invalid synthetic vibration experiment" );
}
/// Synthetic modal mapping, WFS units per DM unit; additive positive plant sign.
Matrix mapping( double scale /**< [in] Explicit physical gain, relative to identification calibration. */ )
{
    Matrix g( 2, 2 );
    g << 1.f, .08f, -.05f, .9f;
    return scale * g;
}
/// Round the positive bound inward so float commands never exceed the declared double limit.
Matrix bounded( const Matrix &command /**< [in] Finite proposed vector. */,
                double        limit /**< [in] Positive synthetic absolute limit. */ )
{
    float bound = float( limit );
    if( double( bound ) > limit )
        bound = std::nextafter( bound, 0.f );
    return command.array().max( -bound ).min( bound ).matrix();
}
/// Independent canonical state oracle: newest-first y, actual past u, causally selected sensor histories.
Matrix state( const VibrationConfig             &c /**< [in] Dimensions. */,
              const VibrationSource             &s /**< [in] Immutable source. */,
              const std::vector<VibrationFrame> &frames /**< [in] This plant's actual recorded history. */,
              int                                origin /**< [in] Latest available WFS identity. */,
              bool                               assisted /**< [in] Include D features. */ )
{
    Matrix x   = Matrix::Zero( ( 2 * c.m_history - 1 ) * 2 + ( assisted ? c.m_accelHistory * 2 : 0 ), 1 );
    int    row = 0;
    for( int j = 0; j < c.m_history; ++j, row += 2 )
        if( origin >= j )
            x.middleRows( row, 2 ) = frames[origin - j].m_measured;
    for( int j = 0; j < c.m_history - 1; ++j, row += 2 )
        if( origin > j )
            x.middleRows( row, 2 ) = frames[origin - j - 1].m_applied;
    if( assisted )
        for( int j = 0; j < c.m_accelHistory; ++j, row += 2 )
            if( origin >= j && s.m_sensorIndex[origin - j] >= 0 )
                x.middleRows( row, 2 ) = s.m_acceleration[s.m_sensorIndex[origin - j]];
    return x;
}
/// Preserve origin coefficients until post-hoc scoring with actual future inputs, never using an optimal future input.
struct Forecast
{
    Matrix        m_base, m_c;    ///< Theta_known * state and C frozen at the actual decision.
    int           m_origin{ -1 }; ///< WFS origin; target sequence is [origin+F,...,origin+1].
    std::uint64_t m_epoch{ 0 };   ///< Supervisor episode prevents validation across resets.
};
} //namespace

std::uint64_t vibration_time( int frame )
{
    return 1000000000ULL + std::uint64_t( frame + 1 ) * period;
}

VibrationSource vibration_source( const VibrationConfig &c )
{
    validate( c );
    VibrationSource s;
    const int       n = c.m_train + c.m_evaluate + c.m_future + c.m_dmExtraFrames + 2;
    for( auto *v : { &s.m_disturbance, &s.m_acceleration, &s.m_wfsNoise, &s.m_exploration } )
        v->assign( n, Matrix::Zero( 2, 1 ) );
    s.m_availableNs.resize( n );
    s.m_sensorNs.resize( n );
    s.m_sensorAvailableNs.resize( n );
    s.m_sensorIndex.assign( n, -1 );
    std::mt19937 wfs( c.m_seed + 1 ), sensor( c.m_seed + 2 ), excitation( c.m_seed + 3 ), force( c.m_seed + 4 ),
        jitter( c.m_seed + 5 ), unrelated( c.m_seed + 6 );
    std::array<std::array<double, 2>, 2> previous{}, current{};
    std::array<double, 2>                broadPrevious{}, broad{};
    for( int mode = 0; mode < 2; ++mode )
        for( int peak = 0; peak < 2; ++peak )
        {
            const double w = 2 * pi * ( peak ? 56 : 28 ) / 2000, phase = .7 * mode + .4 * peak;
            const double amp     = ( peak ? .45 : .8 ) * ( mode ? .8 : 1 );
            previous[mode][peak] = amp * std::sin( phase - w );
            current[mode][peak]  = amp * std::sin( phase );
        }
    for( int i = 0; i < n; ++i )
    {
        s.m_availableNs[i] =
            vibration_time( i ) + c.m_wfsDelayFrames * period + ( c.m_jitterNs ? jitter() % ( c.m_jitterNs + 1 ) : 0 );
        const auto signedJitter =
            c.m_jitterNs ? std::int64_t( jitter() % ( 2 * c.m_jitterNs + 1 ) ) - std::int64_t( c.m_jitterNs ) : 0;
        // The disturbance state/force clock ticks 125 us before WFS; jitter affects the reported tag.
        const auto physicalSampleNs = vibration_time( i ) - 125000;
        s.m_sensorNs[i]             = physicalSampleNs + signedJitter;
        s.m_sensorAvailableNs[i] =
            std::max( physicalSampleNs, s.m_sensorNs[i] ) + 1000 + c.m_sensorDelayFrames * period;
        Matrix acceleration = Matrix::Zero( 2, 1 );
        for( int mode = 0; mode < 2; ++mode )
        {
            // Acceleration is computed from PRESENT state and PRESENT force; next position is then integrated.
            double position = 0, secondDifference = 0;
            for( int peak = 0; peak < 2; ++peak )
            {
                const double w = 2 * pi * ( peak ? 56 : 28 ) / 2000;
                const double a = 2 * ( std::cos( w ) - 1 ) * current[mode][peak];
                if( peak ? c.m_peak56 : c.m_peak28 )
                {
                    position += current[mode][peak];
                    secondDifference += a;
                }
                const double next    = 2 * current[mode][peak] - previous[mode][peak] + a;
                previous[mode][peak] = current[mode][peak];
                current[mode][peak]  = next;
            }
            const double innovation = .045 * noise( force );
            const double a          = -.12 * broad[mode] - ( broad[mode] - broadPrevious[mode] ) + innovation;
            if( c.m_broadband )
            {
                position += broad[mode];
                secondDifference += a;
            }
            const double nextBroad     = 2 * broad[mode] - broadPrevious[mode] + a;
            broadPrevious[mode]        = broad[mode];
            broad[mode]                = nextBroad;
            s.m_disturbance[i]( mode ) = position;
            acceleration( mode )    = secondDifference / .01; // Explicit synthetic acceleration normalization, not SI.
            s.m_wfsNoise[i]( mode ) = c.m_wfsNoise * noise( wfs );
            s.m_exploration[i]( mode ) = noise( excitation );
        }
        Matrix mix( 2, 2 );
        mix << 1.f, .2f, -.1f, 1.f;
        s.m_acceleration[i] = c.m_coherence * mix * acceleration;
        for( int mode = 0; mode < 2; ++mode )
            s.m_acceleration[i]( mode ) += std::sqrt( 1 - c.m_coherence * c.m_coherence ) * 6 * noise( unrelated ) +
                                           c.m_accelNoise * noise( sensor );
        for( int j = i; j >= 0; --j )
        {
            if( vibration_time( i ) - s.m_sensorNs[j] > 1500000 )
                break;
            const bool missing = j >= c.m_missingFrame && j < c.m_missingFrame + c.m_missingFrames;
            if( !missing && s.m_sensorAvailableNs[j] <= s.m_availableNs[i] && s.m_sensorNs[j] <= vibration_time( i ) )
            {
                s.m_sensorIndex[i] = j;
                break;
            }
        }
    }
    return s;
}

VibrationResult vibration_run( const VibrationConfig &c, const VibrationSource &s )
{
    const bool unchanged =
        !c.m_stability || ( !c.m_stability->m_neutralReturn && ( !c.m_stability->m_symmetricRls || c.m_qrd ) );
    if( unchanged && !c.m_freezeFrames )
    {
        auto reference = committed_vibration_run( c, s );
        for( int model = 0; model < 5; ++model )
            for( std::size_t i = 0; i < reference.m_frames[model].size(); ++i )
            {
                auto &f    = reference.m_frames[model][i];
                f.m_action = int(
                    i < std::size_t( c.m_train )
                        ? SimAction::Calibration
                        : ( model == 0 ? SimAction::Neutral : ( f.m_valid ? SimAction::Model : SimAction::Hold ) ) );
                f.m_correctionAccepted = i >= std::size_t( c.m_train ) && model > 0 && f.m_valid;
            }
        return reference;
    }
    validate( c );
    const int n = c.m_train + c.m_evaluate;
    for( const auto *series : { &s.m_disturbance, &s.m_acceleration, &s.m_wfsNoise, &s.m_exploration } )
    {
        if( series->size() < std::size_t( n + c.m_future + c.m_dmExtraFrames ) )
            throw std::invalid_argument( "Synthetic source is too short" );
        for( const auto &value : *series )
            if( value.rows() != 2 || value.cols() != 1 || !value.allFinite() )
                throw std::invalid_argument( "Invalid synthetic modal payload" );
    }
    if( s.m_availableNs.size() < std::size_t( n ) || s.m_sensorNs.size() < std::size_t( n ) ||
        s.m_sensorAvailableNs.size() < std::size_t( n ) || s.m_sensorIndex.size() < std::size_t( n ) )
        throw std::invalid_argument( "Missing synthetic timing provenance" );
    for( int i = 0; i < n; ++i )
    {
        const int a = s.m_sensorIndex[i];
        if( s.m_availableNs[i] < vibration_time( i ) || ( i && s.m_availableNs[i] <= s.m_availableNs[i - 1] ) ||
            a < -1 || a > i ||
            ( a >= 0 &&
              ( s.m_sensorNs[a] > vibration_time( i ) || s.m_sensorAvailableNs[a] < s.m_sensorNs[a] ||
                s.m_sensorAvailableNs[a] > s.m_availableNs[i] || vibration_time( i ) - s.m_sensorNs[a] > 1500000 ) ) )
            throw std::invalid_argument( "Ambiguous or noncausal synthetic timing" );
    }
    ShadowConfig cfg;
    cfg.m_modes        = 2;
    cfg.m_history      = c.m_history;
    cfg.m_future       = c.m_future;
    cfg.m_accelHistory = c.m_accelHistory;
    cfg.m_prior        = c.m_prior;
    cfg.m_lambda       = c.m_lambda;
    cfg.m_gamma        = c.m_gamma;
    cfg.m_useQrd       = c.m_qrd;
    VibrationResult result;
    for( int controller = 0; controller < 5; ++controller )
    {
        auto &frames = result.m_frames[controller];
        frames.resize( n );
        std::vector<Forecast> forecasts( n );
        // No device/app types are constructed. Workers are deterministic manual consumers, not OS threads.
        StabilityWorker      worker( cfg, c.m_capacity, c.m_stability && c.m_stability->m_symmetricRls && !c.m_qrd );
        PredictiveController legacy( 2, c.m_history, c.m_future, 0, c.m_gamma, c.m_lambda, c.m_prior, 0, 0 );
        legacy.use_qrd                          = c.m_qrd;
        const auto                     selected = controller == 4 ? ShadowMode::AccelAssisted : ShadowMode::Standard;
        OfflineSnapshot                snapshot( cfg, selected );
        OfflineRecord                  record( cfg );
        std::array<std::deque<int>, 2> admitted;
        Matrix                         dm = Matrix::Zero( 2, 1 ), held = dm;
        int                            received = -1, submitted = -1, legacyReceived = -1;
        std::uint64_t                  legacyUpdates = 0;
        const StabilityPolicy          defaultPolicy;
        const auto                    &policy = c.m_stability ? *c.m_stability : defaultPolicy;
        const bool         recoveryEnabled    = controller >= 3 && policy.m_neutralReturn && c.m_controlGain != 0;
        RecoverySupervisor recovery(
            c, ( c.m_future + 2 * c.m_history - 1 ) * 2 + ( controller == 4 ? c.m_accelHistory * 2 : 0 ), policy );
        std::size_t   matured      = 0;
        std::uint64_t historyEpoch = 0;
        for( int k = 0; k < n; ++k )
        {
            auto        &f   = frames[k];
            const auto   now = vibration_time( k );
            const Matrix g   = mapping( k < c.m_train ? 1 : c.m_dmScale );
            f.m_residual     = s.m_disturbance[k] + g * dm;
            f.m_measured     = f.m_residual + s.m_wfsNoise[k];
            while( received + 1 <= k && s.m_availableNs[received + 1] <= now )
                ++received;
            f.m_origin = received;
            if( k == c.m_resetFrame )
            {
                worker.reset( selected );
                legacy.reset();
                legacyUpdates = 0;
                ++historyEpoch;
            }
            if( recoveryEnabled )
                while( matured < std::size_t( k ) )
                {
                    const auto &past = forecasts[matured];
                    if( past.m_origin >= 0 && past.m_origin + c.m_future > received )
                        break;
                    if( past.m_origin >= 0 && past.m_epoch == historyEpoch )
                    {
                        Matrix u( 2 * c.m_future, 1 ), y( 2 * c.m_future, 1 );
                        for( int j = 0; j < c.m_future; ++j )
                        {
                            u.middleRows( 2 * j, 2 ) = frames[past.m_origin + c.m_future - j - 1].m_applied;
                            y.middleRows( 2 * j, 2 ) = frames[past.m_origin + c.m_future - j].m_measured;
                        }
                        recovery.observe( ( past.m_base + past.m_c * u - y ).squaredNorm() / ( 2 * c.m_future ),
                                          matured < std::size_t( c.m_train ) );
                    }
                    ++matured;
                }
            const bool stall   = k >= c.m_stallFrame && k < c.m_stallFrame + c.m_stallFrames;
            const auto service = [&]()
            {
                if( controller < 3 || stall )
                    return;
                for( int lane = 0; lane < 2; ++lane )
                    for( int budget = 0;
                         budget < 4 && !admitted[lane].empty() && admitted[lane].front() + c.m_publicationFrames <= k;
                         ++budget )
                    {
                        if( !worker.manual_step( lane ? ShadowMode::AccelAssisted : ShadowMode::Standard, now ) )
                            break;
                        admitted[lane].pop_front();
                    }
            };
            service();
            Matrix proposal = Matrix::Zero( 2, 1 );
            if( controller >= 3 && received >= std::max( c.m_history, c.m_accelHistory ) - 1 )
            {
                const Matrix      x = state( c, s, frames, received, controller == 4 );
                OfflinePrediction prediction;
                bool              featuresValid = true;
                if( controller == 4 )
                    for( int j = 0; j < c.m_accelHistory; ++j )
                        featuresValid = featuresValid && s.m_sensorIndex[received - j] >= 0;
                f.m_valid =
                    featuresValid &&
                    worker.predict(
                        selected, 1, received, vibration_time( received ), now, c.m_maxAgeNs, x, proposal, prediction );
                f.m_ready         = prediction.m_model.m_status.m_ready;
                f.m_version       = prediction.m_model.m_version;
                f.m_rank          = prediction.m_model.m_status.m_rank;
                f.m_ageNs         = prediction.m_ageNs;
                f.m_modelSequence = prediction.m_model.m_sequence;
                f.m_reason        = int( prediction.m_reason );
                if( f.m_valid && worker.read_snapshot( selected, snapshot ) )
                {
                    forecasts[k].m_origin = received;
                    forecasts[k].m_epoch  = historyEpoch;
                    forecasts[k].m_base   = snapshot.m_theta.leftCols( x.rows() ) * x;
                    forecasts[k].m_c      = snapshot.m_theta.rightCols( 2 * c.m_future );
                    f.m_thetaNorm         = snapshot.m_theta.norm();
                    if( controller == 4 )
                        f.m_dNorm =
                            snapshot.m_theta.middleCols( ( 2 * c.m_history - 1 ) * 2, 2 * c.m_accelHistory ).norm();
                }
            }
            if( controller == 2 && received > legacyReceived )
            {
                // Preserve the original requested-command history, scaled penalty and extra feedback (gain=0 here).
                Matrix        commandTheta;
                std::uint64_t commandVersion = legacyUpdates;
                for( int i = legacyReceived + 1; i <= received; ++i )
                {
                    commandTheta             = legacy.get_reference_prediction_matrix();
                    commandVersion           = legacyUpdates;
                    const Matrix exploration = k < c.m_train ? bounded( .3f * s.m_exploration[k], c.m_limit )
                                                             : ( .005f * s.m_exploration[k] ).eval();
                    proposal                 = legacy.calculate_command( frames[i].m_measured, exploration );
                    if( i >= c.m_future + c.m_history )
                    {
                        legacy.update_system();
                        ++legacyUpdates;
                        if( k >= c.m_train - 1 )
                            legacy.update_controller();
                    }
                }
                legacyReceived    = received;
                f.m_modelSequence = received > 0 ? received - 1 : 0;
                f.m_ageNs         = now - vibration_time( f.m_modelSequence );
                f.m_valid         = commandVersion >= std::uint64_t( ( c.m_future + 2 * c.m_history - 1 ) * 2 ) &&
                                    f.m_ageNs <= c.m_maxAgeNs && proposal.allFinite() && commandTheta.allFinite();
                f.m_ready         = f.m_valid;
                f.m_version       = commandVersion;
                if( f.m_valid )
                {
                    const Matrix &theta   = commandTheta;
                    f.m_thetaNorm         = theta.norm();
                    forecasts[k].m_origin = received;
                    forecasts[k].m_base =
                        theta.leftCols( ( 2 * c.m_history - 1 ) * 2 ) * state( c, s, frames, received, false );
                    forecasts[k].m_c = theta.rightCols( 2 * c.m_future );
                }
            }
            if( k < c.m_train )
                f.m_requested = .3f * s.m_exploration[k]; // Explicit common calibration excitation, not a prediction.
            else if( controller == 0 )
                f.m_requested = Matrix::Zero( 2, 1 );
            else if( controller == 1 )
            {
                Matrix futureDm = dm;
                for( int j = 0; j < c.m_dmExtraFrames; ++j )
                {
                    const int    source  = k + j - c.m_dmExtraFrames;
                    const Matrix pending = source >= 0 ? frames[source].m_requested : Matrix::Zero( 2, 1 ).eval();
                    futureDm             = c.m_dmPole * futureDm + ( 1 - c.m_dmPole ) * pending;
                }
                // Deliberately noncausal oracle: true future disturbance and true post-calibration response.
                f.m_requested = ( -g.inverse() * s.m_disturbance[k + c.m_dmExtraFrames + 1] - c.m_dmPole * futureDm ) /
                                ( 1 - c.m_dmPole );
                f.m_valid = f.m_ready = true;
            }
            else
            {
                f.m_requested = f.m_valid
                                    ? ( c.m_controlGain *
                                        ( proposal + ( controller == 2 ? Matrix::Zero( 2, 1 ).eval()
                                                                       : ( .005f * s.m_exploration[k] ).eval() ) ) )
                                          .eval()
                                    : held;
                if( !f.m_valid && !( f.m_requested.array() == held.array() ).all() )
                    ++result.m_invalidChanges[controller];
            }
            f.m_requested    = bounded( f.m_requested, c.m_limit );
            SimAction action = k < c.m_train ? SimAction::Calibration
                                             : ( controller == 0 ? SimAction::Neutral
                                                                 : ( f.m_valid ? SimAction::Model : SimAction::Hold ) );
            if( recoveryEnabled && k >= c.m_train )
            {
                bool features = received >= std::max( c.m_history, c.m_accelHistory ) - 1;
                if( controller == 4 && features )
                    for( int j = 0; j < c.m_accelHistory; ++j )
                        features = features && s.m_sensorIndex[received - j] >= 0;
                const bool healthy = features && worker.read_snapshot( selected, snapshot ) &&
                                     snapshot.m_info.m_measurementNs <= now &&
                                     now - snapshot.m_info.m_measurementNs <= c.m_maxAgeNs &&
                                     now - vibration_time( received ) <= c.m_maxAgeNs;
                // Retain Phase 10's rounded exploration intermediate before combining it with feedback.
                const Matrix feedback =
                    ( c.m_controlGain * ( proposal + ( .005f * s.m_exploration[k] ).eval() ) ).eval();
                f.m_requested = bounded( recovery.command( f.m_valid,
                                                           healthy,
                                                           feedback,
                                                           held,
                                                           bounded( .3f * s.m_exploration[k], c.m_limit ),
                                                           c.m_limit,
                                                           action ),
                                         c.m_limit );
                if( recovery.take_reset() )
                {
                    worker.reset( selected );
                    ++historyEpoch;
                }
            }
            if( recoveryEnabled && k < c.m_train )
                recovery.calibrate( f.m_requested );
            f.m_action   = int( action );
            f.m_recovery = int( recovery.state() );
            if( action == SimAction::Model && !f.m_valid )
                ++result.m_invalidChanges[controller];
            f.m_correctionAccepted  = action == SimAction::Model && f.m_valid;
            f.m_confidenceBound     = recovery.threshold();
            f.m_slewX               = recovery.step_bound()( 0 );
            f.m_slewY               = recovery.step_bound()( 1 );
            f.m_newCommand          = ( f.m_requested.array() != held.array() ).any();
            held                    = f.m_requested;
            const int appliedSource = k - ( k < c.m_train ? 0 : c.m_dmExtraFrames );
            f.m_applied = appliedSource >= 0 ? frames[appliedSource].m_requested : Matrix::Zero( 2, 1 ).eval();
            if( k >= c.m_freezeFrame && k < c.m_freezeFrame + c.m_freezeFrames )
                f.m_applied = k ? frames[k - 1].m_applied : Matrix::Zero( 2, 1 ).eval();
            dm = c.m_dmPole * dm + ( 1 - c.m_dmPole ) * f.m_applied;
            if( controller >= 3 )
            {
                while( submitted < received )
                {
                    const int i              = ++submitted;
                    record.m_data.m_index    = i;
                    record.m_measurementNs   = vibration_time( i );
                    record.m_exposureStartNs = record.m_measurementNs - period;
                    record.m_availableNs     = s.m_availableNs[i];
                    record.m_appliedNs = record.m_appliedAvailableNs = record.m_measurementNs;
                    record.m_data.m_residual                         = frames[i].m_measured;
                    record.m_data.m_appliedCommand                   = frames[i].m_applied;
                    record.m_data.m_wfsValid = record.m_data.m_appliedValid = record.m_data.m_timingValid = true;
                    const int a                = s.m_sensorIndex[i];
                    record.m_data.m_accelValid = a >= 0;
                    if( a >= 0 )
                    {
                        record.m_data.m_acceleration = s.m_acceleration[a];
                        record.m_accelNs             = s.m_sensorNs[a];
                        record.m_accelAvailableNs    = s.m_sensorAvailableNs[a];
                    }
                    const auto accepted = worker.submit_record( record, now );
                    if( accepted.m_standard )
                        admitted[0].push_back( i );
                    if( accepted.m_assisted )
                        admitted[1].push_back( i );
                }
                service();
            }
        }
        result.m_overflows[controller] = worker.stats( selected ).m_overflows;
        // Post-hoc conditional prediction score. Future applied inputs are read ONLY after the entire simulation.
        for( int k = 0; k < n; ++k )
        {
            const auto &forecast = forecasts[k];
            const int   i        = forecast.m_origin;
            if( i < 0 || i + c.m_future >= n )
                continue;
            Matrix future( 2 * c.m_future, 1 ), target( 2 * c.m_future, 1 );
            for( int j = 0; j < c.m_future; ++j )
            {
                future.middleRows( j * 2, 2 ) = frames[i + c.m_future - j - 1].m_applied;
                target.middleRows( j * 2, 2 ) = frames[i + c.m_future - j].m_measured;
            }
            frames[k].m_predictionSquared = ( forecast.m_base + forecast.m_c * future - target ).squaredNorm();
            frames[k].m_predictionCount   = 2 * c.m_future;
        }
    }
    return result;
}
} //namespace libXWCTest::loPredCtrlAccTest
