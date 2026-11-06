/** \file async_benchmark.cpp
 * \brief Paced 2-kHz offline producer and independent training workers with 4/8-kHz synthetic sensor events.
 */
#include "async_test_data.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <thread>

namespace
{
using namespace DDSPC;
using libXWCTest::loPredCtrlAccTest::AsyncTestInputs;
constexpr std::uint64_t period = 500000; ///< The simulated WFS period, not an instrument timing measurement.
constexpr int           warm   = 512;    ///< Same warmup count as the Phase 7 component benchmark.

/// Per-frame fixed telemetry; vectors containing these records are allocated before worker startup.
struct FrameTiming
{
    double                m_frameUs{ 0 }, m_lateUs{ 0 }; ///< Full producer execution and arrival scheduling lateness.
    bool                  m_deadline{ false };           ///< End exceeds its absolute scheduled 500-us window.
    std::array<double, 2> m_predictionUs{}, m_ageUs{},
        m_delta{};                                  ///< Handoff+prediction time, source age and sync-fit difference.
    std::array<std::uint64_t, 2>           m_lag{}; ///< Validation-record lag in sequence units, within an epoch only.
    std::array<OfflinePredictionReason, 2> m_reason{}; ///< Exact rejection/availability diagnostics.
};

/// Apply deterministic stream gaps, sensor/timing loss and restart; no calculated command replaces the applied input.
void inject( int            frame /**< [in] Producer ordinal. */,
             int            samples /**< [in] Measured frames. */,
             bool           stress /**< [in] Enable faults. */,
             OfflineRecord &record /**< [in,out] Synthetic payload. */ )
{
    if( !stress )
        return;
    const int restart = warm + samples * 4 / 5;
    if( frame >= warm + samples * 3 / 5 )
        record.m_data.m_index += 3;
    if( frame >= restart )
    {
        record.m_streamEpoch  = 2;
        record.m_data.m_index = frame - restart;
    }
    if( frame == warm + samples / 2 + 20 )
        record.m_data.m_accelValid = false;
    if( frame == warm + samples * 2 / 3 )
        record.m_data.m_timingValid = false;
}

/// Print the original sorted-index p50/p95/p99/p99.9/max rule; empty cohorts are explicitly zero-count.
void quantiles( std::vector<double> values /**< [in] Microsecond samples, copied outside timed execution. */ )
{
    std::sort( values.begin(), values.end() );
    for( double q : { .5, .95, .99, .999, 1. } )
        std::cout << ',' << ( values.empty() ? 0 : values[std::size_t( q * ( values.size() - 1 ) )] );
}

/// Run one complete paced producer/two-worker experiment; precomputed synchronous output is outside timing.
bool run( int  modes /**< [in] Controlled modes. */,
          bool qrd /**< [in] Learner. */,
          int  sensorHz /**< [in] Synthetic rate. */,
          bool stress /**< [in] Fault injection. */,
          int  samples /**< [in] Measured producer frames. */ )
{
    ShadowConfig cfg;
    cfg.m_modes                              = modes;
    cfg.m_useQrd                             = qrd;
    cfg.m_prior                              = .1f;
    cfg.m_lambda                             = .3f;
    const int                          total = warm + samples;
    std::vector<std::array<Matrix, 2>> reference( total );
    ShadowController                   synchronous( cfg );
    synchronous.training( ShadowMode::Standard, true );
    synchronous.training( ShadowMode::AccelAssisted, true );
    AsyncTestInputs oracleInput( sensorHz, 1000000000 );
    OfflineRecord   record( cfg );
    std::uint64_t   lastEpoch = 1;
    for( int i = 0; i < total; ++i )
    {
        oracleInput.next( i, record );
        inject( i, samples, stress, record );
        if( record.m_streamEpoch != lastEpoch )
        {
            synchronous.invalidate( ShadowRecord{} );
            lastEpoch = record.m_streamEpoch;
        }
        if( stress && i == warm + samples / 2 )
            synchronous.reset( ShadowMode::Standard );
        if( stress && i == warm + samples * 3 / 4 )
            synchronous.reset( ShadowMode::AccelAssisted );
        synchronous.process( record.m_data );
        for( int lane = 0; lane < 2; ++lane )
        {
            const auto mode    = lane ? ShadowMode::AccelAssisted : ShadowMode::Standard;
            reference[i][lane] = Matrix::Zero( modes, 1 );
            if( synchronous.status( mode ).m_ready )
                reference[i][lane] = synchronous.proposal( mode );
        }
    }
    // All payloads, histories and telemetry are fixed before the two worker threads are created.
    OfflineAsyncShadow                                   async( cfg, stress ? 8 : 32, total );
    std::array<std::unique_ptr<PredictiveController>, 2> history;
    std::array<Matrix, 2>                                states, outputs;
    std::array<std::uint64_t, 2>                         counts{};
    std::array<bool, 2>                                  stateReady{};
    for( int lane = 0; lane < 2; ++lane )
    {
        history[lane] = std::make_unique<PredictiveController>(
            modes, 5, 3, 0, 1, .3f, .1f, lane ? 2 : 0, lane ? 20 : 0, HistoryMode::Causal );
        states[lane]  = Matrix::Zero( 9 * modes + ( lane ? 40 : 0 ), 1 );
        outputs[lane] = Matrix::Zero( modes, 1 );
    }
    Matrix                   empty;
    std::vector<FrameTiming> frames( samples );
    std::uint64_t            eigenRequests = 0, cppRequests = 0;
    std::uint64_t            previousSequence = 0, previousEpoch = 0;
    bool                     previousValid = false;
    async.start();
    const auto      origin = offline_now_ns() + 5000000;
    AsyncTestInputs input( sensorHz, origin );
    for( int i = 0; i < total; ++i )
    {
        const auto scheduled = origin + ( i + 1 ) * period;
        std::this_thread::sleep_until( std::chrono::steady_clock::time_point( std::chrono::nanoseconds( scheduled ) ) );
        if( stress && i >= warm && i % 127 == 0 )
            std::this_thread::sleep_for( std::chrono::microseconds( 90 ) );
        DDSPCBenchmark::eigenRequests = DDSPCBenchmark::cppRequests = 0;
        DDSPCBenchmark::enabled                                     = true;
        const auto begin                                            = offline_now_ns();
        input.next( i, record );
        inject( i, samples, stress, record );
        if( stress && i == warm + samples / 4 )
        {
            async.stall_until( ShadowMode::Standard, begin + 8000000 );
            async.stall_until( ShadowMode::AccelAssisted, begin + 16000000 );
        }
        if( stress && i == warm + samples / 2 )
            async.reset( ShadowMode::Standard );
        if( stress && i == warm + samples * 3 / 4 )
            async.reset( ShadowMode::AccelAssisted );
        const bool common     = record.m_data.m_timingValid;
        const bool contiguous = previousValid && common && record.m_streamEpoch == previousEpoch &&
                                record.m_data.m_index == previousSequence + 1;
        if( !contiguous )
            for( int lane = 0; lane < 2; ++lane )
            {
                history[lane]->reset_buffers();
                counts[lane]     = 0;
                stateReady[lane] = false;
            }
        if( common )
            for( int lane = 0; lane < 2; ++lane )
            {
                if( lane && !record.m_data.m_accelValid )
                {
                    history[lane]->reset_buffers();
                    counts[lane]     = 0;
                    stateReady[lane] = false;
                    continue;
                }
                history[lane]->append_sample( counts[lane]++,
                                              record.m_data.m_residual,
                                              record.m_data.m_appliedCommand,
                                              lane ? record.m_data.m_acceleration : empty );
                stateReady[lane] = counts[lane] >= std::uint64_t( lane ? 20 : 5 );
                if( stateReady[lane] )
                    history[lane]->get_prediction_state( states[lane] );
            }
        previousSequence = record.m_data.m_index;
        previousEpoch    = record.m_streamEpoch;
        previousValid    = common;
        async.submit( record, offline_now_ns() );
        FrameTiming timing;
        timing.m_lateUs = double( begin - scheduled ) / 1000;
        for( int lane = 0; lane < 2; ++lane )
        {
            const auto        mode = lane ? ShadowMode::AccelAssisted : ShadowMode::Standard;
            OfflinePrediction result;
            const auto        before    = offline_now_ns();
            const bool        valid     = stateReady[lane] && async.predict( mode,
                                                                             record.m_streamEpoch,
                                                                             record.m_data.m_index,
                                                                             record.m_measurementNs,
                                                                             before,
                                                                             2000000,
                                                                             states[lane],
                                                                             outputs[lane],
                                                                             result );
            timing.m_predictionUs[lane] = double( offline_now_ns() - before ) / 1000;
            timing.m_reason[lane]       = result.m_reason;
            if( result.m_model.m_publication && before >= result.m_model.m_measurementNs )
                timing.m_ageUs[lane] = double( before - result.m_model.m_measurementNs ) / 1000;
            if( result.m_model.m_streamEpoch == record.m_streamEpoch &&
                result.m_model.m_sequence <= record.m_data.m_index )
                timing.m_lag[lane] = record.m_data.m_index - result.m_model.m_sequence;
            if( valid )
                timing.m_delta[lane] = ( outputs[lane] - reference[i][lane] ).cwiseAbs().maxCoeff();
        }
        const auto end          = offline_now_ns();
        timing.m_frameUs        = double( end - begin ) / 1000;
        timing.m_deadline       = end > scheduled + period;
        DDSPCBenchmark::enabled = false;
        if( i >= warm )
        {
            frames[i - warm] = timing;
            eigenRequests += DDSPCBenchmark::eigenRequests;
            cppRequests += DDSPCBenchmark::cppRequests;
        }
    }
    const auto drainUntil = offline_now_ns() + 2000000000;
    while( ( async.stats( ShadowMode::Standard ).m_depth || async.stats( ShadowMode::AccelAssisted ).m_depth ) &&
           offline_now_ns() < drainUntil )
        std::this_thread::sleep_for( std::chrono::microseconds( 50 ) ); // Shutdown coordinator, outside frame timing.
    async.stop();
    for( int lane = 0; lane < 2; ++lane )
    {
        const auto          mode = lane ? ShadowMode::AccelAssisted : ShadowMode::Standard;
        std::vector<double> prediction, execution, lateness, training, publication, recordPublication, age;
        std::uint64_t valid = 0, stale = 0, busy = 0, unready = 0, invalid = 0, executionMiss = 0, deadlineMiss = 0,
                      maxLag = 0, workerEigen = 0, workerCpp = 0;
        double        maxDelta = 0;
        for( const auto &f : frames )
        {
            prediction.push_back( f.m_predictionUs[lane] );
            execution.push_back( f.m_frameUs );
            lateness.push_back( f.m_lateUs );
            if( f.m_ageUs[lane] )
                age.push_back( f.m_ageUs[lane] );
            valid += f.m_reason[lane] == OfflinePredictionReason::Valid;
            stale += f.m_reason[lane] == OfflinePredictionReason::Stale;
            busy += f.m_reason[lane] == OfflinePredictionReason::Busy;
            unready += f.m_reason[lane] == OfflinePredictionReason::Unready;
            invalid += f.m_reason[lane] == OfflinePredictionReason::Invalidated;
            executionMiss += f.m_frameUs > 500;
            deadlineMiss += f.m_deadline;
            maxLag   = std::max( maxLag, f.m_lag[lane] );
            maxDelta = std::max( maxDelta, f.m_delta[lane] );
        }
        for( std::size_t j = 0; j < async.event_count( mode ); ++j )
        {
            const auto &e = async.events( mode )[j];
            if( e.m_measurementNs < origin + ( warm + 1 ) * period )
                continue;
            training.push_back( e.m_trainingNs / 1000. );
            publication.push_back( e.m_publicationNs / 1000. );
            recordPublication.push_back( ( e.m_completedNs - e.m_measurementNs ) / 1000. );
            workerEigen += e.m_eigenRequests;
            workerCpp += e.m_cppRequests;
        }
        const auto stats = async.stats( mode );
        std::cout << modes << ',' << ( qrd ? "qr" : "rls" ) << ',' << sensorHz << ','
                  << ( stress ? "stress" : "normal" ) << ',' << ( lane ? "assisted" : "standard" ) << ',' << samples;
        quantiles( prediction );
        quantiles( execution );
        quantiles( lateness );
        quantiles( training );
        quantiles( publication );
        quantiles( recordPublication );
        quantiles( age );
        std::cout << ',' << training.size() << ',' << valid << ',' << stale << ',' << busy << ',' << unready << ','
                  << invalid << ',' << stats.m_highWater << ',' << async.drops() << ',' << async.overflows() << ','
                  << async.drops() - async.overflows() << ',' << stats.m_discarded << ',' << stats.m_depth << ','
                  << stats.m_publishMisses << ',' << stats.m_traceDrops << ',' << maxLag << ',' << executionMiss << ','
                  << deadlineMiss << ',' << eigenRequests << ',' << cppRequests << ',' << workerEigen << ','
                  << workerCpp << ',' << maxDelta << ',' << input.sensor_events() << '\n';
    }
    return eigenRequests == 0 && cppRequests == 0;
}
} // namespace

/// Run offline paced scenarios; no hostname, device, application or stream is contacted.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Optional sample count, at least 100. */ )
{
    const int samples = argc == 2 ? std::stoi( argv[1] ) : 2000;
    if( samples < 100 || samples > 10000 )
        return 2;
    // Preserve the allocation instrumentation self-test before claiming any zero-allocation measurements.
    DDSPCBenchmark::enabled       = true;
    DDSPCBenchmark::eigenRequests = DDSPCBenchmark::cppRequests = 0;
    Matrix known                                                = Matrix::Ones( 256, 1 );
    void  *request                                              = ::operator new( 73 );
    ::operator delete( request );
    DDSPCBenchmark::enabled = false;
    if( DDSPCBenchmark::eigenRequests != 1 || DDSPCBenchmark::cppRequests != 1 || known.sum() != 256 )
        return 3;
    std::cerr
        << "compiler=" << __VERSION__ << " Eigen=" << EIGEN_WORLD_VERSION << '.' << EIGEN_MAJOR_VERSION << '.'
        << EIGEN_MINOR_VERSION
        << " period_ns=500000 warmup=512 max_model_age_ns=2000000 clock=steady_clock no_affinity no_RT_priority\n";
    std::cout << std::setprecision( 9 ) << "modes,learner,sensor_hz,scenario,model,n";
    for( auto metric :
         { "prediction", "frame", "arrival_late", "training", "publication", "record_publication", "age" } )
        for( auto q : { "p50", "p95", "p99", "p999", "max" } )
            std::cout << ',' << metric << '_' << q << "_us";
    std::cout
        << ",trained_n,valid,stale,busy,unready,invalidated,queue_high,drops,queue_overflows,invalid_inputs,discarded,"
           "pending,publish_misses,trace_drops,max_sequence_lag,execution_misses,absolute_deadline_misses,frontend_"
           "eigen,frontend_cpp,worker_eigen,worker_cpp,max_sync_proposal_difference,sensor_events\n";
    for( int modes : { 2, 10 } )
        for( bool qrd : { false, true } )
            for( int rate : { 4000, 8000 } )
                for( bool stress : { false, true } )
                    if( !run( modes, qrd, rate, stress, samples ) )
                        return 4;
}
