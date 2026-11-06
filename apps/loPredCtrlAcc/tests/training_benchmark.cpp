/** \file training_benchmark.cpp
 * \brief Offline per-worker inclusive wall/CPU phase profiles and numerical fingerprints, without scheduling waits.
 */
#include "async_test_data.hpp"
#include "../training_profile.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace
{
/// Print phase quantiles; collection/formatting are outside all measured scopes.
void quantiles( std::vector<double> values /**< [in] Microsecond values. */ )
{
    std::sort( values.begin(), values.end() );
    for( double q : { .5, .95, .99, .999, 1. } )
        std::cout << ',' << values[std::size_t( q * ( values.size() - 1 ) )];
}
/// Extend a diagnostic hash with raw scalar bytes, preserving signs and full floating-point representations.
void hash_bytes( std::uint64_t &hash /**< [in,out] FNV-1a state. */,
                 const void    *bytes /**< [in] Raw payload. */,
                 std::size_t    count /**< [in] Bytes. */ )
{
    for( std::size_t j = 0; j < count; ++j )
        hash = ( hash ^ static_cast<const unsigned char *>( bytes )[j] ) * 1099511628211ULL;
}
/// Hash all numerical outputs/diagnostics at every version, excluding nondeterministic timing metadata.
void hash_snapshot( std::uint64_t                &hash /**< [in,out] Replay digest. */,
                    const DDSPC::OfflineSnapshot &s /**< [in] Snapshot. */ )
{
    for( const auto *v : { &s.m_theta, &s.m_gain, &s.m_originState, &s.m_originProposal } )
        hash_bytes( hash, v->data(), v->size() * sizeof( float ) );
    const auto &status = s.m_info.m_status;
    hash_bytes( hash, &status.m_rmse, sizeof( status.m_rmse ) );
    for( auto value : { status.m_samples, status.m_updates, status.m_predictions } )
        hash_bytes( hash, &value, sizeof( value ) );
    for( bool value : { status.m_ready, status.m_training, status.m_predictionValid, status.m_dataValid } )
        hash_bytes( hash, &value, sizeof( value ) );
    for( int value : { status.m_rank, static_cast<int>( status.m_reason ) } )
        hash_bytes( hash, &value, sizeof( value ) );
}
} //namespace
/// Run fully excited two/ten-mode profiles serially; inclusive phase times isolate CPU from OS preemption.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Optional sample count. */ )
{
    using namespace DDSPC;
    const int     n = argc >= 2 ? std::stoi( argv[1] ) : 2000;
    std::ofstream trace;
    if( argc == 3 )
        trace.open( argv[2], std::ios::binary );
    if( n < 100 || ( argc == 3 && !trace ) || !DDSPCProfile::stamp( CLOCK_THREAD_CPUTIME_ID ) )
        return 2;
    const char *names[] = { "history",
                            "weights",
                            "covariance",
                            "qr_factor",
                            "qr_solve",
                            "rank",
                            "reference",
                            "diagnostics",
                            "snapshot",
                            "queue",
                            "processing" };
    std::cout << std::setprecision( 9 )
              << "modes,learner,model,phase,n,wall_p50_us,wall_p95_us,wall_p99_us,wall_p999_us,wall_max_us,cpu_p50_us,"
                 "cpu_p95_us,cpu_p99_us,cpu_p999_us,cpu_max_us,phase_calls\n";
    for( int modes : { 2, 10 } )
        for( bool qrd : { false, true } )
        {
            ShadowConfig cfg;
            cfg.m_modes  = modes;
            cfg.m_useQrd = qrd;
            cfg.m_prior  = .1f;
            cfg.m_lambda = .3f;
            OfflineAsyncShadow                             pair( cfg, 8, n + 512 );
            OfflineRecord                                  record( cfg );
            libXWCTest::loPredCtrlAccTest::AsyncTestInputs input( 8000, 1000000000 );
            std::array<OfflineSnapshot, 2> snapshots{ OfflineSnapshot( cfg, ShadowMode::Standard ),
                                                      OfflineSnapshot( cfg, ShadowMode::AccelAssisted ) };
            std::array<std::vector<std::array<DDSPCProfile::Sample, DDSPCProfile::Count>>, 2> timings;
            for( auto &t : timings )
                t.resize( n );
            std::array<std::uint64_t, 2> hashes{ 14695981039346656037ULL, 14695981039346656037ULL };
            for( int i = 0; i < n + 512; ++i )
            {
                input.next( i, record );
                if( !pair.submit( record, record.m_availableNs ) )
                    return 3;
                for( int lane = 0; lane < 2; ++lane )
                {
                    const auto mode       = lane ? ShadowMode::AccelAssisted : ShadowMode::Standard;
                    DDSPCProfile::samples = {};
                    DDSPCProfile::enabled = i >= 512;
                    if( !pair.manual_step( mode, record.m_availableNs ) )
                        return 4;
                    DDSPCProfile::enabled = false;
                    if( i >= 512 )
                        timings[lane][i - 512] = DDSPCProfile::samples;
                    if( !pair.read_snapshot( mode, snapshots[lane] ) )
                        return 5;
                    hash_snapshot( hashes[lane], snapshots[lane] );
                    if( trace.is_open() )
                    {
                        const auto                        &s = snapshots[lane];
                        const std::array<std::uint64_t, 8> header{ std::uint64_t( modes ),
                                                                   std::uint64_t( qrd ),
                                                                   std::uint64_t( lane ),
                                                                   std::uint64_t( i ),
                                                                   std::uint64_t( s.m_theta.size() ),
                                                                   std::uint64_t( s.m_gain.size() ),
                                                                   std::uint64_t( s.m_originState.size() ),
                                                                   std::uint64_t( s.m_originProposal.size() ) };
                        trace.write( reinterpret_cast<const char *>( header.data() ), sizeof( header ) );
                        for( const auto *v : { &s.m_theta, &s.m_gain, &s.m_originState, &s.m_originProposal } )
                            trace.write( reinterpret_cast<const char *>( v->data() ), v->size() * sizeof( float ) );
                        trace.write( reinterpret_cast<const char *>( &s.m_info.m_status.m_rmse ), sizeof( double ) );
                    }
                }
            }
            for( int lane = 0; lane < 2; ++lane )
            {
                std::cerr << modes << ',' << qrd << ',' << lane << ",numerical_hash=" << hashes[lane] << '\n';
                for( int phase = 0; phase < DDSPCProfile::Count; ++phase )
                {
                    std::vector<double> wall( n ), cpu( n );
                    std::uint64_t       calls = 0;
                    for( int i = 0; i < n; ++i )
                    {
                        wall[i] = timings[lane][i][phase].m_wallNs / 1000.;
                        cpu[i]  = timings[lane][i][phase].m_cpuNs / 1000.;
                        calls += timings[lane][i][phase].m_calls;
                    }
                    std::cout << modes << ',' << ( qrd ? "qr" : "rls" ) << ',' << ( lane ? "assisted" : "standard" )
                              << ',' << names[phase] << ',' << n;
                    quantiles( wall );
                    quantiles( cpu );
                    std::cout << ',' << calls << '\n';
                }
            }
        }
}
