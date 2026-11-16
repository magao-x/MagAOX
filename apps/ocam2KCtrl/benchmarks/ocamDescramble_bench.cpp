/** \file ocamDescramble_bench.cpp
 * \brief Compare OCAM descramble implementations using synthetic frames and a real mapping table, without hardware.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sched.h>

#include "../ocamDescrambler.hpp"
#include "../../../libMagAOX/common/streamCopy.hpp"

#ifndef OCAM_BENCH_CXXFLAGS
    #define OCAM_BENCH_CXXFLAGS "not recorded"
#endif

#ifndef OCAM_BENCH_CFLAGS
    #define OCAM_BENCH_CFLAGS "not recorded"
#endif

namespace
{

/// Comparison paths; the memcpy path is a calibration and does not descramble.
enum class Kernel
{
    sdk,
    mapped,
    optimized,
    copy
};

/// Names corresponding to the four comparison kernels.
constexpr std::array<const char *, 4> kernelNames{ "sdk", "mapped", "optimized", "memcpy" };

/// Consume output outside measured intervals so each comparison remains observable.
uint64_t outputChecksum{ 0 };

/// Return monotonic nanoseconds for an individual sample.
int64_t nowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::steady_clock::now().time_since_epoch() )
        .count();
}

/// Prevent the compiler from discarding output stores or moving them across a timing boundary.
void observe( const void *pixels /**< [in] Output whose stores must remain observable. */ )
{
    asm volatile( "" : : "g"( pixels ) : "memory" );
}

/// Return a positive bounded command-line integer, rejecting suffixes and negative values.
size_t positive( const char *text /**< [in] Decimal argument. */, size_t maximum /**< [in] Largest accepted value. */ )
{
    size_t            consumed = 0;
    const std::string value( text );
    const auto        result = std::stoull( value, &consumed );

    if( value.empty() || value.front() == '-' || consumed != value.size() || result == 0 || result > maximum )
    {
        throw std::runtime_error( "invalid positive integer: " + value );
    }

    return result;
}

/// Hash the exact table bytes with FNV-1a for reproducible result identification.
uint64_t tableHash( const std::string &path /**< [in] Descrambling table filename. */ )
{
    std::ifstream file( path, std::ios::binary );

    if( !file )
    {
        throw std::runtime_error( "cannot read table: " + path );
    }

    uint64_t hash = 14695981039346656037ULL;
    char     byte;

    while( file.get( byte ) )
    {
        hash = ( hash ^ static_cast<unsigned char>( byte ) ) * 1099511628211ULL;
    }

    return hash;
}

/// Print the first matching procfs field without interrogating any camera or instrument service.
void printField( const char *path /**< [in] Local procfs filename. */,
                 const char *prefix /**< [in] Field name at the start of a line. */ )
{
    std::ifstream file( path );
    std::string   line;

    while( std::getline( file, line ) )
    {
        if( line.starts_with( prefix ) )
        {
            std::cout << "# " << line << '\n';
            return;
        }
    }
}

/// Print descriptive sample statistics in microseconds; do not subtract timer overhead.
void statistics( const std::string  &label /**< [in] CSV fields identifying the sample population. */,
                 std::vector<double> samples /**< [in] Nanosecond durations, copied for sorting. */ )
{
    const double mean     = std::accumulate( samples.begin(), samples.end(), 0.0 ) / samples.size();
    double       variance = 0;

    for( double sample : samples )
    {
        variance += ( sample - mean ) * ( sample - mean );
    }

    std::sort( samples.begin(), samples.end() );
    const size_t middle = samples.size() / 2;
    const double median = samples.size() % 2 ? samples[middle] : 0.5 * ( samples[middle - 1] + samples[middle] );
    const size_t p95    = static_cast<size_t>( std::ceil( 0.95 * samples.size() ) ) - 1;
    const size_t p99    = static_cast<size_t>( std::ceil( 0.99 * samples.size() ) ) - 1;

    std::cout << label << ',' << samples.size() << ',' << mean * 1e-3 << ',' << median * 1e-3 << ','
              << std::sqrt( variance / samples.size() ) * 1e-3 << ',' << samples[p95] * 1e-3 << ','
              << samples[p99] * 1e-3 << ',' << samples.back() * 1e-3 << '\n';
}

/// Run one comparison path, optionally including the same publication copy as staged frameGrabber loading.
void execute( Kernel   kernel /**< [in] Algorithm to measure. */,
              bool     staged /**< [in] Include private-output to publication-buffer streamCopy. */,
              ocam2_id id /**< [in] Initialized SDK camera identifier. */,
              const MagAOX::app::ocamDescrambler &descrambler /**< [in] Prepared mapping and optimized kernel. */,
              short                              *output /**< [out] Private output buffer. */,
              short       *published /**< [out] Publication buffer used by staged measurements. */,
              const short *raw /**< [in] Synthetic raw frame. */ )
{
    unsigned frame = 0;

    switch( kernel )
    {
    case Kernel::sdk:
        ocam2_descramble( id, &frame, output, raw );
        break;
    case Kernel::mapped:
        descrambler.mapped( output, raw );
        break;
    case Kernel::optimized:
        descrambler.descramble( output, raw );
        break;
    case Kernel::copy:
        std::memcpy( output, raw, descrambler.imagePixels() * sizeof( short ) );
        break;
    }

    if( staged )
    {
        MagAOX::streamCopy( published, output, descrambler.imagePixels() * sizeof( short ) );
    }

    observe( staged ? published : output );
}

/// Verify all generated frames against the SDK before timing either replacement.
void validate( ocam2_id                            id /**< [in] Initialized SDK camera identifier. */,
               const MagAOX::app::ocamDescrambler &descrambler /**< [in] Prepared replacement. */,
               const std::vector<short>           &raw /**< [in] Contiguous synthetic raw frames. */,
               size_t                              buffers /**< [in] Number of raw frames. */ )
{
    std::vector<short> expected( descrambler.imagePixels() );
    std::vector<short> output( expected.size() );
    std::vector<short> published( expected.size() );

    for( size_t buffer = 0; buffer < buffers; ++buffer )
    {
        const short *input = raw.data() + buffer * descrambler.rawPixels();
        unsigned     frame = 0;
        ocam2_descramble( id, &frame, expected.data(), input );

        for( Kernel kernel : { Kernel::mapped, Kernel::optimized } )
        {
            for( size_t pixel = 0; pixel < expected.size(); ++pixel )
            {
                output[pixel]    = static_cast<short>( ~expected[pixel] );
                published[pixel] = output[pixel];
            }

            execute( kernel, true, id, descrambler, output.data(), published.data(), input );

            if( output != expected || published != expected )
            {
                throw std::runtime_error( "replacement output differs from SDK for raw buffer " +
                                          std::to_string( buffer ) );
            }
        }
    }
}

/// Compare kernel-only and staged paths in rotating blocks across repeated rounds.
void benchmark( ocam2_id                            id /**< [in] Initialized SDK camera identifier. */,
                const MagAOX::app::ocamDescrambler &descrambler /**< [in] Prepared replacement. */,
                size_t                              iterations /**< [in] Samples per path per round. */,
                size_t                              buffers /**< [in] Raw frames rotated during this comparison. */,
                size_t                              rounds /**< [in] Repeated rounds with rotating method order. */ )
{
    std::mt19937                       random( 0x0ca2 );
    std::uniform_int_distribution<int> values( -32768, 32767 );
    std::vector<short>                 raw( buffers * descrambler.rawPixels() );
    std::vector<short>                 output( descrambler.imagePixels() );
    std::vector<short>                 published( output.size() );
    std::array<std::vector<double>, 8> samples;

    std::cout << "# alignment_mod64,buffers=" << buffers << ",raw=" << reinterpret_cast<uintptr_t>( raw.data() ) % 64
              << ",raw_stride=" << ( descrambler.rawPixels() * sizeof( short ) ) % 64
              << ",private=" << reinterpret_cast<uintptr_t>( output.data() ) % 64
              << ",published=" << reinterpret_cast<uintptr_t>( published.data() ) % 64 << '\n';

    for( auto &pixel : raw )
    {
        pixel = static_cast<short>( values( random ) );
    }

    validate( id, descrambler, raw, buffers );

    for( auto &pathSamples : samples )
    {
        pathSamples.reserve( iterations * rounds );
    }

    for( size_t method = 0; method < samples.size(); ++method )
    {
        for( size_t warmup = 0; warmup < 1024; ++warmup )
        {
            execute( static_cast<Kernel>( method % 4 ),
                     method >= 4,
                     id,
                     descrambler,
                     output.data(),
                     published.data(),
                     raw.data() + ( warmup % buffers ) * descrambler.rawPixels() );
        }
    }

    constexpr size_t blockSize = 128;

    for( size_t round = 0; round < rounds; ++round )
    {
        for( size_t first = 0; first < iterations; first += blockSize )
        {
            for( size_t order = 0; order < samples.size(); ++order )
            {
                const size_t method = ( order + round + first / blockSize ) % samples.size();

                for( size_t index = first; index < std::min( first + blockSize, iterations ); ++index )
                {
                    const short *input = raw.data() + ( index % buffers ) * descrambler.rawPixels();
                    observe( input );
                    const int64_t begin = nowNs();
                    execute( static_cast<Kernel>( method % 4 ),
                             method >= 4,
                             id,
                             descrambler,
                             output.data(),
                             published.data(),
                             input );
                    const int64_t end = nowNs();
                    samples[method].push_back( static_cast<double>( end - begin ) );
                }

                const auto &result = method >= 4 ? published : output;

                for( short pixel : result )
                {
                    outputChecksum += static_cast<uint16_t>( pixel );
                }
            }
        }

        for( size_t method = 0; method < samples.size(); ++method )
        {
            const auto   first = samples[method].begin() + round * iterations;
            const double mean  = std::accumulate( first, first + iterations, 0.0 ) / iterations;
            std::cout << "# round_mean_us,buffers=" << buffers << ",round=" << round + 1
                      << ",kernel=" << kernelNames[method % 4] << ",staged=" << ( method >= 4 ) << ',' << mean * 1e-3
                      << '\n';
        }
    }

    for( size_t method = 0; method < samples.size(); ++method )
    {
        statistics( std::to_string( buffers ) + ',' + kernelNames[method % 4] + ',' +
                        ( method >= 4 ? "streamCopy" : "kernel" ),
                    std::move( samples[method] ) );
    }
}

} // namespace

/// Run the standalone benchmark with a mapping table, app-supported mode and optional local process affinity.
int main( int    argc /**< [in] Number of command-line arguments. */,
          char **argv /**< [in] Table, mode, iterations, raw-buffer count, optional CPU and round count. */ )
{
    ocam2_id id = -1;

    try
    {
        if( argc < 5 || argc > 7 )
        {
            std::cerr << "Usage: " << argv[0]
                      << " TABLE normal|bin2|bin1x3|bin1x4 ITERATIONS RAW_BUFFERS [CPU [ROUNDS]]\n"
                      << "Each run compares one hot raw frame and RAW_BUFFERS rotating synthetic frames;\n"
                      << "use RAW_BUFFERS=64 for the EDT ring-size comparison. Default ROUNDS=7.\n"
                      << "CPU=-1 retains current affinity; other values pin only this benchmark process.\n"
                      << "No camera, DMA, shared-memory stream, reader or real-time scheduling is used.\n";
            return 1;
        }

        const std::string table      = argv[1];
        const std::string modeName   = argv[2];
        const size_t      iterations = positive( argv[3], 1000000 );
        const size_t      buffers    = positive( argv[4], 1024 );
        const size_t      rounds     = argc == 7 ? positive( argv[6], 100 ) : 7;
        ocam2_mode        mode       = OCAM2_UNKNOWN;

        if( iterations * rounds > 10000000 )
        {
            throw std::runtime_error( "limit samples per path to ten million (iterations times rounds)" );
        }

        if( modeName == "normal" )
        {
            mode = OCAM2_NORMAL;
        }
        else if( modeName == "bin2" )
        {
            mode = OCAM2_BINNING;
        }
        else if( modeName == "bin1x3" )
        {
            mode = OCAM2_BINNING1x3;
        }
        else if( modeName == "bin1x4" )
        {
            mode = OCAM2_BINNING1x4;
        }
        else
        {
            throw std::runtime_error( "unsupported mode: " + modeName );
        }

        if( argc >= 6 && std::string( argv[5] ) != "-1" )
        {
            size_t     consumed = 0;
            const auto cpu      = std::stoul( argv[5], &consumed );
            cpu_set_t  affinity;

            if( consumed != std::strlen( argv[5] ) || cpu >= CPU_SETSIZE )
            {
                throw std::runtime_error( "invalid CPU" );
            }

            CPU_ZERO( &affinity );
            CPU_SET( cpu, &affinity );

            if( sched_setaffinity( 0, sizeof( affinity ), &affinity ) != 0 )
            {
                throw std::runtime_error( "could not set benchmark affinity" );
            }
        }

        std::cout << std::fixed << std::setprecision( 4 );
        std::cout << "# compiler=" << __VERSION__ << "\n# cxxflags=" << OCAM_BENCH_CXXFLAGS
                  << "\n# sdk_cflags=" << OCAM_BENCH_CFLAGS << "\n# table=" << table << "\n# table_fnv1a64=" << std::hex
                  << tableHash( table ) << std::dec << '\n';
        printField( "/proc/sys/kernel/hostname", "" );
        printField( "/proc/cpuinfo", "model name" );
        printField( "/proc/self/status", "Cpus_allowed_list" );

        if( ocam2_init( mode, table.c_str(), &id ) != OCAM2_OK )
        {
            id = -1;
            throw std::runtime_error( "SDK initialization failed" );
        }

        MagAOX::app::ocamDescrambler descrambler;

        if( descrambler.prepare( id ) != 0 )
        {
            throw std::runtime_error( "replacement preparation failed" );
        }

        std::cout << "# mode=" << modeName << ",raw_pixels=" << descrambler.rawPixels()
                  << ",image_pixels=" << descrambler.imagePixels() << ",structured=" << descrambler.structured()
                  << ",iterations=" << iterations << ",rounds=" << rounds << '\n';
        std::cout << "# 128-frame blocks rotate method order; 1024 warmup calls per path.\n"
                  << "# Timings include clock/barrier overhead; output checksums run outside measured intervals.\n"
                  << "# Local synthetic cache behavior does not reproduce DMA or cross-core stream readers.\n"
                  << "buffers,kernel,path,samples,mean_us,median_us,stddev_us,p95_us,p99_us,max_us\n";

        std::vector<double> overhead;
        overhead.reserve( 10000 );

        for( size_t sample = 0; sample < 10000; ++sample )
        {
            const int64_t begin = nowNs();
            observe( &descrambler );
            overhead.push_back( static_cast<double>( nowNs() - begin ) );
        }

        statistics( "0,timer,clock", std::move( overhead ) );
        benchmark( id, descrambler, iterations, 1, rounds );

        if( buffers > 1 )
        {
            benchmark( id, descrambler, iterations, buffers, rounds );
        }

        std::cout << "# validated=all_raw_buffers,checksum=" << outputChecksum << '\n';
        ocam2_exit( id );
        return 0;
    }
    catch( const std::exception &error )
    {
        if( id >= 0 )
        {
            ocam2_exit( id );
        }

        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
