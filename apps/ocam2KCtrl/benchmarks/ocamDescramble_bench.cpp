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
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <sched.h>

#include "../ocamDescrambler.hpp"
#include "ocamDescrambleWorker.hpp"
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
    compact,
    mapped,
    optimized,
    copy,
    rangeSerial,
    pairedSerial,
    halves,
    paired28,
    paired30,
    handoff
};

/// Names corresponding to each comparison kernel, in enumeration order.
constexpr std::array kernelNames{ "sdk",
                                  "compact",
                                  "mapped",
                                  "optimized",
                                  "memcpy",
                                  "range_serial",
                                  "paired_serial",
                                  "parallel_halves",
                                  "parallel_paired28",
                                  "parallel_paired30",
                                  "handoff" };

/// Existing single-thread choices remain available without a worker argument.
constexpr size_t serialKernelCount = static_cast<size_t>( Kernel::rangeSerial );

static_assert( kernelNames.size() == static_cast<size_t>( Kernel::handoff ) + 1 );

using Worker = MagAOX::app::ocamDescrambleWorker;
using Clock  = std::chrono::steady_clock;

/// One measured algorithm with or without its primary-thread publication copy.
struct Method
{
    Kernel m_kernel; ///< Selected algorithm or handoff calibration.
    bool   m_staged; ///< Copy to the publication buffer after all computation finishes.
};

/// Shared SDK reference and optional persistent benchmark-only worker.
struct Context
{
    ocam2_id                            m_id;          ///< Live SDK reference handle.
    const MagAOX::app::ocamDescrambler &m_descrambler; ///< Prepared production helper.
    Worker                             *m_worker;      ///< Optional worker owned by main for the entire invocation.
};

/// Guarded output with an explicitly controlled offset from a 64-byte-aligned base.
class GuardedBuffer
{
    std::vector<short> m_storage; ///< Pixels and untouched prefix/suffix guards.
    size_t             m_begin;   ///< Pixel start within the owning allocation.
    size_t             m_size;    ///< Number of writable pixels.

  public:
    /// Allocate pixels with at least 64 guard bytes before and after the selected output extent.
    GuardedBuffer( size_t pixels /**< [in] Writable pixel count. */,
                   size_t offset /**< [in] Even byte offset from a 64-byte boundary, below 64. */ );

    /// Return the selected writable extent.
    short *data();

    /// Verify every allocation word outside the writable extent remains untouched.
    bool guards() const;
};

GuardedBuffer::GuardedBuffer( size_t pixels, size_t offset ) : m_storage( pixels + 128, 0x4b2d ), m_size( pixels )
{
    const auto base    = reinterpret_cast<uintptr_t>( m_storage.data() + 32 );
    const auto aligned = ( base + 63 ) & ~uintptr_t( 63 );
    m_begin            = ( aligned - reinterpret_cast<uintptr_t>( m_storage.data() ) + offset ) / sizeof( short );
}

short *GuardedBuffer::data()
{
    return m_storage.data() + m_begin;
}

bool GuardedBuffer::guards() const
{
    for( size_t i = 0; i < m_storage.size(); ++i )
    {
        if( ( i < m_begin || i >= m_begin + m_size ) && m_storage[i] != 0x4b2d )
        {
            return false;
        }
    }

    return true;
}

/// Identify methods whose worker must be spinning only during their own comparison block.
bool parallel( Kernel kernel /**< [in] Selected algorithm. */ )
{
    return kernel >= Kernel::halves;
}

/// Select methods while preserving the original ordering when no worker was requested.
std::vector<Method> methods( bool worker /**< [in] Include benchmark-only worker comparisons. */ )
{
    std::vector<Method> result;
    const size_t        count = worker ? kernelNames.size() : serialKernelCount;

    for( bool staged : { false, true } )
    {
        for( size_t kernel = 0; kernel < count; ++kernel )
        {
            if( !staged || static_cast<Kernel>( kernel ) != Kernel::handoff )
            {
                result.push_back( { static_cast<Kernel>( kernel ), staged } );
            }
        }
    }

    return result;
}

/// Park or resume the worker outside all measured and warmup calls for a comparison block.
void selectMethod( const Context &context /**< [in] Optional persistent worker. */,
                   Kernel         kernel /**< [in] Next method. */ )
{
    if( context.m_worker )
    {
        if( parallel( kernel ) )
        {
            context.m_worker->resume();
        }
        else
        {
            context.m_worker->park();
        }
    }
}

/// Pace sample starts outside measured work, counting whole lost periods and avoiding catch-up bursts.
void pace( Clock::time_point       &next /**< [in,out] Next scheduled start. */,
           std::chrono::nanoseconds period /**< [in] Zero for an unpaced run. */,
           uint64_t                &lost /**< [in,out] Whole periods lost to late scheduling. */ )
{
    if( period.count() == 0 )
    {
        return;
    }

    std::this_thread::sleep_until( next );
    const auto current = Clock::now();

    if( current - next >= period )
    {
        lost += ( current - next ) / period;
        next = current;
    }

    next += period;
}

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

/// Return a bounded command-line integer, rejecting suffixes and negative values.
size_t positive( const char *text /**< [in] Decimal argument. */,
                 size_t      maximum /**< [in] Largest accepted value. */,
                 bool        allowZero = false /**< [in] Accept zero for optional settings. */ )
{
    size_t            consumed = 0;
    const std::string value( text );
    const auto        result = std::stoull( value, &consumed );

    if( value.empty() || value.front() == '-' || consumed != value.size() || ( result == 0 && !allowZero ) ||
        result > maximum )
    {
        throw std::runtime_error( "invalid positive integer: " + value );
    }

    return result;
}

/// Read one local CPU-topology field; an empty result means the field is unavailable.
std::string cpuField( int                cpu /**< [in] Logical CPU number. */,
                      const std::string &field /**< [in] Relative sysfs CPU path. */ )
{
    std::ifstream file( "/sys/devices/system/cpu/cpu" + std::to_string( cpu ) + '/' + field );
    std::string   result;
    std::getline( file, result );
    return result;
}

/// Print topology for a verified single-CPU placement without changing scheduler policy or system state.
void printTopology( int cpu /**< [in] Logical CPU number. */, const char *role /**< [in] Primary or worker label. */ )
{
    std::cout << "# topology,role=" << role << ",cpu=" << cpu
              << ",package=" << cpuField( cpu, "topology/physical_package_id" )
              << ",core=" << cpuField( cpu, "topology/core_id" )
              << ",smt_siblings=" << cpuField( cpu, "topology/thread_siblings_list" );

    for( int cache = 0; cache < 8; ++cache )
    {
        const std::string path = "cache/index" + std::to_string( cache ) + '/';

        if( cpuField( cpu, path + "level" ) == "3" )
        {
            std::cout << ",l3_shared_cpus=" << cpuField( cpu, path + "shared_cpu_list" );
            break;
        }
    }

    std::cout << '\n';
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
void execute(
    const Context   &context /**< [in] Prepared helper and optional resumed worker. */,
    Method           method /**< [in] Algorithm and publication-copy selection. */,
    short           *output /**< [out] Private output buffer. */,
    short           *published /**< [out] Publication buffer used by staged measurements. */,
    const short     *raw /**< [in] Synthetic raw frame. */,
    Worker::Timings *timings = nullptr /**< [out] Optional kernel-only diagnostic clocks, disabled normally. */ )
{
    unsigned frame = 0;

    switch( method.m_kernel )
    {
    case Kernel::sdk:
        ocam2_descramble( context.m_id, &frame, output, raw );
        break;
    case Kernel::compact:
        context.m_descrambler.compact( output, raw );
        break;
    case Kernel::mapped:
        context.m_descrambler.mapped( output, raw );
        break;
    case Kernel::optimized:
        context.m_descrambler.descramble( output, raw );
        break;
    case Kernel::copy:
        std::memcpy( output, raw, context.m_descrambler.imagePixels() * sizeof( short ) );
        break;
    case Kernel::rangeSerial:
        context.m_worker->serial( output, raw );
        break;
    case Kernel::pairedSerial:
        context.m_worker->serial( output, raw, true );
        break;
    case Kernel::halves:
    case Kernel::paired28:
    case Kernel::paired30:
    case Kernel::handoff:
    {
        const auto partition = method.m_kernel == Kernel::paired28   ? Worker::Partition::paired28
                               : method.m_kernel == Kernel::paired30 ? Worker::Partition::paired30
                                                                     : Worker::Partition::halves;
        context.m_worker->run( output, raw, partition, method.m_kernel == Kernel::handoff, timings );
        break;
    }
    }

    if( method.m_staged )
    {
        // run() has acknowledged all worker stores before the primary thread begins this copy.
        MagAOX::streamCopy( published, output, context.m_descrambler.imagePixels() * sizeof( short ) );
    }

    observe( method.m_staged ? published : output );
}

/// Verify all generated frames against the SDK before timing each replacement.
void validate( const Context            &context /**< [in] Reference, production helper and optional worker. */,
               const std::vector<short> &raw /**< [in] Contiguous synthetic raw frames. */,
               size_t                    buffers /**< [in] Number of raw frames. */,
               size_t                    outputOffset /**< [in] Additional measured output alignment to validate. */ )
{
    const auto         &descrambler = context.m_descrambler;
    const auto          before      = raw;
    std::vector<short>  expected( descrambler.imagePixels() );
    std::vector<size_t> offsets{ 0, 16, outputOffset };
    std::sort( offsets.begin(), offsets.end() );
    offsets.erase( std::unique( offsets.begin(), offsets.end() ), offsets.end() );

    for( size_t offset : offsets )
    {
        GuardedBuffer output( expected.size(), offset );
        GuardedBuffer published( expected.size(), offset );

        for( size_t buffer = 0; buffer < buffers; ++buffer )
        {
            const short *input = raw.data() + buffer * descrambler.rawPixels();
            unsigned     frame = 0;
            ocam2_descramble( context.m_id, &frame, expected.data(), input );

            for( const auto method : methods( context.m_worker != nullptr ) )
            {
                if( !method.m_staged || method.m_kernel == Kernel::copy )
                {
                    continue;
                }

                for( size_t pixel = 0; pixel < expected.size(); ++pixel )
                {
                    output.data()[pixel]    = static_cast<short>( ~expected[pixel] );
                    published.data()[pixel] = output.data()[pixel];
                }

                selectMethod( context, method.m_kernel );
                execute( context, method, output.data(), published.data(), input );

                if( !std::equal( expected.begin(), expected.end(), output.data() ) ||
                    !std::equal( expected.begin(), expected.end(), published.data() ) || !output.guards() ||
                    !published.guards() )
                {
                    throw std::runtime_error( std::string( kernelNames[static_cast<size_t>( method.m_kernel )] ) +
                                              " output/guard mismatch for raw buffer " + std::to_string( buffer ) +
                                              " at output offset " + std::to_string( offset ) );
                }
            }
        }
    }

    selectMethod( context, Kernel::sdk );

    if( raw != before )
    {
        throw std::runtime_error( "raw input changed during validation" );
    }
}

/// Print output ownership boundaries so cache-line sharing is explicit for each parallel partition.
void printSplits( short *output /**< [in] Selected aligned output start. */ )
{
    const auto address  = reinterpret_cast<uintptr_t>( output );
    const auto boundary = [&]( size_t row /**< [in] Output row beginning the next owner's region. */ )
    {
        return ( address + row * OCAM2_IMAGE_WIDTH_BINNING2x2 * sizeof( short ) ) % 64;
    };

    std::cout << "# split=halves,main_rows=60,worker_rows=60,boundary_mod64=" << boundary( 60 )
              << ",shared_boundary_lines=" << ( boundary( 60 ) != 0 ) << '\n';

    for( size_t pairs : { 28u, 30u } )
    {
        std::cout << "# split=paired" << pairs << ",main_pairs=" << pairs << ",worker_pairs=" << 60 - pairs
                  << ",rows=y_and_119-y,boundary_mod64=" << boundary( pairs ) << ':' << boundary( 120 - pairs )
                  << ",shared_boundary_lines=" << ( boundary( pairs ) != 0 ) + ( boundary( 120 - pairs ) != 0 ) << '\n';
    }
}

/// Collect separately labeled diagnostic clocks without adding them to headline samples.
void diagnostics( const Context            &context /**< [in] Prepared helper and persistent worker. */,
                  const std::vector<short> &raw /**< [in] Synthetic input ring. */,
                  size_t                    buffers /**< [in] Raw frame count. */,
                  size_t                    iterations /**< [in] Separate diagnostic samples per method. */,
                  short                    *output /**< [out] Private pixels. */,
                  short                    *published /**< [out] Publication pixels. */,
                  std::chrono::nanoseconds  period /**< [in] Optional cadence outside measured work. */ )
{
    std::cout << "# separate_instrumented_pass,samples_per_path=" << iterations
              << ",main_and_worker=kernel_regions_only,total=dispatch_plus_completion_plus_optional_copy\n";

    for( const auto method : methods( true ) )
    {
        if( !parallel( method.m_kernel ) || method.m_kernel == Kernel::handoff )
        {
            continue;
        }

        std::vector<double> total, main, worker;
        total.reserve( iterations );
        main.reserve( iterations );
        worker.reserve( iterations );
        uint64_t lost = 0;
        auto     next = Clock::now();
        selectMethod( context, method.m_kernel );

        for( size_t index = 0; index < iterations; ++index )
        {
            pace( next, period, lost );
            const short    *input = raw.data() + ( index % buffers ) * context.m_descrambler.rawPixels();
            Worker::Timings timing{};
            const auto      begin = nowNs();
            execute( context, method, output, published, input, &timing );
            total.push_back( nowNs() - begin );
            main.push_back( timing.mainNs );
            worker.push_back( timing.workerNs );
        }

        selectMethod( context, Kernel::sdk );
        const std::string label = std::to_string( buffers ) + ',' + kernelNames[static_cast<size_t>( method.m_kernel )];
        const std::string suffix = method.m_staged ? "_streamCopy" : "_kernel";
        statistics( label + ",instrumented_total" + suffix, std::move( total ) );
        statistics( label + ",instrumented_main" + suffix, std::move( main ) );
        statistics( label + ",instrumented_worker" + suffix, std::move( worker ) );
        std::cout << "# instrumented_pacing_lost_slots," << label << ',' << suffix << ',' << lost << '\n';
    }
}

/// Compare kernel-only and staged paths in rotating blocks, parking the worker outside parallel blocks.
void benchmark( const Context &context /**< [in] SDK reference, production helper and optional worker. */,
                size_t         iterations /**< [in] Samples per path per round. */,
                size_t         buffers /**< [in] Raw frames rotated during this comparison. */,
                size_t         rounds /**< [in] Repeated rounds with rotating method order. */,
                size_t         outputOffset /**< [in] Even output byte offset; used when the worker is enabled. */,
                bool instrumented /**< [in] Run separate instrumented measurements after the headline pass. */,
                std::chrono::nanoseconds period /**< [in] Optional cadence outside measured work. */ )
{
    const auto                        &descrambler = context.m_descrambler;
    const auto                         paths       = methods( context.m_worker != nullptr );
    std::mt19937                       random( 0x0ca2 );
    std::uniform_int_distribution<int> values( -32768, 32767 );
    std::vector<short>                 raw( buffers * descrambler.rawPixels() );
    // Preserve the existing allocation style when the optional worker is absent.
    std::vector<short>             serialOutput( context.m_worker ? 0 : descrambler.imagePixels() );
    std::vector<short>             serialPublished( serialOutput.size() );
    std::unique_ptr<GuardedBuffer> alignedOutput, alignedPublished;

    if( context.m_worker )
    {
        alignedOutput    = std::make_unique<GuardedBuffer>( descrambler.imagePixels(), outputOffset );
        alignedPublished = std::make_unique<GuardedBuffer>( descrambler.imagePixels(), outputOffset );
    }

    short                           *output    = context.m_worker ? alignedOutput->data() : serialOutput.data();
    short                           *published = context.m_worker ? alignedPublished->data() : serialPublished.data();
    std::vector<std::vector<double>> samples( paths.size() );
    std::vector<uint64_t>            pacingLost( paths.size(), 0 );

    std::cout << "# alignment_mod64,buffers=" << buffers << ",raw=" << reinterpret_cast<uintptr_t>( raw.data() ) % 64
              << ",raw_stride=" << ( descrambler.rawPixels() * sizeof( short ) ) % 64
              << ",private=" << reinterpret_cast<uintptr_t>( output ) % 64
              << ",published=" << reinterpret_cast<uintptr_t>( published ) % 64 << '\n';

    if( context.m_worker )
    {
        printSplits( output );
    }

    for( auto &pixel : raw )
    {
        pixel = static_cast<short>( values( random ) );
    }

    validate( context, raw, buffers, outputOffset );

    for( auto &pathSamples : samples )
    {
        pathSamples.reserve( iterations * rounds );
    }

    for( const auto method : paths )
    {
        selectMethod( context, method.m_kernel );

        for( size_t warmup = 0; warmup < 1024; ++warmup )
        {
            execute( context, method, output, published, raw.data() + ( warmup % buffers ) * descrambler.rawPixels() );
        }

        selectMethod( context, Kernel::sdk );
    }

    constexpr size_t blockSize = 128;

    for( size_t round = 0; round < rounds; ++round )
    {
        for( size_t first = 0; first < iterations; first += blockSize )
        {
            for( size_t order = 0; order < paths.size(); ++order )
            {
                const size_t methodIndex = ( order + round + first / blockSize ) % paths.size();
                const auto   method      = paths[methodIndex];
                selectMethod( context, method.m_kernel );
                auto next = Clock::now();

                for( size_t index = first; index < std::min( first + blockSize, iterations ); ++index )
                {
                    pace( next, period, pacingLost[methodIndex] );
                    const short *input = raw.data() + ( index % buffers ) * descrambler.rawPixels();
                    observe( input );
                    const int64_t begin = nowNs();
                    execute( context, method, output, published, input );
                    const int64_t end = nowNs();
                    samples[methodIndex].push_back( static_cast<double>( end - begin ) );
                }

                selectMethod( context, Kernel::sdk );

                if( method.m_kernel != Kernel::handoff )
                {
                    const short *result = method.m_staged ? published : output;

                    for( size_t pixel = 0; pixel < descrambler.imagePixels(); ++pixel )
                    {
                        outputChecksum += static_cast<uint16_t>( result[pixel] );
                    }
                }
            }
        }

        for( size_t method = 0; method < paths.size(); ++method )
        {
            const auto   first = samples[method].begin() + round * iterations;
            const double mean  = std::accumulate( first, first + iterations, 0.0 ) / iterations;
            std::cout << "# round_mean_us,buffers=" << buffers << ",round=" << round + 1
                      << ",kernel=" << kernelNames[static_cast<size_t>( paths[method].m_kernel )]
                      << ",staged=" << paths[method].m_staged << ',' << mean * 1e-3 << '\n';
        }
    }

    for( size_t method = 0; method < paths.size(); ++method )
    {
        const std::string label = std::to_string( buffers ) + ',' +
                                  kernelNames[static_cast<size_t>( paths[method].m_kernel )] + ',' +
                                  ( paths[method].m_staged ? "streamCopy" : "kernel" );
        statistics( label, std::move( samples[method] ) );
        std::cout << "# pacing_lost_slots," << label << ',' << pacingLost[method] << '\n';
    }

    if( instrumented )
    {
        diagnostics( context, raw, buffers, iterations, output, published, period );
    }

    if( context.m_worker && ( !alignedOutput->guards() || !alignedPublished->guards() ) )
    {
        throw std::runtime_error( "output guards changed during measurements" );
    }
}

} // namespace

/// Run the standalone benchmark with a mapping table, app-supported mode and optional local process affinity.
int main( int    argc /**< [in] Number of command-line arguments. */,
          char **argv /**< [in] Table, mode, samples, buffers, CPUs, rounds, alignment, diagnostics and cadence. */ )
{
    ocam2_id id = -1;

    try
    {
        if( argc < 5 || argc > 11 )
        {
            std::cerr << "Usage: " << argv[0] << " TABLE normal|bin2|bin1x3|bin1x4 ITERATIONS RAW_BUFFERS"
                      << " [CPU [ROUNDS [WORKER_CPU [OUTPUT_OFFSET_BYTES [DIAGNOSTICS [PERIOD_US]]]]]]\n"
                      << "Each run compares one hot raw frame and RAW_BUFFERS rotating synthetic frames;\n"
                      << "use RAW_BUFFERS=64 for the EDT ring-size comparison. Default ROUNDS=7.\n"
                      << "CPU=-1 retains current affinity; other values pin only this benchmark process.\n"
                      << "WORKER_CPU defaults to -1 (disabled); a worker requires bin2 and a distinct physical core.\n"
                      << "Worker output uses a 64-byte-aligned base plus even OUTPUT_OFFSET_BYTES (0..62, default 0).\n"
                      << "DIAGNOSTICS=1 adds a separate instrumented pass; default 0 leaves kernel clocks disabled.\n"
                      << "PERIOD_US defaults to 0 (unpaced); pacing sleeps outside measured calls.\n"
                      << "No camera, DMA, shared-memory stream, reader or real-time scheduling is used.\n";
            return 1;
        }

        const std::string table        = argv[1];
        const std::string modeName     = argv[2];
        const size_t      iterations   = positive( argv[3], 1000000 );
        const size_t      buffers      = positive( argv[4], 1024 );
        const size_t      rounds       = argc >= 7 ? positive( argv[6], 100 ) : 7;
        const int         callerCpu    = argc >= 6 && std::string( argv[5] ) != "-1"
                                             ? static_cast<int>( positive( argv[5], CPU_SETSIZE - 1, true ) )
                                             : -1;
        const int         workerCpu    = argc >= 8 && std::string( argv[7] ) != "-1"
                                             ? static_cast<int>( positive( argv[7], CPU_SETSIZE - 1, true ) )
                                             : -1;
        const size_t      outputOffset = argc >= 9 ? positive( argv[8], 62, true ) : 0;
        const bool        instrumented = argc >= 10 ? positive( argv[9], 1, true ) != 0 : false;
        const auto        period = std::chrono::microseconds( argc == 11 ? positive( argv[10], 1000000, true ) : 0 );
        ocam2_mode        mode   = OCAM2_UNKNOWN;

        if( iterations * rounds > 10000000 )
        {
            throw std::runtime_error( "limit samples per path to ten million (iterations times rounds)" );
        }

        if( outputOffset % sizeof( short ) != 0 || ( workerCpu < 0 && ( outputOffset != 0 || instrumented ) ) )
        {
            throw std::runtime_error(
                "output offset must be even; alignment and worker diagnostics require WORKER_CPU" );
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

        cpu_set_t originalAffinity;
        CPU_ZERO( &originalAffinity );

        if( sched_getaffinity( 0, sizeof( originalAffinity ), &originalAffinity ) != 0 )
        {
            throw std::runtime_error( "could not read initial allowed CPU mask" );
        }

        if( workerCpu >= 0 )
        {
            if( mode != OCAM2_BINNING2x2 || callerCpu < 0 || callerCpu == workerCpu ||
                !CPU_ISSET( callerCpu, &originalAffinity ) || !CPU_ISSET( workerCpu, &originalAffinity ) )
            {
                throw std::runtime_error( "worker requires bin2 and two distinct CPUs in the original allowed mask" );
            }

            const auto mainPackage   = cpuField( callerCpu, "topology/physical_package_id" );
            const auto workerPackage = cpuField( workerCpu, "topology/physical_package_id" );
            const auto mainCore      = cpuField( callerCpu, "topology/core_id" );
            const auto workerCore    = cpuField( workerCpu, "topology/core_id" );

            if( mainPackage.empty() || workerPackage.empty() || mainCore.empty() || workerCore.empty() ||
                ( mainPackage == workerPackage && mainCore == workerCore ) )
            {
                throw std::runtime_error( "could not verify distinct physical cores; SMT siblings are not accepted" );
            }
        }

        if( callerCpu >= 0 )
        {
            cpu_set_t affinity;
            CPU_ZERO( &affinity );
            CPU_SET( callerCpu, &affinity );

            if( !CPU_ISSET( callerCpu, &originalAffinity ) ||
                sched_setaffinity( 0, sizeof( affinity ), &affinity ) != 0 )
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

        if( callerCpu >= 0 )
        {
            printTopology( callerCpu, "primary" );
        }

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

        std::unique_ptr<Worker> worker;

        if( workerCpu >= 0 )
        {
            worker = std::make_unique<Worker>( descrambler, callerCpu, workerCpu );
            printTopology( worker->cpu(), "worker" );
            std::cout
                << "# affinity_verified,primary=" << callerCpu << ",worker=" << worker->cpu()
                << ",worker_lifetime=invocation,worker_parked_for_all_serial_blocks=1\n"
                << "# parallel timer includes dispatch and completion; streamCopy runs on primary after completion.\n"
                << "# range_serial/paired_serial use benchmark rows/pairs; optimized is the production helper.\n"
                << "# handoff processes no pixels; parallel wait is a spin, parked outside each parallel block.\n";
        }

        const Context context{ id, descrambler, worker.get() };

        std::cout << "# mode=" << modeName << ",raw_pixels=" << descrambler.rawPixels()
                  << ",image_pixels=" << descrambler.imagePixels() << ",structured=" << descrambler.structured()
                  << ",iterations=" << iterations << ",rounds=" << rounds << ",period_us=" << period.count() << '\n';
        std::cout << "# 128-frame blocks rotate method order; 1024 unpaced warmup calls per path.\n"
                  << "# Pacing and worker park/resume occur outside timers; lost slots count whole late periods.\n"
                  << "# compact follows the SDK bin2 source-loop order, with mapped fallback in other modes.\n"
                  << "# This source order does not prescribe compiler or hardware memory-operation order.\n"
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
        benchmark( context, iterations, 1, rounds, outputOffset, instrumented, period );

        if( buffers > 1 )
        {
            benchmark( context, iterations, buffers, rounds, outputOffset, instrumented, period );
        }

        std::cout << "# validated=all_raw_buffers_with_output_guards_and_offsets_0_16_requested,checksum="
                  << outputChecksum << '\n';
        worker.reset();
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
