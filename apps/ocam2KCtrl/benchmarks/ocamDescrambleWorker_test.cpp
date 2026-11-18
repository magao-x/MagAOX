/** \file ocamDescrambleWorker_test.cpp
 * \brief Standalone hardware-free checks of the benchmark's blocking descramble worker.
 * \ingroup ocam2KCtrl_descramble_unit_test
 *
 * \see MagAOX::app::ocamDescrambleWorker
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sched.h>
#include <unistd.h>

#include "ocamDescrambleWorker.hpp"
#include "../tests/ocamDescrambleFixture.hpp"

namespace
{

using MagAOX::app::ocamDescrambler;
using MagAOX::app::ocamDescrambleWorker;

/// \cond OCAM2KCTRL_WORKER_TEST_SUPPORT

/// Number of successful checks, written only by the calling thread.
size_t checks{ 0 };

/// Fail the executable immediately when a functional invariant is violated.
void check( bool condition /**< [in] Required invariant. */, const char *message /**< [in] Failure explanation. */ )
{
    if( !condition )
    {
        throw std::runtime_error( message );
    }

    ++checks;
}

/// Verify rejection uses the documented exception category.
template <typename exceptionT, typename callableT>
void expectThrow( callableT &&callable /**< [in] Operation required to reject invalid setup or state. */,
                  const char *message /**< [in] Failure explanation when no expected exception occurs. */ )
{
    bool caught = false;

    try
    {
        callable();
    }
    catch( const exceptionT & )
    {
        caught = true;
    }

    check( caught, message );
}

/// Pin the calling thread for this process's checks and restore its original affinity afterward.
class scopedAffinity
{
    cpu_set_t m_original{}; ///< Calling thread's original allowed CPU mask.

  public:
    /// Save the allowed mask and select one currently allowed CPU.
    explicit scopedAffinity( int cpu /**< [in] Logical CPU for the test's calling thread. */ );

    /// Restore the original mask after all worker instances have been destroyed.
    ~scopedAffinity();
};

scopedAffinity::scopedAffinity( int cpu )
{
    check( sched_getaffinity( 0, sizeof( m_original ), &m_original ) == 0, "cannot read caller affinity" );
    check( cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET( cpu, &m_original ), "caller CPU is not allowed" );
    cpu_set_t selected;
    CPU_ZERO( &selected );
    CPU_SET( cpu, &selected );
    check( sched_setaffinity( 0, sizeof( selected ), &selected ) == 0, "cannot pin caller" );
}

scopedAffinity::~scopedAffinity()
{
    sched_setaffinity( 0, sizeof( m_original ), &m_original );
}

/// Own a valid synthetic SDK table and the currently initialized SDK handle.
class sdkFixture
{
    std::string m_path;     ///< Temporary table filename removed after releasing SDK ownership.
    ocam2_id    m_id{ -1 }; ///< Initialized SDK handle; zero is valid and negative means closed.

  public:
    /// Write a complete normal-mode table with an in-file string terminator.
    explicit sdkFixture( const std::vector<int> &table /**< [in] Valid source-index table. */ );

    /// Release the SDK global mapping before deleting the temporary table.
    ~sdkFixture();

    /// Replace the current SDK mode, releasing the previous handle first.
    ocam2_id open( ocam2_mode mode /**< [in] SDK mode to initialize. */ );
};

sdkFixture::sdkFixture( const std::vector<int> &table )
{
    char path[] = "/tmp/ocam-worker-test-XXXXXX";
    int  fd     = mkstemp( path );
    check( fd >= 0, "cannot create temporary SDK table" );
    m_path = path;
    ::close( fd );
    std::ofstream file( m_path, std::ios::binary );
    check( file.good(), "cannot open temporary SDK table" );

    for( int index : table )
    {
        file << index << ',';
    }

    file.put( '\0' );
    file.close();
    check( !file.fail(), "cannot write temporary SDK table" );
}

sdkFixture::~sdkFixture()
{
    if( m_id >= 0 )
    {
        ocam2_exit( m_id );
    }

    std::remove( m_path.c_str() );
}

ocam2_id sdkFixture::open( ocam2_mode mode )
{
    if( m_id >= 0 )
    {
        ocam2_exit( m_id );
        m_id = -1;
    }

    check( ocam2_init( mode, m_path.c_str(), &m_id ) == OCAM2_OK, "cannot initialize fixture SDK mode" );
    return m_id;
}

/// Fill one exact-size raw frame with changing 16-bit patterns and a known SDK frame number.
void fillRaw( std::vector<short> &raw /**< [out] Complete raw frame. */,
              unsigned            sequence /**< [in] Distinguishes repeated jobs and signed bit patterns. */ )
{
    uint32_t random = 0x12345678u + sequence;

    for( size_t index = 0; index < raw.size(); ++index )
    {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        uint16_t bits = static_cast<uint16_t>( index + sequence );

        if( sequence % 4 == 1 )
        {
            bits = static_cast<uint16_t>( ~bits );
        }
        else if( sequence % 4 == 2 )
        {
            bits = index % 2 ? 0x8000 : 0x7fff;
        }
        else if( sequence % 4 == 3 )
        {
            bits = static_cast<uint16_t>( random );
        }

        std::memcpy( raw.data() + index, &bits, sizeof( bits ) );
    }

    const unsigned number = 0xabcd0000u + sequence;
    std::memcpy( reinterpret_cast<char *>( raw.data() ) + OCAM2_IMAGE_NB_OFFSET, &number, sizeof( number ) );
}

/// Compare whole outputs and every guard word, including both sides of each split boundary.
void verifyOutput( const std::vector<short> &storage /**< [in] Guarded output allocation. */,
                   const short              *output /**< [in] First output sample within storage. */,
                   const std::vector<short> &expected /**< [in] Reference SDK pixels. */,
                   short                     guard /**< [in] Guard word outside the output region. */ )
{
    const size_t offset = output - storage.data();
    check( std::equal( expected.begin(), expected.end(), output ), "worker output differs from SDK" );
    check( std::all_of( storage.begin(),
                        storage.begin() + offset,
                        [guard]( short value )
                        {
                            return value == guard;
                        } ),
           "worker changed leading output guard" );
    check( std::all_of( storage.begin() + offset + expected.size(),
                        storage.end(),
                        [guard]( short value )
                        {
                            return value == guard;
                        } ),
           "worker changed trailing output guard" );
}

/// \endcond

/// Check every split, repeated blocking jobs, source reuse, output alignment, and park/resume cycles.
/** \see MagAOX::app::ocamDescrambleWorker::run
 * \see MagAOX::app::ocamDescrambleWorker::serial
 * \see MagAOX::app::ocamDescrambleWorker::park
 * \see MagAOX::app::ocamDescrambleWorker::resume
 */
void checkFrames( ocam2_id               id /**< [in] SDK reference handle. */,
                  const ocamDescrambler &descrambler /**< [in] Prepared structured bin2 helper. */,
                  int                    callerCpu /**< [in] Exactly pinned calling CPU. */,
                  int                    workerCpu /**< [in] Separate allowed worker CPU. */ )
{
    const short          guard = 0x5a5a;
    const std::array     partitions{ ocamDescrambleWorker::Partition::halves,
                                 ocamDescrambleWorker::Partition::paired28,
                                 ocamDescrambleWorker::Partition::paired30 };
    std::vector<short>   raw( descrambler.rawPixels() );
    std::vector<short>   expected( descrambler.imagePixels() );
    ocamDescrambleWorker worker( descrambler, callerCpu, workerCpu );
    check( worker.cpu() == workerCpu, "worker affinity readback differs from requested CPU" );

    for( size_t offset : { 0u, 1u, 8u } )
    {
        std::vector<short> storage( expected.size() + 128, guard );
        const auto         aligned = ( reinterpret_cast<uintptr_t>( storage.data() ) + 63 ) & ~uintptr_t( 63 );
        short             *output  = reinterpret_cast<short *>( aligned ) + 32 + offset;

        expectThrow<std::logic_error>(
            [&]()
            {
                worker.run( output, raw.data(), partitions[0] );
            },
            "parked worker accepted a job" );
        worker.resume();

        expectThrow<std::logic_error>(
            [&]()
            {
                worker.run( output, raw.data(), static_cast<ocamDescrambleWorker::Partition>( -1 ) );
            },
            "worker accepted an invalid partition" );
        check( std::all_of( storage.begin(),
                            storage.end(),
                            [guard]( short value )
                            {
                                return value == guard;
                            } ),
               "rejected job changed the output" );

        for( unsigned sequence = 0; sequence < 16; ++sequence )
        {
            fillRaw( raw, sequence );
            const auto before = raw;
            unsigned   number = 0;
            ocam2_descramble( id, &number, expected.data(), raw.data() );
            check( number == 0xabcd0000u + sequence, "SDK frame-number reference changed" );
            for( bool paired : { false, true } )
            {
                std::fill( storage.begin(), storage.end(), guard );
                worker.serial( output, raw.data(), paired );
                verifyOutput( storage, output, expected, guard );
            }

            for( const auto partition : partitions )
            {
                std::fill( storage.begin(), storage.end(), guard );
                ocamDescrambleWorker::Timings timings{};
                worker.run( output, raw.data(), partition, false, sequence % 2 ? &timings : nullptr );
                verifyOutput( storage, output, expected, guard );
                check( raw == before, "worker changed its raw input" );
            }

            // run() has completed both owners, so its input can be overwritten immediately.
            std::fill( raw.begin(), raw.end(), short( -1234 ) );
            worker.park();
            verifyOutput( storage, output, expected, guard );
            std::fill( storage.begin(), storage.end(), guard );
            worker.resume();
            check( std::all_of( storage.begin(),
                                storage.end(),
                                [guard]( short value )
                                {
                                    return value == guard;
                                } ),
                   "resume repeated the previous completed job" );
        }

        const auto before = raw;

        for( const auto partition : partitions )
        {
            worker.run( output, raw.data(), partition, true );
            check( std::all_of( storage.begin(),
                                storage.end(),
                                [guard]( short value )
                                {
                                    return value == guard;
                                } ),
                   "handoff-only job changed the output" );
            check( raw == before, "handoff-only job changed the input" );
        }

        worker.park();
    }
}

/// Reject invalid startup, verify completed-job destruction, and retain owned mapping data after SDK reconfiguration.
/** \see MagAOX::app::ocamDescrambleWorker::ocamDescrambleWorker */
void checkLifecycle( sdkFixture &fixture /**< [in,out] SDK mode owner. */,
                     int         callerCpu /**< [in] Exactly pinned calling CPU. */,
                     int         workerCpu /**< [in] Separate allowed worker CPU. */ )
{
    ocamDescrambler descrambler;
    expectThrow<std::runtime_error>(
        [&]()
        {
            ocamDescrambleWorker worker( descrambler, callerCpu, workerCpu );
        },
        "worker accepted an unprepared helper" );
    check( descrambler.prepare( fixture.open( OCAM2_NORMAL ) ) == 0, "cannot prepare normal mode" );
    expectThrow<std::runtime_error>(
        [&]()
        {
            ocamDescrambleWorker worker( descrambler, callerCpu, workerCpu );
        },
        "worker accepted a non-bin2 helper" );
    const auto id = fixture.open( OCAM2_BINNING2x2 );
    check( descrambler.prepare( id ) == 0 && descrambler.structured(), "cannot prepare structured bin2 helper" );

    for( int cpu : { -1, CPU_SETSIZE, CPU_SETSIZE - 1, callerCpu } )
    {
        expectThrow<std::runtime_error>(
            [&]()
            {
                ocamDescrambleWorker worker( descrambler, callerCpu, cpu );
            },
            "worker accepted invalid or shared CPU affinity" );
    }

    expectThrow<std::runtime_error>(
        [&]()
        {
            ocamDescrambleWorker worker( descrambler, workerCpu, callerCpu );
        },
        "worker accepted an incorrectly pinned caller" );

    std::vector<short> raw( descrambler.rawPixels() );
    std::vector<short> output( descrambler.imagePixels() );
    std::vector<short> expected( output.size() );

    for( unsigned sequence = 0; sequence < 4; ++sequence )
    {
        fillRaw( raw, sequence );
        unsigned number = 0;
        ocam2_descramble( id, &number, expected.data(), raw.data() );

        {
            ocamDescrambleWorker worker( descrambler, callerCpu, workerCpu );
            worker.resume();
            worker.run( output.data(), raw.data(), ocamDescrambleWorker::Partition::paired28 );
            // Destruction follows the blocking call; no concurrent destruction is part of this API.
        }

        std::fill( raw.begin(), raw.end(), short( -1 ) );
        check( output == expected, "worker destruction lost or repeated a completed job" );
    }

    checkFrames( id, descrambler, callerCpu, workerCpu );

    fillRaw( raw, 19 );
    unsigned number = 0;
    ocam2_descramble( id, &number, expected.data(), raw.data() );

    {
        ocamDescrambleWorker worker( descrambler, callerCpu, workerCpu );
        descrambler.clear();
        fixture.open( OCAM2_NORMAL );
        worker.resume();
        worker.run( output.data(), raw.data(), ocamDescrambleWorker::Partition::paired30 );
    }

    check( output == expected, "worker retained references to released helper or SDK mapping data" );
}

} // namespace

/// Run correctness and lifecycle checks only, without acquiring frames or contacting an instrument service.
int main( int    argc /**< [in] Optional caller and worker CPU arguments. */,
          char **argv /**< [in] Program name followed by two logical CPU indices. */ )
{
    try
    {
        check( argc == 1 || argc == 3, "usage: ocamDescrambleWorker_test [caller-cpu worker-cpu]" );
        const int      callerCpu = argc == 3 ? std::stoi( argv[1] ) : 2;
        const int      workerCpu = argc == 3 ? std::stoi( argv[2] ) : 4;
        scopedAffinity affinity( callerCpu );

        {
            sdkFixture fixture( libXWCTest::ocam2KCtrlTest::ocamDescrambleTable() );
            checkLifecycle( fixture, callerCpu, workerCpu );
        }

        auto table = libXWCTest::ocam2KCtrlTest::ocamDescrambleTable();
        ++table[0];
        sdkFixture      fixture( table );
        ocamDescrambler descrambler;
        check( descrambler.prepare( fixture.open( OCAM2_BINNING2x2 ) ) == 0 && !descrambler.structured(),
               "cannot prepare valid non-affine bin2 fixture" );
        expectThrow<std::runtime_error>(
            [&]()
            {
                ocamDescrambleWorker worker( descrambler, callerCpu, workerCpu );
            },
            "worker accepted a non-affine bin2 helper" );

        std::cout << "Passed " << checks << " worker checks (caller CPU " << callerCpu << ", worker CPU " << workerCpu
                  << ").\n";
        return 0;
    }
    catch( const std::exception &error )
    {
        std::cerr << "Worker check failed after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
