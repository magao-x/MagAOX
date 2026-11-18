/** \file ocamDescrambleParallel_test.cpp
 * \brief Hardware-free correctness and concurrent control checks for the runtime descramble worker.
 * \ingroup ocam2KCtrl_descramble_unit_test
 *
 * \see MagAOX::app::ocamDescrambleParallel
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unistd.h>

#include "../ocamDescrambleParallel.hpp"
#include "../tests/ocamDescrambleFixture.hpp"

namespace
{

using MagAOX::app::ocamDescrambleParallel;
using MagAOX::app::ocamDescrambler;
using Partition = ocamDescrambleParallel::Partition;

/// \cond OCAM2KCTRL_PARALLEL_TEST_SUPPORT

/// Successful invariant count, modified only by the owner thread.
size_t checks{ 0 };

/// Fail the executable when a functional invariant is violated.
void check( bool condition /**< [in] Required invariant. */, const char *message /**< [in] Failure explanation. */ )
{
    if( !condition )
    {
        throw std::runtime_error( message );
    }

    ++checks;
}

/// Own a synthetic SDK map and its current handle independently of worker ownership.
class sdkFixture
{
    std::string m_path;     ///< Temporary complete CSV table removed after SDK cleanup.
    ocam2_id    m_id{ -1 }; ///< SDK handle, with zero a valid initialized handle.

  public:
    /// Write the same structured normal table used by the standalone mapping tests.
    sdkFixture();

    /// Release any live SDK handle and the temporary table.
    ~sdkFixture();

    /// Change SDK mode while preserving the source fixture table.
    ocam2_id open( ocam2_mode mode /**< [in] Mode to initialize. */ );
};

sdkFixture::sdkFixture()
{
    char      path[] = "/tmp/ocam-parallel-test-XXXXXX";
    const int fd     = mkstemp( path );
    check( fd >= 0, "cannot create fixture" );
    ::close( fd );
    m_path = path;
    std::ofstream file( m_path, std::ios::binary );

    for( int index : libXWCTest::ocam2KCtrlTest::ocamDescrambleTable() )
    {
        file << index << ',';
    }

    file.put( '\0' );
    file.close();
    check( !file.fail(), "cannot write fixture" );
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

    check( ocam2_init( mode, m_path.c_str(), &m_id ) == OCAM2_OK, "cannot initialize SDK" );
    return m_id;
}

/// Join the external worker even when an invariant throws during a check.
class workerOwner
{
    ocamDescrambleParallel &m_worker; ///< Core whose buffers outlive the joined thread.
    std::thread             m_thread; ///< Normal-priority worker owned by this local fixture.

  public:
    /// Launch the worker without RT scheduling or instrument CPU placement.
    explicit workerOwner( ocamDescrambleParallel &worker /**< [in,out] Core started in its parked state. */ );

    /// Stop and join before core or pixel allocations can be destroyed.
    ~workerOwner();
};

workerOwner::workerOwner( ocamDescrambleParallel &worker )
    : m_worker( worker ), m_thread(
                              [&worker]()
                              {
                                  worker.workerMain();
                              } )
{
}

workerOwner::~workerOwner()
{
    m_worker.requestStop();
    m_thread.join();
}

/// Generate raw signed bit patterns which distinguish every repeated frame and split boundary.
void fillRaw( std::vector<short> &raw /**< [out] Complete raw frame. */,
              unsigned            sequence /**< [in] Frame variation. */ )
{
    uint32_t value = 0x12345678u + sequence;

    for( size_t index = 0; index < raw.size(); ++index )
    {
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        const uint16_t bits = sequence % 3 == 0   ? static_cast<uint16_t>( index + sequence )
                              : sequence % 3 == 1 ? static_cast<uint16_t>( ~index )
                                                  : static_cast<uint16_t>( value );
        std::memcpy( raw.data() + index, &bits, sizeof( bits ) );
    }
}

/// Check full pixels and both guards; immediate inspection also checks publication after worker completion.
void verifyOutput( const std::vector<short> &storage /**< [in] Guarded destination. */,
                   const short              *output /**< [in] First destination pixel. */,
                   const std::vector<short> &expected /**< [in] SDK output. */ )
{
    const size_t offset = output - storage.data();
    check( std::equal( expected.begin(), expected.end(), output ), "parallel pixels differ from SDK" );
    check( std::all_of( storage.begin(),
                        storage.begin() + offset,
                        []( short value )
                        {
                            return value == 0x5a5a;
                        } ),
           "leading output guard changed" );
    check( std::all_of( storage.begin() + offset + expected.size(),
                        storage.end(),
                        []( short value )
                        {
                            return value == 0x5a5a;
                        } ),
           "trailing output guard changed" );
}

/// \endcond

/// Check all partitions, aligned/misaligned outputs, source immutability, handoff, park, and owned map lifetime.
/** \see MagAOX::app::ocamDescrambleParallel::prepare
 * \see MagAOX::app::ocamDescrambleParallel::run
 * \see MagAOX::app::ocamDescrambleParallel::serial
 * \see MagAOX::app::ocamDescrambleParallel::clear
 */
void checkFrames( sdkFixture &fixture /**< [in,out] Real SDK reference and map owner. */ )
{
    ocamDescrambler        helper;
    ocamDescrambleParallel worker;
    check( worker.prepare( helper ) < 0, "unprepared helper accepted" );
    check( helper.prepare( fixture.open( OCAM2_NORMAL ) ) == 0, "normal helper preparation failed" );
    check( worker.prepare( helper ) < 0, "unsupported normal map accepted" );
    const auto id = fixture.open( OCAM2_BINNING2x2 );
    check( helper.prepare( id ) == 0 && helper.structured(), "bin2 helper preparation failed" );
    check( worker.prepare( helper ) == 0, "worker map preparation failed" );
    std::vector<short> raw( helper.rawPixels() );
    std::vector<short> expected( helper.imagePixels() );
    std::vector<short> storage( expected.size() + 128, 0x5a5a );
    const auto         aligned = ( reinterpret_cast<uintptr_t>( storage.data() ) + 63 ) & ~uintptr_t( 63 );
    short             *base    = reinterpret_cast<short *>( aligned ) + 32;
    check( !worker.started(), "worker started without external thread" );
    check( !worker.waitUntilStarted( std::chrono::milliseconds( 1 ) ), "absent worker reported startup" );
    check( worker.resume() < 0, "absent worker resumed" );
    check( worker.run( base, raw.data(), Partition::paired30 ) < 0, "absent worker accepted buffers" );
    workerOwner owner( worker );
    check( worker.waitUntilStarted(), "worker failed startup acknowledgement" );
    check( !worker.running() && worker.isParked(), "worker did not start parked" );
    check( worker.run( base, raw.data(), Partition::paired30 ) < 0, "parked worker accepted buffers" );

    for( size_t offset : { 0u, 1u, 8u } )
    {
        short *output = base + offset;

        for( unsigned sequence = 0; sequence < 12; ++sequence )
        {
            fillRaw( raw, sequence );
            const auto before = raw;
            unsigned   number = 0;
            ocam2_descramble( id, &number, expected.data(), raw.data() );
            check( worker.resume() == 0, "worker resume failed" );

            for( bool paired : { false, true } )
            {
                std::fill( storage.begin(), storage.end(), 0x5a5a );
                worker.serial( output, raw.data(), paired );
                verifyOutput( storage, output, expected );
            }

            for( auto partition : { Partition::halves, Partition::paired28, Partition::paired30 } )
            {
                std::fill( storage.begin(), storage.end(), 0x5a5a );
                ocamDescrambleParallel::Timings timings{};
                check( worker.run( output, raw.data(), partition, false, sequence % 2 ? &timings : nullptr ) == 0,
                       "valid worker job rejected" );
                verifyOutput( storage, output, expected );
                check( raw == before, "worker modified source" );
            }

            std::fill( raw.begin(), raw.end(), short( -1234 ) );
            worker.park();
            verifyOutput( storage, output, expected );
            check( !worker.running() && worker.isParked(), "park returned before acknowledgement" );
            std::fill( storage.begin(), storage.end(), 0x5a5a );
            check( worker.resume() == 0 && !worker.isParked(), "parked worker could not resume" );
            check( worker.run( output, raw.data(), Partition::paired30, true ) == 0, "handoff job failed" );
            check( std::all_of( storage.begin(),
                                storage.end(),
                                []( short value )
                                {
                                    return value == 0x5a5a;
                                } ),
                   "handoff or resume replayed output writes" );
            check( worker.run( output, raw.data(), static_cast<Partition>( -1 ) ) < 0, "invalid split accepted" );
        }
    }

    fillRaw( raw, 73 );
    unsigned number = 0;
    ocam2_descramble( id, &number, expected.data(), raw.data() );
    helper.clear();
    fixture.open( OCAM2_NORMAL );
    check( worker.run( base, raw.data(), Partition::paired30 ) == 0, "owned mapping lost after SDK reconfigure" );
    check( std::equal( expected.begin(), expected.end(), base ), "worker retained released SDK map" );
    worker.clear();
    check( !worker.prepared() && !worker.running() && worker.isParked(), "clear did not park and invalidate mapping" );
    check( worker.resume() < 0, "cleared map resumed" );
    check( helper.prepare( fixture.open( OCAM2_BINNING2x2 ) ) == 0, "replacement map preparation failed" );
    check( worker.prepare( helper ) == 0 && worker.resume() == 0, "parked reprepare failed" );
    check( worker.run( base, raw.data(), Partition::halves ) == 0, "reprepared map rejected" );
    check( std::equal( expected.begin(), expected.end(), base ), "reprepared output differs" );
}

/// Race control requests against admitted jobs, then immediately reuse buffers before joining the worker.
/** \see MagAOX::app::ocamDescrambleParallel::requestPark
 * \see MagAOX::app::ocamDescrambleParallel::requestStop
 * \see MagAOX::app::ocamDescrambleParallel::inFlight
 */
void checkControlRaces( sdkFixture &fixture /**< [in,out] Real SDK reference owner. */ )
{
    ocamDescrambler helper;
    const auto      id = fixture.open( OCAM2_BINNING2x2 );
    check( helper.prepare( id ) == 0, "race map preparation failed" );
    std::vector<short> raw( helper.rawPixels() );
    std::vector<short> expected( helper.imagePixels() );
    std::vector<short> output( expected.size() );
    fillRaw( raw, 53 );
    unsigned number = 0;
    ocam2_descramble( id, &number, expected.data(), raw.data() );
    unsigned observedJobs = 0;

    for( bool stop : { false, true } )
    {
        for( unsigned iteration = 0; iteration < 24; ++iteration )
        {
            ocamDescrambleParallel worker;
            check( worker.prepare( helper ) == 0, "race worker preparation failed" );
            workerOwner owner( worker );
            check( worker.waitUntilStarted() && worker.resume() == 0, "race worker startup failed" );
            std::atomic<bool> begin{ false };
            std::atomic<bool> finish{ false };
            std::atomic<bool> observed{ false };
            std::thread       controller(
                [&]()
                {
                    begin.store( true, std::memory_order_release );

                    while( !worker.inFlight() && !finish.load( std::memory_order_acquire ) )
                    {
                        std::this_thread::yield();
                    }

                    observed.store( worker.inFlight(), std::memory_order_relaxed );

                    if( stop )
                    {
                        worker.requestStop();
                    }
                    else
                    {
                        worker.requestPark();
                    }
                } );

            while( !begin.load( std::memory_order_acquire ) )
            {
                std::this_thread::yield();
            }

            bool wholeFrames = true;

            for( unsigned job = 0; job < 256; ++job )
            {
                std::fill( output.begin(), output.end(), short( -777 ) );
                const int result = worker.run( output.data(), raw.data(), Partition::paired30 );
                wholeFrames      = wholeFrames && ( result == 0 ? output == expected
                                                                : std::all_of( output.begin(),
                                                                          output.end(),
                                                                          []( short value )
                                                                          {
                                                                              return value == -777;
                                                                          } ) );

                if( result < 0 )
                {
                    break;
                }
            }

            finish.store( true, std::memory_order_release );
            controller.join();
            observedJobs += observed.load() ? 1 : 0;
            check( wholeFrames, "control request caused partial frame or early error return" );
            worker.park();
            check( !worker.running() && !worker.inFlight() && worker.isParked(), "park left active buffer users" );
            std::fill( output.begin(), output.end(), short( -1234 ) );
            std::fill( raw.begin(), raw.end(), short( 321 ) );
            check( worker.run( output.data(), raw.data(), Partition::halves ) < 0,
                   "paused worker accepted another frame" );
            check( std::all_of( output.begin(),
                                output.end(),
                                []( short value )
                                {
                                    return value == -1234;
                                } ),
                   "completed or rejected job wrote reused buffers" );

            if( stop )
            {
                check( worker.stopped() && worker.resume() < 0, "stopped worker resumed" );
            }
            else
            {
                check( worker.resume() == 0, "park request prevented later resume" );
                check( worker.run( output.data(), raw.data(), Partition::paired28 ) == 0, "resumed frame failed" );
                check( std::all_of( output.begin(),
                                    output.end(),
                                    []( short value )
                                    {
                                        return value == 321;
                                    } ),
                       "resumed frame reused old source" );
            }

            fillRaw( raw, 53 );
        }
    }

    check( observedJobs > 0, "control tests never observed an admitted job" );
    std::cout << "Observed " << observedJobs << " admitted-job control races.\n";
}

} // namespace

/// Run host-only worker protocol checks without any camera, RT scheduler, or instrument cpuset operation.
int main()
{
    try
    {
        sdkFixture fixture;
        checkFrames( fixture );
        checkControlRaces( fixture );
        std::cout << "Passed " << checks << " parallel worker checks.\n";
        return 0;
    }
    catch( const std::exception &error )
    {
        std::cerr << "Parallel worker check failed after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
