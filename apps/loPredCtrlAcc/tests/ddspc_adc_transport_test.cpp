/** \file ddspc_adc_transport_test.cpp
 * \brief Actual ADC worker recovery against a private offline ImageStreamIO API double.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../adc_reader.hpp"
#include "adc_transport_fixture.hpp"
#include <chrono>
#include <limits>
#include <thread>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond ADC_TRANSPORT_WAIT
namespace
{
/// Wait at most one second for an actual offline worker's state; never opens a native stream.
template <typename Predicate>
bool await_adc( Predicate predicate /**< [in] Thread-safe diagnostic predicate. */ )
{
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds( 1 );
    while( std::chrono::steady_clock::now() < end )
    {
        if( predicate() )
        {
            return true;
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }
    return false;
}
} // namespace
/// \endcond

/** \brief The actual worker attaches, copies raw metadata, disconnects, reconnects and releases its reader slot. */
TEST_CASE( "Native ADC worker recovery uses an offline ImageStreamIO API double", "[loPredCtrlAcc][adc][threaded]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcReader::start;
    DDSPC::AdcReader::stop;
    DDSPC::AdcReader::samples;
    DDSPC::AdcReader::copy_after;
#endif
    // clang-format on
    for( const auto depth : { 1u, 4u } )
    {
        AdcTransportFixture fixture( "ddspc-adc-worker", depth );
        DDSPC::AdcConfig    config;
        config.m_stream   = "ddspc-adc-worker";
        config.m_maxAgeNs = 2000000000;
        DDSPC::AdcReader reader( config );
        reader.start();
        REQUIRE( await_adc( [&] { return reader.status().m_valid; } ) );
        REQUIRE( reader.samples().size() == 1 );
        REQUIRE( reader.status().m_latest.m_frame.m_raw[0] == 2049 );
        REQUIRE( reader.status().m_latest.m_acceleration[1] == Approx( -.0083799 / 1.02 ) );
        REQUIRE( reader.status().m_source.m_depth == depth );
        std::array<DDSPC::AdcSample, 2> storage;
        auto batch = reader.copy_after( {}, storage, std::numeric_limits<std::int64_t>::max() );
        REQUIRE( batch.m_count == 1 );
        REQUIRE( storage[0].m_frame.m_raw == reader.samples()[0].m_frame.m_raw );
        REQUIRE( batch.m_status.m_source.m_depth == depth );
        REQUIRE( reader.copy_after( batch.m_next, storage, std::numeric_limits<std::int64_t>::max() ).m_count == 0 );
        fixture.disconnect();
        REQUIRE( await_adc( [&] { return reader.status().m_state == DDSPC::AdcState::Disconnected; } ) );
        REQUIRE( reader.samples().empty() );
        fixture.reconnect();
        REQUIRE( await_adc( [&] { return reader.status().m_valid && reader.status().m_generation == 2; } ) );
        fixture.replace();
        REQUIRE( await_adc( [&] { return reader.status().m_valid && reader.status().m_generation == 3; } ) );
        REQUIRE( reader.samples().size() == 1 );
        batch = reader.copy_after( batch.m_next, storage, std::numeric_limits<std::int64_t>::max() );
        REQUIRE( batch.m_discontinuity );
        REQUIRE( batch.m_count == 1 );
        REQUIRE( batch.m_missing == 0 );
        REQUIRE( storage[0].m_generation == 3 );
        reader.stop();
        REQUIRE( fixture.reader() == 0 );
        REQUIRE_FALSE( reader.status().m_valid );
        REQUIRE( reader.samples().empty() );
        reader.stop();
    }
}

/** \brief An absent input never creates native resources or prevents the worker from joining at shutdown. */
TEST_CASE( "Native ADC worker stops cleanly while its input is absent", "[loPredCtrlAcc][adc][threaded]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::AdcReader::start;
    DDSPC::AdcReader::stop;
#endif
    // clang-format on
    DDSPC::AdcConfig config;
    config.m_stream = "ddspc-adc-absent"; // Unregistered API-double name; never the native namespace.
    DDSPC::AdcReader reader( config );
    reader.start();
    std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
    const auto before = std::chrono::steady_clock::now();
    reader.stop();
    REQUIRE( std::chrono::steady_clock::now() - before < std::chrono::seconds( 1 ) );
    REQUIRE( reader.status().m_state == DDSPC::AdcState::Disconnected );
    REQUIRE( reader.status().m_generation == 0 );
    REQUIRE_FALSE( reader.status().m_valid );
    REQUIRE( reader.samples().empty() );
}

} // namespace libXWCTest::loPredCtrlAccTest
