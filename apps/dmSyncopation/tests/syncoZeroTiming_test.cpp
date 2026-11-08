/** \file syncoZeroTiming_test.cpp
 * \brief Exact-identity main-impact measurements for coordinated zero-command tests.
 * \ingroup dmSyncopation_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../syncoZeroTiming.hpp"

namespace libXWCTest
{
namespace dmSyncopationTest
{
using MagAOX::app::synco::ZeroTiming;

/// \cond ZERO_TIMING_TEST_FIXTURE
/// Construct one successful native SDK endpoint with independent external and monotonic clocks.
MagAOX::dmCompletion::Record zeroSdkRecord( uint64_t counter  = 1 /**< [in] Exact combined input counter. */,
                                            uint64_t sequence = 1 /**< [in] Native publisher attempt sequence. */ )
{
    MagAOX::dmCompletion::Record record;
    record.kind            = MagAOX::dmCompletion::Kind::sdk;
    record.state           = MagAOX::dmCompletion::State::complete;
    record.session         = 12345;
    record.epoch           = 1;
    record.sequence        = sequence;
    record.inputInode      = 1000;
    record.inputCounter    = counter;
    record.publicationSec  = 2000000000;
    record.publicationNsec = int64_t( counter );
    record.startedNs       = 1000000 + int64_t( counter ) * 500000;
    record.finishedNs      = record.startedNs + 107000;
    record.outcome         = MagAOX::dmTiming::Outcome::success;
    record.callbackResult  = 0;
    record.backendResult   = 0;
    return record;
}

/// Derive an exact reader endpoint whose entry precedes the same native SDK entry.
MagAOX::dmCompletion::Record
zeroReaderRecord( const MagAOX::dmCompletion::Record &sdk /**< [in] Native endpoint for the same combined command. */ )
{
    auto record  = sdk;
    record.kind  = MagAOX::dmCompletion::Kind::readers;
    record.state = MagAOX::dmCompletion::State::released;
    record.startedNs -= 17000;
    record.finishedNs = record.startedNs + 22000;
    return record;
}
/// \endcond

/// Exact main, reader and SDK identities form independent duration populations.
TEST_CASE( "zero trial joins exact main SDK and reader identities", "[dmSyncopation][zeroTiming]" )
{
    ZeroTiming timing;
    const auto sdk = zeroSdkRecord();
    timing.main( ZeroTiming::key( sdk ), 80, sdk.startedNs - 32000 );
    timing.reader( zeroReaderRecord( sdk ) );
    timing.sdk( sdk, sdk.finishedNs + 1000 );
    const auto &summary = timing.summary();
    REQUIRE( summary.classified == 1 );
    REQUIRE( summary.matched == 1 );
    REQUIRE( summary.readerMatched == 1 );
    REQUIRE( summary.primaryToSdk.count == 1 );
    REQUIRE( summary.primaryToSdk.meanNs == 32000 );
    REQUIRE( summary.controllerToSdk.meanNs == 17000 );
    REQUIRE( summary.sdkService.meanNs == 107000 );
    REQUIRE( summary.primaryCounter == 80 );
    REQUIRE( summary.combinedCounter == sdk.inputCounter );
    REQUIRE( summary.invalid == 0 );
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::ZeroTiming::main;
    MagAOX::app::synco::ZeroTiming::reader;
    MagAOX::app::synco::ZeroTiming::sdk;
    MagAOX::app::synco::ZeroTiming::summary;
#endif
    // clang-format on
}

/// Out-of-order endpoint captures wait for classification rather than inventing a main.
TEST_CASE( "zero trial supports endpoint arrival before main classification", "[dmSyncopation][zeroTiming]" )
{
    ZeroTiming timing;
    const auto sdk = zeroSdkRecord();
    timing.sdk( sdk, sdk.finishedNs );
    timing.reader( zeroReaderRecord( sdk ) );
    REQUIRE( timing.summary().matched == 0 );
    timing.main( ZeroTiming::key( sdk ), 80, sdk.startedNs - 30000 );
    REQUIRE( timing.summary().matched == 1 );
    REQUIRE( timing.summary().readerMatched == 1 );
    REQUIRE( timing.summary().primaryToSdk.meanNs == 30000 );
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::ZeroTiming::main;
    MagAOX::app::synco::ZeroTiming::reader;
    MagAOX::app::synco::ZeroTiming::sdk;
#endif
    // clang-format on
}

/// Every identity component is required, including producer publication time.
TEST_CASE( "zero trial never joins nearby mismatched commands", "[dmSyncopation][zeroTiming]" )
{
    ZeroTiming     timing;
    const auto     sdk   = zeroSdkRecord();
    auto           wrong = ZeroTiming::key( sdk );
    const unsigned field = GENERATE( 0, 1, 2, 3 );

    switch( field )
    {
    case 0:
        ++wrong.inode;
        break;
    case 1:
        ++wrong.counter;
        break;
    case 2:
        ++wrong.sec;
        break;
    case 3:
        ++wrong.nsec;
        break;
    }

    timing.main( wrong, 80, sdk.startedNs - 30000 );
    timing.sdk( sdk, sdk.finishedNs );
    timing.reader( zeroReaderRecord( sdk ) );
    REQUIRE( timing.summary().matched == 0 );
    REQUIRE( timing.summary().primaryToSdk.count == 0 );
    REQUIRE( timing.summary().controllerToSdk.count == 0 );
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::ZeroTiming::equal;
    MagAOX::app::synco::ZeroTiming::key;
#endif
    // clang-format on
}

/// Late primary observation retains a negative signed delay rather than clipping it.
TEST_CASE( "zero trial keeps signed observer to SDK offsets", "[dmSyncopation][zeroTiming]" )
{
    ZeroTiming timing;
    const auto sdk = zeroSdkRecord();
    timing.main( ZeroTiming::key( sdk ), 80, sdk.startedNs + 1000 );
    timing.sdk( sdk, sdk.finishedNs );
    REQUIRE( timing.summary().primaryToSdk.meanNs == -1000 );
    REQUIRE( timing.summary().primaryToSdk.minNs == -1000 );
    REQUIRE( timing.summary().primaryToSdk.maxNs == -1000 );
    REQUIRE( timing.summary().invalid == 0 );
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::ZeroTiming::main;
    MagAOX::app::synco::ZeroTiming::sdk;
#endif
    // clang-format on
}

/// Torn, future and wrong-protocol captures supply no native SDK timing.
TEST_CASE( "zero trial rejects unusable SDK evidence", "[dmSyncopation][zeroTiming]" )
{
    ZeroTiming timing;
    auto       sdk = zeroSdkRecord();
    timing.main( ZeroTiming::key( sdk ), 80, sdk.startedNs - 30000 );
    const unsigned fault = GENERATE( 0, 1, 2, 3 );

    if( fault == 0 )
    {
        timing.sdk( sdk, sdk.finishedNs, false );
    }
    else if( fault == 1 )
    {
        timing.sdk( sdk, sdk.finishedNs - 1 );
    }
    else if( fault == 2 )
    {
        sdk.kind = MagAOX::dmCompletion::Kind::readers;
        timing.sdk( sdk, sdk.finishedNs );
    }
    else
    {
        sdk.finishedNs = sdk.startedNs - 1;
        timing.sdk( sdk, sdk.startedNs + 107000 );
    }

    REQUIRE( timing.summary().invalid == 1 );
    REQUIRE( timing.summary().matched == 0 );
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::ZeroTiming::sdk;
#endif
    // clang-format on
}

/// Duplicates cannot increase matched populations, and changed duplicates are invalid.
TEST_CASE( "zero trial deduplicates endpoint and classification observations", "[dmSyncopation][zeroTiming]" )
{
    ZeroTiming timing;
    auto       sdk    = zeroSdkRecord();
    auto       reader = zeroReaderRecord( sdk );

    for( unsigned repeat = 0; repeat < 2; ++repeat )
    {
        timing.main( ZeroTiming::key( sdk ), 80, sdk.startedNs - 30000 );
        timing.reader( reader );
        timing.sdk( sdk, sdk.finishedNs + repeat );
    }

    REQUIRE( timing.summary().matched == 1 );
    REQUIRE( timing.summary().readerMatched == 1 );
    REQUIRE( timing.summary().duplicates == 3 );
    ++sdk.finishedNs;
    timing.sdk( sdk, sdk.finishedNs );
    ++reader.finishedNs;
    timing.reader( reader );
    REQUIRE( timing.summary().invalid == 2 );
    REQUIRE( timing.summary().matched == 1 );
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::ZeroTiming::main;
    MagAOX::app::synco::ZeroTiming::reader;
    MagAOX::app::synco::ZeroTiming::sdk;
#endif
    // clang-format on
}

/// Failed native results are counted but never presented as successful main submission.
TEST_CASE( "zero trial records failed native outcomes", "[dmSyncopation][zeroTiming]" )
{
    ZeroTiming timing;
    auto       sdk = zeroSdkRecord();
    timing.main( ZeroTiming::key( sdk ), 80, sdk.startedNs - 30000 );
    sdk.state          = MagAOX::dmCompletion::State::error;
    sdk.outcome        = MagAOX::dmTiming::Outcome::failed;
    sdk.callbackResult = -1;
    sdk.backendResult  = 12;
    timing.sdk( sdk, sdk.finishedNs );
    timing.sdk( sdk, sdk.finishedNs );
    REQUIRE( timing.summary().failed == 1 );
    REQUIRE( timing.summary().duplicates == 1 );
    REQUIRE( timing.summary().matched == 0 );
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::ZeroTiming::sdk;
#endif
    // clang-format on
}

/// Source changes revoke pending history while SDK sequence gaps remain explicit.
TEST_CASE( "zero trial accounts for native source epochs and missing sequences", "[dmSyncopation][zeroTiming]" )
{
    ZeroTiming timing;
    const auto first = zeroSdkRecord();
    timing.sdk( first, first.finishedNs );
    auto third = zeroSdkRecord( 3, 4 );
    timing.sdk( third, third.finishedNs );
    REQUIRE( timing.summary().sdkGaps == 2 );
    auto next = zeroSdkRecord( 4, 1 );
    timing.main( ZeroTiming::key( next ), 81, next.startedNs - 30000 );
    next.session = 67890;
    timing.sdk( next, next.finishedNs );
    REQUIRE( timing.summary().sourceResets == 1 );
    REQUIRE( timing.summary().evicted == 1 );
    REQUIRE( timing.summary().matched == 0 );
    timing.main( ZeroTiming::key( next ), 81, next.startedNs - 30000 );
    REQUIRE( timing.summary().matched == 1 );
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::ZeroTiming::sdk;
    MagAOX::app::synco::ZeroTiming::main;
#endif
    // clang-format on
}

/// Fixed-capacity overwrite and trial reset have explicit, testable accounting.
TEST_CASE( "zero trial keeps bounded pending measurement history", "[dmSyncopation][zeroTiming]" )
{
    ZeroTiming timing;
    const auto first       = zeroSdkRecord();
    const auto replacement = zeroSdkRecord( 257, 257 );
    timing.main( ZeroTiming::key( first ), 80, first.startedNs - 30000 );
    timing.main( ZeroTiming::key( replacement ), 336, replacement.startedNs - 30000 );
    REQUIRE( timing.summary().evicted == 1 );
    timing.sdk( replacement, replacement.finishedNs );
    timing.reader( zeroReaderRecord( replacement ) );
    REQUIRE( timing.summary().matched == 1 );
    timing.reset();
    REQUIRE( timing.summary().classified == 0 );
    REQUIRE( timing.summary().matched == 0 );
    REQUIRE( timing.summary().sdkGaps == 0 );
    timing.main( ZeroTiming::key( first ), 80, first.startedNs - 30000 );
    timing.sdk( first, first.finishedNs );
    REQUIRE( timing.summary().matched == 1 );
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::ZeroTiming::reset;
    MagAOX::app::synco::ZeroTiming::main;
#endif
    // clang-format on
}

/// Delayed old native generations cannot evict current classified commands.
TEST_CASE( "zero trial rejects stale native generations", "[dmSyncopation][zeroTiming]" )
{
    ZeroTiming timing;
    auto       sdk   = zeroSdkRecord();
    sdk.epoch        = 2;
    sdk.publisherPid = 99;
    timing.sdk( sdk, sdk.finishedNs );
    const auto next = zeroSdkRecord( 2, 2 );
    timing.main( ZeroTiming::key( next ), 81, next.startedNs - 30000 );
    const unsigned fault = GENERATE( 0, 1, 2 );

    if( fault == 0 )
    {
        --sdk.session;
    }
    else if( fault == 1 )
    {
        --sdk.epoch;
    }
    else
    {
        --sdk.publisherPid;
    }

    timing.sdk( sdk, next.finishedNs );
    REQUIRE( timing.summary().invalid == 1 );
    REQUIRE( timing.summary().sourceResets == 0 );
    REQUIRE( timing.summary().evicted == 0 );
    auto correct         = next;
    correct.epoch        = 2;
    correct.publisherPid = 99;
    timing.sdk( correct, correct.finishedNs );
    REQUIRE( timing.summary().matched == 1 );
    // clang-format off
#ifdef DMSYNCOPATION_TEST_DOXYGEN_REF
    MagAOX::app::synco::ZeroTiming::sdk;
#endif
    // clang-format on
}

} // namespace dmSyncopationTest
} // namespace libXWCTest
