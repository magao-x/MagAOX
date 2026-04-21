/** \file modalPSDs_test.cpp
 * \brief Catch2 tests for the modalPSDs app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup modalPSDs_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../modalPSDs.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup modalPSDs_unit_test modalPSDs Unit Tests
 * \brief Unit tests for the modalPSDs application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `modalPSDs` unit tests.
/** \ingroup modalPSDs_unit_test
 */
namespace modalPSDsTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class modalPSDs_test : public modalPSDs
{

  public:
    using snapshotT = ampCircBuffT::snapshotT;

    modalPSDs_test( const std::string &device )
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP( psdTime );
        XWCTEST_SETUP_INDI_NEW_PROP( psdAvgTime );
        XWCTEST_SETUP_INDI_NEW_PROP( meanTime );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_fpsSource, modeamps, fps );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_loop, loopdev, loop_state );
    }

    void setWindowSizes( cbIndexT tsSize, cbIndexT meanSize )
    {
        m_tsSize   = tsSize;
        m_meanSize = meanSize;
        m_tsPtrs.resize( m_tsSize );
        m_meanPtrs.resize( m_meanSize );
    }

    void setModeCount( size_t nModes )
    {
        m_nModes = nModes;
    }

    void setCircBuffEntries( cbIndexT entries )
    {
        m_ampCircBuff.maxEntries( entries );
    }

    cbIndexT circBuffSize() const
    {
        return m_ampCircBuff.size();
    }

    void pushSample( realT *ptr )
    {
        m_ampCircBuff.nextEntry( ptr );
    }

    int processImageForTest( realT *ptr )
    {
        dev::shmimT dummy;
        return processImage( ptr, dummy );
    }

    bool loadWindows( snapshotT &sn )
    {
        return loadPsdInputWindows( sn );
    }

    realT tsValue( size_t n ) const
    {
        return *( m_tsPtrs.at( n ) );
    }

    realT meanValue( size_t n ) const
    {
        return *( m_meanPtrs.at( n ) );
    }

    cbIndexT latestRef( const snapshotT &sn, cbIndexT count ) const
    {
        return latestWindowRefEntry( sn, count );
    }

    cbIndexT precedingRef( const snapshotT &sn, cbIndexT refEntry, cbIndexT count ) const
    {
        return precedingWindowRefEntry( sn, refEntry, count );
    }

    void setPSDTiming( realT psdTime, realT psdAvgTime, realT meanTime, realT psdOverlapFraction )
    {
        m_psdTime.store( psdTime );
        m_psdAvgTime.store( psdAvgTime );
        m_meanTime.store( meanTime );
        m_psdOverlapFraction = psdOverlapFraction;
    }

    void setPSDHistoryFloor( int nPSDHistory )
    {
        m_nPSDHistory = nPSDHistory;
    }

    void
    configureLoopState( const std::string &device, const std::string &property, const std::string &element = "toggle" )
    {
        m_loopStateDevice   = device;
        m_loopStateProperty = property;
        m_loopStateElement  = element;
        m_useLoopState      = !m_loopStateDevice.empty();
        m_loopClosed.store( m_useLoopState == false, std::memory_order_release );

        m_indiP_loop.setDevice( device );
        m_indiP_loop.setName( property );
    }

    void setLoopClosedForTest( bool loopClosed )
    {
        m_loopClosed.store( loopClosed, std::memory_order_release );
    }

    bool acceptLoopStateFrameForTest() const
    {
        return acceptLoopStateFrame();
    }

    bool usesLoopStateForTest() const
    {
        return m_useLoopState;
    }

    bool loopClosedForTest() const
    {
        return m_loopClosed.load( std::memory_order_acquire );
    }

    int desiredPSDAverageCountForTest() const
    {
        return desiredPSDAverageCount();
    }

    cbIndexT desiredMeanSampleCountForTest( realT fps ) const
    {
        return desiredMeanSampleCount( fps );
    }

    cbIndexT requiredInputHistoryDepthForTest()
    {
        return requiredInputHistoryDepth();
    }

    static cbIndexT circularEntryAdvanceForTest( cbIndexT from, cbIndexT to, cbIndexT maxEntries )
    {
        return circularEntryAdvance( from, to, maxEntries );
    }

    std::vector<double> recomputeMeanSumsForTest() const
    {
        std::vector<double> meanSums;
        recomputeMeanSums( meanSums );
        return meanSums;
    }

    std::vector<realT> cacheMeanHeadForTest( cbIndexT count ) const
    {
        std::vector<realT> meanHeadCache;
        cacheMeanHead( meanHeadCache, count );
        return meanHeadCache;
    }

    void rollMeanSumsForTest( std::vector<double>      &meanSums,
                              const std::vector<realT> &meanHeadCache,
                              cbIndexT                  advance ) const
    {
        rollMeanSums( meanSums, meanHeadCache, advance );
    }

    static void updatePlaneSumForTest( std::vector<double>      &planeSum,
                                       const std::vector<realT> &addPlane,
                                       const std::vector<realT> *removePlane = nullptr )
    {
        updatePlaneSum(
            planeSum, addPlane.data(), removePlane == nullptr ? nullptr : removePlane->data(), addPlane.size() );
    }

    uint32_t rawPSDHistoryDepthForTest() const
    {
        return rawPSDHistoryDepth();
    }

    uint32_t publishedRawPSDHistoryDepthForTest() const
    {
        return publishedRawPSDHistoryDepth();
    }
};
/// \endcond

/// Verify modalPSDs callback validation and PSD-window extraction logic behave as expected.
/**
 * \ingroup modalPSDs_unit_test
 */
SCENARIO( "INDI Callbacks", "[modalPSDs]" )
{
    // clang-format off
    #ifdef MODALPSDS_TEST_DOXYGEN_REF
    modalPSDs::newCallBack_m_indiP_psdTime( pcf::IndiProperty() );
    modalPSDs::newCallBack_m_indiP_psdAvgTime( pcf::IndiProperty() );
    modalPSDs::newCallBack_m_indiP_meanTime( pcf::IndiProperty() );
    modalPSDs::setCallBack_m_indiP_fpsSource( pcf::IndiProperty() );
    modalPSDs::loadPsdInputWindows( *(ampCircBuffT::snapshotT *)nullptr );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( modalPSDs, psdTime );
    XWCTEST_INDI_NEW_CALLBACK( modalPSDs, psdAvgTime );
    XWCTEST_INDI_NEW_CALLBACK( modalPSDs, meanTime );
    XWCTEST_INDI_SET_CALLBACK( modalPSDs, m_indiP_fpsSource, modeamps, fps );
    XWCTEST_INDI_SET_CALLBACK( modalPSDs, m_indiP_loop, loopdev, loop_state );
}

/// Verify modalPSDs derives PSD averaging depth from psdTime and psdAvgTime while mean sizing follows meanTime.
/**
 * \ingroup modalPSDs_unit_test
 */
TEST_CASE( "modalPSDs PSD averaging and mean windows are decoupled", "[modalPSDs]" )
{
    modalPSDs_test app( "modalPSDs_test" );

    // clang-format off
    #ifdef MODALPSDS_TEST_DOXYGEN_REF
    modalPSDs::desiredPSDAverageCount();
    modalPSDs::desiredMeanSampleCount( 0 );
    modalPSDs::requiredInputHistoryDepth();
    modalPSDs::rawPSDHistoryDepth();
    #endif
    // clang-format on

    app.setPSDHistoryFloor( 100 );

    app.setPSDTiming( 1.0F, 10.0F, 60.0F, 0.5F );
    app.setWindowSizes( 2000, app.desiredMeanSampleCountForTest( 1000.0F ) );
    REQUIRE( app.desiredPSDAverageCountForTest() == 20 );
    REQUIRE( app.desiredMeanSampleCountForTest( 1000.0F ) == 60000 );
    REQUIRE( app.requiredInputHistoryDepthForTest() == 62002 );
    REQUIRE( app.rawPSDHistoryDepthForTest() == 0 );
    REQUIRE( app.publishedRawPSDHistoryDepthForTest() == 100 );

    app.setPSDTiming( 1.0F, 60.0F, 60.0F, 0.5F );
    app.setWindowSizes( 2000, app.desiredMeanSampleCountForTest( 1000.0F ) );
    REQUIRE( app.desiredPSDAverageCountForTest() == 120 );
    REQUIRE( app.desiredMeanSampleCountForTest( 1000.0F ) == 60000 );
    REQUIRE( app.requiredInputHistoryDepthForTest() == 62002 );
    REQUIRE( app.rawPSDHistoryDepthForTest() == 21 );
    REQUIRE( app.publishedRawPSDHistoryDepthForTest() == 100 );

    app.setPSDTiming( 1.0F, 60.0F, 15.0F, 0.5F );
    app.setWindowSizes( 2000, app.desiredMeanSampleCountForTest( 1000.0F ) );
    REQUIRE( app.desiredPSDAverageCountForTest() == 120 );
    REQUIRE( app.desiredMeanSampleCountForTest( 1000.0F ) == 15000 );
    REQUIRE( app.requiredInputHistoryDepthForTest() == 17002 );
    REQUIRE( app.rawPSDHistoryDepthForTest() == 21 );
}

SCENARIO( "PSD input windows come from one validated snapshot", "[modalPSDs]" )
{
    GIVEN( "A full fixed-size circular buffer and configured PSD window lengths" )
    {
        modalPSDs_test app( "modalPSDs_test" );
        app.setWindowSizes( 3, 2 );
        app.setCircBuffEntries( 8 );

        modalPSDs::realT samples[9];
        for( int n = 0; n < 9; ++n )
        {
            samples[n] = static_cast<modalPSDs::realT>( n );
        }

        for( int n = 0; n < 8; ++n )
        {
            app.pushSample( &samples[n] );
        }

        WHEN( "The PSD and mean windows are loaded before wraparound" )
        {
            modalPSDs_test::snapshotT sn;

            REQUIRE( app.loadWindows( sn ) );
            REQUIRE( sn.maxEntries == 8 );
            REQUIRE( sn.validEntries == 8 );

            REQUIRE( app.tsValue( 0 ) == 4 );
            REQUIRE( app.tsValue( 1 ) == 5 );
            REQUIRE( app.tsValue( 2 ) == 6 );

            REQUIRE( app.meanValue( 0 ) == 2 );
            REQUIRE( app.meanValue( 1 ) == 3 );
        }

        WHEN( "The buffer has wrapped once" )
        {
            app.pushSample( &samples[8] );

            modalPSDs_test::snapshotT sn;

            REQUIRE( app.loadWindows( sn ) );
            REQUIRE( sn.latest == 0 );
            REQUIRE( sn.earliest == 1 );

            REQUIRE( app.tsValue( 0 ) == 5 );
            REQUIRE( app.tsValue( 1 ) == 6 );
            REQUIRE( app.tsValue( 2 ) == 7 );

            REQUIRE( app.meanValue( 0 ) == 3 );
            REQUIRE( app.meanValue( 1 ) == 4 );
        }
    }
}

/// Verify modalPSDs can read both windows when the circular buffer is sized to the exact required history depth.
/**
 * \ingroup modalPSDs_unit_test
 */
TEST_CASE( "modalPSDs PSD input windows use the exact required history depth", "[modalPSDs]" )
{
    modalPSDs_test app( "modalPSDs_test_required_history_depth" );

    app.setWindowSizes( 3, 2 );
    app.setCircBuffEntries( 7 );

    modalPSDs::realT samples[7];
    for( int n = 0; n < 7; ++n )
    {
        samples[n] = static_cast<modalPSDs::realT>( n );
        app.pushSample( &samples[n] );
    }

    modalPSDs_test::snapshotT sn;

    REQUIRE( app.loadWindows( sn ) );
    REQUIRE( sn.maxEntries == 7 );
    REQUIRE( sn.validEntries == 7 );

    REQUIRE( app.meanValue( 0 ) == 1 );
    REQUIRE( app.meanValue( 1 ) == 2 );

    REQUIRE( app.tsValue( 0 ) == 3 );
    REQUIRE( app.tsValue( 1 ) == 4 );
    REQUIRE( app.tsValue( 2 ) == 5 );
}

/// Verify modalPSDs rolling mean updates match a full recomputation after one overlap advance.
/**
 * \ingroup modalPSDs_unit_test
 */
TEST_CASE( "modalPSDs rolling mean update matches full recompute", "[modalPSDs]" )
{
    modalPSDs_test app( "modalPSDs_test_rolling_mean" );

    // clang-format off
    #ifdef MODALPSDS_TEST_DOXYGEN_REF
    modalPSDs::circularEntryAdvance( 0, 0, 0 );
    modalPSDs::recomputeMeanSums( *(std::vector<double> *)nullptr );
    modalPSDs::rollMeanSums( *(std::vector<double> *)nullptr, *(std::vector<modalPSDs::realT> *)nullptr, 0 );
    modalPSDs::cacheMeanHead( *(std::vector<modalPSDs::realT> *)nullptr, 0 );
    #endif
    // clang-format on

    app.setModeCount( 2 );
    app.setWindowSizes( 4, 4 );
    app.setCircBuffEntries( 10 );

    modalPSDs::realT samples[12][2];
    for( int n = 0; n < 12; ++n )
    {
        samples[n][0] = static_cast<modalPSDs::realT>( n );
        samples[n][1] = static_cast<modalPSDs::realT>( 100 + 2 * n );
    }

    for( int n = 0; n < 10; ++n )
    {
        app.pushSample( samples[n] );
    }

    modalPSDs_test::snapshotT firstSnap;
    REQUIRE( app.loadWindows( firstSnap ) );

    auto firstSums = app.recomputeMeanSumsForTest();
    auto firstHead = app.cacheMeanHeadForTest( 2 );

    modalPSDs::cbIndexT firstTsRef   = app.latestRef( firstSnap, 4 );
    modalPSDs::cbIndexT firstMeanRef = app.precedingRef( firstSnap, firstTsRef, 4 );

    app.pushSample( samples[10] );
    app.pushSample( samples[11] );

    modalPSDs_test::snapshotT secondSnap;
    REQUIRE( app.loadWindows( secondSnap ) );

    modalPSDs::cbIndexT secondTsRef   = app.latestRef( secondSnap, 4 );
    modalPSDs::cbIndexT secondMeanRef = app.precedingRef( secondSnap, secondTsRef, 4 );

    REQUIRE( modalPSDs_test::circularEntryAdvanceForTest( firstMeanRef, secondMeanRef, secondSnap.maxEntries ) == 2 );

    auto rolledSums = firstSums;
    app.rollMeanSumsForTest( rolledSums, firstHead, 2 );

    auto recomputedSums = app.recomputeMeanSumsForTest();

    REQUIRE( rolledSums.size() == recomputedSums.size() );
    for( size_t n = 0; n < rolledSums.size(); ++n )
    {
        REQUIRE( rolledSums[n] == Approx( recomputedSums[n] ) );
    }
}

/// Verify modalPSDs rolling PSD-sum updates match a full recomputation when one plane enters and one leaves.
/**
 * \ingroup modalPSDs_unit_test
 */
TEST_CASE( "modalPSDs rolling PSD sum update matches full recompute", "[modalPSDs]" )
{
    // clang-format off
    #ifdef MODALPSDS_TEST_DOXYGEN_REF
    modalPSDs::updatePlaneSum( *(std::vector<double> *)nullptr, (const modalPSDs::realT *)nullptr, (const modalPSDs::realT *)nullptr, 0 );
    #endif
    // clang-format on

    std::vector<double>           rollingSum{ 12.0, 15.0, 18.0 };
    std::vector<double>           recomputedSum( 3, 0.0 );
    std::vector<modalPSDs::realT> oldPlane{ 1.0F, 2.0F, 3.0F };
    std::vector<modalPSDs::realT> keepPlaneA{ 4.0F, 5.0F, 6.0F };
    std::vector<modalPSDs::realT> keepPlaneB{ 7.0F, 8.0F, 9.0F };
    std::vector<modalPSDs::realT> newPlane{ 10.0F, 11.0F, 12.0F };

    modalPSDs_test::updatePlaneSumForTest( rollingSum, newPlane, &oldPlane );

    modalPSDs_test::updatePlaneSumForTest( recomputedSum, keepPlaneA );
    modalPSDs_test::updatePlaneSumForTest( recomputedSum, keepPlaneB );
    modalPSDs_test::updatePlaneSumForTest( recomputedSum, newPlane );

    REQUIRE( rollingSum.size() == recomputedSum.size() );
    for( size_t n = 0; n < rollingSum.size(); ++n )
    {
        REQUIRE( rollingSum[n] == Approx( recomputedSum[n] ) );
    }
}

/// Verify modalPSDs optionally gates PSD ingestion on the configured loop-state property.
/**
 * \ingroup modalPSDs_unit_test
 */
TEST_CASE( "modalPSDs loop-state gating is optional and blocks open-loop frames", "[modalPSDs]" )
{
    modalPSDs_test app( "modalPSDs_test_loop_state" );

    // clang-format off
    #ifdef MODALPSDS_TEST_DOXYGEN_REF
    modalPSDs::acceptLoopStateFrame();
    modalPSDs::setCallBack_m_indiP_loop( pcf::IndiProperty() );
    #endif
    // clang-format on

    modalPSDs::realT sample[1] = { 1.0F };

    app.setCircBuffEntries( 4 );
    REQUIRE( app.usesLoopStateForTest() == false );
    REQUIRE( app.acceptLoopStateFrameForTest() == true );
    REQUIRE( app.processImageForTest( sample ) == 0 );
    REQUIRE( app.circBuffSize() == 1 );

    app.setCircBuffEntries( 4 );
    app.configureLoopState( "loopdev", "loop_state" );
    REQUIRE( app.usesLoopStateForTest() == true );
    REQUIRE( app.loopClosedForTest() == false );
    REQUIRE( app.acceptLoopStateFrameForTest() == false );
    REQUIRE( app.processImageForTest( sample ) == 0 );
    REQUIRE( app.circBuffSize() == 0 );

    app.setLoopClosedForTest( true );
    REQUIRE( app.loopClosedForTest() == true );
    REQUIRE( app.acceptLoopStateFrameForTest() == true );
    REQUIRE( app.processImageForTest( sample ) == 0 );
    REQUIRE( app.circBuffSize() == 1 );
}

SCENARIO( "Snapshot-based circular-buffer loads reject stale snapshots", "[modalPSDs]" )
{
    GIVEN( "A full fixed-size circular buffer" )
    {
        modalPSDs::ampCircBuffT buff;
        buff.maxEntries( 4 );

        modalPSDs::realT samples[5];
        for( int n = 0; n < 5; ++n )
        {
            samples[n] = static_cast<modalPSDs::realT>( n );
        }

        for( int n = 0; n < 4; ++n )
        {
            buff.nextEntry( &samples[n] );
        }

        modalPSDs::realT *dest[2] = { nullptr, nullptr };

        WHEN( "The producer has not advanced" )
        {
            auto sn = buff.snapshot();

            REQUIRE( buff.loadSequence( sn, 1, 2, dest ) );
            REQUIRE( *dest[0] == 1 );
            REQUIRE( *dest[1] == 2 );
        }

        WHEN( "The producer advances after the snapshot" )
        {
            auto sn = buff.snapshot();
            buff.nextEntry( &samples[4] );

            REQUIRE_FALSE( buff.loadSequence( sn, 1, 2, dest ) );
        }
    }
}

//} //namespace modalPSDs_test

} // namespace modalPSDsTest

} // namespace libXWCTest
