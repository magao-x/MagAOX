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
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_fpsSource, modeamps, fps )
    }

    void setWindowSizes( cbIndexT tsSize, cbIndexT meanSize )
    {
        m_tsSize   = tsSize;
        m_meanSize = meanSize;
        m_tsPtrs.resize( m_tsSize );
        m_meanPtrs.resize( m_meanSize );
    }

    void setCircBuffEntries( cbIndexT entries )
    {
        m_ampCircBuff.maxEntries( entries );
    }

    void pushSample( realT *ptr )
    {
        m_ampCircBuff.nextEntry( ptr );
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
    modalPSDs::setCallBack_m_indiP_fpsSource( pcf::IndiProperty() );
    modalPSDs::loadPsdInputWindows( *(ampCircBuffT::snapshotT *)nullptr );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( modalPSDs, psdTime );
    XWCTEST_INDI_NEW_CALLBACK( modalPSDs, psdAvgTime );
    XWCTEST_INDI_SET_CALLBACK( modalPSDs, m_indiP_fpsSource, modeamps, fps );
}

} // namespace modalPSDsTest

} // namespace libXWCTest

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
