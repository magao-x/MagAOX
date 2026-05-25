/** \file observerCtrl_test.cpp
 * \brief Catch2 tests for the observerCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup observerCtrl_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../observerCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup observerCtrl_unit_test observerCtrl Unit Tests
 * \brief Unit tests for the observerCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `observerCtrl` unit tests.
/** \ingroup observerCtrl_unit_test
 */
namespace observerCtrlTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class observerCtrl_test : public observerCtrl
{
  public:
    observerCtrl_test( const std::string &device )
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP( observers );
        XWCTEST_SETUP_INDI_NEW_PROP( obsName );
        XWCTEST_SETUP_INDI_NEW_PROP( observing );
        XWCTEST_SETUP_INDI_NEW_PROP( sws );
    }

    void configureStreamWriters( const std::vector<std::string> &writers,
                                 const std::vector<std::string> &defWriters = std::vector<std::string>() )
    {
        m_streamWriters    = writers;
        m_defStreamWriters = defWriters;

        m_indiP_sws = pcf::IndiProperty( pcf::IndiProperty::Switch );
        m_indiP_sws.setDevice( m_configName );
        m_indiP_sws.setName( "writers" );
        m_indiP_sws.setPerm( pcf::IndiProperty::ReadWrite );
        m_indiP_sws.setState( pcf::IndiProperty::Idle );
        m_indiP_sws.setRule( pcf::IndiProperty::AnyOfMany );

        m_indiP_streamWriterWriting.clear();
        m_streamWriterSelectable.clear();
        m_streamWriterDevices.clear();
        m_streamWriterWriting.clear();
        m_streamWriterWritingKnown.clear();
        m_streamWriterStartedByObserver.clear();
        m_indiSetCallBacks.clear();

        for( const auto &writer : writers )
        {
            m_indiP_sws.add( pcf::IndiElement( writer, pcf::IndiElement::Off ) );
            REQUIRE( registerStreamWriter( writer, true ) == 0 );
        }

        for( const auto &writer : defWriters )
        {
            REQUIRE( registerStreamWriter( writer, false ) == 0 );
        }
    }

    void setWriterSelected( const std::string &writerName, pcf::IndiElement::SwitchStateType state )
    {
        m_indiP_sws[writerName].setSwitchState( state );
    }

    int setWriterWritingState( const std::string &writerName, pcf::IndiElement::SwitchStateType state )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );

        ip.setDevice( streamWriterDeviceName( writerName ) );
        ip.setName( "writing" );
        ip.add( pcf::IndiElement( "toggle" ) );
        ip["toggle"].setSwitchState( state );

        return setCallBack_streamWriterWriting( ip );
    }

    bool beginWriter( const std::string &writerName )
    {
        return beginObservationStreamWriter( writerName );
    }

    bool endWriter( const std::string &writerName )
    {
        return endObservationStreamWriter( writerName );
    }

    bool writerWritingKnown( const std::string &writerName ) const
    {
        return m_streamWriterWritingKnown.at( writerName );
    }

    bool writerWriting( const std::string &writerName ) const
    {
        return m_streamWriterWriting.at( writerName );
    }

    bool writerExposed( const std::string &writerName ) const
    {
        return m_indiP_sws.find( writerName );
    }
};
/// \endcond

/// Verify the observerCtrl INDI callback validators accept only the expected properties.
/**
 * \ingroup observerCtrl_unit_test
 */
TEST_CASE( "observerCtrl INDI callbacks validate device and property names", "[observerCtrl]" )
{
    // clang-format off
    #ifdef OBSERVERCTRL_TEST_DOXYGEN_REF
    observerCtrl::newCallBack_m_indiP_observers( pcf::IndiProperty() );
    observerCtrl::newCallBack_m_indiP_obsName( pcf::IndiProperty() );
    observerCtrl::newCallBack_m_indiP_observing( pcf::IndiProperty() );
    observerCtrl::newCallBack_m_indiP_sws( pcf::IndiProperty() );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, observers );
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, obsName );
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, observing );
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, sws );
}

/// Verify observerCtrl tracks remote stream writer writing state updates.
/**
 * \ingroup observerCtrl_unit_test
 */
TEST_CASE( "observerCtrl tracks remote stream writer writing state", "[observerCtrl]" )
{
    // clang-format off
    #ifdef OBSERVERCTRL_TEST_DOXYGEN_REF
    observerCtrl::setCallBack_streamWriterWriting( pcf::IndiProperty() );
    #endif
    // clang-format on

    observerCtrl_test app( "observerCtrl_test" );
    app.configureStreamWriters( { "camwfs" } );

    REQUIRE_FALSE( app.writerWritingKnown( "camwfs" ) );

    REQUIRE( app.setWriterWritingState( "camwfs", pcf::IndiElement::On ) == 0 );
    REQUIRE( app.writerWritingKnown( "camwfs" ) );
    REQUIRE( app.writerWriting( "camwfs" ) );

    REQUIRE( app.setWriterWritingState( "camwfs", pcf::IndiElement::Off ) == 0 );
    REQUIRE_FALSE( app.writerWriting( "camwfs" ) );
}

/// Verify observerCtrl only stops stream writers it started for the current observation.
/**
 * \ingroup observerCtrl_unit_test
 */
TEST_CASE( "observerCtrl only stops stream writers it started", "[observerCtrl]" )
{
    // clang-format off
    #ifdef OBSERVERCTRL_TEST_DOXYGEN_REF
    observerCtrl::beginObservationStreamWriter( std::string() );
    observerCtrl::endObservationStreamWriter( std::string() );
    #endif
    // clang-format on

    observerCtrl_test app( "observerCtrl_test" );
    app.configureStreamWriters( { "camsci1", "camwfs" } );

    app.setWriterSelected( "camsci1", pcf::IndiElement::On );
    app.setWriterSelected( "camwfs", pcf::IndiElement::On );

    REQUIRE( app.setWriterWritingState( "camsci1", pcf::IndiElement::On ) == 0 );
    REQUIRE( app.setWriterWritingState( "camwfs", pcf::IndiElement::Off ) == 0 );

    REQUIRE_FALSE( app.beginWriter( "camsci1" ) );
    REQUIRE( app.beginWriter( "camwfs" ) );

    REQUIRE_FALSE( app.endWriter( "camsci1" ) );
    REQUIRE( app.endWriter( "camwfs" ) );
    REQUIRE_FALSE( app.endWriter( "camwfs" ) );
}

/// Verify observerCtrl always manages configured default writers without exposing them in INDI.
/**
 * \ingroup observerCtrl_unit_test
 */
TEST_CASE( "observerCtrl default stream writers are managed but not selectable", "[observerCtrl]" )
{
    observerCtrl_test app( "observerCtrl_test" );
    app.configureStreamWriters( { "camsci1" }, { "camlowfs" } );

    REQUIRE( app.writerExposed( "camsci1" ) );
    REQUIRE_FALSE( app.writerExposed( "camlowfs" ) );

    REQUIRE( app.setWriterWritingState( "camlowfs", pcf::IndiElement::Off ) == 0 );
    REQUIRE( app.beginWriter( "camlowfs" ) );
    REQUIRE( app.endWriter( "camlowfs" ) );
}

/// Verify observerCtrl does not claim ownership when a writer state has not been received yet.
/**
 * \ingroup observerCtrl_unit_test
 */
TEST_CASE( "observerCtrl does not stop writers with unknown initial state", "[observerCtrl]" )
{
    observerCtrl_test app( "observerCtrl_test" );
    app.configureStreamWriters( {}, { "camlowfs" } );

    REQUIRE( app.beginWriter( "camlowfs" ) );
    REQUIRE_FALSE( app.endWriter( "camlowfs" ) );
}

} // namespace observerCtrlTest

} // namespace libXWCTest
