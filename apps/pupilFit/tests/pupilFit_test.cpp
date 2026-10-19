/** \file pupilFit_test.cpp
 * \brief Offline pupil fitting, telemetry, and averaging regression tests.
 * \ingroup pupilFit_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../pupilFit.hpp"
#include "../../../libMagAOX/logger/generated/logMemberAccessor.hpp"

#include <filesystem>
#include <fstream>

namespace libXWCTest
{
namespace pupilFitTest
{
using namespace MagAOX::app;
using namespace MagAOX::logger;

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Offline harness with a real telemetry queue and no application threads or instrument connections.
class pupilFitHarness : public pupilFit
{
  public:
    /// Construct a synthetic-image fitter with private temporary logs.
    pupilFitHarness( int pupils /**< [in] active pupil count */ );

    /// Flush logs and remove only this harness's temporary directory.
    ~pupilFitHarness();

    /// Fit a copy of one input using the production fit-and-record path.
    int fit( mx::improc::eigenImage<float> image /**< [in] synthetic input */,
             uint64_t                      frame /**< [in] source counter */,
             bool                          reference = false /**< [in] identify calibration fits */ );

    /// Drain telemetry synchronously and read every serialized record.
    std::vector<flatlogs::bufferPtrT> records();

    /// Reset or stop temporal averaging through the production callback.
    int averaging( bool enabled /**< [in] requested toggle state */ );

    using pupilFit::checkRecordTimes;
    using pupilFit::m_avg_dx;
    using pupilFit::m_avg_dy;
    using pupilFit::m_fitter;
    using pupilFit::m_indiP_avg;
    using pupilFit::m_indiP_refmode;
    using pupilFit::m_indiP_thresh;
    using pupilFit::m_indiP_update;
    using pupilFit::m_navg;
    using pupilFit::m_threshold;
    using pupilFit::m_updated;
    using pupilFit::newCallBack_m_indiP_refmode;
    using pupilFit::newCallBack_m_indiP_thresh;
    using pupilFit::newCallBack_m_indiP_update;
    using pupilFit::recordFitState;
    using pupilFit::updatePupilAverages;

    /// Exercise failed-frame publication guards with local image metadata.
    int processFailure( mx::improc::eigenImage<float> &image /**< [in] invalid image */ );
    using pupilFit::m_setD1;
    using pupilFit::m_setD2;
    using pupilFit::m_setD3;
    using pupilFit::m_setD4;
    using pupilFit::m_setx1;
    using pupilFit::m_setx2;
    using pupilFit::m_setx3;
    using pupilFit::m_setx4;
    using pupilFit::m_sety1;
    using pupilFit::m_sety2;
    using pupilFit::m_sety3;
    using pupilFit::m_sety4;
    using telemeterT::m_maxInterval;

  private:
    /// Unique scratch directory owned by this harness.
    std::string m_root;
};

pupilFitHarness::pupilFitHarness( int pupils )
{
    char root[] = "/tmp/pupilFit_test_XXXXXX";
    REQUIRE( mkdtemp( root ) != nullptr );
    m_root       = root;
    m_configName = "pupilFit_test";
    m_tel.logPath( m_root );
    m_tel.logName( m_configName );
    m_tel.logExt( "bintel" );
    m_tel.logLevel( logPrio::LOG_TELEM );
    m_log.logPath( m_root );
    m_log.logName( "pupilFit_test_logs" );
    m_numPupils             = pupils;
    shmimMonitorT::m_width  = 24;
    shmimMonitorT::m_height = 24;
    m_fitter.m_numPupils    = pupils;
    REQUIRE( m_fitter.setSize( 12, 12 ) == 0 );
    createROIndiNumber( m_indiP_avg, "average", "Average" );
    for( auto name : { "x", "y", "D", "dx", "dy", "dD" } )
        m_indiP_avg.add( pcf::IndiElement( name ) );
    createStandardIndiToggleSw( m_indiP_averaging, "averaging", "Averaging" );
    createStandardIndiNumber<float>( m_indiP_thresh, "threshold", 0, 1, 0, "%0.2f", "Threshold" );
    createStandardIndiSelectionSw( m_indiP_refmode, "setpt_mode", { "default", "refim", "user" }, "Reference Mode" );
    createStandardIndiRequestSw( m_indiP_update, "setpt_current", "Set Reference" );
    int index = 0;
    for( auto *prop : { &m_indiP_quad1, &m_indiP_quad2, &m_indiP_quad3, &m_indiP_quad4 } )
    {
        createROIndiNumber( *prop, "quadrant" + std::to_string( ++index ), "Quadrant" );
        for( auto name : { "x", "y", "D" } )
            prop->add( pcf::IndiElement( name, 10.0f * index ) );
    }
}

pupilFitHarness::~pupilFitHarness()
{
    m_tel.logShutdown( true );
    m_tel.logThreadExec();
    m_log.logShutdown( true );
    m_log.logThreadExec();
    std::filesystem::remove_all( m_root );
}

int pupilFitHarness::fit( mx::improc::eigenImage<float> image, uint64_t frame, bool reference )
{
    std::lock_guard<std::mutex>   guard( m_indiMutex );
    mx::improc::eigenImage<float> edge;
    return fitImage( image, edge, frame, { 123, 456 }, reference );
}

int pupilFitHarness::processFailure( mx::improc::eigenImage<float> &image )
{
    IMAGE_METADATA metadata{};
    metadata.cnt0                   = 99;
    metadata.atime                  = { 123, 456 };
    shmimMonitorT::m_imageStream.md = &metadata;
    m_fitIm.resize( 24, 24 );
    int result                      = processImage( image.data(), MagAOX::app::dev::shmimT() );
    shmimMonitorT::m_imageStream.md = nullptr;
    return result;
}

int pupilFitHarness::averaging( bool enabled )
{
    auto request = m_indiP_averaging;
    request["toggle"].setSwitchState( enabled ? pcf::IndiElement::On : pcf::IndiElement::Off );
    return newCallBack_m_indiP_averaging( request );
}

std::vector<flatlogs::bufferPtrT> pupilFitHarness::records()
{
    m_tel.logShutdown( true );
    m_tel.logThreadExec();
    std::vector<std::filesystem::path> files;
    for( const auto &entry : std::filesystem::recursive_directory_iterator( m_root ) )
        if( entry.path().extension() == ".bintel" )
            files.push_back( entry.path() );
    std::sort( files.begin(), files.end() );
    std::vector<flatlogs::bufferPtrT> result;
    for( const auto &path : files )
    {
        std::ifstream     input( path, std::ios::binary );
        std::vector<char> bytes( ( std::istreambuf_iterator<char>( input ) ), {} );
        size_t            offset = 0;
        while( offset < bytes.size() )
        {
            REQUIRE( bytes.size() - offset >= flatlogs::logHeader::minHeadSize );
            auto size = flatlogs::logHeader::totalSize( bytes.data() + offset );
            REQUIRE( size <= bytes.size() - offset );
            flatlogs::bufferPtrT buffer( static_cast<char *>( ::operator new( size ) ) );
            std::memcpy( buffer.get(), bytes.data() + offset, size );
            REQUIRE(
                logVerify( flatlogs::logHeader::eventCode( buffer ), buffer, flatlogs::logHeader::msgLen( buffer ) ) );
            result.push_back( buffer );
            offset += size;
        }
    }
    return result;
}

/// Generate resolved circular pupils with different known intensity levels.
mx::improc::eigenImage<float> synthetic( int pupils /**< [in] active pupil count */ )
{
    mx::improc::eigenImage<float> image( 24, 24 );
    image.setConstant( 10 );
    for( int p = 0; p < pupils; ++p )
    {
        float x = 5.5 + ( p == 1 || ( pupils == 4 && p == 3 ) ? 12 : 0 );
        float y = 5.5 + ( p >= 2 ? 12 : 0 );
        if( pupils == 3 && p == 2 )
            x += 6;
        for( int i = 0; i < 24; ++i )
            for( int j = 0; j < 24; ++j )
                if( ( i - x ) * ( i - x ) + ( j - y ) * ( j - y ) < 25 )
                    image( i, j ) = 100 + 10 * p;
    }
    return image;
}
/// \endcond

/// Every fit produces a record, preserving all active pupils, metadata, and reference identity.
/** \ingroup pupilFit_unit_test */
TEST_CASE( "pupilFit records repeated fits and reference fits", "[pupilFit]" )
{
    // clang-format off
    #ifdef PUPILFIT_TEST_DOXYGEN_REF
    pupilFit::fitImage();
    pupilFit::recordFitState();
    pupilFitter<float>::fit();
    #endif
    // clang-format on
    for( int count : { 3, 4 } )
    {
        pupilFitHarness app( count );
        auto            image = synthetic( count );
        REQUIRE( app.fit( image, 41 ) == 0 );
        REQUIRE( app.fit( image, 42 ) == 0 );
        REQUIRE( app.fit( image, 7, true ) == 0 );
        auto records = app.records();
        REQUIRE( records.size() == 4 );
        REQUIRE( flatlogs::logHeader::eventCode( records[0] ) == telem_pupilfit_state::eventCode );
        for( size_t i = 1; i < records.size(); ++i )
        {
            REQUIRE( flatlogs::logHeader::eventCode( records[i] ) == telem_pupilfit::eventCode );
            auto *payload = flatlogs::logHeader::messageBuffer( records[i] );
            auto *fb      = GetTelem_pupilfit_fb( payload );
            CHECK( fb->num_pupils() == count );
            REQUIRE( fb->pupils()->size() == static_cast<unsigned>( count ) );
            CHECK( fb->valid() );
            CHECK( fb->frame() == ( i == 3 ? 7 : 40 + i ) );
            CHECK( fb->acq_sec() == 123 );
            CHECK( fb->acq_nsec() == 456 );
            CHECK( fb->reference() == ( i == 3 ) );
            for( int p = 0; p < count; ++p )
            {
                CHECK( fb->pupils()->Get( p )->med() == Approx( 100 + 10 * p ) );
                CHECK( fb->pupils()->Get( p )->bg() == Approx( app.m_fitter.m_bg[p] ) );
                CHECK( fb->pupils()->Get( p )->bg() == Approx( 10 ).margin( 1.2 ) );
                CHECK( fb->pupils()->Get( p )->D() == Approx( 10 ).margin( 1 ) );
            }
            std::ostringstream text, json;
            logStdFormat( text, records[i] );
            logJsonFormat( json, records[i] );
            CHECK( text.str().find( "[average]" ) != std::string::npos );
            CHECK( json.str().find( "\"average\": {" ) != std::string::npos );
        }
        auto *first  = flatlogs::logHeader::messageBuffer( records[1] );
        auto *second = flatlogs::logHeader::messageBuffer( records[2] );
        CHECK( telem_pupilfit::values<&PupilFitResult::x>( first ) ==
               telem_pupilfit::values<&PupilFitResult::x>( second ) );
    }
}

/// Failed input, unusable contrast, non-finite pixels, and invalid settings never serialize stale results.
/** \ingroup pupilFit_unit_test */
TEST_CASE( "pupilFit marks failed attempts without stale pupils", "[pupilFit]" )
{
    // clang-format off
    #ifdef PUPILFIT_TEST_DOXYGEN_REF
    pupilFit::fitImage();
    pupilFitter<float>::fit();
    #endif
    // clang-format on
    pupilFitHarness app( 4 );
    auto            image = synthetic( 4 );
    REQUIRE( app.fit( image, 1 ) == 0 );
    SECTION( "flat image" )
    {
        image.setConstant( 10 );
    }
    SECTION( "NaN pixel" )
    {
        image( 1, 1 ) = std::numeric_limits<float>::quiet_NaN();
    }
    SECTION( "infinite pixel" )
    {
        image( 1, 1 ) = std::numeric_limits<float>::infinity();
    }
    SECTION( "wrong dimensions" )
    {
        image.resize( 12, 12 );
        image.setZero();
    }
    SECTION( "bad quantile" )
    {
        app.m_fitter.m_pupMedIndex = 1;
    }
    SECTION( "non-finite threshold" )
    {
        app.m_threshold = std::numeric_limits<float>::quiet_NaN();
    }
    REQUIRE( app.fit( image, 2 ) == -1 );
    auto  records = app.records();
    auto *payload = flatlogs::logHeader::messageBuffer( records.back() );
    auto *fb      = GetTelem_pupilfit_fb( payload );
    CHECK_FALSE( fb->valid() );
    CHECK( fb->frame() == 2 );
    CHECK( fb->pupils()->size() == 0 );
    CHECK( mx::math::isNan( telem_pupilfit::average<&PupilFitResult::x>( payload ) ) );
    std::ostringstream json;
    logJsonFormat( json, records.back() );
    CHECK( json.str().find( "\"average\": null" ) != std::string::npos );
}

/// Failed live frames return before INDI, output-stream publication, or temporal accumulation.
/** \ingroup pupilFit_unit_test */
TEST_CASE( "pupilFit rejects failed live frames before publication", "[pupilFit]" )
{
    // clang-format off
    #ifdef PUPILFIT_TEST_DOXYGEN_REF
    pupilFit::processImage();
    #endif
    // clang-format on
    pupilFitHarness app( 3 );
    REQUIRE( app.averaging( true ) == 0 );
    auto image = synthetic( 3 );
    image.setZero();
    REQUIRE( app.processFailure( image ) == -1 );
    CHECK( app.m_navg == 0 );
    CHECK_FALSE( app.m_updated );
    auto  records = app.records();
    auto *fb      = GetTelem_pupilfit_fb( flatlogs::logHeader::messageBuffer( records.back() ) );
    CHECK_FALSE( fb->valid() );
    CHECK( fb->frame() == 99 );
}

/// Verify wire contracts, binary size, derived JSON, and rejection of malformed records.
/** \ingroup pupilFit_unit_test */
TEST_CASE( "pupilFit telemetry verifies wire contracts and derived means", "[pupilFit]" )
{
    std::vector<PupilFitResult> pupils = { { 3, 6, 4, 100, 10 }, { 6, 12, 8, 110, 10 }, { 9, 18, 12, 120, 10 } };
    telem_pupilfit::messageT    message( 42, { 123, 456 }, false, 3, true, pupils );
    flatlogs::bufferPtrT        buffer;
    REQUIRE( flatlogs::logHeader::createLog<telem_pupilfit>( buffer, { 1, 0 }, message, logPrio::LOG_TELEM ) == 0 );
    REQUIRE( logVerify( telem_pupilfit::eventCode, buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    CHECK( sizeof( PupilFitResult ) == 5 * sizeof( float ) );
    auto *payload = flatlogs::logHeader::messageBuffer( buffer );
    CHECK( telem_pupilfit::average<&PupilFitResult::x>( payload ) == Approx( 6 ) );
    CHECK( telem_pupilfit::average<&PupilFitResult::D>( payload ) == Approx( 8 ) );
    auto json =
        telem_pupilfit::msgJSON( payload, message.builder.GetSize(), telem_pupilfit_bfbs, telem_pupilfit_bfbs_len );
    flatbuffers::Parser parser;
    REQUIRE( parser.Deserialize( telem_pupilfit_bfbs, telem_pupilfit_bfbs_len ) );
    parser.opts.skip_unexpected_fields_in_json = true;
    REQUIRE( parser.ParseJson( json.c_str() ) );
    CHECK( json.find( "\"average\": {\"x\": 6" ) != std::string::npos );
    telem_pupilfit::messageT missing( 42, { 123, 456 }, false, 4, true, pupils );
    REQUIRE( flatlogs::logHeader::createLog<telem_pupilfit>( buffer, { 1, 0 }, missing, logPrio::LOG_TELEM ) == 0 );
    CHECK_FALSE( logVerify( telem_pupilfit::eventCode, buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    pupils[1] = PupilFitResult( std::numeric_limits<float>::infinity(), 12, 8, 110, 10 );
    telem_pupilfit::messageT nonfinite( 42, { 123, 456 }, false, 3, true, pupils );
    REQUIRE( flatlogs::logHeader::createLog<telem_pupilfit>( buffer, { 1, 0 }, nonfinite, logPrio::LOG_TELEM ) == 0 );
    CHECK_FALSE( logVerify( telem_pupilfit::eventCode, buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    std::memset( flatlogs::logHeader::messageBuffer( buffer ), 0xff, 4 );
    CHECK_FALSE( logVerify( telem_pupilfit::eventCode, buffer, flatlogs::logHeader::msgLen( buffer ) ) );
    CHECK( telem_pupilfit::getAccessor( "absent" ).accessor == nullptr );
    CHECK( telem_pupilfit_state::getAccessor( "absent" ).accessor == nullptr );
}

/// Only state changes or periodic requests repeat state, and averaging resets create new windows.
/** \ingroup pupilFit_unit_test */
TEST_CASE( "pupilFit records state changes and averaging boundaries", "[pupilFit]" )
{
    // clang-format off
    #ifdef PUPILFIT_TEST_DOXYGEN_REF
    pupilFit::recordFitState();
    pupilFit::checkRecordTimes();
    pupilFit::recordTelem();
    pupilFit::newCallBack_m_indiP_averaging();
    #endif
    // clang-format on
    pupilFitHarness app( 3 );
    REQUIRE( app.recordFitState() == 0 );
    REQUIRE( app.recordFitState() == 0 );
    app.m_setx1     = 123;
    app.m_threshold = 0.4;
    REQUIRE( app.recordFitState() == 0 );
    REQUIRE( app.averaging( true ) == 0 );
    REQUIRE( app.averaging( true ) == 0 );
    REQUIRE( app.averaging( false ) == 0 );
    app.m_maxInterval = -1;
    REQUIRE( app.checkRecordTimes() == 0 );
    auto                                         records = app.records();
    std::vector<const Telem_pupilfit_state_fb *> states;
    for( auto &record : records )
    {
        auto code = flatlogs::logHeader::eventCode( record );
        CHECK( code != telem_pupilfit::eventCode );
        if( code == telem_pupilfit_state::eventCode )
            states.push_back( GetTelem_pupilfit_state_fb( flatlogs::logHeader::messageBuffer( record ) ) );
    }
    REQUIRE( states.size() == 6 );
    CHECK( states[0]->setpoints()->size() == 3 );
    CHECK( states[1]->setpoints()->Get( 0 )->x() == 123 );
    CHECK( states[1]->threshold() == Approx( 0.4 ) );
    CHECK( states[2]->averaging() );
    CHECK( states[2]->window() == 1 );
    CHECK( states[3]->averaging() );
    CHECK( states[3]->window() == 2 );
    CHECK_FALSE( states[4]->averaging() );
    CHECK( states[4]->window() == 2 );
    CHECK( states[5]->window() == 2 );
}

/// Configuration callbacks snapshot changed settings and identify live-fit setpoint capture.
/** \ingroup pupilFit_unit_test */
TEST_CASE( "pupilFit callbacks record reference and threshold changes", "[pupilFit]" )
{
    // clang-format off
    #ifdef PUPILFIT_TEST_DOXYGEN_REF
    pupilFit::newCallBack_m_indiP_thresh();
    pupilFit::newCallBack_m_indiP_refmode();
    pupilFit::newCallBack_m_indiP_update();
    #endif
    // clang-format on
    pupilFitHarness app( 3 );
    auto            threshold = app.m_indiP_thresh;
    threshold["target"].set( 0.6f );
    REQUIRE( app.newCallBack_m_indiP_thresh( threshold ) == 0 );
    threshold["target"].set( -0.1f );
    REQUIRE( app.newCallBack_m_indiP_thresh( threshold ) == -1 );
    CHECK( app.m_threshold == Approx( 0.6f ) );
    auto mode = app.m_indiP_refmode;
    mode["refim"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_refmode( mode ) == 0 );
    auto current = app.m_indiP_update;
    current["request"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_update( current ) == 0 );
    auto records = app.records();
    REQUIRE( records.size() == 3 );
    auto *fb = GetTelem_pupilfit_state_fb( flatlogs::logHeader::messageBuffer( records.back() ) );
    CHECK( fb->requested_mode() == USEREFIM );
    CHECK( fb->active_mode() == USECURRENTSET );
    CHECK( fb->threshold() == Approx( 0.6f ) );
    REQUIRE( fb->setpoints()->size() == 3 );
    CHECK( fb->setpoints()->Get( 0 )->x() == 10 );
    CHECK( fb->setpoints()->Get( 2 )->D() == 30 );
}

/// Three-pupil means and offsets exclude the inactive fourth entry and agree with derived telemetry.
/** \ingroup pupilFit_unit_test */
TEST_CASE( "pupilFit averages only active pupils", "[pupilFit]" )
{
    // clang-format off
    #ifdef PUPILFIT_TEST_DOXYGEN_REF
    pupilFit::updatePupilAverages();
    #endif
    // clang-format on
    for( int count : { 3, 4 } )
    {
        pupilFitHarness             app( count );
        std::vector<PupilFitResult> pupils;
        for( int i = 0; i < 4; ++i )
        {
            app.m_fitter.m_avgx[i] = 3 * ( i + 1 );
            app.m_fitter.m_avgy[i] = 6 * ( i + 1 );
            app.m_fitter.m_avgr[i] = 2 * ( i + 1 );
            if( i < count )
                pupils.emplace_back( 3 * ( i + 1 ), 6 * ( i + 1 ), 4 * ( i + 1 ), 100, 10 );
        }
        app.m_setx1 = app.m_setx2 = app.m_setx3 = 1;
        app.m_sety1 = app.m_sety2 = app.m_sety3 = 2;
        app.m_setD1 = app.m_setD2 = app.m_setD3 = 3;
        app.m_setx4                             = 101;
        app.m_sety4                             = 102;
        app.m_setD4                             = 103;
        app.updatePupilAverages();
        telem_pupilfit::messageT message( 1, { 1, 0 }, false, count, true, pupils );
        auto                    *payload = message.builder.GetBufferPointer();
        CHECK( app.m_indiP_avg["x"].get<float>() == Approx( telem_pupilfit::average<&PupilFitResult::x>( payload ) ) );
        CHECK( app.m_indiP_avg["y"].get<float>() == Approx( telem_pupilfit::average<&PupilFitResult::y>( payload ) ) );
        CHECK( app.m_indiP_avg["D"].get<float>() == Approx( telem_pupilfit::average<&PupilFitResult::D>( payload ) ) );
        CHECK( app.m_avg_dx == Approx( count == 3 ? 5 : -18.5 ) );
        CHECK( app.m_avg_dy == Approx( count == 3 ? 10 : -12 ) );
        CHECK( app.m_indiP_avg["dD"].get<float>() == Approx( count == 3 ? 5 : -18 ) );
        CHECK( logMemberAccessor( telem_pupilfit::eventCode, "x" ).valType == logMeta::Vector_Float );
        CHECK( logMemberAccessor( telem_pupilfit::eventCode, "avg_x" ).valType == logMeta::Float );
        CHECK( logMemberAccessor( telem_pupilfit_state::eventCode, "set_x" ).metaType == logMeta::State );
    }
}

} // namespace pupilFitTest
} // namespace libXWCTest
