/** \file logMeta_test.cpp
 * \brief Tests for log metadata helper behavior.
 * \ingroup logger_files
 */

#include "../../../tests/testXWC.hpp"

#include "../logMeta.hpp"
#include "../types/telem_teldata.hpp"
#include "../types/text_log.hpp"

#include "logTestFiles.hpp"

namespace libXWCTest
{

/** \defgroup logger_unit_test libXWC::logger Unit Tests
 * \ingroup unit_test
 */

/// Namespace for XWC::logger tests.
/** \ingroup logger_unit_test
 */
namespace loggerTest
{

/** \defgroup logMeta_unit_test logMeta Unit Tests
 * \ingroup logger_unit_test
 */

/// Namespace for XWC::logger::logMeta tests.
/** \ingroup logMeta_unit_test
 */
namespace logMetaTest
{

/// Validate shortest-path angle deltas across wrap boundaries.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "Log metadata angle deltas wrap correctly", "[libMagAOX::logger::logMeta]" )
{
    using MagAOX::logger::logMetaAngleDelta;

    REQUIRE( logMetaAngleDelta( 10.0, 20.0 ) == Approx( 10.0 ) );
    REQUIRE( logMetaAngleDelta( 20.0, 10.0 ) == Approx( -10.0 ) );
    REQUIRE( logMetaAngleDelta( -179.0, 179.0 ) == Approx( -2.0 ) );
    REQUIRE( logMetaAngleDelta( 179.0, -179.0 ) == Approx( 2.0 ) );
    REQUIRE( logMetaAngleDelta( 359.0, 1.0 ) == Approx( 2.0 ) );
    REQUIRE( logMetaAngleDelta( 1.0, 359.0 ) == Approx( -2.0 ) );
}

/// Validate interpolated angle normalization across wrap boundaries.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "Log metadata angle interpolation normalizes wrapped midpoints", "[libMagAOX::logger::logMeta]" )
{
    using MagAOX::logger::logMetaAngleDelta;
    using MagAOX::logger::logMetaNormalizeInterpolatedAngle;

    auto midpoint = []( double a0, double a1 )
    {
        double interp = a0 + 0.5 * logMetaAngleDelta( a0, a1 );
        return logMetaNormalizeInterpolatedAngle( interp, a0, a1 );
    };

    REQUIRE( midpoint( -179.0, 179.0 ) == Approx( -180.0 ) );
    REQUIRE( midpoint( 179.0, -179.0 ) == Approx( -180.0 ) );
    REQUIRE( midpoint( 359.0, 1.0 ) == Approx( 0.0 ) );
    REQUIRE( midpoint( 1.0, 359.0 ) == Approx( 0.0 ) );
    REQUIRE( midpoint( 10.0, 20.0 ) == Approx( 15.0 ) );
    REQUIRE( midpoint( 20.0, 10.0 ) == Approx( 15.0 ) );
}

/// Validate angle normalization edge cases.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "Log metadata angle normalization handles exact and repeated wraps", "[libMagAOX::logger::logMeta]" )
{
    using MagAOX::logger::logMetaNormalizeAngle180;
    using MagAOX::logger::logMetaNormalizeAngle360;

    REQUIRE( logMetaNormalizeAngle360( 0.0 ) == Approx( 0.0 ) );
    REQUIRE( logMetaNormalizeAngle360( 360.0 ) == Approx( 0.0 ) );
    REQUIRE( logMetaNormalizeAngle360( 720.0 ) == Approx( 0.0 ) );
    REQUIRE( logMetaNormalizeAngle360( -1.0 ) == Approx( 359.0 ) );

    REQUIRE( logMetaNormalizeAngle180( 180.0 ) == Approx( -180.0 ) );
    REQUIRE( logMetaNormalizeAngle180( 540.0 ) == Approx( -180.0 ) );
    REQUIRE( logMetaNormalizeAngle180( -181.0 ) == Approx( 179.0 ) );
}

/// Validate that parallactic angle metadata uses wrapped interpolation.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "telem_teldata parallactic angle accessor is continuous angle metadata", "[libMagAOX::logger::logMeta]" )
{
    MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_teldata::getAccessor( "pa" );

    REQUIRE( lmd.keyword == "PARANG" );
    REQUIRE( lmd.valType == MagAOX::logger::logMeta::valTypes::Double );
    REQUIRE( lmd.metaType == MagAOX::logger::logMeta::metaTypes::Continuous_Angle );
}

/// State metadata read across the change from v1 to v2 files.
/**
 * \ingroup logMeta_unit_test
 */
TEST_CASE( "State metadata across v1 and v2 log files", "[libMagAOX::logger::logMeta]" )
{
    using namespace MagAOX::logger;

    const time_t t0 = 1732000000; // 2024-11-19 07:06:40

    tempDir td( "logMeta_cutover" );

    std::vector<size_t> offs1, offs2;
    std::vector<char>   bytes1 =
        fileBytes( offs1, flatlogs::logVersion1, { { t0, 0, "A" }, { t0 + 1, 0, "A" }, { t0 + 3, 0, "bad" } } );

    // The last v1 entry's message fails its schema check: v1 has no checksum, so it loads, and is caught later.
    size_t   badMsg = offs1[2] + flatlogs::logHeaderV1::headerSize( bytes1.data() + offs1[2] );
    uint32_t root   = 0xFFFFFFF0;
    memcpy( bytes1.data() + badMsg, &root, sizeof( root ) );

    auto sfn1 = writeLogFile( td.m_path, "dev1", "binlog", t0, 0, bytes1 );
    auto sfn2 = writeLogFile( td.m_path,
                              "dev1",
                              "binlog2",
                              t0 + 10,
                              0,
                              fileBytes( offs2, flatlogs::logVersion2, { { t0 + 10, 0, "A" }, { t0 + 20, 0, "B" } } ) );

    logMap<> lm;
    lm.m_appToFileMap["dev1"].insert( sfn1 );
    lm.m_appToFileMap["dev1"].insert( sfn2 );

    cerrCapture cap;

    SECTION( "The value before the exposure is in a v1 file, and the next one is in a v2 file" )
    {
        std::string val;
        std::string reason;
        REQUIRE( getLogStateVal( val,
                                 lm,
                                 "dev1",
                                 eventCodes::TEXT_LOG,
                                 flatlogs::timespecX( t0 + 2, 0 ),
                                 flatlogs::timespecX( t0 + 15, 0 ),
                                 &text_log::message,
                                 -1,
                                 &reason ) == 0 );
        REQUIRE( val == "A" );

        // The unverifiable v1 entry between them was skipped and reported once.
        REQUIRE( cap.count( "Unverifiable log entry skipped while processing FITS metadata: app=dev1 ev=10 source=" +
                            sfn1.fullName() + " sourceByte=" + std::to_string( offs1[2] ) ) == 1 );
        REQUIRE( lm.recoverableErrors() == 1 );
    }

    SECTION( "A value that changes during the exposure is taken from the v2 file" )
    {
        std::string val;
        REQUIRE( getLogStateVal( val,
                                 lm,
                                 "dev1",
                                 eventCodes::TEXT_LOG,
                                 flatlogs::timespecX( t0 + 2, 0 ),
                                 flatlogs::timespecX( t0 + 25, 0 ),
                                 &text_log::message ) == 0 );
        REQUIRE( val == "B" );
    }

    SECTION( "The prior value alone is found by a lookup inside the v2 file" )
    {
        logEntryRecord prior;
        REQUIRE( lm.getPriorEntry( prior, "dev1", eventCodes::TEXT_LOG, flatlogs::timespecX( t0 + 15, 0 ) ) == 0 );
        REQUIRE( verifyLogEntry( eventCodes::TEXT_LOG, prior ) );
        REQUIRE( prior.m_integrity == logIntegrity::verified );
        REQUIRE( text_log::message( prior.message() ) == "A" );

        logEntryRecord none;
        REQUIRE( verifyLogEntry( eventCodes::TEXT_LOG, none ) == false );
    }
}

} // namespace logMetaTest
} // namespace loggerTest
} // namespace libXWCTest
