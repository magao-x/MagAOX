/** \file logMeta_test.cpp
 * \brief Tests for log metadata helper behavior.
 * \ingroup logger_files
 */

#include "../../../tests/testXWC.hpp"

#include "../logMeta.hpp"
#include "../types/telem_teldata.hpp"

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

} // namespace logMetaTest
} // namespace loggerTest
} // namespace libXWCTest
