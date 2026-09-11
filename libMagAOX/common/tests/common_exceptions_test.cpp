/** \file common_exceptions_test.cpp
 * \brief Catch2 tests for the xwcException class in libMagAOX/common/exceptions.hpp.
 *
 * The tests construct exceptions directly and read back the message, source location, and
 * code through the accessors. No setup is needed.
 *
 * History:
 */
#include "../../../tests/catch2/catch.hpp"

#include "../exceptions.hpp"

#include <string>

namespace libXWCTest
{
namespace commonTest
{

/** \defgroup common_unit_test libXWC::common Unit Tests
 * \ingroup unit_test
 */

/// Namespace for XWC::xwcException tests
/** \ingroup common_unit_test
 *
 */
namespace xwcExceptionTest
{

/// Construct xwcException with and without a code and check its accessors.
/** The what() text must contain the message. The file name must be this file, which shows
 * the source location is captured at the construction site. The line must be positive.
 * \ingroup common_unit_test
 */
TEST_CASE( "xwcException contains the message, source location, and code", "[libMagAOX::xwcException]" )
{
    SECTION( "An xwcException constructed from a message" )
    {
        MagAOX::xwcException e( "test error" );

        REQUIRE( e.message() == "test error" );
        REQUIRE( std::string( e.what() ).find( "test error" ) != std::string::npos );
        REQUIRE( std::string( e.file_name() ).find( "common_exceptions_test" ) != std::string::npos );
        REQUIRE( e.line() > 0 );
        REQUIRE( e.code() == 0 );
    }

    SECTION( "An xwcException constructed from a message and a code" )
    {
        MagAOX::xwcException e( "coded error", 42 );

        REQUIRE( e.message() == "coded error" );
        REQUIRE( e.code() == 42 );
        REQUIRE( std::string( e.what() ).find( "coded error" ) != std::string::npos );
        REQUIRE( std::string( e.what() ).find( "code: 42" ) != std::string::npos );
        REQUIRE( e.line() > 0 );
    }
}

} // namespace xwcExceptionTest
} // namespace commonTest
} // namespace libXWCTest
