/** \file stateRuleEngine_test.cpp
 * \brief Catch2 tests for the stateRuleEngine app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup stateRuleEngine_files
 */

#include "../../../tests/testXWC.hpp"

#include "../stateRuleEngine.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup stateRuleEngine_unit_test stateRuleEngine Unit Tests
 * \brief Unit tests for the stateRuleEngine application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `stateRuleEngine` unit tests.
/** \ingroup stateRuleEngine_unit_test
 */
namespace stateRuleEngineTest
{

/// Verify the placeholder stateRuleEngine test harness instantiates the app cleanly.
/**
 * \ingroup stateRuleEngine_unit_test
 */
TEST_CASE( "stateRuleEngine placeholder harness instantiates the app", "[stateRuleEngine]" )
{
    // clang-format off
    #ifdef STATERULEENGINE_TEST_DOXYGEN_REF
    stateRuleEngine();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        stateRuleEngine app;

        REQUIRE( true );
    }
}

} // namespace stateRuleEngineTest

} // namespace libXWCTest
