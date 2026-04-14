/** \file psfAcq_test.cpp
 * \brief Catch2 tests for the psfAcq app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup psfAcq_files
 */

#include "../../../tests/testXWC.hpp"

#include "../psfAcq.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup psfAcq_unit_test psfAcq Unit Tests
 * \brief Unit tests for the psfAcq application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `psfAcq` unit tests.
/** \ingroup psfAcq_unit_test
 */
namespace psfAcqTest
{

/// Verify the placeholder psfAcq test harness instantiates the app cleanly.
/**
 * \ingroup psfAcq_unit_test
 */
TEST_CASE( "psfAcq placeholder harness instantiates the app", "[psfAcq]" )
{
    // clang-format off
    #ifdef PSFACQ_TEST_DOXYGEN_REF
    psfAcq();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        psfAcq app;

        REQUIRE( true );
    }
}

} // namespace psfAcqTest

} // namespace libXWCTest
