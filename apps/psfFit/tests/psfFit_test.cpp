/** \file psfFit_test.cpp
 * \brief Catch2 tests for the psfFit app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup psfFit_files
 */

#include "../../../tests/testXWC.hpp"

#include "../psfFit.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup psfFit_unit_test psfFit Unit Tests
 * \brief Unit tests for the psfFit application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `psfFit` unit tests.
/** \ingroup psfFit_unit_test
 */
namespace psfFitTest
{

/// Verify the placeholder psfFit test harness instantiates the app cleanly.
/**
 * \ingroup psfFit_unit_test
 */
TEST_CASE( "psfFit placeholder harness instantiates the app", "[psfFit]" )
{
    // clang-format off
    #ifdef PSFFIT_TEST_DOXYGEN_REF
    psfFit();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        psfFit app;

        REQUIRE( true );
    }
}

} // namespace psfFitTest

} // namespace libXWCTest
