/** \file shmimIntegrator_test.cpp
 * \brief Catch2 tests for the shmimIntegrator app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup shmimIntegrator_files
 */

#include "../../../tests/testXWC.hpp"

#include "../shmimIntegrator.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup shmimIntegrator_unit_test shmimIntegrator Unit Tests
 * \brief Unit tests for the shmimIntegrator application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `shmimIntegrator` unit tests.
/** \ingroup shmimIntegrator_unit_test
 */
namespace shmimIntegratorTest
{

/// Verify the placeholder shmimIntegrator test harness instantiates the app cleanly.
/**
 * \ingroup shmimIntegrator_unit_test
 */
TEST_CASE( "shmimIntegrator placeholder harness instantiates the app", "[shmimIntegrator]" )
{
    // clang-format off
    #ifdef SHMIMINTEGRATOR_TEST_DOXYGEN_REF
    shmimIntegrator();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        shmimIntegrator app;

        REQUIRE( true );
    }
}

} // namespace shmimIntegratorTest

} // namespace libXWCTest
