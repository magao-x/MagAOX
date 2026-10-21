/** \file shmimDelta_test.cpp
 * \brief Catch2 tests for the shmimDelta utility.
 *
 * \author Codex
 */

#include "../../tests/catch2/catch.hpp"

#include <type_traits>

#include "../shmimDelta/shmimDelta.hpp"

namespace shmimDelta_test
{

/// Verify shmimDelta declares the expected application type.
/**
 * \ingroup shmimDelta
 */
SCENARIO( "shmimDelta declares the expected application type", "[shmimDelta]" )
{
    REQUIRE( std::is_base_of_v<mx::app::application, shmimDelta> );
}

} // namespace shmimDelta_test
