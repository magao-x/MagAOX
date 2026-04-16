/** \file shmimInfo_test.cpp
 * \brief Catch2 tests for the shmimInfo utility.
 *
 * \author Codex
 */

#include "../../tests/catch2/catch.hpp"

#include <type_traits>

#include "../shmimInfo/shmimInfo.hpp"

namespace shmimInfo_test
{

SCENARIO( "shmimInfo declares the expected application type", "[shmimInfo]" )
{
    REQUIRE( std::is_base_of_v<mx::app::application, shmimInfo> );
}

} // namespace shmimInfo_test
