/** \file template_test.cpp
  * \brief Catch2 tests for the template app.
  *
  * History:
  */
#include "../../../tests/catch2/catch.hpp"

#include "../config.hpp"
#include "../defaults.hpp"
#include "../environment.hpp"
#include "../paths.hpp"


namespace template_test
{

SCENARIO( "xxxx", "[template]" )
{
   GIVEN("xxxxx")
   {
      int rv;

      WHEN("xxxx")
      {
         rv = 0;

         REQUIRE(rv == 0);
      }
   }
}
} //namespace template_test
