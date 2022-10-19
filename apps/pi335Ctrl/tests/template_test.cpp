/** \file template_test.cpp
  * \brief Catch2 tests for the template app.
  *
  * History:
  */
#include "../../../tests/catch2/catch.hpp"

#include "../template.hpp"

using namespace MagAOX::app;

namespace template_test 
{

SCENARIO( "xxxx", "[template]" )
{
   GIVEN("xxxxx")
   {
      int rv;

      WHEN("xxxx")
      {
         rv = [some test];

         REQUIRE(rv == 0);
      }
   }
}
} //namespace template_test 
