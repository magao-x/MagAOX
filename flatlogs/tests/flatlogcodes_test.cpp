/** \file template_test.cpp
  * \brief Catch2 tests for the template app.
  *
  * History:
  */
#include "../../tests/catch2/catch.hpp"

#include "../include/flatlogs/flatlogs.hpp"
#include "../include/flatlogs/logDefs.hpp"
#include "../include/flatlogs/logHeader.hpp"
#include "../include/flatlogs/logPriority.hpp"
#include "../include/flatlogs/logStdFormat.hpp"
#include "../include/flatlogs/timespecX.hpp"


namespace template_test
{

SCENARIO( "xxxx", "[template]" )
{
   GIVEN("xxxxx")
   {
      WHEN("xxxx")
      {
         int rv = 0;

         REQUIRE(rv == 0);
      }
   }
}
} //namespace template_test
