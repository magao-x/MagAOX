/** \file template_test.cpp
  * \brief Catch2 tests for the template app.
  *
  * History:
  */
#include "../../../tests/catch2/catch.hpp"

#include "../netSerial.hpp"
#include "../telnetConn.hpp"
#include "../ttyErrors.hpp"
#include "../ttyUSB.hpp"
#include "../usbDevice.hpp"

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
