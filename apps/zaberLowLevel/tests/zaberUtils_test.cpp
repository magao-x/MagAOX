/** \file zaberUtils_test.cpp
  * \brief Catch2 tests for the zaberUtils in the zaberLowLevel app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */
#include "../../../tests/catch2/catch.hpp"

#include "../zaberUtils.hpp"

using namespace MagAOX::app;

namespace zaberUtils_test 
{

SCENARIO( "Parsing the system.serial response", "[zaberUtils]" )
{
   GIVEN("A valid response to system.serial")
   {
      int rv;

      WHEN("Valid response")
      {
         std::string tstr = "@01 0 OK IDLE WR 49822@02 0 OK IDLE WR 49820@03 0 OK IDLE WR 49821\n";
 
         std::vector<int> address;
         std::vector<std::string> serial;
         
         rv = parseSystemSerial(address, serial, tstr);
         
         REQUIRE(rv == 0);
         REQUIRE( address[0] == 1 );
         REQUIRE( address[1] == 2 );
         REQUIRE( address[2] == 3 );
         
         REQUIRE( serial[0] == "49822" );
         REQUIRE( serial[1] == "49820" );
         REQUIRE( serial[2] == "49821" );
      }
   }   
}
   
} //namespace zaberUtils_test 
