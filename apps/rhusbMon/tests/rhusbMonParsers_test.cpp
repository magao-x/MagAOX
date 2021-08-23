/** \file rhusbMonParsers_test.cpp
  * \brief Catch2 tests for the parsers in the rhusbMon app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */
#include "../../../tests/catch2/catch.hpp"

#include "../rhusbMonParsers.hpp"

using namespace MagAOX::app;

namespace rhusbMonParsers_test 
{

SCENARIO( "Parsing the temp response", "[rhusbMonParsers]" )
{
   GIVEN("A valid response to C from the RH USB probe")
   {
      int rv;

      WHEN("Valid temp response, 20.0")
      {
         std::string tstr = "20.0 C\r\n>";
         float temp;

         rv = RH::parseC(temp, tstr);

         REQUIRE(rv == 0);
         REQUIRE(temp == (float) 20.0);
      }

      WHEN("Valid temp response -1.2")
      {
         std::string tstr = "-1.2 C\r\n>";
         float temp;

         rv = RH::parseC(temp, tstr);

         REQUIRE(rv == 0);
         REQUIRE(temp == (float) -1.2);
      }

      WHEN("Valid temp response 1.2")
      {
         std::string tstr = "1.2 C\r\n>";
         float temp;

         rv = RH::parseC(temp, tstr);

         REQUIRE(rv == 0);
         REQUIRE(temp == (float) 1.2);
      }

      WHEN("Valid temp response 01.2")
      {
         std::string tstr = "01.2 C\r\n>";
         float temp;

         rv = RH::parseC(temp, tstr);

         REQUIRE(rv == 0);
         REQUIRE(temp == (float) 1.2);
      }
   }

   GIVEN("Invalid responses to C from the RH USB probe")
   {
      int rv;

      WHEN("Invalid temp response 20.0 C\r>")
      {
         std::string tstr = "20.0 C\r>";
         float temp;

         rv = RH::parseC(temp, tstr);

         REQUIRE(rv == -1);
      }

      WHEN("Invalid temp response 20.0C\r>")
      {
         std::string tstr = "20.0 C\r>";
         float temp;

         rv = RH::parseC(temp, tstr);

         REQUIRE(rv == -1);
      }

      WHEN("Invalid temp response 20.0 \r\n>")
      {
         std::string tstr = "20.0 \r\n>";
         float temp;

         rv = RH::parseC(temp, tstr);

         REQUIRE(rv == -1);
      }

      WHEN("Invalid temp response 20.0 >")
      {
         std::string tstr = "20.0 >";
         float temp;

         rv = RH::parseC(temp, tstr);

         REQUIRE(rv == -1);
      }

      WHEN("Invalid temp response ' C\r\n>'")
      {
         std::string tstr = " C\r\n>";
         float temp;

         rv = RH::parseC(temp, tstr);

         REQUIRE(rv == -2);
      }

      WHEN("Invalid temp response 'A C\r\n>'")
      {
         std::string tstr = "A C\r\n>";
         float temp;

         rv = RH::parseC(temp, tstr);

         REQUIRE(rv == -3);
      }
   }
}

SCENARIO( "Parsing the humidity response", "[rhusbMonParsers]" )
{
   GIVEN("A valid response to H from the RH USB probe")
   {
      int rv;

      WHEN("Valid temp response")
      {
         std::string tstr = "20.0 %RH\r\n>";
         float temp;

         rv = RH::parseH(temp, tstr);

         REQUIRE(rv == 0);
         REQUIRE(temp == (float) 20.0);
      }

      WHEN("Valid temp response")
      {
         std::string tstr = "1.2 %RH\r\n>";
         float temp;

         rv = RH::parseH(temp, tstr);

         REQUIRE(rv == 0);
         REQUIRE(temp == (float) 1.2);
      }

      WHEN("Valid temp response")
      {
         std::string tstr = "09.9 %RH\r\n>";
         float temp;

         rv = RH::parseH(temp, tstr);

         REQUIRE(rv == 0);
         REQUIRE(temp == (float) 9.9);
      }

      WHEN("Valid temp response")
      {
         std::string tstr = "0.2 %RH\r\n>";
         float temp;

         rv = RH::parseH(temp, tstr);

         REQUIRE(rv == 0);
         REQUIRE(temp == (float) 0.2);
      }
   }

   GIVEN("Invalid responses to H from the RH USB probe")
   {
      int rv;

      WHEN("Invalid humidity response 20.0 %RH\r>")
      {
         std::string tstr = "20.0 %RH\r>";
         float temp;

         rv = RH::parseH(temp, tstr);

         REQUIRE(rv == -1);
      }

      WHEN("Invalid humidity response 20.0%RH\r>")
      {
         std::string tstr = "20.0 %RH\r>";
         float temp;

         rv = RH::parseH(temp, tstr);

         REQUIRE(rv == -1);
      }

      WHEN("Invalid humidity response 20.0 \r\n>")
      {
         std::string tstr = "20.0 \r\n>";
         float temp;

         rv = RH::parseH(temp, tstr);

         REQUIRE(rv == -1);
      }

      WHEN("Invalid humidity response 20.0 >")
      {
         std::string tstr = "20.0 >";
         float temp;

         rv = RH::parseH(temp, tstr);

         REQUIRE(rv == -1);
      }

      WHEN("Invalid humidity response ' %RH\r\n>'")
      {
         std::string tstr = " %RH\r\n>";
         float temp;

         rv = RH::parseH(temp, tstr);

         REQUIRE(rv == -2);
      }

      WHEN("Invalid humidity response 'A %RH\r\n>'")
      {
         std::string tstr = "A %RH\r\n>";
         float temp;

         rv = RH::parseH(temp, tstr);

         REQUIRE(rv == -3);
      }

      WHEN("Invalid humidity response '-1.2 %RH\r\n>'")
      {
         std::string tstr = "-1.2 %RH\r\n>";
         float temp;

         rv = RH::parseH(temp, tstr);

         REQUIRE(rv == -3);
      }
   }
}

} //namespace ocamUtils_test 
