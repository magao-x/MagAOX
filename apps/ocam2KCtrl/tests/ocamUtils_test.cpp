
// #define CATCH_CONFIG_MAIN
// #include "../../../tests/catch2/catch.hpp"

#include "../../../tests/testMagAOX.hpp"

#include "../ocamUtils.hpp"

using namespace MagAOX::app;



SCENARIO( "Parsing the temp response", "[ocamUtils]" )
{
   GIVEN("A valid response to temp from the OCAM")
   {
      int rv;

      WHEN("Valid temp response")
      {
         std::string tstr = "Temperatures : CCD[26.3] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] SET[200]\nCooling Power [102]mW.\n\n";
         ocamTemps temps;

         rv = parseTemps(temps, tstr);

         REQUIRE(rv == 0);
         REQUIRE(temps.CCD == (float) 26.3);
         REQUIRE(temps.CPU == (float) 41);
         REQUIRE(temps.POWER == (float) 34);
         REQUIRE(temps.BIAS == (float) 47);
         REQUIRE(temps.WATER == (float) 24.2);
         REQUIRE(temps.LEFT == (float) 33);
         REQUIRE(temps.RIGHT == (float) 38);
         REQUIRE(temps.SET == (float) 20.0);
         REQUIRE(temps.COOLING_POWER == (float) 102);
      }


   }
   
   GIVEN("An invalid response to temp from the OCAM, too short")
   {
      int rv;

      WHEN("Temp response is too short")
      {
         std::string tstr = "Temperatures : CCD[26.3] CPU[41] POWER[34] BIAS[47] WATER[24.2] LEFT[33] RIGHT[38] SET[200]\nCooling Power";
         ocamTemps temps;

         rv = parseTemps(temps, tstr);

         REQUIRE(rv == -1);
      }


   }
}


