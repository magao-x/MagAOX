/** \file ocamUtils_test.cpp
  * \brief Catch2 tests for the ocamUtils in the ocam2KCtrl app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */
#include "../../../tests/catch2/catch.hpp"

#include "../ocamUtils.hpp"

using namespace MagAOX::app;

namespace ocamUtils_test 
{

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

SCENARIO( "Parsing the gain response", "[ocamUtils]" )
{
   GIVEN("A valid response to gain from the OCAM")
   {
      int rv;

      WHEN("Valid gain response, gain=2")
      {
         std::string tstr = "Gain set to 2 \n\n";

         unsigned emgain = 1;
         
         rv = parseEMGain(emgain, tstr);

         REQUIRE(rv == 0);
         REQUIRE(emgain == 2);
      }


   }
   
   GIVEN("A valid response to gain from the OCAM")
   {
      int rv;

      WHEN("Valid gain response, gain=512")
      {
         std::string tstr = "Gain set to 512 \n\n";

         unsigned emgain = 1;
         
         rv = parseEMGain(emgain, tstr);

         REQUIRE(rv == 0);
         REQUIRE(emgain == 512);
      }

      

   }
   
   GIVEN("An invalid response to gain from the OCAM")
   {
      int rv;

      WHEN("Invalid gain response, too short, no trailing space")
      {
         std::string tstr = "Gain set to 512\n\n";

         unsigned emgain = 1;
         
         rv = parseEMGain(emgain, tstr);

         REQUIRE(rv == -1);
         REQUIRE(emgain == 0);
      }

      WHEN("Invalid gain response, too short, no gain")
      {
         std::string tstr = "Gain set to \n\n";

         unsigned emgain = 1;
         
         rv = parseEMGain(emgain, tstr);

         REQUIRE(rv == -1);
         REQUIRE(emgain == 0);
      }

      WHEN("Invalid gain response, too long")
      {
         std::string tstr = "Gain set to 512 rubbish added\n\n";

         unsigned emgain = 1;
         
         rv = parseEMGain(emgain, tstr);

         REQUIRE(rv == -1);
         REQUIRE(emgain == 0);
      }
      
      WHEN("Invalid gain response, low gain")
      {
         std::string tstr = "Gain set to 0 \n\n";

         unsigned emgain = 1;
         
         rv = parseEMGain(emgain, tstr);

         REQUIRE(rv == -1);
         REQUIRE(emgain == 0);
      }
      
      WHEN("Invalid gain response, high gain")
      {
         std::string tstr = "Gain set to 601 \n\n";

         unsigned emgain = 1;
         
         rv = parseEMGain(emgain, tstr);

         REQUIRE(rv == -1);
         REQUIRE(emgain == 0);
      }
      
      WHEN("Invalid gain response, bad gain")
      {
         std::string tstr = "Gain set to x \n\n";

         unsigned emgain = 1;
         
         rv = parseEMGain(emgain, tstr);

         REQUIRE(rv == -1);
         REQUIRE(emgain == 0);
      }
   }
}
   
} //namespace ocamUtils_test 
