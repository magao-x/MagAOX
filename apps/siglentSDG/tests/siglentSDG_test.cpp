
#define CATCH_CONFIG_MAIN
#include "../../../tests/catch2/catch.hpp"


#include "../siglentSDG.hpp"

using namespace MagAOX::app;



SCENARIO( "Parsing the OUTP? response", "[siglentSDG]" ) 
{
   GIVEN("A default constructed siglentSDG")
   {
      siglentSDG sdg;
      
      int rv;
      
      WHEN("Valid OUTP passed, off")
      {
         int channel = -10;
         int outp = -10;
         
         rv = sdg.parseOUTP(channel, outp, "C1:OUTP OFF,LOAD,HZ,PLRT,NOR");
         
         REQUIRE(rv == 0);
         REQUIRE(channel == 1);
         REQUIRE(outp == 0);         
      }
      
      WHEN("Valid OUTP passed, on")
      {
         int channel = -10;
         int outp = -10;
         
         rv = sdg.parseOUTP(channel, outp, "C2:OUTP ON,LOAD,HZ,PLRT,NOR");
         
         REQUIRE(rv == 0);
         REQUIRE(channel == 2);
         REQUIRE(outp == 1);         
      }
      
      WHEN("Valid OUTP passed, two-digit channel on")
      {
         int channel = -10;
         int outp = -10;
         
         rv = sdg.parseOUTP(channel, outp, "C35:OUTP ON,LOAD,HZ,PLRT,NOR");
         
         REQUIRE(rv == 0);
         REQUIRE(channel == 35);
         REQUIRE(outp == 1);         
      }
      
      WHEN("Invalid OUTP passed, no :")
      {
         int channel = -10;
         int outp = -10;
         
         rv = sdg.parseOUTP(channel, outp, "C2 OUTP ON,LOAD,HZ,PLRT,NOR");
         
         REQUIRE(rv == -1);
      }
      
      WHEN("Invalid OUTP passed, no sp")
      {
         int channel = -10;
         int outp = -10;
         
         rv = sdg.parseOUTP(channel, outp, "C2:OUTPON,LOAD,HZ,PLRT,NOR");
         
         REQUIRE(rv == -1);
      }
      
      WHEN("Invalid OUTP passed, end before N in ON")
      {
         int channel = -10;
         int outp = -10;
         
         rv = sdg.parseOUTP(channel, outp, "C2:OUTP O");
         
         REQUIRE(rv == -1);
      }
      
      WHEN("Invalid OUTP passed, end before :")
      {
         int channel = -10;
         int outp = -10;
         
         rv = sdg.parseOUTP(channel, outp, "C2");
         
         REQUIRE(rv == -1);
      }
   }
}
      
