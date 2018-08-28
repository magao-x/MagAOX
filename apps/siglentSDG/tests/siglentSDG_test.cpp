
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

      WHEN("Wrong Command Reply")
      {
         int channel = -10;
         int outp = -10;

         rv = sdg.parseOUTP(channel, outp, "C1:BSWV WVTP,SINE,FRQ,10HZ,PERI,0.1S,AMP,2V,AMPVRMS,0.707Vrms,OFST,0V,HLEV,1V,LLEV,-1V,PHSE,0");

         REQUIRE(rv == -1);
      }
   }
}
SCENARIO( "Parsing the OSWV? response", "[siglentSDG]" )
{
   GIVEN("A default constructed siglentSDG")
   {
      siglentSDG sdg;

      int rv;

      WHEN("Valid OSWV passed")
      {
         int channel = -10;
         std::string wvtp;
         double freq;
         double peri;
         double amp;
         double ampvrms;
         double ofst;
         double hlev;
         double llev;
         double phse;
         std::string resp="C1:BSWV WVTP,SINE,FRQ,10.123HZ,PERI,0.8345S,AMP,2.567V,AMPVRMS,0.707Vrms,OFST,0.34V,HLEV,1.3V,LLEV,-2.567V,PHSE,4.3567";
         rv = sdg.parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 1);
         REQUIRE(wvtp == "SINE");
         REQUIRE(freq == 10.123);
         REQUIRE(peri == 0.8345);
         REQUIRE(amp == 2.567);
         REQUIRE(ampvrms == 0.707);
         REQUIRE(ofst == 0.34);
         REQUIRE(hlev == 1.3 );
         REQUIRE(llev == -2.567 );
         REQUIRE(phse ==4.3567 );
      }
   }
}
