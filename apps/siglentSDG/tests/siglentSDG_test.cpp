/** \file siglentSDG_test.cpp
  * \brief Catch2 tests for the siglentSDG app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */

#include "../../../tests/catch2/catch.hpp"

#include "../siglentSDG.hpp"

using namespace MagAOX::app;

namespace siglentSDG_test 
{

SCENARIO( "Parsing the OUTP? response", "[siglentSDG]" )
{
   GIVEN("A valid response to OUTP from the SDG")
   {
      int rv;

      WHEN("Valid OUTP passed, off")
      {
         int channel = -10;
         int outp = -10;

         rv = parseOUTP(channel, outp, "C1:OUTP OFF,LOAD,HZ,PLRT,NOR");

         REQUIRE(rv == 0);
         REQUIRE(channel == 1);
         REQUIRE(outp == 0);
      }

      WHEN("Valid OUTP passed, on")
      {
         int channel = -10;
         int outp = -10;

         rv = parseOUTP(channel, outp, "C2:OUTP ON,LOAD,HZ,PLRT,NOR");

         REQUIRE(rv == 0);
         REQUIRE(channel == 2);
         REQUIRE(outp == 1);
      }

      WHEN("Valid OUTP passed, two-digit channel on")
      {
         int channel = -10;
         int outp = -10;

         rv = parseOUTP(channel, outp, "C35:OUTP ON,LOAD,HZ,PLRT,NOR");

         REQUIRE(rv == 0);
         REQUIRE(channel == 35);
         REQUIRE(outp == 1);
      }
   }
   GIVEN("An invalid response to OUTP from the SDG")
   {
      int rv;
      
      WHEN("Invalid OUTP passed, no sp")
      {
         int channel = -10;
         int outp = -10;

         rv = parseOUTP(channel, outp, "C2:OUTPON,LOAD,HZ,PLRT,NOR");

         REQUIRE(rv == -2);
      }

      WHEN("Invalid OUTP passed, end before N in ON")
      {
         int channel = -10;
         int outp = -10;

         rv = parseOUTP(channel, outp, "C2:OUTP O");

         REQUIRE(rv == -6);
      }

      WHEN("Invalid OUTP passed, end before :")
      {
         int channel = -10;
         int outp = -10;

         rv = parseOUTP(channel, outp, "C2");

         REQUIRE(rv == -1);
      }

      WHEN("Wrong Command Reply")
      {
         int channel = -10;
         int outp = -10;

         rv = parseOUTP(channel, outp, "C1:BSWV WVTP,SINE,FRQ,10HZ,PERI,0.1S,AMP,2V,AMPVRMS,0.707Vrms,OFST,0V,HLEV,1V,LLEV,-1V,PHSE,0");

         REQUIRE(rv == -2);
      }
   }
}

SCENARIO( "Parsing the BSWV? response", "[siglentSDG]" )
{
   GIVEN("A valid response to BSWV from the SDG")
   {
      int rv;

      WHEN("Valid BSWV passed")
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
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

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
   
   GIVEN("An invalid response to BSWV from the SDG")
   {
      int rv;

      WHEN("An invalid BSWV passed - not enough args")
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
         std::string resp="C1:BSWV WVTP";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -1);
      }
      
      WHEN("An invalid BSWV passed - wrong response")
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
         std::string resp="C1:MDWV WVTP,SINE,FRQ,10.123HZ,PERI,0.8345S,AMP,2.567V,AMPVRMS,0.707Vrms,OFST,0.34V,HLEV,1.3V,LLEV,-2.567V,PHSE,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -2);
      }
   
      WHEN("An invalid BSWV passed - bad channel spec, no C")
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
         std::string resp="X1:BSWV WVTP,SINE,FRQ,10.123HZ,PERI,0.8345S,AMP,2.567V,AMPVRMS,0.707Vrms,OFST,0.34V,HLEV,1.3V,LLEV,-2.567V,PHSE,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -3);
      }
      
      WHEN("An invalid BSWV passed - bad channel spec, too short ")
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
         std::string resp="C:BSWV WVTP,SINE,FRQ,10.123HZ,PERI,0.8345S,AMP,2.567V,AMPVRMS,0.707Vrms,OFST,0.34V,HLEV,1.3V,LLEV,-2.567V,PHSE,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -4);
      }
      
      WHEN("An invalid BSWV passed - bad WVTP indicator")
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
         std::string resp="C1:BSWV WVTQ,SINE,FRQ,10.123HZ,PERI,0.8345S,AMP,2.567V,AMPVRMS,0.707Vrms,OFST,0.34V,HLEV,1.3V,LLEV,-2.567V,PHSE,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -5);
      }
      
      WHEN("An invalid BSWV passed - wvtp not SINE")
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
         std::string resp="C1:BSWV WVTP,UPIY,FRQ,10.123HZ,PERI,0.8345S,AMP,2.567V,AMPVRMS,0.707Vrms,OFST,0.34V,HLEV,1.3V,LLEV,-2.567V,PHSE,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -6);
      }

      WHEN("An invalid BSWV passed - bad FRQ indicator")
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
         std::string resp="C1:BSWV WVTP,SINE,FRZ,10.123HZ,PERI,0.8345S,AMP,2.567V,AMPVRMS,0.707Vrms,OFST,0.34V,HLEV,1.3V,LLEV,-2.567V,PHSE,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -10);
      }
      
      WHEN("An invalid BSWV passed - bad PERI indicator")
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
         std::string resp="C1:BSWV WVTP,SINE,FRQ,10.123HZ,PERZ,0.8345S,AMP,2.567V,AMPVRMS,0.707Vrms,OFST,0.34V,HLEV,1.3V,LLEV,-2.567V,PHSE,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -11);
      }
      
      WHEN("An invalid BSWV passed - bad AMP indicator")
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
         std::string resp="C1:BSWV WVTP,SINE,FRQ,10.123HZ,PERI,0.8345S,A/P,2.567V,AMPVRMS,0.707Vrms,OFST,0.34V,HLEV,1.3V,LLEV,-2.567V,PHSE,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -12);
      }
      
      WHEN("An invalid BSWV passed - bad AMPVRMS indicator")
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
         std::string resp="C1:BSWV WVTP,SINE,FRQ,10.123HZ,PERI,0.8345S,AMP,2.567V,APVRMS,0.707Vrms,OFST,0.34V,HLEV,1.3V,LLEV,-2.567V,PHSE,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -13);
      }
      
      WHEN("An invalid BSWV passed - bad OFST indicator")
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
         std::string resp="C1:BSWV WVTP,SINE,FRQ,10.123HZ,PERI,0.8345S,AMP,2.567V,AMPVRMS,0.707Vrms,O,0.34V,HLEV,1.3V,LLEV,-2.567V,PHSE,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -14);
      }
      
      WHEN("An invalid BSWV passed - bad HLEV indicator")
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
         std::string resp="C1:BSWV WVTP,SINE,FRQ,10.123HZ,PERI,0.8345S,AMP,2.567V,AMPVRMS,0.707Vrms,OFST,0.34V,HLV,1.3V,LLEV,-2.567V,PHSE,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -15);
      }
      
      WHEN("An invalid BSWV passed - bad LLEV indicator")
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
         std::string resp="C1:BSWV WVTP,SINE,FRQ,10.123HZ,PERI,0.8345S,AMP,2.567V,AMPVRMS,0.707Vrms,OFST,0.34V,HLEV,1.3V,QLEV,-2.567V,PHSE,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -16);
      }
      
      WHEN("An invalid BSWV passed - bad PHSE indicator")
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
         std::string resp="C1:BSWV WVTP,SINE,FRQ,10.123HZ,PERI,0.8345S,AMP,2.567V,AMPVRMS,0.707Vrms,OFST,0.34V,HLEV,1.3V,LLEV,-2.567V,XXXXX,4.3567";
         rv = parseBSWV(channel, wvtp, freq, peri, amp, ampvrms, ofst, hlev, llev, phse, resp);

         REQUIRE(rv == -17);
      }
   }
}

SCENARIO( "Parsing the MDWV? response", "[siglentSDG]" )
{
   GIVEN("A valid response to MDWV from the SDG")
   {
      int rv;

      WHEN("Valid MDWV passed, with state off")
      {
         int channel = -10;
         std::string state;
         std::string resp="C1:MDWV STATE,OFF";
         rv = parseMDWV(channel, state, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 1);
         REQUIRE(state == "OFF");
     }
     
      WHEN("Valid MDWV passed, with state on")
      {
        //We ignore the rest of the string
         int channel = -10;
         std::string state;
         std::string resp="C1:MDWV STATE,ON,AM,MDSP,SINE,SRC,INT,FRQ,100HZ,DEPTH,100,CARR,WVTP,SINE,FRQ,1000HZ,AMP,4V,AMPVRMS,1.414Vrms,OFST,0V,PHSE,0";
         rv = parseMDWV(channel, state, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 1);
         REQUIRE(state == "ON");
     }
   }
   
   GIVEN("An invalid response to MDWV from the SDG")
   {
      int rv;

      WHEN("invalid MDWV passed - too short")
      {
         int channel = -10;
         std::string state;
         std::string resp="C1:MDWV S";
         rv = parseMDWV(channel, state, resp);

         REQUIRE(rv == -1);
      }
     
      WHEN("invalid MDWV passed - wrong command")
      {
         int channel = -10;
         std::string state;
         std::string resp="C1:MDWQ STATE,OFF";
         rv = parseMDWV(channel, state, resp);

         REQUIRE(rv == -2);
      }
     
      WHEN("invalid MDWV passed - no C")
      {
         int channel = -10;
         std::string state;
         std::string resp="X1:MDWV STATE,OFF";
         rv = parseMDWV(channel, state, resp);

         REQUIRE(rv == -3);
      }
     
      WHEN("invalid MDWV passed - no channel")
      {
         int channel = -10;
         std::string state;
         std::string resp="C:MDWV STATE,OFF";
         rv = parseMDWV(channel, state, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 0);
      }   
     
      WHEN("invalid MDWV passed - no STATE")
      {
         int channel = -10;
         std::string state;
         std::string resp="C1:MDWV STAT,OFF";
         rv = parseMDWV(channel, state, resp);

         REQUIRE(rv == -4);
      }
   }
}

SCENARIO( "Parsing the SWWV? response", "[siglentSDG]" )
{
   GIVEN("A valid response to SWWV from the SDG")
   {
      int rv;

      WHEN("Valid SWWV passed, with state off")
      {
         int channel = -10;
         std::string state;
         std::string resp="C1:SWWV STATE,OFF";
         rv = parseSWWV(channel, state, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 1);
         REQUIRE(state == "OFF");
     }
     
      WHEN("Valid SWWV passed, with state on")
      {
        //We ignore the rest of the string
         int channel = -10;
         std::string state;
         std::string resp="C1:SWWV STATE,ON,TIME,1S,STOP,1500HZ,START,500HZ,TRSR,INT,TRMD,OFF,SWMD,LINE,DIR,UP,SYM,0.000000,CARR,WVTP,SINE,FRQ,1000HZ,AMP,4V,AMPVRMS,1.414Vrms,OFST,0V,PHSE,0";
         rv = parseSWWV(channel, state, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 1);
         REQUIRE(state == "ON");
     }
   }
   
   GIVEN("An invalid response to SWWV from the SDG")
   {
      int rv;

      WHEN("invalid SWWV passed - too short")
      {
         int channel = -10;
         std::string state;
         std::string resp="C1:SWWV S";
         rv = parseSWWV(channel, state, resp);

         REQUIRE(rv == -1);
      }
     
      WHEN("invalid SWWV passed - wrong command")
      {
         int channel = -10;
         std::string state;
         std::string resp="C1:SWWQ STATE,OFF";
         rv = parseSWWV(channel, state, resp);

         REQUIRE(rv == -2);
      }
     
      WHEN("invalid SWWV passed - no C")
      {
         int channel = -10;
         std::string state;
         std::string resp="X1:SWWV STATE,OFF";
         rv = parseSWWV(channel, state, resp);

         REQUIRE(rv == -3);
      }
     
      WHEN("invalid SWWV passed - no channel")
      {
         int channel = -10;
         std::string state;
         std::string resp="C:SWWV STATE,OFF";
         rv = parseSWWV(channel, state, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 0);
      }   
     
      WHEN("invalid SWWV passed - no STATE")
      {
         int channel = -10;
         std::string state;
         std::string resp="C1:SWWV STAT,OFF";
         rv = parseSWWV(channel, state, resp);

         REQUIRE(rv == -4);
      }
   }
}


SCENARIO( "Parsing the BTWV? response", "[siglentSDG]" )
{
   GIVEN("A valid response to BTWV from the SDG")
   {
      int rv;

      WHEN("Valid BTWV passed, with state off")
      {
         int channel = -10;
         std::string state;
         std::string resp="C1:BTWV STATE,OFF";
         rv = parseBTWV(channel, state, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 1);
         REQUIRE(state == "OFF");
     }
     
      WHEN("Valid BTWV passed, with state on")
      {
        //We ignore the rest of the string
         int channel = -10;
         std::string state;
         std::string resp="C1:BTWV STATE,ON,PRD,0.01S,STPS,0,TRSR,INT,TRMD,OFF,TIME,1,DLAY,5.21035e-07S,GATE_NCYC,NCYC,CARR,WVTP,SINE,FRQ,1000HZ,AMP,4V,AMPVRMS,1.414Vrms,OFST,0V,PHSE,0";
         rv = parseBTWV(channel, state, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 1);
         REQUIRE(state == "ON");
     }
   }
   
   GIVEN("An invalid response to BTWV from the SDG")
   {
      int rv;

      WHEN("invalid BTWV passed - too short")
      {
         int channel = -10;
         std::string state;
         std::string resp="C1:BTWV S";
         rv = parseBTWV(channel, state, resp);

         REQUIRE(rv == -1);
      }
     
      WHEN("invalid BTWV passed - wrong command")
      {
         int channel = -10;
         std::string state;
         std::string resp="C1:BTWQ STATE,OFF";
         rv = parseBTWV(channel, state, resp);

         REQUIRE(rv == -2);
      }
     
      WHEN("invalid BTWV passed - no C")
      {
         int channel = -10;
         std::string state;
         std::string resp="X1:BTWV STATE,OFF";
         rv = parseBTWV(channel, state, resp);

         REQUIRE(rv == -3);
      }
     
      WHEN("invalid BTWV passed - no channel")
      {
         int channel = -10;
         std::string state;
         std::string resp="C:BTWV STATE,OFF";
         rv = parseBTWV(channel, state, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 0);
      }   
     
      WHEN("invalid BTWV passed - no STATE")
      {
         int channel = -10;
         std::string state;
         std::string resp="C1:BTWV STAT,OFF";
         rv = parseBTWV(channel, state, resp);

         REQUIRE(rv == -4);
      }
   }
}

SCENARIO( "Parsing the ARWV? response", "[siglentSDG]" )
{
   GIVEN("A valid response to ARWV from the SDG")
   {
      int rv;

      WHEN("Valid ARWV passed, with index 0")
      {
         int channel = -10;
         int index = -10;
         ;
         std::string resp="C1:ARWV INDEX,0,NAME,";
         rv = parseARWV(channel, index, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 1);
         REQUIRE(index == 0);
     }
     
      WHEN("Valid ARWV passed, with index 1")
      {
        //We ignore the rest of the string
         int channel = -10;
         int index = -10;
         std::string resp="C2:ARWV INDEX,1,NAME,";
         rv = parseARWV(channel, index, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 2);
         REQUIRE(index == 1);
     }
   }
   
   GIVEN("An invalid response to ARWV from the SDG")
   {
      int rv;

      WHEN("invalid ARWV passed - too short")
      {
         int channel = -10;
         int index = -10;
         std::string resp="C1:ARWV I";
         rv = parseARWV(channel, index, resp);

         REQUIRE(rv == -1);
      }
     
      WHEN("invalid ARWV passed - wrong command")
      {
         int channel = -10;
         int index = -10;
         std::string resp="C1:ARWQ INDEX,0";
         rv = parseARWV(channel, index, resp);

         REQUIRE(rv == -2);
      }
     
      WHEN("invalid ARWV passed - no C")
      {
         int channel = -10;
         int index = -10;
         std::string resp="X1:ARWV INDEX,0";
         rv = parseARWV(channel, index, resp);

         REQUIRE(rv == -3);
      }
     
      WHEN("invalid ARWV passed - no channel")
      {
         int channel = -10;
         int index = -10;
         std::string resp="C:ARWV INDEX,0";
         rv = parseARWV(channel, index, resp);

         REQUIRE(rv == 0);
         REQUIRE(channel == 0);
      }   
     
      WHEN("invalid ARWV passed - no INDEX")
      {
         int channel = -10;
         int index = -10;
         std::string resp="C1:ARWV INDX,0";
         rv = parseARWV(channel, index, resp);

         REQUIRE(rv == -4);
      }
   }
}

} //namespace siglentSDG_test 
