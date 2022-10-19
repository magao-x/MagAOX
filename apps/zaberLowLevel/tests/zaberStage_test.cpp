/** \file zaberStage_test.cpp
  * \brief Catch2 tests for the zaberStage in the zaberLowLevel app
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */
#include "../../../tests/catch2/catch.hpp"

#include "../zaberLowLevel.hpp"

#include "../za_serial.c" //to allow test to compile

using namespace MagAOX::app;

namespace zaberStage_test 
{

SCENARIO( "Parsing the warnings response", "[zaberStage]" )
{
   GIVEN("A valid response to the warnings query")
   {
      int rv;

      WHEN("Valid response, no warnings")
      {
         zaberStage zstg;
         
         std::string tstr = "00";
 
         rv = zstg.parseWarnings(tstr);
         
         REQUIRE(rv == 0);
         REQUIRE(zstg.warningState() == false);
         REQUIRE(zstg.warnFD() == false);
         REQUIRE(zstg.warnFQ() == false);
         REQUIRE(zstg.warnFS() == false);
         REQUIRE(zstg.warnFT() == false);
         REQUIRE(zstg.warnFB() == false);
         REQUIRE(zstg.warnFP() == false);
         REQUIRE(zstg.warnFE() == false);
         REQUIRE(zstg.warnWH() == false);
         REQUIRE(zstg.warnWL() == false);
         REQUIRE(zstg.warnWP() == false);
         REQUIRE(zstg.warnWV() == false);
         REQUIRE(zstg.warnWT() == false);
         REQUIRE(zstg.warnWM() == false);
         REQUIRE(zstg.warnWR() == false);
         REQUIRE(zstg.warnNC() == false);
         REQUIRE(zstg.warnNI() == false);
         REQUIRE(zstg.warnND() == false);
         REQUIRE(zstg.warnNU() == false);
         REQUIRE(zstg.warnNJ() == false);
         REQUIRE(zstg.warnUNK()== false);
      }
   
      WHEN("Valid response, one warning")
      {
         zaberStage zstg;
         
         std::string tstr = "01 WR";
 
         rv = zstg.parseWarnings(tstr);
         
         REQUIRE(rv == 0);
         REQUIRE(zstg.warningState() == true);
         REQUIRE(zstg.warnFD() == false);
         REQUIRE(zstg.warnFQ() == false);
         REQUIRE(zstg.warnFS() == false);
         REQUIRE(zstg.warnFT() == false);
         REQUIRE(zstg.warnFB() == false);
         REQUIRE(zstg.warnFP() == false);
         REQUIRE(zstg.warnFE() == false);
         REQUIRE(zstg.warnWH() == false);
         REQUIRE(zstg.warnWL() == false);
         REQUIRE(zstg.warnWP() == false);
         REQUIRE(zstg.warnWV() == false);
         REQUIRE(zstg.warnWT() == false);
         REQUIRE(zstg.warnWM() == false);
         REQUIRE(zstg.warnWR() == true);
         REQUIRE(zstg.warnNC() == false);
         REQUIRE(zstg.warnNI() == false);
         REQUIRE(zstg.warnND() == false);
         REQUIRE(zstg.warnNU() == false);
         REQUIRE(zstg.warnNJ() == false);
         REQUIRE(zstg.warnUNK()== false);
      }
      
      WHEN("Valid response, five warnings")
      {
         zaberStage zstg;
         
         std::string tstr = "05 FD FQ FS FT FB";
 
         rv = zstg.parseWarnings(tstr);
         
         REQUIRE(rv == 0);
         REQUIRE(zstg.warningState() == true);
         REQUIRE(zstg.warnFD() == true);
         REQUIRE(zstg.warnFQ() == true);
         REQUIRE(zstg.warnFS() == true);
         REQUIRE(zstg.warnFT() == true);
         REQUIRE(zstg.warnFB() == true);
         REQUIRE(zstg.warnFP() == false);
         REQUIRE(zstg.warnFE() == false);
         REQUIRE(zstg.warnWH() == false);
         REQUIRE(zstg.warnWL() == false);
         REQUIRE(zstg.warnWP() == false);
         REQUIRE(zstg.warnWV() == false);
         REQUIRE(zstg.warnWT() == false);
         REQUIRE(zstg.warnWM() == false);
         REQUIRE(zstg.warnWR() == false);
         REQUIRE(zstg.warnNC() == false);
         REQUIRE(zstg.warnNI() == false);
         REQUIRE(zstg.warnND() == false);
         REQUIRE(zstg.warnNU() == false);
         REQUIRE(zstg.warnNJ() == false);
         REQUIRE(zstg.warnUNK()== false);
      }
      
      WHEN("Valid response, ten warnings")
      {
         zaberStage zstg;
         
         std::string tstr = "10 FP FE WH WL WP WV WT WM WR NC";
 
         rv = zstg.parseWarnings(tstr);
         
         REQUIRE(rv == 0);
         REQUIRE(zstg.warningState() == true);
         REQUIRE(zstg.warnFD() == false);
         REQUIRE(zstg.warnFQ() == false);
         REQUIRE(zstg.warnFS() == false);
         REQUIRE(zstg.warnFT() == false);
         REQUIRE(zstg.warnFB() == false);
         REQUIRE(zstg.warnFP() == true);
         REQUIRE(zstg.warnFE() == true);
         REQUIRE(zstg.warnWH() == true);
         REQUIRE(zstg.warnWL() == true);
         REQUIRE(zstg.warnWP() == true);
         REQUIRE(zstg.warnWV() == true);
         REQUIRE(zstg.warnWT() == true);
         REQUIRE(zstg.warnWM() == true);
         REQUIRE(zstg.warnWR() == true);
         REQUIRE(zstg.warnNC() == true);
         REQUIRE(zstg.warnNI() == false);
         REQUIRE(zstg.warnND() == false);
         REQUIRE(zstg.warnNU() == false);
         REQUIRE(zstg.warnNJ() == false);
         REQUIRE(zstg.warnUNK()== false);
      }
      WHEN("Valid response, 2 warnings")
      {
         zaberStage zstg;
         
         std::string tstr = "02 NI ND";
      
         rv = zstg.parseWarnings(tstr);
         
         REQUIRE(rv == 0);
         REQUIRE(zstg.warningState() == true);
         REQUIRE(zstg.warnFD() == false);
         REQUIRE(zstg.warnFQ() == false);
         REQUIRE(zstg.warnFS() == false);
         REQUIRE(zstg.warnFT() == false);
         REQUIRE(zstg.warnFB() == false);
         REQUIRE(zstg.warnFP() == false);
         REQUIRE(zstg.warnFE() == false);
         REQUIRE(zstg.warnWH() == false);
         REQUIRE(zstg.warnWL() == false);
         REQUIRE(zstg.warnWP() == false);
         REQUIRE(zstg.warnWV() == false);
         REQUIRE(zstg.warnWT() == false);
         REQUIRE(zstg.warnWM() == false);
         REQUIRE(zstg.warnWR() == false);
         REQUIRE(zstg.warnNC() == false);
         REQUIRE(zstg.warnNI() == true);
         REQUIRE(zstg.warnND() == true);
         REQUIRE(zstg.warnNU() == false);
         REQUIRE(zstg.warnNJ() == false);
         REQUIRE(zstg.warnUNK()== false);
      }
      WHEN("Valid response, 3 warnings")
      {
         zaberStage zstg;
         
         std::string tstr = "03 NU NJ UN";
      
         rv = zstg.parseWarnings(tstr);
         
         REQUIRE(rv == 0);
         REQUIRE(zstg.warningState() == true);
         REQUIRE(zstg.warnFD() == false);
         REQUIRE(zstg.warnFQ() == false);
         REQUIRE(zstg.warnFS() == false);
         REQUIRE(zstg.warnFT() == false);
         REQUIRE(zstg.warnFB() == false);
         REQUIRE(zstg.warnFP() == false);
         REQUIRE(zstg.warnFE() == false);
         REQUIRE(zstg.warnWH() == false);
         REQUIRE(zstg.warnWL() == false);
         REQUIRE(zstg.warnWP() == false);
         REQUIRE(zstg.warnWV() == false);
         REQUIRE(zstg.warnWT() == false);
         REQUIRE(zstg.warnWM() == false);
         REQUIRE(zstg.warnWR() == false);
         REQUIRE(zstg.warnNC() == false);
         REQUIRE(zstg.warnNI() == false);
         REQUIRE(zstg.warnND() == false);
         REQUIRE(zstg.warnNU() == true);
         REQUIRE(zstg.warnNJ() == true);
         REQUIRE(zstg.warnUNK()== true);
      }
   }   
}
   
} //namespace zaberStage_test 
