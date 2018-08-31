

#define CATCH_CONFIG_MAIN
#include "../../../tests/catch2/catch.hpp"


#include "../sysMonitor.hpp"

SCENARIO( "System monitor is constructed and CPU temperature results are passed in", "[sysMonitor]" ) 
{
   GIVEN("A default constructed system monitor object and an empty vector for temperatures")
   {
      sysMonitor sm;
      int rv;
      std::vector<float> temps;

      /****
       * Test following parsing functions
       * int parseCPUTemperatures(std::string, std::vector<float>&);
       * int parseCPULoads(std::string, float&);
       * int parseDiskTemperature(std::string, float&);
       * int parseDiskUsage(std::string, float&);
       * int parseRamUsage(std::string, float&);
      ****/
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPUTemperatures("Core 0:         +42.0°C  (high = +100.0°C, crit = +100.0°C)", temps);
         REQUIRE(rv == 0);
         REQUIRE(temps.size() == 1);
      }
      
      WHEN("Another correct line is given")
      {
         rv = sm.parseCPUTemperatures("     Core   1:    +45.0°C    (high = +100.0°C, crit = +100.0°C)", temps);
         REQUIRE(rv == 0);
         REQUIRE(temps.size() == 2);
      }
      
      WHEN("Another correct line is given")
      {
         rv = sm.parseCPUTemperatures("Core 2:      +91.0°C  (high = +100.0°C, crit = +100.0°C)", temps);
         REQUIRE(rv == 0);
         REQUIRE(temps.size() == 3);
      }
      
      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPUTemperatures("coretemp-isa-0000", temps);
         REQUIRE(rv == 1);
         REQUIRE(temps.size() == 3);
      }
      
      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPUTemperatures("Core 3:+91.0°C  (high = +100.0°C, crit = +100.0°C)", temps);
         REQUIRE(rv == 1);
         REQUIRE(temps.size() == 3);
      }

      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPUTemperatures("Core2:      +91.0°C(high =+100.0°C, crit= +100.0°C)", temps);
         REQUIRE(rv == 1);
         REQUIRE(temps.size() == 3);
      }
   }
}
/*
SCENARIO( "remoteDriver parses remote driver spec strings", "[remoteDriver]" ) 
{
   GIVEN("A default constructed remoteDriver")
   {
      remoteDriver rd;
      int rv;
      WHEN("A nominal remote driver spec is parsed (no port)")
      {
         rv = rd.parse("newDrive@remote");
         REQUIRE(rv == 0);
         REQUIRE(rd.name() == "newDrive");
         REQUIRE(rd.host() == "remote");
         REQUIRE(rd.port() == INDI_DEFAULT_PORT);
         REQUIRE(rd.hostSpec() == "remote:7624");
         REQUIRE(rd.fullSpec() == "newDrive@remote:7624");
 
      }
      
      WHEN("A whitespace-full remote driver spec is parsed (no port)")
      {
         rv = rd.parse("next drive @ 9. 8. 7 .6");
         REQUIRE(rv == 0);
         REQUIRE(rd.name() == "nextdrive");
         REQUIRE(rd.host() == "9.8.7.6");
         REQUIRE(rd.port() == INDI_DEFAULT_PORT);
      }
      
      WHEN("A nominal remote driver spec is parsed (w/ port)")
      {
         rv = rd.parse("nowDrive@remoteish:7");
         REQUIRE(rv == 0);
         REQUIRE(rd.name() == "nowDrive");
         REQUIRE(rd.host() == "remoteish");
         REQUIRE(rd.port() == 7);
      }
      
      WHEN("A whitespace-full remote driver spec is parsed (w/ port)")
      {
         rv = rd.parse("next B drive @ 9. 3. 7 .6 : 6 5 7");
         REQUIRE(rv == 0);
         REQUIRE(rd.name() == "nextBdrive");
         REQUIRE(rd.host() == "9.3.7.6");
         REQUIRE(rd.port() == 657);
      }
      
      WHEN("A valid remote driver spec is parsed (w/ empty port)")
      {
         rv = rd.parse("Q@B:");
         REQUIRE(rv == 0);
         REQUIRE(rd.name() == "Q");
         REQUIRE(rd.host() == "B");
         REQUIRE(rd.port() == INDI_DEFAULT_PORT);
      }
      
      WHEN("An invalid driver spec is parsed, @ first")
      {
         rv = rd.parse("@dname");
         REQUIRE(rv < 0);
         REQUIRE(rd.name() == "");
         REQUIRE(rd.host() == "");
      }
      
      WHEN("An invalid driver spec is parsed, @ last")
      {
         rv = rd.parse("@dname");
         REQUIRE(rv < 0);
         REQUIRE(rd.name() == "");
         REQUIRE(rd.host() == "");
      }
      
      WHEN("An invalid driver spec is parsed, no @")
      {
         rv = rd.parse("dname");
         REQUIRE(rv < 0);
         REQUIRE(rd.name() == "");
         REQUIRE(rd.host() == "");
      }
      
      WHEN("An invalid driver spec is parsed, : before @")
      {
         rv = rd.parse("dname:vg@:367");
         REQUIRE(rv < 0);
         REQUIRE(rd.name() == "");
         REQUIRE(rd.host() == "");
         REQUIRE(rd.port() == INDI_DEFAULT_PORT);
      }
      
   }
}
*/