/****
   * Test following parsing functions
   * int parseCPUTemperatures(std::string, std::vector<float>&);
   * int parseCPULoads(std::string, float&);
   * int parseDiskTemperature(std::string, float&);
   * int parseDiskUsage(std::string, float&);
   * int parseRamUsage(std::string, float&);
   ****/

#define CATCH_CONFIG_MAIN
#include "../../../tests/catch2/catch.hpp"


#include "../sysMonitor.hpp"

SCENARIO( "System monitor is constructed and CPU temperature results are passed in", "[sysMonitor]" ) 
{
   GIVEN("A default constructed system monitor object and an empty vector for temperatures")
   {
      MagAOX::app::sysMonitor sm;
      int rv;
      std::vector<float> temps;

      // Fails with whitespace in front, but is this necessary to correct for?
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPUTemperatures("Core 0:         +42.0°C  (high = +100.0°C, crit = +100.0°C)", temps);
         REQUIRE(rv == 0);
         REQUIRE(temps.size() == 1);
      }
      
      WHEN("Another correct line is given")
      {
         rv = sm.parseCPUTemperatures("Core   1:    +45.0°C    (high = +100.0°C, crit = +100.0°C)", temps);
         REQUIRE(rv == 0);
         REQUIRE(temps.size() == 1);
      }
      
      WHEN("Another correct line is given")
      {
         rv = sm.parseCPUTemperatures("Core 2:      +91.0°C  (high = +100.0°C, crit = +100.0°C)", temps);
         REQUIRE(rv == 0);
         REQUIRE(temps.size() == 1);
      }
      
      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPUTemperatures("coretemp-isa-0000", temps);
         REQUIRE(rv == 1);
         REQUIRE(temps.size() == 0);
      }
      
      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPUTemperatures("Core 3:+91.0°C  (high = +100.0°C, crit = +100.0°C)", temps);
         REQUIRE(rv == 1);
         REQUIRE(temps.size() == 0);
      }

      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPUTemperatures("Core2:      +91.0°C(high =+100.0°C, crit= +100.0°C)", temps);
         REQUIRE(rv == 1);
         REQUIRE(temps.size() == 0);
      }
   }
}

SCENARIO( "System monitor is constructed and CPU load results are passed in", "[sysMonitor]" ) 
{
   GIVEN("A default constructed system monitor object and an empty vector for loads")
   {
      MagAOX::app::sysMonitor sm;
      int rv;
      std::vector<float> loads;

      // Fails with whitespace in front, but is this necessary to correct for?
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPUTemperatures("02:35:43 PM    0    6.57    0.02    1.32    0.24    0.00    0.00    0.00    0.00    0.00   91.85", loads);
         REQUIRE(rv == 0);
         REQUIRE(loads.size() == 1);
      }
      
      WHEN("Another correct line is given")
      {
         rv = sm.parseCPUTemperatures("", loads);
         REQUIRE(rv == 0);
         REQUIRE(loads.size() == 1);
      }
      
      WHEN("Another correct line is given")
      {
         rv = sm.parseCPUTemperatures("", loads);
         REQUIRE(rv == 0);
         REQUIRE(loads.size() == 1);
      }
      
      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPUTemperatures("02:35:43 PM  CPU    %%usr   %%nice    %%sys %%iowait    %%irq   %%soft  %%steal  %%guest  %%gnice   %%idle", loads);
         REQUIRE(rv == 1);
         REQUIRE(loads.size() == 0);
      }
      
      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPUTemperatures("02:35:43 PM  all    6.59    0.02    1.32    0.42    0.00    0.00    0.00    0.00    0.00   91.65", loads);
         REQUIRE(rv == 1);
         REQUIRE(loads.size() == 0);
      }

      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPUTemperatures("02:35:43 PM0    6.57    zzz50.02    2vc51.32    0.24    0.00    0.00  c3t  c350.00    0.00    0.00   ct3591.85
", loads);
         REQUIRE(rv == 1);
         REQUIRE(loads.size() == 0);
      }
   }
}