/****
   * Test following parsing functions
   * int parseCPUTemperatures(std::string, std::vector<float>&);
   * int parseCPULoads(std::string, float&);
   * int parseDiskTemperature(std::string, float&);
   * int parseDiskUsage(std::string, float&);
   * int parseRamUsage(std::string, float&);
   * To use:
   * In ~MagAOX/tests, compile this file with:
   * `make -f singleTest.mk testfile=../apps/sysMonitor/tests/sysMonitor_test.cpp`
   * and run program with:
   * `./singleTest`
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
      float temps = -1;

      // Fails with whitespace in front, but is this necessary to correct for?
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPUTemperatures("Core 0:         +42.0°C  (high = +100.0°C, crit = +100.0°C)", temps);
         REQUIRE(rv == 0);
         REQUIRE(temps == 42);
      }
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPUTemperatures("Core 1:         +45.0°C    (high = +100.0°C, crit = +100.0°C)", temps);
         REQUIRE(rv == 0);
         REQUIRE(temps == 45);
      }
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPUTemperatures("Core 2:         +91.0°C  (high = +100.0°C, crit = +100.0°C)", temps);
         REQUIRE(rv == 0);
         REQUIRE(temps == 91);
      }
      
      WHEN("Blank line is given")
      {
         rv = sm.parseCPUTemperatures("", temps);
         REQUIRE(rv == -1);
         REQUIRE(temps == -1);
      }

      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPUTemperatures("coretemp-isa-0000", temps);
         REQUIRE(rv == -1);
         REQUIRE(temps == -1);
      }
      
      WHEN("Corrupted line is given")
      {
         rv = sm.parseCPUTemperatures("Core 3:+91.0°C  (high = +100.0°C, crit = +100.0°C)", temps);
         REQUIRE(rv == -1);
         REQUIRE(temps == -1);
      }

      WHEN("Corrupted line is given")
      {
         rv = sm.parseCPUTemperatures("Core2:      +91.0°C(high =+100.0°C, crit= +100.0°C)", temps);
         REQUIRE(rv == -1);
         REQUIRE(temps == -1);
      }
   }
}

SCENARIO( "System monitor is constructed and CPU load results are passed in", "[sysMonitor]" ) 
{
   GIVEN("A default constructed system monitor object and an empty vector for loads")
   {
      MagAOX::app::sysMonitor sm;
      int rv;
      float loads = -1;

      // Fails with whitespace in front, but is this necessary to correct for?
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPULoads("02:35:43 PM    0    6.57    0.02    1.32    0.24    0.00    0.00    0.00    0.00    0.00   91.85", loads);
         REQUIRE(rv == 0);
         REQUIRE((loads - 0.0815) < 0.0005);
      }
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPULoads("10:32:28 AM    1    6.54    0.21    2.75   24.64    0.00    0.06    0.00    0.00    0.00   65.81", loads);
         REQUIRE(rv == 0);
         REQUIRE((loads - 0.3419) < 0.0005);
      }
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPULoads("10:32:28 AM    3    4.24    0.03    1.97    5.52    0.00    0.00    0.00    0.00    0.00   88.24", loads);
         REQUIRE(rv == 0);
         REQUIRE((loads - 0.1176) < 0.0005);
      }
      
      WHEN("Blank line is given")
      {
         rv = sm.parseCPULoads("", loads);
         REQUIRE(rv == -1);
         REQUIRE(loads == -1);
      }

      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPULoads("02:35:43 PM  CPU    %%usr   %%nice    %%sys %%iowait    %%irq   %%soft  %%steal  %%guest  %%gnice   %%idle", loads);
         REQUIRE(rv == -1);
         REQUIRE(loads == -1);
      }

      WHEN("Corrupted line is given")
      {
         rv = sm.parseCPULoads("10:32:28AM    2    5.24    0.14    2.70_1.41    0.00    0.00    0.00    0.00    0.00   80.50  ncawd vexing", loads);
         REQUIRE(rv == -1);
         REQUIRE(loads == -1);
      }
   }
}

SCENARIO( "System monitor is constructed and disk temperature result is passed in", "[sysMonitor]" ) 
{
   GIVEN("A default constructed system monitor object and an empty float for temperature")
   {
      MagAOX::app::sysMonitor sm;
      int rv;
      float hdd_temp = -1;

      // Fails with whitespace in front, but is this necessary to correct for?
      
      WHEN("Correct line is given for hard drive")
      {
         rv = sm.parseDiskTemperature("/dev/sda: ST1000LM024 HN-M101MBB: 31°C", hdd_temp);
         REQUIRE(rv == 0);
         REQUIRE(hdd_temp == 31);
      }
      
      WHEN("Correct line is given for ssd")
      {
         rv = sm.parseDiskTemperature("/dev/sda: Samsung SSD 860 EVO 500GB: 27°C", hdd_temp);
         REQUIRE(rv == 0);
         REQUIRE(hdd_temp == 27);
      }
      
      WHEN("Correct line is given for ssd")
      {
         rv = sm.parseDiskTemperature("/dev/sdd: Samsung SSD 860 EVO 1TB: 100°C", hdd_temp);
         REQUIRE(rv == 0);
         REQUIRE(hdd_temp == 100);
      }
      
      WHEN("Blank line is given")
      {
         rv = sm.parseDiskTemperature("", hdd_temp);
         REQUIRE(rv == -1);
         REQUIRE(hdd_temp == -1);
      }

      WHEN("Incorrect line is given")
      {
         rv = sm.parseDiskTemperature("/dev/sda: ST1000LM024_HN-M101MBB: 999999", hdd_temp);
         REQUIRE(rv == -1);
         REQUIRE(hdd_temp == -1);
      }

      WHEN("Corrupted line is given")
      {
         rv = sm.parseDiskTemperature("/dev/sdaT10 00L M0 24N-M101 MBB:31°CMBB", hdd_temp);
         REQUIRE(rv == -1);
         REQUIRE(hdd_temp == -1);
      }
   }
}

SCENARIO( "System monitor is constructed and disk usage result is passed in", "[sysMonitor]" ) 
{
   GIVEN("A default constructed system monitor object and an empty float for usage")
   {
      MagAOX::app::sysMonitor sm;
      int rv;
      float rootUsage = -1;
      float dataUsage = -1;
      float bootUsage = -1;

      // Fails with whitespace in front, but is this necessary to correct for?
      
      WHEN("Correct line is given for root")
      {
         rv = sm.parseDiskUsage("/dev/mapper/cl-root  52403200 12321848  40081352  24% /", rootUsage, dataUsage, bootUsage);
         REQUIRE(rv == 0);
         REQUIRE((rootUsage - 0.24f) < 0.0005);
      }
      
      WHEN("Correct line for /data is given")
      {
         rv = sm.parseDiskUsage("/dev/md124     1952297568    81552 1952216016   1% /data", rootUsage, dataUsage, bootUsage);
         REQUIRE(rv == 0);
         REQUIRE((dataUsage - 0.01f) < 0.0005);
      }
      
      WHEN("Correct line for /boot is given")
      {
         rv = sm.parseDiskUsage("/dev/md126         484004   289264     194740  60% /boot", rootUsage, dataUsage, bootUsage);
         REQUIRE(rv == 0);
         REQUIRE((bootUsage - 0.6f) < 0.0005);
      }
      
      WHEN("Blank line is given")
      {
         rv = sm.parseDiskUsage("", rootUsage, dataUsage, bootUsage);
         REQUIRE(rv == -1);
         REQUIRE(rootUsage == -1);
         REQUIRE(dataUsage == -1);
         REQUIRE(bootUsage == -1);
      }

      WHEN("Incorrect line is given")
      {
         rv = sm.parseDiskUsage("/dev/mapper/cl-root2403200 12321848  40081352  24% / 23e32 dwwe", rootUsage, dataUsage, bootUsage);
         REQUIRE(rv == -1);
         REQUIRE(rootUsage == -1);
         REQUIRE(dataUsage == -1);
         REQUIRE(bootUsage == -1);
      }

      WHEN("Corrupted line is given")
      {
         rv = sm.parseDiskUsage("/dev/mapper/cl-root  52403200 12321848  40081352  aa% /", rootUsage, dataUsage, bootUsage);
         REQUIRE(rv == -1);
         REQUIRE(rootUsage == -1);
         REQUIRE(dataUsage == -1);
         REQUIRE(bootUsage == -1);
      }
   }
}

SCENARIO( "System monitor is constructed and ram usage result is passed in", "[sysMonitor]" ) 
{
   GIVEN("A default constructed system monitor object and an float for usage")
   {
      MagAOX::app::sysMonitor sm;
      int rv;
      float ramUsage = -1;

      // Fails with whitespace in front, but is this necessary to correct for?
      
      WHEN("Correct line is given")
      {
         rv = sm.parseRamUsage("Mem:           7714        1308        4550         288        1855        5807", ramUsage);
         REQUIRE(rv == 0);
         REQUIRE(ramUsage == (float) 1308/7714);
      }
      
      WHEN("Correct line is given")
      {
         rv = sm.parseRamUsage("Mem:           7777        7700        4550         288        1855        5807", ramUsage);
         REQUIRE(rv == 0);
         REQUIRE(ramUsage == (float)7700/7777);
      }
      
      WHEN("Blank line is given")
      {
         rv = sm.parseRamUsage("", ramUsage);
         REQUIRE(rv == -1);
         REQUIRE(ramUsage == -1);
      }

      WHEN("Incorrect line is given")
      {
         rv = sm.parseRamUsage("Swap:          7935           0        7935", ramUsage);
         REQUIRE(rv == -1);
         REQUIRE(ramUsage == -1);
      }


      WHEN("Corrupted line is given")
      {
         rv = sm.parseRamUsage("Mem:           1308        7714        4550         288        1855        5807", ramUsage);
         REQUIRE(rv == -1);
         REQUIRE(ramUsage == -1);
      }
   }
}
