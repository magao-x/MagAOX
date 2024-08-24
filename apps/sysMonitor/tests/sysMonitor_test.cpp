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

#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../sysMonitor.hpp"

using namespace MagAOX::app;

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
         rv = sm.parseCPUTemperatures(temps, "Core 0:         +42.0°C  (high = +100.0°C, crit = +100.0°C)");
         REQUIRE(rv == 0);
         REQUIRE(temps == 42);
      }
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPUTemperatures(temps, "Core 1:         +45.0°C    (high = +100.0°C, crit = +100.0°C)");
         REQUIRE(rv == 0);
         REQUIRE(temps == 45);
      }
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPUTemperatures(temps, "Core 2:         +91.0°C  (high = +100.0°C, crit = +100.0°C)");
         REQUIRE(rv == 0);
         REQUIRE(temps == 91);
      }
      
      WHEN("Blank line is given")
      {
         rv = sm.parseCPUTemperatures(temps, "");
         REQUIRE(rv == -1);
         REQUIRE(temps == -999);
      }

      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPUTemperatures(temps, "coretemp-isa-0000");
         REQUIRE(rv == -1);
         REQUIRE(temps == -999);
      }
      
      WHEN("Corrupted line is given")
      {
         rv = sm.parseCPUTemperatures(temps, "Core 3:+91.0° XXXXXXX");
         REQUIRE(rv == -1);
         REQUIRE(temps == -999);
      }

      WHEN("Corrupted line is given")
      {
         rv = sm.parseCPUTemperatures(temps, "Core2:      +91.0°C(high =+100.0°C, crit= +100.0°C)");
         REQUIRE(rv == -1);
         REQUIRE(temps == -999);
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
         rv = sm.parseCPULoads(loads, "02:35:43 PM    0    6.57    0.02    1.32    0.24    0.00    0.00    0.00    0.00    0.00   91.85");
         REQUIRE(rv == 0);
         REQUIRE((loads - 0.0815) < 0.0005);
      }
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPULoads(loads, "10:32:28 AM    1    6.54    0.21    2.75   24.64    0.00    0.06    0.00    0.00    0.00   65.81");
         REQUIRE(rv == 0);
         REQUIRE((loads - 0.3419) < 0.0005);
      }
      
      WHEN("Correct line is given")
      {
         rv = sm.parseCPULoads(loads, "10:32:28 AM    3    4.24    0.03    1.97    5.52    0.00    0.00    0.00    0.00    0.00   88.24");
         REQUIRE(rv == 0);
         REQUIRE((loads - 0.1176) < 0.0005);
      }
      
      WHEN("Blank line is given")
      {
         rv = sm.parseCPULoads(loads, "");
         REQUIRE(rv == -1);
         REQUIRE(loads == -1);
      }

      WHEN("Incorrect line is given")
      {
         rv = sm.parseCPULoads(loads, "02:35:43 PM  CPU    %%usr   %%nice    %%sys %%iowait    %%irq   %%soft  %%steal  %%guest  %%gnice   %%idle");
         REQUIRE(rv == -1);
         REQUIRE(loads == -1);
      }

      WHEN("Corrupted line is given")
      {
         rv = sm.parseCPULoads(loads, "10:32:28AM    2    5.24    0.14    2.70_1.41    0.00    0.00    0.00    0.00    0.00   80.50  ncawd vexing");
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
      std::string dname;
      
      // Fails with whitespace in front, but is this necessary to correct for?
      
      WHEN("Correct line is given for hard drive")
      {
         rv = sm.parseDiskTemperature(dname, hdd_temp, "/dev/sda: ST1000LM024 HN-M101MBB: 31°C");
         REQUIRE(rv == 0);
         REQUIRE(dname == "sda");
         REQUIRE(hdd_temp == 31);
      }
      
      WHEN("Correct line is given for ssd")
      {
         rv = sm.parseDiskTemperature(dname, hdd_temp, "/dev/sda: Samsung SSD 860 EVO 500GB: 27°C");
         REQUIRE(rv == 0);
         REQUIRE(dname == "sda");
         REQUIRE(hdd_temp == 27);
      }
      
      WHEN("Correct line is given for ssd")
      {
         rv = sm.parseDiskTemperature(dname, hdd_temp, "/dev/sdd: Samsung SSD 860 EVO 1TB: 100°C");
         REQUIRE(rv == 0);
         REQUIRE(dname == "sdd");
         REQUIRE(hdd_temp == 100);
      }
      
      WHEN("Blank line is given")
      {
         rv = sm.parseDiskTemperature(dname,  hdd_temp,"");
         REQUIRE(rv == -1);
         REQUIRE(dname == "");
         REQUIRE(hdd_temp == -999);
      }

      WHEN("Incorrect line is given")
      {
         rv = sm.parseDiskTemperature(dname, hdd_temp,"/dev/sda: ST1000LM024_HN-M101MBB: 999999");
         REQUIRE(rv == -1);
         REQUIRE(dname == "");
         REQUIRE(hdd_temp == -999);
      }

      WHEN("Corrupted line is given")
      {
         rv = sm.parseDiskTemperature(dname, hdd_temp, "/dev/sdaT10 00L M0 24N-M101 MBB:31°CMBB");
         REQUIRE(rv == -1);
         REQUIRE(dname == "");
         REQUIRE(hdd_temp == -999);
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

namespace SYSMONTEST
{

class sysMonitor_test : public sysMonitor 
{

public:
    sysMonitor_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(setlat);
    }
};

SCENARIO( "INDI Callbacks", "[sysMonitor]" )
{
    XWCTEST_INDI_NEW_CALLBACK( sysMonitor, setlat);
}

}//namespace SYSMONTEST