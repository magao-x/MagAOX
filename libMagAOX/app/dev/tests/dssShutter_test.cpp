/** \file dssShutter_test.cpp
  * \brief Catch2 tests for the MagAOX::app::dev::dssShutter device mixin.
  *
  * The mixin is exercised through the dssShutterTest harness declared in dssShutter_test.hpp.
  * It never connects to INDI. The harness hides registerIndiPropertySet, threadStart,
  * sendNewProperty, and recordCamera. This lets each test force a failure path and simulate
  * the digital I/O hardware that drives and senses the shutter. The open and shut background
  * threads are started for real. Log calls are captured in static members and checked by
  * count and by message text.
  *
  * The configuration tests write /tmp/dssShutter_test.conf.
  *
  * \ingroup testing
  */

#include "../../../../tests/catch2/catch.hpp"

#include <mx/sys/timeUtils.hpp>

#include "dssShutter_test.hpp"

using namespace MagAOX::app;

/** \defgroup dssShutter_tests libXWC::app::dev::dssShutter Unit Tests
 * \ingroup app_dev_unit_tests
 */
namespace dssShutter_tests
{

/// Write a complete dssShutter config file to /tmp with the given wait and timeout values.
void writeShutterConfig( const std::string &wait = "100", const std::string &timeout = "2000" )
{
   mx::app::writeConfigFile( "/tmp/dssShutter_test.conf",
      { "shutter",     "shutter",      "shutter",   "shutter",       "shutter",         "shutter", "shutter" },
      { "powerDevice", "powerChannel", "dioDevice", "sensorChannel", "triggerChannel",  "wait",    "timeout" },
      { "pdu",         "shutterPower", "dio",       "shutterSensor", "shutterTrigger",   wait,      timeout } );
}

/// Verify that setupConfig and loadConfig read every shutter option from a config file.
/**
 * \ingroup dssShutter_tests
 */
SCENARIO( "dssShutter configuration", "[dev::dssShutter]" )
{
   GIVEN("a complete shutter config section")
   {
      writeShutterConfig("150", "3000");

      mx::app::appConfigurator config;
      dssShutterTest sh;

      int rv = sh.setupConfig(config);
      REQUIRE( rv == 0 );

      config.readConfig("/tmp/dssShutter_test.conf");

      rv = sh.loadConfig(config);
      REQUIRE( rv == 0 );

      REQUIRE( sh.powerDeviceVal() == "pdu" );
      REQUIRE( sh.powerChannelVal() == "shutterPower" );
      REQUIRE( sh.dioDeviceVal() == "dio" );
      REQUIRE( sh.sensorChannelVal() == "shutterSensor" );
      REQUIRE( sh.triggerChannelVal() == "shutterTrigger" );
      REQUIRE( sh.shutterWaitVal() == 150 );
      REQUIRE( sh.shutterTimeoutVal() == 3000 );
   }
}

/// Verify that appStartup registers the three INDI properties and starts both worker threads.
/** Each INDI registration failure and each thread-start failure is then forced in turn.
 * Every forced failure must return -1 and produce exactly one log entry.
 *
 * \ingroup dssShutter_tests
 */
SCENARIO( "dssShutter appStartup", "[dev::dssShutter]" )
{
   GIVEN("a fully configured shutter")
   {
      writeShutterConfig("1", "5");

      mx::app::appConfigurator config;

      WHEN("startup succeeds")
      {
         dssShutterTest sh;
         sh.setupConfig(config);
         config.readConfig("/tmp/dssShutter_test.conf");
         sh.loadConfig(config);

         int rv = sh.appStartup();
         REQUIRE( rv == 0 );

         REQUIRE( sh.indiPowerChannel().getName() == "shutterPower" );
         REQUIRE( sh.indiSensorChannel().getName() == "shutterSensor" );
         REQUIRE( sh.indiTriggerChannel().getName() == "shutterTrigger" );

         REQUIRE( sh.openThreadJoinable() == true );
         REQUIRE( sh.shutThreadJoinable() == true );

         sh.forceShutdown(1);
         rv = sh.appShutdown();
         REQUIRE( rv == 0 );

         REQUIRE( sh.openThreadJoinable() == false );
         REQUIRE( sh.shutThreadJoinable() == false );
      }

      WHEN("powerChannel registration fails")
      {
         dssShutterTest sh;
         sh.setupConfig(config);
         config.readConfig("/tmp/dssShutter_test.conf");
         sh.loadConfig(config);
         sh.m_failPowerRegister = true;

         dssShutterTest::resetLogState();
         int rv = sh.appStartup();
         REQUIRE( rv == -1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );

         REQUIRE( sh.openThreadJoinable() == false );
         REQUIRE( sh.shutThreadJoinable() == false );
      }

      WHEN("sensorChannel registration fails")
      {
         dssShutterTest sh;
         sh.setupConfig(config);
         config.readConfig("/tmp/dssShutter_test.conf");
         sh.loadConfig(config);
         sh.m_failSensorRegister = true;

         dssShutterTest::resetLogState();
         int rv = sh.appStartup();
         REQUIRE( rv == -1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
      }

      WHEN("triggerChannel registration fails")
      {
         dssShutterTest sh;
         sh.setupConfig(config);
         config.readConfig("/tmp/dssShutter_test.conf");
         sh.loadConfig(config);
         sh.m_failTriggerRegister = true;

         dssShutterTest::resetLogState();
         int rv = sh.appStartup();
         REQUIRE( rv == -1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
      }

      WHEN("open thread fails to start")
      {
         dssShutterTest sh;
         sh.setupConfig(config);
         config.readConfig("/tmp/dssShutter_test.conf");
         sh.loadConfig(config);
         sh.m_failOpenThreadStart = true;

         dssShutterTest::resetLogState();
         int rv = sh.appStartup();
         REQUIRE( rv == -1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
         REQUIRE( sh.openThreadJoinable() == false );

         sh.forceShutdown(1);
         sh.appShutdown();
      }

      WHEN("shut thread fails to start")
      {
         dssShutterTest sh;
         sh.setupConfig(config);
         config.readConfig("/tmp/dssShutter_test.conf");
         sh.loadConfig(config);
         sh.m_failShutThreadStart = true;

         dssShutterTest::resetLogState();
         int rv = sh.appStartup();
         REQUIRE( rv == -1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );

         // The open thread did start for real in this case, so it must be stopped and joined.
         REQUIRE( sh.openThreadJoinable() == true );
         sh.forceShutdown(1);
         sh.appShutdown();
         REQUIRE( sh.openThreadJoinable() == false );
      }
   }
}

/// Verify that appLogic derives the shutter status and state from the power and sensor states.
/** Also verify that onPowerOff and whilePowerOff delegate to appLogic.
 *
 * \ingroup dssShutter_tests
 */
SCENARIO( "dssShutter appLogic", "[dev::dssShutter]" )
{
   GIVEN("a shutter instance")
   {
      dssShutterTest sh;

      WHEN("power state is unknown")
      {
         sh.setPowerState(-1);
         sh.setSensorState(-1);
         int rv = sh.appLogic();
         REQUIRE( rv == 0 );
         REQUIRE( sh.m_shutterStatus == "UNKNOWN" );
         REQUIRE( sh.m_shutterState == -1 );
      }

      WHEN("power is off")
      {
         sh.setPowerState(0);
         int rv = sh.appLogic();
         REQUIRE( rv == 0 );
         REQUIRE( sh.m_shutterStatus == "POWEROFF" );
         REQUIRE( sh.m_shutterState == -1 );
      }

      WHEN("power is on and sensor reads shut")
      {
         sh.setPowerState(1);
         sh.setSensorState(0);
         int rv = sh.appLogic();
         REQUIRE( rv == 0 );
         REQUIRE( sh.m_shutterStatus == "READY" );
         REQUIRE( sh.m_shutterState == 0 );
      }

      WHEN("power is on and sensor reads open")
      {
         sh.setPowerState(1);
         sh.setSensorState(1);
         int rv = sh.appLogic();
         REQUIRE( rv == 0 );
         REQUIRE( sh.m_shutterStatus == "READY" );
         REQUIRE( sh.m_shutterState == 1 );
      }

      WHEN("power is on and sensor state is unknown")
      {
         sh.setPowerState(1);
         sh.setSensorState(-1);
         sh.m_shutterState = 42; // Sentinel value. appLogic must leave it alone when the sensor state is unknown.
         int rv = sh.appLogic();
         REQUIRE( rv == 0 );
         REQUIRE( sh.m_shutterStatus == "READY" );
         REQUIRE( sh.m_shutterState == 42 );
      }

      WHEN("onPowerOff and whilePowerOff delegate to appLogic")
      {
         sh.setPowerState(0);
         REQUIRE( sh.onPowerOff() == 0 );
         REQUIRE( sh.m_shutterStatus == "POWEROFF" );

         sh.setPowerState(1);
         sh.setSensorState(1);
         REQUIRE( sh.whilePowerOff() == 0 );
         REQUIRE( sh.m_shutterStatus == "READY" );
         REQUIRE( sh.m_shutterState == 1 );
      }
   }
}

/// Verify that appShutdown returns 0 when appStartup was never called and no threads exist.
/**
 * \ingroup dssShutter_tests
 */
SCENARIO( "dssShutter appShutdown with no threads started", "[dev::dssShutter]" )
{
   GIVEN("a shutter instance that never called appStartup")
   {
      dssShutterTest sh;

      REQUIRE( sh.openThreadJoinable() == false );
      REQUIRE( sh.shutThreadJoinable() == false );

      int rv = sh.appShutdown();
      REQUIRE( rv == 0 );
   }
}

/// Verify that setShutterState rejects invalid requests and hands valid ones to the worker threads.
/** The trigger is made to stick so that open() and shut() always fail. The test polls the
 * request flags until the thread has processed the request and then checks the log count.
 *
 * \ingroup dssShutter_tests
 */
SCENARIO( "dssShutter setShutterState and background threads", "[dev::dssShutter]" )
{
   GIVEN("a running shutter with a stuck trigger (so open/shut always fail)")
   {
      writeShutterConfig("1", "5");

      mx::app::appConfigurator config;

      dssShutterTest sh;
      sh.setupConfig(config);
      config.readConfig("/tmp/dssShutter_test.conf");
      sh.loadConfig(config);

      sh.setPowerState(1);
      sh.setSensorState(0);
      sh.setTriggerState(0);
      sh.setTriggerResponds(false); // The simulated hardware never responds, so open() and shut() always fail.

      REQUIRE( sh.appStartup() == 0 );

      WHEN("an invalid shutter state is requested")
      {
         dssShutterTest::resetLogState();
         int rv = sh.setShutterState(2);
         REQUIRE( rv == -1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
         REQUIRE( dssShutterTest::s_lastLogMessage.find("invalid shutter request") != std::string::npos );
      }

      WHEN("open is requested and processed by the open thread")
      {
         dssShutterTest::resetLogState();

         int rv = sh.setShutterState(1);
         REQUIRE( rv == 0 );

         bool processed = false;
         for(int i = 0; i < 150 && !processed; ++i)
         {
            mx::sys::milliSleep(20);
            if( !sh.doOpen() ) processed = true;
         }
         REQUIRE( processed == true );

         // open() logs one software_error because the trigger is stuck.
         // openThreadExec logs a second software_error because open() returned less than zero.
         REQUIRE( dssShutterTest::s_logCount == 2 );
      }

      WHEN("shut is requested and processed by the shut thread")
      {
         sh.setSensorState(1); // The sensor must read open or shut() returns early as already shut.

         dssShutterTest::resetLogState();

         int rv = sh.setShutterState(0);
         REQUIRE( rv == 0 );

         bool processed = false;
         for(int i = 0; i < 150 && !processed; ++i)
         {
            mx::sys::milliSleep(20);
            if( !sh.doShut() ) processed = true;
         }
         REQUIRE( processed == true );

         // Same two log entries as the open case. One from shut() and one from shutThreadExec.
         REQUIRE( dssShutterTest::s_logCount == 2 );
      }

      sh.forceShutdown(1);
      sh.appShutdown();
   }
}

/// Verify every return path of open().
/** The cases are power off, already open, success on the first pulse, a trigger that never
 * responds, a sensor that never confirms, a trigger that sticks on the second pulse, and
 * success on the second pulse. The sendNewProperty hook changes the simulated hardware at
 * the chosen pulse. Small wait and timeout values keep the retries fast.
 *
 * \ingroup dssShutter_tests
 */
SCENARIO( "dssShutter open", "[dev::dssShutter]" )
{
   GIVEN("a shutter instance")
   {
      dssShutterTest sh;
      sh.setShutterWait(1);
      sh.setShutterTimeout(2);

      WHEN("power is not on")
      {
         sh.setPowerState(0);
         REQUIRE( sh.open() == 0 );
         REQUIRE( sh.m_recordCameraCalls == 0 );
      }

      WHEN("the shutter already reads open")
      {
         sh.setPowerState(1);
         sh.setSensorState(1);
         REQUIRE( sh.open() == 0 );
         REQUIRE( sh.m_recordCameraCalls == 1 ); // open() calls recordCamera once before it checks the sensor.
      }

      WHEN("the first attempt succeeds")
      {
         sh.setPowerState(1);
         sh.setSensorState(0);
         sh.setTriggerState(0);

         dssShutterTest::resetLogState();
         // The sensor confirms open right after the first trigger pulse.
         sh.m_sendNewPropertyHook = [&sh](int callNum)
         {
            if(callNum == 1) sh.setSensorState(1);
         };

         REQUIRE( sh.open() == 0 );
         REQUIRE( sh.m_recordCameraCalls == 2 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
         REQUIRE( dssShutterTest::s_lastLogMessage == "shutter open" );
      }

      WHEN("the trigger never responds")
      {
         sh.setPowerState(1);
         sh.setSensorState(0);
         sh.setTriggerState(0);
         sh.setTriggerResponds(false);

         dssShutterTest::resetLogState();

         REQUIRE( sh.open() == -1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
         REQUIRE( dssShutterTest::s_lastLogMessage.find("shutter trigger did not change state") != std::string::npos );
      }

      WHEN("the trigger responds but the sensor never confirms, failing entirely")
      {
         sh.setPowerState(1);
         sh.setSensorState(0);
         sh.setTriggerState(0);
         sh.setTriggerResponds(true);

         dssShutterTest::resetLogState();

         REQUIRE( sh.open() == -1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
         REQUIRE( dssShutterTest::s_lastLogMessage.find("shutter failed to open") != std::string::npos );
      }

      WHEN("the trigger sticks after the first attempt, so the second try reports it never changed")
      {
         sh.setPowerState(1);
         sh.setSensorState(0);
         sh.setTriggerState(0);
         sh.setTriggerResponds(true);

         // The first pulse moves the trigger. The trigger then sticks for the second pulse.
         sh.m_sendNewPropertyHook = [&sh](int callNum)
         {
            if(callNum == 1) sh.setTriggerResponds(false);
         };

         dssShutterTest::resetLogState();

         REQUIRE( sh.open() == -1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
         REQUIRE( dssShutterTest::s_lastLogMessage.find("shutter trigger did not change state") != std::string::npos );
      }

      WHEN("the second attempt succeeds")
      {
         sh.setPowerState(1);
         sh.setSensorState(0);
         sh.setTriggerState(0);
         sh.setTriggerResponds(true);

         // The sensor confirms open only after the second trigger pulse.
         sh.m_sendNewPropertyHook = [&sh](int callNum)
         {
            if(callNum == 2) sh.setSensorState(1);
         };

         dssShutterTest::resetLogState();

         REQUIRE( sh.open() == 0 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
         REQUIRE( dssShutterTest::s_lastLogMessage == "shutter open" );
      }
   }
}

/// Verify every return path of shut().
/** The cases mirror the open() scenario with the sensor starting in the open state.
 *
 * \ingroup dssShutter_tests
 */
SCENARIO( "dssShutter shut", "[dev::dssShutter]" )
{
   GIVEN("a shutter instance")
   {
      dssShutterTest sh;
      sh.setShutterWait(1);
      sh.setShutterTimeout(2);

      WHEN("power is not on")
      {
         sh.setPowerState(0);
         REQUIRE( sh.shut() == 0 );
      }

      WHEN("the shutter already reads shut")
      {
         sh.setPowerState(1);
         sh.setSensorState(0);
         REQUIRE( sh.shut() == 0 );
      }

      WHEN("the first attempt succeeds")
      {
         sh.setPowerState(1);
         sh.setSensorState(1);
         sh.setTriggerState(0);

         dssShutterTest::resetLogState();
         // The sensor confirms shut right after the first trigger pulse.
         sh.m_sendNewPropertyHook = [&sh](int callNum)
         {
            if(callNum == 1) sh.setSensorState(0);
         };

         REQUIRE( sh.shut() == 0 );
         REQUIRE( sh.m_recordCameraCalls == 1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
         REQUIRE( dssShutterTest::s_lastLogMessage == "shutter shut" );
      }

      WHEN("the trigger never responds")
      {
         sh.setPowerState(1);
         sh.setSensorState(1);
         sh.setTriggerState(0);
         sh.setTriggerResponds(false);

         dssShutterTest::resetLogState();

         REQUIRE( sh.shut() == -1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
         REQUIRE( dssShutterTest::s_lastLogMessage.find("shutter trigger did not change state") != std::string::npos );
      }

      WHEN("the trigger responds but the sensor never confirms, failing entirely")
      {
         sh.setPowerState(1);
         sh.setSensorState(1);
         sh.setTriggerState(0);
         sh.setTriggerResponds(true);

         dssShutterTest::resetLogState();

         REQUIRE( sh.shut() == -1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
         REQUIRE( dssShutterTest::s_lastLogMessage.find("shutter failed to shut") != std::string::npos );
      }

      WHEN("the trigger sticks after the first attempt, so the second try reports it never changed")
      {
         sh.setPowerState(1);
         sh.setSensorState(1);
         sh.setTriggerState(0);
         sh.setTriggerResponds(true);

         // The first pulse moves the trigger. The trigger then sticks for the second pulse.
         sh.m_sendNewPropertyHook = [&sh](int callNum)
         {
            if(callNum == 1) sh.setTriggerResponds(false);
         };

         dssShutterTest::resetLogState();

         REQUIRE( sh.shut() == -1 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
         REQUIRE( dssShutterTest::s_lastLogMessage.find("shutter trigger did not change state") != std::string::npos );
      }

      WHEN("the second attempt succeeds")
      {
         sh.setPowerState(1);
         sh.setSensorState(1);
         sh.setTriggerState(0);
         sh.setTriggerResponds(true);

         // The sensor confirms shut only after the second trigger pulse.
         sh.m_sendNewPropertyHook = [&sh](int callNum)
         {
            if(callNum == 2) sh.setSensorState(0);
         };

         dssShutterTest::resetLogState();

         REQUIRE( sh.shut() == 0 );
         REQUIRE( dssShutterTest::s_logCount == 1 );
         REQUIRE( dssShutterTest::s_lastLogMessage == "shutter shut" );
      }
   }
}

/// Verify the INDI callbacks for the power, sensor, and trigger channels.
/** Each callback must ignore a property with the wrong device or name, ignore a property
 * with no value element, map valid values to the matching state, and map any other value
 * to unknown. The static st_ wrappers must forward to the member callbacks.
 *
 * \ingroup dssShutter_tests
 */
SCENARIO( "dssShutter INDI callbacks", "[dev::dssShutter]" )
{
   GIVEN("a started shutter with named INDI properties")
   {
      writeShutterConfig("1", "5");

      mx::app::appConfigurator config;

      dssShutterTest sh;
      sh.setupConfig(config);
      config.readConfig("/tmp/dssShutter_test.conf");
      sh.loadConfig(config);
      REQUIRE( sh.appStartup() == 0 );

      WHEN("the power channel callback is exercised")
      {
         pcf::IndiProperty mismatched;
         mismatched.setDevice("other");
         mismatched.setName("other");
         REQUIRE( sh.setCallBack_powerChannel(mismatched) == 0 );
         REQUIRE( sh.powerState() == -1 );

         pcf::IndiProperty ip;
         ip.setDevice("pdu");
         ip.setName("shutterPower");
         REQUIRE( sh.setCallBack_powerChannel(ip) == 0 ); // There is no "state" element yet, so the state stays unknown.
         REQUIRE( sh.powerState() == -1 );

         ip.add( pcf::IndiElement("state", std::string("On")) );
         REQUIRE( sh.setCallBack_powerChannel(ip) == 0 );
         REQUIRE( sh.powerState() == 1 );

         ip["state"].setValue( std::string("Off") );
         REQUIRE( sh.setCallBack_powerChannel(ip) == 0 );
         REQUIRE( sh.powerState() == 0 );

         ip["state"].setValue( std::string("Weird") );
         REQUIRE( sh.setCallBack_powerChannel(ip) == 0 );
         REQUIRE( sh.powerState() == -1 );

         ip["state"].setValue( std::string("On") );
         REQUIRE( dev::dssShutter<dssShutterTest>::st_setCallBack_powerChannel(&sh, ip) == 0 );
         REQUIRE( sh.powerState() == 1 );
      }

      WHEN("the sensor channel callback is exercised")
      {
         pcf::IndiProperty mismatched;
         mismatched.setDevice("dio");
         mismatched.setName("other");
         REQUIRE( sh.setCallBack_sensorChannel(mismatched) == 0 );
         REQUIRE( sh.sensorState() == -1 );

         pcf::IndiProperty ip;
         ip.setDevice("dio");
         ip.setName("shutterSensor");
         REQUIRE( sh.setCallBack_sensorChannel(ip) == 0 ); // There is no "current" element yet, so the state stays unknown.
         REQUIRE( sh.sensorState() == -1 );

         ip.add( pcf::IndiElement("current", (int) 1) );
         REQUIRE( sh.setCallBack_sensorChannel(ip) == 0 );
         REQUIRE( sh.sensorState() == 1 );

         ip["current"].setValue( (int) 0 );
         REQUIRE( sh.setCallBack_sensorChannel(ip) == 0 );
         REQUIRE( sh.sensorState() == 0 );

         ip["current"].setValue( (int) 7 );
         REQUIRE( sh.setCallBack_sensorChannel(ip) == 0 );
         REQUIRE( sh.sensorState() == -1 );

         ip["current"].setValue( (int) 1 );
         REQUIRE( dev::dssShutter<dssShutterTest>::st_setCallBack_sensorChannel(&sh, ip) == 0 );
         REQUIRE( sh.sensorState() == 1 );
      }

      WHEN("the trigger channel callback is exercised")
      {
         pcf::IndiProperty mismatched;
         mismatched.setDevice("dio");
         mismatched.setName("other");
         REQUIRE( sh.setCallBack_triggerChannel(mismatched) == 0 );
         REQUIRE( sh.triggerState() == -1 );

         pcf::IndiProperty ip;
         ip.setDevice("dio");
         ip.setName("shutterTrigger");
         REQUIRE( sh.setCallBack_triggerChannel(ip) == 0 ); // There is no "current" element yet, so the state stays unknown.
         REQUIRE( sh.triggerState() == -1 );

         ip.add( pcf::IndiElement("current", (int) 1) );
         REQUIRE( sh.setCallBack_triggerChannel(ip) == 0 );
         REQUIRE( sh.triggerState() == 1 );

         ip["current"].setValue( (int) 0 );
         REQUIRE( sh.setCallBack_triggerChannel(ip) == 0 );
         REQUIRE( sh.triggerState() == 0 );

         ip["current"].setValue( (int) 7 );
         REQUIRE( sh.setCallBack_triggerChannel(ip) == 0 );
         REQUIRE( sh.triggerState() == -1 );

         ip["current"].setValue( (int) 1 );
         REQUIRE( dev::dssShutter<dssShutterTest>::st_setCallBack_triggerChannel(&sh, ip) == 0 );
         REQUIRE( sh.triggerState() == 1 );
      }

      sh.forceShutdown(1);
      sh.appShutdown();
   }
}

// This block is never compiled. It exists so Doxygen links this test file to the functions it covers.
#ifdef DSSSHUTTER_TEST_DOXYGEN_REF
MagAOX::app::dev::dssShutter<dssShutter_tests::dssShutterTest>::setupConfig(*(mx::app::appConfigurator*)0);
MagAOX::app::dev::dssShutter<dssShutter_tests::dssShutterTest>::loadConfig(*(mx::app::appConfigurator*)0);
MagAOX::app::dev::dssShutter<dssShutter_tests::dssShutterTest>::appStartup();
MagAOX::app::dev::dssShutter<dssShutter_tests::dssShutterTest>::appLogic();
MagAOX::app::dev::dssShutter<dssShutter_tests::dssShutterTest>::appShutdown();
MagAOX::app::dev::dssShutter<dssShutter_tests::dssShutterTest>::onPowerOff();
MagAOX::app::dev::dssShutter<dssShutter_tests::dssShutterTest>::whilePowerOff();
MagAOX::app::dev::dssShutter<dssShutter_tests::dssShutterTest>::setShutterState(0);
MagAOX::app::dev::dssShutter<dssShutter_tests::dssShutterTest>::open();
MagAOX::app::dev::dssShutter<dssShutter_tests::dssShutterTest>::shut();
#endif

} // namespace dssShutter_tests
