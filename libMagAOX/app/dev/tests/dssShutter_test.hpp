/** \file dssShutter_test.hpp
 * \brief Test harness for the MagAOX::app::dev::dssShutter device mixin.
 *
 * One harness class, dssShutterTest, drives the real dssShutter code against a simulated
 * digital I/O shutter. The harness hides registerIndiPropertySet, threadStart, sendNewProperty,
 * and recordCamera. This lets each test force a failure path and stand in for the hardware that
 * drives and senses the shutter. The open and shut background threads are started for real.
 * Log calls are captured in static members and checked by count and by message text.
 *
 * The common parts of every dev:: harness, such as the FIFO-less indiDriver and the registration
 * fault injection, come from appHarnessBase in testHarnessCommon.hpp.
 *
 * \ingroup testing
 */

#include "../../../../tests/catch2/catch.hpp"

#include <functional>

#include "../../MagAOXApp.hpp"
#include "../dssShutter.hpp"
#include "testHarnessCommon.hpp"

// LCOV_EXCL_START

namespace dssShutter_tests
{

/// Test harness for dev::dssShutter.
/** Provides the members and methods that dssShutter requires of its derived class.
  * It also provides hooks that let a test force specific error paths. Examples are
  * INDI property registration failures and thread-start failures. The trigger and
  * sensor state members stand in for the digital I/O hardware of a real shutter.
  *
  * \ingroup dssShutter_tests
  */
struct dssShutterTest : public MagAOX::app::dev::testHarness::appHarnessBase,
                        public MagAOX::app::dev::dssShutter<dssShutterTest>
{
   typedef MagAOX::app::dev::testHarness::appHarnessBase baseT;
   typedef MagAOX::app::dev::dssShutter<dssShutterTest> dssShutterT;

   friend dssShutterT;

   // Members that dssShutter reads and writes through derived().
   std::string m_shutterStatus;
   int m_shutterState {-2};

   // Log capture. The log() below mirrors the MagAOXApp::log signature so it hides it.
   static std::string s_lastLogMessage;
   static flatlogs::logPrioT s_lastLogLevel;
   static int s_logCount;

   // Flags that make registerIndiPropertySet fail for one property.
   bool m_failPowerRegister {false};
   bool m_failSensorRegister {false};
   bool m_failTriggerRegister {false};

   // Flags that make threadStart fail for one thread.
   bool m_failOpenThreadStart {false};
   bool m_failShutThreadStart {false};

   // Record of recordCamera calls.
   int m_recordCameraCalls {0};
   bool m_lastRecordCameraForce {false};

   // Simulated hardware behind sendNewProperty.
   int m_sendNewPropertyCalls {0};
   bool m_triggerResponds {true}; ///< If true, the simulated trigger channel follows the commanded value.
   std::function<void(int)> m_sendNewPropertyHook; ///< Called with the 1-based call number after each sendNewProperty.

   dssShutterTest() : baseT( "dssShutterTest" )
   {
      resetLogState();
   }

   ~dssShutterTest() noexcept
   {}

   static void resetLogState()
   {
      s_lastLogMessage.clear();
      s_lastLogLevel = flatlogs::logPrio::LOG_DEFAULT;
      s_logCount = 0;
   }

   /// Capture dssShutter log messages instead of sending them to the normal logger.
   template <typename logT, int retval = 0>
   static int log( const typename logT::messageT &msg, flatlogs::logPrioT level = flatlogs::logPrio::LOG_DEFAULT )
   {
      s_lastLogMessage = logT::msgString( const_cast<uint8_t *>( msg.builder.GetBufferPointer() ), msg.builder.GetSize() );
      s_lastLogLevel = level;
      ++s_logCount;
      return retval;
   }

   int setupConfig( mx::app::appConfigurator & config )
   {
      dssShutterT::setupConfig(config);
      return 0;
   }

   int loadConfig( mx::app::appConfigurator & config )
   {
      dssShutterT::loadConfig(config);
      return 0;
   }

   int appStartup()
   {
      return dssShutterT::appStartup();
   }

   int appLogic()
   {
      return dssShutterT::appLogic();
   }

   int appShutdown()
   {
      return dssShutterT::appShutdown();
   }

   int onPowerOff()
   {
      return dssShutterT::onPowerOff();
   }

   int whilePowerOff()
   {
      return dssShutterT::whilePowerOff();
   }

   /// Hides the base registerIndiPropertySet so appStartup's registration calls can be forced to fail.
   /** Unlike the base version this never registers anything for real. It only names the property
     * and returns -1 when the matching failure flag is set.
     */
   int registerIndiPropertySet( pcf::IndiProperty &prop,
                                 const std::string &devName,
                                 const std::string &propName,
                                 int (*callBack)( void *, const pcf::IndiProperty & )
                               )
   {
      static_cast<void>(callBack);

      prop = pcf::IndiProperty();
      prop.setDevice(devName);
      prop.setName(propName);

      if( &prop == &(this->dssShutterT::m_indiP_powerChannel) && m_failPowerRegister ) return -1;
      if( &prop == &m_indiP_sensorChannel && m_failSensorRegister ) return -1;
      if( &prop == &m_indiP_triggerChannel && m_failTriggerRegister ) return -1;

      return 0;
   }

   /// Hides MagAOXApp<false>::threadStart so the open/shut thread starts can be forced to fail.
   template<class thisPtr, class Function>
   int threadStart( std::thread & thrd,
                     bool & thrdInit,
                     pid_t & tpid,
                     pcf::IndiProperty & thProp,
                     int thrdPrio,
                     const std::string & cpuset,
                     const std::string & thrdName,
                     thisPtr * thrdThis,
                     Function && thrdStart
                   )
   {
      if( thrdName == "open" && m_failOpenThreadStart ) return -1;
      if( thrdName == "shut" && m_failShutThreadStart ) return -1;

      return baseT::threadStart( thrd, thrdInit, tpid, thProp, thrdPrio, cpuset, thrdName, thrdThis, std::forward<Function>(thrdStart) );
   }

   /// Required by dssShutter via derived(). Counts calls instead of recording the camera.
   int recordCamera( bool force = false )
   {
      ++m_recordCameraCalls;
      m_lastRecordCameraForce = force;
      return 0;
   }

   /// Required by dssShutter via derived(). Simulates the digital I/O hardware.
   /** When m_triggerResponds is true the trigger state follows the commanded value.
     * The optional hook runs after each call so a test can change the sensor or
     * trigger behavior at a chosen point in the open or shut sequence.
     */
   template<typename T>
   int sendNewProperty( const pcf::IndiProperty & ipSend, const std::string & el, const T & newVal )
   {
      static_cast<void>(ipSend);
      static_cast<void>(el);

      ++m_sendNewPropertyCalls;

      if( m_triggerResponds )
      {
         m_triggerState = static_cast<int>(newVal);
      }

      if( m_sendNewPropertyHook )
      {
         m_sendNewPropertyHook( m_sendNewPropertyCalls );
      }

      return 0;
   }

   // Public wrappers around protected dssShutter and MagAOXApp state so tests can read and set it.

   void forceShutdown( int v )
   {
      m_shutdown = v;
   }

   void setPowerState( int v ) { dssShutterT::m_powerState = v; }
   int powerState() const { return dssShutterT::m_powerState; }

   void setSensorState( int v ) { m_sensorState = v; }
   int sensorState() const { return m_sensorState; }

   void setTriggerState( int v ) { m_triggerState = v; }
   int triggerState() const { return m_triggerState; }

   void setTriggerResponds( bool v ) { m_triggerResponds = v; }

   void setShutterWait( unsigned v ) { m_shutterWait = v; }
   void setShutterTimeout( unsigned v ) { m_shutterTimeout = v; }

   bool doOpen() const { return m_doOpen; }
   bool doShut() const { return m_doShut; }

   std::string powerDeviceVal() const { return dssShutterT::m_powerDevice; }
   std::string powerChannelVal() const { return dssShutterT::m_powerChannel; }
   std::string dioDeviceVal() const { return m_dioDevice; }
   std::string sensorChannelVal() const { return m_sensorChannel; }
   std::string triggerChannelVal() const { return m_triggerChannel; }
   unsigned shutterWaitVal() const { return m_shutterWait; }
   unsigned shutterTimeoutVal() const { return m_shutterTimeout; }

   pcf::IndiProperty & indiPowerChannel() { return dssShutterT::m_indiP_powerChannel; }
   pcf::IndiProperty & indiSensorChannel() { return m_indiP_sensorChannel; }
   pcf::IndiProperty & indiTriggerChannel() { return m_indiP_triggerChannel; }

   bool openThreadJoinable() { return m_openThread.joinable(); }
   bool shutThreadJoinable() { return m_shutThread.joinable(); }
};

std::string dssShutterTest::s_lastLogMessage;
flatlogs::logPrioT dssShutterTest::s_lastLogLevel = flatlogs::logPrio::LOG_DEFAULT;
int dssShutterTest::s_logCount = 0;

} // namespace dssShutter_tests

// LCOV_EXCL_STOP
