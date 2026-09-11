/** \file outletController_test.hpp
 * \brief Test harness for the MagAOX::app::dev::outletController device mixin.
 *
 * One harness class, outletControllerTest, drives the real outletController code against
 * four simulated outlets. The outlet hooks record the state and the time of each switch in
 * memory. The harness can make one outlet fail through m_failOutlet. The common parts of
 * every dev:: harness, such as the FIFO-less indiDriver and the registration fault
 * injection through m_regFailAt, come from appHarnessBase in testHarnessCommon.hpp.
 *
 * \ingroup testing
 */

#include "../../../../tests/catch2/catch.hpp"

#include <mx/sys/timeUtils.hpp>

// OUTLET_CTRL_TEST_NOINDI is deliberately not defined here. Defining it would compile out the
// indi::updateIfChanged() calls in outletController. Those calls are safe without a driver
// because updateIfChanged() in indiUtils returns immediately when the driver pointer is null.
// Leaving them in lets these tests run the real code paths.
//#define OUTLET_CTRL_TEST_NOLOG
#include "../../MagAOXApp.hpp"
#include "../outletController.hpp"
#include "testHarnessCommon.hpp"

// LCOV_EXCL_START

namespace outletController_tests
{

/// Test harness for dev::outletController.
/** Provides the four simulated outlets and the outlet hooks that outletController requires
  * of its derived class. Each hook records the new state and a timestamp so tests can check
  * both the result and the order of switching. One outlet can be made to fail through
  * m_failOutlet. The n-th INDI registration can be made to fail through m_regFailAt, which
  * comes from appHarnessBase.
  *
  * \ingroup outletController_tests
  */
struct outletControllerTest : public MagAOX::app::dev::testHarness::appHarnessBase,
                              public MagAOX::app::dev::outletController<outletControllerTest>
{
   typedef MagAOX::app::dev::testHarness::appHarnessBase baseT;
   typedef MagAOX::app::dev::outletController<outletControllerTest> outletControllerT;

   std::vector<double> m_timestamps;

   /// Outlet number for which turnOutletOn() and turnOutletOff() return -1.
   /// This simulates a hardware failure on one outlet so the error-return paths
   /// in turnChannelOn() and turnChannelOff() can be tested. The default of -1 means never fail.
   int m_failOutlet {-1};

   outletControllerTest() : baseT( "outletControllerTest" )
   {
      setNumberOfOutlets(4);
      m_timestamps.resize(4,0);
      turnOutletOff(0);
      turnOutletOff(1);
      turnOutletOff(2);
      turnOutletOff(3);
   }

   ~outletControllerTest() noexcept
   {}

   int setupConfig( mx::app::appConfigurator & config)
   {
      return outletControllerT::setupConfig(config);
   }

   int loadConfig( mx::app::appConfigurator & config)
   {
      return outletControllerT::loadConfig(config);
   }

   int appStartup()
   {
      return outletControllerT::appStartup();
   }

   int appLogic()
   {
      return 0;
   }

   int appShutdown()
   {
      return 0;
   }

   int updateOutletState( int outletNum )
   {
      return m_outletStates[outletNum];
   }

   int turnOutletOn( int outletNum )
   {
      if(m_failOutlet == outletNum) return -1;

      m_outletStates[outletNum] = 2;
      mx::sys::nanoSleep(1);
      m_timestamps[outletNum] = mx::sys::get_curr_time();

      return 0;
   }

   int turnOutletOff( int outletNum )
   {
      if(m_failOutlet == outletNum) return -1;

      m_outletStates[outletNum] = 0;
      mx::sys::nanoSleep(1);
      m_timestamps[outletNum] = mx::sys::get_curr_time();

      return 0;
   }

};

} // namespace outletController_tests

// LCOV_EXCL_STOP
