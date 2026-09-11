//#define CATCH_CONFIG_MAIN
/** \file outletController_test.cpp
  * \brief Catch2 tests for the MagAOX::app::dev::outletController device mixin.
  *
  * The component under test is the CRTP mixin in libMagAOX/app/dev/outletController.hpp.
  * It maps named channels onto numbered power outlets and sequences them on and off.
  *
  * The tests use the outletControllerTest harness declared in outletController_test.hpp. It
  * derives from appHarnessBase with four simulated outlets. The outlet hooks record the state
  * and the time of each switch in memory. The harness can make one outlet fail and can make the
  * n-th INDI property registration fail. Config files are written under /tmp and read back
  * through a real appConfigurator. INDI callbacks are driven with hand-built
  * pcf::IndiProperty objects. No INDI server is needed. One scenario installs a FIFO-less
  * INDI driver so updateINDI() runs past its null-driver guard.
  *
  * \ingroup outletController_tests
  */
#include "../../../../tests/catch2/catch.hpp"

#include "outletController_test.hpp"

using namespace MagAOX::app;

/** \defgroup outletController_tests libXWC::app::dev::outletController Unit Tests
 * \ingroup app_dev_unit_tests
*/
namespace outletController_tests
{

/// outletController Configuration
/**
 * \ingroup outletController_tests
 */
SCENARIO( "outletController Configuration", "[outletController]" )
{
   GIVEN("a config file with 4 channels for 4 outlets")
   {
      WHEN("using outlet keyword, only outlet specified")
      {
         mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1", "channel2",     "channel3",      "channel4"},
                                                        {"outlet",   "outlet",       "outlet",          "outlet"},
                                                        {"0",         "1",             "2",                "3"} );

         mx::app::appConfigurator config;
         config.readConfig("/tmp/outletController_test.conf");

         outletControllerTest pdt;
         int rv;
         rv = pdt.setupConfig(config);
         REQUIRE( rv == 0);

         rv = pdt.loadConfig(config);
         REQUIRE( rv == 0);
         REQUIRE( pdt.numChannels() == 4);

         std::vector<size_t> outlets, onOrder, offOrder;
         std::vector<unsigned> onDelays, offDelays;
         outlets = pdt.channelOutlets("channel1");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 0 );

         onOrder = pdt.channelOnOrder("channel1");
         REQUIRE( onOrder.size() == 0);
         offOrder = pdt.channelOffOrder("channel1");
         REQUIRE( offOrder.size() == 0);
         onDelays = pdt.channelOnDelays("channel1");
         REQUIRE( onDelays.size() == 0);
         offDelays = pdt.channelOffDelays("channel1");
         REQUIRE( offDelays.size() == 0);

         outlets = pdt.channelOutlets("channel2");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 1 );

         onOrder = pdt.channelOnOrder("channel2");
         REQUIRE( onOrder.size() == 0);
         offOrder = pdt.channelOffOrder("channel2");
         REQUIRE( offOrder.size() == 0);
         onDelays = pdt.channelOnDelays("channel2");
         REQUIRE( onDelays.size() == 0);
         offDelays = pdt.channelOffDelays("channel2");
         REQUIRE( offDelays.size() == 0);

         outlets = pdt.channelOutlets("channel3");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 2 );

         onOrder = pdt.channelOnOrder("channel3");
         REQUIRE( onOrder.size() == 0);
         offOrder = pdt.channelOffOrder("channel3");
         REQUIRE( offOrder.size() == 0);
         onDelays = pdt.channelOnDelays("channel3");
         REQUIRE( onDelays.size() == 0);
         offDelays = pdt.channelOffDelays("channel3");
         REQUIRE( offDelays.size() == 0);

         outlets = pdt.channelOutlets("channel4");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 3 );

         onOrder = pdt.channelOnOrder("channel4");
         REQUIRE( onOrder.size() == 0);
         offOrder = pdt.channelOffOrder("channel4");
         REQUIRE( offOrder.size() == 0);
         onDelays = pdt.channelOnDelays("channel4");
         REQUIRE( onDelays.size() == 0);
         offDelays = pdt.channelOffDelays("channel4");
         REQUIRE( offDelays.size() == 0);

      }

      WHEN("using outlet keyword, all specified")
      {
         mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1", "channel1", "channel1", "channel1", "channel1",  "channel2",  "channel2", "channel2", "channel2", "channel2",  "channel3", "channel3", "channel3", "channel3", "channel3",   "channel4",  "channel4", "channel4", "channel4", "channel4"  },
                                                        {"outlet",   "onOrder",  "offOrder", "onDelays", "offDelays", "outlet",    "onOrder",  "offOrder", "onDelays", "offDelays", "outlet",   "onOrder",  "offOrder", "onDelays", "offDelays",  "outlet",   "onOrder",  "offOrder", "onDelays", "offDelays"  },
                                                        {"0",        "0",        "0",        "100",      "120",       "1",         "0",        "0",        "105",      "130",       "2",        "0",        "0",        "107",      "132",        "3",      "0",        "0",        "108",      "133"});

         mx::app::appConfigurator config;
         config.readConfig("/tmp/outletController_test.conf");

         outletControllerTest pdt;
         int rv;
         rv = pdt.setupConfig(config);
         REQUIRE( rv == 0);

         rv = pdt.loadConfig(config);
         REQUIRE( rv == 0);
         REQUIRE( pdt.numChannels() == 4);

         std::vector<size_t> outlets, onOrder, offOrder;
         std::vector<unsigned> onDelays, offDelays;

         outlets = pdt.channelOutlets("channel1");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 0 );

         onOrder = pdt.channelOnOrder("channel1");
         REQUIRE( onOrder.size() == 1);
         REQUIRE( onOrder[0] == 0);
         offOrder = pdt.channelOffOrder("channel1");
         REQUIRE( offOrder.size() == 1);
         REQUIRE( offOrder[0] == 0);
         onDelays = pdt.channelOnDelays("channel1");
         REQUIRE( onDelays.size() == 1);
         REQUIRE( onDelays[0] == 100);
         offDelays = pdt.channelOffDelays("channel1");
         REQUIRE( offDelays.size() == 1);
         REQUIRE( offDelays[0] == 120);

         outlets = pdt.channelOutlets("channel2");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 1 );

         onOrder = pdt.channelOnOrder("channel2");
         REQUIRE( onOrder.size() == 1);
         REQUIRE( onOrder[0] == 0);
         offOrder = pdt.channelOffOrder("channel2");
         REQUIRE( offOrder.size() == 1);
         REQUIRE( offOrder[0] == 0);
         onDelays = pdt.channelOnDelays("channel2");
         REQUIRE( onDelays.size() == 1);
         REQUIRE( onDelays[0] == 105);
         offDelays = pdt.channelOffDelays("channel2");
         REQUIRE( offDelays.size() == 1);
         REQUIRE( offDelays[0] == 130);

         outlets = pdt.channelOutlets("channel3");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 2 );

         onOrder = pdt.channelOnOrder("channel3");
         REQUIRE( onOrder.size() == 1);
         REQUIRE( onOrder[0] == 0);
         offOrder = pdt.channelOffOrder("channel3");
         REQUIRE( offOrder.size() == 1);
         REQUIRE( offOrder[0] == 0);
         onDelays = pdt.channelOnDelays("channel3");
         REQUIRE( onDelays.size() == 1);
         REQUIRE( onDelays[0] == 107);
         offDelays = pdt.channelOffDelays("channel3");
         REQUIRE( offDelays.size() == 1);
         REQUIRE( offDelays[0] == 132);

         outlets = pdt.channelOutlets("channel4");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 3 );

         onOrder = pdt.channelOnOrder("channel4");
         REQUIRE( onOrder.size() == 1);
         REQUIRE( onOrder[0] == 0);
         offOrder = pdt.channelOffOrder("channel4");
         REQUIRE( offOrder.size() == 1);
         REQUIRE( offOrder[0] == 0);
         onDelays = pdt.channelOnDelays("channel4");
         REQUIRE( onDelays.size() == 1);
         REQUIRE( onDelays[0] == 108);
         offDelays = pdt.channelOffDelays("channel4");
         REQUIRE( offDelays.size() == 1);
         REQUIRE( offDelays[0] == 133);
      }

      WHEN("using outlets keyword, only outlet specified")
      {
         mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1",     "channel2",     "channel3",      "channel4"},
                                                        {"outlets",       "outlets",       "outlets",          "outlets"},
                                                        {"0",             "1",             "2",                "3"} );

         mx::app::appConfigurator config;
         config.readConfig("/tmp/outletController_test.conf");

         outletControllerTest pdt;
         int rv;
         rv = pdt.setupConfig(config);
         REQUIRE( rv == 0);

         rv = pdt.loadConfig(config);
         REQUIRE( rv == 0);
         REQUIRE( pdt.numChannels() == 4);

         std::vector<size_t> outlets, onOrder, offOrder;
         std::vector<unsigned> onDelays, offDelays;
         outlets = pdt.channelOutlets("channel1");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 0 );

         onOrder = pdt.channelOnOrder("channel1");
         REQUIRE( onOrder.size() == 0);
         offOrder = pdt.channelOffOrder("channel1");
         REQUIRE( offOrder.size() == 0);
         onDelays = pdt.channelOnDelays("channel1");
         REQUIRE( onDelays.size() == 0);
         offDelays = pdt.channelOffDelays("channel1");
         REQUIRE( offDelays.size() == 0);

         outlets = pdt.channelOutlets("channel2");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 1 );

         onOrder = pdt.channelOnOrder("channel2");
         REQUIRE( onOrder.size() == 0);
         offOrder = pdt.channelOffOrder("channel2");
         REQUIRE( offOrder.size() == 0);
         onDelays = pdt.channelOnDelays("channel2");
         REQUIRE( onDelays.size() == 0);
         offDelays = pdt.channelOffDelays("channel2");
         REQUIRE( offDelays.size() == 0);

         outlets = pdt.channelOutlets("channel3");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 2 );

         onOrder = pdt.channelOnOrder("channel3");
         REQUIRE( onOrder.size() == 0);
         offOrder = pdt.channelOffOrder("channel3");
         REQUIRE( offOrder.size() == 0);
         onDelays = pdt.channelOnDelays("channel3");
         REQUIRE( onDelays.size() == 0);
         offDelays = pdt.channelOffDelays("channel3");
         REQUIRE( offDelays.size() == 0);

         outlets = pdt.channelOutlets("channel4");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 3 );

         onOrder = pdt.channelOnOrder("channel4");
         REQUIRE( onOrder.size() == 0);
         offOrder = pdt.channelOffOrder("channel4");
         REQUIRE( offOrder.size() == 0);
         onDelays = pdt.channelOnDelays("channel4");
         REQUIRE( onDelays.size() == 0);
         offDelays = pdt.channelOffDelays("channel4");
         REQUIRE( offDelays.size() == 0);
      }

      WHEN("using outlets keyword, all specified")
      {
         mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1", "channel1", "channel1", "channel1", "channel1",  "channel2", "channel2", "channel2", "channel2", "channel2",  "channel3", "channel3", "channel3", "channel3", "channel3",   "channel4",  "channel4", "channel4", "channel4", "channel4"  },
                                                        {"outlets",  "onOrder",  "offOrder", "onDelays", "offDelays", "outlets",  "onOrder",  "offOrder", "onDelays", "offDelays", "outlets",  "onOrder",  "offOrder", "onDelays", "offDelays",  "outlets",   "onOrder",  "offOrder", "onDelays", "offDelays"  },
                                                        {"0",        "0",        "0",        "100",      "120",       "1",        "0",        "0",        "105",      "130",       "2",        "0",        "0",        "107",      "132",        "3",          "0",        "0",        "108",      "133"});

         mx::app::appConfigurator config;
         config.readConfig("/tmp/outletController_test.conf");

         outletControllerTest pdt;
         int rv;
         rv = pdt.setupConfig(config);
         REQUIRE( rv == 0);

         rv = pdt.loadConfig(config);
         REQUIRE( rv == 0);
         REQUIRE( pdt.numChannels() == 4);

         std::vector<size_t> outlets, onOrder, offOrder;
         std::vector<unsigned> onDelays, offDelays;

         outlets = pdt.channelOutlets("channel1");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 0 );

         onOrder = pdt.channelOnOrder("channel1");
         REQUIRE( onOrder.size() == 1);
         REQUIRE( onOrder[0] == 0);
         offOrder = pdt.channelOffOrder("channel1");
         REQUIRE( offOrder.size() == 1);
         REQUIRE( offOrder[0] == 0);
         onDelays = pdt.channelOnDelays("channel1");
         REQUIRE( onDelays.size() == 1);
         REQUIRE( onDelays[0] == 100);
         offDelays = pdt.channelOffDelays("channel1");
         REQUIRE( offDelays.size() == 1);
         REQUIRE( offDelays[0] == 120);

         outlets = pdt.channelOutlets("channel2");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 1 );

         onOrder = pdt.channelOnOrder("channel2");
         REQUIRE( onOrder.size() == 1);
         REQUIRE( onOrder[0] == 0);
         offOrder = pdt.channelOffOrder("channel2");
         REQUIRE( offOrder.size() == 1);
         REQUIRE( offOrder[0] == 0);
         onDelays = pdt.channelOnDelays("channel2");
         REQUIRE( onDelays.size() == 1);
         REQUIRE( onDelays[0] == 105);
         offDelays = pdt.channelOffDelays("channel2");
         REQUIRE( offDelays.size() == 1);
         REQUIRE( offDelays[0] == 130);

         outlets = pdt.channelOutlets("channel3");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 2 );

         onOrder = pdt.channelOnOrder("channel3");
         REQUIRE( onOrder.size() == 1);
         REQUIRE( onOrder[0] == 0);
         offOrder = pdt.channelOffOrder("channel3");
         REQUIRE( offOrder.size() == 1);
         REQUIRE( offOrder[0] == 0);
         onDelays = pdt.channelOnDelays("channel3");
         REQUIRE( onDelays.size() == 1);
         REQUIRE( onDelays[0] == 107);
         offDelays = pdt.channelOffDelays("channel3");
         REQUIRE( offDelays.size() == 1);
         REQUIRE( offDelays[0] == 132);

         outlets = pdt.channelOutlets("channel4");
         REQUIRE( outlets.size() == 1);
         REQUIRE( outlets[0] == 3 );

         onOrder = pdt.channelOnOrder("channel4");
         REQUIRE( onOrder.size() == 1);
         REQUIRE( onOrder[0] == 0);
         offOrder = pdt.channelOffOrder("channel4");
         REQUIRE( offOrder.size() == 1);
         REQUIRE( offOrder[0] == 0);
         onDelays = pdt.channelOnDelays("channel4");
         REQUIRE( onDelays.size() == 1);
         REQUIRE( onDelays[0] == 108);
         offDelays = pdt.channelOffDelays("channel4");
         REQUIRE( offDelays.size() == 1);
         REQUIRE( offDelays[0] == 133);
      }
   }

   GIVEN("a config file with 2 channels for 4 outlets")
   {
      WHEN("using outlet keyword, only outlet specified")
      {
         mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1",     "channel2" },
                                                        {"outlet",       "outlet"   },
                                                        {"0,1",             "2,3"   } );

         mx::app::appConfigurator config;
         config.readConfig("/tmp/outletController_test.conf");

         outletControllerTest pdt;
         int rv;
         rv = pdt.setupConfig(config);
         REQUIRE( rv == 0);

         rv = pdt.loadConfig(config);
         REQUIRE( rv == 0);
         REQUIRE( pdt.numChannels() == 2);

         std::vector<size_t> outlets, onOrder, offOrder;
         std::vector<unsigned> onDelays, offDelays;
         outlets = pdt.channelOutlets("channel1");
         REQUIRE( outlets.size() == 2);
         REQUIRE( outlets[0] == 0 );
         REQUIRE( outlets[1] == 1 );

         onOrder = pdt.channelOnOrder("channel1");
         REQUIRE( onOrder.size() == 0);
         offOrder = pdt.channelOffOrder("channel1");
         REQUIRE( offOrder.size() == 0);
         onDelays = pdt.channelOnDelays("channel1");
         REQUIRE( onDelays.size() == 0);
         offDelays = pdt.channelOffDelays("channel1");
         REQUIRE( offDelays.size() == 0);

         outlets = pdt.channelOutlets("channel2");
         REQUIRE( outlets.size() == 2);
         REQUIRE( outlets[0] == 2 );
         REQUIRE( outlets[1] == 3 );

         onOrder = pdt.channelOnOrder("channel2");
         REQUIRE( onOrder.size() == 0);
         offOrder = pdt.channelOffOrder("channel2");
         REQUIRE( offOrder.size() == 0);
         onDelays = pdt.channelOnDelays("channel2");
         REQUIRE( onDelays.size() == 0);
         offDelays = pdt.channelOffDelays("channel2");
         REQUIRE( offDelays.size() == 0);


      }

      WHEN("using outlet keyword, all specified")
      {
         mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1","channel1", "channel1", "channel1", "channel1",  "channel2", "channel2", "channel2", "channel2", "channel2"   },
                                                        {"outlet",  "onOrder",  "offOrder", "onDelays", "offDelays", "outlet",   "onOrder",  "offOrder", "onDelays", "offDelays"  },
                                                        {"0,1",     "0,1",      "1,0",      "0,105",    "0,107",     "2,3",      "1,0",      "0,1",      "0,106",    "0,108"      } );

         mx::app::appConfigurator config;
         config.readConfig("/tmp/outletController_test.conf");

         outletControllerTest pdt;
         int rv;
         rv = pdt.setupConfig(config);
         REQUIRE( rv == 0);

         rv = pdt.loadConfig(config);
         REQUIRE( rv == 0);
         REQUIRE( pdt.numChannels() == 2);

         std::vector<size_t> outlets, onOrder, offOrder;
         std::vector<unsigned> onDelays, offDelays;
         outlets = pdt.channelOutlets("channel1");
         REQUIRE( outlets.size() == 2);
         REQUIRE( outlets[0] == 0 );
         REQUIRE( outlets[1] == 1 );

         onOrder = pdt.channelOnOrder("channel1");
         REQUIRE( onOrder.size() == 2);
         REQUIRE( onOrder[0] == 0 );
         REQUIRE( onOrder[1] == 1 );
         offOrder = pdt.channelOffOrder("channel1");
         REQUIRE( offOrder.size() == 2);
         REQUIRE( offOrder[0] == 1 );
         REQUIRE( offOrder[1] == 0 );
         onDelays = pdt.channelOnDelays("channel1");
         REQUIRE( onDelays.size() == 2);
         REQUIRE( onDelays[0] == 0 );
         REQUIRE( onDelays[1] == 105 );
         offDelays = pdt.channelOffDelays("channel1");
         REQUIRE( offDelays.size() == 2);
         REQUIRE( offDelays[0] == 0 );
         REQUIRE( offDelays[1] == 107 );

         outlets = pdt.channelOutlets("channel2");
         REQUIRE( outlets.size() == 2);
         REQUIRE( outlets[0] == 2 );
         REQUIRE( outlets[1] == 3 );

         onOrder = pdt.channelOnOrder("channel2");
         REQUIRE( onOrder.size() == 2);
         REQUIRE( onOrder[0] == 1 );
         REQUIRE( onOrder[1] == 0 );
         offOrder = pdt.channelOffOrder("channel2");
         REQUIRE( offOrder.size() == 2);
         REQUIRE( offOrder[0] == 0 );
         REQUIRE( offOrder[1] == 1 );
         onDelays = pdt.channelOnDelays("channel2");
         REQUIRE( onDelays.size() == 2);
         REQUIRE( onDelays[0] == 0 );
         REQUIRE( onDelays[1] == 106 );
         offDelays = pdt.channelOffDelays("channel2");
         REQUIRE( offDelays.size() == 2);
         REQUIRE( offDelays[0] == 0 );
         REQUIRE( offDelays[1] == 108 );
      }

      WHEN("using outlets keyword, only outlet specified")
      {
         mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1",     "channel2" },
                                                        {"outlets",       "outlets"   },
                                                        {"0,1",             "2,3"   } );

         mx::app::appConfigurator config;
         config.readConfig("/tmp/outletController_test.conf");

         outletControllerTest pdt;
         int rv;
         rv = pdt.setupConfig(config);
         REQUIRE( rv == 0);

         rv = pdt.loadConfig(config);
         REQUIRE( rv == 0);
         REQUIRE( pdt.numChannels() == 2);

         std::vector<size_t> outlets, onOrder, offOrder;
         std::vector<unsigned> onDelays, offDelays;
         outlets = pdt.channelOutlets("channel1");
         REQUIRE( outlets.size() == 2);
         REQUIRE( outlets[0] == 0 );
         REQUIRE( outlets[1] == 1 );

         onOrder = pdt.channelOnOrder("channel1");
         REQUIRE( onOrder.size() == 0);
         offOrder = pdt.channelOffOrder("channel1");
         REQUIRE( offOrder.size() == 0);
         onDelays = pdt.channelOnDelays("channel1");
         REQUIRE( onDelays.size() == 0);
         offDelays = pdt.channelOffDelays("channel1");
         REQUIRE( offDelays.size() == 0);


         outlets = pdt.channelOutlets("channel2");
         REQUIRE( outlets.size() == 2);
         REQUIRE( outlets[0] == 2 );
         REQUIRE( outlets[1] == 3 );

         onOrder = pdt.channelOnOrder("channel2");
         REQUIRE( onOrder.size() == 0);
         offOrder = pdt.channelOffOrder("channel2");
         REQUIRE( offOrder.size() == 0);
         onDelays = pdt.channelOnDelays("channel2");
         REQUIRE( onDelays.size() == 0);
         offDelays = pdt.channelOffDelays("channel2");
         REQUIRE( offDelays.size() == 0);
      }

      WHEN("using outlets keyword, all specified")
      {
         mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1","channel1", "channel1", "channel1", "channel1",  "channel2", "channel2", "channel2", "channel2", "channel2"   },
                                                        {"outlets",  "onOrder",  "offOrder", "onDelays", "offDelays", "outlets",   "onOrder",  "offOrder", "onDelays", "offDelays"  },
                                                        {"0,2",     "0,1",      "1,0",      "0,105",    "0,107",     "1,3",      "1,0",      "0,1",      "0,106",    "0,108"      } );

         mx::app::appConfigurator config;
         config.readConfig("/tmp/outletController_test.conf");

         outletControllerTest pdt;
         int rv;
         rv = pdt.setupConfig(config);
         REQUIRE( rv == 0);

         rv = pdt.loadConfig(config);
         REQUIRE( rv == 0);
         REQUIRE( pdt.numChannels() == 2);

         std::vector<size_t> outlets, onOrder, offOrder;
         std::vector<unsigned> onDelays, offDelays;
         outlets = pdt.channelOutlets("channel1");
         REQUIRE( outlets.size() == 2);
         REQUIRE( outlets[0] == 0 );
         REQUIRE( outlets[1] == 2 );

         onOrder = pdt.channelOnOrder("channel1");
         REQUIRE( onOrder.size() == 2);
         REQUIRE( onOrder[0] == 0 );
         REQUIRE( onOrder[1] == 1 );
         offOrder = pdt.channelOffOrder("channel1");
         REQUIRE( offOrder.size() == 2);
         REQUIRE( offOrder[0] == 1 );
         REQUIRE( offOrder[1] == 0 );
         onDelays = pdt.channelOnDelays("channel1");
         REQUIRE( onDelays.size() == 2);
         REQUIRE( onDelays[0] == 0 );
         REQUIRE( onDelays[1] == 105 );
         offDelays = pdt.channelOffDelays("channel1");
         REQUIRE( offDelays.size() == 2);
         REQUIRE( offDelays[0] == 0 );
         REQUIRE( offDelays[1] == 107 );

         outlets = pdt.channelOutlets("channel2");
         REQUIRE( outlets.size() == 2);
         REQUIRE( outlets[0] == 1 );
         REQUIRE( outlets[1] == 3 );

         onOrder = pdt.channelOnOrder("channel2");
         REQUIRE( onOrder.size() == 2);
         REQUIRE( onOrder[0] == 1 );
         REQUIRE( onOrder[1] == 0 );
         offOrder = pdt.channelOffOrder("channel2");
         REQUIRE( offOrder.size() == 2);
         REQUIRE( offOrder[0] == 0 );
         REQUIRE( offOrder[1] == 1 );
         onDelays = pdt.channelOnDelays("channel2");
         REQUIRE( onDelays.size() == 2);
         REQUIRE( onDelays[0] == 0 );
         REQUIRE( onDelays[1] == 106 );
         offDelays = pdt.channelOffDelays("channel2");
         REQUIRE( offDelays.size() == 2);
         REQUIRE( offDelays[0] == 0 );
         REQUIRE( offDelays[1] == 108 );
      }
   }
}

/// outletController Operation
/**
 * \ingroup outletController_tests
 */
SCENARIO( "outletController Operation", "[outletController]" )
{
   GIVEN("a config file with 4 channels for 4 outlets, only outlet specified")
   {
      mx::app::writeConfigFile( "/tmp/outletController_test.conf", 
                                {"channel1", "channel2",     "channel3",      "channel4"},
                                {"outlet",   "outlet",       "outlet",          "outlet"},
                                {"0",         "1",             "2",                "3"} );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      pdt.setupConfig(config);
      pdt.loadConfig(config);

      WHEN("test device startup outlet states")
      {
         //Verify outlet startup state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state at startup
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );
      }

      WHEN("operating a single channel")
      {
         //Turn on channel1
         pdt.turnChannelOn("channel1");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 2 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         //Turn off channel1
         pdt.turnChannelOff("channel1");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         //Turn on channel2
         pdt.turnChannelOn("channel2");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 2 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 2 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         //Turn off channel2
         pdt.turnChannelOff("channel2");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         //Turn on channel3
         pdt.turnChannelOn("channel3");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 2 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 2 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         //Turn off channel3
         pdt.turnChannelOff("channel3");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         //Turn on channel4
         pdt.turnChannelOn("channel4");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 2 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 2 );

         //Turn off channel4
         pdt.turnChannelOff("channel4");

         //Verify outlet startup state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state at startup
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );
      }

      WHEN("operating multiple channels")
      {
         //Turn on channel1&2
         pdt.turnChannelOn("channel1");
         pdt.turnChannelOn("channel2");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 2 );
         REQUIRE( pdt.outletState(1) == 2 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 2 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         //Turn off channel1&2
         pdt.turnChannelOff("channel1");
         pdt.turnChannelOff("channel2");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         //Turn on channel3&4
         pdt.turnChannelOn("channel3");
         pdt.turnChannelOn("channel4");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 2 );
         REQUIRE( pdt.outletState(3) == 2 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 2 );
         REQUIRE( pdt.channelState("channel4") == 2 );

         //Turn off channel3&4
         pdt.turnChannelOff("channel3");
         pdt.turnChannelOff("channel4");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         //Turn on channel1&3
         pdt.turnChannelOn("channel1");
         pdt.turnChannelOn("channel3");

         //Verify outlet state
         REQUIRE( pdt.m_outletStates[0] == 2);
         REQUIRE( pdt.outletState(0) == 2 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 2 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 2 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         //Turn off channel1&3
         pdt.turnChannelOff("channel1");
         pdt.turnChannelOff("channel3");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         //Turn on channel2&4
         pdt.turnChannelOn("channel2");
         pdt.turnChannelOn("channel4");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 2 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 2 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 2 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 2 );

         //Turn off channel2&4
         pdt.turnChannelOff("channel2");
         pdt.turnChannelOff("channel4");

         //Verify outlet startup state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state at startup
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );
      }

      WHEN("outlets intermediate")
      {
         pdt.m_outletStates[0] = 1;

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 1 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 1 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         pdt.m_outletStates[0] = 0;

         pdt.m_outletStates[1] = 1;

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 1 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 1 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         pdt.m_outletStates[1] = 0;

         pdt.m_outletStates[2] = 1;

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 1 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 1 );
         REQUIRE( pdt.channelState("channel4") == 0 );

         pdt.m_outletStates[2] = 0;

         pdt.m_outletStates[3] = 1;

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 1 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
         REQUIRE( pdt.channelState("channel3") == 0 );
         REQUIRE( pdt.channelState("channel4") == 1 );

         pdt.m_outletStates[3] = 0;
      }

   }

   GIVEN("a config file with 2 channels for 4 outlets, only outlet specified")
   {
      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1",     "channel2" },
                                                        {"outlet",       "outlet"   },
                                                        {"0,1",             "2,3"   } );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      pdt.setupConfig(config);
      pdt.loadConfig(config);

      WHEN("test device startup outlet states")
      {
         //Verify outlet startup state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state at startup
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
      WHEN("operating a single channel")
      {
         //Turn on channel1
         pdt.turnChannelOn("channel1");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 2 );
         REQUIRE( pdt.outletState(1) == 2 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //verify outlet order
         REQUIRE( pdt.m_timestamps[1] > pdt.m_timestamps[0]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn off channel1
         pdt.turnChannelOff("channel1");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn on channel2
         pdt.turnChannelOn("channel2");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 2 );
         REQUIRE( pdt.outletState(3) == 2 );

         //verify outlet order
         REQUIRE( pdt.m_timestamps[3] > pdt.m_timestamps[2]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 2 );

         //Turn off channel2
         pdt.turnChannelOff("channel2");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
      WHEN("operating two channels")
      {
         //Turn on channels
         pdt.turnChannelOn("channel1");
         pdt.turnChannelOn("channel2");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 2 );
         REQUIRE( pdt.outletState(1) == 2 );
         REQUIRE( pdt.outletState(2) == 2 );
         REQUIRE( pdt.outletState(3) == 2 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 2 );

         //Turn off channel1&2
         pdt.turnChannelOff("channel1");
         pdt.turnChannelOff("channel2");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );

      }
      WHEN("outlets intermediate")
      {
         pdt.m_outletStates[0] = 2;

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 2 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         REQUIRE( pdt.channelState("channel1") == 1);
         REQUIRE( pdt.channelState("channel2") == 0);

         pdt.turnChannelOn("channel2");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 2 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 2 );
         REQUIRE( pdt.outletState(3) == 2 );

         REQUIRE( pdt.channelState("channel1") == 1);
         REQUIRE( pdt.channelState("channel2") == 2);

         pdt.m_outletStates[0] = 0;

         REQUIRE( pdt.channelState("channel1") == 0);
         REQUIRE( pdt.channelState("channel2") == 2);

         pdt.turnChannelOff("channel2");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         REQUIRE( pdt.channelState("channel1") == 0);
         REQUIRE( pdt.channelState("channel2") == 0);


         pdt.m_outletStates[2] = 1;

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 1 );
         REQUIRE( pdt.outletState(3) == 0 );

         REQUIRE( pdt.channelState("channel1") == 0);
         REQUIRE( pdt.channelState("channel2") == 1);

         pdt.turnChannelOn("channel1");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 2 );
         REQUIRE( pdt.outletState(1) == 2 );
         REQUIRE( pdt.outletState(2) == 1 );
         REQUIRE( pdt.outletState(3) == 0 );

         REQUIRE( pdt.channelState("channel1") == 2);
         REQUIRE( pdt.channelState("channel2") == 1);

         pdt.m_outletStates[2] = 0;

         REQUIRE( pdt.channelState("channel1") == 2);
         REQUIRE( pdt.channelState("channel2") == 0);

         pdt.turnChannelOff("channel1");

         //Verify outlet state
         REQUIRE( pdt.outletState(0) == 0 );
         REQUIRE( pdt.outletState(1) == 0 );
         REQUIRE( pdt.outletState(2) == 0 );
         REQUIRE( pdt.outletState(3) == 0 );

         REQUIRE( pdt.channelState("channel1") == 0);
         REQUIRE( pdt.channelState("channel2") == 0);
      }
   }
   GIVEN("a config file with 2 channels for 4 outlets, onOrder specified")
   {
      //Here we are just testing order, so we don't need to verify outlet state anymore

      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1",  "channel1",   "channel2", "channel2" },
                                                        {"outlet",  "onOrder",     "outlet", "onOrder"   },
                                                        {"0,1",     "0,1",        "2,3",     "0,1"   } );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      pdt.setupConfig(config);
      pdt.loadConfig(config);

      WHEN("test device startup channel states")
      {
         //Verify channel state at startup
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
      WHEN("operating a single channel")
      {
         //Turn on channel1
         pdt.turnChannelOn("channel1");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[1] > pdt.m_timestamps[0]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn off channel1
         pdt.turnChannelOff("channel1");

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn on channel2
         pdt.turnChannelOn("channel2");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[3] > pdt.m_timestamps[2]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 2 );

         //Turn off channel2
         pdt.turnChannelOff("channel2");

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
   }
   GIVEN("a config file with 2 channels for 4 outlets, onOrder reversed")
   {
      //Here we are just testing order, so we don't need to verify outlet state anymore

      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1",  "channel1",   "channel2", "channel2" },
                                                        {"outlet",  "onOrder",     "outlet", "onOrder"   },
                                                        {"0,1",     "1,0",        "2,3",     "1,0"   } );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      pdt.setupConfig(config);
      pdt.loadConfig(config);

      WHEN("test device startup channel states")
      {
         //Verify channel state at startup
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
      WHEN("operating a single channel")
      {
         //Turn on channel1
         pdt.turnChannelOn("channel1");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[0] > pdt.m_timestamps[1]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn off channel1
         pdt.turnChannelOff("channel1");

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn on channel2
         pdt.turnChannelOn("channel2");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[2] > pdt.m_timestamps[3]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 2 );

         //Turn off channel2
         pdt.turnChannelOff("channel2");

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
   }
   GIVEN("a config file with 2 channels for 4 outlets, onOrder and offOrder specified, the same")
   {
      //Here we are just testing order, so we don't need to verify outlet state anymore

      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1",  "channel1",  "channel1",  "channel2", "channel2", "channel2" },
                                                        {"outlet",  "onOrder",  "offOrder",   "outlet", "onOrder", "offOrder"   },
                                                        {"0,1",     "0,1",       "0,1",  "2,3",     "0,1" , "0,1"  } );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      pdt.setupConfig(config);
      pdt.loadConfig(config);

      WHEN("test device startup channel states")
      {
         //Verify channel state at startup
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
      WHEN("operating a single channel")
      {
         //Turn on channel1
         pdt.turnChannelOn("channel1");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[1] > pdt.m_timestamps[0]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn off channel1
         pdt.turnChannelOff("channel1");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[1] > pdt.m_timestamps[0]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn on channel2
         pdt.turnChannelOn("channel2");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[3] > pdt.m_timestamps[2]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 2 );

         //Turn off channel2
         pdt.turnChannelOff("channel2");

         REQUIRE( pdt.m_timestamps[3] >= pdt.m_timestamps[2]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
   }
   GIVEN("a config file with 2 channels for 4 outlets, onOrder and offOrder specified, different")
   {
      //Here we are just testing order, so we don't need to verify outlet state anymore

      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1",  "channel1",  "channel1",  "channel2", "channel2", "channel2" },
                                                        {"outlet",  "onOrder",  "offOrder",   "outlet", "onOrder", "offOrder"   },
                                                        {"0,1",     "0,1",       "1,0",  "2,3",     "0,1" , "1,0"  } );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      pdt.setupConfig(config);
      pdt.loadConfig(config);

      WHEN("test device startup channel states")
      {
         //Verify channel state at startup
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
      WHEN("operating a single channel")
      {
         //Turn on channel1
         pdt.turnChannelOn("channel1");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[1] > pdt.m_timestamps[0]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 0 );


         //Turn off channel1
         pdt.turnChannelOff("channel1");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[0] > pdt.m_timestamps[1]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn on channel2
         pdt.turnChannelOn("channel2");


         //verify outlet order
         REQUIRE( pdt.m_timestamps[3] > pdt.m_timestamps[2]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 2 );

         //Turn off channel2
         pdt.turnChannelOff("channel2");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[2] > pdt.m_timestamps[3]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
   }
   GIVEN("a config file with 2 channels for 4 outlets, onOrder and offOrder specified, different, reversed")
   {
      //Here we are just testing order, so we don't need to verify outlet state anymore

      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1",  "channel1",  "channel1",  "channel2", "channel2", "channel2" },
                                                        {"outlet",  "onOrder",  "offOrder",   "outlet", "onOrder", "offOrder"   },
                                                        {"0,1",     "1,0",       "0,1",  "2,3",     "1,0" , "0,1"  } );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      pdt.setupConfig(config);
      pdt.loadConfig(config);

      WHEN("test device startup channel states")
      {
         //Verify channel state at startup
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
      WHEN("operating a single channel")
      {
         //Turn on channel1
         pdt.turnChannelOn("channel1");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[0] > pdt.m_timestamps[1]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 0 );


         //Turn off channel1
         pdt.turnChannelOff("channel1");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[1] > pdt.m_timestamps[0]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn on channel2
         pdt.turnChannelOn("channel2");


         //verify outlet order
         REQUIRE( pdt.m_timestamps[2] > pdt.m_timestamps[3]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 2 );

         //Turn off channel2
         pdt.turnChannelOff("channel2");

         //verify outlet order
         REQUIRE( pdt.m_timestamps[3] > pdt.m_timestamps[2]);

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
   }
}

SCENARIO( "outletController Operation with delays", "[outletController]" )
{
   std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
   std::cout << "[outletController] Testing delays ... \n";
   GIVEN("a config file with 2 channels for 4 outlets, onDelays specified")
   {
      //Here we are just testing delays, so we don't need to verify outlet state anymore

      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1",  "channel1",   "channel2", "channel2" },
                                                        {"outlet",  "onDelays",   "outlet", "onDelays"   },
                                                        {"0,1",     "0,350",        "2,3",     "0,150"   } );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      pdt.setupConfig(config);
      pdt.loadConfig(config);

      WHEN("operating a single channel")
      {
         //Turn on channel1
         pdt.turnChannelOn("channel1");

         //verify outlet delay
         REQUIRE( pdt.m_timestamps[1] >= Approx(pdt.m_timestamps[0]+0.350));
         std::cout << "Ch1 On Delay was " << (pdt.m_timestamps[1] - pdt.m_timestamps[0])*1000 << " msec, expected 350.\n";

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn off channel1
         pdt.turnChannelOff("channel1");
         std::cout << "Ch1 Off Delay was " << (pdt.m_timestamps[1] - pdt.m_timestamps[0])*1000 << " msec, expected ~0.\n";

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn on channel2
         pdt.turnChannelOn("channel2");


         //verify outlet delay
         REQUIRE( pdt.m_timestamps[3] >= Approx(pdt.m_timestamps[2]+0.150));
         std::cout << "Ch2 On Delay was " << (pdt.m_timestamps[3] - pdt.m_timestamps[2])*1000 << " msec, expected 150.\n";

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 2 );

         //Turn off channel2
         pdt.turnChannelOff("channel2");
         std::cout << "Ch2 Off Delay was " << (pdt.m_timestamps[3] - pdt.m_timestamps[2])*1000 << " msec, expected ~0.\n";

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
   }
   GIVEN("a config file with 2 channels for 4 outlets, offDelays specified")
   {
      //Here we are just testing delays, so we don't need to verify outlet state anymore

      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1",  "channel1",   "channel2", "channel2" },
                                                        {"outlet",  "offDelays",   "outlet", "offDelays"   },
                                                        {"0,1",     "0,550",        "2,3",     "0,750"   } );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      pdt.setupConfig(config);
      pdt.loadConfig(config);

      WHEN("operating a single channel")
      {
         //Turn on channel1
         pdt.turnChannelOn("channel1");

         std::cout << "Ch1 On Delay was " << (pdt.m_timestamps[1] - pdt.m_timestamps[0])*1000 << " msec, expected ~0.\n";

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn off channel1
         pdt.turnChannelOff("channel1");

         REQUIRE( pdt.m_timestamps[1] >= Approx(pdt.m_timestamps[0]+0.550));
         std::cout << "Ch1 Off Delay was " << (pdt.m_timestamps[1] - pdt.m_timestamps[0])*1000 << " msec, expected 550.\n";


         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn on channel2
         pdt.turnChannelOn("channel2");


         //verify outlet delay
         std::cout << "Ch1 On Delay was " << (pdt.m_timestamps[3] - pdt.m_timestamps[2])*1000 << " msec, expected ~0.\n";

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 2 );

         //Turn off channel2
         pdt.turnChannelOff("channel2");
         REQUIRE( pdt.m_timestamps[3] >= Approx(pdt.m_timestamps[2]+0.750));
         std::cout << "Ch1 On Delay was " << (pdt.m_timestamps[3] - pdt.m_timestamps[2])*1000 << " msec, expected 750.\n";

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
   }
   GIVEN("a config file with 2 channels for 4 outlets, onDelays and offDelays specified, off order reversed")
   {
      //Here we are just testing delays, so we don't need to verify outlet state anymore

      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1", "channel1", "channel1", "channel1", "channel1",  "channel2", "channel2", "channel2",  "channel2", "channel2" },
                                                     {"outlet",   "onOrder",  "onDelays", "offOrder", "offDelays", "outlet",   "onOrder",  "onDelays", "offOrder", "offDelays"   },
                                                     {"0,1",      "0,1",      "0,350",    "1,0",      "0,450",     "2,3",      "0,1",      "0,150",      "1,0",      "0,75"   } );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      pdt.setupConfig(config);
      pdt.loadConfig(config);

      WHEN("operating a single channel")
      {
         //Turn on channel1
         pdt.turnChannelOn("channel1");

         //verify outlet delay
         REQUIRE( pdt.m_timestamps[1] >= Approx(pdt.m_timestamps[0]+0.350));
         std::cout << "Ch1 On Delay was " << (pdt.m_timestamps[1] - pdt.m_timestamps[0])*1000 << " msec, expected 350.\n";

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 2 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn off channel1
         pdt.turnChannelOff("channel1");
         REQUIRE( pdt.m_timestamps[0] >= Approx(pdt.m_timestamps[1]+0.450));
         std::cout << "Ch1 Off Delay was " << (pdt.m_timestamps[0] - pdt.m_timestamps[1])*1000 << " msec, expected 450.\n";

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );

         //Turn on channel2
         pdt.turnChannelOn("channel2");


         //verify outlet delay
         REQUIRE( pdt.m_timestamps[3] >= Approx(pdt.m_timestamps[2]+0.150));
         std::cout << "Ch2 On Delay was " << (pdt.m_timestamps[3] - pdt.m_timestamps[2])*1000 << " msec, expected 150.\n";

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 2 );

         //Turn off channel2
         pdt.turnChannelOff("channel2");
         REQUIRE( pdt.m_timestamps[2] >= Approx(pdt.m_timestamps[3]+0.075));
         std::cout << "Ch2 Off Delay was " << (pdt.m_timestamps[2] - pdt.m_timestamps[3])*1000 << " msec, expected 75.\n";

         //Verify channel state
         REQUIRE( pdt.channelState("channel1") == 0 );
         REQUIRE( pdt.channelState("channel2") == 0 );
      }
   }
   std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
}

/// outletController Configuration Error Handling
/**
 * \ingroup outletController_tests
 *
 * Verify that loadConfig() returns -1 for each kind of malformed channel section.
 * Each GIVEN writes a config file under /tmp with one specific defect, such as a
 * missing outlet value, an out-of-range outlet number, or an order or delay list
 * whose size does not match the outlet list.
 */
SCENARIO( "outletController Configuration Error Handling", "[outletController]" )
{
   GIVEN("a channel section with the outlet keyword present but no value")
   {
      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1"},
                                                        {"outlet"},
                                                        {""} );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      int rv = pdt.setupConfig(config);
      REQUIRE( rv == 0);

      WHEN("loadConfig is called")
      {
         rv = pdt.loadConfig(config);
         REQUIRE( rv == -1);
      }
   }

   GIVEN("a channel section with an outlet number that is out of range")
   {
      // The harness has 4 outlets numbered 0 to 3, so outlet 10 is out of range.
      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1"},
                                                        {"outlet"},
                                                        {"10"} );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      int rv = pdt.setupConfig(config);
      REQUIRE( rv == 0);

      WHEN("loadConfig is called")
      {
         rv = pdt.loadConfig(config);
         REQUIRE( rv == -1);
      }
   }

   GIVEN("a channel section with a mismatched onOrder size")
   {
      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1", "channel1"},
                                                        {"outlet",   "onOrder"},
                                                        {"0,1",      "0"} );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      int rv = pdt.setupConfig(config);
      REQUIRE( rv == 0);

      WHEN("loadConfig is called")
      {
         rv = pdt.loadConfig(config);
         REQUIRE( rv == -1);
      }
   }

   GIVEN("a channel section with a mismatched offOrder size")
   {
      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1", "channel1"},
                                                        {"outlet",   "offOrder"},
                                                        {"0,1",      "0"} );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      int rv = pdt.setupConfig(config);
      REQUIRE( rv == 0);

      WHEN("loadConfig is called")
      {
         rv = pdt.loadConfig(config);
         REQUIRE( rv == -1);
      }
   }

   GIVEN("a channel section with a mismatched onDelays size")
   {
      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1", "channel1"},
                                                        {"outlet",   "onDelays"},
                                                        {"0,1",      "100"} );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      int rv = pdt.setupConfig(config);
      REQUIRE( rv == 0);

      WHEN("loadConfig is called")
      {
         rv = pdt.loadConfig(config);
         REQUIRE( rv == -1);
      }
   }

   GIVEN("a channel section with a mismatched offDelays size")
   {
      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1", "channel1"},
                                                        {"outlet",   "offDelays"},
                                                        {"0,1",      "100"} );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      int rv = pdt.setupConfig(config);
      REQUIRE( rv == 0);

      WHEN("loadConfig is called")
      {
         rv = pdt.loadConfig(config);
         REQUIRE( rv == -1);
      }
   }
}

/// outletController updateOutletStates
/**
 * \ingroup outletController_tests
 *
 * Verify the default updateOutletStates() implementation. On the success path
 * every outlet updates cleanly and the function returns 0. On the error path one
 * outlet reports a negative state. That value is returned and the loop stops early.
 */
SCENARIO( "outletController updateOutletStates default implementation", "[outletController]" )
{
   GIVEN("a controller with 4 outlets, all in a valid state")
   {
      outletControllerTest pdt;
      pdt.setNumberOfOutlets(4);

      WHEN("updateOutletStates is called and all outlets report valid states")
      {
         int rv = pdt.updateOutletStates();
         REQUIRE( rv == 0);
      }

      WHEN("updateOutletStates is called and an outlet reports an error")
      {
         // The harness updateOutletState() returns the stored state, so a negative value acts as an error.
         pdt.m_outletStates[2] = -7;
         int rv = pdt.updateOutletStates();
         REQUIRE( rv == -7);
         pdt.m_outletStates[2] = 0; // Restore the outlet state.
      }
   }
}

/// outletController turnChannelOn/turnChannelOff edge cases
/**
 * \ingroup outletController_tests
 *
 * Verify the branches of turnChannelOn() and turnChannelOff() that ordinary
 * operation does not reach. These are the null-mutex fallback, the no-op when a
 * channel is already on or already off, the skip when the state delay has not
 * elapsed, and the error returns when an outlet fails. Outlet failures are
 * injected through the harness member m_failOutlet.
 */
SCENARIO( "outletController turnChannelOn/turnChannelOff edge cases", "[outletController]" )
{
   GIVEN("a config file with 2 channels for 4 outlets")
   {
      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1",  "channel2" },
                                                        {"outlet",   "outlet"   },
                                                        {"0,1",      "2,3"   } );

      mx::app::appConfigurator config;
      config.readConfig("/tmp/outletController_test.conf");

      outletControllerTest pdt;
      pdt.setupConfig(config);
      pdt.loadConfig(config);

      WHEN("the channel mutex is null")
      {
         // Force the null-mutex fallback path by clearing channelSpec::m_mutex.
         // The mutex object itself is still owned and deleted through m_channelMutexes, so this is safe.
         pdt.m_channels["channel1"].m_mutex = nullptr;

         int rv = pdt.turnChannelOn("channel1");
         REQUIRE( rv == 0 );
         REQUIRE( pdt.channelState("channel1") == 2 );

         pdt.m_channels["channel1"].m_mutex = nullptr;
         rv = pdt.turnChannelOff("channel1");
         REQUIRE( rv == 0 );
         REQUIRE( pdt.channelState("channel1") == 0 );
      }

      WHEN("turning on a channel that is already on")
      {
         pdt.turnChannelOn("channel1");
         REQUIRE( pdt.channelState("channel1") == 2 );

         int rv = pdt.turnChannelOn("channel1");
         REQUIRE( rv == 0 );
         REQUIRE( pdt.channelState("channel1") == 2 );
      }

      WHEN("turning off a channel that is already off")
      {
         REQUIRE( pdt.channelState("channel1") == 0 );

         int rv = pdt.turnChannelOff("channel1");
         REQUIRE( rv == 0 );
         REQUIRE( pdt.channelState("channel1") == 0 );
      }

      WHEN("the state delay has not elapsed when turning a channel on")
      {
         pdt.m_stateDelay = 5.0; // A large delay in seconds so it cannot elapse during the test.

         // First transition. The default m_stateTime is old, so this proceeds normally.
         int rv = pdt.turnChannelOn("channel1");
         REQUIRE( rv == 0 );
         REQUIRE( pdt.channelState("channel1") == 2 );

         // Force the channel to appear off without going through turnChannelOff().
         // This keeps m_stateTime recent, that is within the delay window.
         pdt.m_outletStates[0] = 0;
         pdt.m_outletStates[1] = 0;
         REQUIRE( pdt.channelState("channel1") == 0 );

         // Turning it on again should now be skipped by the delay logic.
         rv = pdt.turnChannelOn("channel1");
         REQUIRE( rv == 0 );
         REQUIRE( pdt.outletState(0) == 0 ); // Unchanged because turnOutletOn() was not called.
         REQUIRE( pdt.outletState(1) == 0 );
      }

      WHEN("the state delay has not elapsed when turning a channel off")
      {
         pdt.m_stateDelay = 5.0; // A large delay in seconds so it cannot elapse during the test.

         // Turn the channel on. This sets m_stateTime to now.
         int rv = pdt.turnChannelOn("channel1");
         REQUIRE( rv == 0 );
         REQUIRE( pdt.channelState("channel1") == 2 );

         // Turning it off immediately should be skipped by the delay logic.
         rv = pdt.turnChannelOff("channel1");
         REQUIRE( rv == 0 );
         REQUIRE( pdt.channelState("channel1") == 2 ); // Unchanged. The channel is still on.
      }

      WHEN("the first outlet fails to turn on")
      {
         pdt.m_failOutlet = 0; // The first outlet of channel1.

         int rv = pdt.turnChannelOn("channel1");
         REQUIRE( rv == -1 );

         pdt.m_failOutlet = -1; // Stop injecting failures.
      }

      WHEN("a subsequent outlet fails to turn on")
      {
         pdt.m_failOutlet = 1; // The second outlet of channel1.

         int rv = pdt.turnChannelOn("channel1");
         REQUIRE( rv == -1 );
         REQUIRE( pdt.outletState(0) == 2 ); // The first outlet succeeded before the failure.

         pdt.m_failOutlet = -1; // Stop injecting failures.
      }

      WHEN("the first outlet fails to turn off")
      {
         pdt.turnChannelOn("channel1");
         REQUIRE( pdt.channelState("channel1") == 2 );

         pdt.m_failOutlet = 0; // The first outlet of channel1.

         int rv = pdt.turnChannelOff("channel1");
         REQUIRE( rv == -1 );

         pdt.m_failOutlet = -1; // Stop injecting failures.
      }

      WHEN("a subsequent outlet fails to turn off")
      {
         pdt.turnChannelOn("channel1");
         REQUIRE( pdt.channelState("channel1") == 2 );

         pdt.m_failOutlet = 1; // The second outlet of channel1.

         int rv = pdt.turnChannelOff("channel1");
         REQUIRE( rv == -1 );
         REQUIRE( pdt.outletState(0) == 0 ); // The first outlet succeeded before the failure.

         pdt.m_failOutlet = -1; // Stop injecting failures.
      }
   }
}

/// outletController stateIntToString
/**
 * \ingroup outletController_tests
 *
 * Verify all four branches of the free function MagAOX::app::dev::stateIntToString().
 * The branches return Off, Int, On, and Unk. This function is not header-only. It is
 * compiled into libMagAOX.a from outletController.cpp.
 */
SCENARIO( "outletController stateIntToString", "[outletController]" )
{
   GIVEN("the four possible outlet state values")
   {
      WHEN("the state is OUTLET_STATE_OFF")
      {
         REQUIRE( dev::stateIntToString(OUTLET_STATE_OFF) == "Off" );
      }

      WHEN("the state is OUTLET_STATE_INTERMEDIATE")
      {
         REQUIRE( dev::stateIntToString(OUTLET_STATE_INTERMEDIATE) == "Int" );
      }

      WHEN("the state is OUTLET_STATE_ON")
      {
         REQUIRE( dev::stateIntToString(OUTLET_STATE_ON) == "On" );
      }

      WHEN("the state is an unknown/unexpected value")
      {
         REQUIRE( dev::stateIntToString(OUTLET_STATE_UNKNOWN) == "Unk" );
         REQUIRE( dev::stateIntToString(42) == "Unk" );
      }
   }
}

/// outletController appStartup, setupINDI, newCallBack_channels, updateINDI
/**
 * \ingroup outletController_tests
 *
 * Verify the INDI side of the mixin. appStartup() and setupINDI() must register the
 * outlet and channel properties. Every registration failure must be propagated as -1.
 * newCallBack_channels() must reject requests when the app is not READY and must
 * dispatch on and off requests when it is. updateINDI() must return 0 both without a
 * driver and with a real FIFO-less driver installed.
 *
 * The harness appStartup() forwards to the real dev::outletController appStartup() so
 * these paths run. Registration failures are injected through m_regFailAt.
 */
SCENARIO( "outletController appStartup, setupINDI, newCallBack_channels, updateINDI", "[outletController]" )
{
   GIVEN("a config file with 4 channels for 4 outlets")
   {
      mx::app::writeConfigFile( "/tmp/outletController_test_startup.conf", {"channel1", "channel2", "channel3", "channel4"},
                                                     {"outlet",   "outlet",   "outlet",   "outlet"},
                                                     {"0",         "1",        "2",         "3"} );

      WHEN("appStartup succeeds")
      {
         mx::app::appConfigurator config;
         config.readConfig("/tmp/outletController_test_startup.conf");

         {
            outletControllerTest pdt;
            REQUIRE( pdt.setupConfig(config) == 0 );
            REQUIRE( pdt.loadConfig(config) == 0 );

            REQUIRE( pdt.appStartup() == 0 );
         }

         // setupINDI() delegates to appStartup(). A fresh instance is needed to call it because
         // registerIndiPropertyNew() rejects registering the same properties twice on one instance.
         // MagAOXApp also forbids a second live instance, so the instance above must go out of
         // scope first. That is why it is wrapped in its own block.
         outletControllerTest pdt2;
         REQUIRE( pdt2.setupConfig(config) == 0 );
         REQUIRE( pdt2.loadConfig(config) == 0 );
         REQUIRE( pdt2.setupINDI() == 0 );
      }

      WHEN("every registration failure is propagated")
      {
         // First run a successful appStartup() to count the register calls it makes.
         // Then run it once per call, failing a different call each time.
         int totalCalls = 0;
         {
            mx::app::appConfigurator config;
            config.readConfig("/tmp/outletController_test_startup.conf");
            outletControllerTest pdt;
            REQUIRE( pdt.setupConfig(config) == 0 );
            REQUIRE( pdt.loadConfig(config) == 0 );
            REQUIRE( pdt.appStartup() == 0 );
            totalCalls = pdt.m_regCallCount;
         }
         REQUIRE( totalCalls > 0 );

         for( int failAt = 1; failAt <= totalCalls; ++failAt )
         {
            mx::app::appConfigurator config;
            config.readConfig("/tmp/outletController_test_startup.conf");

            outletControllerTest pdt;
            REQUIRE( pdt.setupConfig(config) == 0 );
            REQUIRE( pdt.loadConfig(config) == 0 );
            pdt.m_regFailAt = failAt;

            REQUIRE( pdt.appStartup() == -1 );
         }
      }

      WHEN("newCallBack_channels dispatches based on state/target")
      {
         mx::app::appConfigurator config;
         config.readConfig("/tmp/outletController_test_startup.conf");

         outletControllerTest pdt;
         REQUIRE( pdt.setupConfig(config) == 0 );
         REQUIRE( pdt.loadConfig(config) == 0 );
         REQUIRE( pdt.appStartup() == 0 );

         // The app state is not READY, so the request is rejected regardless of its content.
         pcf::IndiProperty notReady( pcf::IndiProperty::Text );
         notReady.setDevice( pdt.configName() );
         notReady.setName( "channel1" );
         REQUIRE( pdt.newCallBack_channels( notReady ) == -1 );

         pdt.state( MagAOX::app::stateCodes::READY );

         pcf::IndiProperty onReq( pcf::IndiProperty::Text );
         onReq.setDevice( pdt.configName() );
         onReq.setName( "channel1" );
         onReq.add( pcf::IndiElement( "target" ) );
         onReq["target"].setValue( "on" );
         REQUIRE( pdt.newCallBack_channels( onReq ) == 0 );
         REQUIRE( pdt.updateOutletState( 0 ) == OUTLET_STATE_ON );

         pcf::IndiProperty offReq( pcf::IndiProperty::Text );
         offReq.setDevice( pdt.configName() );
         offReq.setName( "channel1" );
         offReq.add( pcf::IndiElement( "state" ) );
         offReq["state"].setValue( "off" );
         REQUIRE( pdt.newCallBack_channels( offReq ) == 0 );
         REQUIRE( pdt.updateOutletState( 0 ) == OUTLET_STATE_OFF );

         // A value that is neither on nor off takes no action and returns 0.
         pcf::IndiProperty neither( pcf::IndiProperty::Text );
         neither.setDevice( pdt.configName() );
         neither.setName( "channel1" );
         neither.add( pcf::IndiElement( "state" ) );
         neither["state"].setValue( "sideways" );
         REQUIRE( pdt.newCallBack_channels( neither ) == 0 );

         // Also go through the static function that registerIndiPropertyNew() installs as the callback.
         // The mixin casts the void pointer back to the harness type.
         REQUIRE( outletControllerTest::st_newCallBack_channels( &pdt, onReq ) == 0 );
      }

      WHEN("updateINDI publishes outlet and channel states with a real driver")
      {
         mx::app::appConfigurator config;
         config.readConfig("/tmp/outletController_test_startup.conf");

         outletControllerTest pdt;
         REQUIRE( pdt.setupConfig(config) == 0 );
         REQUIRE( pdt.loadConfig(config) == 0 );
         REQUIRE( pdt.appStartup() == 0 );

         REQUIRE( pdt.updateINDI() == 0 ); // No driver yet, so updateINDI() returns early.

         pdt.setupRealDriver();
         REQUIRE( pdt.updateINDI() == 0 );

         // Change one outlet so the update has a state change to publish.
         pdt.turnOutletOn(0);
         REQUIRE( pdt.updateINDI() == 0 );
      }
   }
}

} //namespace outletController_tests
