//#define CATCH_CONFIG_MAIN
#include "../../../../tests/catch2/catch.hpp"

#include <mx/sys/timeUtils.hpp>

#define OUTLET_CTRL_TEST_NOINDI
#define OUTLET_CTRL_TEST_NOLOG
#include "../outletController.hpp"

namespace outletController_tests 
{
   
struct outletControllerTest : public MagAOX::app::dev::outletController<outletControllerTest>
{
   std::vector<double> m_timestamps;
   
   outletControllerTest()
   {
      setNumberOfOutlets(4);
      m_timestamps.resize(4,0);
      turnOutletOff(0);
      turnOutletOff(1);
      turnOutletOff(2);
      turnOutletOff(3);
   }

   virtual int updateOutletState( int outletNum )
   {
      return m_outletStates[outletNum];
   }
   
   
   virtual int turnOutletOn( int outletNum )
   {
      m_outletStates[outletNum] = 2;
      mx::sys::nanoSleep(1);
      m_timestamps[outletNum] = mx::sys::get_curr_time();
      
      return 0;
   }
   
   virtual int turnOutletOff( int outletNum )
   {
      m_outletStates[outletNum] = 0;
      mx::sys::nanoSleep(1);
      m_timestamps[outletNum] = mx::sys::get_curr_time();
      
      return 0;
   }

};


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

SCENARIO( "outletController Operation", "[outletController]" ) 
{
   GIVEN("a config file with 4 channels for 4 outlets, only outlet specified")
   {
      mx::app::writeConfigFile( "/tmp/outletController_test.conf", {"channel1", "channel2",     "channel3",      "channel4"},
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
         
         //Turn off channel1
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
         
         //Turn off channel1
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
         
         //Turn off channel1
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
         
         //Turn off channel2&4
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

} //namespace outletController_tests 
