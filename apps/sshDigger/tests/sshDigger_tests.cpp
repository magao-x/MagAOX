//#define CATCH_CONFIG_MAIN
#include "../../../tests/catch2/catch.hpp"

#include <mx/timeUtils.hpp>

#define SSHDIGGER_TEST_NOINDI
#define SSHDIGGER_TEST_NOLOG
#include "../sshDigger.hpp"

using namespace MagAOX::app;

class sshDigger_test : public sshDigger
{
public:
   void configName(const std::string & cn) 
   {
      m_configName = cn;
   }
   
   std::string remoteHost() {return m_remoteHost;}
   int localPort() {return m_localPort;}
   int remotePort() {return m_remotePort;}
};


SCENARIO( "sshDigger Configuration", "[sshDigger]" ) 
{
   GIVEN("a config file with 1 tunnel")
   {
      WHEN("the tunnel is fully specified and matches configName")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",      "tunnel1",     "tunnel1" },
                                                               {"remoteHost",   "localPort",       "remotePort" },
                                                               {"exao2",         "80",             "81" } );
      
      
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel1");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == 0);
         
         REQUIRE( dig.remoteHost() == "exao2");
         REQUIRE( dig.localPort() == 80 );
         REQUIRE( dig.remotePort() == 81 );
         
      }
      
      WHEN("no unused sections")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {},
                                                               {},
                                                               {} );
      
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel2");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == SSHDIGGER_E_NOTUNNELS);
         
         
      }
      
      WHEN("the tunnel does not match configName")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",      "tunnel1",     "tunnel1" },
                                                               {"remoteHost",   "localPort",       "remotePort" },
                                                               {"exao2",         "80",             "81" } );
      
      
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel2");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == SSHDIGGER_E_NOTUNNELFOUND);
         
         
      }
      
      WHEN("no remote host")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",     "tunnel1" },
                                                               {"localPort",       "remotePort" },
                                                               {"80",             "81" } );
      
      
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel1");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == SSHDIGGER_E_NOHOSTNAME );
         
      }
      
      WHEN("no local port")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",      "tunnel1" },
                                                               {"remoteHost",   "remotePort" },
                                                               {"exao2",        "81" } );
         
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel1");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == SSHDIGGER_E_NOLOCALPORT );
         
      }
      
      WHEN("no remote port")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",      "tunnel1" },
                                                               {"remoteHost",   "localPort" },
                                                               {"exao2",        "80" } );
         
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel1");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == SSHDIGGER_E_NOREMOTEPORT );
         
      }
      
   }
   GIVEN("a config file with 2 tunnels")
   {
      WHEN("the tunnels are fully specified and match their configNames")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",      "tunnel1",   "tunnel1",    "tunnel2",      "tunnel2",   "tunnel2" },
                                                               {"remoteHost",   "localPort", "remotePort", "remoteHost",   "localPort", "remotePort" },
                                                               {"exao2",         "80",       "81",         "exao3",         "85",       "86" } );
      
      
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");

         int rv;
         
         sshDigger_test dig;
         dig.configName("tunnel1");
         
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == 0);
         REQUIRE( dig.remoteHost() == "exao2");
         REQUIRE( dig.localPort() == 80 );
         REQUIRE( dig.remotePort() == 81 );
         
         dig.configName("tunnel2");

         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == 0);
         REQUIRE( dig.remoteHost() == "exao3");
         REQUIRE( dig.localPort() == 85 );
         REQUIRE( dig.remotePort() == 86 );
         
      }
      
      
      WHEN("No tunnels match configName")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",      "tunnel1",   "tunnel1",    "tunnel2",      "tunnel2",   "tunnel2" },
                                                               {"remoteHost",   "localPort", "remotePort", "remoteHost",   "localPort", "remotePort" },
                                                               {"exao2",         "80",       "81",         "exao3",         "85",       "86" } );      
      
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel3");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == SSHDIGGER_E_NOTUNNELFOUND);
         
         
      }
      
      WHEN("no remote host in first tunnel")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",   "tunnel1",    "tunnel2",      "tunnel2",   "tunnel2" },
                                                               {"localPort", "remotePort", "remoteHost",   "localPort", "remotePort" },
                                                               {"80",       "81",         "exao3",         "85",       "86" } );
         
      
      
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel1");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == SSHDIGGER_E_NOHOSTNAME );
         
         
         dig.configName("tunnel2");

         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == 0);
         REQUIRE( dig.remoteHost() == "exao3");
         REQUIRE( dig.localPort() == 85 );
         REQUIRE( dig.remotePort() == 86 );         
         
         
      }
      
      WHEN("no remote host in 2nd tunnel")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",      "tunnel1",   "tunnel1",    "tunnel2",   "tunnel2" },
                                                               {"remoteHost",   "localPort", "remotePort", "localPort", "remotePort" },
                                                               {"exao2",         "80",       "81",         "85",       "86" } );             
      
      
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel2");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == SSHDIGGER_E_NOHOSTNAME );
         
         
         dig.configName("tunnel1");

         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == 0);
         REQUIRE( dig.remoteHost() == "exao2");
         REQUIRE( dig.localPort() == 80 );
         REQUIRE( dig.remotePort() == 81 );         
         
         
      }
      
      WHEN("no local port in first tunnel")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",   "tunnel1",    "tunnel2",      "tunnel2",   "tunnel2" },
                                                               {"remoteHost", "remotePort", "remoteHost",   "localPort", "remotePort" },
                                                               {"exao2",       "81",         "exao3",         "85",       "86" } );
         
      
      
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel1");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == SSHDIGGER_E_NOLOCALPORT );
         
         
         dig.configName("tunnel2");

         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == 0);
         REQUIRE( dig.remoteHost() == "exao3");
         REQUIRE( dig.localPort() == 85 );
         REQUIRE( dig.remotePort() == 86 );         
         
         
      }
         
      WHEN("no local port in second tunnel")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",      "tunnel1",   "tunnel1",    "tunnel2",    "tunnel2" },
                                                               {"remoteHost",   "localPort", "remotePort", "remoteHost", "remotePort" },
                                                               {"exao2",         "80",       "81",         "exao3",      "86" } );      
         
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel2");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == SSHDIGGER_E_NOLOCALPORT );
         
         
         dig.configName("tunnel1");

         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == 0);
         REQUIRE( dig.remoteHost() == "exao2");
         REQUIRE( dig.localPort() == 80 );
         REQUIRE( dig.remotePort() == 81 );         
         
      }   
      
      WHEN("no remote port in first tunnel")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",    "tunnel1",   "tunnel2",      "tunnel2",   "tunnel2" },
                                                               {"remoteHost", "localPort", "remoteHost",   "localPort", "remotePort" },
                                                               {"exao2",      "80",        "exao3",         "85",       "86" } );
         
      
      
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel1");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == SSHDIGGER_E_NOREMOTEPORT );
         
         
         dig.configName("tunnel2");

         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == 0);
         REQUIRE( dig.remoteHost() == "exao3");
         REQUIRE( dig.localPort() == 85 );
         REQUIRE( dig.remotePort() == 86 );         
         
      }
      
      WHEN("no remote port in second tunnel")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",      "tunnel1",   "tunnel1",    "tunnel2",    "tunnel2" },
                                                               {"remoteHost",   "localPort", "remotePort", "remoteHost", "localPort" },
                                                               {"exao2",         "80",       "81",         "exao3",      "85" } );      
         
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel2");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == SSHDIGGER_E_NOREMOTEPORT );
         
         
         dig.configName("tunnel1");

         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == 0);
         REQUIRE( dig.remoteHost() == "exao2");
         REQUIRE( dig.localPort() == 80 );
         REQUIRE( dig.remotePort() == 81 );         
         
      }   
   }
} 

SCENARIO( "sshDigger tunnel exec preparation", "[sshDigger]" ) 
{
   GIVEN("a config file with 1 tunnel")
   {
      WHEN("creating the tunnelSpec")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",      "tunnel1",     "tunnel1" },
                                                               {"remoteHost",   "localPort",       "remotePort" },
                                                               {"exao2",         "80",             "81" } );
      
      
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel1");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == 0);
         
         REQUIRE( dig.tunnelSpec() == "80:localhost:81");
      }
      
      WHEN("creating the exec argv vector")
      {
         mx::app::writeConfigFile( "/tmp/sshDigger_test.conf", {"tunnel1",      "tunnel1",     "tunnel1" },
                                                               {"remoteHost",   "localPort",       "remotePort" },
                                                               {"exao2",         "80",             "81" } );
      
      
         mx::app::appConfigurator config;
         config.readConfig("/tmp/sshDigger_test.conf");
         
         sshDigger_test dig;
         dig.configName("tunnel1");
         int rv;
         rv = dig.loadConfigImpl(config);
         REQUIRE( rv == 0);
         
         std::vector<std::string> argsV;
         dig.genArgsV(argsV);
         
         REQUIRE( argsV[0] == "autossh");
         REQUIRE( argsV[1] == "-M0");
         REQUIRE( argsV[2] == "-nNTL");
         REQUIRE( argsV[3] == "80:localhost:81");
         REQUIRE( argsV[4] == "exao2");
      }
   }
}
