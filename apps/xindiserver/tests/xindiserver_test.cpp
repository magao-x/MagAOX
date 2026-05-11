/** \file xindiserver_test.cpp
 * \brief Catch2 tests for the xindiserver app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup xindiserver_files
 */

#include "../../../tests/testXWC.hpp"

#include "../xindiserver.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup xindiserver_unit_test xindiserver Unit Tests
 * \brief Unit tests for the xindiserver application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `xindiserver` unit tests.
/** \ingroup xindiserver_unit_test
 */
namespace xindiserverTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
struct xindiserver_test
{
    void indiserver_m( xindiserver &xi, const int &m )
    {
        xi.indiserver_m = m;
    }
    void indiserver_n( xindiserver &xi, const bool &n )
    {
        xi.indiserver_n = n;
    }
    void indiserver_p( xindiserver &xi, const int &p )
    {
        xi.indiserver_p = p;
    }
    void indiserver_v( xindiserver &xi, const int &v )
    {
        xi.indiserver_v = v;
    }
    void indiserver_x( xindiserver &xi, const bool &x )
    {
        xi.indiserver_x = x;
    }
    void m_local( xindiserver &xi, const std::vector<std::string> &ml )
    {
        xi.m_local = ml;
    }
    void m_remote( xindiserver &xi, const std::vector<std::string> &mr )
    {
        xi.m_remote = mr;
    }

    tunnelMapT &tunnelMap( xindiserver &xi )
    {
        return xi.m_tunnels;
    }
};
/// \endcond

/// Verify xindiserver builds command lines and tunnel-derived driver lists consistently.
/**
 * \ingroup xindiserver_unit_test
 */
SCENARIO( "xindiserver constructs inserver options", "[xindiserver]" )
{
    // clang-format off
   #ifdef XINDISERVER_TEST_DOXYGEN_REF
   xindiserver::constructIndiserverCommand( *(std::vector<std::string> *)nullptr );
   xindiserver::addRemoteDrivers( *(std::vector<std::string> *)nullptr );
   xindiserver::addLocalDrivers( *(std::vector<std::string> *)nullptr );
   loadSSHTunnelConfigs( *(tunnelMapT *)nullptr, *(mx::app::appConfigurator *)nullptr );
   #endif
    // clang-format on

    GIVEN( "A default constructed xindiserver" )
    {
        xindiserver      xi;
        xindiserver_test xi_test;

        int rv;

        WHEN( "Option m with argument provided" )
        {
            std::vector<std::string> clargs;
            xi_test.indiserver_m( xi, 100 );

            rv = xi.constructIndiserverCommand( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 3 );
            REQUIRE( clargs[0] == "indiserver" );
            REQUIRE( clargs[1] == "-m" );
            REQUIRE( clargs[2] == "100" );
        }

        WHEN( "Option n provided" )
        {
            std::vector<std::string> clargs;
            xi_test.indiserver_n( xi, true );

            rv = xi.constructIndiserverCommand( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 2 );
            REQUIRE( clargs[0] == "indiserver" );
            REQUIRE( clargs[1] == "-n" );
        }

        WHEN( "Option p provided with argument" )
        {
            std::vector<std::string> clargs;
            xi_test.indiserver_p( xi, 2000 );

            rv = xi.constructIndiserverCommand( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 3 );
            REQUIRE( clargs[0] == "indiserver" );
            REQUIRE( clargs[1] == "-p" );
            REQUIRE( clargs[2] == "2000" );
        }

        WHEN( "1 Option v provided with argument (-v)" )
        {
            std::vector<std::string> clargs;
            xi_test.indiserver_v( xi, 1 );

            rv = xi.constructIndiserverCommand( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 2 );
            REQUIRE( clargs[0] == "indiserver" );
            REQUIRE( clargs[1] == "-v" );
        }

        WHEN( "2 Option v provided with argument (-vv)" )
        {
            std::vector<std::string> clargs;
            xi_test.indiserver_v( xi, 2 );

            rv = xi.constructIndiserverCommand( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 2 );
            REQUIRE( clargs[0] == "indiserver" );
            REQUIRE( clargs[1] == "-vv" );
        }

        WHEN( "3 Option v provided with argument (3==>-vvv)" )
        {
            std::vector<std::string> clargs;
            xi_test.indiserver_v( xi, 3 );

            rv = xi.constructIndiserverCommand( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 2 );
            REQUIRE( clargs[0] == "indiserver" );
            REQUIRE( clargs[1] == "-vvv" );
        }

        WHEN( "Option v provided with argument (4==>-vvv)" )
        {
            std::vector<std::string> clargs;
            xi_test.indiserver_v( xi, 4 );

            rv = xi.constructIndiserverCommand( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 2 );
            REQUIRE( clargs[0] == "indiserver" );
            REQUIRE( clargs[1] == "-vvv" );
        }

        WHEN( "Option v provided with argument (0==>)" )
        {
            std::vector<std::string> clargs;
            xi_test.indiserver_v( xi, 0 );

            rv = xi.constructIndiserverCommand( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 1 );
            REQUIRE( clargs[0] == "indiserver" );
        }

        WHEN( "Option x provided" )
        {
            std::vector<std::string> clargs;
            xi_test.indiserver_x( xi, true );

            rv = xi.constructIndiserverCommand( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 2 );
            REQUIRE( clargs[0] == "indiserver" );
            REQUIRE( clargs[1] == "-x" );
        }

        WHEN( "All options provided" )
        {
            std::vector<std::string> clargs;
            xi_test.indiserver_m( xi, 100 );
            xi_test.indiserver_n( xi, true );
            xi_test.indiserver_p( xi, 2000 );
            xi_test.indiserver_v( xi, 2 );
            xi_test.indiserver_x( xi, true );

            rv = xi.constructIndiserverCommand( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 8 );
            REQUIRE( clargs[0] == "indiserver" );
            REQUIRE( clargs[1] == "-m" );
            REQUIRE( clargs[2] == "100" );
            REQUIRE( clargs[3] == "-n" );
            REQUIRE( clargs[4] == "-p" );
            REQUIRE( clargs[5] == "2000" );
            REQUIRE( clargs[6] == "-vv" );
            REQUIRE( clargs[7] == "-x" );
        }
    }
}

} // namespace xindiserverTest

} // namespace libXWCTest

SCENARIO( "xindiserver constructs local driver arguments", "[xindiserver]" )
{
    GIVEN( "A default constructed xindiserver" )
    {
        xindiserver      xi;
        xindiserver_test xi_test;

        int rv;

        WHEN( "Single local driver" )
        {
            std::vector<std::string> ml( { "driverX" } );
            xi_test.m_local( xi, ml );

            std::vector<std::string> clargs;
            rv = xi.addLocalDrivers( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 1 );
            REQUIRE( clargs[0] == "/opt/MagAOX/drivers/driverX" );
        }

        WHEN( "Two local drivers" )
        {
            std::vector<std::string> ml( { "driverY", "driverZ" } );
            xi_test.m_local( xi, ml );

            std::vector<std::string> clargs;
            rv = xi.addLocalDrivers( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 2 );
            REQUIRE( clargs[0] == "/opt/MagAOX/drivers/driverY" );
            REQUIRE( clargs[1] == "/opt/MagAOX/drivers/driverZ" );
        }

        WHEN( "Three local drivers" )
        {
            std::vector<std::string> ml( { "driverX", "driverY", "driverZ" } );
            xi_test.m_local( xi, ml );

            std::vector<std::string> clargs;
            rv = xi.addLocalDrivers( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 3 );
            REQUIRE( clargs[0] == "/opt/MagAOX/drivers/driverX" );
            REQUIRE( clargs[1] == "/opt/MagAOX/drivers/driverY" );
            REQUIRE( clargs[2] == "/opt/MagAOX/drivers/driverZ" );
        }

        WHEN( "Three local drivers, with an error (@)" )
        {
            std::vector<std::string> ml( { "driverX", "driver@Y", "driverZ" } );
            xi_test.m_local( xi, ml );

            std::vector<std::string> clargs;
            rv = xi.addLocalDrivers( clargs );
            REQUIRE( rv == XINDISERVER_E_BADDRIVERSPEC );
        }

        WHEN( "Three local drivers, with an error (/)" )
        {
            std::vector<std::string> ml( { "driver/X", "driverY", "driverZ" } );
            xi_test.m_local( xi, ml );

            std::vector<std::string> clargs;
            rv = xi.addLocalDrivers( clargs );
            REQUIRE( rv == XINDISERVER_E_BADDRIVERSPEC );
        }

        WHEN( "Three local drivers, with an error (:)" )
        {
            std::vector<std::string> ml( { "driverX", "driverY", "driver:Z" } );
            xi_test.m_local( xi, ml );

            std::vector<std::string> clargs;
            rv = xi.addLocalDrivers( clargs );
            REQUIRE( rv == XINDISERVER_E_BADDRIVERSPEC );
        }

        WHEN( "Three local drivers, duplicate" )
        {
            std::vector<std::string> ml( { "driverX", "driverY", "driverX" } );
            xi_test.m_local( xi, ml );

            std::vector<std::string> clargs;
            rv = xi.addLocalDrivers( clargs );
            REQUIRE( rv == XINDISERVER_E_DUPLICATEDRIVER );
        }
    }
}

SCENARIO( "xindiserver constructs remote driver arguments", "[xindiserver]" )
{
    GIVEN( "A default constructed xindiserver" )
    {
        xindiserver      xi;
        xindiserver_test xi_test;

        int rv;

        WHEN( "Single remote driver, single remote host" )
        {
            std::vector<std::string> mr( { "driverX@host1" } );
            xi_test.m_remote( xi, mr );

            mx::app::writeConfigFile( "/tmp/xindiserver_test.conf",
                                      { "host1", "host1", "host1" },
                                      { "remoteHost", "localPort", "remotePort" },
                                      { "host1", "1000", "81" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/xindiserver_test.conf" );

            loadSSHTunnelConfigs( xi_test.tunnelMap( xi ), config );

            std::vector<std::string> clargs;
            rv = xi.addRemoteDrivers( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 1 );
            REQUIRE( clargs[0] == "driverX@localhost:1000" );
        }

        WHEN( "Two remote drivers, single remote host" )
        {
            std::vector<std::string> mr( { "driverX@host1", "driverY@host1" } );
            xi_test.m_remote( xi, mr );

            mx::app::writeConfigFile( "/tmp/xindiserver_test.conf",
                                      { "host1", "host1", "host1" },
                                      { "remoteHost", "localPort", "remotePort" },
                                      { "host1", "1000", "81" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/xindiserver_test.conf" );

            xi_test.tunnelMap( xi ).clear(); // make sure we don't hold over
            loadSSHTunnelConfigs( xi_test.tunnelMap( xi ), config );

            std::vector<std::string> clargs;
            rv = xi.addRemoteDrivers( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 2 );
            REQUIRE( clargs[0] == "driverX@localhost:1000" );
            REQUIRE( clargs[1] == "driverY@localhost:1000" );
        }

        WHEN( "Two remote drivers, two remote hosts" )
        {
            std::vector<std::string> mr( { "driverX@host1", "driverY@host2" } );
            xi_test.m_remote( xi, mr );

            mx::app::writeConfigFile(
                "/tmp/xindiserver_test.conf",
                { "host1", "host1", "host1", "host2", "host2", "host2" },
                { "remoteHost", "localPort", "remotePort", "remoteHost", "localPort", "remotePort" },
                { "host1", "1000", "81", "host2", "1002", "86" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/xindiserver_test.conf" );

            xi_test.tunnelMap( xi ).clear(); // make sure we don't hold over
            loadSSHTunnelConfigs( xi_test.tunnelMap( xi ), config );

            std::vector<std::string> clargs;
            rv = xi.addRemoteDrivers( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 2 );
            REQUIRE( clargs[0] == "driverX@localhost:1000" );
            REQUIRE( clargs[1] == "driverY@localhost:1002" );
        }

        WHEN( "Three remote drivers, two remote hosts, in order" )
        {
            std::vector<std::string> mr( { "driverX@host1", "driverZ@host1", "driverY@host2" } );
            xi_test.m_remote( xi, mr );

            mx::app::writeConfigFile(
                "/tmp/xindiserver_test.conf",
                { "host1", "host1", "host1", "host2", "host2", "host2" },
                { "remoteHost", "localPort", "remotePort", "remoteHost", "localPort", "remotePort" },
                { "host1", "1000", "81", "host2", "1002", "86" } );

            mx::app::appConfigurator config;
            config.readConfig( "/tmp/xindiserver_test.conf" );

            xi_test.tunnelMap( xi ).clear(); // make sure we don't hold over
            loadSSHTunnelConfigs( xi_test.tunnelMap( xi ), config );

            std::vector<std::string> clargs;
            rv = xi.addRemoteDrivers( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 3 );

            REQUIRE( clargs[0] == "driverX@localhost:1000" );
            REQUIRE( clargs[1] == "driverZ@localhost:1000" );
            REQUIRE( clargs[2] == "driverY@localhost:1002" );
        }

        WHEN( "Three remote drivers, two remote hosts, arb order" )
        {
            std::vector<std::string> mr( { "driverX@host1", "driverZ@host2", "driverY@host1" } );
            xi_test.m_remote( xi, mr );

            mx::app::writeConfigFile(
                "/tmp/xindiserver_test.conf",
                { "host1", "host1", "host1", "host2", "host2", "host2" },
                { "remoteHost", "localPort", "remotePort", "remoteHost", "localPort", "remotePort" },
                { "host1", "1000", "81", "host2", "1002", "86" } );

            mx::app::appConfigurator config;
            config.readConfig( "/tmp/xindiserver_test.conf" );

            xi_test.tunnelMap( xi ).clear(); // make sure we don't hold over
            loadSSHTunnelConfigs( xi_test.tunnelMap( xi ), config );

            std::vector<std::string> clargs;
            rv = xi.addRemoteDrivers( clargs );
            REQUIRE( rv == 0 );
            REQUIRE( clargs.size() == 3 );

            REQUIRE( clargs[0] == "driverX@localhost:1000" );
            REQUIRE( clargs[1] == "driverZ@localhost:1002" );
            REQUIRE( clargs[2] == "driverY@localhost:1000" );
        }

        WHEN( "Three remote drivers, two remote hosts, error in host" )
        {
            std::vector<std::string> mr( { "driverX@host1", "driverZ@host2", "driverY@host1" } );
            xi_test.m_remote( xi, mr );

            mx::app::writeConfigFile( "/tmp/xindiserver_test.conf",
                                      { "host1", "host1", "host1", "host2", "host2" },
                                      { "remoteHost", "localPort", "remotePort", "remoteHost", "localPort" },
                                      { "exao2", "1000", "81", "exao2", "1002" } );

            mx::app::appConfigurator config;
            config.readConfig( "/tmp/xindiserver_test.conf" );

            xi_test.tunnelMap( xi ).clear(); // make sure we don't hold over
            loadSSHTunnelConfigs( xi_test.tunnelMap( xi ), config );

            std::vector<std::string> clargs;
            rv = xi.addRemoteDrivers( clargs );
            REQUIRE( rv == XINDISERVER_E_TUNNELNOTFOUND );
        }

        WHEN( "Three remote drivers, two remote hosts, error in driver" )
        {
            std::vector<std::string> mr( { "driverX", "driverZ@host2", "driverY@host1" } );
            xi_test.m_remote( xi, mr );

            mx::app::writeConfigFile(
                "/tmp/xindiserver_test.conf",
                { "host1", "host1", "host1", "host2", "host2", "host2" },
                { "remoteHost", "localPort", "remotePort", "remoteHost", "localPort", "remotePort" },
                { "host1", "1000", "81", "host2", "1002", "86" } );

            mx::app::appConfigurator config;
            config.readConfig( "/tmp/xindiserver_test.conf" );

            xi_test.tunnelMap( xi ).clear(); // make sure we don't hold over
            loadSSHTunnelConfigs( xi_test.tunnelMap( xi ), config );

            std::vector<std::string> clargs;
            rv = xi.addRemoteDrivers( clargs );
            REQUIRE( rv == XINDISERVER_E_BADDRIVERSPEC );
        }

        WHEN( "Three remote drivers, two remote hosts, duplicate driver" )
        {
            std::vector<std::string> mr( { "driverX@host2", "driverX@host1", "driverY@host1" } );
            xi_test.m_remote( xi, mr );

            mx::app::writeConfigFile(
                "/tmp/xindiserver_test.conf",
                { "host1", "host1", "host1", "host2", "host2", "host2" },
                { "remoteHost", "localPort", "remotePort", "remoteHost", "localPort", "remotePort" },
                { "host1", "1000", "81", "host2", "1002", "86" } );

            mx::app::appConfigurator config;
            config.readConfig( "/tmp/xindiserver_test.conf" );
            loadSSHTunnelConfigs( xi_test.tunnelMap( xi ), config );

            std::vector<std::string> clargs;
            rv = xi.addRemoteDrivers( clargs );
            REQUIRE( rv == XINDISERVER_E_DUPLICATEDRIVER );
        }
    }
}

SCENARIO( "xindiserver constructs both local and remote driver arguments", "[xindiserver]" )
{
    GIVEN( "A default constructed xindiserver" )
    {
        xindiserver      xi;
        xindiserver_test xi_test;

        int rv;

        WHEN( "single local driver, single remote driver, single remote host" )
        {

            std::vector<std::string> ml( { "driverX" } );
            xi_test.m_local( xi, ml );

            std::vector<std::string> mr( { "driverY@host1" } );
            xi_test.m_remote( xi, mr );

            mx::app::writeConfigFile( "/tmp/xindiserver_test.conf",
                                      { "host1", "host1", "host1" },
                                      { "remoteHost", "localPort", "remotePort" },
                                      { "host1", "1000", "81" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/xindiserver_test.conf" );

            loadSSHTunnelConfigs( xi_test.tunnelMap( xi ), config );

            std::vector<std::string> clargs;
            rv = xi.addRemoteDrivers( clargs );
            REQUIRE( rv == 0 );

            rv = xi.addLocalDrivers( clargs );
            REQUIRE( rv == 0 );

            REQUIRE( clargs.size() == 2 );

            REQUIRE( clargs[0] == "driverY@localhost:1000" );
            REQUIRE( clargs[1] == "/opt/MagAOX/drivers/driverX" );
        }

        WHEN( "single local driver, single remote driver, single remote host -- duplicate driver" )
        {

            std::vector<std::string> ml( { "driverX" } );
            xi_test.m_local( xi, ml );

            std::vector<std::string> mr( { "driverX@host1" } );
            xi_test.m_remote( xi, mr );

            mx::app::writeConfigFile( "/tmp/xindiserver_test.conf",
                                      { "host1", "host1", "host1" },
                                      { "remoteHost", "localPort", "remotePort" },
                                      { "host1", "1000", "81" } );
            mx::app::appConfigurator config;
            config.readConfig( "/tmp/xindiserver_test.conf" );

            loadSSHTunnelConfigs( xi_test.tunnelMap( xi ), config );

            std::vector<std::string> clargs;
            rv = xi.addRemoteDrivers( clargs );
            REQUIRE( rv == 0 );

            rv = xi.addLocalDrivers( clargs );
            REQUIRE( rv == XINDISERVER_E_DUPLICATEDRIVER );
        }
    }
}
