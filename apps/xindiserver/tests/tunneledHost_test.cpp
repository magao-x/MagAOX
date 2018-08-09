
#define CATCH_CONFIG_MAIN
#include "../../../tests/catch2/catch.hpp"


#include "../tunneledHost.hpp"

SCENARIO( "tunneledHost is constructed and modified", "[tunneledHost]" ) 
{
   GIVEN("A default constructed tunneledHost")
   {
      tunneledHost th;
      int rv;
      
      REQUIRE(th.name() == "");
      REQUIRE(th.remotePort() == INDI_DEFAULT_PORT);
      REQUIRE(th.localPort() == 0);
      
      WHEN("The name is changed")
      {
         rv = th.name("remHost");
         REQUIRE(rv == 0);
         REQUIRE(th.name() == "remHost");
      }
      
      WHEN("The remote port is changed")
      {
         rv = th.remotePort(6);
         REQUIRE(rv == 0);
         REQUIRE(th.remotePort() == 6);
      }

      WHEN("The local port is changed")
      {
         rv = th.localPort(7);
         REQUIRE(rv == 0);
         REQUIRE(th.localPort() == 7);
      }
   }
}

SCENARIO( "tunneled host parses host spec strings", "[tunneledHost]" ) 
{
   GIVEN("A default constructed remoteDriver")
   {

      tunneledHost th;

      int rv ;
 
      WHEN("Parsing a nominal spec, with default remote port")
      {
         rv = th.parse( "128.168.34.56:7623" );
         REQUIRE(rv == 0);
         
         REQUIRE( th.name() == "128.168.34.56");
         REQUIRE( th.remotePort() == INDI_DEFAULT_PORT);
         REQUIRE( th.localPort() == 7623);
         REQUIRE( th.remoteSpec() == "128.168.34.56:7624");
         REQUIRE( th.fullSpec() == "128.168.34.56:7624:7623");
      }

      WHEN("Parsing a nominal spec, with remote port")
      {
         rv = th.parse( "localhost:7625:7624" );
         REQUIRE(rv == 0);
         
         REQUIRE( th.name() == "localhost");
         REQUIRE( th.remotePort() == 7625);
         REQUIRE( th.localPort() == 7624);
      }

      WHEN("Parsing a conforming full spec, full of whitespace")
      {
         rv = th.parse( " new host : 82 37 : 1845 " );
         REQUIRE(rv == 0);
         
         REQUIRE( th.name() == "newhost");
         REQUIRE( th.remotePort() == 8237);
         REQUIRE( th.localPort() == 1845);
      }
      

      WHEN("Empty remote port with ::.")
      {
         rv = th.parse( "host::7627" );
         REQUIRE(rv == 0);
         
         REQUIRE( th.name() == "host");
         REQUIRE( th.remotePort() == INDI_DEFAULT_PORT);
         REQUIRE( th.localPort() == 7627);
      }

      WHEN("No local port supplied, no :.")
      {
         rv = th.parse( "host" );
         REQUIRE(rv < 0);
         
         REQUIRE( th.name() == "");
         REQUIRE( th.remotePort() == INDI_DEFAULT_PORT);
         REQUIRE( th.localPort() == 0);
      }
      
      WHEN("Empty local port supplied, trailing:.")
      {
         rv = th.parse( "host:" );
         REQUIRE(rv < 0);
         
         REQUIRE( th.name() == "");
         REQUIRE( th.remotePort() == INDI_DEFAULT_PORT);
         REQUIRE( th.localPort() == 0);
      }

      WHEN("Empty local port supplied, trailing::.")
      {
         rv = th.parse( "host::" );
         REQUIRE(rv < 0);
         
         REQUIRE( th.name() == "");
         REQUIRE( th.remotePort() == INDI_DEFAULT_PORT);
         REQUIRE( th.localPort() == 0);
      }
      
      WHEN("No host supplied, only ports.")
      {
         rv = th.parse( ":6000:6001" );
         REQUIRE(rv < 0);
         
         REQUIRE( th.name() == "");
         REQUIRE( th.remotePort() == INDI_DEFAULT_PORT);
         REQUIRE( th.localPort() == 0);
      }
   }
}
