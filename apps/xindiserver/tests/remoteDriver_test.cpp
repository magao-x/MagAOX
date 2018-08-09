

#define CATCH_CONFIG_MAIN
#include "../../../tests/catch2/catch.hpp"


#include "../remoteDriver.hpp"

SCENARIO( "remoteDriver is constructed and modified", "[remoteDriver]" ) 
{
   GIVEN("A default constructed remoteDriver")
   {
      remoteDriver rd;
      int rv;
      
      REQUIRE(rd.name() == "");
      REQUIRE(rd.host() == "");
      REQUIRE(rd.port() == INDI_DEFAULT_PORT);
      
      WHEN("The name is changed")
      {
         rv = rd.name("remDrive");
         REQUIRE(rv == 0);
         REQUIRE(rd.name() == "remDrive");
      }
      
      WHEN("An invalid name is specified, includes @")
      {
         rv = rd.name("@ghj");
         REQUIRE(rv < 0);
         REQUIRE(rd.name() == "");
      }
      
      WHEN("An invalid name is specified, includes :")
      {
         rv = rd.name("ghj:345");
         REQUIRE(rv < 0);
         REQUIRE(rd.name() == "");
      }
      
      WHEN("The host is changed")
      {
         rv = rd.host("1.2.3.4");
         REQUIRE(rv == 0);
         REQUIRE(rd.host() == "1.2.3.4");
      }
      
      WHEN("An invalid host is specified, includes @")
      {
         rv = rd.host("1.2.@3.4");
         REQUIRE(rv < 0);
         REQUIRE(rd.host() == "");
      }
      
      WHEN("An invalid host is specified, includes :")
      {
         rv = rd.host("1.2.3:4");
         REQUIRE(rv < 0);
         REQUIRE(rd.host() == "");
      }
      
      WHEN("The port is changed")
      {
         rv = rd.port(256);
         REQUIRE(rd.port() == 256);
      }
   }
   
}

SCENARIO( "remoteDriver parses remote driver spec strings", "[remoteDriver]" ) 
{
   GIVEN("A default constructed remoteDriver")
   {
      remoteDriver rd;
      int rv;
      WHEN("A nominal remote driver spec is parsed (no port)")
      {
         rv = rd.parse("newDrive@remote");
         REQUIRE(rv == 0);
         REQUIRE(rd.name() == "newDrive");
         REQUIRE(rd.host() == "remote");
         REQUIRE(rd.port() == INDI_DEFAULT_PORT);
         REQUIRE(rd.hostSpec() == "remote:7624");
         REQUIRE(rd.fullSpec() == "newDrive@remote:7624");
 
      }
      
      WHEN("A whitespace-full remote driver spec is parsed (no port)")
      {
         rv = rd.parse("next drive @ 9. 8. 7 .6");
         REQUIRE(rv == 0);
         REQUIRE(rd.name() == "nextdrive");
         REQUIRE(rd.host() == "9.8.7.6");
         REQUIRE(rd.port() == INDI_DEFAULT_PORT);
      }
      
      WHEN("A nominal remote driver spec is parsed (w/ port)")
      {
         rv = rd.parse("nowDrive@remoteish:7");
         REQUIRE(rv == 0);
         REQUIRE(rd.name() == "nowDrive");
         REQUIRE(rd.host() == "remoteish");
         REQUIRE(rd.port() == 7);
      }
      
      WHEN("A whitespace-full remote driver spec is parsed (w/ port)")
      {
         rv = rd.parse("next B drive @ 9. 3. 7 .6 : 6 5 7");
         REQUIRE(rv == 0);
         REQUIRE(rd.name() == "nextBdrive");
         REQUIRE(rd.host() == "9.3.7.6");
         REQUIRE(rd.port() == 657);
      }
      
      WHEN("A valid remote driver spec is parsed (w/ empty port)")
      {
         rv = rd.parse("Q@B:");
         REQUIRE(rv == 0);
         REQUIRE(rd.name() == "Q");
         REQUIRE(rd.host() == "B");
         REQUIRE(rd.port() == INDI_DEFAULT_PORT);
      }
      
      WHEN("An invalid driver spec is parsed, @ first")
      {
         rv = rd.parse("@dname");
         REQUIRE(rv < 0);
         REQUIRE(rd.name() == "");
         REQUIRE(rd.host() == "");
      }
      
      WHEN("An invalid driver spec is parsed, @ last")
      {
         rv = rd.parse("@dname");
         REQUIRE(rv < 0);
         REQUIRE(rd.name() == "");
         REQUIRE(rd.host() == "");
      }
      
      WHEN("An invalid driver spec is parsed, no @")
      {
         rv = rd.parse("dname");
         REQUIRE(rv < 0);
         REQUIRE(rd.name() == "");
         REQUIRE(rd.host() == "");
      }
      
      WHEN("An invalid driver spec is parsed, : before @")
      {
         rv = rd.parse("dname:vg@:367");
         REQUIRE(rv < 0);
         REQUIRE(rd.name() == "");
         REQUIRE(rd.host() == "");
         REQUIRE(rd.port() == INDI_DEFAULT_PORT);
      }
      
   }
}
