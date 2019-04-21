/** \file xtChannels_test.cpp
  * \brief Catch2 tests for the xtChannels struct in the Ctrl app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup xt1211Ctrl_files
  * 
  * History:
  *  -- Created 2019-04-21 by JRM
  */
#include "../../../tests/catch2/catch.hpp"

#include "../xtChannels.hpp"


namespace xtChannels_test 
{

SCENARIO( "Setting channels from registers", "[xtChannels]" )
{
   GIVEN("A set of input registers read from the device")
   {
      int rv;

      WHEN("Individual channels set")
      {
         xt1121Channels xtc;
         
         uint16_t registers[4];// = {0,0,0,0};
         
         //Channel 0
         memset(registers,0, sizeof(registers));
         registers[0] = 1;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == true);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 1
         memset(registers,0, sizeof(registers));
         registers[0] = 2;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == true);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 2
         memset(registers,0, sizeof(registers));
         registers[0] = 4;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == true);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 3
         memset(registers,0, sizeof(registers));
         registers[0] = 8;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == true);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 4
         memset(registers,0, sizeof(registers));
         registers[1] = 1;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == true);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 5
         memset(registers,0, sizeof(registers));
         registers[1] = 2;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == true);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 6
         memset(registers,0, sizeof(registers));
         registers[1] = 4;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == true);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 7
         memset(registers,0, sizeof(registers));
         registers[1] = 8;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == true);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 8
         memset(registers,0, sizeof(registers));
         registers[2] = 1;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == true);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 9
         memset(registers,0, sizeof(registers));
         registers[2] = 2;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == true);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 10
         memset(registers,0, sizeof(registers));
         registers[2] = 4;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == true);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 11
         memset(registers,0, sizeof(registers));
         registers[2] = 8;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == true);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 12
         memset(registers,0, sizeof(registers));
         registers[3] = 1;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == true);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 13
         memset(registers,0, sizeof(registers));
         registers[3] = 2;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == true);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 14
         memset(registers,0, sizeof(registers));
         registers[3] = 4;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == true);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 15
         memset(registers,0, sizeof(registers));
         registers[3] = 8;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == true);
      }

      WHEN("Multiple channels set")
      {
         xt1121Channels xtc;
         
         uint16_t registers[4];// = {0,0,0,0};
         
         //Channel 0 and 1
         memset(registers,0, sizeof(registers));
         registers[0] = 1;
         registers[0] += 2;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == true);
         REQUIRE(xtc.channel(1) == true);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 0 and 2
         memset(registers,0, sizeof(registers));
         registers[0] = 1;
         registers[0] += 4;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == true);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == true);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 0 and 3
         memset(registers,0, sizeof(registers));
         registers[0] = 1;
         registers[0] += 8;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == true);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == true);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 0 and 4
         memset(registers,0, sizeof(registers));
         registers[0] = 1;
         registers[1] = 1;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == true);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == true);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 0 and 7
         memset(registers,0, sizeof(registers));
         registers[0] = 1;
         registers[1] = 8;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == true);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == true);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == false);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 8 and 14
         memset(registers,0, sizeof(registers));
         registers[2] = 1;
         registers[3] = 4;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == false);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == true);
         REQUIRE(xtc.channel(9) == false);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == true);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 0, 9 and 14
         memset(registers,0, sizeof(registers));
         registers[0] = 1;
         registers[2] = 2;
         registers[3] = 4;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == true);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == false);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == true);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == true);
         REQUIRE(xtc.channel(15) == false);
         
         //Channel 0, 7,9 and 14
         memset(registers,0, sizeof(registers));
         registers[0] = 1;
         registers[1] = 8;
         registers[2] = 2;
         registers[3] = 4;
         rv = xtc.readRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(xtc.channel(0) == true);
         REQUIRE(xtc.channel(1) == false);
         REQUIRE(xtc.channel(2) == false);
         REQUIRE(xtc.channel(3) == false);
         REQUIRE(xtc.channel(4) == false);
         REQUIRE(xtc.channel(5) == false);
         REQUIRE(xtc.channel(6) == false);
         REQUIRE(xtc.channel(7) == true);
         REQUIRE(xtc.channel(8) == false);
         REQUIRE(xtc.channel(9) == true);
         REQUIRE(xtc.channel(10) == false);
         REQUIRE(xtc.channel(11) == false);
         REQUIRE(xtc.channel(12) == false);
         REQUIRE(xtc.channel(13) == false);
         REQUIRE(xtc.channel(14) == true);
         REQUIRE(xtc.channel(15) == false);
      }

   }
}

SCENARIO( "Setting registers from channels", "[xtChannels]" )
{
   GIVEN("A set of input registers to send to the device")
   {
      int rv;

      WHEN("Individual channels set, all able to output")
      {
         xt1121Channels xtc;
         
         uint16_t registers[4];
         
         //Channel 0
         xtc.clearAll();
         xtc.setChannel(0);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 1);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 1
         xtc.clearAll();
         xtc.setChannel(1);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 2);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         
         //Channel 2
         xtc.clearAll();
         xtc.setChannel(2);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 4);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 3
         xtc.clearAll();
         xtc.setChannel(3);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 8);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 4
         xtc.clearAll();
         xtc.setChannel(4);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 1);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 5
         xtc.clearAll();
         xtc.setChannel(5);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 2);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 6
         xtc.clearAll();
         xtc.setChannel(6);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 4);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 7
         xtc.clearAll();
         xtc.setChannel(7);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 8);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 8
         xtc.clearAll();
         xtc.setChannel(8);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 1);
         REQUIRE(registers[3] == 0);
         
         //Channel 9
         xtc.clearAll();
         xtc.setChannel(9);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 2);
         REQUIRE(registers[3] == 0);
         
         //Channel 10
         xtc.clearAll();
         xtc.setChannel(10);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 4);
         REQUIRE(registers[3] == 0);
         
         //Channel 11
         xtc.clearAll();
         xtc.setChannel(11);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 8);
         REQUIRE(registers[3] == 0);
         
         //Channel 12
         xtc.clearAll();
         xtc.setChannel(12);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 1);
         
         //Channel 13
         xtc.clearAll();
         xtc.setChannel(13);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 2);
         
         //Channel 14
         xtc.clearAll();
         xtc.setChannel(14);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 4);
         
         //Channel 15
         xtc.clearAll();
         xtc.setChannel(15);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 8);
         
      }
      
      WHEN("Multiple channels set, all able to output")
      {
         xt1121Channels xtc;
         
         uint16_t registers[4];
         
         //Channel 0 and 1
         xtc.clearAll();
         xtc.setChannel(0);
         xtc.setChannel(1);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 3);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 0 and 2
         xtc.clearAll();
         xtc.setChannel(0);
         xtc.setChannel(2);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 5);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 0 and 3
         xtc.clearAll();
         xtc.setChannel(0);
         xtc.setChannel(3);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 9);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 0 and 4
         xtc.clearAll();
         xtc.setChannel(0);
         xtc.setChannel(4);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 1);
         REQUIRE(registers[1] == 1);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 6,9,15
         xtc.clearAll();
         xtc.setChannel(6);
         xtc.setChannel(9);
         xtc.setChannel(15);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 4);
         REQUIRE(registers[2] == 2);
         REQUIRE(registers[3] == 8);
      }
      
      WHEN("Individual channels set, some input only")
      {
         xt1121Channels xtc;
         xtc.setInputOnly(0);
         xtc.setInputOnly(5);
         xtc.setInputOnly(10);
         xtc.setInputOnly(15);
         
         uint16_t registers[4];
         
         //Channel 0 -- input only
         xtc.clearAll();
         xtc.setChannel(0);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 1
         xtc.clearAll();
         xtc.setChannel(1);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 2);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         
         //Channel 2
         xtc.clearAll();
         xtc.setChannel(2);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 4);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 3
         xtc.clearAll();
         xtc.setChannel(3);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 8);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 4
         xtc.clearAll();
         xtc.setChannel(4);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 1);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 5 -- input only
         xtc.clearAll();
         xtc.setChannel(5);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 6
         xtc.clearAll();
         xtc.setChannel(6);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 4);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 7
         xtc.clearAll();
         xtc.setChannel(7);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 8);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 8
         xtc.clearAll();
         xtc.setChannel(8);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 1);
         REQUIRE(registers[3] == 0);
         
         //Channel 9
         xtc.clearAll();
         xtc.setChannel(9);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 2);
         REQUIRE(registers[3] == 0);
         
         //Channel 10 -- input only
         xtc.clearAll();
         xtc.setChannel(10);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 11
         xtc.clearAll();
         xtc.setChannel(11);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 8);
         REQUIRE(registers[3] == 0);
         
         //Channel 12
         xtc.clearAll();
         xtc.setChannel(12);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 1);
         
         //Channel 13
         xtc.clearAll();
         xtc.setChannel(13);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 2);
         
         //Channel 14
         xtc.clearAll();
         xtc.setChannel(14);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 4);
         
         //Channel 15 -- input only
         xtc.clearAll();
         xtc.setChannel(15);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
      }
      
      WHEN("Multiple channels set, some input only")
      {
         xt1121Channels xtc;
         xtc.setInputOnly(0);
         xtc.setInputOnly(5);
         xtc.setInputOnly(10);
         xtc.setInputOnly(15);
         
         uint16_t registers[4];
         
         //Channel 0(input only) and 1
         xtc.clearAll();
         xtc.setChannel(0);
         xtc.setChannel(1);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 2);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 0(input only) and 2
         xtc.clearAll();
         xtc.setChannel(0);
         xtc.setChannel(2);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 4);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 0(input only) and 3
         xtc.clearAll();
         xtc.setChannel(0);
         xtc.setChannel(3);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 8);
         REQUIRE(registers[1] == 0);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 0(input only) and 4
         xtc.clearAll();
         xtc.setChannel(0);
         xtc.setChannel(4);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 1);
         REQUIRE(registers[2] == 0);
         REQUIRE(registers[3] == 0);
         
         //Channel 6,9,15(input only)
         xtc.clearAll();
         xtc.setChannel(6);
         xtc.setChannel(9);
         xtc.setChannel(15);
         
         memset(registers,0, sizeof(registers));
         rv = xtc.setRegisters(registers);
         REQUIRE(rv == 0);
         REQUIRE(registers[0] == 0);
         REQUIRE(registers[1] == 4);
         REQUIRE(registers[2] == 2);
         REQUIRE(registers[3] == 0);
      }
   }
}


} //namespace xtChannels_test
   
