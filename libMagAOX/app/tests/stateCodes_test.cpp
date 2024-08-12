//#define CATCH_CONFIG_MAIN
#include "../../../tests/catch2/catch.hpp"

#include <mx/sys/timeUtils.hpp>

#include "../stateCodes.hpp"


using namespace MagAOX::app;

SCENARIO( "Getting State Strings From Codes", "[stateCodes]" ) 
{
    GIVEN("a valid state code")
    {
        std::string str;

        str = stateCodes::codeText(stateCodes::FAILURE);
        REQUIRE(str == "FAILURE");

        str = stateCodes::codeText(stateCodes::ERROR);
        REQUIRE(str == "ERROR");

        str = stateCodes::codeText(stateCodes::UNINITIALIZED);
        REQUIRE(str == "UNINITIALIZED");

        str = stateCodes::codeText(stateCodes::INITIALIZED);
        REQUIRE(str == "INITIALIZED");

        str = stateCodes::codeText(stateCodes::NODEVICE);
        REQUIRE(str == "NODEVICE");

        str = stateCodes::codeText(stateCodes::POWEROFF);
        REQUIRE(str == "POWEROFF");

        str = stateCodes::codeText(stateCodes::POWERON);
        REQUIRE(str == "POWERON");

        str = stateCodes::codeText(stateCodes::NOTCONNECTED);
        REQUIRE(str == "NOTCONNECTED");

        str = stateCodes::codeText(stateCodes::CONNECTED);
        REQUIRE(str == "CONNECTED");

        str = stateCodes::codeText(stateCodes::LOGGEDIN);
        REQUIRE(str == "LOGGEDIN");

        str = stateCodes::codeText(stateCodes::CONFIGURING);
        REQUIRE(str == "CONFIGURING");

        str = stateCodes::codeText(stateCodes::NOTHOMED);
        REQUIRE(str == "NOTHOMED");

        str = stateCodes::codeText(stateCodes::HOMING);
        REQUIRE(str == "HOMING");

        str = stateCodes::codeText(stateCodes::OPERATING);
        REQUIRE(str == "OPERATING");

        str = stateCodes::codeText(stateCodes::READY);
        REQUIRE(str == "READY");

        str = stateCodes::codeText(stateCodes::SHUTDOWN);
        REQUIRE(str == "SHUTDOWN");
    }
}

SCENARIO( "Getting State Codes From Strings", "[stateCodes]" ) 
{
    GIVEN("a string using stateCodeFast")
    {
        WHEN("valid strings")
        {
            stateCodes::stateCodeT sc;

            sc = stateCodes::str2CodeFast("FAILURE");
            REQUIRE(sc == stateCodes::FAILURE);

            sc = stateCodes::str2CodeFast("ERROR");
            REQUIRE(sc == stateCodes::ERROR);

            sc = stateCodes::str2CodeFast("UNINITIALIZED");
            REQUIRE(sc == stateCodes::UNINITIALIZED);

            sc = stateCodes::str2CodeFast("INITIALIZED");
            REQUIRE(sc == stateCodes::INITIALIZED);

            sc = stateCodes::str2CodeFast("NODEVICE");
            REQUIRE(sc == stateCodes::NODEVICE);

            sc = stateCodes::str2CodeFast("POWEROFF");
            REQUIRE(sc == stateCodes::POWEROFF);

            sc = stateCodes::str2CodeFast("POWERON");
            REQUIRE(sc == stateCodes::POWERON);

            sc = stateCodes::str2CodeFast("NOTCONNECTED");
            REQUIRE(sc == stateCodes::NOTCONNECTED);

            sc = stateCodes::str2CodeFast("CONNECTED");
            REQUIRE(sc == stateCodes::CONNECTED);

            sc = stateCodes::str2CodeFast("LOGGEDIN");
            REQUIRE(sc == stateCodes::LOGGEDIN);

            sc = stateCodes::str2CodeFast("CONFIGURING");
            REQUIRE(sc == stateCodes::CONFIGURING);

            sc = stateCodes::str2CodeFast("NOTHOMED");
            REQUIRE(sc == stateCodes::NOTHOMED);

            sc = stateCodes::str2CodeFast("HOMING");
            REQUIRE(sc == stateCodes::HOMING);

            sc = stateCodes::str2CodeFast("OPERATING");
            REQUIRE(sc == stateCodes::OPERATING);

            sc = stateCodes::str2CodeFast("READY");
            REQUIRE(sc == stateCodes::READY);

            sc = stateCodes::str2CodeFast("SHUTDOWN");
            REQUIRE(sc == stateCodes::SHUTDOWN);

        }

        WHEN("strings too short")
        {
            stateCodes::stateCodeT sc;

            sc = stateCodes::str2CodeFast("CON");
            REQUIRE(sc == -999);

            sc = stateCodes::str2CodeFast("CO");
            REQUIRE(sc == -999);

            sc = stateCodes::str2CodeFast("NOD");
            REQUIRE(sc == -999);

            sc = stateCodes::str2CodeFast("NOT");
            REQUIRE(sc == -999);

            sc = stateCodes::str2CodeFast("NO");
            REQUIRE(sc == -999);

            sc = stateCodes::str2CodeFast("POWER");
            REQUIRE(sc == -999);

            sc = stateCodes::str2CodeFast("POW");
            REQUIRE(sc == -999);

        }
    }

    GIVEN("a string using stateCode")
    {
        WHEN("valid strings")
        {
            stateCodes::stateCodeT sc;

            sc = stateCodes::str2Code("FAILURE");
            REQUIRE(sc == stateCodes::FAILURE);

            sc = stateCodes::str2Code("ERROR");
            REQUIRE(sc == stateCodes::ERROR);

            sc = stateCodes::str2Code("UNINITIALIZED");
            REQUIRE(sc == stateCodes::UNINITIALIZED);

            sc = stateCodes::str2Code("INITIALIZED");
            REQUIRE(sc == stateCodes::INITIALIZED);

            sc = stateCodes::str2Code("NODEVICE");
            REQUIRE(sc == stateCodes::NODEVICE);

            sc = stateCodes::str2Code("POWEROFF");
            REQUIRE(sc == stateCodes::POWEROFF);

            sc = stateCodes::str2Code("POWERON");
            REQUIRE(sc == stateCodes::POWERON);

            sc = stateCodes::str2Code("NOTCONNECTED");
            REQUIRE(sc == stateCodes::NOTCONNECTED);

            sc = stateCodes::str2Code("CONNECTED");
            REQUIRE(sc == stateCodes::CONNECTED);

            sc = stateCodes::str2Code("LOGGEDIN");
            REQUIRE(sc == stateCodes::LOGGEDIN);

            sc = stateCodes::str2Code("CONFIGURING");
            REQUIRE(sc == stateCodes::CONFIGURING);

            sc = stateCodes::str2Code("NOTHOMED");
            REQUIRE(sc == stateCodes::NOTHOMED);

            sc = stateCodes::str2Code("HOMING");
            REQUIRE(sc == stateCodes::HOMING);

            sc = stateCodes::str2Code("OPERATING");
            REQUIRE(sc == stateCodes::OPERATING);

            sc = stateCodes::str2Code("READY");
            REQUIRE(sc == stateCodes::READY);

            sc = stateCodes::str2Code("SHUTDOWN");
            REQUIRE(sc == stateCodes::SHUTDOWN);

        }

        WHEN("invalid strings")
        {
            stateCodes::stateCodeT sc;

            sc = stateCodes::str2Code("FAILUR");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("ERRR");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("UNIITIALIZED");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("INITALIZED");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("NODVICE");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("POEROFF");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("POWRON");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("NOTCNNECTED");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("CONNETED");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("LOGGEIN");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("CONFIURING");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("NOHOMED");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("HOMNG");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("OPETING");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("REDY");
            REQUIRE(sc == -999);

            sc = stateCodes::str2Code("SHTDOWN");
            REQUIRE(sc == -999);

        }
    }
}

