//#define CATCH_CONFIG_MAIN
#include "../../../tests/catch2/catch.hpp"

#include "../indiUtils.hpp"
using namespace MagAOX::app::indi;
   

SCENARIO( "Parsing INDI unique key", "[indiUtils]" ) 
{
    GIVEN("valid keys")
    {
        WHEN("standard dev.prop")
        {
            std::string devName;
            std::string propName;

            int rv = parseIndiKey( devName, propName, "dev.prop" );

            REQUIRE( rv == 0 );
            REQUIRE( devName == "dev" );
            REQUIRE( propName == "prop" );
        }


    }

    GIVEN("invalid keys")
    {
        WHEN("empty")
        {
            std::string devName;
            std::string propName;

            int rv = parseIndiKey( devName, propName, "" );

            REQUIRE( rv == -1 );
        }

        WHEN(". only")
        {
            std::string devName;
            std::string propName;

            int rv = parseIndiKey( devName, propName, "." );

            REQUIRE( rv == -1 );
        }

        WHEN("no .")
        {
            std::string devName;
            std::string propName;

            int rv = parseIndiKey( devName, propName, "nada" );

            REQUIRE( rv == -2 );
        }

        WHEN("dev.")
        {
            std::string devName;
            std::string propName;

            int rv = parseIndiKey( devName, propName, "dev." );

            REQUIRE( rv == -4 );
        }

        WHEN(".prop")
        {
            std::string devName;
            std::string propName;

            int rv = parseIndiKey( devName, propName, ".prop" );

            REQUIRE( rv == -3 );
        }
    }
}


