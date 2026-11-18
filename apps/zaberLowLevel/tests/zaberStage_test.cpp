/** \file zaberStage_test.cpp
 * \brief Catch2 tests for the zaberStage in the zaberLowLevel app
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup zaberLowLevel_files
 */

#include "../../../tests/testXWC.hpp"

#include "../zaberLowLevel.hpp"

extern "C"
{
#include "../za_serial.c" //to allow test to compile
}

using namespace MagAOX::app;

namespace libXWCTest
{

/** \addtogroup zaberLowLevel_unit_test
 * \brief Additional unit tests for the zaberLowLevel application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `zaberLowLevel` unit tests.
/** \ingroup zaberLowLevel_unit_test
 */
namespace zaberLowLevelTest
{

/// Verify `zaberStage` classifies replies and decodes warning tokens from ASCII responses.
/**
 * \ingroup zaberLowLevel_unit_test
 */
SCENARIO( "Classifying decoded device messages", "[zaberStage]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVEL_TEST_DOXYGEN_REF
    zaberStage<zaberLowLevel>::isCommandReply( za_reply() );
    zaberStage<zaberLowLevel>::parseWarnings( "" );
    #endif
    // clang-format on

    GIVEN( "A configured stage and decoded ASCII messages" )
    {
        zaberLowLevel zll;

        zaberStage<zaberLowLevel> zstg( &zll );

        REQUIRE( zstg.deviceAddress( 1 ) == 0 );

        za_reply rep{};
        rep.device_address = 1;

        WHEN( "The message is a normal reply for the same device" )
        {
            rep.message_type = '@';

            REQUIRE( zstg.isCommandReply( rep ) == true );
        }

        WHEN( "The message is an alert for the same device" )
        {
            rep.message_type = '!';

            REQUIRE( zstg.isCommandReply( rep ) == false );
        }

        WHEN( "The message is an info message for the same device" )
        {
            rep.message_type = '#';

            REQUIRE( zstg.isCommandReply( rep ) == false );
        }

        WHEN( "The message is a reply for a different device" )
        {
            rep.message_type   = '@';
            rep.device_address = 2;

            REQUIRE( zstg.isCommandReply( rep ) == false );
        }
    }
}

SCENARIO( "Parsing the warnings response", "[zaberStage]" )
{
    GIVEN( "A valid response to the warnings query" )
    {
        int rv;

        WHEN( "Valid response, no warnings" )
        {
            zaberLowLevel zll;

            zaberStage<zaberLowLevel> zstg( &zll );

            std::string tstr = "00";

            rv = zstg.parseWarnings( tstr );

            REQUIRE( rv == 0 );
            REQUIRE( zstg.warningState() == false );
            REQUIRE( zstg.warnFD() == false );
            REQUIRE( zstg.warnFQ() == false );
            REQUIRE( zstg.warnFS() == false );
            REQUIRE( zstg.warnFT() == false );
            REQUIRE( zstg.warnFB() == false );
            REQUIRE( zstg.warnFP() == false );
            REQUIRE( zstg.warnFE() == false );
            REQUIRE( zstg.warnWH() == false );
            REQUIRE( zstg.warnWL() == false );
            REQUIRE( zstg.warnWP() == false );
            REQUIRE( zstg.warnWV() == false );
            REQUIRE( zstg.warnWT() == false );
            REQUIRE( zstg.warnWM() == false );
            REQUIRE( zstg.warnWR() == false );
            REQUIRE( zstg.warnNC() == false );
            REQUIRE( zstg.warnNI() == false );
            REQUIRE( zstg.warnND() == false );
            REQUIRE( zstg.warnNU() == false );
            REQUIRE( zstg.warnNJ() == false );
            REQUIRE( zstg.warnUNK() == false );
        }

        WHEN( "Valid response, one warning" )
        {
            zaberLowLevel zll;

            zaberStage<zaberLowLevel> zstg( &zll );

            std::string tstr = "01 WR";

            rv = zstg.parseWarnings( tstr );

            REQUIRE( rv == 0 );
            REQUIRE( zstg.warningState() == true );
            REQUIRE( zstg.warnFD() == false );
            REQUIRE( zstg.warnFQ() == false );
            REQUIRE( zstg.warnFS() == false );
            REQUIRE( zstg.warnFT() == false );
            REQUIRE( zstg.warnFB() == false );
            REQUIRE( zstg.warnFP() == false );
            REQUIRE( zstg.warnFE() == false );
            REQUIRE( zstg.warnWH() == false );
            REQUIRE( zstg.warnWL() == false );
            REQUIRE( zstg.warnWP() == false );
            REQUIRE( zstg.warnWV() == false );
            REQUIRE( zstg.warnWT() == false );
            REQUIRE( zstg.warnWM() == false );
            REQUIRE( zstg.warnWR() == true );
            REQUIRE( zstg.warnNC() == false );
            REQUIRE( zstg.warnNI() == false );
            REQUIRE( zstg.warnND() == false );
            REQUIRE( zstg.warnNU() == false );
            REQUIRE( zstg.warnNJ() == false );
            REQUIRE( zstg.warnUNK() == false );
        }

        WHEN( "Valid response, five warnings" )
        {
            zaberLowLevel zll;

            zaberStage<zaberLowLevel> zstg( &zll );

            std::string tstr = "05 FD FQ FS FT FB";

            rv = zstg.parseWarnings( tstr );

            REQUIRE( rv == 0 );
            REQUIRE( zstg.warningState() == true );
            REQUIRE( zstg.warnFD() == true );
            REQUIRE( zstg.warnFQ() == true );
            REQUIRE( zstg.warnFS() == true );
            REQUIRE( zstg.warnFT() == true );
            REQUIRE( zstg.warnFB() == true );
            REQUIRE( zstg.warnFP() == false );
            REQUIRE( zstg.warnFE() == false );
            REQUIRE( zstg.warnWH() == false );
            REQUIRE( zstg.warnWL() == false );
            REQUIRE( zstg.warnWP() == false );
            REQUIRE( zstg.warnWV() == false );
            REQUIRE( zstg.warnWT() == false );
            REQUIRE( zstg.warnWM() == false );
            REQUIRE( zstg.warnWR() == false );
            REQUIRE( zstg.warnNC() == false );
            REQUIRE( zstg.warnNI() == false );
            REQUIRE( zstg.warnND() == false );
            REQUIRE( zstg.warnNU() == false );
            REQUIRE( zstg.warnNJ() == false );
            REQUIRE( zstg.warnUNK() == false );
        }

        WHEN( "Valid response, ten warnings" )
        {
            zaberLowLevel zll;

            zaberStage<zaberLowLevel> zstg( &zll );

            std::string tstr = "10 FP FE WH WL WP WV WT WM WR NC";

            rv = zstg.parseWarnings( tstr );

            REQUIRE( rv == 0 );
            REQUIRE( zstg.warningState() == true );
            REQUIRE( zstg.warnFD() == false );
            REQUIRE( zstg.warnFQ() == false );
            REQUIRE( zstg.warnFS() == false );
            REQUIRE( zstg.warnFT() == false );
            REQUIRE( zstg.warnFB() == false );
            REQUIRE( zstg.warnFP() == true );
            REQUIRE( zstg.warnFE() == true );
            REQUIRE( zstg.warnWH() == true );
            REQUIRE( zstg.warnWL() == true );
            REQUIRE( zstg.warnWP() == true );
            REQUIRE( zstg.warnWV() == true );
            REQUIRE( zstg.warnWT() == true );
            REQUIRE( zstg.warnWM() == true );
            REQUIRE( zstg.warnWR() == true );
            REQUIRE( zstg.warnNC() == true );
            REQUIRE( zstg.warnNI() == false );
            REQUIRE( zstg.warnND() == false );
            REQUIRE( zstg.warnNU() == false );
            REQUIRE( zstg.warnNJ() == false );
            REQUIRE( zstg.warnUNK() == false );
        }
        WHEN( "Valid response, 2 warnings" )
        {
            zaberLowLevel zll;

            zaberStage<zaberLowLevel> zstg( &zll );

            std::string tstr = "02 NI ND";

            rv = zstg.parseWarnings( tstr );

            REQUIRE( rv == 0 );
            REQUIRE( zstg.warningState() == true );
            REQUIRE( zstg.warnFD() == false );
            REQUIRE( zstg.warnFQ() == false );
            REQUIRE( zstg.warnFS() == false );
            REQUIRE( zstg.warnFT() == false );
            REQUIRE( zstg.warnFB() == false );
            REQUIRE( zstg.warnFP() == false );
            REQUIRE( zstg.warnFE() == false );
            REQUIRE( zstg.warnWH() == false );
            REQUIRE( zstg.warnWL() == false );
            REQUIRE( zstg.warnWP() == false );
            REQUIRE( zstg.warnWV() == false );
            REQUIRE( zstg.warnWT() == false );
            REQUIRE( zstg.warnWM() == false );
            REQUIRE( zstg.warnWR() == false );
            REQUIRE( zstg.warnNC() == false );
            REQUIRE( zstg.warnNI() == true );
            REQUIRE( zstg.warnND() == true );
            REQUIRE( zstg.warnNU() == false );
            REQUIRE( zstg.warnNJ() == false );
            REQUIRE( zstg.warnUNK() == false );
        }
        WHEN( "Valid response, 3 warnings" )
        {
            zaberLowLevel zll;

            zaberStage<zaberLowLevel> zstg( &zll );

            std::string tstr = "03 NU NJ UN";

            rv = zstg.parseWarnings( tstr );

            REQUIRE( rv == 0 );
            REQUIRE( zstg.warningState() == true );
            REQUIRE( zstg.warnFD() == false );
            REQUIRE( zstg.warnFQ() == false );
            REQUIRE( zstg.warnFS() == false );
            REQUIRE( zstg.warnFT() == false );
            REQUIRE( zstg.warnFB() == false );
            REQUIRE( zstg.warnFP() == false );
            REQUIRE( zstg.warnFE() == false );
            REQUIRE( zstg.warnWH() == false );
            REQUIRE( zstg.warnWL() == false );
            REQUIRE( zstg.warnWP() == false );
            REQUIRE( zstg.warnWV() == false );
            REQUIRE( zstg.warnWT() == false );
            REQUIRE( zstg.warnWM() == false );
            REQUIRE( zstg.warnWR() == false );
            REQUIRE( zstg.warnNC() == false );
            REQUIRE( zstg.warnNI() == false );
            REQUIRE( zstg.warnND() == false );
            REQUIRE( zstg.warnNU() == true );
            REQUIRE( zstg.warnNJ() == true );
            REQUIRE( zstg.warnUNK() == true );
        }

        WHEN( "Truncated response ends before the first warning token" )
        {
            zaberLowLevel zll;

            zaberStage<zaberLowLevel> zstg( &zll );

            std::string tstr = "01 ";

            rv = zstg.parseWarnings( tstr );

            REQUIRE( rv < 0 );
            REQUIRE( zstg.warnUNK() == false );
        }

        WHEN( "Truncated response ends inside a later warning token" )
        {
            zaberLowLevel zll;

            zaberStage<zaberLowLevel> zstg( &zll );

            std::string tstr = "06 6";

            rv = zstg.parseWarnings( tstr );

            REQUIRE( rv < 0 );
            REQUIRE( zstg.warnUNK() == false );
        }
    }
}

} // namespace zaberLowLevelTest

} // namespace libXWCTest
