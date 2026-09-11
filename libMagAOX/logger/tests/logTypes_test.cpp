/** \file logTypes_test.cpp
 * \brief Tests for the formatting and validation branches of individual log types, and
 *        for the unrecognized event code fallback of the code-generated dispatchers.
 *
 * Technique: each test builds a real log entry in memory with logHeader::createLog and
 * then calls msgString(), verify(), or an accessor of the type directly. No files are
 * written. These tests cover branches that the generated per-type tests cannot reach,
 * because the generator uses one fixed dummy value for every bool field.
 * \ingroup logger_files
 */

#include "../../../tests/testXWC.hpp"

#include <sstream>

#include "../generated/logTypes.hpp"
#include "../generated/logVerify.hpp"
#include "../generated/logCodeValid.hpp"
#include "../generated/logStdFormat.hpp"

namespace libXWCTest
{

namespace loggerTest
{

/** \defgroup logTypes_formatting_unit_test log type formatting Unit Tests
 * \ingroup logger_unit_test
 */

/// Namespace for XWC::logger log type formatting/validation tests
/** \ingroup logTypes_formatting_unit_test
 *
 */
namespace logTypesTest
{

/// Builds a full log entry for TYPE from msg. Returns both the buffer and the message
/// length that verify() and msgString() need. This is the same construction the generated
/// per-type tests use, factored out here for these supplementary branch coverage checks.

/// The timestamp every log in this file is created with. 2024-11-21 08:00:00 UTC, an
/// arbitrary fixed time. Nothing here depends on the value.
static const time_t kLogTime = 1732176000;

template <class TYPE>
std::pair<flatlogs::bufferPtrT, flatlogs::msgLenT> makeLog( const typename TYPE::messageT &msg )
{
    flatlogs::msgLenT    len = TYPE::length( msg );
    flatlogs::bufferPtrT buf;
    flatlogs::logHeader::createLog<TYPE>(
        buf, flatlogs::timespecX( kLogTime, 0 ), msg, flatlogs::logPrio::LOG_TELEM );
    return { buf, len };
}

/// logCodeValid, logVerify, logStdFormat, eventCode, and eventCodeName each dispatch on
/// an event code or name through a switch or if chain with one case per real log type.
/// The generated tests exercise those cases one type at a time. Each dispatcher also has
/// its own fallback for an event code or name with no matching type. That fallback is
/// checked directly here because no per-type test can reach it.
/**
 * \ingroup logTypes_formatting_unit_test
 */
TEST_CASE( "logCodeValid, logVerify, logStdFormat, eventCode, and eventCodeName fall back "
           "gracefully for an unrecognized event code",
           "[libMagAOX::logger::logCodes]" )
{
    const flatlogs::eventCodeT unknownCode = 60999; // arbitrary. Above every generated code and below UNKNOWN, which is 65535 and is a real value

    SECTION( "logCodeValid reports the code as not valid" )
    {
        REQUIRE( MagAOX::logger::logCodeValid( unknownCode ) == false );
    }

    SECTION( "logVerify reports failure without touching the buffer" )
    {
        flatlogs::bufferPtrT buf;
        REQUIRE( MagAOX::logger::logVerify( unknownCode, buf, 0 ) == false );
    }

    SECTION( "logStdFormat writes an unknown-log-type message instead of formatting anything" )
    {
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<MagAOX::logger::git_state>( buf,
                                                                    flatlogs::timespecX( kLogTime, 0 ),
                                                                    MagAOX::logger::git_state::messageT( "repo", "sha", false ),
                                                                    flatlogs::logPrio::LOG_NOTICE );
        flatlogs::logHeader::eventCode( buf, unknownCode );

        std::ostringstream oss;
        MagAOX::logger::logStdFormat( oss, buf );

        REQUIRE( oss.str().find( "Unknown log type" ) != std::string::npos );
    }

    SECTION( "the short/min/json format dispatchers have the same unknown-log-type fallback" )
    {
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<MagAOX::logger::git_state>( buf,
                                                                    flatlogs::timespecX( kLogTime, 0 ),
                                                                    MagAOX::logger::git_state::messageT( "repo", "sha", false ),
                                                                    flatlogs::logPrio::LOG_NOTICE );
        flatlogs::logHeader::eventCode( buf, unknownCode );

        std::ostringstream ossShort, ossMin, ossJson;
        MagAOX::logger::logShortStdFormat( ossShort, "testapp", buf );
        MagAOX::logger::logMinStdFormat( ossMin, buf );
        MagAOX::logger::logJsonFormat( ossJson, buf );

        REQUIRE( ossShort.str().find( "Unknown log type" ) != std::string::npos );
        REQUIRE( ossMin.str().find( "Unknown log type" ) != std::string::npos );
        REQUIRE( ossJson.str().find( "Unknown log type" ) != std::string::npos );
    }

    SECTION( "logShortStdFormat truncates application names longer than 15 characters" )
    {
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<MagAOX::logger::git_state>( buf,
                                                                    flatlogs::timespecX( kLogTime, 0 ),
                                                                    MagAOX::logger::git_state::messageT( "repo", "sha", false ),
                                                                    flatlogs::logPrio::LOG_NOTICE );

        std::ostringstream oss;
        MagAOX::logger::logShortStdFormat( oss, "a_very_long_application_name", buf );
        REQUIRE( !oss.str().empty() );

        // A name of exactly 16 characters exercises the early exit of the truncation loop.
        std::ostringstream oss2;
        MagAOX::logger::logShortStdFormat( oss2, "sixteen_chars_ab", buf );
        REQUIRE( !oss2.str().empty() );
    }

    SECTION( "flatbuffer_log::msgJSON reports a genuinely invalid binary schema" )
    {
        flatlogs::bufferPtrT buf;
        flatlogs::logHeader::createLog<MagAOX::logger::git_state>( buf,
                                                                    flatlogs::timespecX( kLogTime, 0 ),
                                                                    MagAOX::logger::git_state::messageT( "repo", "sha", false ),
                                                                    flatlogs::logPrio::LOG_NOTICE );

        // A few bytes of garbage is not a valid flatbuffers binary schema. So
        // Parser::Deserialize() genuinely fails and the error branch runs. It returns
        // empty JSON rather than proceeding into GenText, which would crash.
        const uint8_t notASchema[4] = { 0xde, 0xad, 0xbe, 0xef };
        std::string out = MagAOX::logger::git_state::msgJSON( flatlogs::logHeader::messageBuffer( buf ),
                                                              flatlogs::logHeader::msgLen( buf ),
                                                              notASchema,
                                                              sizeof( notASchema ) );
        REQUIRE( out == "{}" );
    }

    SECTION( "eventCode maps an unrecognized name to UNKNOWN" )
    {
        REQUIRE( MagAOX::logger::eventCode( "not_a_real_log_type" ) == MagAOX::logger::eventCodes::UNKNOWN );
    }

    SECTION( "eventCodeName maps an unrecognized code to a placeholder string" )
    {
        REQUIRE( MagAOX::logger::eventCodeName( unknownCode ) == "unknown event code" );
    }
}

/// The per-type generated tests always construct bool-gated fields as true, because that
/// is the dummy value the generator uses for every bool field. So the false side of the
/// msgString() branching in each type is never reached there. These are genuine type
/// specific formatting branches, not generated dispatch code. So they are covered here
/// directly rather than by complicating the single bool dummy value of the generator.
/**
 * \ingroup logTypes_formatting_unit_test
 */
TEST_CASE( "telem_dmspeck, telem_sparkleclock, telem_fxngen, and telem_poltrack format "
           "their false/off/not-tracking states",
           "[libMagAOX::logger::telem_dmspeck]" )
{
    SECTION( "a telem_dmspeck message that is not modulating" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_dmspeck>(
            MagAOX::logger::telem_dmspeck::messageT( false, false, 1.0f, { 1.0f }, { 2.0f }, { 3.0f }, { true } ) ); // distinct placeholder values
        REQUIRE( MagAOX::logger::telem_dmspeck::msgString( flatlogs::logHeader::messageBuffer( buf ), len ) ==
                 "[speckles] not modulating" );
    }

    SECTION( "a telem_dmspeck message that is modulating by frequency, not trigger" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_dmspeck>(
            MagAOX::logger::telem_dmspeck::messageT( true, false, 12.5f, { 1.0f }, { 2.0f }, { 3.0f }, { true, false } ) ); // 12.5 is checked in the output below
        std::string s = MagAOX::logger::telem_dmspeck::msgString( flatlogs::logHeader::messageBuffer( buf ), len );
        REQUIRE( s.find( "at 12.5" ) != std::string::npos );
        REQUIRE( s.find( "+-" ) != std::string::npos );
    }

    SECTION( "a telem_sparkleclock message that is not modulating" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_sparkleclock>(
            MagAOX::logger::telem_sparkleclock::messageT( false, false, 1.0f, 2.0f, { 1.0f }, 3.0f, 4.0f ) ); // distinct placeholder values
        REQUIRE( MagAOX::logger::telem_sparkleclock::msgString( flatlogs::logHeader::messageBuffer( buf ), len ) ==
                 "[sparkleclock] not modulating" );
    }

    SECTION( "a telem_sparkleclock message that is modulating by frequency, not trigger" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_sparkleclock>(
            MagAOX::logger::telem_sparkleclock::messageT( true, false, 12.5f, 2.0f, { 1.0f }, 3.0f, 4.0f ) ); // 12.5 is checked in the output below
        std::string s = MagAOX::logger::telem_sparkleclock::msgString( flatlogs::logHeader::messageBuffer( buf ), len );
        REQUIRE( s.find( "at 12.5" ) != std::string::npos );
    }

    SECTION( "a telem_fxngen message with both channels' sync off" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_fxngen>( MagAOX::logger::telem_fxngen::messageT(
            1, 1.0, 1.0, 1.0, 1.0, 0, 1, 1.0, 1.0, 1.0, 1.0, 0, (uint8_t)0, (uint8_t)0, 1.0, 1.0 ) );
        std::string s = MagAOX::logger::telem_fxngen::msgString( flatlogs::logHeader::messageBuffer( buf ), len );
        REQUIRE( s.find( "SYNC OFF" ) != std::string::npos );
    }

    SECTION( "a telem_poltrack message that is not tracking" )
    {
        auto [buf, len] =
            makeLog<MagAOX::logger::telem_poltrack>( MagAOX::logger::telem_poltrack::messageT( 1.0f, 1.0f, "home", false ) ); // placeholder values
        std::string s = MagAOX::logger::telem_poltrack::msgString( flatlogs::logHeader::messageBuffer( buf ), len );
        REQUIRE( s.find( "NONE" ) != std::string::npos );
    }
}

/// ttmmod_params, outlet_state, and outlet_channel_state each map a small integer state
/// to a name in msgString(). Every named value is formatted once and checked.
/**
 * \ingroup logTypes_formatting_unit_test
 */
TEST_CASE( "ttmmod_params and outlet/outlet-channel state format every named state value",
           "[libMagAOX::logger::ttmmod_params]" )
{
    SECTION( "ttmmod_params in each of its named modState values" )
    {
        struct
        {
            uint8_t     state;
            std::string expect;
        } cases[] = { { 0, "OFF" }, { 1, "REST" }, { 2, "INT" }, { 3, "SET" } };

        for( auto &c : cases )
        {
            auto [buf, len] =
                makeLog<MagAOX::logger::ttmmod_params>( MagAOX::logger::ttmmod_params::messageT( c.state, 0.0, 0.0, 0.0, 0.0 ) );
            REQUIRE( MagAOX::logger::ttmmod_params::msgString( flatlogs::logHeader::messageBuffer( buf ), len ) == c.expect );
            // modState() duplicates the state name mapping of msgString() for the
            // metadata accessor system rather than delegating to it. So it is exercised
            // directly too.
            REQUIRE( MagAOX::logger::ttmmod_params::modState( flatlogs::logHeader::messageBuffer( buf ) ) == c.expect );
        }

        auto [buf, len] = makeLog<MagAOX::logger::ttmmod_params>( MagAOX::logger::ttmmod_params::messageT( 4, 10.0, 0.5, 0.0, 0.0 ) ); // state 4 is modulating. 10.0 and 0.5 are checked in the output below
        std::string s = MagAOX::logger::ttmmod_params::msgString( flatlogs::logHeader::messageBuffer( buf ), len );
        REQUIRE( s.find( "MOD" ) != std::string::npos );
        REQUIRE( s.find( "Freq: 10" ) != std::string::npos );
        REQUIRE( s.find( "Rad: 0.5" ) != std::string::npos );
        REQUIRE( MagAOX::logger::ttmmod_params::modState( flatlogs::logHeader::messageBuffer( buf ) ) == "MOD" );
    }

    SECTION( "outlet_state in each of its named state values" )
    {
        struct
        {
            uint8_t     state;
            std::string expect;
        } cases[] = { { 0, "OFF" }, { 1, "INT" }, { 2, "ON" } };

        for( auto &c : cases )
        {
            auto [buf, len] = makeLog<MagAOX::logger::outlet_state>( MagAOX::logger::outlet_state::messageT( 1, c.state ) );
            std::string s = MagAOX::logger::outlet_state::msgString( flatlogs::logHeader::messageBuffer( buf ), len );
            REQUIRE( s.find( c.expect ) != std::string::npos );
        }
    }

    SECTION( "outlet_channel_state in each of its named state values" )
    {
        struct
        {
            uint8_t     state;
            std::string expect;
        } cases[] = { { 0, "OFF" }, { 1, "INT" }, { 2, "ON" } };

        for( auto &c : cases )
        {
            auto [buf, len] =
                makeLog<MagAOX::logger::outlet_channel_state>( MagAOX::logger::outlet_channel_state::messageT( "ch1", c.state ) );
            std::string s = MagAOX::logger::outlet_channel_state::msgString( flatlogs::logHeader::messageBuffer( buf ), len );
            REQUIRE( s.find( c.expect ) != std::string::npos );
        }
    }
}

/// telem_blockgains prints a placeholder instead of a value when a gain vector and its
/// constant flag vector have different lengths. Each of the three vector pairs is tried.
/**
 * \ingroup logTypes_formatting_unit_test
 */
TEST_CASE( "telem_blockgains reports a placeholder when a gain vector and its "
           "constant-flag vector have mismatched lengths",
           "[libMagAOX::logger::telem_blockgains]" )
{
    SECTION( "gains and gains_constant of different lengths" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_blockgains>( MagAOX::logger::telem_blockgains::messageT(
            { 1.0f, 2.0f }, { 1 }, std::vector<float>{}, std::vector<uint8_t>{}, std::vector<float>{}, std::vector<uint8_t>{} ) );
        std::string s = MagAOX::logger::telem_blockgains::msgString( flatlogs::logHeader::messageBuffer( buf ), len );
        REQUIRE( s.find( "1.0" ) != std::string::npos );
    }

    SECTION( "mcs and mcs_constant of different lengths" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_blockgains>( MagAOX::logger::telem_blockgains::messageT(
            std::vector<float>{}, std::vector<uint8_t>{}, { 1.0f, 2.0f }, { 1 }, std::vector<float>{}, std::vector<uint8_t>{} ) );
        MagAOX::logger::telem_blockgains::msgString( flatlogs::logHeader::messageBuffer( buf ), len );
    }

    SECTION( "lims and lims_constant of different lengths" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_blockgains>( MagAOX::logger::telem_blockgains::messageT(
            std::vector<float>{}, std::vector<uint8_t>{}, std::vector<float>{}, std::vector<uint8_t>{}, { 1.0f, 2.0f }, { 1 } ) );
        MagAOX::logger::telem_blockgains::msgString( flatlogs::logHeader::messageBuffer( buf ), len );
    }
}

/// telem_observer has a hand-written verify() that rejects a truncated buffer and any
/// string field that contains a non-printable character. Each string field is tried in
/// turn.
/**
 * \ingroup logTypes_formatting_unit_test
 */
TEST_CASE( "telem_observer's verify() catches malformed buffers and non-printable field content",
           "[libMagAOX::logger::telem_observer]" )
{
    SECTION( "a truncated buffer" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_observer>(
            MagAOX::logger::telem_observer::messageT( "observer@example.com", "observer", true, "target", "operator@example.com" ) ); // valid placeholder fields
        REQUIRE( MagAOX::logger::telem_observer::verify( buf, 1 ) == false );
    }

    SECTION( "a non-printable character in each string field in turn" )
    {
        std::string bad( 1, '\x01' );

        auto [buf1, len1] = makeLog<MagAOX::logger::telem_observer>(
            MagAOX::logger::telem_observer::messageT( bad, "observer", true, "target", "operator@example.com" ) );
        REQUIRE( MagAOX::logger::telem_observer::verify( buf1, len1 ) == false );

        auto [buf2, len2] = makeLog<MagAOX::logger::telem_observer>(
            MagAOX::logger::telem_observer::messageT( "observer@example.com", bad, true, "target", "operator@example.com" ) );
        REQUIRE( MagAOX::logger::telem_observer::verify( buf2, len2 ) == false );

        auto [buf3, len3] = makeLog<MagAOX::logger::telem_observer>(
            MagAOX::logger::telem_observer::messageT( "observer@example.com", "observer", true, bad, "operator@example.com" ) );
        REQUIRE( MagAOX::logger::telem_observer::verify( buf3, len3 ) == false );

        auto [buf4, len4] = makeLog<MagAOX::logger::telem_observer>(
            MagAOX::logger::telem_observer::messageT( "observer@example.com", "observer", true, "target", bad ) );
        REQUIRE( MagAOX::logger::telem_observer::verify( buf4, len4 ) == false );
    }
}

/// telem_stdcam maps small integer fields to state names in msgString(). Each section
/// formats one combination and also checks the matching accessor function.
/**
 * \ingroup logTypes_formatting_unit_test
 */
TEST_CASE( "telem_stdcam formats every named shutter/sync/crop/led state", "[libMagAOX::logger::telem_stdcam]" )
{
    // Builds a message with placeholder values for the fields this test does not care
    // about.
    auto build = []( int8_t shutterState, uint8_t synchro, float vshift, uint8_t cropMode, int8_t led )
    {
        return MagAOX::logger::telem_stdcam::messageT( "mode", 0.0f, 0.0f, 1, 1, 1, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                        (uint8_t)0, (uint8_t)0, "", "", shutterState, synchro, vshift, cropMode,
                                                        "", "", "", led );
    };

    SECTION( "shutter state unknown (-1), sync off, vshift at zero, crop off (-1), led off (-1)" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_stdcam>( build( -1, 0, 0.0f, (uint8_t)-1, -1 ) );
        void       *msgBuffer = flatlogs::logHeader::messageBuffer( buf );
        std::string s         = MagAOX::logger::telem_stdcam::msgString( msgBuffer, len );
        REQUIRE( s.find( "UNKN" ) != std::string::npos );
        REQUIRE( s.find( "Sync: OFF" ) != std::string::npos );
        REQUIRE( s.find( "vshift: ---" ) != std::string::npos );
        REQUIRE( s.find( "crop: ---" ) != std::string::npos );
        REQUIRE( s.find( "led: ---" ) != std::string::npos );
        // shutterState() duplicates the shutter state name mapping of msgString() for the
        // metadata accessor system rather than delegating to it. So it is exercised
        // directly.
        REQUIRE( MagAOX::logger::telem_stdcam::shutterState( msgBuffer ) == "UNKNOWN" );
    }

    SECTION( "shutter shut (0), crop on (1), led on (1)" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_stdcam>( build( 0, 1, 1.0f, 1, 1 ) );
        void       *msgBuffer = flatlogs::logHeader::messageBuffer( buf );
        std::string s         = MagAOX::logger::telem_stdcam::msgString( msgBuffer, len );
        REQUIRE( s.find( "SHUT" ) != std::string::npos );
        REQUIRE( s.find( "crop: ON" ) != std::string::npos );
        REQUIRE( s.find( "led: ON" ) != std::string::npos );
        REQUIRE( MagAOX::logger::telem_stdcam::shutterState( msgBuffer ) == "SHUT" );
        // cropMode() and led() likewise duplicate the on and off mapping of msgString()
        // for the metadata accessor system as their own bool-valued accessors.
        REQUIRE( MagAOX::logger::telem_stdcam::cropMode( msgBuffer ) == true );
        REQUIRE( MagAOX::logger::telem_stdcam::led( msgBuffer ) == true );
    }

    SECTION( "shutter open (1)" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_stdcam>( build( 1, 1, 1.0f, 0, 0 ) );
        void       *msgBuffer = flatlogs::logHeader::messageBuffer( buf );
        std::string s         = MagAOX::logger::telem_stdcam::msgString( msgBuffer, len );
        REQUIRE( s.find( "OPEN" ) != std::string::npos );
        REQUIRE( MagAOX::logger::telem_stdcam::shutterState( msgBuffer ) == "OPEN" );
    }
}

/// The measuring field of telem_pokecenter and telem_pokeloop is a uint8_t, not a bool.
/// So the fixed dummy value of the generator does not cover it at all. The field gates
/// their own constructors. When measuring is 0 the constructor short-circuits and skips
/// serializing the rest of the fields.
/**
 * \ingroup logTypes_formatting_unit_test
 */
TEST_CASE( "telem_pokecenter and telem_pokeloop format every named measuring state",
           "[libMagAOX::logger::telem_pokecenter]" )
{
    SECTION( "telem_pokecenter not measuring (multi-vector constructor)" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_pokecenter>( MagAOX::logger::telem_pokecenter::messageT(
            (uint8_t)0, 1.0f, 2.0f, std::vector<float>{ 3.0f }, std::vector<float>{ 4.0f } ) );
        REQUIRE( MagAOX::logger::telem_pokecenter::measuring( flatlogs::logHeader::messageBuffer( buf ) ) == false );
        REQUIRE( MagAOX::logger::telem_pokecenter::msgString( flatlogs::logHeader::messageBuffer( buf ), len ) ==
                 "not measuring" );
    }

    SECTION( "telem_pokecenter not measuring (combined-vector constructor)" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_pokecenter>(
            MagAOX::logger::telem_pokecenter::messageT( (uint8_t)0, 1.0f, 2.0f, std::vector<float>{ 3.0f, 4.0f } ) );
        REQUIRE( MagAOX::logger::telem_pokecenter::msgString( flatlogs::logHeader::messageBuffer( buf ), len ) ==
                 "not measuring" );
    }

    SECTION( "telem_pokecenter measuring a single poke, with mismatched poke_x/poke_y lengths" )
    {
        auto [buf, len] = makeLog<MagAOX::logger::telem_pokecenter>( MagAOX::logger::telem_pokecenter::messageT(
            (uint8_t)1, 1.0f, 2.0f, std::vector<float>{ 3.0f, 4.0f }, std::vector<float>{ 5.0f } ) );
        std::string s = MagAOX::logger::telem_pokecenter::msgString( flatlogs::logHeader::messageBuffer( buf ), len );
        REQUIRE( s.find( "single" ) != std::string::npos );
        REQUIRE( s.find( "[poke-avg] ? [pokes] ?" ) != std::string::npos );
    }

    SECTION( "telem_pokeloop not measuring" )
    {
        auto [buf, len] =
            makeLog<MagAOX::logger::telem_pokeloop>( MagAOX::logger::telem_pokeloop::messageT( (uint8_t)0, 1.0f, 2.0f, 3ULL ) ); // distinct placeholder values
        REQUIRE( MagAOX::logger::telem_pokeloop::measuring( flatlogs::logHeader::messageBuffer( buf ) ) == false );
        REQUIRE( MagAOX::logger::telem_pokeloop::msgString( flatlogs::logHeader::messageBuffer( buf ), len ) ==
                 "[pokeloop] not measuring" );
    }
}

} // namespace logTypesTest
} // namespace loggerTest
} // namespace libXWCTest
