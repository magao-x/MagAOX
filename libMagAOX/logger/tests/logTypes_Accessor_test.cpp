/** \file logType_Accessor_test.hpp
 * \brief Tests for the log type accessors class
 * \ingroup logger_files
 */

#include "../../../tests/testXWC.hpp"

#include "../generated/logTypes.hpp"

namespace libXWCTest
{

/** \defgroup logger_unit_test libXWC::logger Unit Tests
 * \ingroup unit_test
 */

/// Namespace for XWC::logger tests
/** \ingroup logger_unit_test
 *
 */
namespace loggerTest
{

/** \defgroup logTypes_unit_test log types Unit Tests
 * \ingroup logger_unit_test
 */

/// Namespace for XWC::logger::logType_Accessor tests
/** \ingroup logTypes_unit_test
 *
 */
namespace logTypeAccessorTest
{

/// Call to accessor with invalid member
/**
 * \ingroup logTypes_unit_test
 */
TEST_CASE( "Call to accessor with invalid member", "[libMagAOX::logger::logTypes_Accessor]" )
{
    SECTION( "ao_operator" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::ao_operator::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "config_log" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::config_log::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "git_state" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::git_state::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "indidriver_start" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::indidriver_start::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "indidriver_stop" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::indidriver_stop::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "loop_closed" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::loop_closed::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "loop_open" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::loop_open::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "loop_paused" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::loop_paused::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "observer" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::observer::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "ocam_temps" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::ocam_temps::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "outlet_channel_state" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::outlet_channel_state::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "outlet_state" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::outlet_state::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "pico_channel" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::pico_channel::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "saving_start" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::saving_start::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "saving_state_change" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::saving_state_change::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "saving_stop" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::saving_stop::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "software_log" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::software_log::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "state_change" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::state_change::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "string_log" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::string_log::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_blockgains" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_blockgains::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_chrony_stats" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_chrony_stats::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_chrony_status" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_chrony_status::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_cooler" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_cooler::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_coreloads" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_coreloads::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_coretemps" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_coretemps::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_dmmodes" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_dmmodes::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_dmspeck" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_dmspeck::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_drivetemps" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_drivetemps::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_fgtimings" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_fgtimings::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_fxngen" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_fxngen::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_loopgain" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_loopgain::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_observer" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_observer::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_pi335" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_pi335::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_pico" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_pico::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_pokecenter" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_pokecenter::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_pokeloop" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_pokeloop::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_position" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_position::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_adctrack" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_adctrack::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_rhusb" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_rhusb::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_saving_state" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_saving_state::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_saving" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_saving::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_sparkleclock" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_sparkleclock::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_stage" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_stage::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_stdcam" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_stdcam::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_telcat" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_telcat::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_teldata" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_teldata::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_telenv" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_telenv::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_telpos" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_telpos::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_telsee" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_telsee::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_telvane" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_telvane::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_temps" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_temps::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_usage" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_usage::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "telem_zaber" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::telem_zaber::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "text_log" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::text_log::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "ttmmode_params" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::ttmmod_params::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }

    SECTION( "user_log" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::user_log::getAccessor( "" );

        REQUIRE( lmd.keyword == "" );
        REQUIRE( lmd.comment == "" );
        REQUIRE( lmd.format == "" );
        REQUIRE( lmd.valType == -1 );
        REQUIRE( lmd.metaType == -1 );
        REQUIRE( lmd.accessor == nullptr );
        REQUIRE( lmd.hierarch == true );
    }
    /*
    SECTION("")
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::::getAccessor("");

        REQUIRE(lmd.keyword == "");
        REQUIRE(lmd.comment == "");
        REQUIRE(lmd.format == "");
        REQUIRE(lmd.valType == -1);
        REQUIRE(lmd.metaType == -1);
        REQUIRE(lmd.accessor == nullptr);
        REQUIRE(lmd.hierarch == true);
    }
    */
}

/// string_log has no eventCode or defaultLevel of its own. It is a base reused by the
/// messageT of other log types, for example text_log. So it never gets its own generated
/// per-type test, and nothing else exercises a real, recognized member name through its
/// getAccessor(). This test calls it with the "message" member directly.
/**
 * \ingroup logTypes_unit_test
 */
TEST_CASE( "Call to accessor with a valid member", "[libMagAOX::logger::logTypes_Accessor]" )
{
    SECTION( "string_log" )
    {
        MagAOX::logger::logMetaDetail lmd = MagAOX::logger::string_log::getAccessor( "message" );

        REQUIRE( lmd.keyword == "MESSAGE" );
        REQUIRE( lmd.valType == MagAOX::logger::logMeta::valTypes::String );
        REQUIRE( lmd.accessor != nullptr );
    }
}

} // namespace logTypeAccessorTest
} // namespace loggerTest
} // namespace libXWCTest
