
#include "../../tests/catch2/catch.hpp"

#include "../logsurgeon.hpp"

TEST_CASE( "configuring logsurgeon", "[logsurgeon::config]" )
{
    SECTION( "default construction" )
    {
        logsurgeon ls;

        REQUIRE( ls.fname() == "" );
        REQUIRE( ls.checkOnly() == false );
    }

    SECTION( "default configuration without setting file" )
    {
        logsurgeon ls;

        std::vector<std::string> sargv( { "lstest" } );

        int                 argc = sargv.size();
        std::vector<char *> argv( argc + 1, nullptr );

        for( size_t n = 0; n < sargv.size(); ++n )
        {
            argv[n] = sargv[n].data();
        }

        int rv = ls.main( argc, argv.data() );

        REQUIRE( ls.fname() == "" );
        REQUIRE( ls.checkOnly() == false );
        REQUIRE( rv == logsurgeon::file_not_specified ); // fails b/c no file specified
    }

    SECTION( "setting file with -F on CL" )
    {
        logsurgeon ls;

        std::vector<std::string> sargv( { "lstest", "-F/tmp/nologtotest.binlog" } );

        int                 argc = sargv.size();
        std::vector<char *> argv( argc + 1, nullptr );

        for( size_t n = 0; n < sargv.size(); ++n )
        {
            argv[n] = sargv[n].data();
        }

        int rv = ls.main( argc, argv.data() );

        REQUIRE( ls.fname() == "/tmp/nologtotest.binlog" );
        REQUIRE( ls.checkOnly() == false );
        REQUIRE( rv == logsurgeon::file_not_found );
    }

    SECTION( "setting file with --file on CL" )
    {
        logsurgeon ls;

        std::vector<std::string> sargv( { "lstest", "--file=/tmp/nologtotest.binlog" } );

        int                 argc = sargv.size();
        std::vector<char *> argv( argc + 1, nullptr );

        for( size_t n = 0; n < sargv.size(); ++n )
        {
            argv[n] = sargv[n].data();
        }

        int rv = ls.main( argc, argv.data() );

        REQUIRE( ls.fname() == "/tmp/nologtotest.binlog" );
        REQUIRE( ls.checkOnly() == false );
        REQUIRE( rv == logsurgeon::file_not_found );
    }

    SECTION( "setting file with -F and checkOnly with -C on CL" )
    {
        logsurgeon ls;

        std::vector<std::string> sargv( { "lstest", "-F/tmp/nologtotest.binlog", "-C" } );

        int                 argc = sargv.size();
        std::vector<char *> argv( argc + 1, nullptr );

        for( size_t n = 0; n < sargv.size(); ++n )
        {
            argv[n] = sargv[n].data();
        }

        int rv = ls.main( argc, argv.data() );

        REQUIRE( ls.fname() == "/tmp/nologtotest.binlog" );
        REQUIRE( ls.checkOnly() == true );
        REQUIRE( rv == logsurgeon::file_not_found );
    }

    SECTION( "setting file with -F and checkOnly with --check-only on CL" )
    {
        logsurgeon ls;

        std::vector<std::string> sargv( { "lstest", "-F/tmp/nologtotest.binlog", "--check-only" } );

        int                 argc = sargv.size();
        std::vector<char *> argv( argc + 1, nullptr );

        for( size_t n = 0; n < sargv.size(); ++n )
        {
            argv[n] = sargv[n].data();
        }

        int rv = ls.main( argc, argv.data() );

        REQUIRE( ls.fname() == "/tmp/nologtotest.binlog" );
        REQUIRE( ls.checkOnly() == true );
        REQUIRE( rv == logsurgeon::file_not_found );
    }

    SECTION( "setting file with config file" )
    {
        mx::app::writeConfigFile(
            "/tmp/logsurgeon_config_test.conf", { "" }, { "file" }, { "/tmp/nologtotest.binlog" } );
        logsurgeon ls;

        std::vector<std::string> sargv( { "lstest", "-c", "/tmp/logsurgeon_config_test.conf" } );

        int                 argc = sargv.size();
        std::vector<char *> argv( argc + 1, nullptr );

        for( size_t n = 0; n < sargv.size(); ++n )
        {
            argv[n] = sargv[n].data();
        }

        int rv = ls.main( argc, argv.data() );

        REQUIRE( ls.fname() == "/tmp/nologtotest.binlog" );
        REQUIRE( ls.checkOnly() == false );
        REQUIRE( rv == logsurgeon::file_not_found );
    }

    SECTION( "setting file and checkOnly to false with config file" )
    {
        mx::app::writeConfigFile( "/tmp/logsurgeon_config_test.conf",
                                  { "", "" },
                                  { "file", "check-only" },
                                  { "/tmp/nologtotest.binlog", "false" } );
        logsurgeon ls;

        std::vector<std::string> sargv( { "lstest", "-c", "/tmp/logsurgeon_config_test.conf" } );

        int                 argc = sargv.size();
        std::vector<char *> argv( argc + 1, nullptr );

        for( size_t n = 0; n < sargv.size(); ++n )
        {
            argv[n] = sargv[n].data();
        }

        int rv = ls.main( argc, argv.data() );

        REQUIRE( ls.fname() == "/tmp/nologtotest.binlog" );
        REQUIRE( ls.checkOnly() == false );
        REQUIRE( rv == logsurgeon::file_not_found );
    }

    SECTION( "setting file and checkOnly to true with config file" )
    {
        mx::app::writeConfigFile( "/tmp/logsurgeon_config_test.conf",
                                  { "", "" },
                                  { "file", "check-only" },
                                  { "/tmp/nologtotest.binlog", "true" } );
        logsurgeon ls;

        std::vector<std::string> sargv( { "lstest", "-c", "/tmp/logsurgeon_config_test.conf" } );

        int                 argc = sargv.size();
        std::vector<char *> argv( argc + 1, nullptr );

        for( size_t n = 0; n < sargv.size(); ++n )
        {
            argv[n] = sargv[n].data();
        }

        int rv = ls.main( argc, argv.data() );

        REQUIRE( ls.fname() == "/tmp/nologtotest.binlog" );
        REQUIRE( ls.checkOnly() == true );
        REQUIRE( rv == logsurgeon::file_not_found );
    }

    SECTION( "setting file and checkOnly to false with config file, overriden on CLI with -C" )
    {
        mx::app::writeConfigFile( "/tmp/logsurgeon_config_test.conf",
                                  { "", "" },
                                  { "file", "check-only" },
                                  { "/tmp/nologtotest.binlog", "false" } );
        logsurgeon ls;

        std::vector<std::string> sargv( { "lstest", "-c", "/tmp/logsurgeon_config_test.conf", "-C" } );

        int                 argc = sargv.size();
        std::vector<char *> argv( argc + 1, nullptr );

        for( size_t n = 0; n < sargv.size(); ++n )
        {
            argv[n] = sargv[n].data();
        }

        int rv = ls.main( argc, argv.data() );

        REQUIRE( ls.fname() == "/tmp/nologtotest.binlog" );
        REQUIRE( ls.checkOnly() == true );
        REQUIRE( rv == logsurgeon::file_not_found );
    }

    SECTION( "setting file and checkOnly to false with config file, overriden on CLI with --check-only" )
    {
        mx::app::writeConfigFile( "/tmp/logsurgeon_config_test.conf",
                                  { "", "" },
                                  { "file", "check-only" },
                                  { "/tmp/nologtotest.binlog", "false" } );
        logsurgeon ls;

        std::vector<std::string> sargv( { "lstest", "-c", "/tmp/logsurgeon_config_test.conf", "--check-only" } );

        int                 argc = sargv.size();
        std::vector<char *> argv( argc + 1, nullptr );

        for( size_t n = 0; n < sargv.size(); ++n )
        {
            argv[n] = sargv[n].data();
        }

        int rv = ls.main( argc, argv.data() );

        REQUIRE( ls.fname() == "/tmp/nologtotest.binlog" );
        REQUIRE( ls.checkOnly() == true );
        REQUIRE( rv == logsurgeon::file_not_found );
    }
}
