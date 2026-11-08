/** \file po4ao_route_io_test.cpp
 * \brief Execute the production adapter against private fake utilities and ABI records.
 * \ingroup po4ao_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../po4aoRouteIO.hpp"
#include "po4aoRouteTestAbi.hpp"
#include <filesystem>
#include <cstring>
#include <fstream>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

namespace libXWCTest::po4aoTest
{
/// \cond TEST_HARNESS
/// Write a complete offline record.
template <class T>
void record( const std::string &path /**< [in] Private test path. */, const T &value /**< [in] Record. */ )
{
    std::ofstream output( path, std::ios::binary );
    output.write( reinterpret_cast<const char *>( &value ), sizeof( value ) );
}
/// Build a private fake HOWFS loop; test compilation replaces every installed executable/path.
void fixture()
{
    namespace fs = std::filesystem;
    fs::remove_all( ".build/routing" );
    fs::create_directories( ".build/routing/root" );
    fs::create_directories( ".build/routing/shm" );
    fs::create_directories( ".build/routing/tweeter-vispyr-calibs/default" );
    for( const auto *name : { "aol1_modevalWFS", "po4ao_output" } )
        std::ofstream( ".build/routing/shm/" + std::string( name ) + ".im.shm" ) << "source";
    fs::create_symlink( fs::absolute( ".build/routing/shm/aol1_modevalWFS.im.shm" ),
                        ".build/routing/shm/aol1_mfiltINPUT.im.shm" );
    std::ofstream( ".build/routing/shm/aol1_calib_source.txt" )
        << fs::absolute( ".build/routing/tweeter-vispyr-calibs/default" ).string();
    std::ofstream( ".build/routing/shm/aol1_calib_loaded.txt" ) << "before";
    std::ofstream( ".build/routing/root/switch.sh" )
        << "#!/bin/sh\nln -sfn \"$PWD/../shm/$1.im.shm\" ../shm/aol1_mfiltINPUT.im.shm\n";
    std::ofstream( ".build/routing/root/calibrate.sh" ) << "#!/bin/sh\necho after > ../shm/aol1_calib_loaded.txt\n";
    for( const auto *name : { "switch.sh", "calibrate.sh" } )
        fs::permissions( ".build/routing/root/" + std::string( name ), fs::perms::owner_all );
    FUNCTION_PARAMETER_STRUCT_MD md{};
    std::strcpy( md.name, "mfilt-1" );
    md.NBparamMAX = 2;
    md.runpid     = getpid();
    FUNCTION_PARAMETER params[2]{};
    std::strcpy( params[0].keywordfull, "mfilt-1.loopON" );
    params[0].type = FPTYPE_ONOFF;
    std::strcpy( params[1].keywordfull, "mfilt-1.inmval" );
    std::strcpy( params[1].val.string[0], "aol1_mfiltINPUT" );
    std::ofstream out( ".build/routing/shm/mfilt-1.fps.shm", std::ios::binary );
    out.write( reinterpret_cast<char *>( &md ), sizeof( md ) );
    out.write( reinterpret_cast<char *>( params ), sizeof( params ) );
}
/// \endcond

/** \brief Real child execution and completion checks use the required directory and fixed arguments. */
TEST_CASE( "Routing adapter executes private utilities and verifies calibration", "[po4ao][routingIO]" )
{
    fixture();
    MagAOX::app::Po4aoRouteIO io;
    auto                      deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );
    REQUIRE( io.capture().route == "aol1_modevalWFS" );
    io.switchInput( "po4ao_output", deadline );
    REQUIRE( io.route() == "po4ao_output" );
    REQUIRE_NOTHROW( io.applyDefault( deadline ) );
    REQUIRE_THROWS( io.switchInput( "not-a-route", deadline ) );
    io.m_cancelled.store( true );
    REQUIRE_THROWS( io.switchInput( "aol1_modevalWFS", deadline ) );
    REQUIRE( io.route() == "po4ao_output" );
}

/** \brief A changed alias and zero exit status do not establish downstream consumption. */
TEST_CASE( "Routing adapter requires advancing consumption on the correct inode", "[po4ao][routingIO]" )
{
    fixture();
    MagAOX::app::Po4aoRouteIO io;
    PROCESSINFO               info{};
    info.PID             = getpid();
    info.loopstat        = PROCESSINFO_LOOPSTAT_ACTIVE;
    info.triggerstreamID = 1;
    struct stat st{};
    REQUIRE( stat( ".build/routing/shm/aol1_modevalWFS.im.shm", &st ) == 0 );
    const auto path         = ".build/routing/shm/proc.mfilt-1." + std::to_string( getpid() ) + ".shm";
    info.triggerstreaminode = st.st_ino;
    record( path, info );
    REQUIRE_THROWS(
        io.verifyConsumption( "aol1_modevalWFS", std::chrono::steady_clock::now() + std::chrono::milliseconds( 70 ) ) );
    std::thread producer(
        [&]
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
            info.triggerstreamcnt = 10;
            info.loopcnt          = 10;
            record( path, info );
        } );
    std::exception_ptr error;
    try
    {
        io.verifyConsumption( "aol1_modevalWFS", std::chrono::steady_clock::now() + std::chrono::seconds( 1 ) );
    }
    catch( ... )
    {
        error = std::current_exception();
    }
    producer.join();
    REQUIRE_FALSE( error );
}

/** \brief A child timeout terminates its process group and never pretends calibration completed. */
TEST_CASE( "Routing utility failure and timeout remain failures", "[po4ao][routingIO]" )
{
    fixture();
    MagAOX::app::Po4aoRouteIO io;
    io.capture();
    // Exit zero without a fresh marker is not calibration completion.
    std::ofstream( ".build/routing/root/calibrate.sh" ) << "#!/bin/sh\nexit 0\n";
    REQUIRE_THROWS( io.applyDefault( std::chrono::steady_clock::now() + std::chrono::milliseconds( 60 ) ) );
    std::ofstream( ".build/routing/root/calibrate.sh" ) << "#!/bin/sh\nexit 2\n";
    REQUIRE_THROWS( io.applyDefault( std::chrono::steady_clock::now() + std::chrono::seconds( 1 ) ) );
    std::ofstream( ".build/routing/root/calibrate.sh" )
        << "#!/bin/sh\nsleep 5\necho late > ../shm/aol1_calib_loaded.txt\n";
    REQUIRE_THROWS( io.applyDefault( std::chrono::steady_clock::now() + std::chrono::milliseconds( 50 ) ) );
    std::string marker;
    std::ifstream( ".build/routing/shm/aol1_calib_loaded.txt" ) >> marker;
    REQUIRE( marker == "before" );
}
/** \brief Closed-loop, damaged ABI and a wrong consumer inode cannot pass preflight/completion. */
TEST_CASE( "Routing adapter refuses incompatible readbacks", "[po4ao][routingIO]" )
{
    fixture();
    MagAOX::app::Po4aoRouteIO io;
    SECTION( "truncated FPS" )
    {
        std::filesystem::resize_file( ".build/routing/shm/mfilt-1.fps.shm", 1 );
        REQUIRE_THROWS( io.capture() );
    }
    SECTION( "operator loop is closed" )
    {
        std::fstream file( ".build/routing/shm/mfilt-1.fps.shm", std::ios::binary | std::ios::in | std::ios::out );
        FUNCTION_PARAMETER p{};
        file.seekg( sizeof( FUNCTION_PARAMETER_STRUCT_MD ) );
        file.read( reinterpret_cast<char *>( &p ), sizeof( p ) );
        p.fpflag |= FPFLAG_ONOFF;
        file.seekp( sizeof( FUNCTION_PARAMETER_STRUCT_MD ) );
        file.write( reinterpret_cast<char *>( &p ), sizeof( p ) );
        file.close();
        REQUIRE_THROWS( io.capture() );
        REQUIRE( io.route() == "aol1_modevalWFS" );
    }
    SECTION( "wrong inode" )
    {
        PROCESSINFO info{};
        info.PID                = getpid();
        info.loopstat           = PROCESSINFO_LOOPSTAT_ACTIVE;
        info.triggerstreamID    = 1;
        info.triggerstreaminode = 0;
        info.triggerstreamcnt   = 10;
        info.loopcnt            = 10;
        record( ".build/routing/shm/proc.mfilt-1." + std::to_string( getpid() ) + ".shm", info );
        REQUIRE_THROWS( io.verifyConsumption( "aol1_modevalWFS",
                                              std::chrono::steady_clock::now() + std::chrono::milliseconds( 60 ) ) );
    }
}
} // namespace libXWCTest::po4aoTest
