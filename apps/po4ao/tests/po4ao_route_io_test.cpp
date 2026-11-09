/** \file po4ao_route_io_test.cpp
 * \brief Execute the production adapter against private fake utilities and ABI records.
 * \ingroup po4ao_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../po4aoRouting.hpp"
#include "po4aoRouteTestAbi.hpp"
#include <filesystem>
#include <cstring>
#include <fstream>
#include <thread>
#include <future>
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
/// Publish a private FPS record and Linux-format process identity; no instrument paths are used.
void writerRecord( bool running /**< [in] Whether the fake writer is running. */,
                   long generation         = 1 /**< [in] Fake process start ticks. */,
                   long metadataGeneration = 1 /**< [in] Independent FPS start timestamp. */ )
{
    FUNCTION_PARAMETER_STRUCT_MD md{};
    std::strcpy( md.name, "wfs2cmodeval-1" );
    md.NBparamMAX             = 1;
    md.runpid                 = running ? getpid() : 0;
    md.runpidstarttime.tv_sec = metadataGeneration;
    const auto proc           = std::string( ".build/routing/proc/" ) + std::to_string( getpid() );
    std::filesystem::create_directories( proc );
    std::ofstream process( proc + "/stat" );
    process << getpid() << " (fake writer) S";
    for( int field = 4; field < 22; ++field )
        process << " 0";
    process << " " << generation << " 0\n";
    std::ofstream( proc + "/cmdline" ) << "milk-wfs2cmodeval-1";
    const std::string path = ".build/routing/shm/wfs2cmodeval-1.fps.shm";
    std::ofstream     fps( path, std::ios::binary );
    fps.write( reinterpret_cast<const char *>( &md ), sizeof( md ) );
    FUNCTION_PARAMETER payload{};
    fps.write( reinterpret_cast<const char *>( &payload ), sizeof( payload ) );
}
/// Build a private fake HOWFS loop; test compilation replaces every installed executable/path.
void fixture()
{
    namespace fs = std::filesystem;
    fs::remove_all( ".build/routing" );
    fs::create_directories( ".build/routing/root" );
    fs::create_directories( ".build/routing/shm" );
    writerRecord( false );
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
    writerRecord( true );
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
/** \brief A calibration marker cannot release NN while the queued writer start remains unobserved. */
TEST_CASE( "Calibration marker alone cannot complete restart", "[po4ao][reviewRegression]" )
{
    fixture();
    MagAOX::app::Po4aoRouteIO io;
    SECTION( "stopped with queued work" )
    {
        io.capture();
    }
    SECTION( "old process still running" )
    {
        writerRecord( true );
        io.capture();
    }
    SECTION( "writer started between preflight and reservation" )
    {
        io.capture();
        writerRecord( true );
        io.switchInput( "po4ao_output", std::chrono::steady_clock::now() + std::chrono::seconds( 2 ) );
    }
    SECTION( "metadata changed without a new process" )
    {
        writerRecord( true );
        io.capture();
        writerRecord( true, 1, 2 );
    }
    REQUIRE_THROWS_WITH( io.applyDefault( std::chrono::steady_clock::now() + std::chrono::seconds( 2 ) ),
                         "CACAO calibration restart not acknowledged" );
}
/** \brief A delayed new writer must be observed before the adapter can return for NN restoration. */
TEST_CASE( "Calibration waits for a fresh writer process", "[po4ao][routingIO]" )
{
    fixture();
    writerRecord( true );
    MagAOX::app::Po4aoRouteIO io;
    io.capture();
    // The old writer has stopped, but the queued start has not executed.
    writerRecord( false );
    const auto  deadline  = std::chrono::steady_clock::now() + std::chrono::seconds( 3 );
    auto        completed = std::async( std::launch::async, [&] { io.applyDefault( deadline ); } );
    std::string marker;
    do
    {
        std::ifstream( ".build/routing/shm/aol1_calib_loaded.txt" ) >> marker;
        if( marker == "after" )
            break;
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    } while( std::chrono::steady_clock::now() < deadline );
    REQUIRE( marker == "after" );
    REQUIRE( completed.wait_for( std::chrono::milliseconds( 60 ) ) == std::future_status::timeout );
    writerRecord( true, 2, 2 );
    REQUIRE_NOTHROW( completed.get() );
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::Po4aoRouteIO::applyDefault;
    MagAOX::app::cacaoWriter::runningIdentity;
#endif
    // clang-format on
}
} // namespace libXWCTest::po4aoTest
