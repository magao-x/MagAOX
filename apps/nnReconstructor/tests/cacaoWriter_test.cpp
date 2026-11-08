/** \file cacaoWriter_test.cpp
 * \brief Production FPS/FIFO I/O against local fixtures; no instrument connection.
 * \ingroup nnReconstructor_unit_test
 */
#define CATCH_CONFIG_MAIN
#include "../../../tests/testXWC.hpp"
#include "../cacaoWriter.hpp"
#include "cacaoWriterTestAbi.hpp"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace libXWCTest::nnReconstructorTest
{
/// \cond NNRECONSTRUCTOR_TEST_SUPPORT
/// Own disposable metadata and Linux-format process records for the real transport.
struct WriterFixture
{
    std::string                  m_root{ ".build/cacaoWriter" }; ///< Test-only directory.
    MagAOX::app::cacaoWriter     m_writer{ m_root, m_root };     ///< Production I/O under test.
    FUNCTION_PARAMETER_STRUCT_MD m_metadata{};                   ///< Injectable FPS evidence.

    /// Create an FPS fixture with the test process identity.
    WriterFixture();

    /// Write complete metadata and fake process identity records.
    void save( const std::string &ticks = "123" /**< [in] Simulated Linux start time. */ );
};

WriterFixture::WriterFixture()
{
    std::filesystem::remove_all( m_root );
    std::filesystem::create_directories( m_root + "/" + std::to_string( getpid() ) );
    std::strcpy( m_metadata.name, m_writer.m_name.c_str() );
    m_metadata.runpid = getpid();
    m_writer.m_fifo   = m_root + "/control.fifo";
    save();
}

void WriterFixture::save( const std::string &ticks )
{
    std::ofstream fps( m_root + "/" + m_writer.m_name + ".fps.shm", std::ios::binary );
    fps.write( reinterpret_cast<const char *>( &m_metadata ), sizeof( m_metadata ) );
    FUNCTION_PARAMETER payload{};
    fps.write( reinterpret_cast<const char *>( &payload ), sizeof( payload ) );
    const auto    path = m_root + "/" + std::to_string( getpid() );
    std::ofstream process( path + "/stat" );
    process << getpid() << " (writer with spaces) S";
    for( int field = 4; field < 22; ++field )
        process << " 0";
    process << " " << ticks << " 0\n";
    std::ofstream( path + "/cmdline" ) << "milk-wfs2cmodeval-1";
}
/// \endcond

/** \brief A new FPS timestamp alone cannot acknowledge a new writer process. */
TEST_CASE( "CACAO process identity distinguishes actual restart evidence", "[nnReconstructor][cacaoWriter]" )
{
    WriterFixture f;
    const auto    original = f.m_writer.runningIdentity();
    REQUIRE_FALSE( original.empty() );
    f.m_metadata.runpidstarttime.tv_sec = 1;
    f.save();
    REQUIRE( f.m_writer.runningIdentity() == original );
    f.m_metadata.runpidstarttime.tv_sec = 2;
    f.save( "456" );
    REQUIRE( f.m_writer.runningIdentity() != original );
    f.m_metadata.runpid = 0;
    f.m_metadata.status = 0;
    f.save();
    REQUIRE( f.m_writer.runningIdentity().empty() );
}

/** \brief RUN state requires complete FPS metadata and matching process identity. */
TEST_CASE( "CACAO writer verifies metadata and process lifetime", "[nnReconstructor][cacaoWriter]" )
{
    WriterFixture f;
    REQUIRE( f.m_writer.running() );
    SECTION( "Reused PID" )
    {
        f.save( "999" );
        REQUIRE_THROWS_WITH( f.m_writer.running(), "CACAO RUN PID was reused" );
    }
    SECTION( "Unrelated process" )
    {
        f.m_writer.m_processMatch = "another-writer";
        REQUIRE_THROWS( f.m_writer.running() );
    }
    SECTION( "Uncompleted stop" )
    {
        f.m_metadata.runpid = 0;
        f.m_metadata.status = FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN;
        f.save();
        REQUIRE_THROWS( f.m_writer.running() );
        f.m_metadata.status = 0;
        f.save();
        REQUIRE_FALSE( f.m_writer.running() );
    }
    SECTION( "Truncated metadata" )
    {
        std::ofstream( f.m_root + "/" + f.m_writer.m_name + ".fps.shm" ) << "short";
        REQUIRE_THROWS( f.m_writer.running() );
    }
    SECTION( "Wrong layout" )
    {
        f.m_metadata.NBparamMAX = 100;
        f.save();
        REQUIRE_THROWS( f.m_writer.running() );
    }
    SECTION( "Nonpositive parameter count" )
    {
        f.m_metadata.NBparamMAX = 0;
        f.save();
        REQUIRE_THROWS( f.m_writer.running() );
        f.m_metadata.NBparamMAX = -1;
        f.save();
        REQUIRE_THROWS( f.m_writer.running() );
    }
    SECTION( "Partial parameter payload" )
    {
        const auto path = f.m_root + "/" + f.m_writer.m_name + ".fps.shm";
        const auto size = std::filesystem::file_size( path );
        std::filesystem::resize_file( path, size - 1 );
        REQUIRE_THROWS( f.m_writer.running() );
        f.save();
        std::filesystem::resize_file( path, size + 1 );
        REQUIRE_THROWS( f.m_writer.running() );
    }
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::cacaoWriter::running;
#endif
    // clang-format on
}

/** \brief FIFO requests are bounded and never interpreted as writer completion. */
TEST_CASE( "CACAO FIFO delivery is independent of completion", "[nnReconstructor][cacaoWriter]" )
{
    WriterFixture f;
    REQUIRE_THROWS( f.m_writer.command( false ) );
    std::ofstream( f.m_writer.m_fifo ) << "ordinary file";
    REQUIRE_THROWS( f.m_writer.command( false ) );
    std::filesystem::remove( f.m_writer.m_fifo );
    REQUIRE( mkfifo( f.m_writer.m_fifo.c_str(), 0600 ) == 0 );
    REQUIRE_THROWS( f.m_writer.command( false ) ); // No reader: open must not block.
    const int reader = open( f.m_writer.m_fifo.c_str(), O_RDONLY | O_NONBLOCK );
    REQUIRE( reader >= 0 );
    f.m_writer.command( false );
    char       request[128]{};
    const auto size = read( reader, request, sizeof( request ) );
    close( reader );
    REQUIRE( size > 0 );
    REQUIRE( std::string( request, size ) == "runstop wfs2cmodeval-1\n" );
    REQUIRE( f.m_writer.running() ); // Delivery alone did not change process evidence.
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::cacaoWriter::command;
#endif
    // clang-format on
}
} // namespace libXWCTest::nnReconstructorTest
