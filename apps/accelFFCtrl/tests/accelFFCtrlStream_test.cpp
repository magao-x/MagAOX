/** \file accelFFCtrlStream_test.cpp
 * \brief Opt-in native ADC reader tests in a process-private ImageStreamIO namespace.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelFFCtrlStream.hpp"

#include <cstdlib>
#include <filesystem>
#include <unistd.h>

namespace libXWCTest
{
namespace accelFFCtrlTest
{
using namespace MagAOX::app::accelFFCtrlDetail;

/// \cond ACCELFFCTRL_NATIVE_FIXTURE
/// Own a synthetic raw ADC stream only after verifying its private, externally selected namespace.
class nativeInput
{
    bool m_created{ false }; ///< Destruction is permitted only after this fixture created the image.

  public:
    IMAGE       m_image{}; ///< Sole synthetic producer mapping.
    std::string m_name;    ///< Unique non-actuator test stream name.

    /// Create a small temporal raw-pair ring; MILK_SHM_DIR must already select an empty private directory.
    explicit nativeInput( uint32_t depth /**< [in] retained pair count */ );

    /// Destroy only this fixture's private synthetic image.
    ~nativeInput();

    /// Publish a synthetic plane following frameGrabber's ordering and timestamp conventions.
    void publish( uint64_t                counter /**< [in] source identity */,
                  int64_t                 acquired /**< [in] realtime acquisition ns */,
                  std::array<uint16_t, 2> codes /**< [in] raw pair */ );
};

nativeInput::nativeInput( uint32_t depth )
{
    const char *directory = std::getenv( "MILK_SHM_DIR" );
    REQUIRE( directory != nullptr );
    REQUIRE( std::string( directory ).starts_with( "/tmp/accelFFCtrl-native-" ) );
    static unsigned serial = 0;
    m_name                 = "accff-private-" + std::to_string( getpid() ) + "-" + std::to_string( ++serial );
    char path[1024];
    REQUIRE( ImageStreamIO_filename( path, sizeof( path ), m_name.c_str() ) == 0 );
    REQUIRE( std::filesystem::path( path ).parent_path() == std::filesystem::path( directory ) );
    REQUIRE_FALSE( std::filesystem::exists( path ) );
    uint32_t size[3]{ 2, 1, depth };
    REQUIRE( ImageStreamIO_createIm_gpu( &m_image,
                                         m_name.c_str(),
                                         3,
                                         size,
                                         _DATATYPE_UINT16,
                                         -1,
                                         1,
                                         IMAGE_NB_SEMAPHORE,
                                         0,
                                         CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                         0 ) == 0 );
    m_created = true;
}

nativeInput::~nativeInput()
{
    if( m_created )
        ImageStreamIO_destroyIm( &m_image );
}

void nativeInput::publish( uint64_t counter, int64_t acquired, std::array<uint16_t, 2> codes )
{
    const auto plane  = ( counter - 1 ) % m_image.md->size[2];
    m_image.md->write = 1;
    std::memcpy( m_image.array.UI16 + 2 * plane, codes.data(), 2 * sizeof( uint16_t ) );
    m_image.md->atime             = { acquired / 1000000000, acquired % 1000000000 };
    m_image.md->writetime         = { ( acquired + 10000 ) / 1000000000, ( acquired + 10000 ) % 1000000000 };
    m_image.md->cnt1              = plane;
    m_image.md->cnt0              = counter;
    m_image.atimearray[plane]     = m_image.md->atime;
    m_image.writetimearray[plane] = m_image.md->writetime;
    m_image.cntarray[plane]       = counter;
    m_image.md->write             = 0;
    ImageStreamIO_sempost( &m_image, -1 );
}
/// \endcond

/** \brief Native reader copies all retained pairs and leaves producer payload/metadata intact, including depth-one
 * loss.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl private native retained ADC input", "[.][accelFFCtrl-native]" )
{
    constexpr int64_t epoch = 1700000000000000000;
    for( uint32_t depth : { 1, 8 } )
    {
        nativeInput    producer( depth );
        adcStream      reader;
        inputProcessor processor;
        REQUIRE( reader.open( producer.m_name ) );
        REQUIRE( reader.inode() != 0 );
        processor.attached();
        producer.publish( 1, epoch + 1000000000, { 2017, 2031 } );
        auto poll = [&]( int64_t mono )
        { return processor.drain( reader, [mono] { return inputClock{ epoch + mono, mono, 10 }; } ); };
        REQUIRE( poll( 1000050000 ) == 1 );
        REQUIRE( reader.frame( 0 ).m_adc == std::array<uint16_t, 2>{ 2017, 2031 } );
        for( uint64_t n = 2; n <= 5; ++n )
            producer.publish( n, epoch + 1000000000 + ( n - 1 ) * 125000, { uint16_t( 2048 + n ), 2048 } );
        REQUIRE( poll( 1000550000 ) == ( depth == 1 ? 1 : 4 ) );
        REQUIRE( processor.diagnostics().m_dropped == ( depth == 1 ? 3 : 0 ) );
        REQUIRE( producer.m_image.md->cnt0 == 5 );
        REQUIRE( producer.m_image.array.UI16[2 * producer.m_image.md->cnt1] == 2053 );
        REQUIRE( reader.header().m_uint16 );
        producer.m_image.md->write = 1;
        REQUIRE( poll( 1000550000 ) == 0 );
        producer.m_image.md->write = 0;
        timespec now;
        REQUIRE( clock_gettime( CLOCK_MONOTONIC, &now ) == 0 );
        REQUIRE( reader.wait( int64_t( now.tv_sec ) * 1000000000 + now.tv_nsec + 100000 ) );
        reader.close();
        REQUIRE_FALSE( reader.header().m_valid );
        REQUIRE( producer.m_image.md->cnt0 == 5 );
        // Reattachment starts a new generation and binds only the newest publication.
        processor.disconnected();
        REQUIRE( reader.open( producer.m_name ) );
        processor.attached();
        REQUIRE( poll( 1000600000 ) == 1 );
        REQUIRE( processor.diagnostics().m_generation == 2 );
    }
}

/** \brief Missing names and DM-prefixed names cannot attach; incompatible native types cannot feed the estimator.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl native input validation", "[.][accelFFCtrl-native]" )
{
    nativeInput producer( 4 );
    adcStream   reader;
    REQUIRE_FALSE( reader.open( "dm01disp00" ) ); // Rejected before any native filename lookup/open.
    REQUIRE_FALSE( reader.open( "accff-private-missing" ) );
    REQUIRE( reader.open( producer.m_name ) );
    producer.m_image.md->datatype = _DATATYPE_FLOAT;
    REQUIRE_FALSE( reader.header().m_uint16 );
    producer.m_image.md->datatype = _DATATYPE_UINT16;
    producer.m_image.md->size[2]  = 5;
    REQUIRE_FALSE( reader.header().m_valid );
    producer.m_image.md->size[2] = 4;
    REQUIRE( reader.frame( 4 ).m_counter == 0 );
}

} // namespace accelFFCtrlTest
} // namespace libXWCTest
