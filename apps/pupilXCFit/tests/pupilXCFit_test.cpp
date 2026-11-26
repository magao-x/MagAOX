/** \file pupilXCFit_test.cpp
 * \brief Offline tests of pyramid pupil cross-correlation, reference changes, and archived results.
 * \ingroup pupilXCFit_unit_test
 */

#include "../../../tests/testXWC.hpp"

#include <filesystem>
#include <fstream>

#include "../pupilXCFit.hpp"
#include "../../../libMagAOX/logger/generated/logMemberAccessor.hpp"

namespace libXWCTest
{
namespace pupilXCFitTest
{

using namespace MagAOX::app;

/// \cond TEST_HARNESS
using Image = mx::improc::eigenImage<float>;

/// Generate an asymmetric rectangular image translated by the specified x and y offsets.
Image scene( float dx = 0, /**< [in] x translation */
             float dy = 0 /**< [in] y translation */ );

Image scene( float dx, float dy )
{
    Image image( 96, 84 );

    for( Eigen::Index col = 0; col < image.cols(); ++col )
    {
        for( Eigen::Index row = 0; row < image.rows(); ++row )
        {
            const float x     = row - dx - 35;
            const float y     = col - dy - 32;
            image( row, col ) = std::exp( -( x * x / 19 + y * y / 40 ) ) +
                                0.6f * std::exp( -( ( x - 18 ) * ( x - 18 ) / 27 + ( y + 11 ) * ( y + 11 ) / 17 ) );
        }
    }

    return image;
}

/// Deliver frames directly through the production callbacks without attaching to local streams.
class Harness : public pupilXCFit
{
  public:
    /// Prepare a live attachment, with a selectable startup reference.
    explicit Harness( bool fit = false /**< [in] true to select the fit reference at startup */ );

    /// Clean up the fixture's private archive, if one was enabled.
    ~Harness() noexcept override;

    /// Deliver an image to either reference monitor.
    template <bool fitT>
    void reference( Image  &image /**< [in] reference frame */,
                    uint8_t type = _DATATYPE_FLOAT /**< [in] stream type */ );

    /// Deliver a live frame with the fixture acquisition time.
    void live( Image &image /**< [in] live frame */ );

    /// Apply a reference change through its INDI callback.
    void select( const std::string &element /**< [in] reference selector element */ );

    /// Claim a pending frame through the production worker callback.
    void claim();

    /// Read the previously claimed result through the output callback.
    shiftT claimed();

    /// Claim and read the next pending output.
    shiftT output();

    /// Return the production output timestamp.
    timespec timestamp() const;

    /// Enable the regular telemetry queue with a private destination.
    void enableTelemetry();

    /// Drain the writer synchronously and return verified records from the archive.
    std::vector<flatlogs::bufferPtrT> archivedFits();

    /// Exercise a reference attachment loss.
    using pupilXCFit::shmimMonitorTiming;

  private:
    /// Metadata for the fake live attachment.
    IMAGE_METADATA m_metadata{};

    /// Private directory for the optional telemetry archive.
    std::filesystem::path m_telemetryPath;
};

Harness::Harness( bool fit )
{
    shmimMonitorT::m_width          = 96;
    shmimMonitorT::m_height         = 84;
    shmimMonitorT::m_dataType       = _DATATYPE_FLOAT;
    shmimMonitorT::m_imageStream.md = &m_metadata;
    wfsRefMonitorT::m_shmimName     = "camwfs_ref";
    fitRefMonitorT::m_shmimName     = "test_fit_reference";
    m_metadata.atime                = { 123, 456 };
    m_fitSelected                   = fit;
    createStandardIndiSelectionSw( m_indiP_reference, "reference", { "wfs", "fit" } );
    REQUIRE( allocate( dev::shmimT() ) == 0 );
}

Harness::~Harness() noexcept
{
    if( !m_telemetryPath.empty() )
    {
        std::error_code error;
        std::filesystem::remove_all( m_telemetryPath, error );
    }
}

template <bool fitT>
void Harness::reference( Image &image, uint8_t type )
{
    using monitorT       = dev::shmimMonitor<pupilXCFit, pupilXCReferenceT<fitT>>;
    monitorT::m_width    = image.rows();
    monitorT::m_height   = image.cols();
    monitorT::m_dataType = type;
    REQUIRE( processImage( image.data(), pupilXCReferenceT<fitT>() ) == 0 );
}

void Harness::live( Image &image )
{
    ++m_metadata.cnt0;
    REQUIRE( processImage( image.data(), dev::shmimT() ) == 0 );
}

void Harness::select( const std::string &element )
{
    pcf::IndiProperty request( pcf::IndiProperty::Switch );
    request.setName( m_indiP_reference.getName() );
    request.add( pcf::IndiElement( element, pcf::IndiElement::On ) );
    REQUIRE( newCallBack_m_indiP_reference( request ) == 0 );
}

void Harness::claim()
{
    REQUIRE( acquireAndCheckValid() == 0 );
}

auto Harness::claimed() -> shiftT
{
    shiftT result{};
    REQUIRE( loadImageIntoStream( result.data() ) == 0 );
    return result;
}

auto Harness::output() -> shiftT
{
    claim();
    return claimed();
}

timespec Harness::timestamp() const
{
    return frameGrabberT::m_currImageTimestamp;
}

void Harness::enableTelemetry()
{
    char directory[] = "/tmp/pupilXCFit_test_XXXXXX";
    REQUIRE( mkdtemp( directory ) != nullptr );
    m_telemetryPath = directory;
    m_tel.logPath( directory );
    m_tel.logName( "pupilXCFit_test" );
    m_tel.logExt( "bintel" );
    m_tel.logLevel( logPrio::LOG_TELEM );
}

std::vector<flatlogs::bufferPtrT> Harness::archivedFits()
{
    m_tel.logShutdown( true );
    m_tel.logThreadExec();

    std::vector<flatlogs::bufferPtrT> records;
    for( const auto &entry : std::filesystem::recursive_directory_iterator( m_telemetryPath ) )
    {
        if( entry.path().extension() != ".bintel" )
        {
            continue;
        }

        std::ifstream     input( entry.path(), std::ios::binary );
        std::vector<char> bytes( ( std::istreambuf_iterator<char>( input ) ), {} );
        size_t            offset = 0;

        while( offset < bytes.size() )
        {
            REQUIRE( bytes.size() - offset >= flatlogs::logHeader::minHeadSize );
            const auto size = flatlogs::logHeader::totalSize( bytes.data() + offset );
            REQUIRE( size >= flatlogs::logHeader::minHeadSize );
            REQUIRE( size <= bytes.size() - offset );
            flatlogs::bufferPtrT record( static_cast<char *>( ::operator new( size ) ) );
            std::memcpy( record.get(), bytes.data() + offset, size );
            REQUIRE( flatlogs::logHeader::eventCode( record ) == telem_pupilxcfit::eventCode );
            REQUIRE( logVerify( telem_pupilxcfit::eventCode, record, flatlogs::logHeader::msgLen( record ) ) );
            records.push_back( record );
            offset += size;
        }
    }

    return records;
}
/// \endcond

/// Verify live-minus-reference shifts, subpixel resolution, and transfer of the acquisition timestamp.
TEST_CASE( "pupilXCFit keeps ImageStreamIO axis order and acquisition time", "[pupilXCFit]" )
{
    // clang-format off
    #ifdef PUPILXCFIT_TEST_DOXYGEN_REF
    pupilXCFit::processImage;
    pupilXCFit::measure;
    pupilXCFit::loadImageIntoStream;
    #endif
    // clang-format on

    Harness app;
    auto    reference = scene();
    app.reference<false>( reference );
    app.live( reference );
    auto shift = app.output();
    CHECK( shift[0] == Approx( 0 ).margin( 0.03 ) );
    CHECK( shift[1] == Approx( 0 ).margin( 0.03 ) );

    auto target = scene( 2.50f, -1.25f );
    app.live( target );
    shift = app.output();
    CHECK( shift[0] == Approx( 2.50 ).margin( 0.04 ) );
    CHECK( shift[1] == Approx( -1.25 ).margin( 0.04 ) );
    CHECK( app.timestamp().tv_sec == 123 );
    CHECK( app.timestamp().tv_nsec == 456 );

    auto distant = scene( -11.5f, 0 );
    app.live( distant );
    CHECK( mx::math::isNan( app.output()[0] ) );
}

/// Verify reference failure, replacement and disconnect discard the superseded measurements.
TEST_CASE( "pupilXCFit waits for the selected reference and recovers after invalidation", "[pupilXCFit]" )
{
    // clang-format off
    #ifdef PUPILXCFIT_TEST_DOXYGEN_REF
    pupilXCFit::invalidateReference;
    pupilXCFit::prepareReference;
    pupilXCFit::shmimMonitorTiming;
    #endif
    // clang-format on

    Harness app( true );
    auto    wfs = scene();
    auto    fit = scene( 3, -2 );
    app.reference<false>( wfs );
    app.live( wfs );
    CHECK( mx::math::isNan( app.output()[0] ) );

    app.reference<true>( fit, _DATATYPE_UINT16 );
    app.live( wfs );
    CHECK( mx::math::isNan( app.output()[0] ) );

    Image mismatched = fit.leftCols( 83 );
    app.reference<true>( mismatched );
    app.live( wfs );
    CHECK( mx::math::isNan( app.output()[0] ) );

    app.reference<true>( fit );
    app.live( wfs );
    auto shift = app.output();
    CHECK( shift[0] == Approx( -3 ).margin( 0.03 ) );
    CHECK( shift[1] == Approx( 2 ).margin( 0.03 ) );

    app.live( wfs );
    app.claim();
    app.reference<false>( wfs );
    CHECK( app.claimed()[0] == Approx( -3 ).margin( 0.03 ) );
    app.select( "wfs" );
    CHECK( mx::math::isNan( app.claimed()[0] ) );
    CHECK( mx::math::isNan( app.output()[0] ) );
    app.live( wfs );
    CHECK( app.output()[0] == Approx( 0 ).margin( 0.03 ) );

    Image invalid = Image::Zero( 96, 84 );
    app.reference<false>( invalid );
    app.live( wfs );
    CHECK( mx::math::isNan( app.output()[0] ) );
    app.reference<false>( wfs );
    invalid( 0, 0 ) = std::numeric_limits<float>::infinity();
    app.live( invalid );
    CHECK( mx::math::isNan( app.output()[0] ) );
    app.live( wfs );
    app.claim();
    app.shmimMonitorTiming( pupilXCReferenceT<false>(), dev::shmimMonitorEvent::disconnected );
    CHECK( mx::math::isNan( app.claimed()[0] ) );
    app.reference<false>( wfs );
    app.live( wfs );
    CHECK( app.output()[0] == Approx( 0 ).margin( 0.03 ) );
}

/// Verify that successful, unavailable and rejected correlations carry the selected stream in the archive.
TEST_CASE( "pupilXCFit archives each live fit with its selected reference", "[pupilXCFit]" )
{
    // clang-format off
    #ifdef PUPILXCFIT_TEST_DOXYGEN_REF
    pupilXCFit::processImage;
    pupilXCFit::measure;
    MagAOX::logger::telem_pupilxcfit::verify;
    MagAOX::logger::telem_pupilxcfit::getAccessor;
    #endif
    // clang-format on

    Harness app;
    app.enableTelemetry();
    auto wfs     = scene();
    auto fit     = scene( 3, -2 );
    auto target  = scene( 2.50f, -1.25f );
    auto distant = scene( -8.5f, -2 );
    app.reference<false>( wfs );
    app.live( target );
    app.select( "fit" );
    app.live( target );
    app.reference<true>( fit );
    app.live( target );
    app.live( distant );
    CHECK( mx::math::isNan( app.output()[0] ) );

    auto records = app.archivedFits();
    REQUIRE( records.size() == 4 );
    const auto record = [&]( size_t index )
    {
        return GetTelem_pupilxcfit_fb( flatlogs::logHeader::messageBuffer( records[index] ) );
    };

    CHECK( record( 0 )->frame() == 1 );
    CHECK( record( 0 )->acq_sec() == 123 );
    CHECK( record( 0 )->acq_nsec() == 456 );
    CHECK( record( 0 )->reference()->str() == "camwfs_ref" );
    CHECK( record( 0 )->valid() );
    REQUIRE( record( 0 )->measurement() );
    CHECK( record( 0 )->measurement()->dx() == Approx( 2.50 ).margin( 0.04 ) );
    CHECK( record( 0 )->measurement()->dy() == Approx( -1.25 ).margin( 0.04 ) );
    CHECK( record( 0 )->measurement()->peak() > 0.98f );

    CHECK( record( 1 )->frame() == 2 );
    CHECK( record( 1 )->reference()->str() == "test_fit_reference" );
    CHECK_FALSE( record( 1 )->valid() );
    CHECK( record( 1 )->measurement() == nullptr );

    CHECK( record( 2 )->frame() == 3 );
    CHECK( record( 2 )->reference()->str() == "test_fit_reference" );
    CHECK( record( 2 )->valid() );
    REQUIRE( record( 2 )->measurement() );
    CHECK( record( 2 )->measurement()->dx() == Approx( -0.5 ).margin( 0.04 ) );
    CHECK( record( 2 )->measurement()->dy() == Approx( 0.75 ).margin( 0.04 ) );

    CHECK( record( 3 )->frame() == 4 );
    CHECK( record( 3 )->reference()->str() == "test_fit_reference" );
    CHECK_FALSE( record( 3 )->valid() );
    REQUIRE( record( 3 )->measurement() );
    CHECK( record( 3 )->measurement()->dx() == Approx( -11.5 ).margin( 0.04 ) );

    std::ostringstream text;
    logStdFormat( text, records[0] );
    CHECK( text.str().find( "reference=camwfs_ref" ) != std::string::npos );
    std::ostringstream json;
    logJsonFormat( json, records[2] );
    CHECK( json.str().find( "\"reference\": \"test_fit_reference\"" ) != std::string::npos );
    CHECK( json.str().find( "\"measurement\"" ) != std::string::npos );

    const auto reference = logMemberAccessor( telem_pupilxcfit::eventCode, "reference" );
    REQUIRE( reference.accessor );
    const auto getReference = reinterpret_cast<std::string ( * )( void * )>( reference.accessor );
    CHECK( getReference( flatlogs::logHeader::messageBuffer( records[1] ) ) == "test_fit_reference" );

    const auto peak = logMemberAccessor( telem_pupilxcfit::eventCode, "peak" );
    REQUIRE( peak.accessor );
    const auto getPeak = reinterpret_cast<float ( * )( void * )>( peak.accessor );
    CHECK( mx::math::isNan( getPeak( flatlogs::logHeader::messageBuffer( records[1] ) ) ) );
    CHECK( getPeak( flatlogs::logHeader::messageBuffer( records[3] ) ) == record( 3 )->measurement()->peak() );
}

} // namespace pupilXCFitTest
} // namespace libXWCTest
