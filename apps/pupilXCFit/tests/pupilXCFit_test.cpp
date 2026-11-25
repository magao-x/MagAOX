/** \file pupilXCFit_test.cpp
 * \brief Offline tests of pyramid pupil cross-correlation and reference changes.
 * \ingroup pupilXCFit_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../pupilXCFit.hpp"

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

    /// Exercise a reference attachment loss.
    using pupilXCFit::shmimMonitorTiming;

  private:
    /// Metadata for the fake live attachment.
    IMAGE_METADATA m_metadata{};
};

Harness::Harness( bool fit )
{
    shmimMonitorT::m_width          = 96;
    shmimMonitorT::m_height         = 84;
    shmimMonitorT::m_dataType       = _DATATYPE_FLOAT;
    shmimMonitorT::m_imageStream.md = &m_metadata;
    m_metadata.atime                = { 123, 456 };
    m_fitSelected                   = fit;
    createStandardIndiSelectionSw( m_indiP_reference, "reference", { "wfs", "fit" } );
    REQUIRE( allocate( dev::shmimT() ) == 0 );
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

} // namespace pupilXCFitTest
} // namespace libXWCTest
