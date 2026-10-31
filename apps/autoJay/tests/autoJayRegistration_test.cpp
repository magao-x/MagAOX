/** \file autoJayRegistration_test.cpp
 * \brief Hardware-free image registration tests for autoJay.
 * \ingroup autoJay_files
 */

#include "../../../tests/testXWC.hpp"

#include <random>

#include "../autoJayRegistration.hpp"

namespace libXWCTest
{

/** \defgroup autoJay_registration_test autoJay Registration Tests
 * \brief Synthetic registration, geometry, and response-quality coverage.
 * \ingroup autoJay_unit_test
 */

/// Synthetic image tests of the autoJay application processing helpers.
namespace autoJayTest
{

using namespace MagAOX::app::autoJayDetail;

/// \cond
namespace
{

/// Describe a sensor-coordinate ROI by its first pixel center and sampled dimensions.
geometry makeGeometry( int    rows = 44, /**< [in] Horizontal sampled image length. */
                       int    cols = 40, /**< [in] Vertical sampled image length. */
                       double x0   = 0,  /**< [in] First horizontal detector pixel center. */
                       double y0   = 0,  /**< [in] First vertical detector pixel center. */
                       int    bx   = 1,  /**< [in] Horizontal binning. */
                       int    by   = 1 /**< [in] Vertical binning. */ );

/// Evaluate an asymmetric analytic spot pattern at physical detector coordinates.
imageT response( const geometry &g,           /**< [in] Applied ROI geometry. */
                 size_t          dm,          /**< [in] DM identity, selecting a different aligned pattern. */
                 double          dx      = 0, /**< [in] Horizontal physical displacement. */
                 double          dy      = 0, /**< [in] Vertical physical displacement. */
                 int             omitted = -1 /**< [in] Optional spot index to hide. */ );

/// Build known illuminated baseline and separate poke stacks with valid noise estimates.
std::array<frameStack, 3> acquisition( const geometry &g,      /**< [in] Applied ROI geometry. */
                                       double          nx = 0, /**< [in] NCPC horizontal displacement. */
                                       double          ny = 0, /**< [in] NCPC vertical displacement. */
                                       double          tx = 0, /**< [in] Tweeter horizontal displacement. */
                                       double          ty = 0 /**< [in] Tweeter vertical displacement. */ );

/// Build an aligned calibration without interacting with files or hardware.
calibration aligned( const geometry &g /**< [in] Shared reference geometry. */ );

geometry makeGeometry( int rows, int cols, double x0, double y0, int bx, int by )
{
    geometry g;
    g.units       = MagAOX::roi::units::sensor;
    g.orientation = "native";
    g.region      = { static_cast<float>( x0 + 0.5 * ( rows - 1 ) * bx ),
                      static_cast<float>( y0 + 0.5 * ( cols - 1 ) * by ),
                      rows * bx,
                      cols * by,
                      bx,
                      by };
    return g;
}

imageT response( const geometry &g, size_t dm, double dx, double dy, int omitted )
{
    imageT       image( static_cast<int>( MagAOX::roi::imageSize( g.region.w, g.region.bin_x, g.units ) ),
                  static_cast<int>( MagAOX::roi::imageSize( g.region.h, g.region.bin_y, g.units ) ) );
    const double x0 =
        MagAOX::roi::physical( g.region.x, g.region.bin_x, g.units ) - 0.5 * ( image.rows() - 1 ) * g.region.bin_x;
    const double y0 =
        MagAOX::roi::physical( g.region.y, g.region.bin_y, g.units ) - 0.5 * ( image.cols() - 1 ) * g.region.bin_y;
    const std::array<double, 4> px{ 10, 27, 17, 34 }, py{ 10, 13, 29, 31 }, amplitude{ 50, 75, 60, 90 };
    for( Eigen::Index y = 0; y < image.cols(); ++y )
    {
        for( Eigen::Index x = 0; x < image.rows(); ++x )
        {
            double value = 0;
            for( int p = 0; p < 4; ++p )
            {
                if( p == omitted )
                    continue;
                const double rx = x0 + x * g.region.bin_x - px[p] - dm * 1.2 - dx;
                const double ry = y0 + y * g.region.bin_y - py[p] + dm * 0.7 - dy;
                value += amplitude[p] * std::exp( -( rx * rx + ry * ry ) / ( 2 * 2.1 * 2.1 ) );
            }
            image( x, y ) = value;
        }
    }
    return image;
}

std::array<frameStack, 3> acquisition( const geometry &g, double nx, double ny, double tx, double ty )
{
    std::array<frameStack, 3> frames;
    imageT                    n = response( g, 0, nx, ny ), t = response( g, 1, tx, ty );
    for( auto &frame : frames )
    {
        frame.mean         = imageT::Constant( n.rows(), n.cols(), 500 );
        frame.variance     = imageT::Constant( n.rows(), n.cols(), 0.01 );
        frame.count        = 5;
        frame.firstCounter = 10;
        frame.lastCounter  = 14;
    }
    frames[1].mean += n;
    frames[2].mean += t;
    return frames;
}

calibration aligned( const geometry &g )
{
    calibration reference;
    reference.cameraGeometry  = g;
    reference.dm[0].reference = response( g, 0 );
    reference.dm[1].reference = response( g, 1 );
    return reference;
}

} // namespace
/// \endcond

/** \brief Verify physical displacement signs and aligned-reference subtraction.
 * \ingroup autoJay_registration_test
 */
TEST_CASE( "autoJay registers independent subpixel shifts and common motion", "[autoJay][registration]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJayDetail::registerPair;
    #endif
    // clang-format on
    const auto g         = makeGeometry();
    auto       reference = aligned( g );
    double     nx = 0, ny = 0, tx = 0, ty = 0;
    SECTION( "aligned grids with different absolute positions produce zero" )
    {
    }
    SECTION( "independent subpixel shifts" )
    {
        nx = 1.35;
        ny = -0.65;
        tx = -0.75;
        ty = 1.25;
    }
    SECTION( "equal motion cancels" )
    {
        nx = tx = 1.35;
        ny = ty = -0.65;
    }
    auto result = registerPair( reference, acquisition( g, nx, ny, tx, ty ), g, {} );
    INFO( result.reason );
    REQUIRE( result.valid );
    REQUIRE( result.dm[0].dx == Approx( nx ).margin( 0.1 ) );
    REQUIRE( result.dm[0].dy == Approx( ny ).margin( 0.1 ) );
    REQUIRE( result.dm[1].dx == Approx( tx ).margin( 0.1 ) );
    REQUIRE( result.dm[1].dy == Approx( ty ).margin( 0.1 ) );
    REQUIRE( result.deltaX == Approx( tx - nx ).margin( 0.1 ) );
    REQUIRE( result.deltaY == Approx( ty - ny ).margin( 0.1 ) );
}

/** \brief Changing only the shared camera ROI must not create differential motion.
 * \ingroup autoJay_registration_test
 */
TEST_CASE( "autoJay maps unequal shared ROIs and fractional binned origins", "[autoJay][registration]" )
{
    auto rg = makeGeometry(), lg = makeGeometry( 39, 35, 4, 3 );
    SECTION( "smaller offset live ROI" )
    {
    }
    SECTION( "larger odd live ROI" )
    {
        lg = makeGeometry( 49, 45, -3, -2 );
    }
    SECTION( "anisotropic binning and a fractional sampled-pixel origin" )
    {
        rg = makeGeometry( 24, 40, 0, 0, 2, 1 );
        lg = makeGeometry( 23, 39, 1, 0.5, 2, 1 );
    }
    SECTION( "binned native ROI convention" )
    {
        rg = makeGeometry( 24, 22, 0, 0, 2, 2 );
        lg = makeGeometry( 23, 21, 1, 1, 2, 2 );
        for( auto *g : { &rg, &lg } )
        {
            g->units    = MagAOX::roi::units::binned;
            g->region.x = ( g->region.x + 0.5 ) / g->region.bin_x - 0.5;
            g->region.y = ( g->region.y + 0.5 ) / g->region.bin_y - 0.5;
            g->region.w /= g->region.bin_x;
            g->region.h /= g->region.bin_y;
        }
    }
    auto result = registerPair( aligned( rg ), acquisition( lg, 0.8, -0.5, 0.8, -0.5 ), lg, {} );
    INFO( result.reason );
    REQUIRE( result.valid );
    REQUIRE( result.dm[0].dx == Approx( 0.8 ).margin( 0.1 ) );
    REQUIRE( result.dm[0].dy == Approx( -0.5 ).margin( 0.1 ) );
    REQUIRE( result.deltaX == Approx( 0 ).margin( 0.1 ) );
    REQUIRE( result.deltaY == Approx( 0 ).margin( 0.1 ) );
}

/** \brief Distinctive visible subsets remain measurable despite missing or clipped pokes.
 * \ingroup autoJay_registration_test
 */
TEST_CASE( "autoJay accepts partial distinctive poke patterns", "[autoJay][registration]" )
{
    const auto rg = makeGeometry();
    auto       lg = rg;
    SECTION( "one feature hidden" )
    {
    }
    SECTION( "pokes clipped by a smaller ROI" )
    {
        lg = makeGeometry( 28, 31, 6, 5 );
    }
    auto frames    = acquisition( lg, 0.5, -0.5, -0.5, 0.5 );
    frames[1].mean = frames[0].mean + response( lg, 0, 0.5, -0.5, 1 );
    auto result    = registerPair( aligned( rg ), frames, lg, {} );
    INFO( result.dm[0].dx << ", " << result.dm[0].dy << "; " << result.dm[1].dx << ", " << result.dm[1].dy );
    INFO( result.reason );
    REQUIRE( result.valid );
    REQUIRE( result.deltaX == Approx( -1 ).margin( 0.15 ) );
    REQUIRE( result.deltaY == Approx( 1 ).margin( 0.15 ) );
}

/** \brief Correlation and visible-feature handling retain response-scale invariance and signed structure.
 * \ingroup autoJay_registration_test
 */
TEST_CASE( "autoJay registers scaled and signed responses", "[autoJay][registration]" )
{
    const auto g         = makeGeometry();
    auto       reference = aligned( g );
    auto       frames    = acquisition( g, 0.6, -0.4, -0.6, 0.4 );
    SECTION( "large brightness decrease retains good registration when noise remains low" )
    {
        frames[1].mean = frames[0].mean + 0.1 * response( g, 0, 0.6, -0.4 );
        frames[2].mean = frames[0].mean + 0.1 * response( g, 1, -0.6, 0.4 );
        for( auto &frame : frames )
            frame.variance.setConstant( 0.00001 );
    }
    SECTION( "response contains positive and negative lobes" )
    {
        for( size_t dm = 0; dm < 2; ++dm )
        {
            reference.dm[dm].reference = response( g, dm, 0.2 ) - response( g, dm, -0.2 );
            const double dx = dm == 0 ? 0.6 : -0.6, dy = dm == 0 ? -0.4 : 0.4;
            frames[dm + 1].mean = frames[0].mean + response( g, dm, dx + 0.2, dy ) - response( g, dm, dx - 0.2, dy );
        }
    }
    auto result = registerPair( reference, frames, g, {} );
    INFO( result.reason );
    REQUIRE( result.valid );
    REQUIRE( result.deltaX == Approx( -1.2 ).margin( 0.1 ) );
    REQUIRE( result.deltaY == Approx( 0.8 ).margin( 0.1 ) );
}

/** \brief Reject changed sampling, unsupported metadata, and insufficient geometric support.
 * \ingroup autoJay_registration_test
 */
TEST_CASE( "autoJay rejects incompatible or unusable geometry", "[autoJay][registration]" )
{
    const auto rg        = makeGeometry();
    auto       lg        = rg;
    auto       reference = aligned( rg );
    SECTION( "binning changed" )
    {
        lg = makeGeometry( 22, 20, 0, 0, 2, 2 );
    }
    SECTION( "orientation changed" )
    {
        lg.orientation = "flipped";
    }
    SECTION( "no reference overlap" )
    {
        lg = makeGeometry( 44, 40, 1000, 1000 );
    }
    SECTION( "insufficient reference coverage" )
    {
        lg = makeGeometry( 8, 8, 7, 7 );
    }
    SECTION( "reference pair has different sizes" )
    {
        reference.dm[1].reference.conservativeResize( 43, 40 );
    }
    auto frames = acquisition( lg );
    SECTION( "applied geometry dimensions disagree with stack" )
    {
        ++lg.region.w;
    }
    auto result = registerPair( reference, frames, lg, {} );
    REQUIRE_FALSE( result.valid );
    REQUIRE_FALSE( result.reason.empty() );
}

/** \brief Reject unsupported values before they reach FFT normalization under fast-math builds.
 * \ingroup autoJay_registration_test
 */
TEST_CASE( "autoJay rejects malformed stacks and settings", "[autoJay][registration]" )
{
    const auto           g         = makeGeometry();
    auto                 frames    = acquisition( g );
    auto                 reference = aligned( g );
    registrationSettings settings;
    SECTION( "one frame has no measured noise variance" )
    {
        frames[1].count = 1;
    }
    SECTION( "negative noise variance" )
    {
        frames[2].variance( 0, 0 ) = -1;
    }
    SECTION( "nonfinite raw image" )
    {
        frames[0].mean( 0, 0 ) = std::numeric_limits<float>::quiet_NaN();
    }
    SECTION( "nonfinite noise variance" )
    {
        frames[1].variance( 0, 0 ) = std::numeric_limits<float>::infinity();
    }
    SECTION( "nonfinite reference" )
    {
        reference.dm[0].reference( 0, 0 ) = std::numeric_limits<float>::infinity();
    }
    SECTION( "saturated acquisition" )
    {
        settings.saturation = 550;
    }
    SECTION( "zero search radius" )
    {
        settings.maxShift = 0;
    }
    SECTION( "nonfinite search radius" )
    {
        settings.maxShift = std::numeric_limits<double>::quiet_NaN();
    }
    SECTION( "unrepresentable search radius" )
    {
        settings.maxShift = 1e100;
    }
    SECTION( "finite inputs overflow during baseline subtraction" )
    {
        settings.saturation    = std::numeric_limits<double>::max();
        frames[0].mean( 0, 0 ) = -std::numeric_limits<float>::max();
        frames[1].mean( 0, 0 ) = std::numeric_limits<float>::max();
    }
    auto result = registerPair( reference, frames, g, settings );
    REQUIRE_FALSE( result.valid );
    REQUIRE_FALSE( result.reason.empty() );
}

/** \brief Reject low signal, wrong response sign, and excessive individual or differential motion.
 * \ingroup autoJay_registration_test
 */
TEST_CASE( "autoJay rejects invalid response quality and displacement", "[autoJay][registration]" )
{
    const auto           g      = makeGeometry();
    auto                 frames = acquisition( g );
    registrationSettings settings;
    SECTION( "no poke response" )
    {
        frames[1].mean = frames[0].mean;
    }
    SECTION( "inverted response" )
    {
        frames[1].mean = frames[0].mean - response( g, 0 );
    }
    SECTION( "low response SNR" )
    {
        frames[1].variance.setConstant( 1000 );
    }
    SECTION( "outside individual bound" )
    {
        frames = acquisition( g, 7, 0, 0, 0 );
    }
    SECTION( "at search boundary" )
    {
        frames = acquisition( g, 5, 0, 0, 0 );
    }
    SECTION( "differential exceeds bound even though individuals do not" )
    {
        frames = acquisition( g, -3.5, 0, 3.5, 0 );
    }
    auto result = registerPair( aligned( g ), frames, g, settings );
    INFO( result.reason );
    REQUIRE_FALSE( result.valid );
    REQUIRE_FALSE( result.reason.empty() );
}

/** \brief First capture needs independent signal evidence because noise self-correlates perfectly.
 * \ingroup autoJay_registration_test
 */
TEST_CASE( "autoJay validates first capture without accepting noise as a reference", "[autoJay][registration]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJayDetail::validateCapture;
    #endif
    // clang-format on
    const auto g        = makeGeometry();
    auto       frames   = acquisition( g );
    bool       expected = true;
    SECTION( "valid capture has zero residual without previous references" )
    {
    }
    SECTION( "noise-only capture" )
    {
        std::mt19937                    generator( 42 );
        std::normal_distribution<float> noise( 0, 1 );
        for( size_t dm = 1; dm < 3; ++dm )
        {
            for( Eigen::Index p = 0; p < frames[dm].mean.size(); ++p )
                frames[dm].mean.data()[p] = frames[0].mean.data()[p] + noise( generator );
            frames[dm].variance.setConstant( 1 );
        }
        expected = false;
    }
    SECTION( "periodic template is ambiguous within capture range" )
    {
        for( size_t dm = 1; dm < 3; ++dm )
        {
            for( Eigen::Index y = 0; y < frames[dm].mean.cols(); ++y )
                for( Eigen::Index x = 0; x < frames[dm].mean.rows(); ++x )
                    frames[dm].mean( x, y ) += 30 * std::cos( 2 * M_PI * x / 4.0 ) * std::cos( 2 * M_PI * y / 4.0 );
            // Use only the repeated response, not the distinctive fixture pattern.
            frames[dm].mean -= response( g, dm - 1 );
        }
        expected = false;
    }
    auto result = validateCapture( frames, g, {} );
    INFO( result.reason );
    REQUIRE( result.valid == expected );
    if( expected )
    {
        REQUIRE( result.deltaX == Approx( 0 ).margin( 0.1 ) );
        REQUIRE( result.deltaY == Approx( 0 ).margin( 0.1 ) );
    }
}

/** \brief Visibility refinement must not resolve an intrinsically ambiguous pattern by fitting its mask.
 * \ingroup autoJay_registration_test
 */
TEST_CASE( "autoJay rejects competing grid translations before removing absent reference features",
           "[autoJay][registration]" )
{
    const auto g         = makeGeometry();
    auto       reference = aligned( g );
    auto       frames    = acquisition( g );
    auto      &r         = reference.dm[0].reference;
    r.setZero();
    frames[1].mean = frames[0].mean;
    for( Eigen::Index y = 0; y < r.cols(); ++y )
    {
        for( Eigen::Index x = 0; x < r.rows(); ++x )
        {
            // Two copies of a vertical grid both explain the one surviving live grid at +/-4 pixels.
            for( double yy : { 10.0, 21.0, 31.0 } )
            {
                const double row = ( y - yy ) * ( y - yy );
                r( x, y ) += 70 * ( std::exp( -( ( x - 16 ) * ( x - 16 ) + row ) / 2.0 ) +
                                    std::exp( -( ( x - 24 ) * ( x - 24 ) + row ) / 2.0 ) );
                frames[1].mean( x, y ) += 70 * std::exp( -( ( x - 20 ) * ( x - 20 ) + row ) / 2.0 );
            }
            // The original reference has a distinguishing poke, but it is absent from this acquisition.
            r( x, y ) += 15 * std::exp( -( ( x - 7 ) * ( x - 7 ) + ( y - 6 ) * ( y - 6 ) ) / 2.0 );
        }
    }
    const auto result = registerPair( reference, frames, g, {} );
    INFO( result.reason );
    REQUIRE_FALSE( result.valid );
    REQUIRE( result.dm[0].reason == "ambiguous competing registration peak" );
    REQUIRE( result.dm[0].secondaryRatio >= 0.95 );
}

} // namespace autoJayTest
} // namespace libXWCTest
