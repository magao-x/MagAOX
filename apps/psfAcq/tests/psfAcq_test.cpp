/** \file psfAcq_test.cpp
 * \brief Catch2 tests for the psfAcq app.
 *
 * \ingroup psfAcq_files
 */

#include "../../../tests/testXWC.hpp"

#include "../psfAcq.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup psfAcq_unit_test psfAcq Unit Tests
 * \brief Unit tests for the psfAcq application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `psfAcq` unit tests.
/** \ingroup psfAcq_unit_test
 */
namespace psfAcqTest
{

/// Verify the placeholder psfAcq test harness instantiates the app cleanly.
/**
 * \ingroup psfAcq_unit_test
 */
TEST_CASE( "psfAcq placeholder harness instantiates the app", "[psfAcq]" )
{
    // clang-format off
    #ifdef PSFACQ_TEST_DOXYGEN_REF
    psfAcq();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        psfAcq app;

        REQUIRE( true );
    }
}

/// \cond TEST_HARNESS
/// Expose tracked-star state for offline selection tests.
class psfAcqHarness : public psfAcq
{
  public:
    /// Add a tracked star without an INDI property.
    void addStar( std::size_t id /**< [in] Track id. */,
                  float       x /**< [in] Row coordinate in pixels. */,
                  float       y /**< [in] Column coordinate in pixels. */,
                  float       peak /**< [in] Fitted peak. */,
                  float       seeing /**< [in] Seeing in arcsec. */ );

    using psfAcq::acquireNudge;
    using psfAcq::m_current_acq_star;
    using psfAcq::m_detectedStars;
    using psfAcq::m_plate_scale;
    using psfAcq::m_seeing;
    using psfAcq::m_seeing_star;
    using psfAcq::m_x_center;
    using psfAcq::m_y_center;
    using psfAcq::selectSeeingStar;
};

void psfAcqHarness::addStar( std::size_t id, float x, float y, float peak, float seeing )
{
    Star star;
    star.id     = id;
    star.x      = x;
    star.y      = y;
    star.max    = peak;
    star.seeing = seeing;
    m_detectedStars.push_back( std::move( star ) );
}
/// \endcond

/// A brightness-rank swap between two tracks does not change which star is acquired or used for seeing.
/**
 * \ingroup psfAcq_unit_test
 */
TEST_CASE( "psfAcq selects stars by track id across a brightness swap", "[psfAcq]" )
{
    // clang-format off
    #ifdef PSFACQ_TEST_DOXYGEN_REF
    psfAcq::acquireNudge; psfAcq::selectSeeingStar; psfAcq::starIndexById;
    #endif
    // clang-format on

    psfAcqHarness app;
    app.m_x_center    = 342;
    app.m_y_center    = 320;
    app.m_plate_scale = 0.1;

    // Track 7 starts brighter (rank 0), track 3 dimmer (rank 1).
    app.addStar( 7, 100, 120, 500, 0.8 );
    app.addStar( 3, 300, 220, 400, 1.3 );

    double nx = 0;
    double ny = 0;

    REQUIRE( app.acquireNudge( 3, nx, ny ) == 0 );
    CHECK( nx == Approx( 10.0 ) );
    CHECK( ny == Approx( 4.2 ) );

    app.selectSeeingStar();
    CHECK( app.m_current_acq_star == 7 );
    CHECK( app.m_seeing == Approx( 0.8 ) );

    app.m_seeing_star = 3;
    app.selectSeeingStar();
    CHECK( app.m_current_acq_star == 3 );
    CHECK( app.m_seeing == Approx( 1.3 ) );

    // Swap brightness ranks, as relabelStarsByBrightness would after a peak change.
    std::swap( app.m_detectedStars[0], app.m_detectedStars[1] );

    nx = 0;
    ny = 0;
    REQUIRE( app.acquireNudge( 3, nx, ny ) == 0 );
    CHECK( nx == Approx( 10.0 ) );
    CHECK( ny == Approx( 4.2 ) );

    app.selectSeeingStar();
    CHECK( app.m_current_acq_star == 3 );
    CHECK( app.m_seeing == Approx( 1.3 ) );

    // A lost track is rejected for acquisition, and seeing falls back to the brightest star.
    app.m_detectedStars.erase( app.m_detectedStars.begin() );
    CHECK( app.acquireNudge( 3, nx, ny ) == -1 );

    app.selectSeeingStar();
    CHECK( app.m_seeing_star == -1 );
    CHECK( app.m_current_acq_star == 7 );
    CHECK( app.m_seeing == Approx( 0.8 ) );

    app.m_detectedStars.clear();
    app.selectSeeingStar();
    CHECK( app.m_current_acq_star == -1 );
    CHECK( app.m_seeing == 0 );
}

} // namespace psfAcqTest

} // namespace libXWCTest
