/** \file modalFilter_test.cpp
 * \brief Verify CACAO recurrence, frame ordering, and predictive feedback offline.
 * \ingroup modalFilter_files
 */

#include "../../../tests/testXWC.hpp"
#include "../modalFilter.hpp"

namespace MagAOX::app
{

/// \cond
class modalFilter_test : public modalFilter
{
  public:
    /// Size the offline controller and initialize its INDI switches.
    void prepare( size_t modes /**< [in] Number of modes. */ )
    {
        configureModes( modes );
        REQUIRE( configureAcquisition() == 0 );
        m_reconfig = false;
        createStandardIndiToggleSw( m_indiP_loop, "loop_state" );
        createStandardIndiToggleSw( m_indiP_pcOn, "loop_pcOn" );
        createStandardIndiRequestSw( m_indiP_loopZero, "loop_zero" );
        m_gain = 0.4F;
        m_mult = 0.97F;
    }

    /// Supply a synthetic vector through the normal monitor callback.
    template <ModalFilterStream stream>
    void parameterFrame( std::vector<float> values /**< [in] Parameter vector. */,
                         uint64_t           count = 1 /**< [in] Counter. */ )
    {
        Monitor<stream>::m_width    = values.size();
        Monitor<stream>::m_height   = 1;
        Monitor<stream>::m_dataType = _DATATYPE_FLOAT;
        IMAGE_METADATA metadata{};
        metadata.cnt0                     = count;
        Monitor<stream>::m_imageStream.md = &metadata;
        REQUIRE( allocate( Tag<stream>{} ) == 0 );
        REQUIRE( processImage( values.data(), Tag<stream>{} ) == 0 );
        Monitor<stream>::m_imageStream.md = nullptr;
    }

    /// Submit a WFS sample with a known acquisition time and frame count.
    void submit( std::vector<float> modes /**< [in] WFS values. */, uint64_t count /**< [in] Source counter. */ )
    {
        Sample sample{ std::move( modes ), { 100, static_cast<long>( count ) }, count };
        enqueue( sample );
    }

    /// Read the next output through the normal command path.
    std::vector<float> receive()
    {
        std::vector<float> output( m_modevalSz );
        REQUIRE( acquireAndCheckValid() == 0 );
        REQUIRE( loadImageIntoStream( output.data() ) == 0 );
        return output;
    }

    /// Return the source counter encoded into the output acquisition timestamp.
    uint64_t outputInputCount() const
    {
        return m_currImageTimestamp.tv_nsec;
    }

    /// Apply an ordinary loop state request.
    void loop( bool closed /**< [in] True to close the loop. */ )
    {
        auto request = m_indiP_loop;
        request["toggle"].setSwitchState( closed ? pcf::IndiElement::On : pcf::IndiElement::Off );
        REQUIRE( newCallBack_m_indiP_loop( request ) == 0 );
    }

    /// Make a zero request through INDI.
    void zero()
    {
        auto request = m_indiP_loopZero;
        request["request"].setSwitchState( pcf::IndiElement::On );
        REQUIRE( newCallBack_m_indiP_loopZero( request ) == 0 );
    }

    /// Request predictive control through the optimizer's expected INDI property.
    void requestPredictor()
    {
        auto request = m_indiP_pcOn;
        REQUIRE( request.getName() == "loop_pcOn" );
        request["toggle"].setSwitchState( pcf::IndiElement::On );
        REQUIRE( newCallBack_m_indiP_pcOn( request ) == 0 );
    }

    /// Validate and apply the supplied predictor frame.
    bool predictor( std::vector<float> data /**< [in] Coherent complete frame. */,
                    size_t             width /**< [in] Mode width. */ )
    {
        return installPredictor( data, width, m_modevalSz );
    }

    /// Whether the latest frame used predictive control.
    bool predictorActive() const
    {
        return m_pcActive;
    }

    using modalFilter::acquireAndCheckValid;
};
/// \endcond

} // namespace MagAOX::app

namespace libXWCTest::modalFilterTest
{

/** \defgroup modalFilter_unit_test modalFilter Unit Tests
 * \brief Offline modal filter tests.
 * \ingroup application_unit_test
 */

using MagAOX::app::modalFilter_test;
using MagAOX::app::ModalFilterStream;

/// Exercise the CACAO zero point, clipping, external command reseed, open hold, and explicit zero.
TEST_CASE( "modalFilter matches CACAO integration and command state", "[modalFilter]" )
{
    modalFilter_test app;
    app.prepare( 2 );
    app.parameterFrame<ModalFilterStream::gain>( { 0.75F, 0.5F } );
    app.parameterFrame<ModalFilterStream::mult>( { 0.94F, 0.8F } );
    app.parameterFrame<ModalFilterStream::zero>( { 0.2F, -0.1F } );
    app.parameterFrame<ModalFilterStream::limit>( { 0.5F, 0.25F } );
    app.loop( true );

    // clang-format off
    #ifdef MODALFILTER_TEST_DOXYGEN_REF
    MagAOX::app::modalFilter::processImage( nullptr, MagAOX::app::ModalFilterTag<ModalFilterStream::zero>{} );
    MagAOX::app::modalFilter::acquireAndCheckValid();
    MagAOX::app::modalFilter::loadImageIntoStream( nullptr );
    MagAOX::app::modalFilter::newCallBack_m_indiP_loopZero( pcf::IndiProperty{} );
    #endif
    // clang-format on

    app.submit( { 4, -4 }, 1 );
    CHECK( app.receive() == std::vector<float>{ -0.5F, 0.25F } );
    app.submit( { 0.2F, -0.1F }, 2 );
    const auto held = app.receive();
    CHECK( held[0] == Approx( -0.5F * ( 0.97F * 0.94F ) ) );
    CHECK( held[1] == Approx( 0.25F * ( 0.97F * 0.8F ) ) );

    app.loop( false );
    app.submit( { 100, 100 }, 3 );
    app.loop( true );
    app.submit( { 0.2F, -0.1F }, 4 );
    const auto resumed = app.receive();
    CHECK( resumed[0] == Approx( held[0] * ( 0.97F * 0.94F ) ) );
    CHECK( resumed[1] == Approx( held[1] * ( 0.97F * 0.8F ) ) );

    app.parameterFrame<ModalFilterStream::feedback>( { 0.25F, -0.2F } );
    app.submit( { 0.2F, -0.1F }, 5 );
    const auto seeded = app.receive();
    CHECK( seeded[0] == Approx( 0.25F * ( 0.97F * 0.94F ) ) );
    CHECK( seeded[1] == Approx( -0.2F * ( 0.97F * 0.8F ) ) );

    app.loop( false );
    app.zero();
    CHECK( app.receive() == std::vector<float>{ 0, 0 } );
}

/// Consume each queued WFS frame once and propagate its acquisition timestamp.
TEST_CASE( "modalFilter consumes queued frames in source order from cold start", "[modalFilter]" )
{
    modalFilter_test app;
    app.prepare( 1 );
    app.parameterFrame<ModalFilterStream::limit>( { 10 } );
    app.loop( true );
    app.submit( { 1 }, 1 );
    app.submit( { 2 }, 2 );
    app.submit( { 2 }, 2 );

    CHECK( app.receive()[0] == Approx( -0.4F ) );
    CHECK( app.outputInputCount() == 1 );
    CHECK( app.receive()[0] == Approx( -0.8F - 0.4F * 0.97F ) );
    CHECK( app.outputInputCount() == 2 );
    CHECK( app.acquireAndCheckValid() == 1 );
}

/// Apply different FIR and IIR taps, use SI on a gap, and reject a partial predictor frame.
TEST_CASE( "modalFilter applies a coherent predictive frame after history warms", "[modalFilter]" )
{
    modalFilter_test app;
    app.prepare( 2 );
    app.parameterFrame<ModalFilterStream::limit>( { 10, 10 } );
    app.loop( true );
    app.requestPredictor();

    constexpr size_t         width = MagAOX::app::modalPredictor::width( 3 );
    const std::vector<float> frame{ .6F, .8F, 2, 3, .65F, .12F, 0, .86F, -.04F, .05F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    REQUIRE( app.predictor( frame, width ) );

    // clang-format off
    #ifdef MODALFILTER_TEST_DOXYGEN_REF
    MagAOX::app::modalFilter::installPredictor( frame, width, 2 );
    MagAOX::app::modalFilter::filter( MagAOX::app::modalFilter::Sample{} );
    #endif
    // clang-format on

    app.submit( { 0, 0 }, 1 );
    CHECK( app.receive()[0] == 0 );
    app.submit( { 0, 0 }, 2 );
    CHECK( app.receive()[0] == 0 );
    CHECK_FALSE( app.predictorActive() );

    auto invalid = frame;
    invalid[3]   = 4;
    invalid[0]   = 0;
    CHECK_FALSE( app.predictor( invalid, width ) );
    app.submit( { 1, 1 }, 3 );
    const auto first = app.receive();
    CHECK( first[0] == Approx( -0.6F * 0.86F ) );
    CHECK( first[1] == Approx( -0.4F ) );
    CHECK( app.predictorActive() );

    app.submit( { 0, 0 }, 4 );
    const auto second = app.receive();
    CHECK( second[0] == Approx( 0.6F * 0.04F + 0.8F * 0.65F * first[0] ) );
    app.submit( { 0, 0 }, 6 );
    CHECK( app.receive()[0] == Approx( second[0] * 0.97F ) );
    CHECK_FALSE( app.predictorActive() );
    app.submit( { 0, 0 }, 7 );
    const float beforeReenable = app.receive()[0];
    app.submit( { 1, 0 }, 8 );
    CHECK( app.receive()[0] ==
           Approx( -0.6F * 0.86F + 0.8F * ( 0.65F * beforeReenable + 0.12F * second[0] * 0.97F ) ) );
    CHECK( app.predictorActive() );
}

} // namespace libXWCTest::modalFilterTest
