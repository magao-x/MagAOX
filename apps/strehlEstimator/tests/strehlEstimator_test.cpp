/** \file strehlEstimator_test.cpp
 * \brief Catch2 tests for the strehlEstimator app.
 * \author OpenAI Codex
 *
 * \ingroup strehlEstimator_files
 */

/** \defgroup strehlEstimator_unit_test strehlEstimator Unit Tests
 * \brief Unit tests for the strehlEstimator application.
 *
 * \ingroup application_unit_test
 */

#include "../strehlEstimator.hpp"

#include "../../../tests/testMacrosINDI.hpp"
#include "../../../tests/testXWC.hpp"

#include <cmath>

using namespace MagAOX::app;

namespace libXWCTest
{

/// Namespace for `strehlEstimator` unit tests.
/** \ingroup strehlEstimator_unit_test
 */
namespace strehlEstimatorTest
{

namespace
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Test harness exposing the `strehlEstimator` internals needed by the unit tests.
class strehlEstimator_test : public strehlEstimator
{
  public:
    /// Construct a testable `strehlEstimator` instance with callback keys pre-seeded.
    strehlEstimator_test( const std::string &device )
    {
        m_configName = device;

        setupConfig();

        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_fps, camwfs, fps );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_emg, camwfs, emgain );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_stage, stagebs, presetName );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_tcsi_seeing, tcsi, seeing );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_tcsi_telpos, tcsi, telpos );

        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_mag, star_mag );
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_seeing_magaox, seeing );
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_windSpeed, wind_speed );
        XWCTEST_SETUP_INDI_ARB_NEW_PROP( m_indiP_useEstimates, use_estimates );
    }

    /// Initialize the published INDI properties without starting the shmim-monitor threads.
    int initializePublishedProperties()
    {
        if( createCurrentEstimatedProperty( m_indiP_mag, "star_mag", "Star Magnitude", "Error Budget" ) < 0 )
        {
            return -1;
        }

        if( createCurrentEstimatedProperty( m_indiP_seeing_magaox, "seeing", "Seeing", "Error Budget" ) < 0 )
        {
            return -1;
        }

        if( createStandardIndiSelectionSw( m_indiP_windSpeed,
                                           "wind_speed",
                                           windSpeedSelectionElements(),
                                           windSpeedSelectionLabels(),
                                           "Wind Speed",
                                           "Error Budget" ) < 0 )
        {
            return -1;
        }

        if( createStandardIndiToggleSw( m_indiP_useEstimates, "use_estimates", "Use Estimates", "Error Budget" ) < 0 )
        {
            return -1;
        }

        if( createROIndiNumber( m_indiP_strehl, "strehl_optimal", "Strehl", "Error Budget" ) < 0 )
        {
            return -1;
        }
        m_indiP_strehl.add( pcf::IndiElement( "pyramid", 0.0f ) );

        if( createROIndiNumber( m_indiP_wfe, "wfe_predicted", "WFE", "Error Budget" ) < 0 )
        {
            return -1;
        }
        m_indiP_wfe.add( pcf::IndiElement( "total", 0.0f ) );
        m_indiP_wfe.add( pcf::IndiElement( "measurement", 0.0f ) );
        m_indiP_wfe.add( pcf::IndiElement( "time_delay", 0.0f ) );
        m_indiP_wfe.add( pcf::IndiElement( "fitting", 0.0f ) );

        if( createROIndiNumber( m_indiP_loopSpeedOptimum, "loop_speed_optimum", "Optimum Loop Speed", "Error Budget" ) <
            0 )
        {
            return -1;
        }
        m_indiP_loopSpeedOptimum.add( pcf::IndiElement( "fps", 0.0f ) );
        m_indiP_loopSpeedOptimum.add( pcf::IndiElement( "strehl", 0.0f ) );
        m_indiP_loopSpeedOptimum.add( pcf::IndiElement( "wfe_total", 0.0f ) );
        m_indiP_loopSpeedOptimum.add( pcf::IndiElement( "wfe_measurement", 0.0f ) );
        m_indiP_loopSpeedOptimum.add( pcf::IndiElement( "wfe_time_delay", 0.0f ) );
        m_indiP_loopSpeedOptimum.add( pcf::IndiElement( "wfe_fitting", 0.0f ) );

        updatePlanningProperties();
        updatePredictionOutputs();

        return 0;
    }

    /// Recompute the public prediction properties immediately.
    void refreshPredictions()
    {
        updatePredictionOutputs();
    }

    /// Seed the live photometry inputs and recalculate the live guide-star magnitude.
    void setPhotometry( float counts, int npix )
    {
        m_counts = counts;
        m_npix   = npix;
        calcMag();
    }

    /// Return the current live guide-star magnitude.
    float liveMag() const
    {
        return m_mag;
    }

    /// Return the estimated guide-star magnitude.
    float estimatedMag() const
    {
        return m_magEstimated;
    }

    /// Return the current live seeing.
    float liveSeeing() const
    {
        return m_seeing;
    }

    /// Return the estimated seeing.
    float estimatedSeeing() const
    {
        return m_seeingEstimated;
    }

    /// Return the selected wind speed value in m/s.
    float windSpeed() const
    {
        return m_windSpeed;
    }

    /// Return the selected wind-speed switch element name.
    std::string windSpeedSelection() const
    {
        return windSpeedSelectionName( m_windSpeed );
    }

    /// Return whether estimate overrides are currently enabled.
    bool useEstimates() const
    {
        return m_useEstimates;
    }

    /// Return the selected star magnitude used by the AO model.
    float selectedMagDirect() const
    {
        return selectedStarMag();
    }

    /// Return the selected seeing used by the AO model.
    float selectedSeeingDirect() const
    {
        return selectedSeeing();
    }

    /// Return the published star-magnitude property.
    const pcf::IndiProperty &starMagProperty() const
    {
        return m_indiP_mag;
    }

    /// Return the published seeing property.
    const pcf::IndiProperty &seeingProperty() const
    {
        return m_indiP_seeing_magaox;
    }

    /// Return the published wind-speed property.
    const pcf::IndiProperty &windSpeedProperty() const
    {
        return m_indiP_windSpeed;
    }

    /// Return the published `use_estimates` toggle property.
    const pcf::IndiProperty &useEstimatesProperty() const
    {
        return m_indiP_useEstimates;
    }

    /// Return the published optimum-loop-speed property.
    const pcf::IndiProperty &optimumLoopSpeedProperty() const
    {
        return m_indiP_loopSpeedOptimum;
    }

    /// Return the published current-prediction Strehl.
    float predictedStrehl() const
    {
        return m_indiP_strehl["pyramid"].get<float>();
    }

    /// Return the published optimum-loop-speed FPS.
    float optimumFPS() const
    {
        return m_indiP_loopSpeedOptimum["fps"].get<float>();
    }

    /// Return the published optimum-loop-speed Strehl.
    float optimumStrehl() const
    {
        return m_indiP_loopSpeedOptimum["strehl"].get<float>();
    }

    /// Evaluate the configured AO model at one FPS sample for test-side brute-force comparisons.
    float predictedStrehlAtFps( float fps, bool optimizeTau )
    {
        predictionInputs inputs = snapshotPredictionInputs();
        configureAoSystem( m_aosysScan, inputs, fps, optimizeTau );
        return m_aosysScan.strehl();
    }
};
/// \endcond

/// Build a number-property update payload for the local app.
pcf::IndiProperty makeLocalNumberProperty( const std::string &device, const std::string &name )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );
    ip.setDevice( device );
    ip.setName( name );
    return ip;
}

/// Build a switch-property update payload for the local app.
pcf::IndiProperty makeLocalSwitchProperty( const std::string &device, const std::string &name )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );
    ip.setDevice( device );
    ip.setName( name );
    return ip;
}

/// Build a set-property payload from another device.
pcf::IndiProperty makeRemoteNumberProperty( const std::string &device, const std::string &name )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );
    ip.setDevice( device );
    ip.setName( name );
    return ip;
}

} // namespace

/// Verify the `strehlEstimator` callbacks reject mismatched property identities.
/**
 * \ingroup strehlEstimator_unit_test
 */
TEST_CASE( "strehlEstimator callbacks validate the property identity", "[strehlEstimator][indi]" )
{
    // clang-format off
    #ifdef STREHLESTIMATOR_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( strehlEstimator::setCallBack_m_indiP_fps( std::declval<const pcf::IndiProperty &>() ) );
    XWCTEST_DOXYGEN_REF( strehlEstimator::setCallBack_m_indiP_emg( std::declval<const pcf::IndiProperty &>() ) );
    XWCTEST_DOXYGEN_REF( strehlEstimator::setCallBack_m_indiP_stage( std::declval<const pcf::IndiProperty &>() ) );
    XWCTEST_DOXYGEN_REF( strehlEstimator::setCallBack_m_indiP_tcsi_seeing( std::declval<const pcf::IndiProperty &>() ) );
    XWCTEST_DOXYGEN_REF( strehlEstimator::setCallBack_m_indiP_tcsi_telpos( std::declval<const pcf::IndiProperty &>() ) );
    XWCTEST_DOXYGEN_REF( strehlEstimator::newCallBack_m_indiP_mag( std::declval<const pcf::IndiProperty &>() ) );
    XWCTEST_DOXYGEN_REF( strehlEstimator::newCallBack_m_indiP_seeing_magaox( std::declval<const pcf::IndiProperty &>() ) );
    XWCTEST_DOXYGEN_REF( strehlEstimator::newCallBack_m_indiP_windSpeed( std::declval<const pcf::IndiProperty &>() ) );
    XWCTEST_DOXYGEN_REF( strehlEstimator::newCallBack_m_indiP_useEstimates( std::declval<const pcf::IndiProperty &>() ) );
    #endif
    // clang-format on

    XWCTEST_INDI_SET_CALLBACK( strehlEstimator, m_indiP_fps, camwfs, fps );
    XWCTEST_INDI_SET_CALLBACK( strehlEstimator, m_indiP_emg, camwfs, emgain );
    XWCTEST_INDI_SET_CALLBACK( strehlEstimator, m_indiP_stage, stagebs, presetName );
    XWCTEST_INDI_SET_CALLBACK( strehlEstimator, m_indiP_tcsi_seeing, tcsi, seeing );
    XWCTEST_INDI_SET_CALLBACK( strehlEstimator, m_indiP_tcsi_telpos, tcsi, telpos );

    XWCTEST_INDI_ARBNEW_CALLBACK( strehlEstimator, newCallBack_m_indiP_mag, star_mag );
    XWCTEST_INDI_ARBNEW_CALLBACK( strehlEstimator, newCallBack_m_indiP_seeing_magaox, seeing );
    XWCTEST_INDI_ARBNEW_CALLBACK( strehlEstimator, newCallBack_m_indiP_windSpeed, wind_speed );
    XWCTEST_INDI_ARBNEW_CALLBACK( strehlEstimator, newCallBack_m_indiP_useEstimates, use_estimates );
}

/// Verify published-property initialization creates the planning-input and optimum-speed properties with the expected
/// elements.
/**
 * \ingroup strehlEstimator_unit_test
 */
TEST_CASE( "strehlEstimator startup publishes planning and optimum-speed properties", "[strehlEstimator]" )
{
    strehlEstimator_test app( "strehlEstimator_test" );

    REQUIRE( app.initializePublishedProperties() == 0 );

    REQUIRE( app.starMagProperty().find( "current" ) );
    REQUIRE( app.starMagProperty().find( "estimated" ) );

    REQUIRE( app.seeingProperty().find( "current" ) );
    REQUIRE( app.seeingProperty().find( "estimated" ) );

    REQUIRE( app.windSpeedProperty().find( "slow" ) );
    REQUIRE( app.windSpeedProperty().find( "normal" ) );
    REQUIRE( app.windSpeedProperty().find( "fast" ) );
    REQUIRE( app.windSpeedProperty()[app.windSpeedSelection()].getSwitchState() == pcf::IndiElement::On );

    REQUIRE( app.useEstimatesProperty().find( "toggle" ) );
    REQUIRE( app.useEstimatesProperty()["toggle"].getSwitchState() == pcf::IndiElement::Off );

    REQUIRE( app.optimumLoopSpeedProperty().find( "fps" ) );
    REQUIRE( app.optimumLoopSpeedProperty().find( "strehl" ) );
    REQUIRE( app.optimumLoopSpeedProperty().find( "wfe_total" ) );
    REQUIRE( app.optimumLoopSpeedProperty().find( "wfe_measurement" ) );
    REQUIRE( app.optimumLoopSpeedProperty().find( "wfe_time_delay" ) );
    REQUIRE( app.optimumLoopSpeedProperty().find( "wfe_fitting" ) );
}

/// Verify local star-magnitude and seeing writes only honor `estimated`, while the wind-speed selector updates the
/// planning wind state.
/**
 * \ingroup strehlEstimator_unit_test
 */
TEST_CASE( "strehlEstimator planning inputs honor estimated writes and wind-speed selections", "[strehlEstimator]" )
{
    // clang-format off
    #ifdef STREHLESTIMATOR_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( strehlEstimator::calcMag() );
    XWCTEST_DOXYGEN_REF( strehlEstimator::newCallBack_m_indiP_mag( std::declval<const pcf::IndiProperty &>() ) );
    XWCTEST_DOXYGEN_REF( strehlEstimator::newCallBack_m_indiP_seeing_magaox( std::declval<const pcf::IndiProperty &>() ) );
    XWCTEST_DOXYGEN_REF( strehlEstimator::newCallBack_m_indiP_windSpeed( std::declval<const pcf::IndiProperty &>() ) );
    XWCTEST_DOXYGEN_REF( strehlEstimator::setCallBack_m_indiP_tcsi_seeing( std::declval<const pcf::IndiProperty &>() ) );
    #endif
    // clang-format on

    strehlEstimator_test app( "right" );

    REQUIRE( app.initializePublishedProperties() == 0 );

    app.setPhotometry( 25000.0f, 144 );
    REQUIRE( app.estimatedMag() == Approx( app.liveMag() ) );

    SECTION( "star magnitude current writes are ignored" )
    {
        pcf::IndiProperty ip = makeLocalNumberProperty( "right", "star_mag" );
        ip.add( pcf::IndiElement( "current" ) );
        ip["current"].set( app.liveMag() + 2.0f );

        REQUIRE( app.newCallBack_m_indiP_mag( ip ) == 0 );
        REQUIRE( app.estimatedMag() == Approx( app.liveMag() ) );
    }

    SECTION( "star magnitude estimated writes are accepted" )
    {
        pcf::IndiProperty ip = makeLocalNumberProperty( "right", "star_mag" );
        ip.add( pcf::IndiElement( "estimated" ) );
        ip["estimated"].set( app.liveMag() + 2.0f );

        REQUIRE( app.newCallBack_m_indiP_mag( ip ) == 0 );
        REQUIRE( app.estimatedMag() == Approx( app.liveMag() + 2.0f ) );
    }

    SECTION( "live TCS seeing updates the current value" )
    {
        pcf::IndiProperty ip = makeRemoteNumberProperty( "tcsi", "seeing" );
        ip.add( pcf::IndiElement( "dimm_fwhm_corr" ) );
        ip["dimm_fwhm_corr"].set( 0.83f );

        REQUIRE( app.setCallBack_m_indiP_tcsi_seeing( ip ) == 0 );
        REQUIRE( app.liveSeeing() == Approx( 0.83f ) );
        REQUIRE( app.estimatedSeeing() == Approx( 0.83f ) );
    }

    SECTION( "local seeing current writes are ignored and estimated writes are accepted" )
    {
        pcf::IndiProperty live = makeRemoteNumberProperty( "tcsi", "seeing" );
        live.add( pcf::IndiElement( "dimm_fwhm_corr" ) );
        live["dimm_fwhm_corr"].set( 0.76f );
        REQUIRE( app.setCallBack_m_indiP_tcsi_seeing( live ) == 0 );

        pcf::IndiProperty currentWrite = makeLocalNumberProperty( "right", "seeing" );
        currentWrite.add( pcf::IndiElement( "current" ) );
        currentWrite["current"].set( 0.40f );
        REQUIRE( app.newCallBack_m_indiP_seeing_magaox( currentWrite ) == 0 );
        REQUIRE( app.estimatedSeeing() == Approx( 0.76f ) );

        pcf::IndiProperty estimatedWrite = makeLocalNumberProperty( "right", "seeing" );
        estimatedWrite.add( pcf::IndiElement( "estimated" ) );
        estimatedWrite["estimated"].set( 1.15f );
        REQUIRE( app.newCallBack_m_indiP_seeing_magaox( estimatedWrite ) == 0 );
        REQUIRE( app.estimatedSeeing() == Approx( 1.15f ) );
    }

    SECTION( "wind-speed selection writes update the planning wind speed" )
    {
        pcf::IndiProperty normal = makeLocalSwitchProperty( "right", "wind_speed" );
        normal.add( pcf::IndiElement( "normal" ) );
        normal["normal"].setSwitchState( pcf::IndiElement::On );
        REQUIRE( app.newCallBack_m_indiP_windSpeed( normal ) == 0 );
        REQUIRE( app.windSpeed() == Approx( 18.7f ) );
        REQUIRE( app.windSpeedProperty()["normal"].getSwitchState() == pcf::IndiElement::On );

        pcf::IndiProperty fast = makeLocalSwitchProperty( "right", "wind_speed" );
        fast.add( pcf::IndiElement( "fast" ) );
        fast["fast"].setSwitchState( pcf::IndiElement::On );
        REQUIRE( app.newCallBack_m_indiP_windSpeed( fast ) == 0 );
        REQUIRE( app.windSpeed() == Approx( 23.4f ) );
        REQUIRE( app.windSpeedProperty()["fast"].getSwitchState() == pcf::IndiElement::On );
    }
}

/// Verify estimate selection changes the published predictions and the optimum-loop-speed summary matches the fixed FPS
/// grid.
/**
 * \ingroup strehlEstimator_unit_test
 */
TEST_CASE( "strehlEstimator freezes auto-tracked estimates while use_estimates is enabled", "[strehlEstimator]" )
{
    // clang-format off
    #ifdef STREHLESTIMATOR_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( strehlEstimator::calcMag() );
    XWCTEST_DOXYGEN_REF( strehlEstimator::setCallBack_m_indiP_tcsi_seeing( std::declval<const pcf::IndiProperty &>() ) );
    XWCTEST_DOXYGEN_REF( strehlEstimator::newCallBack_m_indiP_useEstimates( std::declval<const pcf::IndiProperty &>() ) );
    #endif
    // clang-format on

    strehlEstimator_test app( "right" );

    REQUIRE( app.initializePublishedProperties() == 0 );

    app.setPhotometry( 30000.0f, 196 );

    pcf::IndiProperty initialSeeing = makeRemoteNumberProperty( "tcsi", "seeing" );
    initialSeeing.add( pcf::IndiElement( "dimm_fwhm_corr" ) );
    initialSeeing["dimm_fwhm_corr"].set( 0.55f );
    REQUIRE( app.setCallBack_m_indiP_tcsi_seeing( initialSeeing ) == 0 );

    const float frozenMag    = app.estimatedMag();
    const float frozenSeeing = app.estimatedSeeing();

    pcf::IndiProperty useEstimates = makeLocalSwitchProperty( "right", "use_estimates" );
    useEstimates.add( pcf::IndiElement( "toggle" ) );
    useEstimates["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_useEstimates( useEstimates ) == 0 );

    const float frozenPredictedStrehl = app.predictedStrehl();
    const float frozenOptimumFPS      = app.optimumFPS();

    app.setPhotometry( 12000.0f, 196 );

    pcf::IndiProperty updatedSeeing = makeRemoteNumberProperty( "tcsi", "seeing" );
    updatedSeeing.add( pcf::IndiElement( "dimm_fwhm_corr" ) );
    updatedSeeing["dimm_fwhm_corr"].set( 0.92f );
    REQUIRE( app.setCallBack_m_indiP_tcsi_seeing( updatedSeeing ) == 0 );

    REQUIRE( app.useEstimates() == true );
    REQUIRE( app.estimatedMag() == Approx( frozenMag ) );
    REQUIRE( app.estimatedSeeing() == Approx( frozenSeeing ) );
    REQUIRE( app.selectedMagDirect() == Approx( frozenMag ) );
    REQUIRE( app.selectedSeeingDirect() == Approx( frozenSeeing ) );
    REQUIRE( app.predictedStrehl() == Approx( frozenPredictedStrehl ) );
    REQUIRE( app.optimumFPS() == Approx( frozenOptimumFPS ) );
    REQUIRE( app.liveMag() != Approx( frozenMag ) );
    REQUIRE( app.liveSeeing() != Approx( frozenSeeing ) );
}

/// Verify estimate selection changes the published predictions and the optimum-loop-speed summary matches the fixed FPS
/// grid.
/**
 * \ingroup strehlEstimator_unit_test
 */
TEST_CASE( "strehlEstimator uses estimates when requested and scans the fixed loop-speed grid", "[strehlEstimator]" )
{
    // clang-format off
    #ifdef STREHLESTIMATOR_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( strehlEstimator::updatePredictionOutputs() );
    XWCTEST_DOXYGEN_REF( strehlEstimator::updateOptimumLoopSpeed() );
    XWCTEST_DOXYGEN_REF( strehlEstimator::newCallBack_m_indiP_useEstimates( std::declval<const pcf::IndiProperty &>() ) );
    #endif
    // clang-format on

    strehlEstimator_test app( "right" );

    REQUIRE( app.initializePublishedProperties() == 0 );

    app.setPhotometry( 30000.0f, 196 );

    pcf::IndiProperty liveSeeing = makeRemoteNumberProperty( "tcsi", "seeing" );
    liveSeeing.add( pcf::IndiElement( "dimm_fwhm_corr" ) );
    liveSeeing["dimm_fwhm_corr"].set( 0.55f );
    REQUIRE( app.setCallBack_m_indiP_tcsi_seeing( liveSeeing ) == 0 );

    pcf::IndiProperty liveElevation = makeRemoteNumberProperty( "tcsi", "telpos" );
    liveElevation.add( pcf::IndiElement( "el" ) );
    liveElevation["el"].set( 72.0f );
    REQUIRE( app.setCallBack_m_indiP_tcsi_telpos( liveElevation ) == 0 );

    const float liveSelectedMag     = app.selectedMagDirect();
    const float liveSelectedSeeing  = app.selectedSeeingDirect();
    const float livePredictedStrehl = app.predictedStrehl();

    pcf::IndiProperty magEstimate = makeLocalNumberProperty( "right", "star_mag" );
    magEstimate.add( pcf::IndiElement( "estimated" ) );
    magEstimate["estimated"].set( liveSelectedMag + 4.0f );
    REQUIRE( app.newCallBack_m_indiP_mag( magEstimate ) == 0 );

    pcf::IndiProperty seeingEstimate = makeLocalNumberProperty( "right", "seeing" );
    seeingEstimate.add( pcf::IndiElement( "estimated" ) );
    seeingEstimate["estimated"].set( 1.10f );
    REQUIRE( app.newCallBack_m_indiP_seeing_magaox( seeingEstimate ) == 0 );

    pcf::IndiProperty windSelection = makeLocalSwitchProperty( "right", "wind_speed" );
    windSelection.add( pcf::IndiElement( "normal" ) );
    windSelection["normal"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_windSpeed( windSelection ) == 0 );

    pcf::IndiProperty useEstimates = makeLocalSwitchProperty( "right", "use_estimates" );
    useEstimates.add( pcf::IndiElement( "toggle" ) );
    useEstimates["toggle"].setSwitchState( pcf::IndiElement::On );
    REQUIRE( app.newCallBack_m_indiP_useEstimates( useEstimates ) == 0 );

    REQUIRE( app.useEstimates() == true );
    REQUIRE( app.selectedMagDirect() == Approx( app.estimatedMag() ) );
    REQUIRE( app.selectedSeeingDirect() == Approx( app.estimatedSeeing() ) );
    REQUIRE( app.selectedMagDirect() != Approx( liveSelectedMag ) );
    REQUIRE( app.selectedSeeingDirect() != Approx( liveSelectedSeeing ) );
    REQUIRE( app.predictedStrehl() != Approx( livePredictedStrehl ) );

    float bruteForceBestFps    = 0.0f;
    float bruteForceBestStrehl = -1.0f;
    for( int fps = 100; fps <= 3000; fps += 100 )
    {
        float strehl = app.predictedStrehlAtFps( static_cast<float>( fps ), false );
        if( bruteForceBestFps == 0.0f || strehl > bruteForceBestStrehl )
        {
            bruteForceBestFps    = static_cast<float>( fps );
            bruteForceBestStrehl = strehl;
        }
    }

    REQUIRE( app.optimumFPS() >= 100.0f );
    REQUIRE( app.optimumFPS() <= 3000.0f );
    REQUIRE( std::fmod( app.optimumFPS(), 100.0f ) == Approx( 0.0f ).margin( 1.0e-4 ) );
    REQUIRE( app.optimumFPS() == Approx( bruteForceBestFps ) );
    REQUIRE( app.optimumStrehl() == Approx( bruteForceBestStrehl ) );
}

} // namespace strehlEstimatorTest

} // namespace libXWCTest
