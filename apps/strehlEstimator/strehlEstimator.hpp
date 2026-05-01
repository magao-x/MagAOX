/** \file strehlEstimator.hpp
 * \brief Declares the `strehlEstimator` MagAO-X application.
 *
 * \ingroup strehlEstimator_files
 */

#ifndef strehlEstimator_hpp
#define strehlEstimator_hpp

#include <cmath>

#include <mx/ao/analysis/aoSystem.hpp>
using namespace mx::math;

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup strehlEstimator
 * \brief Predicts Strehl and WFE for live or operator-estimated observing conditions.
 *
 * <a href="../handbook/operating/software/apps/XXXXXX.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup strehlEstimator_files
 * \ingroup strehlEstimator
 */

namespace MagAOX
{
namespace app
{

/// Tag type for the live WFS average shmim monitor.
struct wfsavgShmimT
{
    /// Return the configuration section name for this shmim monitor.
    static std::string configSection()
    {
        return "wfsavgShmim";
    };

    /// Return the INDI prefix for this shmim monitor.
    static std::string indiPrefix()
    {
        return "wfsavg";
    };
};

/// Tag type for the WFS mask shmim monitor.
struct wfsmaskShmimT
{
    /// Return the configuration section name for this shmim monitor.
    static std::string configSection()
    {
        return "wfsmaskShmim";
    };

    /// Return the INDI prefix for this shmim monitor.
    static std::string indiPrefix()
    {
        return "wfsmask";
    };
};

/// Predicts Strehl and WFE from live WFS telemetry and optional planning overrides.
/**
 * \ingroup strehlEstimator
 */
class strehlEstimator : public MagAOXApp<true>,
                        dev::shmimMonitor<strehlEstimator, wfsavgShmimT>,
                        dev::shmimMonitor<strehlEstimator, wfsmaskShmimT>
{

    // Give the test harness access.
    friend class strehlEstimator_test;

    friend class dev::shmimMonitor<strehlEstimator, wfsavgShmimT>;
    friend class dev::shmimMonitor<strehlEstimator, wfsmaskShmimT>;

  public:
    typedef dev::shmimMonitor<strehlEstimator, wfsavgShmimT>                              wfsavgShmimMonitorT;
    typedef dev::shmimMonitor<strehlEstimator, wfsmaskShmimT>                             wfsmaskShmimMonitorT;
    typedef mx::AO::analysis::aoSystem<float, mx::AO::analysis::vonKarmanSpectrum<float>> aoSystemT;

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    /// Loop number used to resolve the WFS shmim names.
    int m_loopNum{ 1 };

    /// WFS device providing the live FPS property.
    std::string m_wfsDevice{ "camwfs" };

    /// Beamsplitter stage device used to choose the active photometric calibration.
    std::string m_stagebsDevice{ "stagebs" };

    /// Analog gain factor converting WFS counts into photo-electrons.
    float m_again{ 28.547f };

    /// Active WFS quantum efficiency for the currently selected beamsplitter branch.
    float m_qe{ 0.53f };

    /// Zero-magnitude photon flux for the 65/35 beamsplitter branch.
    float m_F0_6535{ 4.2e10f };

    /// Zero-magnitude photon flux for the Ha/IR beamsplitter branch.
    float m_F0_HaIR{ 5.3e10f };

    /// Effective WFS wavelength in microns for the 65/35 beamsplitter branch.
    float m_lam0_6535{ 0.791f };

    /// Effective WFS wavelength in microns for the Ha/IR beamsplitter branch.
    float m_lam0_HaIR{ 0.837f };

    /// WFS QE for the 65/35 beamsplitter branch.
    float m_qe_6535{ 0.53f };

    /// WFS QE for the Ha/IR beamsplitter branch.
    float m_qe_HaIR{ 0.53f };

    ///@}

    /** \name Runtime State - Data
     *@{
     */

    /// Live WFS frame rate in Hz.
    float m_fps{ 2000.0f };

    /// Live EM gain reported by the WFS camera.
    float m_emg{ 1.0f };

    /// Active zero-magnitude photon flux for the selected beamsplitter branch.
    float m_F0{ m_F0_6535 };

    /// Active WFS/science wavelength in microns for the selected beamsplitter branch.
    float m_lam0{ m_lam0_6535 };

    /// Live seeing estimate in arcseconds from `tcsi.seeing.dimm_fwhm_corr`.
    float m_seeing{ 0.64f };

    /// Live Fried parameter corresponding to `m_seeing`.
    float m_r0{ 0.2063f * 0.5f / 0.64f };

    /// Telescope elevation in degrees.
    float m_elevation{ 90.0f };

    /// Number of illuminated WFS pixels in the current mask.
    int m_npix{ 0 };

    /// Total masked WFS counts used to derive the live guide-star magnitude.
    float m_counts{ 0.0f };

    /// Live guide-star magnitude derived from `m_counts`.
    float m_mag{ 0.0f };

    /// Operator-entered star magnitude used when planning overrides are enabled.
    float m_magEstimated{ 0.0f };

    /// Tracks whether the estimated star magnitude has been explicitly set by an operator.
    bool m_magEstimatedManual{ false };

    /// Operator-entered seeing in arcseconds used when planning overrides are enabled.
    float m_seeingEstimated{ 0.64f };

    /// Tracks whether the estimated seeing has been explicitly set by an operator.
    bool m_seeingEstimatedManual{ false };

    /// Operator-entered wind speed in m/s used for planning calculations.
    float m_windSpeedEstimated{ 10.0f };

    /// Selects whether predicted outputs use the live or estimated planning inputs.
    bool m_useEstimates{ false };

    /// Latest WFS mask image.
    mx::improc::eigenImage<float> m_wfsmask;

    /// Latest WFS average image.
    mx::improc::eigenImage<float> m_wfsavg;

    /// AO model used for the current predicted Strehl and WFE outputs.
    aoSystemT m_aosys;

    /// AO model dedicated to the fixed-FPS optimum-loop-speed scan.
    aoSystemT m_aosysScan;

    /// Latest DIMM elevation-corrected FWHM.
    double m_dimm_fwhm_corr{ 0.0 };

    /// Seconds since midnight of the latest DIMM measurement.
    int m_dimm_time{ 0 };

    /// Latest MAG1 elevation-corrected FWHM.
    double m_mag1_fwhm_corr{ 0.0 };

    /// Seconds since midnight of the latest MAG1 measurement.
    int m_mag1_time{ 0 };

    /// Latest MAG2 elevation-corrected FWHM.
    double m_mag2_fwhm_corr{ 0.0 };

    /// Seconds since midnight of the latest MAG2 measurement.
    int m_mag2_time{ 0 };

    ///@}

  public:
    /// Construct the application with the compiled git-version metadata.
    strehlEstimator();

    /// Destroy the application.
    ~strehlEstimator() noexcept
    {
    }

    /// Declare configuration keys and initialize the AO models.
    virtual void setupConfig();

    /// Load configuration values after `setupConfig()` has registered them.
    /**
     * This is split from `loadConfig()` so the unit tests can call it directly.
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] application configuration source to read from */ );

    /// Load the configured runtime values.
    virtual void loadConfig();

    /// Register INDI properties and transition the app into the operating state.
    virtual int appStartup();

    /// Refresh the AO predictions and service the shmim-monitor state machine.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shut down the shmim monitors.
    virtual int appShutdown();

    /// React to allocation of the WFS average shmim stream.
    int allocate( const wfsavgShmimT &dummy /**< [in] tag distinguishing the shmimMonitor parent */ );

    /// Process one WFS average frame.
    int processImage( void               *curr_src, /**< [in] pointer to the start of the current frame */
                      const wfsavgShmimT &dummy /**< [in] tag distinguishing the shmimMonitor parent */ );

    /// React to allocation of the WFS mask shmim stream.
    int allocate( const wfsmaskShmimT &dummy /**< [in] tag distinguishing the shmimMonitor parent */ );

    /// Process one WFS mask frame.
    int processImage( void                *curr_src, /**< [in] pointer to the start of the current frame */
                      const wfsmaskShmimT &dummy /**< [in] tag distinguishing the shmimMonitor parent */ );

    /// Recalculate the live guide-star magnitude from the current WFS counts.
    void calcMag();

    /// Return the selected star magnitude for prediction calculations.
    float selectedStarMag() const;

    /// Return the selected seeing for prediction calculations.
    float selectedSeeing() const;

    /// Return the selected wind speed for prediction calculations.
    float selectedWindSpeed() const;

    /// Convert seeing in arcseconds to Fried parameter `r0` in meters.
    static float seeingToR0( float seeing /**< [in] seeing in arcseconds */ );

    /// Return whether a value is finite.
    static bool finiteValue( float value /**< [in] value to test */ );

    /// Return whether a value is finite and strictly positive.
    static bool finitePositiveValue( float value /**< [in] value to test */ );

    /// Convert AO model phase variance into WFE in nm RMS at the active wavelength.
    float wfeNm( float variance /**< [in] phase variance at the science wavelength */ ) const;

    /// Create a writable number property with `current` and `estimated` elements.
    int createCurrentEstimatedProperty( pcf::IndiProperty &prop,  /**< [out] property to initialize */
                                        const std::string &name,  /**< [in] INDI property name */
                                        const std::string &label, /**< [in] suggested GUI label */
                                        const std::string &group  /**< [in] suggested GUI group */
    );

    /// Update the published planning-input properties from the current runtime state.
    void updatePlanningProperties();

    /// Configure an AO model for the selected inputs and requested loop speed.
    void configureAoSystem( aoSystemT &aosys,      /**< [in,out] AO model instance to configure */
                            float      fps,        /**< [in] loop speed in Hz */
                            bool       optimizeTau /**< [in] true to preserve the current optimal-tau behavior */
    );

    /// Refresh the predicted Strehl, WFE, and optimum-loop-speed properties.
    void updatePredictionOutputs();

    /// Refresh the fixed-grid optimum-loop-speed summary property.
    void updateOptimumLoopSpeed();

    /** \name INDI - Data
     * @{
     */

    /// Subscription to the live WFS FPS property.
    pcf::IndiProperty m_indiP_fps;

    /// Subscription to the live WFS EM-gain property.
    pcf::IndiProperty m_indiP_emg;

    /// Subscription to the beamsplitter preset state.
    pcf::IndiProperty m_indiP_stage;

    /// Subscription to the TCS seeing property.
    pcf::IndiProperty m_indiP_tcsi_seeing;

    /// Subscription to the TCS telescope position property.
    pcf::IndiProperty m_indiP_tcsi_telpos;

    /// Local writable seeing property exposing `current` and `estimated`.
    pcf::IndiProperty m_indiP_seeing_magaox;

    /// Local writable star-magnitude property exposing `current` and `estimated`.
    pcf::IndiProperty m_indiP_mag;

    /// Local writable wind-speed property exposing `current` and `estimated`.
    pcf::IndiProperty m_indiP_windSpeed;

    /// Local toggle selecting whether predicted outputs use estimated inputs.
    pcf::IndiProperty m_indiP_useEstimates;

    /// Predicted Strehl property for the currently selected conditions.
    pcf::IndiProperty m_indiP_strehl;

    /// Predicted WFE breakdown for the currently selected conditions.
    pcf::IndiProperty m_indiP_wfe;

    /// Summary property for the best fixed-grid loop speed.
    pcf::IndiProperty m_indiP_loopSpeedOptimum;

    /// Callback for live FPS updates.
    INDI_SETCALLBACK_DECL( strehlEstimator, m_indiP_fps );

    /// Callback for live EM-gain updates.
    INDI_SETCALLBACK_DECL( strehlEstimator, m_indiP_emg );

    /// Callback for beamsplitter preset updates.
    INDI_SETCALLBACK_DECL( strehlEstimator, m_indiP_stage );

    /// Callback for live TCS seeing updates.
    INDI_SETCALLBACK_DECL( strehlEstimator, m_indiP_tcsi_seeing );

    /// Callback for live TCS elevation updates.
    INDI_SETCALLBACK_DECL( strehlEstimator, m_indiP_tcsi_telpos );

    /// Callback for local star-magnitude estimate writes.
    INDI_NEWCALLBACK_DECL( strehlEstimator, m_indiP_mag );

    /// Callback for local seeing estimate writes.
    INDI_NEWCALLBACK_DECL( strehlEstimator, m_indiP_seeing_magaox );

    /// Callback for local wind-speed estimate writes.
    INDI_NEWCALLBACK_DECL( strehlEstimator, m_indiP_windSpeed );

    /// Callback for the `use_estimates` toggle.
    INDI_NEWCALLBACK_DECL( strehlEstimator, m_indiP_useEstimates );

    ///@}
};

strehlEstimator::strehlEstimator() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    wfsavgShmimMonitorT::m_getExistingFirst  = true;
    wfsmaskShmimMonitorT::m_getExistingFirst = true;

    return;
}

void strehlEstimator::setupConfig()
{
    m_aosys.loadMagAOX();
    m_aosysScan.loadMagAOX();

    m_windSpeedEstimated = m_aosys.atm.v_wind();

    config.add( "loop.number",
                "",
                "loop.number",
                argType::Required,
                "loop",
                "number",
                false,
                "int",
                "The number of the loop. Used to set shmim names, as in aolN_mgainfact." );

    config.add( "phot.qe_6535",
                "",
                "phot.qe_6535",
                argType::Required,
                "phot",
                "qe_6535",
                false,
                "float",
                "The WFS QE in the 65-35 B/S." );

    config.add( "phot.qe_HaIR",
                "",
                "phot.qe_HaIR",
                argType::Required,
                "phot",
                "qe_HaIR",
                false,
                "float",
                "The WFS QE in the Ha-IR B/S." );

    SHMIMMONITORT_SETUP_CONFIG( wfsavgShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( wfsmaskShmimMonitorT, config );
}

int strehlEstimator::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_loopNum, "loop.number" );

    _config( m_qe_6535, "phot.qe_6535" );
    _config( m_qe_HaIR, "phot.qe_HaIR" );

    char shmim[1024];
    snprintf( shmim, sizeof( shmim ), "aol%d_wfsavg", m_loopNum );
    wfsavgShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( wfsavgShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_wfsmask", m_loopNum );
    wfsmaskShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( wfsmaskShmimMonitorT, _config );

    return 0;
}

void strehlEstimator::loadConfig()
{
    loadConfigImpl( config );
}

int strehlEstimator::createCurrentEstimatedProperty( pcf::IndiProperty &prop,
                                                     const std::string &name,
                                                     const std::string &label,
                                                     const std::string &group )
{
    prop = pcf::IndiProperty( pcf::IndiProperty::Number );
    prop.setDevice( configName() );
    prop.setName( name );
    prop.setPerm( pcf::IndiProperty::ReadWrite );
    prop.setState( pcf::IndiProperty::Idle );

    if( label != "" )
    {
        prop.setLabel( label );
    }

    if( group != "" )
    {
        prop.setGroup( group );
    }

    prop.add( pcf::IndiElement( "current", 0.0f ) );
    prop.add( pcf::IndiElement( "estimated", 0.0f ) );

    return 0;
}

float strehlEstimator::selectedStarMag() const
{
    if( m_useEstimates )
    {
        return m_magEstimated;
    }

    return m_mag;
}

float strehlEstimator::selectedSeeing() const
{
    if( m_useEstimates )
    {
        return m_seeingEstimated;
    }

    return m_seeing;
}

float strehlEstimator::selectedWindSpeed() const
{
    return m_windSpeedEstimated;
}

float strehlEstimator::seeingToR0( float seeing )
{
    return 0.2063f * 0.5f / seeing;
}

bool strehlEstimator::finiteValue( float value )
{
    return std::isfinite( value );
}

bool strehlEstimator::finitePositiveValue( float value )
{
    return finiteValue( value ) && value > 0.0f;
}

float strehlEstimator::wfeNm( float variance ) const
{
    if( variance <= 0.0f )
    {
        return 0.0f;
    }

    return std::sqrt( variance ) * ( 1000.0f * m_lam0 / two_pi<float>() );
}

void strehlEstimator::updatePlanningProperties()
{
    if( !m_indiDriver )
    {
        m_indiP_mag["current"].set( m_mag );
        m_indiP_mag["estimated"].set( m_magEstimated );
        m_indiP_mag.setState( INDI_OK );

        m_indiP_seeing_magaox["current"].set( m_seeing );
        m_indiP_seeing_magaox["estimated"].set( m_seeingEstimated );
        m_indiP_seeing_magaox.setState( INDI_OK );

        m_indiP_windSpeed["current"].set( m_windSpeedEstimated );
        m_indiP_windSpeed["estimated"].set( m_windSpeedEstimated );
        m_indiP_windSpeed.setState( INDI_OK );

        m_indiP_useEstimates["toggle"].setSwitchState( m_useEstimates ? pcf::IndiElement::On : pcf::IndiElement::Off );
        m_indiP_useEstimates.setState( m_useEstimates ? INDI_OK : INDI_IDLE );

        return;
    }

    updatesIfChanged<float>( m_indiP_mag, { "current", "estimated" }, { m_mag, m_magEstimated } );
    updatesIfChanged<float>( m_indiP_seeing_magaox, { "current", "estimated" }, { m_seeing, m_seeingEstimated } );
    updatesIfChanged<float>(
        m_indiP_windSpeed, { "current", "estimated" }, { m_windSpeedEstimated, m_windSpeedEstimated } );
    updateSwitchIfChanged( m_indiP_useEstimates,
                           "toggle",
                           m_useEstimates ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           m_useEstimates ? INDI_OK : INDI_IDLE );
}

void strehlEstimator::configureAoSystem( aoSystemT &aosys, float fps, bool optimizeTau )
{
    aosys.optTau( optimizeTau );
    aosys.starMag( selectedStarMag() );
    aosys.F0( m_qe * m_F0 );
    aosys.lam_wfs( m_lam0 * 1.0e-6f );
    aosys.lam_sci( m_lam0 * 1.0e-6f );
    aosys.ron_wfs( std::vector<float>( { 245.0f / m_emg } ) );
    aosys.npix_wfs( std::vector<float>( { static_cast<float>( m_npix ) } ) );
    aosys.minTauWFS( std::vector<float>( { 1.0f / fps } ) );
    aosys.tauWFS( 1.0f / fps );
    aosys.atm.r_0( seeingToR0( selectedSeeing() ), 0.5e-6f );
    aosys.atm.v_wind( selectedWindSpeed() );
    aosys.zeta( ( 90.0f - m_elevation ) * pi<float>() / 180.0f );
}

void strehlEstimator::updateOptimumLoopSpeed()
{
    float bestFPS            = 0.0f;
    float bestStrehl         = -1.0f;
    float bestTotalWfe       = 0.0f;
    float bestMeasurementWfe = 0.0f;
    float bestTimeDelayWfe   = 0.0f;
    float bestFittingWfe     = 0.0f;

    for( int fps = 100; fps <= 3000; fps += 100 )
    {
        configureAoSystem( m_aosysScan, static_cast<float>( fps ), false );

        float strehl = m_aosysScan.strehl();
        if( !finiteValue( strehl ) )
        {
            continue;
        }

        if( bestFPS == 0.0f || strehl > bestStrehl )
        {
            bestFPS            = static_cast<float>( fps );
            bestStrehl         = strehl;
            bestTotalWfe       = wfeNm( m_aosysScan.wfeVar() );
            bestMeasurementWfe = wfeNm( m_aosysScan.measurementErrorTotal() );
            bestTimeDelayWfe   = wfeNm( m_aosysScan.timeDelayErrorTotal() );
            bestFittingWfe     = wfeNm( m_aosysScan.fittingErrorTotal() );
        }
    }

    if( !m_indiDriver )
    {
        m_indiP_loopSpeedOptimum["fps"].set( bestFPS );
        m_indiP_loopSpeedOptimum["strehl"].set( bestStrehl );
        m_indiP_loopSpeedOptimum["wfe_total"].set( bestTotalWfe );
        m_indiP_loopSpeedOptimum["wfe_measurement"].set( bestMeasurementWfe );
        m_indiP_loopSpeedOptimum["wfe_time_delay"].set( bestTimeDelayWfe );
        m_indiP_loopSpeedOptimum["wfe_fitting"].set( bestFittingWfe );
        m_indiP_loopSpeedOptimum.setState( INDI_OK );
        return;
    }

    updatesIfChanged<float>(
        m_indiP_loopSpeedOptimum,
        { "fps", "strehl", "wfe_total", "wfe_measurement", "wfe_time_delay", "wfe_fitting" },
        { bestFPS, bestStrehl, bestTotalWfe, bestMeasurementWfe, bestTimeDelayWfe, bestFittingWfe } );
}

void strehlEstimator::updatePredictionOutputs()
{
    if( !finitePositiveValue( m_fps ) || !finitePositiveValue( m_emg ) || !finitePositiveValue( m_qe ) ||
        !finitePositiveValue( m_F0 ) || !finitePositiveValue( selectedSeeing() ) ||
        !finitePositiveValue( selectedWindSpeed() ) )
    {
        return;
    }

    configureAoSystem( m_aosys, m_fps, true );

    if( !m_indiDriver )
    {
        m_indiP_strehl["pyramid"].set( m_aosys.strehl() );
        m_indiP_strehl.setState( INDI_OK );

        m_indiP_wfe["total"].set( wfeNm( m_aosys.wfeVar() ) );
        m_indiP_wfe["measurement"].set( wfeNm( m_aosys.measurementErrorTotal() ) );
        m_indiP_wfe["time_delay"].set( wfeNm( m_aosys.timeDelayErrorTotal() ) );
        m_indiP_wfe["fitting"].set( wfeNm( m_aosys.fittingErrorTotal() ) );
        m_indiP_wfe.setState( INDI_OK );
    }
    else
    {
        updateIfChanged( m_indiP_strehl, "pyramid", m_aosys.strehl() );
        updatesIfChanged<float>( m_indiP_wfe,
                                 { "total", "measurement", "time_delay", "fitting" },
                                 { wfeNm( m_aosys.wfeVar() ),
                                   wfeNm( m_aosys.measurementErrorTotal() ),
                                   wfeNm( m_aosys.timeDelayErrorTotal() ),
                                   wfeNm( m_aosys.fittingErrorTotal() ) } );
    }

    updateOptimumLoopSpeed();
}

int strehlEstimator::appStartup()
{
    SHMIMMONITORT_APP_STARTUP( wfsavgShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( wfsmaskShmimMonitorT );

    REG_INDI_SETPROP( m_indiP_fps, m_wfsDevice, "fps" );
    REG_INDI_SETPROP( m_indiP_emg, m_wfsDevice, "emgain" );
    REG_INDI_SETPROP( m_indiP_stage, m_stagebsDevice, "presetName" );
    REG_INDI_SETPROP( m_indiP_tcsi_seeing, "tcsi", "seeing" );
    REG_INDI_SETPROP( m_indiP_tcsi_telpos, "tcsi", "telpos" );

    if( createCurrentEstimatedProperty( m_indiP_mag, "star_mag", "Star Magnitude", "Error Budget" ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from createCurrentEstimatedProperty" } );
    }
    if( registerIndiPropertyNew( m_indiP_mag, INDI_NEWCALLBACK( m_indiP_mag ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from registerIndiPropertyNew" } );
    }

    if( createCurrentEstimatedProperty( m_indiP_seeing_magaox, "seeing", "Seeing", "Error Budget" ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from createCurrentEstimatedProperty" } );
    }
    if( registerIndiPropertyNew( m_indiP_seeing_magaox, INDI_NEWCALLBACK( m_indiP_seeing_magaox ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from registerIndiPropertyNew" } );
    }

    if( createCurrentEstimatedProperty( m_indiP_windSpeed, "wind_speed", "Wind Speed", "Error Budget" ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from createCurrentEstimatedProperty" } );
    }
    if( registerIndiPropertyNew( m_indiP_windSpeed, INDI_NEWCALLBACK( m_indiP_windSpeed ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from registerIndiPropertyNew" } );
    }

    if( createStandardIndiToggleSw( m_indiP_useEstimates, "use_estimates", "Use Estimates", "Error Budget" ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from createStandardIndiToggleSw" } );
    }
    if( registerIndiPropertyNew( m_indiP_useEstimates, INDI_NEWCALLBACK( m_indiP_useEstimates ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from registerIndiPropertyNew" } );
    }

    CREATE_REG_INDI_RO_NUMBER( m_indiP_strehl, "strehl_optimal", "Strehl", "Error Budget" );
    m_indiP_strehl.add( pcf::IndiElement( "pyramid", 0.0f ) );

    CREATE_REG_INDI_RO_NUMBER( m_indiP_wfe, "wfe_predicted", "WFE", "Error Budget" );
    m_indiP_wfe.add( pcf::IndiElement( "total", 0.0f ) );
    m_indiP_wfe.add( pcf::IndiElement( "measurement", 0.0f ) );
    m_indiP_wfe.add( pcf::IndiElement( "time_delay", 0.0f ) );
    m_indiP_wfe.add( pcf::IndiElement( "fitting", 0.0f ) );

    CREATE_REG_INDI_RO_NUMBER( m_indiP_loopSpeedOptimum, "loop_speed_optimum", "Optimum Loop Speed", "Error Budget" );
    m_indiP_loopSpeedOptimum.add( pcf::IndiElement( "fps", 0.0f ) );
    m_indiP_loopSpeedOptimum.add( pcf::IndiElement( "strehl", 0.0f ) );
    m_indiP_loopSpeedOptimum.add( pcf::IndiElement( "wfe_total", 0.0f ) );
    m_indiP_loopSpeedOptimum.add( pcf::IndiElement( "wfe_measurement", 0.0f ) );
    m_indiP_loopSpeedOptimum.add( pcf::IndiElement( "wfe_time_delay", 0.0f ) );
    m_indiP_loopSpeedOptimum.add( pcf::IndiElement( "wfe_fitting", 0.0f ) );

    updatePlanningProperties();
    updatePredictionOutputs();

    state( stateCodes::OPERATING );

    return 0;
}

int strehlEstimator::appLogic()
{
    SHMIMMONITORT_APP_LOGIC( wfsavgShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( wfsmaskShmimMonitorT );

    SHMIMMONITORT_UPDATE_INDI( wfsavgShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( wfsmaskShmimMonitorT );

    updatePlanningProperties();
    updatePredictionOutputs();

    return 0;
}

int strehlEstimator::appShutdown()
{
    SHMIMMONITORT_APP_SHUTDOWN( wfsavgShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( wfsmaskShmimMonitorT );

    return 0;
}

int strehlEstimator::allocate( const wfsavgShmimT &dummy )
{
    static_cast<void>( dummy );

    std::cerr << "Got WFS avg: " << wfsavgShmimMonitorT::m_width << " x " << wfsavgShmimMonitorT::m_height << '\n';
    return 0;
}

int strehlEstimator::processImage( void *curr_src, const wfsavgShmimT &dummy )
{
    static_cast<void>( dummy );

    m_wfsavg = mx::improc::eigenMap<float>(
        reinterpret_cast<float *>( curr_src ), wfsavgShmimMonitorT::m_width, wfsavgShmimMonitorT::m_height );

    if( m_wfsavg.rows() == m_wfsmask.rows() && m_wfsavg.cols() == m_wfsmask.cols() )
    {
        m_counts = ( m_wfsavg * m_wfsmask ).sum();

        std::cerr << "counts: " << m_counts << '\n';

        calcMag();
    }

    return 0;
}

int strehlEstimator::allocate( const wfsmaskShmimT &dummy )
{
    static_cast<void>( dummy );

    std::cerr << "Got WFS mask: " << wfsmaskShmimMonitorT::m_width << " x " << wfsmaskShmimMonitorT::m_height << '\n';
    return 0;
}

int strehlEstimator::processImage( void *curr_src, const wfsmaskShmimT &dummy )
{
    static_cast<void>( dummy );

    m_wfsmask = mx::improc::eigenMap<float>(
        reinterpret_cast<float *>( curr_src ), wfsmaskShmimMonitorT::m_width, wfsmaskShmimMonitorT::m_height );

    m_npix = static_cast<int>( m_wfsmask.sum() );

    if( m_wfsavg.rows() == m_wfsmask.rows() && m_wfsavg.cols() == m_wfsmask.cols() )
    {
        // update counts because we might have been waiting on this.
        m_counts = ( m_wfsavg * m_wfsmask ).sum();

        calcMag();
    }

    return 0;
}

void strehlEstimator::calcMag()
{
    std::cerr << "calcMag: " << m_counts << ' ' << m_again << ' ' << ' ' << m_emg << ' ' << m_fps << ' ' << m_qe << ' '
              << m_F0 << '\n';

    if( !finitePositiveValue( m_counts ) || !finitePositiveValue( m_again ) || !finitePositiveValue( m_emg ) ||
        !finitePositiveValue( m_fps ) || !finitePositiveValue( m_qe ) || !finitePositiveValue( m_F0 ) )
    {
        return;
    }

    m_mag = -2.5f * std::log10( m_counts * m_again / m_emg * m_fps / ( m_qe * m_F0 ) );

    if( !m_magEstimatedManual )
    {
        m_magEstimated = m_mag;
    }

    updatePlanningProperties();
    updatePredictionOutputs();
}

INDI_SETCALLBACK_DEFN( strehlEstimator, m_indiP_fps )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fps, ipRecv );

    if( ipRecv.find( "current" ) )
    {
        float fps = ipRecv["current"].get<float>();

        if( finitePositiveValue( fps ) && fps != m_fps )
        {
            m_fps = fps;
            std::cerr << "Got FPS: " << m_fps << '\n';

            calcMag();
        }
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( strehlEstimator, m_indiP_emg )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_emg, ipRecv );

    if( ipRecv.find( "current" ) )
    {
        float emg = ipRecv["current"].get<float>();

        if( finitePositiveValue( emg ) && emg != m_emg )
        {
            m_emg = emg;
            std::cerr << "Got EMG: " << m_emg << '\n';

            calcMag();
        }
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( strehlEstimator, m_indiP_stage )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stage, ipRecv );

    std::string preset = "none";

    for( auto &&el : ipRecv.getElements() )
    {
        if( el.second.getSwitchState() == pcf::IndiElement::On )
        {
            preset = el.first;
            break;
        }
    }

    std::cerr << "Got stage bs: " << preset << '\n';

    if( preset == "ha-ir" )
    {
        m_F0   = m_F0_HaIR;
        m_lam0 = m_lam0_HaIR;
        m_qe   = m_qe_HaIR;
    }
    else
    {
        m_F0   = m_F0_6535;
        m_lam0 = m_lam0_6535;
        m_qe   = m_qe_6535;
    }

    calcMag();

    return 0;
}

INDI_SETCALLBACK_DEFN( strehlEstimator, m_indiP_tcsi_seeing )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_tcsi_seeing, ipRecv );

    if( ipRecv.find( "dimm_fwhm_corr" ) )
    {
        float seeing = ipRecv["dimm_fwhm_corr"].get<float>();

        if( finitePositiveValue( seeing ) && seeing != m_seeing )
        {
            m_seeing         = seeing;
            m_r0             = seeingToR0( m_seeing );
            m_dimm_fwhm_corr = seeing;

            if( !m_seeingEstimatedManual )
            {
                m_seeingEstimated = m_seeing;
            }

            std::cerr << "Got seeing: " << m_seeing << '\n';

            updatePlanningProperties();
            updatePredictionOutputs();
        }
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( strehlEstimator, m_indiP_tcsi_telpos )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_tcsi_telpos, ipRecv );

    if( ipRecv.find( "el" ) )
    {
        float elevation = ipRecv["el"].get<float>();

        if( finiteValue( elevation ) && elevation != m_elevation )
        {
            m_elevation = elevation;
            std::cerr << "Got elevation: " << m_elevation << '\n';

            updatePredictionOutputs();
        }
    }
    return 0;
}

INDI_NEWCALLBACK_DEFN( strehlEstimator, m_indiP_mag )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_mag, ipRecv );

    if( ipRecv.find( "estimated" ) )
    {
        float mag = ipRecv["estimated"].get<float>();

        if( finiteValue( mag ) && mag != m_magEstimated )
        {
            m_magEstimated       = mag;
            m_magEstimatedManual = true;
            updatePlanningProperties();
            updatePredictionOutputs();
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( strehlEstimator, m_indiP_seeing_magaox )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_seeing_magaox, ipRecv );

    if( ipRecv.find( "estimated" ) )
    {
        float seeing = ipRecv["estimated"].get<float>();

        if( finitePositiveValue( seeing ) && seeing != m_seeingEstimated )
        {
            m_seeingEstimated       = seeing;
            m_seeingEstimatedManual = true;
            updatePlanningProperties();
            updatePredictionOutputs();
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( strehlEstimator, m_indiP_windSpeed )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_windSpeed, ipRecv );

    if( ipRecv.find( "estimated" ) )
    {
        float windSpeed = ipRecv["estimated"].get<float>();

        if( finitePositiveValue( windSpeed ) && windSpeed != m_windSpeedEstimated )
        {
            m_windSpeedEstimated = windSpeed;
            updatePlanningProperties();
            updatePredictionOutputs();
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( strehlEstimator, m_indiP_useEstimates )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_useEstimates, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    bool useEstimates = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;

    if( useEstimates != m_useEstimates )
    {
        m_useEstimates = useEstimates;
        updatePlanningProperties();
        updatePredictionOutputs();
    }
    else
    {
        updateSwitchIfChanged( m_indiP_useEstimates,
                               "toggle",
                               m_useEstimates ? pcf::IndiElement::On : pcf::IndiElement::Off,
                               m_useEstimates ? INDI_OK : INDI_IDLE );
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // strehlEstimator_hpp
