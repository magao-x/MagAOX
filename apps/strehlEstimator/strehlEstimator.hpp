/** \file strehlEstimator.hpp
 * \brief Declares the `strehlEstimator` MagAO-X application.
 *
 * \ingroup strehlEstimator_files
 */

#ifndef strehlEstimator_hpp
#define strehlEstimator_hpp

#include <cmath>
#include <mutex>
#include <sstream>

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

    /// Operator-selected wind speed in m/s used for planning calculations.
    float m_windSpeed{ 9.4f };

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

    /// Protects the live telemetry and planning-input state while prediction snapshots are assembled.
    mutable std::mutex m_stateMutex;

    /// Serializes access to the shared AO-model instances and prediction-property updates.
    mutable std::mutex m_predictionMutex;

    /// Tracks the last suspicious optimum FPS reported in debug logging.
    float m_lastLoggedSuspiciousOptimumFps{ -1.0f };

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

    /// Snapshot of the scalar inputs used to update the AO prediction model.
    struct predictionInputs
    {
        /// Live loop speed in Hz.
        float m_fps{ 0.0f };

        /// Live EM gain.
        float m_emg{ 0.0f };

        /// Active quantum efficiency.
        float m_qe{ 0.0f };

        /// Active zero-magnitude photon flux.
        float m_F0{ 0.0f };

        /// Active wavelength in microns.
        float m_lam0{ 0.0f };

        /// Telescope elevation in degrees.
        float m_elevation{ 0.0f };

        /// Current illuminated WFS-pixel count.
        int m_npix{ 0 };

        /// Live guide-star magnitude.
        float m_mag{ 0.0f };

        /// Operator-entered guide-star magnitude estimate.
        float m_magEstimated{ 0.0f };

        /// Selected guide-star magnitude used for prediction.
        float m_selectedMag{ 0.0f };

        /// Live seeing in arcseconds.
        float m_seeing{ 0.0f };

        /// Operator-entered seeing estimate in arcseconds.
        float m_seeingEstimated{ 0.0f };

        /// Selected seeing used for prediction.
        float m_selectedSeeing{ 0.0f };

        /// Operator-selected wind speed in m/s.
        float m_windSpeed{ 0.0f };

        /// Selected wind speed used for prediction.
        float m_selectedWindSpeed{ 0.0f };

        /// Whether estimated planning inputs are currently selected.
        bool m_useEstimates{ false };
    };

    /// Snapshot the scalar state used by the planning properties and AO-model calculations.
    predictionInputs snapshotPredictionInputs() const;

    /// Return the selected star magnitude for prediction calculations.
    float selectedStarMag() const;

    /// Return the selected seeing for prediction calculations.
    float selectedSeeing() const;

    /// Return the selected wind speed for prediction calculations.
    float selectedWindSpeed() const;

    /// Return the supported wind-speed selection element names.
    static const std::vector<std::string> &windSpeedSelectionElements();

    /// Return the supported wind-speed selection labels.
    static const std::vector<std::string> &windSpeedSelectionLabels();

    /// Convert a wind-speed selection element name into its configured speed in m/s.
    static float windSpeedSelectionValue( const std::string &selection /**< [in] selected wind-speed element name */ );

    /// Return the nearest supported wind-speed selection element name for a speed in m/s.
    static std::string windSpeedSelectionName( float windSpeed /**< [in] wind speed in m/s */ );

    /// Convert seeing in arcseconds to Fried parameter `r0` in meters.
    static float seeingToR0( float seeing /**< [in] seeing in arcseconds */ );

    /// Return whether a value is finite.
    static bool finiteValue( float value /**< [in] value to test */ );

    /// Return whether a value is finite and strictly positive.
    static bool finitePositiveValue( float value /**< [in] value to test */ );

    /// Return whether two scalar AO-model inputs are effectively equal.
    static bool nearlyEqual( float a,      /**< [in] first value */
                             float b,      /**< [in] second value */
                             float relTol, /**< [in] relative tolerance */
                             float absTol  /**< [in] absolute tolerance */
    );

    /// Convert AO model phase variance into WFE in nm RMS at the specified wavelength.
    static float wfeNm( float variance, /**< [in] phase variance at the science wavelength */
                        float lam0      /**< [in] active wavelength in microns */
    );

    /// Create a writable number property with `current` and `estimated` elements.
    int createCurrentEstimatedProperty( pcf::IndiProperty &prop,  /**< [out] property to initialize */
                                        const std::string &name,  /**< [in] INDI property name */
                                        const std::string &label, /**< [in] suggested GUI label */
                                        const std::string &group  /**< [in] suggested GUI group */
    );

    /// Update the published planning-input properties from the current runtime state.
    void updatePlanningProperties();

    /// Configure an AO model for the selected inputs and requested loop speed.
    void configureAoSystem( aoSystemT              &aosys,  /**< [in,out] AO model instance to configure */
                            const predictionInputs &inputs, /**< [in] scalar state snapshot to apply */
                            float                   fps,    /**< [in] loop speed in Hz */
                            bool optimizeTau /**< [in] true to preserve the current optimal-tau behavior */
    );

    /// Refresh the predicted Strehl, WFE, and optimum-loop-speed properties.
    void updatePredictionOutputs();

    /// Refresh the fixed-grid optimum-loop-speed summary property.
    void updateOptimumLoopSpeed( const predictionInputs &inputs /**< [in] scalar state snapshot to evaluate */ );

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

    /// Local writable wind-speed selection property exposing `slow`, `normal`, and `fast`.
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

    /// Callback for local wind-speed selection writes.
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

    m_windSpeed = windSpeedSelectionValue( windSpeedSelectionName( m_aosys.atm.v_wind() ) );

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
    std::lock_guard<std::mutex> lock( m_stateMutex );

    if( m_useEstimates )
    {
        return m_magEstimated;
    }

    return m_mag;
}

float strehlEstimator::selectedSeeing() const
{
    std::lock_guard<std::mutex> lock( m_stateMutex );

    if( m_useEstimates )
    {
        return m_seeingEstimated;
    }

    return m_seeing;
}

float strehlEstimator::selectedWindSpeed() const
{
    std::lock_guard<std::mutex> lock( m_stateMutex );
    return m_windSpeed;
}

const std::vector<std::string> &strehlEstimator::windSpeedSelectionElements()
{
    static const std::vector<std::string> names{ "slow", "normal", "fast" };
    return names;
}

const std::vector<std::string> &strehlEstimator::windSpeedSelectionLabels()
{
    static const std::vector<std::string> labels{ "Slow (9.4 m/s)", "Normal (18.7 m/s)", "Fast (23.4 m/s)" };
    return labels;
}

float strehlEstimator::windSpeedSelectionValue( const std::string &selection )
{
    if( selection == "fast" )
    {
        return 23.4f;
    }

    if( selection == "normal" )
    {
        return 18.7f;
    }

    return 9.4f;
}

std::string strehlEstimator::windSpeedSelectionName( float windSpeed )
{
    float slowDiff   = std::fabs( windSpeed - 9.4f );
    float normalDiff = std::fabs( windSpeed - 18.7f );
    float fastDiff   = std::fabs( windSpeed - 23.4f );

    if( normalDiff < slowDiff && normalDiff <= fastDiff )
    {
        return "normal";
    }

    if( fastDiff < slowDiff && fastDiff < normalDiff )
    {
        return "fast";
    }

    return "slow";
}

strehlEstimator::predictionInputs strehlEstimator::snapshotPredictionInputs() const
{
    predictionInputs inputs;

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );

        inputs.m_fps             = m_fps;
        inputs.m_emg             = m_emg;
        inputs.m_qe              = m_qe;
        inputs.m_F0              = m_F0;
        inputs.m_lam0            = m_lam0;
        inputs.m_elevation       = m_elevation;
        inputs.m_npix            = m_npix;
        inputs.m_mag             = m_mag;
        inputs.m_magEstimated    = m_magEstimated;
        inputs.m_seeing          = m_seeing;
        inputs.m_seeingEstimated = m_seeingEstimated;
        inputs.m_windSpeed       = m_windSpeed;
        inputs.m_useEstimates    = m_useEstimates;
    }

    if( inputs.m_useEstimates )
    {
        inputs.m_selectedMag       = inputs.m_magEstimated;
        inputs.m_selectedSeeing    = inputs.m_seeingEstimated;
        inputs.m_selectedWindSpeed = inputs.m_windSpeed;
    }
    else
    {
        inputs.m_selectedMag       = inputs.m_mag;
        inputs.m_selectedSeeing    = inputs.m_seeing;
        inputs.m_selectedWindSpeed = inputs.m_windSpeed;
    }

    return inputs;
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

bool strehlEstimator::nearlyEqual( float a, float b, float relTol, float absTol )
{
    float diff = std::fabs( a - b );
    if( diff <= absTol )
    {
        return true;
    }

    float scale = std::fabs( a );
    if( std::fabs( b ) > scale )
    {
        scale = std::fabs( b );
    }

    return diff <= relTol * scale;
}

float strehlEstimator::wfeNm( float variance, float lam0 )
{
    if( variance <= 0.0f )
    {
        return 0.0f;
    }

    return std::sqrt( variance ) * ( 1000.0f * lam0 / two_pi<float>() );
}

void strehlEstimator::updatePlanningProperties()
{
    predictionInputs inputs        = snapshotPredictionInputs();
    std::string      windSelection = windSpeedSelectionName( inputs.m_windSpeed );

    if( !m_indiDriver )
    {
        m_indiP_mag["current"].set( inputs.m_mag );
        m_indiP_mag["estimated"].set( inputs.m_magEstimated );
        m_indiP_mag.setState( INDI_OK );

        m_indiP_seeing_magaox["current"].set( inputs.m_seeing );
        m_indiP_seeing_magaox["estimated"].set( inputs.m_seeingEstimated );
        m_indiP_seeing_magaox.setState( INDI_OK );

        for( auto &&el : m_indiP_windSpeed.getElements() )
        {
            m_indiP_windSpeed[el.first].setSwitchState( el.first == windSelection ? pcf::IndiElement::On
                                                                                  : pcf::IndiElement::Off );
        }
        m_indiP_windSpeed.setState( INDI_OK );

        m_indiP_useEstimates["toggle"].setSwitchState( inputs.m_useEstimates ? pcf::IndiElement::On
                                                                             : pcf::IndiElement::Off );
        m_indiP_useEstimates.setState( inputs.m_useEstimates ? INDI_OK : INDI_IDLE );

        return;
    }

    updatesIfChanged<float>( m_indiP_mag, { "current", "estimated" }, { inputs.m_mag, inputs.m_magEstimated } );
    updatesIfChanged<float>(
        m_indiP_seeing_magaox, { "current", "estimated" }, { inputs.m_seeing, inputs.m_seeingEstimated } );
    indi::updateSelectionSwitchIfChanged( m_indiP_windSpeed, windSelection, m_indiDriver, INDI_OK );
    updateSwitchIfChanged( m_indiP_useEstimates,
                           "toggle",
                           inputs.m_useEstimates ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           inputs.m_useEstimates ? INDI_OK : INDI_IDLE );
}

void strehlEstimator::configureAoSystem( aoSystemT &aosys, const predictionInputs &inputs, float fps, bool optimizeTau )
{
    aosys.optTau( optimizeTau );
    aosys.starMag( inputs.m_selectedMag );
    aosys.F0( inputs.m_qe * inputs.m_F0 );
    aosys.lam_wfs( inputs.m_lam0 * 1.0e-6f );
    aosys.lam_sci( inputs.m_lam0 * 1.0e-6f );
    aosys.ron_wfs( std::vector<float>( { 245.0f / inputs.m_emg } ) );
    aosys.npix_wfs( std::vector<float>( { static_cast<float>( inputs.m_npix ) } ) );
    aosys.minTauWFS( std::vector<float>( { 1.0f / fps } ) );
    aosys.tauWFS( 1.0f / fps );
    aosys.atm.r_0( seeingToR0( inputs.m_selectedSeeing ), 0.5e-6f );
    aosys.atm.v_wind( inputs.m_selectedWindSpeed );
    aosys.zeta( ( 90.0f - inputs.m_elevation ) * pi<float>() / 180.0f );
}

void strehlEstimator::updateOptimumLoopSpeed( const predictionInputs &inputs )
{
    struct scanPoint
    {
        int   fps{ 0 };
        float strehl{ 0.0f };
        float wfeMeasurement{ 0.0f };
        float wfeTimeDelay{ 0.0f };
        float wfeFitting{ 0.0f };
        float wfeTotal{ 0.0f };
        float dOpt{ 0.0f };
        int   binOpt{ 0 };
    };

    constexpr float strehlTieTolerance = 1.0e-4f;

    float                  bestFPS            = 0.0f;
    float                  bestStrehl         = -1.0f;
    float                  bestTotalWfe       = 0.0f;
    float                  bestMeasurementWfe = 0.0f;
    float                  bestTimeDelayWfe   = 0.0f;
    float                  bestFittingWfe     = 0.0f;
    std::vector<scanPoint> scanCurve;
    scanCurve.reserve( 30 );

    for( int fps = 100; fps <= 3000; fps += 100 )
    {
        configureAoSystem( m_aosysScan, inputs, static_cast<float>( fps ), false );

        float strehl = m_aosysScan.strehl();
        if( !finiteValue( strehl ) )
        {
            continue;
        }

        scanPoint point;
        point.fps            = fps;
        point.strehl         = strehl;
        point.wfeMeasurement = wfeNm( m_aosysScan.measurementErrorTotal(), inputs.m_lam0 );
        point.wfeTimeDelay   = wfeNm( m_aosysScan.timeDelayErrorTotal(), inputs.m_lam0 );
        point.wfeFitting     = wfeNm( m_aosysScan.fittingErrorTotal(), inputs.m_lam0 );
        point.wfeTotal       = wfeNm( m_aosysScan.wfeVar(), inputs.m_lam0 );
        point.dOpt           = m_aosysScan.d_opt();
        point.binOpt         = m_aosysScan.bin_opt();
        scanCurve.push_back( point );

        if( bestFPS == 0.0f || strehl > bestStrehl + strehlTieTolerance )
        {
            bestFPS            = static_cast<float>( fps );
            bestStrehl         = strehl;
            bestTotalWfe       = point.wfeTotal;
            bestMeasurementWfe = point.wfeMeasurement;
            bestTimeDelayWfe   = point.wfeTimeDelay;
            bestFittingWfe     = point.wfeFitting;
        }
    }

    bool suspiciousWinner = bestFPS > 0.0f && ( bestFPS <= 400.0f || bestStrehl >= 0.98f ||
                                                bestMeasurementWfe == 0.0f || bestTimeDelayWfe == 0.0f );

    if( suspiciousWinner && bestFPS != m_lastLoggedSuspiciousOptimumFps )
    {
        std::ostringstream oss;
        oss << "strehlEstimator suspicious optimum scan" << " use_estimates=" << std::boolalpha << inputs.m_useEstimates
            << " selected_mag=" << inputs.m_selectedMag << " selected_seeing=" << inputs.m_selectedSeeing
            << " selected_wind=" << inputs.m_selectedWindSpeed << " fps_live=" << inputs.m_fps
            << " emg=" << inputs.m_emg << " elevation=" << inputs.m_elevation << " npix=" << inputs.m_npix
            << " winner_fps=" << bestFPS << " winner_strehl=" << bestStrehl << " winner_wfe_total=" << bestTotalWfe
            << " winner_wfe_meas=" << bestMeasurementWfe << " winner_wfe_delay=" << bestTimeDelayWfe
            << " winner_wfe_fit=" << bestFittingWfe << '\n';

        for( const auto &point : scanCurve )
        {
            oss << "  fps=" << point.fps << " strehl=" << point.strehl << " wfe_total=" << point.wfeTotal
                << " wfe_meas=" << point.wfeMeasurement << " wfe_delay=" << point.wfeTimeDelay
                << " wfe_fit=" << point.wfeFitting << " d_opt=" << point.dOpt << " bin_opt=" << point.binOpt << '\n';
        }

        std::cerr << oss.str();
        m_lastLoggedSuspiciousOptimumFps = bestFPS;
    }
    else if( !suspiciousWinner )
    {
        m_lastLoggedSuspiciousOptimumFps = -1.0f;
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
    std::lock_guard<std::mutex> predictionLock( m_predictionMutex );
    constexpr float             predictionRelTol = 1.0e-6f;
    constexpr float             predictionAbsTol = 1.0e-7f;

    predictionInputs inputs = snapshotPredictionInputs();

    if( !finitePositiveValue( inputs.m_fps ) || !finitePositiveValue( inputs.m_emg ) ||
        !finitePositiveValue( inputs.m_qe ) || !finitePositiveValue( inputs.m_F0 ) ||
        !finitePositiveValue( inputs.m_selectedSeeing ) || !finitePositiveValue( inputs.m_selectedWindSpeed ) ||
        inputs.m_npix <= 0 )
    {
        return;
    }

    const float configuredF0   = inputs.m_qe * inputs.m_F0;
    const float configuredLam  = inputs.m_lam0 * 1.0e-6f;
    const float configuredRon  = 245.0f / inputs.m_emg;
    const float configuredNpix = static_cast<float>( inputs.m_npix );
    const float configuredTau  = 1.0f / inputs.m_fps;
    const float configuredZeta = ( 90.0f - inputs.m_elevation ) * pi<float>() / 180.0f;

    bool sameConfiguredInputs =
        m_aosys.optTau() == true && m_aosys.ron_wfs().size() > 0 && m_aosys.npix_wfs().size() > 0 &&
        m_aosys.minTauWFS().size() > 0 &&
        nearlyEqual( m_aosys.starMag(), inputs.m_selectedMag, predictionRelTol, predictionAbsTol ) &&
        nearlyEqual( m_aosys.F0(), configuredF0, predictionRelTol, predictionAbsTol ) &&
        nearlyEqual( m_aosys.lam_wfs(), configuredLam, predictionRelTol, predictionAbsTol ) &&
        nearlyEqual( m_aosys.lam_sci(), configuredLam, predictionRelTol, predictionAbsTol ) &&
        nearlyEqual( m_aosys.ron_wfs( 0 ), configuredRon, predictionRelTol, predictionAbsTol ) &&
        nearlyEqual( m_aosys.npix_wfs( 0 ), configuredNpix, predictionRelTol, predictionAbsTol ) &&
        nearlyEqual( m_aosys.minTauWFS( 0 ), configuredTau, predictionRelTol, predictionAbsTol ) &&
        nearlyEqual( m_aosys.tauWFS(), configuredTau, predictionRelTol, predictionAbsTol ) &&
        nearlyEqual( m_aosys.atm.v_wind(), inputs.m_selectedWindSpeed, predictionRelTol, predictionAbsTol ) &&
        nearlyEqual( m_aosys.zeta(), configuredZeta, predictionRelTol, predictionAbsTol );

    if( sameConfiguredInputs )
    {
        return;
    }

    configureAoSystem( m_aosys, inputs, inputs.m_fps, true );

    if( !m_indiDriver )
    {
        m_indiP_strehl["pyramid"].set( m_aosys.strehl() );
        m_indiP_strehl.setState( INDI_OK );

        m_indiP_wfe["total"].set( wfeNm( m_aosys.wfeVar(), inputs.m_lam0 ) );
        m_indiP_wfe["measurement"].set( wfeNm( m_aosys.measurementErrorTotal(), inputs.m_lam0 ) );
        m_indiP_wfe["time_delay"].set( wfeNm( m_aosys.timeDelayErrorTotal(), inputs.m_lam0 ) );
        m_indiP_wfe["fitting"].set( wfeNm( m_aosys.fittingErrorTotal(), inputs.m_lam0 ) );
        m_indiP_wfe.setState( INDI_OK );
    }
    else
    {
        updateIfChanged( m_indiP_strehl, "pyramid", m_aosys.strehl() );
        updatesIfChanged<float>( m_indiP_wfe,
                                 { "total", "measurement", "time_delay", "fitting" },
                                 { wfeNm( m_aosys.wfeVar(), inputs.m_lam0 ),
                                   wfeNm( m_aosys.measurementErrorTotal(), inputs.m_lam0 ),
                                   wfeNm( m_aosys.timeDelayErrorTotal(), inputs.m_lam0 ),
                                   wfeNm( m_aosys.fittingErrorTotal(), inputs.m_lam0 ) } );
    }

    updateOptimumLoopSpeed( inputs );
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

    if( createStandardIndiSelectionSw( m_indiP_windSpeed,
                                       "wind_speed",
                                       windSpeedSelectionElements(),
                                       windSpeedSelectionLabels(),
                                       "Wind Speed",
                                       "Error Budget" ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from createStandardIndiSelectionSw" } );
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

    auto wfsavg = mx::improc::eigenMap<float>(
        reinterpret_cast<float *>( curr_src ), wfsavgShmimMonitorT::m_width, wfsavgShmimMonitorT::m_height );

    float counts     = 0.0f;
    bool  haveCounts = false;

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );

        m_wfsavg = wfsavg;

        if( m_wfsavg.rows() == m_wfsmask.rows() && m_wfsavg.cols() == m_wfsmask.cols() )
        {
            m_counts   = ( m_wfsavg * m_wfsmask ).sum();
            counts     = m_counts;
            haveCounts = true;
        }
    }

    if( haveCounts )
    {
        std::cerr << "counts: " << counts << '\n';

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

    auto wfsmask = mx::improc::eigenMap<float>(
        reinterpret_cast<float *>( curr_src ), wfsmaskShmimMonitorT::m_width, wfsmaskShmimMonitorT::m_height );

    bool haveCounts = false;

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );

        m_wfsmask = wfsmask;
        m_npix    = static_cast<int>( std::lround( m_wfsmask.sum() ) );

        if( m_wfsavg.rows() == m_wfsmask.rows() && m_wfsavg.cols() == m_wfsmask.cols() )
        {
            // update counts because we might have been waiting on this.
            m_counts   = ( m_wfsavg * m_wfsmask ).sum();
            haveCounts = true;
        }
    }

    if( haveCounts )
    {
        calcMag();
    }

    return 0;
}

void strehlEstimator::calcMag()
{
    float counts  = 0.0f;
    float emg     = 0.0f;
    float fps     = 0.0f;
    float qe      = 0.0f;
    float F0      = 0.0f;
    bool  updated = false;

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );

        counts = m_counts;
        emg    = m_emg;
        fps    = m_fps;
        qe     = m_qe;
        F0     = m_F0;

        if( finitePositiveValue( m_counts ) && finitePositiveValue( m_again ) && finitePositiveValue( m_emg ) &&
            finitePositiveValue( m_fps ) && finitePositiveValue( m_qe ) && finitePositiveValue( m_F0 ) )
        {
            m_mag = -2.5f * std::log10( m_counts * m_again / m_emg * m_fps / ( m_qe * m_F0 ) );

            if( !m_useEstimates && !m_magEstimatedManual )
            {
                m_magEstimated = m_mag;
            }

            updated = true;
        }
    }

    std::cerr << "calcMag: " << counts << ' ' << m_again << ' ' << ' ' << emg << ' ' << fps << ' ' << qe << ' ' << F0
              << '\n';

    if( !updated )
    {
        return;
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

        bool changed = false;

        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            if( finitePositiveValue( fps ) && fps != m_fps )
            {
                m_fps   = fps;
                changed = true;
            }
        }

        if( changed )
        {
            std::cerr << "Got FPS: " << fps << '\n';

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

        bool changed = false;

        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            if( finitePositiveValue( emg ) && emg != m_emg )
            {
                m_emg   = emg;
                changed = true;
            }
        }

        if( changed )
        {
            std::cerr << "Got EMG: " << emg << '\n';

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

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );

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

        bool changed = false;

        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );

            if( finitePositiveValue( seeing ) && seeing != m_seeing )
            {
                m_seeing         = seeing;
                m_r0             = seeingToR0( m_seeing );
                m_dimm_fwhm_corr = seeing;

                if( !m_useEstimates && !m_seeingEstimatedManual )
                {
                    m_seeingEstimated = m_seeing;
                }

                changed = true;
            }
        }

        if( changed )
        {
            std::cerr << "Got seeing: " << seeing << '\n';

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

        bool changed = false;

        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            if( finiteValue( elevation ) && elevation != m_elevation )
            {
                m_elevation = elevation;
                changed     = true;
            }
        }

        if( changed )
        {
            std::cerr << "Got elevation: " << elevation << '\n';

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

        bool changed = false;

        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            if( finiteValue( mag ) && mag != m_magEstimated )
            {
                m_magEstimated       = mag;
                m_magEstimatedManual = true;
                changed              = true;
            }
        }

        if( changed )
        {
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

        bool changed = false;

        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            if( finitePositiveValue( seeing ) && seeing != m_seeingEstimated )
            {
                m_seeingEstimated       = seeing;
                m_seeingEstimatedManual = true;
                changed                 = true;
            }
        }

        if( changed )
        {
            updatePlanningProperties();
            updatePredictionOutputs();
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( strehlEstimator, m_indiP_windSpeed )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_windSpeed, ipRecv );

    std::string selection;
    for( auto &&el : ipRecv.getElements() )
    {
        if( el.second.getSwitchState() == pcf::IndiElement::On )
        {
            selection = el.first;
            break;
        }
    }

    if( selection == "" )
    {
        return 0;
    }

    float windSpeed = windSpeedSelectionValue( selection );
    bool  changed   = false;

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );
        if( finitePositiveValue( windSpeed ) && windSpeed != m_windSpeed )
        {
            m_windSpeed = windSpeed;
            changed     = true;
        }
    }

    updatePlanningProperties();

    if( changed )
    {
        updatePredictionOutputs();
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
    bool changed      = false;

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );
        if( useEstimates != m_useEstimates )
        {
            m_useEstimates = useEstimates;
            changed        = true;
        }
    }

    updatePlanningProperties();

    if( changed )
    {
        updatePredictionOutputs();
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // strehlEstimator_hpp
