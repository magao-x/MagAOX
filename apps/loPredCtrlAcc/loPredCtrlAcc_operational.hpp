/** \file loPredCtrlAcc_operational.hpp
 * \brief WFS-rate closed-loop DDSPC with independent standard/ACC fits and tagged auxiliary DM delivery.
 * \ingroup app_files
 */
#ifndef loPredCtrlAcc_operational_hpp
#define loPredCtrlAcc_operational_hpp

#include <atomic>
#include <mutex>
#include <random>
#include <sstream>

#include <mx/ioutils/fits/fitsFile.hpp>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
#ifndef MAGAOX_CURRENT_SHA1
    #include "../../magaox_git_version.h"
#endif
#include "../../libMagAOX/app/syncoOutput.hpp"
#include "adc_reader.hpp"
#include "native_pipeline.hpp"

namespace MagAOX::app
{
/// Operational loPredCtrl workflow; upstream owns ACC/WFS synchronization and the DM owns its scheduler.
/** Both independent models use identical WFS-rate command histories and objectives. The assisted model additionally
 * receives every individual upstream 8-kHz ACC sample in a rolling history frozen at each 2-kHz WFS origin.
 * Commands are held between WFS decisions and delivered at the existing producer's advertised slots.
 */
class loPredCtrlAcc_operational : public MagAOXApp<true>,
                                  public dev::shmimMonitor<loPredCtrlAcc_operational>,
                                  protected DDSPC::ShadowPredictionSink,
                                  protected dev::dmSyncopationProducer<loPredCtrlAcc_operational>
{
    friend class dev::shmimMonitor<loPredCtrlAcc_operational>;
    friend class dev::dmSyncopationProducer<loPredCtrlAcc_operational>;
    using shmimMonitorT          = dev::shmimMonitor<loPredCtrlAcc_operational>;
    using dmSyncopationProducerT = dev::dmSyncopationProducer<loPredCtrlAcc_operational>;

  protected:
    /// Common model parameters; the RLS/QR implementation is shared with the existing controller.
    DDSPC::ShadowConfig m_settings;

    /// Existing full-image calibration, or the same modal cube used by the operational modal-to-DM path.
    DDSPC::NativeCalibration m_calibration;
    std::string              m_calibrationFile, m_modeCube, m_mask;

    std::uint32_t m_calWidth{ 0 }, m_calHeight{ 0 }; ///< Exact native FITS plane dimensions when using a cube.

    /// Exact owned auxiliary; no configuration can redirect this app to a combined DM destination.
    std::string m_channel{ "dm10disp10_aux" };

    /// Independent raw ADC decoder; absence or invalid calibration affects only the assisted model.
    bool                              m_accelEnabled{ true };
    DDSPC::AdcConfig                  m_adcConfig;
    std::unique_ptr<DDSPC::AdcReader> m_adc;
    DDSPC::AdcCursor                  m_adcCursor;
    std::array<DDSPC::AdcSample, 128> m_adcHistory{};
    unsigned                          m_adcCount{ 0 }, m_adcNext{ 0 };

    /// Existing producer lock diagnostic; there are no local phase estimates or counter anchors.
    std::string       m_syncDevice{ "acc000" }, m_syncProperty{ "timingDiag" };
    std::atomic<bool> m_syncLocked{ false };

    /// Shared WFS freshness policy, independent of ACC synchronization and the model's excitation rank.
    double                    m_inputTimeout{ .01 };
    std::atomic<std::int64_t> m_wfsReceipt{ -1 };
    std::atomic<bool>         m_inputValid{ false }, m_assistedValid{ false }, m_candidateValid{ false };

    /// Serialize numerical state and operator changes; the producer never takes this mutex.
    std::mutex                               m_modelMutex;
    std::unique_ptr<DDSPC::ShadowController> m_models;
    DDSPC::ShadowRecord                      m_pending;
    bool                                     m_havePending{ false };
    std::uint64_t                            m_wfsCounter{ 0 };
    std::array<DDSPC::Matrix, 2>             m_plans;

    /// Matched scalar errors have identical WFS origin, target, horizon and channel; they never train either fit.
    struct PairedScore
    {
        std::uint64_t         m_origin{ 0 }, m_target{ 0 }; ///< Original WFS identities for this bounded slot.
        unsigned              m_seen{ 0 };                  ///< One bit for each independent model.
        std::array<double, 2> m_squared{};                  ///< Origin-only errors for the identical measured target.
    };
    std::vector<PairedScore> m_pairs;
    std::array<double, 2>    m_matchedSquared{};
    std::uint64_t m_matchedTargets{ 0 }; ///< Scalars scored by both models, excluding unmatched gaps and warmup.

    /// Brief payload/acknowledged-request exchange, kept separate from all RLS and QR work.
    std::mutex         m_commandMutex;
    std::vector<float> m_pixels;
    DDSPC::Matrix      m_copiedCommand;
    bool               m_commandKnown{ false };
    SyncoOutput        m_output;

    /// Operator intent; learning OFF freezes both models while prediction can continue.
    bool                           m_learning{ false }, m_predicting{ false };
    std::atomic<DDSPC::ShadowMode> m_mode{ DDSPC::ShadowMode::Standard };
    DDSPC::realT                   m_gain{ .3f };

    /// Original count, Gaussian strength, regularization exploration stages, consumed once per WFS decision.
    struct Stage
    {
        unsigned m_count{ 0 };                  ///< Remaining real WFS decisions.
        float    m_sigma{ 0 }, m_lambda{ 100 }; ///< Modal command strength and common command penalty.
    };
    std::vector<Stage>              m_exploration;
    std::default_random_engine      m_generator;
    std::normal_distribution<float> m_distribution{ 0, 1 };

    /// Public controls and compact diagnostics; names retain the operational loPredCtrl workflow.
    pcf::IndiProperty m_indiP_learning, m_indiP_predicting, m_indiP_learnStd, m_indiP_reset;
    pcf::IndiProperty m_indiP_mode, m_indiP_gain, m_indiP_exploration, m_indiP_sync;
    pcf::IndiProperty m_indiP_control, m_indiP_performance, m_indiP_safety, m_indiP_adc;

    /// Load the existing calibration or selected finite masked cube planes; derive the request projection by QR.
    void loadCalibration();

    /// Allocate the independent pair and pre-sized workspaces without changing operator intent.
    void resetModels();

    /// Freeze the causally available raw ACC history; never generate, interpolate, average or synchronize samples.
    bool acceleration( DDSPC::Matrix &features /**< [out] Newest pair first, four pairs per WFS history. */,
                       std::int64_t   now /**< [in] Frozen WFS decision cutoff. */ );

    /// Assemble the causal sensor vector from the existing reader's owned batch and continuity metadata.
    bool sensorHistory( const DDSPC::AdcBatch            &copied /**< [in] Coherent reader status and count. */,
                        std::span<const DDSPC::AdcSample> batch /**< [in] Samples available at decision time. */,
                        DDSPC::Matrix                    &features /**< [out] Newest-first X/Y pairs. */,
                        std::int64_t                      now /**< [in] Frozen WFS cutoff. */ );

    /// Process one WFS decision using the last transport-acknowledged own request to close the preceding interval.
    int step( const DDSPC::Matrix &measurement /**< [in] Current calibrated WFS modes. */,
              std::uint64_t        counter /**< [in] Original WFS publication counter. */,
              const DDSPC::Matrix &features /**< [in] Individual raw ACC lift. */,
              bool                 assistedValid /**< [in] Upstream lock and complete causal sensor history. */,
              const DDSPC::Matrix &acknowledged /**< [in] Own command copied by decision time, empty if unknown. */,
              std::int64_t         now /**< [in] Current monotonic decision time. */ );

    /// Check only current WFS/selected sensor input; bootstrap prediction may explore before a model is trained.
    bool outputValid() const;

    /// Copy a prepared WFS-rate image for an actual existing producer Request.
    bool outputCandidate( std::span<float> pixels /**< [out] Complete own auxiliary payload. */ );

    /// Retain only the exact transport-copied own contribution, separately from physical application.
    void copied( std::span<const float> pixels /**< [in] Matching acknowledged own payload. */ );

    /// Validate the original exploration CSV grammar before replacing the running sequence.
    static std::vector<Stage> parseExploration( const std::string &text /**< [in] count,sigma,lambda triples. */ );

    /// Parse an entire finite INDI number; the library's stream extraction does not report conversion failures.
    static double number( const pcf::IndiElement &element /**< [in] Original wire value. */ );

    /// Reject mode/reset changes until output is off and its own zero has been acknowledged.
    bool removed() const;

    /// Register operator controls without opening inputs or the auxiliary transport.
    int setupINDI();

    /// Accumulate matched comparison scores on the sole numerical owner.
    void prediction(
        const DDSPC::ShadowPrediction &sample /**< [in] Matured frozen-origin prediction. */ ) noexcept override;

    /// Validate the input vector layout on connection or source replacement.
    int allocate( const dev::shmimT &tag /**< [in] WFS monitor tag. */ );

    /// Run normal WFS-rate learning and prediction on the existing monitor thread.
    int processImage( void              *source /**< [in] Coherent FLOAT WFS modal payload. */,
                      const dev::shmimT &tag /**< [in] WFS monitor tag. */ );

  public:
    /// Set operational model defaults before reading configuration.
    loPredCtrlAcc_operational();

    /// Stop callbacks while the application's command/model state still exists.
    ~loPredCtrlAcc_operational() noexcept;

    /// Register the existing base interfaces and operational configuration.
    virtual void setupConfig();

    /// Validate configuration and calibration; missing ACC configuration leaves standard control available.
    virtual void loadConfig();

    /// Load the base interfaces through their integer-returning configuration macros.
    int loadConfigImpl();

    /// Register controls, start input monitoring, and acquire only the existing auxiliary producer.
    virtual int appStartup();

    /// Publish model/transport state without placing INDI activity in the WFS callback.
    virtual int appLogic();

    /// Remove only this app's contribution before closing producer and input resources.
    virtual int appShutdown();

    /// Set learning for both independent models.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_learning );

    /// Enable prediction/bootstrap exploration, or request acknowledged own-channel zero.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_predicting );

    /// Run the original three-stage standard learning sequence.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_learnStd );

    /// Reset both models after confirmed command removal.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_reset );

    /// Select the active independent fit without changing either model's training.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_mode );

    /// Set the additive current-residual feedback gain used by loPredCtrl.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_gain );

    /// Install an original-format explicit exploration sequence.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_exploration );

    /// Receive the existing upstream synchronization state; it never gates standard DDSPC.
    INDI_SETCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_sync );
};

inline loPredCtrlAcc_operational::loPredCtrlAcc_operational() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_settings.m_modes         = 2;
    m_settings.m_prior         = 1e-5f;
    m_settings.m_accelHistory  = 1;
    m_settings.m_useQrd        = true;
    m_settings.m_scaledPenalty = m_settings.m_externalOrigins = m_settings.m_freezeQualification = true;
    m_settings.m_requireFullRank                                                                 = false;
    m_calibration.m_indices                                                                      = { 0, 1 };
}

inline loPredCtrlAcc_operational::~loPredCtrlAcc_operational() noexcept
{
    m_shutdown = true;
    m_output.stop();
    shmimMonitorT::appShutdown();
}

inline void loPredCtrlAcc_operational::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );
    DMSYNCOPATIONPRODUCER_SETUP_CONFIG( config );
    auto add =
        [this]( const std::string &section, const std::string &key, const std::string &type, const std::string &help )
    { config.add( section + "." + key, "", section + "." + key, argType::Required, section, key, false, type, help ); };
    for( const auto &key : { "gain", "regularization", "gamma", "covariance", "input_timeout" } )
        add( "parameters", key, "float", "Operational loPredCtrl parameter; input_timeout is in seconds." );
    for( const auto &key : { "num_modes", "history", "future", "accel_history", "modal_index0", "modal_index1" } )
        add( "parameters", key, "int", "Model dimensions or WFS modal indices; accel_history counts WFS periods." );
    add( "parameters", "qrd", "bool", "Use QR-RLS for both independent models." );
    for( const auto &key : { "channelName", "modeCube", "mask", "calibration" } )
        add( "dm", key, "string", "Existing own auxiliary/calibration; calibration overrides modeCube/mask." );
    add( "adc", "enabled", "bool", "Use upstream synchronized ADC inputs for the assisted model." );
    for( const auto &key : { "shmimName", "profile", "sync_device", "sync_property" } )
        add( "adc", key, "string", "Existing ADC source or existing upstream lock diagnostic." );
    for( const auto &key : { "min_code", "max_code" } )
        add( "adc", key, "int", "Raw ADC encoding bounds." );
    add( "adc", "capacity", "int", "Existing ADC reader ring capacity." );
    for( const auto &key : { "max_age_us", "max_interval_us", "clock_tolerance_us" } )
        add( "adc", key, "float", "Existing ADC reader validity interval in microseconds." );
    add( "adc", "reject_rails", "bool", "Reject ADC rail samples." );
    for( const auto &key : { "zero", "volts_per_count", "sensitivity", "polarity" } )
        for( int axis = 0; axis < 2; ++axis )
            add( "adc", std::string( key ) + std::to_string( axis ), "float", "Per-axis upstream ADC calibration." );
}

inline void loPredCtrlAcc_operational::loadConfig()
{
    if( loadConfigImpl() < 0 )
        m_shutdown = true;
}

inline int loPredCtrlAcc_operational::loadConfigImpl()
{
    try
    {
        SHMIMMONITOR_LOAD_CONFIG( config );
        DMSYNCOPATIONPRODUCER_LOAD_CONFIG( config );
        config( m_gain, "parameters.gain" );
        config( m_settings.m_lambda, "parameters.regularization" );
        config( m_settings.m_gamma, "parameters.gamma" );
        config( m_settings.m_prior, "parameters.covariance" );
        config( m_settings.m_modes, "parameters.num_modes" );
        config( m_settings.m_history, "parameters.history" );
        config( m_settings.m_future, "parameters.future" );
        config( m_settings.m_useQrd, "parameters.qrd" );
        config( m_inputTimeout, "parameters.input_timeout" );
        int accelHistory = 5;
        config( accelHistory, "parameters.accel_history" );
        config( m_calibration.m_indices[0], "parameters.modal_index0" );
        config( m_calibration.m_indices[1], "parameters.modal_index1" );
        config( m_channel, "dm.channelName" );
        config( m_modeCube, "dm.modeCube" );
        config( m_mask, "dm.mask" );
        config( m_calibrationFile, "dm.calibration" );
        if( m_channel != "dm10disp10_aux" || m_syncopationTiming.empty() || shmimMonitorT::m_shmimName.empty() ||
            !std::isfinite( m_gain ) || std::abs( m_gain ) > 10 || !std::isfinite( m_inputTimeout ) ||
            m_inputTimeout <= 0 || m_inputTimeout > 1 || m_settings.m_modes < 1 || m_settings.m_modes > 2 ||
            accelHistory < 1 || accelHistory > 16 )
            throw std::invalid_argument( "Invalid operational WFS, producer, gain or model configuration" );
        m_settings.m_accelChannels = 8 * accelHistory;
        loadCalibration();
        resetModels();
    }
    catch( const std::exception &error )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
    }
    try
    {
        config( m_accelEnabled, "adc.enabled" );
        if( !m_accelEnabled )
            return 0;
        config( m_adcConfig.m_capacity, "adc.capacity" );
        auto interval = [this]( const char *key, std::int64_t &nanoseconds )
        {
            double microseconds = nanoseconds / 1000.;
            config( microseconds, key );
            if( !std::isfinite( microseconds ) || microseconds < .001 || microseconds > 1e9 )
                throw std::invalid_argument( std::string( "Invalid ADC interval: " ) + key );
            nanoseconds = std::llround( microseconds * 1000 );
        };
        interval( "adc.max_age_us", m_adcConfig.m_maxAgeNs );
        interval( "adc.max_interval_us", m_adcConfig.m_maxIntervalNs );
        interval( "adc.clock_tolerance_us", m_adcConfig.m_clockToleranceNs );
        config( m_adcConfig.m_stream, "adc.shmimName" );
        config( m_adcConfig.m_profile, "adc.profile" );
        config( m_syncDevice, "adc.sync_device" );
        config( m_syncProperty, "adc.sync_property" );
        config( m_adcConfig.m_minCode, "adc.min_code" );
        config( m_adcConfig.m_maxCode, "adc.max_code" );
        config( m_adcConfig.m_rejectRails, "adc.reject_rails" );
        for( unsigned axis = 0; axis < 2; ++axis )
        {
            const auto suffix = std::to_string( axis );
            config( m_adcConfig.m_zero[axis], "adc.zero" + suffix );
            config( m_adcConfig.m_voltsPerCount[axis], "adc.volts_per_count" + suffix );
            config( m_adcConfig.m_sensitivity[axis], "adc.sensitivity" + suffix );
            config( m_adcConfig.m_polarity[axis], "adc.polarity" + suffix );
        }
        DDSPC::validate_adc_config( m_adcConfig );
    }
    catch( const std::exception &error )
    {
        m_accelEnabled = false;
        log<software_error>( { __FILE__, __LINE__, std::string( "ACC assistance unavailable: " ) + error.what() } );
    }
    return 0;
}

inline void loPredCtrlAcc_operational::loadCalibration()
{
    if( !m_calibrationFile.empty() )
        m_calibration = DDSPC::load_native_calibration( m_calibrationFile );
    else
    {
        if( m_modeCube.empty() )
            throw std::invalid_argument( "Supply the operational modal-to-DM modeCube" );
        mx::fits::fitsFile<float>     file;
        mx::improc::eigenCube<float>  cube;
        mx::improc::eigenImage<float> mask;
        if( file.read( cube, m_modeCube ) != mx::error_t::noerror || cube.rows() < 1 || cube.cols() < 1 ||
            cube.planes() < 1 )
            throw std::invalid_argument( "Could not read DM mode cube: " + m_modeCube );
        if( m_mask.empty() )
            mask = mx::improc::eigenImage<float>::Ones( cube.rows(), cube.cols() );
        else if( file.read( mask, m_mask ) != mx::error_t::noerror )
            throw std::invalid_argument( "Could not read DM mask: " + m_mask );
        if( mask.rows() != cube.rows() || mask.cols() != cube.cols() || !mask.allFinite() ||
            cube.rows() * cube.cols() > 4096 )
            throw std::invalid_argument( "Incompatible DM modal cube/mask" );
        m_calWidth  = cube.rows();
        m_calHeight = cube.cols();
        m_calibration.m_shapes.resize( mask.size(), m_settings.m_modes );
        for( int mode = 0; mode < m_settings.m_modes; ++mode )
        {
            const auto index = m_calibration.m_indices[mode];
            if( index >= std::size_t( cube.planes() ) || !cube.image( index ).allFinite() )
                throw std::invalid_argument( "Missing or nonfinite calibrated DM mode" );
            for( int pixel = 0; pixel < mask.size(); ++pixel )
                m_calibration.m_shapes( pixel, mode ) = mask.data()[pixel] > 0 ? cube.image( index ).data()[pixel] : 0;
        }
        Eigen::CompleteOrthogonalDecomposition<DDSPC::Matrix> qr( m_calibration.m_shapes );
        if( qr.rank() != m_settings.m_modes )
            throw std::invalid_argument( "DM modes have deficient rank on the supplied mask" );
        m_calibration.m_projection = qr.pseudoInverse();
        m_calibration.m_layout     = 1;
    }
    const auto &shapes = m_calibration.m_shapes;
    if( shapes.rows() <= 0 || shapes.cols() != m_settings.m_modes || !shapes.allFinite() ||
        m_calibration.m_projection.cols() != shapes.rows() || m_calibration.m_projection.rows() != m_settings.m_modes ||
        !m_calibration.m_projection.allFinite() )
        throw std::invalid_argument( "Missing complete calibrated DM synthesis/projection" );
    m_pixels.assign( shapes.rows(), 0 );
    m_copiedCommand = DDSPC::Matrix::Zero( m_settings.m_modes, 1 );
}

inline void loPredCtrlAcc_operational::resetModels()
{
    m_models = std::make_unique<DDSPC::ShadowController>( m_settings );
    m_models->prediction_sink( this );
    m_pairs.assign( ( m_settings.m_future + 1 ) * m_settings.m_future * m_settings.m_modes, {} );
    m_matchedSquared.fill( 0 );
    m_matchedTargets = 0;
    for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
        m_models->training( mode, m_learning );
    for( auto &plan : m_plans )
        plan = DDSPC::Matrix::Zero( m_settings.m_future * m_settings.m_modes, 1 );
    m_pending     = {};
    m_havePending = false;
    m_wfsCounter  = 0;
}

inline void loPredCtrlAcc_operational::prediction( const DDSPC::ShadowPrediction &sample ) noexcept
{
    if( !sample.m_originValid || sample.m_horizon < 1 || sample.m_horizon > unsigned( m_settings.m_future ) ||
        sample.m_channel >= unsigned( m_settings.m_modes ) )
        return;
    const auto index =
        ( ( sample.m_origin % ( m_settings.m_future + 1 ) ) * m_settings.m_future + sample.m_horizon - 1 ) *
            m_settings.m_modes +
        sample.m_channel;
    auto &pair = m_pairs[index];
    if( pair.m_origin != sample.m_origin || pair.m_target != sample.m_target )
    {
        pair          = {};
        pair.m_origin = sample.m_origin;
        pair.m_target = sample.m_target;
    }
    const unsigned mode  = sample.m_mode == DDSPC::ShadowMode::AccelAssisted;
    const double   error = sample.m_originPrediction - sample.m_measurement;
    if( ( pair.m_seen & ( 1u << mode ) ) || !std::isfinite( error ) )
        return;
    pair.m_seen |= 1u << mode;
    pair.m_squared[mode] = error * error;
    if( pair.m_seen == 3 )
    {
        for( unsigned m = 0; m < 2; ++m )
            m_matchedSquared[m] += pair.m_squared[m];
        ++m_matchedTargets;
    }
}

inline bool loPredCtrlAcc_operational::acceleration( DDSPC::Matrix &features, std::int64_t now )
{
    if( !m_adc )
        return false;
    std::array<DDSPC::AdcSample, 128> batch;
    const auto                        copied = m_adc->copy_after( m_adcCursor, batch, now );
    m_adcCursor                              = copied.m_next;
    return sensorHistory( copied, batch, features, now );
}

inline bool loPredCtrlAcc_operational::sensorHistory( const DDSPC::AdcBatch            &copied,
                                                      std::span<const DDSPC::AdcSample> batch,
                                                      DDSPC::Matrix                    &features,
                                                      std::int64_t                      now )
{
    if( copied.m_count > batch.size() || !m_syncLocked.load() || !copied.m_status.m_valid )
    {
        m_adcCount = m_adcNext = 0;
        return false;
    }
    if( copied.m_discontinuity || copied.m_more )
        m_adcCount = m_adcNext = 0;
    for( unsigned n = 0; n < copied.m_count; ++n )
    {
        m_adcHistory[m_adcNext] = batch[n];
        m_adcNext               = ( m_adcNext + 1 ) % m_adcHistory.size();
        m_adcCount              = std::min<unsigned>( m_adcCount + 1, m_adcHistory.size() );
    }
    const unsigned count = m_settings.m_accelChannels / 2;
    if( !copied.m_count || copied.m_more || m_adcCount < count )
        return false;
    const auto &newest = m_adcHistory[( m_adcNext + m_adcHistory.size() - 1 ) % m_adcHistory.size()];
    if( now < newest.m_receivedNs || now - newest.m_acquiredNs > m_adcConfig.m_maxAgeNs )
        return false;
    features.resize( m_settings.m_accelChannels, 1 );
    for( unsigned n = 0; n < count; ++n )
    {
        const auto &sample = m_adcHistory[( m_adcNext + m_adcHistory.size() - n - 1 ) % m_adcHistory.size()];
        if( sample.m_frame.m_counter + n != newest.m_frame.m_counter || sample.m_segment != newest.m_segment ||
            sample.m_generation != newest.m_generation || sample.m_receivedNs > now )
            return false;
        features( 2 * n )     = sample.m_acceleration[0];
        features( 2 * n + 1 ) = sample.m_acceleration[1];
    }
    return features.allFinite();
}

inline int loPredCtrlAcc_operational::step( const DDSPC::Matrix &measurement,
                                            std::uint64_t        counter,
                                            const DDSPC::Matrix &features,
                                            bool                 assistedValid,
                                            const DDSPC::Matrix &acknowledged,
                                            std::int64_t         now )
{
    std::lock_guard<std::mutex> lock( m_modelMutex );
    if( !m_models || measurement.rows() != m_settings.m_modes || measurement.cols() != 1 || !measurement.allFinite() ||
        !counter )
    {
        m_inputValid.store( false );
        return 0;
    }
    if( counter == m_wfsCounter )
        return 0;
    if( m_wfsCounter && counter != m_wfsCounter + 1 )
    {
        m_models->invalidate( {} );
        m_havePending = false;
    }
    const auto selected = m_mode.load();
    const bool probing  = m_predicting && m_learning && !m_exploration.empty() &&
                          ( selected == DDSPC::ShadowMode::Standard || assistedValid );
    if( probing )
        m_models->regularization( m_exploration.front().m_lambda );
    if( m_havePending )
    {
        m_pending.m_appliedCommand = acknowledged;
        m_pending.m_appliedValid   = acknowledged.size() == m_settings.m_modes;
        m_models->process( m_pending );
    }
    bool          selectedReady = false;
    DDSPC::Matrix command       = m_gain * measurement;
    for( unsigned model = 0; model < 2; ++model )
    {
        const auto    mode = model ? DDSPC::ShadowMode::AccelAssisted : DDSPC::ShadowMode::Standard;
        DDSPC::Matrix empty;
        const bool    ready =
            ( !model || assistedValid ) &&
            m_models->predict_next( mode, counter, measurement, model ? features : empty, m_plans[model] );
        if( mode == selected )
            selectedReady = ready;
        if( m_predicting && mode == selected && ready )
            command += m_plans[model].bottomRows( m_settings.m_modes );
    }
    const bool numericalFault = m_models->status( selected ).m_reason == DDSPC::ShadowReason::NumericalFault;
    if( probing && !numericalFault )
    {
        auto &stage = m_exploration.front();
        for( int mode = 0; mode < m_settings.m_modes; ++mode )
            command( mode ) += stage.m_sigma * m_distribution( m_generator );
        if( !--stage.m_count )
            m_exploration.erase( m_exploration.begin() );
    }
    if( !m_predicting || numericalFault )
        command.setZero();
    const DDSPC::Matrix image  = m_calibration.m_shapes * command;
    const bool          finite = image.allFinite();
    { //mutex scope
        std::lock_guard<std::mutex> commandLock( m_commandMutex );
        if( finite )
            std::copy( image.data(), image.data() + image.size(), m_pixels.begin() );
        m_assistedValid.store( assistedValid );
        m_candidateValid.store( finite && !numericalFault &&
                                ( !m_predicting || selectedReady || probing || ( m_learning && m_gain != 0 ) ) );
        m_wfsReceipt.store( now );
        m_inputValid.store( true );
    }
    m_pending                = {};
    m_pending.m_index        = counter;
    m_pending.m_residual     = measurement;
    m_pending.m_acceleration = features;
    m_pending.m_wfsValid = m_pending.m_timingValid = true;
    m_pending.m_accelValid                         = assistedValid;
    m_havePending                                  = true;
    m_wfsCounter                                   = counter;
    return 0;
}

inline bool loPredCtrlAcc_operational::outputValid() const
{
    const auto now = synco::monotonicNs(), last = m_wfsReceipt.load();
    return m_inputValid.load() && m_candidateValid.load() && last >= 0 && now >= last &&
           now - last <= m_inputTimeout * 1e9 &&
           ( ( m_assistedValid.load() && m_syncLocked.load() ) || m_mode == DDSPC::ShadowMode::Standard );
}

inline bool loPredCtrlAcc_operational::outputCandidate( std::span<float> pixels )
{
    std::lock_guard<std::mutex> lock( m_commandMutex );
    if( pixels.size() != m_pixels.size() )
        return false;
    std::copy( m_pixels.begin(), m_pixels.end(), pixels.begin() );
    return true;
}

inline void loPredCtrlAcc_operational::copied( std::span<const float> pixels )
{
    std::lock_guard<std::mutex>     lock( m_commandMutex );
    Eigen::Map<const DDSPC::Matrix> image( pixels.data(), pixels.size(), 1 );
    m_copiedCommand.noalias() = m_calibration.m_projection * image;
    m_commandKnown            = m_copiedCommand.allFinite();
}

inline int loPredCtrlAcc_operational::allocate( const dev::shmimT & )
{
    m_inputValid.store( false );
    if( shmimMonitorT::m_dataType != _DATATYPE_FLOAT ||
        ( shmimMonitorT::m_width != 1 && shmimMonitorT::m_height != 1 ) ||
        shmimMonitorT::m_width * shmimMonitorT::m_height <=
            *std::max_element( m_calibration.m_indices.begin(), m_calibration.m_indices.begin() + m_settings.m_modes ) )
        return -1;
    std::lock_guard<std::mutex> lock( m_modelMutex );
    m_models->invalidate( {} );
    m_havePending = false;
    m_wfsCounter  = 0;
    return 0;
}

inline int loPredCtrlAcc_operational::processImage( void *source, const dev::shmimT & )
{
    DDSPC::Matrix acknowledged;
    std::int64_t  now;
    { //mutex scope
        std::lock_guard<std::mutex> commandLock( m_commandMutex );
        now = synco::monotonicNs();
        if( m_commandKnown )
            acknowledged = m_copiedCommand;
    }
    const auto &metadata = *shmimMonitorT::m_imageStream.md;
    const auto  counter  = metadata.cnt0;
    if( metadata.write || !source )
        return 0;
    if( metadata.naxis == 3 )
    {
        const auto plane = metadata.cnt1;
        if( plane >= metadata.size[2] ||
            source != shmimMonitorT::m_imageStream.array.F +
                          plane * std::size_t( shmimMonitorT::m_width ) * shmimMonitorT::m_height )
            return 0;
    }
    std::atomic_thread_fence( std::memory_order_acquire );
    DDSPC::Matrix measurement( m_settings.m_modes, 1 ), features;
    for( int mode = 0; mode < m_settings.m_modes; ++mode )
        measurement( mode ) = static_cast<const float *>( source )[m_calibration.m_indices[mode]];
    std::atomic_thread_fence( std::memory_order_acquire );
    if( metadata.write || counter != metadata.cnt0 )
        return 0;
    const bool valid = acceleration( features, now );
    return step( measurement, counter, features, valid, acknowledged, now );
}

inline bool loPredCtrlAcc_operational::removed() const
{
    const auto output = m_output.status();
    return !output.m_requested && output.m_zeroConfirmed;
}

inline std::vector<loPredCtrlAcc_operational::Stage>
loPredCtrlAcc_operational::parseExploration( const std::string &text )
{
    std::vector<Stage> stages;
    std::stringstream  input( text );
    std::string        count, sigma, lambda;
    while( std::getline( input, count, ',' ) )
    {
        if( !std::getline( input, sigma, ',' ) || !std::getline( input, lambda, ',' ) || stages.size() >= 32 )
            throw std::invalid_argument( "Expected count,sigma,regularization exploration triples" );
        std::size_t nc, ns, nl;
        const auto  n = std::stoul( count, &nc );
        const auto  s = std::stof( sigma, &ns ), l = std::stof( lambda, &nl );
        if( nc != count.size() || ns != sigma.size() || nl != lambda.size() || n == 0 || n > 1000000 ||
            !std::isfinite( s ) || s < 0 || !std::isfinite( l ) || l <= 0 )
            throw std::invalid_argument( "Invalid exploration count, strength or regularization" );
        stages.push_back( { unsigned( n ), s, l } );
    }
    if( stages.empty() || text.back() == ',' )
        throw std::invalid_argument( "Empty or incomplete exploration sequence" );
    return stages;
}

inline int loPredCtrlAcc_operational::setupINDI()
{
    createStandardIndiToggleSw( m_indiP_learning, "learn", "Learning", "DDSPC" );
    if( registerIndiPropertyNew( m_indiP_learning, INDI_NEWCALLBACK( m_indiP_learning ) ) < 0 )
        return -1;
    createStandardIndiToggleSw( m_indiP_predicting, "predict", "Prediction", "DDSPC" );
    if( registerIndiPropertyNew( m_indiP_predicting, INDI_NEWCALLBACK( m_indiP_predicting ) ) < 0 )
        return -1;
    createStandardIndiToggleSw( m_indiP_learnStd, "learn_std", "Standard learning sequence", "DDSPC" );
    if( registerIndiPropertyNew( m_indiP_learnStd, INDI_NEWCALLBACK( m_indiP_learnStd ) ) < 0 )
        return -1;
    createStandardIndiRequestSw( m_indiP_reset, "reset_model", "Reset models", "DDSPC" );
    if( registerIndiPropertyNew( m_indiP_reset, INDI_NEWCALLBACK( m_indiP_reset ) ) < 0 )
        return -1;
    createStandardIndiSelectionSw( m_indiP_mode, "ddspc_mode", { "standard", "accel_assisted" }, "Mode", "DDSPC" );
    m_indiP_mode["standard"].setSwitchState( pcf::IndiElement::On );
    if( registerIndiPropertyNew( m_indiP_mode, INDI_NEWCALLBACK( m_indiP_mode ) ) < 0 )
        return -1;
    createStandardIndiNumber<float>( m_indiP_gain, "gain", -10, 10, .001, "%0.4f" );
    m_indiP_gain["current"].set( m_gain );
    m_indiP_gain["target"].set( m_gain );
    if( registerIndiPropertyNew( m_indiP_gain, INDI_NEWCALLBACK( m_indiP_gain ) ) < 0 )
        return -1;
    createStandardIndiText( m_indiP_exploration, "exploration_sequence", "Exploration sequence", "DDSPC" );
    if( registerIndiPropertyNew( m_indiP_exploration, INDI_NEWCALLBACK( m_indiP_exploration ) ) < 0 )
        return -1;
    createROIndiText( m_indiP_control, "ddspc_control", "requested_mode" );
    for( const auto &key : { "standard_ready",
                             "assisted_ready",
                             "learning",
                             "prediction",
                             "accel_sync",
                             "standard_state",
                             "assisted_state" } )
        m_indiP_control.add( pcf::IndiElement( key, "false" ) );
    if( registerIndiPropertyReadOnly( m_indiP_control ) < 0 )
        return -1;
    createROIndiNumber( m_indiP_performance, "ddspc_performance" );
    for( const auto &key : { "standard_updates",
                             "assisted_updates",
                             "standard_rmse",
                             "assisted_rmse",
                             "matched_standard_rmse",
                             "matched_assisted_rmse",
                             "matched_targets",
                             "transport_published",
                             "transport_acks" } )
        m_indiP_performance.add( pcf::IndiElement( key, 0.0 ) );
    if( registerIndiPropertyReadOnly( m_indiP_performance ) < 0 )
        return -1;
    createROIndiText( m_indiP_safety, "safety", "output_state" );
    for( const auto &key : { "dm_write_enabled", "zero_confirmed" } )
        m_indiP_safety.add( pcf::IndiElement( key, "false" ) );
    if( registerIndiPropertyReadOnly( m_indiP_safety ) < 0 )
        return -1;
    createROIndiNumber( m_indiP_adc, "adc_values" );
    for( const auto &key : { "raw0", "raw1", "accel0", "accel1", "rate_hz", "valid" } )
        m_indiP_adc.add( pcf::IndiElement( key, 0.0 ) );
    if( registerIndiPropertyReadOnly( m_indiP_adc ) < 0 )
        return -1;
    return 0;
}

inline int loPredCtrlAcc_operational::appStartup()
{
    if( m_shutdown || !m_models || m_pixels.empty() || setupINDI() < 0 )
        return -1;
    try
    {
        if( m_accelEnabled )
        {
            try
            {
                if( m_syncDevice.empty() || ( m_syncProperty != "timingDiag" && m_syncProperty != "ddspcTiming" ) )
                    throw std::invalid_argument( "Configure the existing upstream ADC lock diagnostic" );
                m_adc = std::make_unique<DDSPC::AdcReader>( m_adcConfig );
                m_adc->start();
                if( registerIndiPropertySet(
                        m_indiP_sync, m_syncDevice, m_syncProperty, INDI_SETCALLBACK( m_indiP_sync ) ) < 0 )
                    throw std::runtime_error( "Could not subscribe to upstream synchronization" );
            }
            catch( const std::exception &error )
            {
                m_adc.reset();
                log<text_log>( std::string( "ACC assistance unavailable; standard DDSPC remains available: " ) +
                               error.what() );
            }
        }
        SyncoOutputConfig output;
        // Operator predict.toggle and the existing producer protocol supply intent/ownership. No extra trust switches.
        output.m_authorized = output.m_routingVerified = output.m_producerDefaults = true;
        output.m_width                                                             = m_calWidth;
        output.m_height                                                            = m_calHeight;
        m_output.configure( output, m_channel, m_syncopationTiming, m_writeLeadUs, m_pixels.size() );
        m_output.start( [this] { return outputValid(); },
                        [this]( const synco::Request &, const synco::Record &, std::int64_t, std::span<float> pixels )
                        { return outputCandidate( pixels ); },
                        [this]( std::span<const float> pixels ) { copied( pixels ); } );
        m_output.initialize();
        SHMIMMONITOR_APP_STARTUP;
    }
    catch( const std::exception &error )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
    }
    state( stateCodes::OPERATING );
    return 0;
}

inline int loPredCtrlAcc_operational::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    SHMIMMONITOR_UPDATE_INDI;
    const auto            output = m_output.status();
    DDSPC::ShadowStatus   standard, assisted;
    bool                  learning, predicting;
    std::uint64_t         matched;
    std::array<double, 2> squared;
    { //mutex scope
        std::lock_guard<std::mutex> modelLock( m_modelMutex );
        if( !output.m_requested )
            m_predicting = false;
        standard   = m_models->status( DDSPC::ShadowMode::Standard );
        assisted   = m_models->status( DDSPC::ShadowMode::AccelAssisted );
        learning   = m_learning;
        predicting = m_predicting;
        matched    = m_matchedTargets;
        squared    = m_matchedSquared;
    }
    const auto now   = synco::monotonicNs();
    const auto last  = m_wfsReceipt.load();
    const bool fresh = m_inputValid.load() && last >= 0 && now >= last && now - last <= m_inputTimeout * 1e9;
    const bool valid = outputValid();
    if( m_adc )
    {
        const auto  adc    = m_adc->status();
        const auto &sample = adc.m_latest;
        updatesIfChanged<double>( m_indiP_adc,
                                  { "raw0", "raw1", "accel0", "accel1", "rate_hz", "valid" },
                                  { double( sample.m_frame.m_raw[0] ),
                                    double( sample.m_frame.m_raw[1] ),
                                    sample.m_acceleration[0],
                                    sample.m_acceleration[1],
                                    adc.m_rateHz,
                                    double( adc.m_valid ) },
                                  adc.m_valid ? INDI_OK : INDI_IDLE );
    }
    const bool selectedReady = m_mode.load() == DDSPC::ShadowMode::Standard ? standard.m_ready : assisted.m_ready;
    updatesIfChanged<std::string>(
        m_indiP_control,
        { "requested_mode",
          "standard_ready",
          "assisted_ready",
          "learning",
          "prediction",
          "accel_sync",
          "standard_state",
          "assisted_state" },
        { m_mode.load() == DDSPC::ShadowMode::Standard ? "standard" : "accel_assisted",
          standard.m_ready && fresh ? "true" : "false",
          assisted.m_ready && fresh && m_assistedValid.load() && m_syncLocked.load() ? "true" : "false",
          learning ? "true" : "false",
          !predicting             ? "off"
          : !valid                ? "inhibited"
          : !selectedReady        ? "warming"
          : output.m_acknowledged ? "active"
                                  : "awaiting_copy",
          m_syncLocked.load() ? "true" : "false",
          !fresh             ? "awaiting_wfs"
          : standard.m_ready ? "ready"
                             : DDSPC::shadow_reason( standard.m_reason ),
          !fresh                    ? "awaiting_wfs"
          : !m_syncLocked.load()    ? "awaiting_upstream_lock"
          : !m_assistedValid.load() ? "awaiting_acc_samples"
          : assisted.m_ready        ? "ready"
                                    : DDSPC::shadow_reason( assisted.m_reason ) },
        INDI_OK );
    updatesIfChanged<double>( m_indiP_performance,
                              { "standard_updates",
                                "assisted_updates",
                                "standard_rmse",
                                "assisted_rmse",
                                "matched_standard_rmse",
                                "matched_assisted_rmse",
                                "matched_targets",
                                "transport_published",
                                "transport_acks" },
                              { double( standard.m_updates ),
                                double( assisted.m_updates ),
                                standard.m_originRmse,
                                assisted.m_originRmse,
                                matched ? std::sqrt( squared[0] / matched ) : 0,
                                matched ? std::sqrt( squared[1] / matched ) : 0,
                                double( matched ),
                                double( output.m_published ),
                                double( output.m_acknowledgements ) },
                              INDI_OK );
    updatesIfChanged<std::string>( m_indiP_safety,
                                   { "output_state", "dm_write_enabled", "zero_confirmed" },
                                   { output.m_reason,
                                     output.m_requested && !output.m_fault ? "true" : "false",
                                     output.m_zeroConfirmed ? "true" : "false" },
                                   output.m_fault ? INDI_ALERT : INDI_OK );
    updateSwitchIfChanged( m_indiP_predicting,
                           "toggle",
                           predicting ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           output.m_fault ? INDI_ALERT : INDI_OK );
    return 0;
}

inline int loPredCtrlAcc_operational::appShutdown()
{
    const bool removed = m_output.stop();
    SHMIMMONITOR_APP_SHUTDOWN;
    m_adc.reset();
    if( !removed )
        log<software_error>(
            { __FILE__, __LINE__, "Own-channel zero acknowledgement unavailable; removal unconfirmed" } );
    return removed ? 0 : -1;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_learning )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_learning, ipRecv );
    if( !ipRecv.find( "toggle" ) )
        return -1;
    const auto value = ipRecv["toggle"].getSwitchState();
    if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        return -1;
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    m_learning = value == pcf::IndiElement::On;
    for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
        m_models->training( mode, m_learning );
    if( !m_learning )
    {
        m_exploration.clear();
        updateSwitchIfChanged( m_indiP_learnStd, "toggle", pcf::IndiElement::Off, INDI_OK );
    }
    updateSwitchIfChanged( m_indiP_learning, "toggle", value, INDI_OK );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_predicting )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_predicting, ipRecv );
    if( !ipRecv.find( "toggle" ) )
        return -1;
    const auto value = ipRecv["toggle"].getSwitchState();
    if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        return -1;
    const bool                  enabled = value == pcf::IndiElement::On;
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    if( enabled && !m_predicting )
    { //mutex scope
        std::lock_guard<std::mutex> commandLock( m_commandMutex );
        std::fill( m_pixels.begin(), m_pixels.end(), 0 );
    }
    if( enabled && !m_models->status( m_mode.load() ).m_ready &&
        !( m_learning && ( m_gain != 0 || !m_exploration.empty() ) ) )
        return -1;
    const bool accepted = m_output.request( enabled );
    m_predicting        = enabled && accepted;
    updateSwitchIfChanged( m_indiP_predicting,
                           "toggle",
                           m_predicting ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           accepted ? INDI_OK : INDI_ALERT );
    return accepted ? 0 : -1;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_learnStd )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_learnStd, ipRecv );
    if( !ipRecv.find( "toggle" ) )
        return -1;
    const auto value = ipRecv["toggle"].getSwitchState();
    if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        return -1;
    const bool                  enabled = value == pcf::IndiElement::On;
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    if( enabled && !m_predicting )
    { //mutex scope
        std::lock_guard<std::mutex> commandLock( m_commandMutex );
        std::fill( m_pixels.begin(), m_pixels.end(), 0 );
    }
    if( enabled && !m_output.request( true ) )
        return -1;
    m_learning = enabled;
    if( enabled )
    {
        m_exploration = parseExploration( "1000,0.1,100,1000,0.1,3,1000,0.1,0.1" );
        m_predicting  = true;
    }
    else
        m_exploration.clear();
    for( auto mode : { DDSPC::ShadowMode::Standard, DDSPC::ShadowMode::AccelAssisted } )
        m_models->training( mode, m_learning );
    updateSwitchIfChanged( m_indiP_learning, "toggle", value, INDI_OK );
    updateSwitchIfChanged( m_indiP_learnStd, "toggle", value, INDI_OK );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_reset )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_reset, ipRecv );
    if( !ipRecv.find( "request" ) || ipRecv["request"].getSwitchState() != pcf::IndiElement::On )
        return 0;
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    if( !removed() || !m_output.clear_fault() )
        return -1;
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    m_predicting = false;
    m_exploration.clear();
    resetModels();
    updateSwitchIfChanged( m_indiP_reset, "request", pcf::IndiElement::Off, INDI_OK );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_mode )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_mode, ipRecv );
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    if( !removed() )
        return -1;
    unsigned          selected = 0;
    DDSPC::ShadowMode mode     = DDSPC::ShadowMode::Standard;
    for( const auto &entry : ipRecv.getElements() )
    {
        if( entry.first != "standard" && entry.first != "accel_assisted" )
            return -1;
        if( entry.second.getSwitchState() == pcf::IndiElement::On )
        {
            ++selected;
            mode = entry.first == "standard" ? DDSPC::ShadowMode::Standard : DDSPC::ShadowMode::AccelAssisted;
        }
        else if( entry.second.getSwitchState() != pcf::IndiElement::Off )
            return -1;
    }
    if( selected != 1 )
        return -1;
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    m_mode.store( mode );
    m_predicting = false;
    updateSwitchIfChanged( m_indiP_mode,
                           "standard",
                           mode == DDSPC::ShadowMode::Standard ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           INDI_OK );
    updateSwitchIfChanged( m_indiP_mode,
                           "accel_assisted",
                           mode == DDSPC::ShadowMode::AccelAssisted ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           INDI_OK );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_gain )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_gain, ipRecv );
    if( !ipRecv.find( "target" ) )
        return -1;
    float gain;
    try
    {
        gain = number( ipRecv["target"] );
    }
    catch( const std::exception & )
    {
        return -1;
    }
    if( !std::isfinite( gain ) || std::abs( gain ) > 10 )
        return -1;
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    m_gain = gain;
    updatesIfChanged<float>( m_indiP_gain, { "current", "target" }, { gain, gain }, INDI_OK );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_exploration )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_exploration, ipRecv );
    if( !ipRecv.find( "target" ) )
        return -1;
    try
    {
        const auto                  text   = ipRecv["target"].get<std::string>();
        auto                        stages = parseExploration( text );
        std::lock_guard<std::mutex> indiLock( m_indiMutex );
        std::lock_guard<std::mutex> modelLock( m_modelMutex );
        m_exploration = std::move( stages );
        updatesIfChanged<std::string>( m_indiP_exploration, { "current", "target" }, { text, text }, INDI_OK );
        return 0;
    }
    catch( const std::exception & )
    {
        return -1;
    }
}

inline INDI_SETCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_sync )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_sync, ipRecv );
    const bool  adc     = m_syncProperty == "ddspcTiming";
    const char *enabled = adc ? "enabled" : "mode_code", *locked = adc ? "valid" : "delay_lock";
    bool        valid = ipRecv.find( enabled ) && ipRecv.find( locked );
    try
    {
        valid = valid && number( ipRecv[enabled] ) == 1 && number( ipRecv[locked] ) == 1;
    }
    catch( const std::exception & )
    {
        valid = false;
    }
    m_syncLocked.store( valid );
    if( !valid )
        m_assistedValid.store( false );
    return 0;
}

inline double loPredCtrlAcc_operational::number( const pcf::IndiElement &element )
{
    const auto   text = element.getValue();
    std::size_t  used;
    const double value = std::stod( text, &used );
    if( used != text.size() || !std::isfinite( value ) )
        throw std::invalid_argument( "Invalid finite INDI number" );
    return value;
}
} // namespace MagAOX::app
#endif
