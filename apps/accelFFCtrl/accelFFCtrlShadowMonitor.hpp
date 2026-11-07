/** \file accelFFCtrlShadowMonitor.hpp
 * \brief Optional model-coordinate shadow diagnostics and exclusive CSV logging; no actuator or stream writer API.
 * \ingroup accelFFCtrl_files
 */
#ifndef accelFFCtrlShadowMonitor_hpp
#define accelFFCtrlShadowMonitor_hpp

#include <cstdio>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#include <unistd.h>

#include "accelFFCtrlInput.hpp"

namespace MagAOX
{
namespace app
{
namespace accelFFCtrlDetail
{
/// Startup-only diagnostic settings, not a DM calibration or an actuator enable.
struct shadowSettings
{
    bool        m_enabled{ false }; ///< Explicit startup opt-in; default observation behavior is unchanged.
    double      m_gain{ 0 }, m_maxWfsAgeUs{ 2000 }; ///< Model-coordinate file gain and metadata freshness bound.
    std::string m_wfsName, m_logFile;               ///< Operator-verified timing stream and new absolute CSV file.

    /// Validate startup values without opening a stream or file.
    bool valid() const;
};

/// Latest bracketed WFS metadata only; no WFS image/mode interpretation or exposure-center claim.
struct wfsTiming
{
    uint64_t m_counter{ 0 }, m_generation{ 0 }; ///< Publication counter and observer attachment generation.
    int64_t  m_acquired{ 0 },
        m_written{ 0 };    ///< Original realtime metadata nanoseconds, never a semaphore-arrival time.
    bool m_valid{ false }; ///< Stable copy and observed counter/time continuity passed.
};

/// New absolute diagnostic values; invalid records never reuse a previous proposed contribution.
struct shadowStatus
{
    bool   m_enabled{ false }, m_valid{ false }, m_wfsValid{ false }; ///< Numerical diagnostics only; never actuation.
    double m_gain{ 0 }, m_wfsAgeUs{ -1 };                             ///< Applied startup gain and metadata age.
    std::array<double, 2> m_unitGain{}, m_proposed{}; ///< Opposing model-pointing coordinates, not DM coefficients.
};

/// Evaluate a future model-coordinate proposal without changing the estimator, timing or WFS/AO state.
shadowStatus
evaluateShadow( const shadowSettings   &settings /**< [in] startup diagnostic controls */,
                const inputDiagnostics &input /**< [in] immutable owner snapshot */,
                const inputClock       &clock /**< [in] current local clock bracket */,
                const wfsTiming        &wfs /**< [in] stable original WFS metadata */,
                bool pending = false /**< [in] estimator reset/configuration awaits owner acknowledgement */ );

/// One app-thread-owned regular CSV file; no overwrite, symlink following or shared-memory writer capability.
class shadowLog
{
    FILE *m_file{ nullptr }; ///< Owned exclusive diagnostic file, never used by the ADC worker.

  public:
    /// Construct with no filesystem access.
    shadowLog() = default;

    /// A log has exactly one owner.
    shadowLog( const shadowLog &other /**< [in] prohibited second owner */ ) = delete;

    /// A log cannot be assigned to another owner.
    shadowLog &operator=( const shadowLog &other /**< [in] prohibited second owner */ ) = delete;

    /// Close only this regular file.
    ~shadowLog();

    /// Create a new absolute .csv file exclusively and write its schema.
    bool open( const std::string &path /**< [in] new private diagnostic filename */ );

    /// Write one status snapshot at app cadence; shutdown records are explicitly zero and invalid.
    bool write( const inputDiagnostics &input /**< [in] owner snapshot and applied calibration */,
                const inputClock       &clock /**< [in] recording clocks */,
                const wfsTiming        &wfs /**< [in] WFS metadata */,
                const shadowStatus     &status /**< [in] newly evaluated diagnostic proposal */,
                bool                    shutdown = false /**< [in] final inhibited record */ );

    /// Release the diagnostic file without removing it.
    void close();
};

inline bool shadowSettings::valid() const
{
    return std::isfinite( m_gain ) && m_gain >= 0 && m_gain <= 1 && std::isfinite( m_maxWfsAgeUs ) &&
           m_maxWfsAgeUs > 0 && m_maxWfsAgeUs <= 1e6 &&
           ( !m_enabled ||
             ( inputNameValid( m_wfsName ) && m_logFile.starts_with( "/" ) && m_logFile.ends_with( ".csv" ) &&
               m_logFile.find_first_of( "\r\n" ) == std::string::npos ) );
}

inline shadowStatus evaluateShadow( const shadowSettings   &settings,
                                    const inputDiagnostics &input,
                                    const inputClock       &clock,
                                    const wfsTiming        &wfs,
                                    bool                    pending )
{
    shadowStatus result;
    result.m_enabled = settings.m_enabled;
    result.m_gain    = settings.m_gain;
    if( !settings.valid() || clock.m_real <= 0 || clock.m_mono <= 0 || clock.m_uncertainty < 0 )
        return result;
    if( wfs.m_acquired > 0 && wfs.m_acquired <= clock.m_real )
        result.m_wfsAgeUs = ( clock.m_real - wfs.m_acquired ) * 1e-3;
    result.m_wfsValid = wfs.m_valid && wfs.m_counter && wfs.m_acquired > 0 && wfs.m_written >= wfs.m_acquired &&
                        wfs.m_written <= clock.m_real && result.m_wfsAgeUs >= 0 &&
                        result.m_wfsAgeUs <= settings.m_maxWfsAgeUs;
    const auto &p     = input.m_prediction;
    if( pending || p.m_state != estimatorState::ready || p.m_sampleTimeNs <= 0 || p.m_sampleTimeNs > clock.m_mono ||
        clock.m_mono - p.m_sampleTimeNs > input.m_config.m_maxAgeNs || p.m_targetTimeNs < clock.m_mono ||
        p.m_targetTimeNs - p.m_sampleTimeNs > input.m_config.m_maxPredictionNs )
        return result;
    if( !std::isfinite( p.m_pointing[0] ) || !std::isfinite( p.m_pointing[1] ) )
        return result;
    for( size_t axis = 0; axis < 2; ++axis )
        result.m_unitGain[axis] = -p.m_pointing[axis];
    result.m_valid = settings.m_enabled && result.m_wfsValid;
    if( result.m_valid )
        for( size_t axis = 0; axis < 2; ++axis )
            result.m_proposed[axis] = settings.m_gain * result.m_unitGain[axis];
    return result;
}

inline shadowLog::~shadowLog()
{
    close();
}

inline bool shadowLog::open( const std::string &path )
{
    close();
    if( !path.starts_with( "/" ) || !path.ends_with( ".csv" ) )
        return false;
    const int fd = ::open( path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600 );
    if( fd < 0 )
        return false;
    m_file = fdopen( fd, "w" );
    if( !m_file )
    {
        ::close( fd );
        return false;
    }
    std::ostringstream header;
    header << "record_realtime_ns,record_monotonic_ns,adc_counter,adc_generation,applied_revision,"
              "adc_sample_monotonic_ns,prediction_target_monotonic_ns,adc_ready,wfs_counter,wfs_generation,"
              "wfs_acquisition_realtime_ns,wfs_write_realtime_ns,wfs_age_us,wfs_valid,shadow_enabled,"
              "shadow_valid,gain,predicted_tip_model_rad,predicted_tilt_model_rad,"
              "unit_gain_tip_model_rad,unit_gain_tilt_model_rad,proposed_tip_model_rad,proposed_tilt_model_rad,"
              "dm_output_enabled,shutdown,clock_uncertainty_ns,prediction_query_monotonic_ns,"
              "adc_dropped,adc_resets,adc_last_event,adc_service_us,adc_effective_horizon_us";
    for( const char *name : parameterNames )
        header << ',' << name;
    header << '\n';
    const auto text = header.str();
    return std::fwrite( text.data(), 1, text.size(), m_file ) == text.size() && std::fflush( m_file ) == 0;
}

inline bool shadowLog::write( const inputDiagnostics &input,
                              const inputClock       &clock,
                              const wfsTiming        &wfs,
                              const shadowStatus     &status,
                              bool                    shutdown )
{
    if( !m_file )
        return false;
    const auto        &p = input.m_prediction;
    std::ostringstream row;
    row << std::setprecision( 17 ) << clock.m_real << ',' << clock.m_mono << ',' << input.m_counter << ','
        << input.m_generation << ',' << input.m_revision << ',' << p.m_sampleTimeNs << ',' << p.m_targetTimeNs << ','
        << ( !shutdown && p.m_state == estimatorState::ready ) << ',' << wfs.m_counter << ',' << wfs.m_generation << ','
        << wfs.m_acquired << ',' << wfs.m_written << ',' << status.m_wfsAgeUs << ','
        << ( !shutdown && status.m_wfsValid ) << ',' << status.m_enabled << ',' << ( !shutdown && status.m_valid )
        << ',' << status.m_gain << ',' << ( shutdown ? 0 : p.m_pointing[0] ) << ','
        << ( shutdown ? 0 : p.m_pointing[1] );
    for( const auto *values : { &status.m_unitGain, &status.m_proposed } )
        for( double value : *values )
            row << ',' << ( shutdown ? 0 : value );
    row << ",0," << shutdown << ',' << clock.m_uncertainty << ','
        << ( p.m_targetTimeNs > 0 ? p.m_targetTimeNs - input.m_config.m_predictionNs : 0 ) << ',' << input.m_dropped
        << ',' << input.m_resets << ',' << int( input.m_lastEvent ) << ',' << input.m_serviceUs << ','
        << input.m_horizonUs;
    for( size_t n = 0; n < parameterNames.size(); ++n )
        row << ',' << parameterValue( input.m_config, n );
    row << '\n';
    const auto text = row.str();
    return std::fwrite( text.data(), 1, text.size(), m_file ) == text.size() && std::fflush( m_file ) == 0;
}

inline void shadowLog::close()
{
    if( m_file )
        std::fclose( m_file );
    m_file = nullptr;
}

} // namespace accelFFCtrlDetail
} // namespace app
} // namespace MagAOX
#endif
