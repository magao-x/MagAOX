/** \file sim_bridge.cpp
 * \brief Test-only array interface to the actual accelSim source and causal DDSPC; no device or shmim linkage.
 */
#include "async_shadow.hpp"
#include "operational_law.hpp"
#include "accelSimModel.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>

namespace
{
using namespace DDSPC;
namespace Accel = MagAOX::app::accelSimDetail;

/// Private source instance; its pointing output goes exclusively to the simulated plant.
struct Source
{
    Accel::model m_model; ///< Existing producer physics/ADC, without its runtime acquisition or output helpers.

    /// Retain only the reference 28/56-Hz modes with an explicitly synthetic amplitude multiplier.
    Source( double        scale /**< [in] Positive simulation amplitude multiplier. */,
            std::uint64_t seed /**< [in] Identical realization for all comparisons. */ );
};
Source::Source( double scale, std::uint64_t seed )
{
    if( !std::isfinite( scale ) || scale <= 0 )
        throw std::invalid_argument( "Invalid simulation amplitude" );
    Accel::parameters config;
    config.m_fps  = 8000;
    config.m_seed = seed;
    auto modes    = Accel::referenceModes( config.m_modeSeed );
    modes.erase( std::remove_if( modes.begin(),
                                 modes.end(),
                                 []( const auto &m ) { return m.m_frequency != 28 && m.m_frequency != 56; } ),
                 modes.end() );
    for( auto &m : modes )
        m.m_accelRms *= scale;
    m_model.configure( config, modes );
}

/// Fixed plain-data FFI diagnostics; no reader has access to source pointing or mutable coefficients.
struct Prediction
{
    int           m_valid{ 0 }, m_ready{ 0 }, m_reason{ 0 }, m_rank{ 0 }; ///< Unchanged snapshot gates.
    std::uint64_t m_version{ 0 }, m_sequence{ 0 }, m_age{ 0 };            ///< Actual evidence used by prediction.
    float         m_ax{ 0 }, m_ay{ 0 }; ///< Decoded four-sample causal mean, in declared feature units.
    double m_fitRmse{ 0 }, m_thetaNorm{ 0 }, m_controlError{ 0 }, m_gainNorm{ 0 }; ///< Diagnostic-only model evidence.
};

/// One simulated trajectory, with independently trained standard/assisted workers and no output sink.
struct Controller
{
    ShadowConfig m_config;               ///< Existing two-mode reference model dimensions/cost.
    ShadowMode   m_selected;             ///< Only the selected proposal is returned; both fits train.
    double       m_wfsUnit, m_accelUnit; ///< Fixed producer-derived normalization, never fitted from a realization.
    Accel::parameters  m_adc;            ///< Active ADC inverse calibration, copied explicitly from producer settings.
    OfflineAsyncShadow m_worker;         ///< Existing bounded model/publication mechanism, manually serviced offline.
    OfflineRecord      m_record;         ///< One pending WFS/accepted-input association.
    OfflineSnapshot    m_snapshot;       ///< Coherent private copy; read-only law input in the opt-in offline profile.
    std::deque<Matrix> m_y, m_u, m_a;    ///< Causal frontend history; u is populated only after plant consumption.
    std::deque<bool>   m_sensorValid;    ///< Assisted predictions require a fully valid causal feature history.
    std::uint64_t      m_next{ 0 };      ///< Required contiguous source frame identity.
    bool               m_pending{ false }; ///< Query must be completed by one confirmed plant read.
#ifdef DDSPC_SIM_LEGACY_LAW
    Matrix        m_operationalGain; ///< Private law-adapter cache; never mutates a published worker snapshot.
    std::uint64_t m_operationalVersion{ std::uint64_t( -1 ) }; ///< Fit identity of the cached law.

    /// Cache Legacy algebra for one fitted version; diagnostic export never authorizes prediction.
    const Matrix &operational_gain();
#endif

    /// Construct a strict QR pair and immutable simulation units.
    Controller( int           selected /**< [in] 0 standard, 1 assisted. */,
                const double *units /**< [in] Fixed WFS/acceleration units and three active ADC parameters. */ );

    /// Assemble current WFS and past consumed inputs; no future command or disturbance is accepted.
    Matrix state() const;
};

/// Preserve the existing general DDSPC configuration; no frequency-specific controller coefficients.
ShadowConfig configuration()
{
    ShadowConfig c;
    c.m_modes        = 2;
    c.m_history      = 6; // Five past inputs retain the extra aoSim delay without shortening disturbance history.
    c.m_future       = 3;
    c.m_accelHistory = 5;
    c.m_gamma        = .9995f;
    c.m_prior        = .001f;
    c.m_lambda       = .02f;
    c.m_useQrd       = true;
#ifdef DDSPC_SIM_WFS_HISTORY
    c.m_history = DDSPC_SIM_WFS_HISTORY;
#endif
#ifdef DDSPC_SIM_QR_GAMMA
    c.m_gamma = DDSPC_SIM_QR_GAMMA;
#endif
#ifdef DDSPC_SIM_CONTROL_PENALTY
    c.m_lambda = DDSPC_SIM_CONTROL_PENALTY;
#endif
#ifdef DDSPC_SIM_ACCEL_HISTORY
    c.m_accelHistory = DDSPC_SIM_ACCEL_HISTORY;
    if( c.m_accelHistory < 1 || c.m_accelHistory > c.m_history )
        throw std::invalid_argument( "Unsupported simulation sensor history" );
#endif
#ifdef DDSPC_SIM_QR_PRIOR
    c.m_prior = DDSPC_SIM_QR_PRIOR;
#endif
    return c;
}
Controller::Controller( int selected, const double *units )
    : m_config( configuration() ), m_selected( selected == 0 ? ShadowMode::Standard : ShadowMode::AccelAssisted ),
      m_wfsUnit( units[0] ), m_accelUnit( units[1] ), m_worker( m_config, 32, 0, 1500000 ), m_record( m_config ),
      m_snapshot( m_config, m_selected )
{
    m_adc.m_zero        = units[2];
    m_adc.m_voltsPerBit = units[3];
    m_adc.m_sensitivity = units[4];
    if( selected < 0 || selected > 1 ||
        !std::all_of( units, units + 5, []( double v ) { return std::isfinite( v ); } ) || m_wfsUnit <= 0 ||
        m_accelUnit <= 0 || units[3] <= 0 || units[4] <= 0 )
        throw std::invalid_argument( "Invalid simulation controller configuration" );
}
Matrix Controller::state() const
{
    const int a = 2 * m_config.m_history, b = 2 * ( m_config.m_history - 1 );
    Matrix    x( a + b + ( m_selected == ShadowMode::AccelAssisted ? 2 * m_config.m_accelHistory : 0 ), 1 );
    for( int i = 0; i < m_config.m_history; ++i )
        x.middleRows( i * 2, 2 ) = m_y[i];
    for( int i = 0; i < m_config.m_history - 1; ++i )
        x.middleRows( a + i * 2, 2 ) = m_u[i];
    if( m_selected == ShadowMode::AccelAssisted )
        for( int i = 0; i < m_config.m_accelHistory; ++i )
            x.middleRows( a + b + i * 2, 2 ) = m_a[i];
    return x;
}
#ifdef DDSPC_SIM_LEGACY_LAW
const Matrix &Controller::operational_gain()
{
    if( m_operationalVersion != m_snapshot.m_info.m_version )
    {
        realT feedback = .3f;
    #ifdef DDSPC_SIM_FEEDBACK_GAIN
        feedback = DDSPC_SIM_FEEDBACK_GAIN;
    #endif
        m_operationalGain = operational_legacy_gain(
            m_snapshot.m_theta, m_config.m_modes, m_config.m_history, m_config.m_lambda, feedback );
        m_operationalVersion = m_snapshot.m_info.m_version;
    }
    return m_operationalGain;
}
#endif
} // namespace

extern "C"
{
    /// Expose the diagnostics layout size so the Python caller verifies its ABI before use.
    std::size_t ddspc_sim_prediction_size()
    {
        return sizeof( Prediction );
    }

    /// Create only the numerical accelSim source; never construct the application or open a stream.
    void *ddspc_sim_source_create( double        scale /**< [in] Explicit synthetic mode amplitude multiplier. */,
                                   std::uint64_t seed /**< [in] Producer RNG identity. */ )
    {
        try
        {
            return new Source( scale, seed );
        }
        catch( ... )
        {
            return nullptr;
        }
    }

    /// Destroy privately owned numerical source storage.
    void ddspc_sim_source_destroy( void *source /**< [in] Source returned by source_create. */ )
    {
        delete static_cast<Source *>( source );
    }

    /// Step the actual producer once; the two simultaneous raw channels and pointing describe the same sample.
    int ddspc_sim_source_step( void          *source /**< [in] Private source. */,
                               std::uint16_t *adc /**< [out] Two raw X/Y UINT16 values. */,
                               double        *pointing /**< [out] Two pointing values in arcsec, plant only. */ )
    {
        try
        {
            const auto &s = static_cast<Source *>( source )->m_model.step();
            std::copy( s.m_adc.begin(), s.m_adc.end(), adc );
            std::copy( s.m_pointing.begin(), s.m_pointing.end(), pointing );
            return 0;
        }
        catch( ... )
        {
            return -1;
        }
    }

    /// Return fixed source units and ADC settings without inspecting future samples or normalizing a realized trace.
    int ddspc_sim_source_units(
        void   *source /**< [in] Private configured source. */,
        double *units /**< [out] 28-Hz pointing RMS, acceleration RMS, zero, volts/count, sensitivity. */ )
    {
        const auto &s = static_cast<Source *>( source )->m_model;
        units[0]      = s.m_modes.front().m_axes[0].m_rms;
        units[1]      = s.m_modes.front().m_spec.m_accelRms;
        units[2]      = s.m_config.m_zero;
        units[3]      = s.m_config.m_voltsPerBit;
        units[4]      = s.m_config.m_sensitivity;
        return 0;
    }

    /// Audit ADC channel encoding against the producer's last noisy acceleration; never a controller input.
    int ddspc_sim_source_audit( void   *source /**< [in] Source after source_step. */,
                                double *acceleration /**< [out] Noisy physical acceleration for channels 0 and 1. */ )
    {
        const auto &a = static_cast<Source *>( source )->m_model.m_sample.m_acceleration;
        std::copy( a.begin(), a.end(), acceleration );
        return 0;
    }

    /// Allocate independent DDSPC fits; the returned interface has no source pointer or physical output facility.
    void *
    ddspc_sim_controller_create( int           selected /**< [in] 0 standard or 1 assisted. */,
                                 const double *units /**< [in] Source units, zero, volts/count and sensitivity. */ )
    {
        try
        {
            return new Controller( selected, units );
        }
        catch( ... )
        {
            return nullptr;
        }
    }

    /// Destroy the private offline workers without any output operation.
    void ddspc_sim_controller_destroy( void *controller /**< [in] Controller returned by controller_create. */ )
    {
        delete static_cast<Controller *>( controller );
    }

    /// Return the actual canonical column count for read-only diagnostic snapshot allocation.
    int
    ddspc_sim_model_features( void *controller /**< [in] Configured private controller, including before training. */ )
    {
        return static_cast<Controller *>( controller )->m_snapshot.m_theta.cols();
    }

    /// Report the actual WFS-history dimension; feature count alone cannot distinguish WFS and ADC lags.
    int ddspc_sim_model_history( void *controller /**< [in] Private configured controller. */ )
    {
        return static_cast<Controller *>( controller )->m_config.m_history;
    }

    /// Copy a coherent fit for offline scoring; return 1 for unready diagnostics, 0 for ready, -1 for unavailable.
    int ddspc_sim_copy_model( void          *controller /**< [in] Private controller with completed calibration. */,
                              float         *theta /**< [out] Canonical prediction coefficients, column-major. */,
                              int            thetaCount /**< [in] Exact destination coefficient count. */,
                              float         *gain /**< [out] Full offline-profile gain, column-major. */,
                              int            gainCount /**< [in] Exact destination gain count. */,
                              std::uint64_t *version /**< [out] Copied coefficient version. */ )
    {
        auto &c = *static_cast<Controller *>( controller );
        if( thetaCount != c.m_snapshot.m_theta.size() || gainCount != c.m_snapshot.m_gain.size() ||
            !c.m_worker.read_snapshot( c.m_selected, c.m_snapshot ) || !c.m_snapshot.m_info.m_version )
            return -1;
        std::copy( c.m_snapshot.m_theta.data(), c.m_snapshot.m_theta.data() + thetaCount, theta );
#ifdef DDSPC_SIM_LEGACY_LAW
        const Matrix &actual = c.operational_gain();
        std::copy( actual.data(), actual.data() + gainCount, gain );
#else
        std::copy( c.m_snapshot.m_gain.data(), c.m_snapshot.m_gain.data() + gainCount, gain );
#endif
        *version = c.m_snapshot.m_info.m_version;
        return c.m_snapshot.m_info.m_status.m_ready ? 0 : 1;
    }

    /// Query from observed WFS and four causal ADC pairs; a separate consumed-input record must finish this frame.
    int ddspc_sim_query( void         *controller /**< [in] Numerical controller. */,
                         std::uint64_t sequence /**< [in] Contiguous WFS counter, starting at zero. */,
                         std::uint64_t now /**< [in] Logical WFS measurement/decision time, ns. */,
                         std::uint64_t firstSample /**< [in] First of four ADC sample times, spaced 125000 ns. */,
                         const float  *wfs /**< [in] Two measured WFS values in arcsec. */,
                         const std::uint16_t *raw /**< [in] Four interleaved X/Y ADC pairs. */,
                         float      *proposal /**< [out] Two proposed commands in arcsec, zero unless valid. */,
                         Prediction *info /**< [out] Snapshot and decoded-feature diagnostics. */ )
    {
        proposal[0] = proposal[1] = 0;
        *info                     = Prediction{};
        try
        {
            auto &c = *static_cast<Controller *>( controller );
            if( c.m_pending || sequence != c.m_next || now < 500000 || firstSample > now ||
                now - firstSample < 376000 || now - firstSample > 1500000 )
                throw std::invalid_argument( "Invalid simulation timing or query order" );
            Matrix y( 2, 1 ), a = Matrix::Zero( 2, 1 );
            bool   valid = true;
            for( int axis = 0; axis < 2; ++axis )
            {
                y( axis ) = wfs[axis] / c.m_wfsUnit;
                for( int sample = 0; sample < 4; ++sample )
                {
                    const auto count = raw[2 * sample + axis];
                    valid            = valid && count > 0 && count < 4095;
                    a( axis ) += float( ( double( count ) - c.m_adc.m_zero ) * c.m_adc.m_voltsPerBit /
                                        c.m_adc.m_sensitivity / c.m_accelUnit / 4 );
                }
            }
            if( !y.allFinite() )
                throw std::invalid_argument( "Invalid WFS" );
            c.m_y.push_front( y );
            c.m_a.push_front( a );
            c.m_sensorValid.push_front( valid );
            if( c.m_y.size() > std::size_t( c.m_config.m_history ) )
            {
                c.m_y.pop_back();
                c.m_a.pop_back();
                c.m_sensorValid.pop_back();
            }
            const auto selected = c.m_selected;
            const bool featureHistory =
                selected == ShadowMode::Standard ||
                std::all_of( c.m_sensorValid.begin(), c.m_sensorValid.end(), []( bool v ) { return v; } );
            if( c.m_y.size() == std::size_t( c.m_config.m_history ) &&
                c.m_u.size() >= std::size_t( c.m_config.m_history - 1 ) && featureHistory )
            {
                Matrix            out( 2, 1 );
                OfflinePrediction p;
                info->m_valid    = c.m_worker.predict( selected, 1, sequence, now, now, 2000000, c.state(), out, p );
                info->m_ready    = p.m_model.m_status.m_ready;
                info->m_reason   = int( p.m_reason );
                info->m_rank     = p.m_model.m_status.m_rank;
                info->m_version  = p.m_model.m_version;
                info->m_sequence = p.m_model.m_sequence;
                info->m_age      = p.m_ageNs;
                if( info->m_valid )
                    for( int axis = 0; axis < 2; ++axis )
                        proposal[axis] = out( axis ) * c.m_wfsUnit;
            }
            const bool copied = c.m_worker.read_snapshot( selected, c.m_snapshot );
            if( copied )
            {
                info->m_fitRmse   = c.m_snapshot.m_info.m_status.m_rmse;
                info->m_thetaNorm = c.m_snapshot.m_theta.norm();
                info->m_gainNorm  = c.m_snapshot.m_gain.norm();
                // Known aoSim lag-zero response in newest-first F=3 coordinates, diagnostic only.
                Matrix expected = Matrix::Zero( 6, 6 );
                expected.block( 0, 2, 2, 2 ).setIdentity();
                expected.block( 2, 4, 2, 2 ).setIdentity();
                info->m_controlError = ( c.m_snapshot.m_theta.rightCols( 6 ) - expected ).norm();
            }
#ifdef DDSPC_SIM_LEGACY_LAW
            if( info->m_valid )
            {
                if( !copied || c.m_snapshot.m_info.m_version != info->m_version )
                    throw std::runtime_error( "Incoherent operational-law snapshot" );
                const Matrix out = c.operational_gain().bottomRows( 2 ) * c.state();
                if( !out.allFinite() )
                    throw std::runtime_error( "Nonfinite operational-law command" );
                for( int axis = 0; axis < 2; ++axis )
                    proposal[axis] = out( axis ) * c.m_wfsUnit;
                info->m_gainNorm = c.operational_gain().norm();
            }
#endif
            info->m_ax              = a( 0 );
            info->m_ay              = a( 1 );
            auto &r                 = c.m_record;
            r.m_data.m_index        = sequence;
            r.m_data.m_residual     = y;
            r.m_data.m_acceleration = a;
            r.m_data.m_wfsValid = r.m_data.m_appliedValid = r.m_data.m_timingValid = true;
            r.m_data.m_accelValid                                                  = valid;
            r.m_measurementNs = r.m_appliedNs = r.m_appliedAvailableNs = r.m_availableNs = now;
            r.m_exposureStartNs                                                          = now - 500000;
            r.m_accelNs                                                                  = firstSample + 375000;
            r.m_accelAvailableNs                                                         = r.m_accelNs + 1000;
            c.m_pending                                                                  = true;
            return 0;
        }
        catch( ... )
        {
            return -1;
        }
    }

    /// Train only after aoSim has read its DM channel; never substitute the proposed command for this observation.
    int ddspc_sim_accept( void        *controller /**< [in] Controller with a pending WFS query. */,
                          const float *consumed /**< [in] Two actual values read by aoSim.update_dm(), arcsec. */ )
    {
        try
        {
            auto &c = *static_cast<Controller *>( controller );
            if( !c.m_pending )
                throw std::invalid_argument( "No pending observation" );
            Matrix u( 2, 1 );
            for( int axis = 0; axis < 2; ++axis )
                u( axis ) = consumed[axis] / c.m_wfsUnit;
            if( !u.allFinite() )
                throw std::invalid_argument( "Invalid consumed input" );
            c.m_record.m_data.m_appliedCommand = u;
            if( !c.m_worker.submit( c.m_record, c.m_record.m_availableNs ) )
                throw std::runtime_error( "Simulation record rejected" );
            c.m_worker.manual_step( ShadowMode::Standard, c.m_record.m_availableNs + 1000 );
            c.m_worker.manual_step( ShadowMode::AccelAssisted, c.m_record.m_availableNs + 1000 );
            c.m_u.push_front( u );
            if( c.m_u.size() > std::size_t( c.m_config.m_history ) )
                c.m_u.pop_back();
            ++c.m_next;
            c.m_pending = false;
            return 0;
        }
        catch( ... )
        {
            return -1;
        }
    }
} // extern C
