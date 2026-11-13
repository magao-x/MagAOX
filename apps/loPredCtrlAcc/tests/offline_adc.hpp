/** \file offline_adc.hpp
 * \brief Owned synthetic ADC input and spectral analysis without clocks, files, streams or runtime libraries.
 * \ingroup loPredCtrlAcc_unit_test
 */
#ifndef loPredCtrlAcc_offline_adc_hpp
#define loPredCtrlAcc_offline_adc_hpp
#include "../adc_input.hpp"
#include <random>

/// \cond OFFLINE_TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
/// Fixed SI waveform/calibration and explicit consumer loss policy; no accelSim dependency.
struct SyntheticAdcConfig
{
    DDSPC::AdcConfig m_calibration; ///< Actual decoder calibration, also used to encode independent synthetic counts.
    std::array<double, 2> m_lowAmplitude{ .15, .12 }, m_highAmplitude{ .08, .06 };      ///< Peak acceleration in m/s².
    double m_lowHz{ 28.5 }, m_highHz{ 56 }, m_forceNoise{ .02 }, m_sensorNoise{ .003 }; ///< SI frequencies/noise.
    std::uint64_t m_seed{ 721 }, m_jitterNs{ 0 }, m_dropEvery{ 0 }; ///< Reproducible input, timing and lost reads.
    std::uint64_t m_dropStart{ 0 }, m_dropCount{ 0 },
        m_dropPeriod{ 0 };       ///< Single/repeated lost-read burst, ADC ticks.
    double m_clockDriftPpm{ 0 }; ///< Independent ADC cadence error relative to WFS; acquisition timestamps stay common.
    std::int64_t m_latencyNs{ 3000 }, m_latencyJitterNs{ 0 }; ///< Receipt latency and independent uniform variation.
};

/// One-plane, memory-only producer; a skipped read loses that payload exactly as a depth-one source can.
class SyntheticAdcSource : public DDSPC::AdcSource
{
    SyntheticAdcConfig    m_config; ///< Immutable scenario; never interprets controller state.
    DDSPC::AdcHeader      m_header{ 3, 2, 1, 1, 0, 0, true, true, true, false }; ///< Exact raw-pair layout.
    DDSPC::AdcFrame       m_frame;      ///< Sole owned plane, including original publication metadata.
    DDSPC::AdcClock       m_clock;      ///< Virtual clock observation, never wall-clock paced.
    std::array<double, 2> m_physical{}; ///< Latent current acceleration, accessible only to the synthetic plant/oracle.
    std::mt19937_64       m_force, m_sensor, m_jitter, m_delivery; ///< Independent waveform, cadence and delivery RNGs.
    std::normal_distribution<double> m_forceNormal, m_sensorNormal; ///< Independent Gaussian samplers.
    std::uint64_t                    m_tick{ 0 }; ///< Generated ADC frames at 125000-ns nominal spacing.

  public:
    /// Validate and construct without performing any external operation.
    explicit SyntheticAdcSource( const SyntheticAdcConfig &config /**< [in] Declared waveform and loss policy. */ );
    /// Generate one acquired/published pair; false asks the consumer to skip this read, not to invent a sample.
    bool advance();
    /// Return owned stable layout/publication metadata.
    DDSPC::AdcHeader header() const override;
    /// Copy the sole owned plane, rejecting other indices.
    DDSPC::AdcFrame frame( std::size_t index /**< [in] Plane zero only. */ ) const override;
    /// Return the known synthetic clock epoch and receipt time.
    DDSPC::AdcClock clock() const override;
    /// Return latent SI acceleration for the synthetic plant; never passed as predictor features.
    std::array<double, 2> physical() const;
};

/// Fit a sinusoid using original acquisition times; quantization/noise and timestamp jitter remain observable.
double tone_amplitude( const std::vector<double>       &values /**< [in] Measured scalar values. */,
                       const std::vector<std::int64_t> &times /**< [in] Matching monotonic ns. */,
                       double                           frequency /**< [in] Positive trial frequency in Hz. */ );
/// Return a Hann-window periodogram, density units value²/Hz, from a uniformly sampled scalar record.
std::vector<double> periodogram( const std::vector<double> &values /**< [in] Uniformly sampled record. */,
                                 double                     rate /**< [in] Samples/second. */,
                                 double                     maxHz = 200 /**< [in] Highest exported frequency. */ );
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
#endif
