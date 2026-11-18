/** \file modalPsdProcessor.hpp
 * \brief Header-only template class for modal PSD noise estimation and
 * disturbance extrapolation.
 *
 * \author OpenAI Codex
 */

#ifndef modalPsdProcessor_hpp
#define modalPsdProcessor_hpp

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <mx/error/error.hpp>
#include <mx/math/func/moffat.hpp>
#include <mx/math/vectorUtils.hpp>

namespace MagAOX
{
namespace app
{

/// Header-only helper for modal PSD noise estimation, extrapolation, and LP
/// continuum shaping.
template <typename realT>
class modalPsdProcessor
{
  public:
    /// The default process-method name.
    static constexpr const char *c_defaultProcessMethod = "legacy";

    /// The default `1/f^a` exponent.
    static constexpr realT c_defaultPowerLawIndex = static_cast<realT>( 8.0 / 3.0 );

    /// The default auto-selected power-law normalization frequency.
    static constexpr realT c_defaultPowerLawNormFreq = static_cast<realT>( 0 );

    /// The default disabled explicit power-law match frequency.
    static constexpr realT c_defaultPowerLawMatchFreq = static_cast<realT>( 0 );

    /// The default domain used for noise-floor estimation.
    static constexpr const char *c_defaultNoiseEstimateDomain = "open-loop";

    /// The default end of the PSD used for noise-floor estimation.
    static constexpr const char *c_defaultNoiseEstimateRange = "high-freq";

    /// The default disabled maximum frequency for low-frequency noise estimation.
    static constexpr realT c_defaultNoiseEstimateLowFreqMaxHz = static_cast<realT>( 0 );

    /// The default statistic used to estimate the flat noise floor.
    static constexpr const char *c_defaultNoiseEstimateStatistic = "percentile";

    /// The default closed-loop to open-loop PSD reconstruction method.
    static constexpr const char *c_defaultClosedLoopOlEstimateMethod = "etf-only";

    /// The default half-width of the local match-frequency fallback window.
    static constexpr realT c_defaultPowerLawMatchFallbackWindowHz = static_cast<realT>( 5 );

    /// The default crossover-selection mode for the power-law handoff.
    static constexpr const char *c_defaultPowerLawCrossoverMode = "manual";

    /// The default median-smoothing width used by the automatic crossover finder.
    static constexpr realT c_defaultPowerLawAutoSmoothWidthHz = static_cast<realT>( 50 );

    /// The default fraction of the maximum sampled frequency searched by the
    /// automatic crossover finder. Set to 0 to disable the cap.
    static constexpr realT c_defaultPowerLawAutoMaxFreqFraction = static_cast<realT>( 0.4 );

    /// The default choice to keep the power-law exponent fixed.
    static constexpr bool c_defaultFitPowerLawIndex = false;

    /// The default choice to include the match point in the exponent fit.
    static constexpr bool c_defaultPowerLawFitIncludesMatchPoint = true;

    /// The default low edge of the power-law exponent fit range.
    static constexpr realT c_defaultPowerLawFitMinFreqHz = static_cast<realT>( 100 );

    /// The default high edge of the power-law exponent fit range.
    static constexpr realT c_defaultPowerLawFitMaxFreqHz = static_cast<realT>( 1000 );

    /// The default width of each exponent-fit median bin.
    static constexpr realT c_defaultPowerLawFitBinWidthHz = static_cast<realT>( 100 );

    /// The default number of bins used to blend between the raw PSD and the
    /// extrapolated continuum.
    static constexpr int c_defaultPowerLawBlendBins = 5;

    /// The default wide smoothing width used by peak detection.
    static constexpr realT c_defaultPeakDetectWidthHz = static_cast<realT>( 50 );

    /// The default strong-peak detection factor.
    static constexpr realT c_defaultPeakDetectFactor = static_cast<realT>( 5 );

    /// The default broad-peak detection factor.
    static constexpr realT c_defaultPeakDetectBroadFactor = static_cast<realT>( 3 );

    /// The default minimum accepted broad-peak width in dex.
    static constexpr realT c_defaultPeakDetectMinWidthLog = static_cast<realT>( 0.02 );

    /// The default lower bound on the Moffat beta parameter.
    static constexpr realT c_defaultPeakMoffatBeta = static_cast<realT>( 6 );

    /// The default number of subtract-and-redetect peak passes.
    static constexpr int c_defaultPeakDetectPasses = 2;

    /// The default dropout-detection threshold factor.
    static constexpr realT c_defaultDropoutGapFactor = static_cast<realT>( 0.2 );

    /// The default factor below the local good-bin scale that a candidate run
    /// must reach to be considered a true dropout.
    static constexpr realT c_defaultDropoutTinyFactor = static_cast<realT>( 1e-6 );

    /// The default maximum repaired dropout-run length in bins.
    static constexpr size_t c_defaultDropoutMaxBins = 4;

    /// The default raw-CL significance threshold multiplier.
    static constexpr realT c_defaultClSignificanceThreshold = static_cast<realT>( 1.1 );

    /// The default minimum fraction of significant raw-CL bins required to keep
    /// processing a mode.
    static constexpr realT c_defaultClMinSignificantFraction = static_cast<realT>( 0.05 );

    /// The default LP-continuum smoothing width.
    static constexpr realT c_defaultLpContinuumWidthHz = static_cast<realT>( 25 );

    /// The default disabled cutoff above which the extrapolation is pure power
    /// law only.
    static constexpr realT c_defaultPowerLawOnlyAboveFreq = static_cast<realT>( 0 );

    /// Configuration of the disturbance-PSD extrapolation model.
    struct processModelConfig
    {
        /// The extrapolation method name.
        std::string m_method{ c_defaultProcessMethod };

        /// The power-law exponent \f$a\f$ in \f$1/f^a\f$.
        realT m_powerLawIndex{ c_defaultPowerLawIndex };

        /// The normalization frequency for the power-law continuum.
        realT m_powerLawNormFreq{ c_defaultPowerLawNormFreq };

        /// The frequency where the power law is forced to match the raw disturbance
        /// PSD.
        realT m_powerLawMatchFreq{ c_defaultPowerLawMatchFreq };

        /// The domain used to estimate the flat noise floor.
        std::string m_noiseEstimateDomain{ c_defaultNoiseEstimateDomain };

        /// Which end of the PSD is used to estimate the flat noise floor.
        std::string m_noiseEstimateRange{ c_defaultNoiseEstimateRange };

        /// The maximum frequency in Hz used by the low-frequency noise estimate, or
        /// 0 to disable.
        realT m_noiseEstimateLowFreqMaxHz{ c_defaultNoiseEstimateLowFreqMaxHz };

        /// Which statistic is used to estimate the flat noise floor from the
        /// selected bins.
        std::string m_noiseEstimateStatistic{ c_defaultNoiseEstimateStatistic };

        /// How to reconstruct the OL PSD from a CL PSD when estimating noise in CL
        /// space.
        std::string m_closedLoopOlEstimateMethod{ c_defaultClosedLoopOlEstimateMethod };

        /// The half-width of the local fallback window used when the match point is
        /// in a trough.
        realT m_powerLawMatchFallbackWindowHz{ c_defaultPowerLawMatchFallbackWindowHz };

        /// How the power-law match/cutoff frequencies are chosen.
        std::string m_powerLawCrossoverMode{ c_defaultPowerLawCrossoverMode };

        /// The median-smoothing width used when automatically locating the
        /// crossover.
        realT m_powerLawAutoSmoothWidthHz{ c_defaultPowerLawAutoSmoothWidthHz };

        /// The fraction of the maximum sampled frequency searched when
        /// automatically locating the crossover. Set to 0 to disable the cap.
        realT m_powerLawAutoMaxFreqFraction{ c_defaultPowerLawAutoMaxFreqFraction };

        /// Whether to fit the power-law exponent from high-frequency bins.
        bool m_fitPowerLawIndex{ c_defaultFitPowerLawIndex };

        /// Above this frequency, force the extrapolation to be power-law only.
        realT m_powerLawOnlyAboveFreq{ c_defaultPowerLawOnlyAboveFreq };

        /// Whether the match point is included directly in the exponent fit.
        bool m_powerLawFitIncludesMatchPoint{ c_defaultPowerLawFitIncludesMatchPoint };

        /// The low edge of the exponent-fit range.
        realT m_powerLawFitMinFreqHz{ c_defaultPowerLawFitMinFreqHz };

        /// The high edge of the exponent-fit range.
        realT m_powerLawFitMaxFreqHz{ c_defaultPowerLawFitMaxFreqHz };

        /// The width of the exponent-fit median bins.
        realT m_powerLawFitBinWidthHz{ c_defaultPowerLawFitBinWidthHz };

        /// The number of bins used to blend between the measured PSD and the
        /// extrapolated continuum.
        int m_powerLawBlendBins{ c_defaultPowerLawBlendBins };

        /// The wide smoothing width used for peak detection.
        realT m_peakDetectWidthHz{ c_defaultPeakDetectWidthHz };

        /// The minimum factor above the smoothed PSD for a strong peak.
        realT m_peakDetectFactor{ c_defaultPeakDetectFactor };

        /// The lower factor used for broad-peak candidates.
        realT m_peakDetectBroadFactor{ c_defaultPeakDetectBroadFactor };

        /// The minimum accepted broad-peak width in log-frequency.
        realT m_peakDetectMinWidthLog{ c_defaultPeakDetectMinWidthLog };

        /// The number of iterative peak-detection passes.
        int m_peakDetectPasses{ c_defaultPeakDetectPasses };

        /// The minimum Moffat beta used for synthesized peaks.
        realT m_peakMoffatBeta{ c_defaultPeakMoffatBeta };

        /// The threshold used to identify dropout bins.
        realT m_dropoutGapFactor{ c_defaultDropoutGapFactor };

        /// The factor below the local good-bin scale that a candidate run must
        /// reach to be considered a true dropout.
        realT m_dropoutTinyFactor{ c_defaultDropoutTinyFactor };

        /// The maximum dropout-run length repaired by the gap-filling logic.
        size_t m_dropoutMaxBins{ c_defaultDropoutMaxBins };

        /// The multiplier above the fitted raw-CL noise floor required for a bin
        /// to be considered significant.
        realT m_clSignificanceThreshold{ c_defaultClSignificanceThreshold };

        /// The minimum fraction of raw-CL bins that must be significant for a
        /// mode to remain active.
        realT m_clMinSignificantFraction{ c_defaultClMinSignificantFraction };
    };

    /// Description of one detected spectral peak.
    struct identifiedPeak1D
    {
        /// The first PSD bin in the detected peak region.
        size_t m_start{ 0 };

        /// The last PSD bin in the detected peak region.
        size_t m_end{ 0 };

        /// The PSD bin containing the detected peak maximum.
        size_t m_peakIndex{ 0 };

        /// The detected peak center frequency in Hz.
        realT m_centerFreq{ 0 };

        /// The peak height above the continuum PSD.
        realT m_peakHeight{ 0 };

        /// The peak full-width at half maximum in Hz.
        realT m_fwhm{ 0 };
    };

    /// Results of modal PSD noise estimation and disturbance extrapolation.
    struct processResults
    {
        /// The extrapolation method used for the disturbance PSD.
        std::string m_processMethod{ c_defaultProcessMethod };

        /// The fitted flat noise floor.
        realT m_noiseFloor{ 0 };

        /// The continuum normalization at `m_powerLawNormFreq`.
        realT m_extrapolation{ 0 };

        /// The power-law exponent used in extrapolation.
        realT m_powerLawIndex{ c_defaultPowerLawIndex };

        /// The resolved normalization frequency of the power-law model.
        realT m_powerLawNormFreq{ 0 };

        /// The frequency where the power law is forced to match the disturbance
        /// PSD.
        realT m_powerLawMatchFreq{ 0 };

        /// The domain used to estimate the flat noise floor.
        std::string m_noiseEstimateDomain{ c_defaultNoiseEstimateDomain };

        /// Which end of the PSD was used to estimate the flat noise floor.
        std::string m_noiseEstimateRange{ c_defaultNoiseEstimateRange };

        /// The maximum frequency in Hz used by the low-frequency noise estimate, or
        /// 0 if disabled.
        realT m_noiseEstimateLowFreqMaxHz{ c_defaultNoiseEstimateLowFreqMaxHz };

        /// Which statistic was used to estimate the flat noise floor from the
        /// selected bins.
        std::string m_noiseEstimateStatistic{ c_defaultNoiseEstimateStatistic };

        /// Which CL-to-OL reconstruction method was used.
        std::string m_closedLoopOlEstimateMethod{ c_defaultClosedLoopOlEstimateMethod };

        /// The half-width of the local match-frequency fallback window.
        realT m_powerLawMatchFallbackWindowHz{ c_defaultPowerLawMatchFallbackWindowHz };

        /// How the power-law match/cutoff frequencies were chosen.
        std::string m_powerLawCrossoverMode{ c_defaultPowerLawCrossoverMode };

        /// The median-smoothing width used when automatically locating the
        /// crossover.
        realT m_powerLawAutoSmoothWidthHz{ c_defaultPowerLawAutoSmoothWidthHz };

        /// The fraction of the maximum sampled frequency searched when
        /// automatically locating the crossover. Set to 0 to disable the cap.
        realT m_powerLawAutoMaxFreqFraction{ c_defaultPowerLawAutoMaxFreqFraction };

        /// Whether the exponent was requested to be fit from the PSD.
        bool m_fitPowerLawIndex{ c_defaultFitPowerLawIndex };

        /// Above this frequency, force the extrapolation to be power-law only.
        realT m_powerLawOnlyAboveFreq{ c_defaultPowerLawOnlyAboveFreq };

        /// Whether the exponent fit succeeded and was applied.
        bool m_powerLawIndexFitSucceeded{ false };

        /// Whether the exponent fit included the explicit match point.
        bool m_powerLawFitIncludesMatchPoint{ c_defaultPowerLawFitIncludesMatchPoint };

        /// The low edge of the exponent-fit range.
        realT m_powerLawFitMinFreqHz{ c_defaultPowerLawFitMinFreqHz };

        /// The high edge of the exponent-fit range.
        realT m_powerLawFitMaxFreqHz{ c_defaultPowerLawFitMaxFreqHz };

        /// The width of the exponent-fit median bins.
        realT m_powerLawFitBinWidthHz{ c_defaultPowerLawFitBinWidthHz };

        /// The number of populated median bins used in the exponent fit.
        size_t m_powerLawFitBinsUsed{ 0 };

        /// The last frequency bin used to anchor the continuum.
        size_t m_powerLawAnchorIndex{ 0 };

        /// The frequency where the continuum takes over.
        realT m_powerLawAnchorFreq{ 0 };

        /// The blend width used at the power-law anchor.
        int m_powerLawBlendBins{ c_defaultPowerLawBlendBins };

        /// The peak-detection smoothing width.
        realT m_peakDetectWidthHz{ c_defaultPeakDetectWidthHz };

        /// The strong peak-detection factor threshold.
        realT m_peakDetectFactor{ c_defaultPeakDetectFactor };

        /// The broad peak-detection factor threshold.
        realT m_peakDetectBroadFactor{ c_defaultPeakDetectBroadFactor };

        /// The minimum accepted broad-peak width in dex.
        realT m_peakDetectMinWidthLog{ c_defaultPeakDetectMinWidthLog };

        /// The number of iterative peak-detection passes.
        int m_peakDetectPasses{ c_defaultPeakDetectPasses };

        /// The Moffat beta used for synthesized peaks.
        realT m_peakMoffatBeta{ c_defaultPeakMoffatBeta };

        /// The threshold used to identify dropout bins.
        realT m_dropoutGapFactor{ c_defaultDropoutGapFactor };

        /// The factor below the local good-bin scale that a candidate run must
        /// reach to be considered a true dropout.
        realT m_dropoutTinyFactor{ c_defaultDropoutTinyFactor };

        /// The maximum repaired dropout-run length in bins.
        size_t m_dropoutMaxBins{ c_defaultDropoutMaxBins };

        /// The LP-only continuum cutoff frequency in Hz.
        realT m_lpContinuumFreq{ 0 };

        /// The LP-only continuum smoothing width in Hz.
        realT m_lpContinuumWidthHz{ c_defaultLpContinuumWidthHz };

        /// The flat noise PSD estimate.
        std::vector<realT> m_noisePsd;

        /// The disturbance PSD used for optimization.
        std::vector<realT> m_processPsd;

        /// The disturbance PSD passed to the LP optimizer.
        std::vector<realT> m_lpProcessPsd;

        /// The unsmoothed, unextrapolated OL disturbance PSD.
        std::vector<realT> m_rawProcessPsd;

        /// The smoothed but still unextrapolated OL disturbance PSD.
        std::vector<realT> m_smoothedProcessPsd;

        /// The peaks detected by the Moffat extrapolator.
        std::vector<identifiedPeak1D> m_peaks;
    };

    /// Build the noise PSD, disturbance PSD, and LP continuum PSD for one mode.
    static mx::error_t
    analyzePsd( processResults &result,                          /**< [out] the populated process-model results */
                const std::vector<realT> &measuredPsd,           /**< [in] the measured one-sided PSD */
                const std::vector<realT> &freq,                  /**< [in] the one-sided frequency grid */
                size_t modeIndex,                                /**< [in] the zero-based mode index */
                const processModelConfig &config,                /**< [in] the disturbance-PSD configuration */
                realT lpContinuumFreq = static_cast<realT>( 0 ), /**< [in] the LP continuum cutoff */
                realT lpContinuumWidthHz = c_defaultLpContinuumWidthHz, /**< [in] LP smoothing width */
                const std::vector<realT> *etfPsd = nullptr,             /**< [in] optional CL ETF^2 correction */
                const std::vector<realT> *ntfPsd = nullptr              /**< [in] optional CL NTF^2 correction */
    );

    /// Estimate the flat noise PSD using the configured modalGainOpt statistic.
    static mx::error_t estimateNoisePsd(
        std::vector<realT> &noisePsd,                                         /**< [out] the flat noise PSD estimate */
        realT &noiseFloor,                                                    /**< [out] the fitted noise floor */
        const std::vector<realT> &measuredPsd,                                /**< [in] the measured one-sided PSD */
        const std::vector<realT> &freq,                                       /**< [in] the one-sided frequency grid */
        size_t modeIndex,                                                     /**< [in] the zero-based mode index */
        std::string noiseEstimateRange = c_defaultNoiseEstimateRange,         /**< [in] which PSD end to use */
        std::string noiseEstimateStatistic = c_defaultNoiseEstimateStatistic, /**< [in] how to summarize the
                                                                                 selected bins */
        realT noiseEstimateLowFreqMaxHz = c_defaultNoiseEstimateLowFreqMaxHz  /**< [in] optional low-frequency
                                                                                 upper limit */
    );

    /// Replace all LP content above a cutoff with a smoothed continuum.
    static mx::error_t applyLpContinuum( std::vector<realT> &lpProcessPsd,     /**< [out] the LP disturbance PSD */
                                         const std::vector<realT> &processPsd, /**< [in] the nominal disturbance PSD */
                                         const std::vector<realT> &freq,       /**< [in] the one-sided frequency grid */
                                         realT cutoffFreq,      /**< [in] the continuum cutoff frequency */
                                         realT continuumWidthHz /**< [in] the smoothing width */
    );

  protected:
    /// Return the index of the first strictly-positive frequency bin.
    static size_t firstPositiveFreqIndex( const std::vector<realT> &freq /**< [in] the one-sided frequency grid */ );

    /// Normalize a noise-estimation-domain name to lowercase hyphenated form.
    static std::string normalizeNoiseEstimateDomain( std::string domain /**< [in] the requested domain name */ );

    /// Normalize a noise-estimation-range name to lowercase hyphenated form.
    static std::string normalizeNoiseEstimateRange( std::string range /**< [in] the requested PSD-end selector */ );

    /// Normalize a noise-estimation-statistic name to lowercase hyphenated form.
    static std::string
    normalizeNoiseEstimateStatistic( std::string statistic /**< [in] the requested noise-fit statistic */ );

    /// Normalize a CL-to-OL PSD reconstruction method name to lowercase
    /// hyphenated form.
    static std::string
    normalizeClosedLoopOlEstimateMethod( std::string method /**< [in] the requested CL-to-OL method */ );

    /// Normalize a power-law crossover mode name to lowercase hyphenated form.
    static std::string normalizePowerLawCrossoverMode( std::string mode /**< [in] the requested crossover mode */ );

    /// Resolve the power-law normalization frequency, defaulting to the first
    /// positive bin.
    static realT resolvePowerLawNormFreq( const std::vector<realT> &freq, /**< [in] the one-sided frequency grid */
                                          realT requestedNormFreq /**< [in] the requested normalization frequency */
    );

    /// Median-smooth a disturbance PSD in log space while preserving the original
    /// sampling.
    static mx::error_t
    buildSmoothedProcessPsd( std::vector<realT> &smoothedProcessPsd,  /**< [out] the smoothed disturbance PSD */
                             const std::vector<realT> &rawProcessPsd, /**< [in] the raw disturbance PSD */
                             const std::vector<realT> &freq,          /**< [in] the one-sided frequency grid */
                             realT smoothWidthHz                      /**< [in] the median-smoothing width in Hz */
    );

    /// Determine the automatic power-law crossover from a median-smoothed
    /// disturbance PSD.
    static mx::error_t findAutoPowerLawCrossoverFreq(
        realT &crossoverFreq,                         /**< [out] the resolved crossover frequency */
        const std::vector<realT> &smoothedProcessPsd, /**< [in] the smoothed disturbance PSD */
        const std::vector<realT> &noisePsd,           /**< [in] the flat noise PSD */
        const std::vector<realT> &freq,               /**< [in] the one-sided frequency grid */
        realT maxFreqFraction                         /**< [in] the maximum searched frequency as a fraction of the
                                                         sampled maximum */
    );

    /// Resolve effective power-law match and cutoff frequencies for manual or
    /// automatic crossover modes.
    static mx::error_t resolvePowerLawCrossoverFrequencies(
        realT &powerLawMatchFreq,                     /**< [in.out] match frequency */
        realT &powerLawOnlyAboveFreq,                 /**< [in.out] cutoff frequency */
        const std::vector<realT> &rawProcessPsd,      /**< [in] raw OL disturbance PSD */
        const std::vector<realT> &smoothedProcessPsd, /**< [in] smoothed OL disturbance PSD */
        const std::vector<realT> &noisePsd,           /**< [in] OL noise PSD */
        const std::vector<realT> &freq,               /**< [in] frequency grid */
        std::string powerLawCrossoverMode,            /**< [in] mode */
        realT powerLawAutoMaxFreqFraction             /**< [in] the maximum searched frequency as a fraction of the
                                                         sampled maximum */
    );

    /// Evaluate the extrapolated power-law continuum at one frequency bin.
    static realT powerLawContinuum( realT extrapolation,            /**< [in] the continuum normalization */
                                    const std::vector<realT> &freq, /**< [in] the one-sided frequency grid */
                                    size_t index,                   /**< [in] the bin index to evaluate */
                                    realT powerLawIndex,            /**< [in] the power-law exponent */
                                    realT powerLawNormFreq          /**< [in] the continuum normalization frequency */
    );

    /// Linearly interpolate a sampled PSD onto an arbitrary frequency.
    static realT interpolatePsdAtFreq( const std::vector<realT> &psd,  /**< [in] the sampled PSD values */
                                       const std::vector<realT> &freq, /**< [in] the sampled frequencies */
                                       realT targetFreq                /**< [in] the desired interpolation frequency */
    );

    /// Force the power-law normalization so the model matches the disturbance PSD
    /// at a chosen frequency.
    static mx::error_t
    matchPowerLawAtFreq( realT &extrapolation,                       /**< [in.out] the continuum normalization */
                         const std::vector<realT> &anchorProcessPsd, /**< [in] the PSD used for continuum anchoring */
                         const std::vector<realT> &freq,             /**< [in] the one-sided frequency grid */
                         realT powerLawIndex,                        /**< [in] the power-law exponent */
                         realT powerLawNormFreq,                     /**< [in] the continuum normalization frequency */
                         realT powerLawMatchFreq,                    /**< [in] the desired match frequency */
                         realT powerLawMatchFallbackWindowHz         /**< [in] the local fallback
                                                                        half-width */
    );

    /// Invert the Moffat FWHM relation to recover the alpha parameter.
    static realT moffatAlphaFromFwhm( realT fwhm, /**< [in] the desired full-width at half-maximum */
                                      realT beta  /**< [in] the Moffat beta parameter */
    );

    /// Evaluate a zero-background Moffat profile from its height, FWHM, and beta.
    static realT moffatValueFromFwhm( realT radius,     /**< [in] the distance from the peak center */
                                      realT peakHeight, /**< [in] the peak height above the continuum */
                                      realT fwhm,       /**< [in] the full-width at half maximum */
                                      realT beta        /**< [in] the Moffat beta parameter */
    );

    /// Find the smallest beta that drives a Moffat wing down to a target level at
    /// a target radius.
    static realT fitMoffatBetaToBoundary( realT peakHeight, /**< [in] the peak height above the continuum */
                                          realT fwhm,       /**< [in] the full-width at half maximum */
                                          realT radius,     /**< [in] the target boundary radius */
                                          realT target,     /**< [in] the desired level at that boundary */
                                          realT betaFloor   /**< [in] the minimum allowed beta */
    );

    /// Linearly interpolate the x-position where a profile crosses a target
    /// level.
    static realT interpolateCrossing( realT x0,    /**< [in] the first x-coordinate */
                                      realT y0,    /**< [in] the first y-coordinate */
                                      realT x1,    /**< [in] the second x-coordinate */
                                      realT y1,    /**< [in] the second y-coordinate */
                                      realT target /**< [in] the y-level to interpolate */
    );

    /// Fit the power-law exponent from binned median log PSD values over a
    /// selected frequency range.
    static mx::error_t fitPowerLawIndexFromBinnedMedians(
        realT &powerLawIndex,                              /**< [out] the fitted exponent */
        size_t &nBinsUsed,                                 /**< [out] the number of populated bins */
        const std::vector<realT> &rawProcessPsd,           /**< [in] disturbance PSD */
        const std::vector<realT> &freq,                    /**< [in] the one-sided frequency grid */
        realT fitMinFreqHz,                                /**< [in] the low edge of the fit range */
        realT fitMaxFreqHz,                                /**< [in] the high edge of the fit range */
        realT fitBinWidthHz,                               /**< [in] the width of the median bins */
        realT includeMatchFreqHz = static_cast<realT>( 0 ) /**< [in] optional explicit match point */
    );

    /// Estimate the `1/f^a` continuum used by the power-law and Moffat
    /// extrapolators.
    static mx::error_t estimatePowerLawContinuum(
        std::vector<realT> &continuumPsd,           /**< [out] the continuum PSD */
        realT &extrapolation,                       /**< [out] the continuum normalization */
        size_t &anchorIndex,                        /**< [out] the last bin used to anchor the fit */
        const std::vector<realT> &rawProcessPsd,    /**< [in] raw disturbance PSD */
        const std::vector<realT> &anchorProcessPsd, /**< [in] smoothed disturbance PSD used for anchoring */
        const std::vector<realT> &noisePsd,         /**< [in] the flat noise PSD */
        const std::vector<realT> &freq,             /**< [in] the one-sided frequency grid */
        realT powerLawIndex,                        /**< [in] the power-law exponent */
        realT powerLawNormFreq,                     /**< [in] the normalization frequency */
        realT powerLawMatchFreq,                    /**< [in] the optional match frequency */
        realT powerLawMatchFallbackWindowHz,        /**< [in] the match fallback
                                                       half-width */
        bool fitPowerLawIndex = false,              /**< [in] whether to fit the exponent */
        realT powerLawFitMinFreqHz = c_defaultPowerLawFitMinFreqHz,                  /**< [in] fit low edge */
        realT powerLawFitMaxFreqHz = c_defaultPowerLawFitMaxFreqHz,                  /**< [in] fit high edge */
        realT powerLawFitBinWidthHz = c_defaultPowerLawFitBinWidthHz,                /**< [in] fit bin width */
        bool powerLawFitIncludesMatchPoint = c_defaultPowerLawFitIncludesMatchPoint, /**< [in] include match point
                                                                                      */
        realT *usedPowerLawIndex = nullptr, /**< [out] the exponent actually used */
        size_t *fitBinsUsed = nullptr       /**< [out] the number of populated fit bins */
    );

    /// Populate a sampled power-law continuum from a resolved normalization.
    static mx::error_t buildPowerLawContinuum( std::vector<realT> &continuumPsd, /**< [out] the sampled continuum PSD */
                                               realT extrapolation,            /**< [in] the continuum normalization */
                                               const std::vector<realT> &freq, /**< [in] the one-sided frequency grid */
                                               realT powerLawIndex,            /**< [in] the power-law exponent */
                                               realT powerLawNormFreq          /**< [in] the normalization frequency */
    );

    /// Blend from a continuum model to the measured disturbance PSD at the anchor
    /// point.
    static mx::error_t
    blendContinuumAtAnchor( std::vector<realT> &processPsd,          /**< [out] the blended disturbance PSD */
                            const std::vector<realT> &rawProcessPsd, /**< [in] measured disturbance PSD */
                            const std::vector<realT> &continuumPsd,  /**< [in] extrapolated continuum PSD */
                            size_t anchorIndex, /**< [in] the last bin where the continuum anchors */
                            int blendBins       /**< [in] the number of handoff bins */
    );

    /// Identify peaks using strong-threshold or broad-but-wide discriminants
    /// above a smoothed PSD.
    static mx::error_t
    identifyMoffatPeaks( std::vector<identifiedPeak1D> &peaks,    /**< [out] the detected peaks */
                         const std::vector<realT> &rawProcessPsd, /**< [in] the raw disturbance PSD */
                         const std::vector<realT> &continuumPsd,  /**< [in] the continuum PSD model */
                         const std::vector<realT> &freq,          /**< [in] the one-sided frequency grid */
                         realT peakDetectWidthHz,                 /**< [in] the wide smoothing width */
                         realT peakDetectFactor,                  /**< [in] the factor above smooth for strong peaks
                                                                   */
                         realT peakDetectBroadFactor,             /**< [in] the lower factor for broad
                                                                     candidates */
                         realT peakDetectMinWidthLog              /**< [in] the minimum accepted broad-peak
                                                                     width */
    );

    /// Add one clipped Moffat peak excess to a PSD model.
    static mx::error_t
    addClippedMoffatPeakExcess( std::vector<realT> &peakModel,       /**< [in.out] the accumulated peak model */
                                const identifiedPeak1D &peak,        /**< [in] the detected peak */
                                const std::vector<realT> &sourcePsd, /**< [in] the source PSD used to bound the peak */
                                const std::vector<realT> &continuumPsd, /**< [in] the continuum PSD */
                                const std::vector<realT> &freq,         /**< [in] the one-sided frequency grid */
                                realT peakMoffatBeta                    /**< [in] the minimum Moffat beta */
    );

    /// Identify peaks iteratively, subtracting each pass's modeled peaks from the
    /// residual.
    static mx::error_t
    identifyMoffatPeaksMultiPass( std::vector<identifiedPeak1D> &peaks,    /**< [out] the detected peaks */
                                  const std::vector<realT> &rawProcessPsd, /**< [in] the raw disturbance PSD */
                                  const std::vector<realT> &continuumPsd,  /**< [in] the continuum PSD model */
                                  const std::vector<realT> &freq,          /**< [in] the one-sided frequency grid */
                                  realT peakDetectWidthHz,                 /**< [in] the wide smoothing width */
                                  realT peakDetectFactor,                  /**< [in] the factor above smooth */
                                  realT peakDetectBroadFactor,             /**< [in] the broad-peak factor */
                                  realT peakDetectMinWidthLog,             /**< [in] the minimum log-width */
                                  int peakDetectPasses,                    /**< [in] the number of iterative passes */
                                  realT peakMoffatBeta                     /**< [in] the minimum Moffat beta */
    );

    /// Build the full Moffat-peak disturbance model from a fixed continuum.
    static mx::error_t
    buildMoffatProcessFromContinuum( std::vector<realT> &processPsd,          /**< [out] the disturbance PSD */
                                     std::vector<identifiedPeak1D> &peaks,    /**< [out] detected peaks */
                                     std::vector<unsigned char> &repairMask,  /**< [out] repair-eligible bins */
                                     const std::vector<realT> &rawProcessPsd, /**< [in] raw disturbance PSD */
                                     const std::vector<realT> &noisePsd,      /**< [in] the flat noise PSD */
                                     const std::vector<realT> &continuumPsd,  /**< [in] the continuum PSD */
                                     const std::vector<realT> &freq,          /**< [in] the one-sided frequency grid */
                                     size_t anchorIndex,                      /**< [in] the continuum handoff bin */
                                     const processModelConfig &config         /**< [in] the process configuration */
    );

    /// Build the power-law-only disturbance model from a fixed continuum.
    static mx::error_t
    buildPowerLawOnlyProcessFromContinuum( std::vector<realT> &processPsd,          /**< [out] the disturbance PSD */
                                           std::vector<unsigned char> &repairMask,  /**< [out] repair-eligible bins */
                                           const std::vector<realT> &rawProcessPsd, /**< [in] raw disturbance PSD */
                                           const std::vector<realT> &noisePsd,      /**< [in] the flat noise PSD */
                                           const std::vector<realT> &continuumPsd,  /**< [in] the continuum PSD */
                                           const std::vector<realT> &freq,  /**< [in] the one-sided frequency grid */
                                           size_t anchorIndex,              /**< [in] the continuum handoff bin */
                                           const processModelConfig &config /**< [in] the process configuration */
    );

    /// Build a disturbance PSD from only the extrapolated `1/f^a` continuum.
    static mx::error_t
    estimateProcessPsdPowerLawOnly( std::vector<realT> &processPsd, /**< [out] the disturbance PSD */
                                    realT &extrapolation,           /**< [out] the power-law continuum anchor */
                                    size_t &anchorIndex,            /**< [out] the last frequency bin used to anchor the
                                                                       fit */
                                    std::vector<unsigned char> &repairMask,     /**< [out] bins eligible for repair */
                                    const std::vector<realT> &measuredPsd,      /**< [in] the measured one-sided PSD */
                                    const std::vector<realT> &anchorProcessPsd, /**< [in] the smoothed disturbance PSD
                                                                                    used for anchoring */
                                    const std::vector<realT> &noisePsd,         /**< [in] the flat noise PSD */
                                    const std::vector<realT> &freq,     /**< [in] the one-sided frequency grid */
                                    const processModelConfig &config,   /**< [in] the disturbance-PSD configuration */
                                    realT *usedPowerLawIndex = nullptr, /**< [out] the exponent actually used */
                                    size_t *fitBinsUsed = nullptr       /**< [out] the number of populated fit bins */
    );

    /// Build a disturbance PSD by combining a `1/f^a` continuum with detected
    /// Moffat peaks.
    static mx::error_t
    estimateProcessPsdMoffatPeaks( std::vector<realT> &processPsd,         /**< [out] the disturbance PSD */
                                   realT &extrapolation,                   /**< [out] the power-law continuum anchor */
                                   std::vector<identifiedPeak1D> &peaks,   /**< [out] the detected peaks */
                                   std::vector<unsigned char> &repairMask, /**< [out] bins eligible for repair */
                                   const std::vector<realT> &measuredPsd,  /**< [in] the measured one-sided PSD */
                                   const std::vector<realT> &anchorProcessPsd, /**< [in] the smoothed disturbance PSD
                                                                                   used for anchoring */
                                   const std::vector<realT> &noisePsd,         /**< [in] the flat noise PSD */
                                   const std::vector<realT> &freq,             /**< [in] the one-sided frequency grid */
                                   const processModelConfig &config /**< [in] the extrapolation configuration */
    );

    /// Build the disturbance PSD used for optimization from the measured PSD and
    /// the flat noise estimate.
    static mx::error_t
    estimateProcessPsd( std::vector<realT> &processPsd,        /**< [out] the disturbance PSD */
                        realT &extrapolation,                  /**< [out] the low-frequency extrapolation anchor */
                        const std::vector<realT> &measuredPsd, /**< [in] the measured one-sided PSD */
                        const std::vector<realT> &noisePsd,    /**< [in] the flat noise PSD estimate */
                        const std::vector<realT> &freq,        /**< [in] the one-sided frequency grid */
                        realT powerLawNormFreq,                /**< [in] the power-law normalization frequency */
                        realT powerLawMatchFreq,               /**< [in] the optional power-law match frequency */
                        realT powerLawMatchFallbackWindowHz    /**< [in] the local match fallback
                                                                  half-width */
    );

    /// Fill isolated or short dropout runs in a disturbance PSD.
    static mx::error_t
    fillProcessPsdDropouts( std::vector<realT> &processPsd,               /**< [in.out] the disturbance PSD */
                            const std::vector<realT> &freq,               /**< [in] the one-sided frequency grid */
                            const std::vector<unsigned char> &repairMask, /**< [in] repair-eligible bins */
                            realT gapFactor,    /**< [in] the threshold used to identify dropout bins */
                            realT tinyFactor,   /**< [in] the factor below the local good-bin scale
                                                   required for a true dropout */
                            size_t maxGapBins,  /**< [in] the maximum repaired gap length */
                            realT powerLawIndex /**< [in] the power-law exponent used to continue a trailing gap */
    );
};

template <typename realT>
mx::error_t modalPsdProcessor<realT>::analyzePsd( processResults &result,
                                                  const std::vector<realT> &measuredPsd,
                                                  const std::vector<realT> &freq,
                                                  size_t modeIndex,
                                                  const processModelConfig &config,
                                                  realT lpContinuumFreq,
                                                  realT lpContinuumWidthHz,
                                                  const std::vector<realT> *etfPsd,
                                                  const std::vector<realT> *ntfPsd )
{
    if( measuredPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Measured PSD and frequency grid must be the same size" );
    }

    if( measuredPsd.size() < 2 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr, "At least two frequency bins are required" );
    }

    result.m_processMethod = config.m_method;
    result.m_powerLawIndex = config.m_powerLawIndex;
    result.m_powerLawNormFreq = resolvePowerLawNormFreq( freq, config.m_powerLawNormFreq );
    result.m_powerLawMatchFreq = config.m_powerLawMatchFreq;
    result.m_noiseEstimateDomain = normalizeNoiseEstimateDomain( config.m_noiseEstimateDomain );
    result.m_noiseEstimateRange = normalizeNoiseEstimateRange( config.m_noiseEstimateRange );
    result.m_noiseEstimateLowFreqMaxHz = config.m_noiseEstimateLowFreqMaxHz;
    result.m_noiseEstimateStatistic = normalizeNoiseEstimateStatistic( config.m_noiseEstimateStatistic );
    result.m_closedLoopOlEstimateMethod = normalizeClosedLoopOlEstimateMethod( config.m_closedLoopOlEstimateMethod );
    result.m_powerLawMatchFallbackWindowHz = config.m_powerLawMatchFallbackWindowHz;
    result.m_powerLawCrossoverMode = normalizePowerLawCrossoverMode( config.m_powerLawCrossoverMode );
    result.m_powerLawAutoSmoothWidthHz = config.m_powerLawAutoSmoothWidthHz;
    result.m_powerLawAutoMaxFreqFraction = config.m_powerLawAutoMaxFreqFraction;
    result.m_fitPowerLawIndex = config.m_fitPowerLawIndex;
    result.m_powerLawOnlyAboveFreq = config.m_powerLawOnlyAboveFreq;
    result.m_powerLawIndexFitSucceeded = false;
    result.m_powerLawFitIncludesMatchPoint = config.m_powerLawFitIncludesMatchPoint;
    result.m_powerLawFitMinFreqHz = config.m_powerLawFitMinFreqHz;
    result.m_powerLawFitMaxFreqHz = config.m_powerLawFitMaxFreqHz;
    result.m_powerLawFitBinWidthHz = config.m_powerLawFitBinWidthHz;
    result.m_powerLawFitBinsUsed = 0;
    result.m_powerLawBlendBins = config.m_powerLawBlendBins;
    result.m_peakDetectWidthHz = config.m_peakDetectWidthHz;
    result.m_peakDetectFactor = config.m_peakDetectFactor;
    result.m_peakDetectBroadFactor = config.m_peakDetectBroadFactor;
    result.m_peakDetectMinWidthLog = config.m_peakDetectMinWidthLog;
    result.m_peakDetectPasses = config.m_peakDetectPasses;
    result.m_peakMoffatBeta = config.m_peakMoffatBeta;
    result.m_dropoutGapFactor = config.m_dropoutGapFactor;
    result.m_dropoutTinyFactor = config.m_dropoutTinyFactor;
    result.m_dropoutMaxBins = config.m_dropoutMaxBins;
    result.m_lpContinuumFreq = lpContinuumFreq;
    result.m_lpContinuumWidthHz = lpContinuumWidthHz;

    if( config.m_dropoutGapFactor <= static_cast<realT>( 0 ) || config.m_dropoutGapFactor >= static_cast<realT>( 1 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Dropout gap factor must be between 0 and 1" );
    }

    if( config.m_dropoutTinyFactor <= static_cast<realT>( 0 ) || config.m_dropoutTinyFactor >= static_cast<realT>( 1 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Dropout tiny factor must be between 0 and 1" );
    }

    if( config.m_powerLawOnlyAboveFreq < static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Power-law-only-above frequency must be non-negative" );
    }

    if( config.m_powerLawMatchFallbackWindowHz < static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Power-law match fallback window must be non-negative" );
    }

    if( config.m_powerLawBlendBins < 0 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg, "Power-law blend bins must be non-negative" );
    }

    if( config.m_dropoutMaxBins < 1 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg, "Dropout max bins must be at least one" );
    }

    if( config.m_fitPowerLawIndex && ( config.m_powerLawFitMinFreqHz <= static_cast<realT>( 0 ) ||
                                       config.m_powerLawFitMaxFreqHz <= config.m_powerLawFitMinFreqHz ||
                                       config.m_powerLawFitBinWidthHz <= static_cast<realT>( 0 ) ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Invalid power-law exponent fit range or bin width" );
    }

    if( result.m_noiseEstimateDomain != "open-loop" && result.m_noiseEstimateDomain != "closed-loop-pre-xfer" )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Unknown noise-estimation domain: " + result.m_noiseEstimateDomain );
    }

    if( result.m_noiseEstimateRange != "high-freq" && result.m_noiseEstimateRange != "low-freq" )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Unknown noise-estimation range: " + result.m_noiseEstimateRange );
    }

    if( result.m_noiseEstimateStatistic != "percentile" && result.m_noiseEstimateStatistic != "minimum" )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Unknown noise-estimation statistic: " +
                                                     result.m_noiseEstimateStatistic );
    }

    if( result.m_noiseEstimateLowFreqMaxHz < static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Low-frequency noise-estimate maximum must be non-negative" );
    }

    if( result.m_closedLoopOlEstimateMethod != "etf-only" && result.m_closedLoopOlEstimateMethod != "ntf-aware" )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Unknown closed-loop OL estimate method: " +
                                                     result.m_closedLoopOlEstimateMethod );
    }

    if( result.m_powerLawCrossoverMode != "manual" && result.m_powerLawCrossoverMode != "auto-smoothed-crossing" )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Unknown power-law crossover mode: " +
                                                     result.m_powerLawCrossoverMode );
    }

    if( result.m_powerLawCrossoverMode == "auto-smoothed-crossing" &&
        result.m_powerLawAutoSmoothWidthHz <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Automatic power-law crossover smoothing width must be positive" );
    }

    if( result.m_powerLawAutoMaxFreqFraction < static_cast<realT>( 0 ) ||
        result.m_powerLawAutoMaxFreqFraction > static_cast<realT>( 1 ) )
    {
        return mx::error_report<mx::verbose::d>(
            mx::error_t::invalidarg,
            "Automatic power-law crossover maximum-frequency fraction must be between 0 and 1" );
    }

    const std::vector<realT> &noiseEstimatePsd = measuredPsd;

    mx::error_t errc = estimateNoisePsd( result.m_noisePsd,
                                         result.m_noiseFloor,
                                         noiseEstimatePsd,
                                         freq,
                                         modeIndex,
                                         result.m_noiseEstimateRange,
                                         result.m_noiseEstimateStatistic,
                                         result.m_noiseEstimateLowFreqMaxHz );
    if( !!errc )
    {
        return errc;
    }

    std::vector<realT> processMeasuredPsd = measuredPsd;
    std::vector<realT> processNoisePsd = result.m_noisePsd;
    if( result.m_noiseEstimateDomain == "closed-loop-pre-xfer" )
    {
        if( etfPsd == nullptr )
        {
            return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                     "Closed-loop noise estimation requires an ETF PSD" );
        }

        if( etfPsd->size() != measuredPsd.size() )
        {
            return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr, "ETF PSD must match the measured PSD size" );
        }

        if( result.m_closedLoopOlEstimateMethod == "ntf-aware" )
        {
            if( ntfPsd == nullptr )
            {
                return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                         "NTF-aware closed-loop noise estimation requires an NTF PSD" );
            }

            if( ntfPsd->size() != measuredPsd.size() )
            {
                return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                         "NTF PSD must match the measured PSD size" );
            }
        }

        const realT tiny = std::numeric_limits<realT>::min();
        for( size_t n = 0; n < measuredPsd.size(); ++n )
        {
            const realT useEtf = std::max( ( *etfPsd )[n], tiny );
            realT closedLoopNoise = result.m_noisePsd[n];
            if( result.m_closedLoopOlEstimateMethod == "ntf-aware" )
            {
                closedLoopNoise = result.m_noisePsd[n] * std::max( ( *ntfPsd )[n], tiny );
            }

            const realT rawClosedLoop = std::max( measuredPsd[n] - closedLoopNoise, tiny );
            const realT openLoopProcess = rawClosedLoop / useEtf;

            // Keep the comparison noise PSD in the same CL-noise domain as the
            // fitted/published noise floor. The OL disturbance estimate is still
            // formed from (CL - noise) / ETF above.
            processNoisePsd[n] = std::max( closedLoopNoise, tiny );
            processMeasuredPsd[n] = openLoopProcess + processNoisePsd[n];
        }
    }

    processModelConfig effectiveConfig = config;
    effectiveConfig.m_method = result.m_processMethod;
    effectiveConfig.m_noiseEstimateDomain = result.m_noiseEstimateDomain;
    effectiveConfig.m_noiseEstimateRange = result.m_noiseEstimateRange;
    effectiveConfig.m_noiseEstimateStatistic = result.m_noiseEstimateStatistic;
    effectiveConfig.m_closedLoopOlEstimateMethod = result.m_closedLoopOlEstimateMethod;
    effectiveConfig.m_powerLawCrossoverMode = result.m_powerLawCrossoverMode;
    effectiveConfig.m_powerLawAutoSmoothWidthHz = result.m_powerLawAutoSmoothWidthHz;
    effectiveConfig.m_powerLawAutoMaxFreqFraction = result.m_powerLawAutoMaxFreqFraction;

    const realT tiny = std::numeric_limits<realT>::min();
    result.m_rawProcessPsd.resize( processMeasuredPsd.size() );
    for( size_t n = 0; n < processMeasuredPsd.size(); ++n )
    {
        result.m_rawProcessPsd[n] = std::max( processMeasuredPsd[n] - processNoisePsd[n], tiny );
    }

    if( effectiveConfig.m_method == "power-law-only" || effectiveConfig.m_method == "moffat-peaks" )
    {
        errc = fillProcessPsdDropouts( result.m_rawProcessPsd,
                                       freq,
                                       {},
                                       effectiveConfig.m_dropoutGapFactor,
                                       effectiveConfig.m_dropoutTinyFactor,
                                       effectiveConfig.m_dropoutMaxBins,
                                       effectiveConfig.m_powerLawIndex );
        if( !!errc )
        {
            return errc;
        }

        for( size_t n = 0; n < processMeasuredPsd.size(); ++n )
        {
            processMeasuredPsd[n] = result.m_rawProcessPsd[n] + processNoisePsd[n];
        }
    }

    if( effectiveConfig.m_powerLawAutoSmoothWidthHz > static_cast<realT>( 0 ) )
    {
        errc = buildSmoothedProcessPsd( result.m_smoothedProcessPsd,
                                        result.m_rawProcessPsd,
                                        freq,
                                        effectiveConfig.m_powerLawAutoSmoothWidthHz );
        if( !!errc )
        {
            return errc;
        }
    }
    else
    {
        result.m_smoothedProcessPsd = result.m_rawProcessPsd;
    }

    errc = resolvePowerLawCrossoverFrequencies( effectiveConfig.m_powerLawMatchFreq,
                                                effectiveConfig.m_powerLawOnlyAboveFreq,
                                                result.m_rawProcessPsd,
                                                result.m_smoothedProcessPsd,
                                                processNoisePsd,
                                                freq,
                                                effectiveConfig.m_powerLawCrossoverMode,
                                                effectiveConfig.m_powerLawAutoMaxFreqFraction );
    if( !!errc )
    {
        return errc;
    }

    result.m_powerLawMatchFreq = effectiveConfig.m_powerLawMatchFreq;
    result.m_powerLawOnlyAboveFreq = effectiveConfig.m_powerLawOnlyAboveFreq;

    result.m_peaks.clear();
    std::vector<unsigned char> processRepairMask;

    if( effectiveConfig.m_method == "legacy" )
    {
        result.m_powerLawIndex = c_defaultPowerLawIndex;
        result.m_powerLawAnchorIndex = 0;
        result.m_powerLawAnchorFreq = 0;
        errc = estimateProcessPsd( result.m_processPsd,
                                   result.m_extrapolation,
                                   processMeasuredPsd,
                                   processNoisePsd,
                                   freq,
                                   effectiveConfig.m_powerLawNormFreq,
                                   effectiveConfig.m_powerLawMatchFreq,
                                   effectiveConfig.m_powerLawMatchFallbackWindowHz );
        if( !!errc )
        {
            return errc;
        }
    }
    else if( effectiveConfig.m_method == "power-law-only" )
    {
        realT usedPowerLawIndex = effectiveConfig.m_powerLawIndex;
        size_t fitBinsUsed = 0;
        errc = estimateProcessPsdPowerLawOnly( result.m_processPsd,
                                               result.m_extrapolation,
                                               result.m_powerLawAnchorIndex,
                                               processRepairMask,
                                               processMeasuredPsd,
                                               result.m_smoothedProcessPsd,
                                               processNoisePsd,
                                               freq,
                                               effectiveConfig,
                                               &usedPowerLawIndex,
                                               &fitBinsUsed );
        if( !!errc )
        {
            return errc;
        }

        result.m_powerLawAnchorFreq =
            result.m_powerLawAnchorIndex < freq.size() ? freq[result.m_powerLawAnchorIndex] : static_cast<realT>( 0 );
        result.m_powerLawIndex = usedPowerLawIndex;
        result.m_powerLawIndexFitSucceeded = effectiveConfig.m_fitPowerLawIndex && fitBinsUsed > 0;
        result.m_powerLawFitBinsUsed = fitBinsUsed;
    }
    else if( effectiveConfig.m_method == "moffat-peaks" )
    {
        size_t anchorIndex = 0;
        errc = estimateProcessPsdMoffatPeaks( result.m_processPsd,
                                              result.m_extrapolation,
                                              result.m_peaks,
                                              processRepairMask,
                                              processMeasuredPsd,
                                              result.m_smoothedProcessPsd,
                                              processNoisePsd,
                                              freq,
                                              effectiveConfig );
        if( !!errc )
        {
            return errc;
        }

        std::vector<realT> rawProcessPsd( processMeasuredPsd.size() );
        const realT tiny = std::numeric_limits<realT>::min();
        for( size_t n = 0; n < processMeasuredPsd.size(); ++n )
        {
            rawProcessPsd[n] = std::max( processMeasuredPsd[n] - processNoisePsd[n], tiny );
        }

        std::vector<realT> continuumPsd;
        realT usedPowerLawIndex = effectiveConfig.m_powerLawIndex;
        size_t fitBinsUsed = 0;
        errc = estimatePowerLawContinuum( continuumPsd,
                                          result.m_extrapolation,
                                          anchorIndex,
                                          rawProcessPsd,
                                          result.m_smoothedProcessPsd,
                                          processNoisePsd,
                                          freq,
                                          effectiveConfig.m_powerLawIndex,
                                          effectiveConfig.m_powerLawNormFreq,
                                          effectiveConfig.m_powerLawMatchFreq,
                                          effectiveConfig.m_powerLawMatchFallbackWindowHz,
                                          effectiveConfig.m_fitPowerLawIndex,
                                          effectiveConfig.m_powerLawFitMinFreqHz,
                                          effectiveConfig.m_powerLawFitMaxFreqHz,
                                          effectiveConfig.m_powerLawFitBinWidthHz,
                                          effectiveConfig.m_powerLawFitIncludesMatchPoint,
                                          &usedPowerLawIndex,
                                          &fitBinsUsed );
        if( !!errc )
        {
            return errc;
        }

        result.m_powerLawAnchorIndex = anchorIndex;
        result.m_powerLawAnchorFreq =
            result.m_powerLawAnchorIndex < freq.size() ? freq[result.m_powerLawAnchorIndex] : static_cast<realT>( 0 );
        result.m_powerLawIndex = usedPowerLawIndex;
        result.m_powerLawIndexFitSucceeded = effectiveConfig.m_fitPowerLawIndex && fitBinsUsed > 0;
        result.m_powerLawFitBinsUsed = fitBinsUsed;
    }
    else
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Unknown process method: " + effectiveConfig.m_method );
    }

    if( effectiveConfig.m_method == "power-law-only" || effectiveConfig.m_method == "moffat-peaks" )
    {
        errc = fillProcessPsdDropouts( result.m_processPsd,
                                       freq,
                                       processRepairMask,
                                       effectiveConfig.m_dropoutGapFactor,
                                       effectiveConfig.m_dropoutTinyFactor,
                                       effectiveConfig.m_dropoutMaxBins,
                                       result.m_powerLawIndex );
        if( !!errc )
        {
            return errc;
        }

        if( effectiveConfig.m_powerLawMatchFreq > static_cast<realT>( 0 ) )
        {
            const realT tiny = std::numeric_limits<realT>::min();
            std::vector<realT> rawProcessPsd( processMeasuredPsd.size() );
            for( size_t n = 0; n < processMeasuredPsd.size(); ++n )
            {
                rawProcessPsd[n] = std::max( processMeasuredPsd[n] - processNoisePsd[n], tiny );
            }

            std::vector<realT> continuumPsd;
            size_t rematchAnchorIndex = 0;
            errc = estimatePowerLawContinuum( continuumPsd,
                                              result.m_extrapolation,
                                              rematchAnchorIndex,
                                              rawProcessPsd,
                                              result.m_smoothedProcessPsd,
                                              processNoisePsd,
                                              freq,
                                              result.m_powerLawIndex,
                                              effectiveConfig.m_powerLawNormFreq,
                                              static_cast<realT>( 0 ),
                                              effectiveConfig.m_powerLawMatchFallbackWindowHz,
                                              false );
            if( !!errc )
            {
                return errc;
            }

            errc = matchPowerLawAtFreq( result.m_extrapolation,
                                        result.m_smoothedProcessPsd,
                                        freq,
                                        result.m_powerLawIndex,
                                        effectiveConfig.m_powerLawNormFreq,
                                        effectiveConfig.m_powerLawMatchFreq,
                                        normalizePowerLawCrossoverMode( effectiveConfig.m_powerLawCrossoverMode ) ==
                                                "auto-smoothed-crossing"
                                            ? static_cast<realT>( 0 )
                                            : effectiveConfig.m_powerLawMatchFallbackWindowHz );
            if( !!errc )
            {
                return errc;
            }

            errc = buildPowerLawContinuum( continuumPsd,
                                           result.m_extrapolation,
                                           freq,
                                           result.m_powerLawIndex,
                                           effectiveConfig.m_powerLawNormFreq );
            if( !!errc )
            {
                return errc;
            }

            result.m_powerLawAnchorIndex = rematchAnchorIndex;
            result.m_powerLawAnchorFreq = result.m_powerLawAnchorIndex < freq.size()
                                              ? freq[result.m_powerLawAnchorIndex]
                                              : static_cast<realT>( 0 );

            if( effectiveConfig.m_method == "power-law-only" )
            {
                errc = buildPowerLawOnlyProcessFromContinuum( result.m_processPsd,
                                                              processRepairMask,
                                                              rawProcessPsd,
                                                              processNoisePsd,
                                                              continuumPsd,
                                                              freq,
                                                              result.m_powerLawAnchorIndex,
                                                              effectiveConfig );
            }
            else
            {
                errc = buildMoffatProcessFromContinuum( result.m_processPsd,
                                                        result.m_peaks,
                                                        processRepairMask,
                                                        rawProcessPsd,
                                                        processNoisePsd,
                                                        continuumPsd,
                                                        freq,
                                                        result.m_powerLawAnchorIndex,
                                                        effectiveConfig );
            }
            if( !!errc )
            {
                return errc;
            }

            errc = fillProcessPsdDropouts( result.m_processPsd,
                                           freq,
                                           processRepairMask,
                                           effectiveConfig.m_dropoutGapFactor,
                                           effectiveConfig.m_dropoutTinyFactor,
                                           effectiveConfig.m_dropoutMaxBins,
                                           result.m_powerLawIndex );
            if( !!errc )
            {
                return errc;
            }
        }
    }

    errc = applyLpContinuum( result.m_lpProcessPsd, result.m_processPsd, freq, lpContinuumFreq, lpContinuumWidthHz );
    if( !!errc )
    {
        return errc;
    }

    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::estimateNoisePsd( std::vector<realT> &noisePsd,
                                                        realT &noiseFloor,
                                                        const std::vector<realT> &measuredPsd,
                                                        const std::vector<realT> &freq,
                                                        size_t modeIndex,
                                                        std::string noiseEstimateRange,
                                                        std::string noiseEstimateStatistic,
                                                        realT noiseEstimateLowFreqMaxHz )
{
    if( measuredPsd.size() < 2 || measuredPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "PSD and frequency grid must match and have at least two bins" );
    }

    noiseEstimateRange = normalizeNoiseEstimateRange( noiseEstimateRange );
    noiseEstimateStatistic = normalizeNoiseEstimateStatistic( noiseEstimateStatistic );
    size_t f0 = measuredPsd.size() / 2;
    size_t f1 = measuredPsd.size();
    if( noiseEstimateRange == "low-freq" )
    {
        f0 = measuredPsd.size() > 1 ? 1 : 0;
        f1 = std::max( f0 + static_cast<size_t>( 1 ), measuredPsd.size() / 2 );
        if( noiseEstimateLowFreqMaxHz > static_cast<realT>( 0 ) )
        {
            size_t cappedF1 = f0;
            while( cappedF1 < measuredPsd.size() && freq[cappedF1] <= noiseEstimateLowFreqMaxHz )
            {
                ++cappedF1;
            }

            f1 = std::max( f0 + static_cast<size_t>( 1 ), std::min( f1, cappedF1 ) );
        }
    }
    else if( noiseEstimateRange != "high-freq" )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Unknown noise-estimation range: " + noiseEstimateRange );
    }

    if( f1 <= f0 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr, "PSD is too short for noise fitting" );
    }

    std::vector<realT> npsd( f1 - f0 );
    const realT tiny = std::numeric_limits<realT>::min();
    for( size_t f = f0; f < f1; ++f )
    {
        npsd[f - f0] = log10( std::max( measuredPsd[f], tiny ) );
    }

    if( noiseEstimateStatistic == "minimum" )
    {
        auto minIt = std::min_element( npsd.begin(), npsd.end() );
        noiseFloor = pow( static_cast<realT>( 10 ), *minIt );
        noisePsd.assign( measuredPsd.size(), noiseFloor );
        return mx::error_t::noerror;
    }

    if( noiseEstimateStatistic != "percentile" )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Unknown noise-estimation statistic: " + noiseEstimateStatistic );
    }

    realT pct = static_cast<realT>( 0.25 );
    if( modeIndex < 2 )
    {
        pct = static_cast<realT>( 0.05 );
    }

    size_t nthIndex = static_cast<size_t>( pct * static_cast<realT>( npsd.size() ) );
    if( nthIndex >= npsd.size() )
    {
        nthIndex = npsd.size() - 1;
    }

    auto nth = npsd.begin() + nthIndex;
    std::nth_element( npsd.begin(), nth, npsd.end() );

    noiseFloor = pow( static_cast<realT>( 10 ), *nth );
    noisePsd.assign( measuredPsd.size(), noiseFloor );

    return mx::error_t::noerror;
}

template <typename realT>
std::string modalPsdProcessor<realT>::normalizeNoiseEstimateDomain( std::string domain )
{
    std::transform( domain.begin(),
                    domain.end(),
                    domain.begin(),
                    []( unsigned char c )
                    {
                        if( c == '_' )
                        {
                            return static_cast<char>( '-' );
                        }

                        return static_cast<char>( std::tolower( c ) );
                    } );

    return domain;
}

template <typename realT>
std::string modalPsdProcessor<realT>::normalizeNoiseEstimateRange( std::string range )
{
    std::transform( range.begin(),
                    range.end(),
                    range.begin(),
                    []( unsigned char c )
                    {
                        if( c == '_' )
                        {
                            return static_cast<char>( '-' );
                        }

                        return static_cast<char>( std::tolower( c ) );
                    } );

    return range;
}

template <typename realT>
std::string modalPsdProcessor<realT>::normalizeNoiseEstimateStatistic( std::string statistic )
{
    std::transform( statistic.begin(),
                    statistic.end(),
                    statistic.begin(),
                    []( unsigned char c )
                    {
                        if( c == '_' )
                        {
                            return static_cast<char>( '-' );
                        }

                        return static_cast<char>( std::tolower( c ) );
                    } );

    return statistic;
}

template <typename realT>
std::string modalPsdProcessor<realT>::normalizeClosedLoopOlEstimateMethod( std::string method )
{
    std::transform( method.begin(),
                    method.end(),
                    method.begin(),
                    []( unsigned char c )
                    {
                        if( c == '_' )
                        {
                            return static_cast<char>( '-' );
                        }

                        return static_cast<char>( std::tolower( c ) );
                    } );

    return method;
}

template <typename realT>
std::string modalPsdProcessor<realT>::normalizePowerLawCrossoverMode( std::string mode )
{
    std::transform( mode.begin(),
                    mode.end(),
                    mode.begin(),
                    []( unsigned char c )
                    {
                        if( c == '_' )
                        {
                            return static_cast<char>( '-' );
                        }

                        return static_cast<char>( std::tolower( c ) );
                    } );

    if( mode == "auto" || mode == "automatic" || mode == "auto-crossing" )
    {
        return "auto-smoothed-crossing";
    }

    return mode;
}

template <typename realT>
size_t modalPsdProcessor<realT>::firstPositiveFreqIndex( const std::vector<realT> &freq )
{
    for( size_t n = 0; n < freq.size(); ++n )
    {
        if( freq[n] > static_cast<realT>( 0 ) )
        {
            return n;
        }
    }

    return 0;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::buildSmoothedProcessPsd( std::vector<realT> &smoothedProcessPsd,
                                                               const std::vector<realT> &rawProcessPsd,
                                                               const std::vector<realT> &freq,
                                                               realT smoothWidthHz )
{
    if( rawProcessPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Smoothed disturbance PSD inputs must have "
                                                 "matching PSD and frequency sizes" );
    }

    if( rawProcessPsd.size() < 3 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Smoothed disturbance PSD requires at least three bins" );
    }

    if( smoothWidthHz <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Smoothed disturbance PSD width must be positive" );
    }

    const size_t firstPositive = firstPositiveFreqIndex( freq );
    if( firstPositive >= freq.size() || freq[firstPositive] <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Automatic power-law crossover requires positive frequencies" );
    }

    if( firstPositive + 1 >= freq.size() )
    {
        smoothedProcessPsd = rawProcessPsd;
        return mx::error_t::noerror;
    }

    const realT df = freq[firstPositive + 1] - freq[firstPositive];
    if( df <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Automatic power-law crossover requires increasing frequency bins" );
    }

    int win = std::max( 3, static_cast<int>( std::lround( smoothWidthHz / df ) ) );
    if( win % 2 == 0 )
    {
        ++win;
    }

    const realT tiny = std::numeric_limits<realT>::min();
    std::vector<realT> logRaw( rawProcessPsd.size() );
    for( size_t n = 0; n < rawProcessPsd.size(); ++n )
    {
        logRaw[n] = log10( std::max( rawProcessPsd[n], tiny ) );
    }

    std::vector<realT> logSmooth;
    mx::math::vectorSmoothMedian( logSmooth, logRaw, win );
    smoothedProcessPsd.resize( rawProcessPsd.size() );
    for( size_t n = 0; n < rawProcessPsd.size(); ++n )
    {
        smoothedProcessPsd[n] = pow( static_cast<realT>( 10 ), logSmooth[n] );
    }

    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::findAutoPowerLawCrossoverFreq( realT &crossoverFreq,
                                                                     const std::vector<realT> &smoothedProcessPsd,
                                                                     const std::vector<realT> &noisePsd,
                                                                     const std::vector<realT> &freq,
                                                                     realT maxFreqFraction )
{
    if( smoothedProcessPsd.size() != noisePsd.size() || smoothedProcessPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Automatic power-law crossover inputs must have "
                                                 "matching PSD and frequency sizes" );
    }

    if( smoothedProcessPsd.size() < 3 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Automatic power-law crossover requires at least three bins" );
    }

    const size_t firstPositive = firstPositiveFreqIndex( freq );
    if( firstPositive >= freq.size() || freq[firstPositive] <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Automatic power-law crossover requires positive frequencies" );
    }

    size_t lastSearch = freq.size() - 1;
    if( maxFreqFraction > static_cast<realT>( 0 ) && maxFreqFraction < static_cast<realT>( 1 ) )
    {
        realT maxSearchFreq = maxFreqFraction * freq.back();
        auto upper =
            std::upper_bound( freq.begin() + static_cast<std::ptrdiff_t>( firstPositive ), freq.end(), maxSearchFreq );
        if( upper == freq.begin() + static_cast<std::ptrdiff_t>( firstPositive ) )
        {
            lastSearch = firstPositive;
        }
        else
        {
            lastSearch = static_cast<size_t>( ( upper - freq.begin() ) - 1 );
        }
    }

    if( lastSearch <= firstPositive )
    {
        crossoverFreq = freq[firstPositive];
        return mx::error_t::noerror;
    }

    bool foundCrossing = false;
    realT lastCrossingFreq = freq[firstPositive];
    for( size_t n = firstPositive; n < lastSearch; ++n )
    {
        realT d0 = smoothedProcessPsd[n] - noisePsd[n];
        realT d1 = smoothedProcessPsd[n + 1] - noisePsd[n + 1];

        if( d0 == static_cast<realT>( 0 ) && d1 == static_cast<realT>( 0 ) )
        {
            lastCrossingFreq = freq[n + 1];
            foundCrossing = true;
            continue;
        }

        if( ( d0 <= static_cast<realT>( 0 ) && d1 >= static_cast<realT>( 0 ) ) ||
            ( d0 >= static_cast<realT>( 0 ) && d1 <= static_cast<realT>( 0 ) ) )
        {
            realT alpha = static_cast<realT>( 0 );
            if( d0 != d1 )
            {
                alpha = d0 / ( d0 - d1 );
            }

            alpha = std::max( static_cast<realT>( 0 ), std::min( static_cast<realT>( 1 ), alpha ) );
            lastCrossingFreq = freq[n] + alpha * ( freq[n + 1] - freq[n] );
            foundCrossing = true;
        }
    }

    if( foundCrossing )
    {
        crossoverFreq = lastCrossingFreq;
        return mx::error_t::noerror;
    }

    size_t minimumIndex = firstPositive;
    realT minimumPsd = smoothedProcessPsd[firstPositive];
    for( size_t n = firstPositive + 1; n <= lastSearch; ++n )
    {
        if( smoothedProcessPsd[n] <= minimumPsd )
        {
            minimumPsd = smoothedProcessPsd[n];
            minimumIndex = n;
        }
    }

    crossoverFreq = freq[minimumIndex];
    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::resolvePowerLawCrossoverFrequencies( realT &powerLawMatchFreq,
                                                                           realT &powerLawOnlyAboveFreq,
                                                                           const std::vector<realT> &rawProcessPsd,
                                                                           const std::vector<realT> &smoothedProcessPsd,
                                                                           const std::vector<realT> &noisePsd,
                                                                           const std::vector<realT> &freq,
                                                                           std::string powerLawCrossoverMode,
                                                                           realT powerLawAutoMaxFreqFraction )
{
    powerLawCrossoverMode = normalizePowerLawCrossoverMode( powerLawCrossoverMode );
    if( powerLawCrossoverMode != "auto-smoothed-crossing" )
    {
        return mx::error_t::noerror;
    }

    static_cast<void>( rawProcessPsd );

    realT crossoverFreq = static_cast<realT>( 0 );
    mx::error_t errc =
        findAutoPowerLawCrossoverFreq( crossoverFreq, smoothedProcessPsd, noisePsd, freq, powerLawAutoMaxFreqFraction );
    if( !!errc )
    {
        return errc;
    }

    auto upper = std::lower_bound( freq.begin(), freq.end(), crossoverFreq );
    if( upper == freq.end() )
    {
        crossoverFreq = freq.back();
    }
    else
    {
        crossoverFreq = *upper;
    }

    powerLawMatchFreq = crossoverFreq;
    powerLawOnlyAboveFreq = crossoverFreq;
    return mx::error_t::noerror;
}

template <typename realT>
realT modalPsdProcessor<realT>::resolvePowerLawNormFreq( const std::vector<realT> &freq, realT requestedNormFreq )
{
    if( requestedNormFreq > static_cast<realT>( 0 ) )
    {
        return requestedNormFreq;
    }

    size_t refIndex = firstPositiveFreqIndex( freq );
    if( refIndex < freq.size() && freq[refIndex] > static_cast<realT>( 0 ) )
    {
        return freq[refIndex];
    }

    return static_cast<realT>( 1 );
}

template <typename realT>
realT modalPsdProcessor<realT>::powerLawContinuum(
    realT extrapolation, const std::vector<realT> &freq, size_t index, realT powerLawIndex, realT powerLawNormFreq )
{
    const realT refFreq = resolvePowerLawNormFreq( freq, powerLawNormFreq );
    const realT useFreq = freq[index] > static_cast<realT>( 0 ) ? freq[index] : refFreq;

    return extrapolation * pow( refFreq / useFreq, powerLawIndex );
}

template <typename realT>
realT modalPsdProcessor<realT>::interpolatePsdAtFreq( const std::vector<realT> &psd,
                                                      const std::vector<realT> &freq,
                                                      realT targetFreq )
{
    if( psd.empty() || psd.size() != freq.size() )
    {
        return std::numeric_limits<realT>::quiet_NaN();
    }

    auto upper = std::lower_bound( freq.begin(), freq.end(), targetFreq );
    if( upper == freq.begin() )
    {
        return psd.front();
    }

    if( upper == freq.end() )
    {
        return psd.back();
    }

    size_t hi = upper - freq.begin();
    if( *upper == targetFreq )
    {
        return psd[hi];
    }

    size_t lo = hi - 1;
    if( freq[hi] == freq[lo] )
    {
        return psd[lo];
    }

    realT alpha = ( targetFreq - freq[lo] ) / ( freq[hi] - freq[lo] );
    return ( static_cast<realT>( 1 ) - alpha ) * psd[lo] + alpha * psd[hi];
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::matchPowerLawAtFreq( realT &extrapolation,
                                                           const std::vector<realT> &anchorProcessPsd,
                                                           const std::vector<realT> &freq,
                                                           realT powerLawIndex,
                                                           realT powerLawNormFreq,
                                                           realT powerLawMatchFreq,
                                                           realT powerLawMatchFallbackWindowHz )
{
    if( powerLawMatchFreq <= static_cast<realT>( 0 ) )
    {
        return mx::error_t::noerror;
    }

    if( anchorProcessPsd.size() != freq.size() || anchorProcessPsd.empty() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Anchor disturbance PSD and frequency grid must have the same size" );
    }

    size_t firstPositive = firstPositiveFreqIndex( freq );
    if( firstPositive >= freq.size() || freq[firstPositive] <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg, "Frequency grid must contain positive bins" );
    }

    if( powerLawMatchFreq < freq[firstPositive] || powerLawMatchFreq > freq.back() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Power-law match frequency is outside the sampled frequency range" );
    }

    const realT tiny = std::numeric_limits<realT>::min();
    const realT refFreq = resolvePowerLawNormFreq( freq, powerLawNormFreq );
    realT matchPsd = interpolatePsdAtFreq( anchorProcessPsd, freq, powerLawMatchFreq );
    if( !std::isfinite( matchPsd ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Could not interpolate the disturbance PSD at the match frequency" );
    }

    std::vector<realT> localPositivePsd;
    const realT localMinFreq = std::max( freq[firstPositive], powerLawMatchFreq - powerLawMatchFallbackWindowHz );
    const realT localMaxFreq = std::min( freq.back(), powerLawMatchFreq + powerLawMatchFallbackWindowHz );
    size_t localStart = std::lower_bound( freq.begin(), freq.end(), localMinFreq ) - freq.begin();
    size_t localEnd = std::upper_bound( freq.begin(), freq.end(), localMaxFreq ) - freq.begin();
    for( size_t n = localStart; n < localEnd; ++n )
    {
        if( anchorProcessPsd[n] > tiny )
        {
            localPositivePsd.push_back( anchorProcessPsd[n] );
        }
    }

    if( !localPositivePsd.empty() )
    {
        size_t mid = localPositivePsd.size() / 2;
        std::nth_element( localPositivePsd.begin(), localPositivePsd.begin() + mid, localPositivePsd.end() );
        realT localMedian = localPositivePsd[mid];
        if( localPositivePsd.size() % 2 == 0 )
        {
            auto lowerIt = std::max_element( localPositivePsd.begin(), localPositivePsd.begin() + mid );
            localMedian = static_cast<realT>( 0.5 ) * ( localMedian + *lowerIt );
        }

        if( matchPsd <= tiny || matchPsd < static_cast<realT>( 0.5 ) * localMedian )
        {
            matchPsd = localMedian;
        }
    }

    extrapolation = std::max( matchPsd, tiny ) * pow( powerLawMatchFreq / refFreq, powerLawIndex );
    return mx::error_t::noerror;
}

template <typename realT>
realT modalPsdProcessor<realT>::moffatAlphaFromFwhm( realT fwhm, realT beta )
{
    return fwhm / ( static_cast<realT>( 2 ) *
                    sqrt( pow( static_cast<realT>( 2 ), static_cast<realT>( 1 ) / beta ) - static_cast<realT>( 1 ) ) );
}

template <typename realT>
realT modalPsdProcessor<realT>::moffatValueFromFwhm( realT radius, realT peakHeight, realT fwhm, realT beta )
{
    const realT alpha = moffatAlphaFromFwhm( fwhm, beta );
    return mx::math::func::moffat<realT>( radius,
                                          static_cast<realT>( 0 ),
                                          peakHeight,
                                          static_cast<realT>( 0 ),
                                          alpha,
                                          beta );
}

template <typename realT>
realT modalPsdProcessor<realT>::fitMoffatBetaToBoundary(
    realT peakHeight, realT fwhm, realT radius, realT target, realT betaFloor )
{
    if( peakHeight <= static_cast<realT>( 0 ) || fwhm <= static_cast<realT>( 0 ) || radius <= static_cast<realT>( 0 ) ||
        target <= static_cast<realT>( 0 ) )
    {
        return betaFloor;
    }

    realT low = std::max( betaFloor, static_cast<realT>( 1.0e-3 ) );
    if( moffatValueFromFwhm( radius, peakHeight, fwhm, low ) <= target )
    {
        return low;
    }

    realT high = std::max( static_cast<realT>( 2 ) * low, low + static_cast<realT>( 0.5 ) );
    while( high < static_cast<realT>( 128 ) && moffatValueFromFwhm( radius, peakHeight, fwhm, high ) > target )
    {
        high *= static_cast<realT>( 2 );
    }

    if( moffatValueFromFwhm( radius, peakHeight, fwhm, high ) > target )
    {
        return high;
    }

    for( int n = 0; n < 60; ++n )
    {
        realT mid = static_cast<realT>( 0.5 ) * ( low + high );
        if( moffatValueFromFwhm( radius, peakHeight, fwhm, mid ) > target )
        {
            low = mid;
        }
        else
        {
            high = mid;
        }
    }

    return high;
}

template <typename realT>
realT modalPsdProcessor<realT>::interpolateCrossing( realT x0, realT y0, realT x1, realT y1, realT target )
{
    if( y1 == y0 )
    {
        return static_cast<realT>( 0.5 ) * ( x0 + x1 );
    }

    realT alpha = ( target - y0 ) / ( y1 - y0 );
    if( alpha < static_cast<realT>( 0 ) )
    {
        alpha = static_cast<realT>( 0 );
    }
    else if( alpha > static_cast<realT>( 1 ) )
    {
        alpha = static_cast<realT>( 1 );
    }

    return x0 + alpha * ( x1 - x0 );
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::fitPowerLawIndexFromBinnedMedians( realT &powerLawIndex,
                                                                         size_t &nBinsUsed,
                                                                         const std::vector<realT> &rawProcessPsd,
                                                                         const std::vector<realT> &freq,
                                                                         realT fitMinFreqHz,
                                                                         realT fitMaxFreqHz,
                                                                         realT fitBinWidthHz,
                                                                         realT includeMatchFreqHz )
{
    const realT originalPowerLawIndex = powerLawIndex;

    if( rawProcessPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Raw disturbance PSD and frequency grid must have the same size" );
    }

    if( fitMinFreqHz <= static_cast<realT>( 0 ) || fitMaxFreqHz <= fitMinFreqHz ||
        fitBinWidthHz <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Invalid power-law exponent fit range or bin width" );
    }

    if( fitMinFreqHz >= freq.back() )
    {
        nBinsUsed = 0;
        powerLawIndex = std::max( originalPowerLawIndex, static_cast<realT>( 0 ) );
        return mx::error_t::noerror;
    }

    const realT tiny = std::numeric_limits<realT>::min();
    std::vector<realT> x;
    std::vector<realT> y;

    size_t idx = std::lower_bound( freq.begin(), freq.end(), fitMinFreqHz ) - freq.begin();
    for( realT binStart = fitMinFreqHz; binStart < fitMaxFreqHz; binStart += fitBinWidthHz )
    {
        realT binEnd = std::min( binStart + fitBinWidthHz, fitMaxFreqHz );
        realT binCenter = static_cast<realT>( 0.5 ) * ( binStart + binEnd );
        if( binCenter <= static_cast<realT>( 0 ) )
        {
            continue;
        }

        std::vector<realT> logPsd;
        while( idx < freq.size() && freq[idx] < binEnd )
        {
            if( rawProcessPsd[idx] > tiny )
            {
                logPsd.push_back( log10( rawProcessPsd[idx] ) );
            }

            ++idx;
        }

        if( logPsd.empty() )
        {
            continue;
        }

        size_t mid = logPsd.size() / 2;
        std::nth_element( logPsd.begin(), logPsd.begin() + mid, logPsd.end() );
        realT medianLogPsd = logPsd[mid];
        if( logPsd.size() % 2 == 0 )
        {
            auto lowerIt = std::max_element( logPsd.begin(), logPsd.begin() + mid );
            medianLogPsd = static_cast<realT>( 0.5 ) * ( medianLogPsd + *lowerIt );
        }

        x.push_back( log10( binCenter ) );
        y.push_back( medianLogPsd );
    }

    if( includeMatchFreqHz > static_cast<realT>( 0 ) &&
        ( includeMatchFreqHz < fitMinFreqHz || includeMatchFreqHz >= fitMaxFreqHz ) )
    {
        realT matchPsd = interpolatePsdAtFreq( rawProcessPsd, freq, includeMatchFreqHz );
        if( std::isfinite( matchPsd ) && matchPsd > tiny )
        {
            x.push_back( log10( includeMatchFreqHz ) );
            y.push_back( log10( matchPsd ) );
        }
    }

    nBinsUsed = x.size();
    if( nBinsUsed < 2 )
    {
        nBinsUsed = 0;
        powerLawIndex = std::max( originalPowerLawIndex, static_cast<realT>( 0 ) );
        return mx::error_t::noerror;
    }

    realT meanX = 0;
    realT meanY = 0;
    for( size_t n = 0; n < nBinsUsed; ++n )
    {
        meanX += x[n];
        meanY += y[n];
    }

    meanX /= static_cast<realT>( nBinsUsed );
    meanY /= static_cast<realT>( nBinsUsed );

    realT varX = 0;
    realT covXY = 0;
    for( size_t n = 0; n < nBinsUsed; ++n )
    {
        realT dx = x[n] - meanX;
        realT dy = y[n] - meanY;
        varX += dx * dx;
        covXY += dx * dy;
    }

    if( varX <= static_cast<realT>( 0 ) )
    {
        nBinsUsed = 0;
        powerLawIndex = std::max( originalPowerLawIndex, static_cast<realT>( 0 ) );
        return mx::error_t::noerror;
    }

    powerLawIndex = -covXY / varX;
    if( !std::isfinite( powerLawIndex ) )
    {
        nBinsUsed = 0;
        powerLawIndex = std::max( originalPowerLawIndex, static_cast<realT>( 0 ) );
        return mx::error_t::noerror;
    }

    if( powerLawIndex < static_cast<realT>( 0 ) )
    {
        powerLawIndex = static_cast<realT>( 0 );
    }

    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::estimatePowerLawContinuum( std::vector<realT> &continuumPsd,
                                                                 realT &extrapolation,
                                                                 size_t &anchorIndex,
                                                                 const std::vector<realT> &rawProcessPsd,
                                                                 const std::vector<realT> &anchorProcessPsd,
                                                                 const std::vector<realT> &noisePsd,
                                                                 const std::vector<realT> &freq,
                                                                 realT powerLawIndex,
                                                                 realT powerLawNormFreq,
                                                                 realT powerLawMatchFreq,
                                                                 realT powerLawMatchFallbackWindowHz,
                                                                 bool fitPowerLawIndex,
                                                                 realT powerLawFitMinFreqHz,
                                                                 realT powerLawFitMaxFreqHz,
                                                                 realT powerLawFitBinWidthHz,
                                                                 bool powerLawFitIncludesMatchPoint,
                                                                 realT *usedPowerLawIndex,
                                                                 size_t *fitBinsUsed )
{
    if( rawProcessPsd.size() != noisePsd.size() || rawProcessPsd.size() != freq.size() ||
        anchorProcessPsd.size() != rawProcessPsd.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Raw PSD, anchor PSD, noise PSD, and frequency "
                                                 "grid must have the same size" );
    }

    if( rawProcessPsd.size() < 2 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr, "At least two frequency bins are required" );
    }

    if( powerLawIndex < static_cast<realT>( 0 ) && !fitPowerLawIndex )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg, "Power-law index must be non-negative" );
    }

    const realT tiny = std::numeric_limits<realT>::min();
    size_t localFitBinsUsed = 0;
    realT localPowerLawIndex = powerLawIndex;
    if( fitPowerLawIndex )
    {
        mx::error_t errc = fitPowerLawIndexFromBinnedMedians( localPowerLawIndex,
                                                              localFitBinsUsed,
                                                              rawProcessPsd,
                                                              freq,
                                                              powerLawFitMinFreqHz,
                                                              powerLawFitMaxFreqHz,
                                                              powerLawFitBinWidthHz,
                                                              powerLawFitIncludesMatchPoint ? powerLawMatchFreq
                                                                                            : static_cast<realT>( 0 ) );
        if( !!errc )
        {
            localPowerLawIndex = std::max( powerLawIndex, static_cast<realT>( 0 ) );
            localFitBinsUsed = 0;
        }
    }

    const size_t refIndex = firstPositiveFreqIndex( freq );
    const realT refFreq = resolvePowerLawNormFreq( freq, powerLawNormFreq );
    extrapolation = 0;
    anchorIndex = std::min( refIndex, freq.size() - 1 );
    int nExtrap = 0;

    size_t fMax = static_cast<size_t>( static_cast<realT>( 0.05 ) * static_cast<realT>( freq.size() ) );
    if( fMax < 2 )
    {
        fMax = std::min<size_t>( freq.size(), 2 );
    }

    for( size_t f = 1; f < fMax; ++f )
    {
        if( anchorProcessPsd[f] <= static_cast<realT>( 0.1 ) * noisePsd[f] )
        {
            continue;
        }

        extrapolation += log10( anchorProcessPsd[f] * pow( freq[f] / refFreq, localPowerLawIndex ) );
        anchorIndex = f;
        ++nExtrap;
    }

    if( nExtrap > 0 )
    {
        extrapolation = pow( static_cast<realT>( 10 ), extrapolation / static_cast<realT>( nExtrap ) );
    }
    else
    {
        const realT useFreq = freq[refIndex] > static_cast<realT>( 0 ) ? freq[refIndex] : refFreq;
        extrapolation =
            std::max( anchorProcessPsd[refIndex] * static_cast<realT>( pow( useFreq / refFreq, localPowerLawIndex ) ),
                      tiny );
    }

    mx::error_t errc = matchPowerLawAtFreq( extrapolation,
                                            anchorProcessPsd,
                                            freq,
                                            localPowerLawIndex,
                                            refFreq,
                                            powerLawMatchFreq,
                                            powerLawMatchFallbackWindowHz );
    if( !!errc )
    {
        return errc;
    }

    errc = buildPowerLawContinuum( continuumPsd, extrapolation, freq, localPowerLawIndex, refFreq );
    if( !!errc )
    {
        return errc;
    }

    if( usedPowerLawIndex )
    {
        *usedPowerLawIndex = localPowerLawIndex;
    }

    if( fitBinsUsed )
    {
        *fitBinsUsed = localFitBinsUsed;
    }

    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::buildPowerLawContinuum( std::vector<realT> &continuumPsd,
                                                              realT extrapolation,
                                                              const std::vector<realT> &freq,
                                                              realT powerLawIndex,
                                                              realT powerLawNormFreq )
{
    const realT tiny = std::numeric_limits<realT>::min();

    continuumPsd.resize( freq.size() );
    for( size_t n = 0; n < freq.size(); ++n )
    {
        continuumPsd[n] =
            std::max( powerLawContinuum( extrapolation, freq, n, powerLawIndex, powerLawNormFreq ), tiny );
    }

    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::blendContinuumAtAnchor( std::vector<realT> &processPsd,
                                                              const std::vector<realT> &rawProcessPsd,
                                                              const std::vector<realT> &continuumPsd,
                                                              size_t anchorIndex,
                                                              int blendBins )
{
    if( rawProcessPsd.size() != continuumPsd.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Raw disturbance PSD and continuum PSD must have the same size" );
    }

    if( rawProcessPsd.empty() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr, "At least one frequency bin is required" );
    }

    if( blendBins < 1 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg, "Blend width must be positive" );
    }

    const realT tiny = std::numeric_limits<realT>::min();
    anchorIndex = std::min( anchorIndex, rawProcessPsd.size() - 1 );
    size_t blendEnd = std::min( anchorIndex + static_cast<size_t>( blendBins ), rawProcessPsd.size() - 1 );

    processPsd.resize( rawProcessPsd.size() );
    for( size_t n = 0; n < processPsd.size(); ++n )
    {
        if( n <= anchorIndex )
        {
            processPsd[n] = std::max( rawProcessPsd[n], tiny );
            continue;
        }

        if( n >= blendEnd || blendEnd == anchorIndex )
        {
            processPsd[n] = std::max( continuumPsd[n], tiny );
            continue;
        }

        realT w = static_cast<realT>( n - anchorIndex ) / static_cast<realT>( blendEnd - anchorIndex );
        realT logBlend = ( static_cast<realT>( 1 ) - w ) * log10( std::max( rawProcessPsd[n], tiny ) ) +
                         w * log10( std::max( continuumPsd[n], tiny ) );
        processPsd[n] = pow( static_cast<realT>( 10 ), logBlend );
    }

    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::identifyMoffatPeaks( std::vector<identifiedPeak1D> &peaks,
                                                           const std::vector<realT> &rawProcessPsd,
                                                           const std::vector<realT> &continuumPsd,
                                                           const std::vector<realT> &freq,
                                                           realT peakDetectWidthHz,
                                                           realT peakDetectFactor,
                                                           realT peakDetectBroadFactor,
                                                           realT peakDetectMinWidthLog )
{
    if( rawProcessPsd.size() != continuumPsd.size() || rawProcessPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Raw disturbance PSD, continuum PSD, and "
                                                 "frequency grid must have the same size" );
    }

    if( rawProcessPsd.size() < 3 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr, "At least three frequency bins are required" );
    }

    if( peakDetectWidthHz <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg, "Peak-detection width must be positive" );
    }

    if( peakDetectFactor <= static_cast<realT>( 1 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg, "Peak-detection factor must exceed unity" );
    }

    if( peakDetectBroadFactor <= static_cast<realT>( 1 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Broad peak-detection factor must exceed unity" );
    }

    if( peakDetectMinWidthLog <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Broad peak minimum log-width must be positive" );
    }

    const realT df = freq[1] - freq[0];
    if( df <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg, "Frequency spacing must be positive" );
    }

    int win = std::max( 3, static_cast<int>( std::lround( peakDetectWidthHz / df ) ) );
    if( win % 2 == 0 )
    {
        ++win;
    }

    const realT tiny = std::numeric_limits<realT>::min();
    std::vector<realT> logRaw( rawProcessPsd.size() );
    for( size_t n = 0; n < rawProcessPsd.size(); ++n )
    {
        logRaw[n] = log10( std::max( rawProcessPsd[n], tiny ) );
    }

    std::vector<realT> logSmooth;
    std::vector<realT> smoothPsd( rawProcessPsd.size() );
    mx::math::vectorSmoothMedian( logSmooth, logRaw, win );
    for( size_t n = 0; n < smoothPsd.size(); ++n )
    {
        smoothPsd[n] = pow( static_cast<realT>( 10 ), logSmooth[n] );
    }

    std::vector<unsigned char> strongMask( rawProcessPsd.size(), 0 );
    std::vector<unsigned char> candidateMask( rawProcessPsd.size(), 0 );
    for( size_t n = 1; n < rawProcessPsd.size(); ++n )
    {
        if( rawProcessPsd[n] > peakDetectFactor * smoothPsd[n] )
        {
            strongMask[n] = 1;
            candidateMask[n] = 1;
        }

        if( rawProcessPsd[n] > peakDetectBroadFactor * smoothPsd[n] )
        {
            candidateMask[n] = 1;
        }
    }

    peaks.clear();
    std::vector<unsigned char> peakHasStrong;
    bool inPeak = false;
    bool currentHasStrong = false;
    identifiedPeak1D currentPeak;

    for( size_t n = 1; n < rawProcessPsd.size(); ++n )
    {
        if( candidateMask[n] )
        {
            if( !inPeak )
            {
                inPeak = true;
                currentHasStrong = ( strongMask[n] != 0 );
                currentPeak = identifiedPeak1D{};
                currentPeak.m_start = n;
                currentPeak.m_end = n;
                currentPeak.m_peakIndex = n;
            }
            else
            {
                currentPeak.m_end = n;
                currentHasStrong = currentHasStrong || ( strongMask[n] != 0 );
                if( rawProcessPsd[n] > rawProcessPsd[currentPeak.m_peakIndex] )
                {
                    currentPeak.m_peakIndex = n;
                }
            }
        }
        else if( inPeak )
        {
            inPeak = false;
            peaks.push_back( currentPeak );
            peakHasStrong.push_back( currentHasStrong ? 1 : 0 );
        }
    }

    if( inPeak )
    {
        peaks.push_back( currentPeak );
        peakHasStrong.push_back( currentHasStrong ? 1 : 0 );
    }

    std::vector<identifiedPeak1D> validPeaks;
    validPeaks.reserve( peaks.size() );

    for( size_t p = 0; p < peaks.size(); ++p )
    {
        identifiedPeak1D peak = peaks[p];
        peak.m_centerFreq = freq[peak.m_peakIndex];
        peak.m_peakHeight = rawProcessPsd[peak.m_peakIndex] - continuumPsd[peak.m_peakIndex];
        if( peak.m_peakHeight <= static_cast<realT>( 0 ) )
        {
            continue;
        }

        const realT halfLevel = continuumPsd[peak.m_peakIndex] + static_cast<realT>( 0.5 ) * peak.m_peakHeight;
        const realT defaultFwhm = std::max( df, static_cast<realT>( peak.m_end - peak.m_start + 1 ) * df );

        realT leftCross = peak.m_centerFreq - static_cast<realT>( 0.5 ) * defaultFwhm;
        for( size_t n = peak.m_peakIndex; n > 0; --n )
        {
            if( rawProcessPsd[n - 1] <= halfLevel )
            {
                leftCross =
                    interpolateCrossing( freq[n - 1], rawProcessPsd[n - 1], freq[n], rawProcessPsd[n], halfLevel );
                break;
            }
        }

        realT rightCross = peak.m_centerFreq + static_cast<realT>( 0.5 ) * defaultFwhm;
        for( size_t n = peak.m_peakIndex; n + 1 < rawProcessPsd.size(); ++n )
        {
            if( rawProcessPsd[n + 1] <= halfLevel )
            {
                rightCross =
                    interpolateCrossing( freq[n], rawProcessPsd[n], freq[n + 1], rawProcessPsd[n + 1], halfLevel );
                break;
            }
        }

        peak.m_fwhm = rightCross - leftCross;
        if( peak.m_fwhm <= static_cast<realT>( 0 ) )
        {
            peak.m_fwhm = defaultFwhm;
        }

        const realT minPositiveFreq = std::max( freq[1], tiny );
        realT lowerFreq = std::max( peak.m_centerFreq - static_cast<realT>( 0.5 ) * peak.m_fwhm, minPositiveFreq );
        realT upperFreq = std::max( peak.m_centerFreq + static_cast<realT>( 0.5 ) * peak.m_fwhm, lowerFreq );
        realT logWidth = log10( upperFreq / lowerFreq );

        if( !peakHasStrong[p] && logWidth < peakDetectMinWidthLog )
        {
            continue;
        }

        validPeaks.push_back( peak );
    }

    peaks.swap( validPeaks );
    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::addClippedMoffatPeakExcess( std::vector<realT> &peakModel,
                                                                  const identifiedPeak1D &peak,
                                                                  const std::vector<realT> &sourcePsd,
                                                                  const std::vector<realT> &continuumPsd,
                                                                  const std::vector<realT> &freq,
                                                                  realT peakMoffatBeta )
{
    if( peakModel.size() != sourcePsd.size() || peakModel.size() != continuumPsd.size() ||
        peakModel.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Peak model, source PSD, continuum PSD, and "
                                                 "frequency grid must have the same size" );
    }

    if( peak.m_peakIndex >= peakModel.size() || peak.m_fwhm <= static_cast<realT>( 0 ) )
    {
        return mx::error_t::noerror;
    }

    realT peakBeta = peakMoffatBeta;

    if( peak.m_start > 0 && peak.m_start <= peak.m_peakIndex )
    {
        const size_t edge = peak.m_start - 1;
        const realT radius = peak.m_centerFreq - freq[edge];
        peakBeta = std::max(
            peakBeta,
            fitMoffatBetaToBoundary( peak.m_peakHeight, peak.m_fwhm, radius, continuumPsd[edge], peakMoffatBeta ) );
    }

    if( peak.m_end + 1 < peakModel.size() && peak.m_end >= peak.m_peakIndex )
    {
        const size_t edge = peak.m_end + 1;
        const realT radius = freq[edge] - peak.m_centerFreq;
        peakBeta = std::max(
            peakBeta,
            fitMoffatBetaToBoundary( peak.m_peakHeight, peak.m_fwhm, radius, continuumPsd[edge], peakMoffatBeta ) );
    }

    const realT alpha = moffatAlphaFromFwhm( peak.m_fwhm, peakBeta );
    realT peakAtCenter = mx::math::func::moffat<realT>( peak.m_centerFreq,
                                                        static_cast<realT>( 0 ),
                                                        peak.m_peakHeight,
                                                        peak.m_centerFreq,
                                                        alpha,
                                                        peakBeta );
    if( peakAtCenter <= continuumPsd[peak.m_peakIndex] )
    {
        return mx::error_t::noerror;
    }

    size_t left = peak.m_peakIndex;
    while( left > 0 && sourcePsd[left - 1] > continuumPsd[left - 1] )
    {
        --left;
    }

    size_t right = peak.m_peakIndex;
    while( right + 1 < peakModel.size() && sourcePsd[right + 1] > continuumPsd[right + 1] )
    {
        ++right;
    }

    for( size_t n = left; n <= right; ++n )
    {
        realT peakValue = mx::math::func::moffat<realT>( freq[n],
                                                         static_cast<realT>( 0 ),
                                                         peak.m_peakHeight,
                                                         peak.m_centerFreq,
                                                         alpha,
                                                         peakBeta );
        realT measuredExcess = std::max( sourcePsd[n] - continuumPsd[n], static_cast<realT>( 0 ) );
        if( measuredExcess > static_cast<realT>( 0 ) )
        {
            peakModel[n] += std::min( peakValue, measuredExcess );
        }
    }

    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::identifyMoffatPeaksMultiPass( std::vector<identifiedPeak1D> &peaks,
                                                                    const std::vector<realT> &rawProcessPsd,
                                                                    const std::vector<realT> &continuumPsd,
                                                                    const std::vector<realT> &freq,
                                                                    realT peakDetectWidthHz,
                                                                    realT peakDetectFactor,
                                                                    realT peakDetectBroadFactor,
                                                                    realT peakDetectMinWidthLog,
                                                                    int peakDetectPasses,
                                                                    realT peakMoffatBeta )
{
    if( peakDetectPasses < 1 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg, "Peak-detection passes must be positive" );
    }

    const realT tiny = std::numeric_limits<realT>::min();
    std::vector<realT> workingExcess( rawProcessPsd.size() );
    for( size_t n = 0; n < workingExcess.size(); ++n )
    {
        workingExcess[n] = std::max( rawProcessPsd[n] - continuumPsd[n], static_cast<realT>( 0 ) );
    }

    peaks.clear();

    for( int pass = 0; pass < peakDetectPasses; ++pass )
    {
        std::vector<realT> workingPsd( rawProcessPsd.size() );
        for( size_t n = 0; n < workingPsd.size(); ++n )
        {
            workingPsd[n] = std::max( continuumPsd[n] + workingExcess[n], tiny );
        }

        std::vector<identifiedPeak1D> passPeaks;
        mx::error_t errc = identifyMoffatPeaks( passPeaks,
                                                workingPsd,
                                                continuumPsd,
                                                freq,
                                                peakDetectWidthHz,
                                                peakDetectFactor,
                                                peakDetectBroadFactor,
                                                peakDetectMinWidthLog );
        if( !!errc )
        {
            return errc;
        }

        if( passPeaks.empty() )
        {
            break;
        }

        std::vector<realT> passModel( rawProcessPsd.size(), static_cast<realT>( 0 ) );
        for( size_t p = 0; p < passPeaks.size(); ++p )
        {
            errc =
                addClippedMoffatPeakExcess( passModel, passPeaks[p], workingPsd, continuumPsd, freq, peakMoffatBeta );
            if( !!errc )
            {
                return errc;
            }
        }

        bool subtracted = false;
        for( size_t n = 0; n < workingExcess.size(); ++n )
        {
            realT newExcess = std::max( workingExcess[n] - passModel[n], static_cast<realT>( 0 ) );
            if( newExcess < workingExcess[n] )
            {
                subtracted = true;
            }
            workingExcess[n] = newExcess;
        }

        peaks.insert( peaks.end(), passPeaks.begin(), passPeaks.end() );
        if( !subtracted )
        {
            break;
        }
    }

    std::sort( peaks.begin(),
               peaks.end(),
               []( const identifiedPeak1D &a, const identifiedPeak1D &b ) { return a.m_peakIndex < b.m_peakIndex; } );

    auto newEnd = std::unique( peaks.begin(),
                               peaks.end(),
                               []( const identifiedPeak1D &a, const identifiedPeak1D &b )
                               { return a.m_peakIndex == b.m_peakIndex; } );
    peaks.erase( newEnd, peaks.end() );

    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::buildMoffatProcessFromContinuum( std::vector<realT> &processPsd,
                                                                       std::vector<identifiedPeak1D> &peaks,
                                                                       std::vector<unsigned char> &repairMask,
                                                                       const std::vector<realT> &rawProcessPsd,
                                                                       const std::vector<realT> &noisePsd,
                                                                       const std::vector<realT> &continuumPsd,
                                                                       const std::vector<realT> &freq,
                                                                       size_t anchorIndex,
                                                                       const processModelConfig &config )
{
    if( rawProcessPsd.size() != noisePsd.size() || rawProcessPsd.size() != continuumPsd.size() ||
        rawProcessPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Raw disturbance PSD, noise PSD, continuum PSD, "
                                                 "and frequency grid must have the same size" );
    }

    const realT tiny = std::numeric_limits<realT>::min();

    mx::error_t errc = identifyMoffatPeaksMultiPass( peaks,
                                                     rawProcessPsd,
                                                     continuumPsd,
                                                     freq,
                                                     config.m_peakDetectWidthHz,
                                                     config.m_peakDetectFactor,
                                                     config.m_peakDetectBroadFactor,
                                                     config.m_peakDetectMinWidthLog,
                                                     config.m_peakDetectPasses,
                                                     config.m_peakMoffatBeta );
    if( !!errc )
    {
        return errc;
    }

    std::vector<realT> peakModel( rawProcessPsd.size(), static_cast<realT>( 0 ) );
    for( size_t p = 0; p < peaks.size(); ++p )
    {
        errc = addClippedMoffatPeakExcess( peakModel,
                                           peaks[p],
                                           rawProcessPsd,
                                           continuumPsd,
                                           freq,
                                           config.m_peakMoffatBeta );
        if( !!errc )
        {
            return errc;
        }
    }

    std::vector<realT> highFreqModel( rawProcessPsd.size() );
    for( size_t n = 0; n < highFreqModel.size(); ++n )
    {
        if( config.m_powerLawOnlyAboveFreq > static_cast<realT>( 0 ) && freq[n] >= config.m_powerLawOnlyAboveFreq )
        {
            highFreqModel[n] = std::max( continuumPsd[n], tiny );
            continue;
        }

        realT measuredExcess = std::max( rawProcessPsd[n] - continuumPsd[n], static_cast<realT>( 0 ) );
        highFreqModel[n] = std::max( continuumPsd[n] + std::min( peakModel[n], measuredExcess ), tiny );
    }

    std::vector<realT> extrapolatedPsd;
    errc = blendContinuumAtAnchor( extrapolatedPsd,
                                   rawProcessPsd,
                                   highFreqModel,
                                   anchorIndex,
                                   config.m_powerLawBlendBins );
    if( !!errc )
    {
        return errc;
    }

    processPsd.resize( rawProcessPsd.size() );
    repairMask.assign( rawProcessPsd.size(), 0 );
    size_t blendEnd =
        std::min( anchorIndex + static_cast<size_t>( config.m_powerLawBlendBins ), rawProcessPsd.size() - 1 );
    for( size_t n = 0; n <= blendEnd; ++n )
    {
        repairMask[n] = 1;
    }

    for( size_t p = 0; p < peaks.size(); ++p )
    {
        size_t peakStart = std::min( peaks[p].m_start, repairMask.size() - 1 );
        size_t peakEnd = std::min( peaks[p].m_end, repairMask.size() - 1 );
        for( size_t n = peakStart; n <= peakEnd; ++n )
        {
            repairMask[n] = 1;
        }
    }

    if( config.m_powerLawOnlyAboveFreq > static_cast<realT>( 0 ) )
    {
        for( size_t n = 0; n < repairMask.size(); ++n )
        {
            if( freq[n] >= config.m_powerLawOnlyAboveFreq )
            {
                repairMask[n] = 0;
            }
        }
    }

    for( size_t n = 0; n < processPsd.size(); ++n )
    {
        if( config.m_powerLawOnlyAboveFreq > static_cast<realT>( 0 ) && freq[n] >= config.m_powerLawOnlyAboveFreq )
        {
            processPsd[n] = std::max( continuumPsd[n], tiny );
            continue;
        }

        if( rawProcessPsd[n] > noisePsd[n] )
        {
            processPsd[n] = std::max( rawProcessPsd[n], tiny );
        }
        else
        {
            processPsd[n] = std::max( extrapolatedPsd[n], tiny );
        }
    }

    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::buildPowerLawOnlyProcessFromContinuum( std::vector<realT> &processPsd,
                                                                             std::vector<unsigned char> &repairMask,
                                                                             const std::vector<realT> &rawProcessPsd,
                                                                             const std::vector<realT> &noisePsd,
                                                                             const std::vector<realT> &continuumPsd,
                                                                             const std::vector<realT> &freq,
                                                                             size_t anchorIndex,
                                                                             const processModelConfig &config )
{
    if( rawProcessPsd.size() != noisePsd.size() || rawProcessPsd.size() != continuumPsd.size() ||
        rawProcessPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Raw disturbance PSD, noise PSD, continuum PSD, "
                                                 "and frequency grid must have the same size" );
    }

    const realT tiny = std::numeric_limits<realT>::min();
    std::vector<realT> extrapolatedPsd;
    bool hardAutoHandoff =
        normalizePowerLawCrossoverMode( config.m_powerLawCrossoverMode ) == "auto-smoothed-crossing" &&
        config.m_powerLawMatchFreq > static_cast<realT>( 0 );
    if( hardAutoHandoff )
    {
        extrapolatedPsd = continuumPsd;
    }
    else
    {
        mx::error_t errc = blendContinuumAtAnchor( extrapolatedPsd,
                                                   rawProcessPsd,
                                                   continuumPsd,
                                                   anchorIndex,
                                                   config.m_powerLawBlendBins );
        if( !!errc )
        {
            return errc;
        }
    }

    processPsd.resize( rawProcessPsd.size() );
    repairMask.assign( rawProcessPsd.size(), 1 );
    for( size_t n = 0; n < processPsd.size(); ++n )
    {
        if( config.m_powerLawOnlyAboveFreq > static_cast<realT>( 0 ) && freq[n] >= config.m_powerLawOnlyAboveFreq )
        {
            processPsd[n] = std::max( continuumPsd[n], tiny );
            repairMask[n] = 0;
        }
        else if( hardAutoHandoff )
        {
            processPsd[n] = std::max( rawProcessPsd[n], tiny );
        }
        else if( rawProcessPsd[n] > noisePsd[n] )
        {
            processPsd[n] = std::max( rawProcessPsd[n], tiny );
        }
        else
        {
            processPsd[n] = std::max( extrapolatedPsd[n], tiny );
        }
    }

    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::estimateProcessPsdPowerLawOnly( std::vector<realT> &processPsd,
                                                                      realT &extrapolation,
                                                                      size_t &anchorIndex,
                                                                      std::vector<unsigned char> &repairMask,
                                                                      const std::vector<realT> &measuredPsd,
                                                                      const std::vector<realT> &anchorProcessPsd,
                                                                      const std::vector<realT> &noisePsd,
                                                                      const std::vector<realT> &freq,
                                                                      const processModelConfig &config,
                                                                      realT *usedPowerLawIndex,
                                                                      size_t *fitBinsUsed )
{
    if( measuredPsd.size() != noisePsd.size() || measuredPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>(
            mx::error_t::sizeerr,
            "Measured PSD, noise PSD, and frequency grid must have the same size" );
    }

    const realT tiny = std::numeric_limits<realT>::min();
    std::vector<realT> rawProcessPsd( measuredPsd.size() );
    for( size_t n = 0; n < measuredPsd.size(); ++n )
    {
        rawProcessPsd[n] = std::max( measuredPsd[n] - noisePsd[n], tiny );
    }

    std::vector<realT> continuumPsd;
    mx::error_t errc = estimatePowerLawContinuum( continuumPsd,
                                                  extrapolation,
                                                  anchorIndex,
                                                  rawProcessPsd,
                                                  anchorProcessPsd,
                                                  noisePsd,
                                                  freq,
                                                  config.m_powerLawIndex,
                                                  config.m_powerLawNormFreq,
                                                  config.m_powerLawMatchFreq,
                                                  config.m_powerLawMatchFallbackWindowHz,
                                                  config.m_fitPowerLawIndex,
                                                  config.m_powerLawFitMinFreqHz,
                                                  config.m_powerLawFitMaxFreqHz,
                                                  config.m_powerLawFitBinWidthHz,
                                                  config.m_powerLawFitIncludesMatchPoint,
                                                  usedPowerLawIndex,
                                                  fitBinsUsed );
    if( !!errc )
    {
        return errc;
    }

    if( normalizePowerLawCrossoverMode( config.m_powerLawCrossoverMode ) == "auto-smoothed-crossing" &&
        config.m_powerLawMatchFreq > static_cast<realT>( 0 ) )
    {
        realT exactPowerLawIndex = config.m_powerLawIndex;
        if( usedPowerLawIndex != nullptr )
        {
            exactPowerLawIndex = *usedPowerLawIndex;
        }

        errc = matchPowerLawAtFreq( extrapolation,
                                    anchorProcessPsd,
                                    freq,
                                    exactPowerLawIndex,
                                    config.m_powerLawNormFreq,
                                    config.m_powerLawMatchFreq,
                                    static_cast<realT>( 0 ) );
        if( !!errc )
        {
            return errc;
        }

        errc =
            buildPowerLawContinuum( continuumPsd, extrapolation, freq, exactPowerLawIndex, config.m_powerLawNormFreq );
        if( !!errc )
        {
            return errc;
        }
    }

    return buildPowerLawOnlyProcessFromContinuum( processPsd,
                                                  repairMask,
                                                  rawProcessPsd,
                                                  noisePsd,
                                                  continuumPsd,
                                                  freq,
                                                  anchorIndex,
                                                  config );
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::estimateProcessPsdMoffatPeaks( std::vector<realT> &processPsd,
                                                                     realT &extrapolation,
                                                                     std::vector<identifiedPeak1D> &peaks,
                                                                     std::vector<unsigned char> &repairMask,
                                                                     const std::vector<realT> &measuredPsd,
                                                                     const std::vector<realT> &anchorProcessPsd,
                                                                     const std::vector<realT> &noisePsd,
                                                                     const std::vector<realT> &freq,
                                                                     const processModelConfig &config )
{
    if( measuredPsd.size() != noisePsd.size() || measuredPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>(
            mx::error_t::sizeerr,
            "Measured PSD, noise PSD, and frequency grid must have the same size" );
    }

    if( measuredPsd.size() < 3 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr, "At least three frequency bins are required" );
    }

    if( config.m_peakMoffatBeta <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg, "Peak Moffat beta must be positive" );
    }

    const realT tiny = std::numeric_limits<realT>::min();
    std::vector<realT> rawProcessPsd( measuredPsd.size() );
    for( size_t n = 0; n < measuredPsd.size(); ++n )
    {
        rawProcessPsd[n] = std::max( measuredPsd[n] - noisePsd[n], tiny );
    }

    std::vector<realT> continuumPsd;
    size_t anchorIndex = 0;
    mx::error_t errc = estimatePowerLawContinuum( continuumPsd,
                                                  extrapolation,
                                                  anchorIndex,
                                                  rawProcessPsd,
                                                  anchorProcessPsd,
                                                  noisePsd,
                                                  freq,
                                                  config.m_powerLawIndex,
                                                  config.m_powerLawNormFreq,
                                                  config.m_powerLawMatchFreq,
                                                  config.m_powerLawMatchFallbackWindowHz,
                                                  config.m_fitPowerLawIndex,
                                                  config.m_powerLawFitMinFreqHz,
                                                  config.m_powerLawFitMaxFreqHz,
                                                  config.m_powerLawFitBinWidthHz,
                                                  config.m_powerLawFitIncludesMatchPoint );
    if( !!errc )
    {
        return errc;
    }

    return buildMoffatProcessFromContinuum( processPsd,
                                            peaks,
                                            repairMask,
                                            rawProcessPsd,
                                            noisePsd,
                                            continuumPsd,
                                            freq,
                                            anchorIndex,
                                            config );
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::estimateProcessPsd( std::vector<realT> &processPsd,
                                                          realT &extrapolation,
                                                          const std::vector<realT> &measuredPsd,
                                                          const std::vector<realT> &noisePsd,
                                                          const std::vector<realT> &freq,
                                                          realT powerLawNormFreq,
                                                          realT powerLawMatchFreq,
                                                          realT powerLawMatchFallbackWindowHz )
{
    if( measuredPsd.size() != noisePsd.size() || measuredPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>(
            mx::error_t::sizeerr,
            "Measured PSD, noise PSD, and frequency grid must have the same size" );
    }

    if( measuredPsd.size() < 2 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr, "At least two frequency bins are required" );
    }

    const realT tiny = std::numeric_limits<realT>::min();
    const size_t refIndex = firstPositiveFreqIndex( freq );
    const realT refFreq = resolvePowerLawNormFreq( freq, powerLawNormFreq );
    processPsd.resize( measuredPsd.size() );
    for( size_t f = 0; f < measuredPsd.size(); ++f )
    {
        processPsd[f] = measuredPsd[f] - noisePsd[f];
    }

    extrapolation = 0;
    int nExtrap = 0;

    size_t fMax = static_cast<size_t>( static_cast<realT>( 0.05 ) * static_cast<realT>( freq.size() ) );
    if( fMax < 2 )
    {
        fMax = std::min<size_t>( freq.size(), 2 );
    }

    for( size_t f = 1; f < fMax; ++f )
    {
        if( processPsd[f] <= static_cast<realT>( 0.1 ) * noisePsd[f] )
        {
            continue;
        }

        extrapolation += log10( processPsd[f] * pow( freq[f] / refFreq, c_defaultPowerLawIndex ) );
        ++nExtrap;
    }

    if( nExtrap > 0 )
    {
        extrapolation = pow( static_cast<realT>( 10 ), extrapolation / static_cast<realT>( nExtrap ) );
    }
    else
    {
        const realT useFreq = freq[refIndex] > static_cast<realT>( 0 ) ? freq[refIndex] : refFreq;
        extrapolation =
            std::max( processPsd[refIndex] * static_cast<realT>( pow( useFreq / refFreq, c_defaultPowerLawIndex ) ),
                      tiny );
    }

    mx::error_t errc = matchPowerLawAtFreq( extrapolation,
                                            processPsd,
                                            freq,
                                            c_defaultPowerLawIndex,
                                            refFreq,
                                            powerLawMatchFreq,
                                            powerLawMatchFallbackWindowHz );
    if( !!errc )
    {
        return errc;
    }

    std::vector<realT> toSmooth( freq.size() );
    std::vector<realT> l10( freq.size() );
    std::vector<realT> smoothed( freq.size() );
    std::vector<int> smoothWidths( freq.size() );

    for( size_t f = 0; f < freq.size(); ++f )
    {
        if( processPsd[f] < static_cast<realT>( 0 ) )
        {
            const realT useFreq = freq[f] > static_cast<realT>( 0 ) ? freq[f] : refFreq;
            toSmooth[f] = extrapolation * pow( refFreq / useFreq, c_defaultPowerLawIndex );
        }
        else
        {
            toSmooth[f] = processPsd[f];
        }

        toSmooth[f] = std::max( toSmooth[f], tiny );
        smoothWidths[f] = f < 2 ? 2 : 2 + static_cast<int>( static_cast<realT>( f ) / static_cast<realT>( 10 ) );
        l10[f] = log10( toSmooth[f] );
    }

    mx::math::vectorSmoothMean( smoothed, l10, smoothWidths );

    for( size_t f = 0; f < freq.size(); ++f )
    {
        smoothed[f] = pow( static_cast<realT>( 10 ), smoothed[f] );
        if( processPsd[f] < noisePsd[f] )
        {
            processPsd[f] = smoothed[f];
        }

        processPsd[f] = std::max( processPsd[f], tiny );
    }

    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::fillProcessPsdDropouts( std::vector<realT> &processPsd,
                                                              const std::vector<realT> &freq,
                                                              const std::vector<unsigned char> &repairMask,
                                                              realT gapFactor,
                                                              realT tinyFactor,
                                                              size_t maxGapBins,
                                                              realT powerLawIndex )
{
    if( processPsd.size() < 3 )
    {
        return mx::error_t::noerror;
    }

    if( processPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Dropout repair frequency grid must match the disturbance PSD size" );
    }

    if( !repairMask.empty() && repairMask.size() != processPsd.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Dropout repair mask must match the disturbance PSD size" );
    }

    if( gapFactor <= static_cast<realT>( 0 ) || gapFactor >= static_cast<realT>( 1 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Dropout gap factor must be between 0 and 1" );
    }

    if( tinyFactor <= static_cast<realT>( 0 ) || tinyFactor >= static_cast<realT>( 1 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg,
                                                 "Dropout tiny factor must be between 0 and 1" );
    }

    const realT tiny = std::numeric_limits<realT>::min();
    (void)maxGapBins;

    realT refFreq = static_cast<realT>( 1 );
    for( size_t n = 0; n < freq.size(); ++n )
    {
        if( freq[n] > static_cast<realT>( 0 ) )
        {
            refFreq = freq[n];
            break;
        }
    }

    std::vector<realT> sourcePsd = processPsd;
    std::vector<realT> updatedPsd = processPsd;
    bool changed = false;

    if( repairMask.empty() || repairMask[0] != 0 )
    {
        realT runMax = sourcePsd[0];
        for( size_t end = 0; end + 2 < sourcePsd.size(); ++end )
        {
            if( !repairMask.empty() && repairMask[end] == 0 )
            {
                break;
            }

            runMax = std::max( runMax, sourcePsd[end] );

            realT right1 = std::max( sourcePsd[end + 1], tiny );
            realT right2 = std::max( sourcePsd[end + 2], tiny );
            realT flankMin = std::min( right1, right2 );
            realT flankMax = std::max( right1, right2 );
            if( runMax >= gapFactor * flankMin )
            {
                continue;
            }

            if( runMax > tinyFactor * flankMax )
            {
                continue;
            }

            realT xLeft = log10( std::max( freq[end + 1], refFreq ) );
            realT xRight = log10( std::max( freq[end + 2], refFreq ) );
            if( xRight <= xLeft )
            {
                xRight = xLeft + static_cast<realT>( 1 );
            }

            realT yLeft = log10( std::max( sourcePsd[end + 1], tiny ) );
            realT yRight = log10( std::max( sourcePsd[end + 2], tiny ) );

            for( size_t fill = 0; fill <= end; ++fill )
            {
                realT xFill = log10( std::max( freq[fill], refFreq ) );
                realT alpha = ( xFill - xLeft ) / ( xRight - xLeft );
                realT fillValue = pow( static_cast<realT>( 10 ), yLeft + alpha * ( yRight - yLeft ) );
                changed = changed || fillValue != updatedPsd[fill];
                updatedPsd[fill] = std::max( fillValue, tiny );
            }

            if( updatedPsd.size() > 1 )
            {
                changed = changed || updatedPsd[0] != updatedPsd[1];
                updatedPsd[0] = updatedPsd[1];
            }

            sourcePsd = updatedPsd;
            break;
        }
    }

    size_t n = 1;
    while( n + 1 < sourcePsd.size() )
    {
        if( !repairMask.empty() && repairMask[n] == 0 )
        {
            ++n;
            continue;
        }

        realT leftVal = std::max( sourcePsd[n - 1], tiny );
        if( sourcePsd[n] >= gapFactor * leftVal )
        {
            ++n;
            continue;
        }

        realT runMax = sourcePsd[n];
        bool repaired = false;
        for( size_t end = n; end + 1 < sourcePsd.size(); ++end )
        {
            if( !repairMask.empty() && repairMask[end] == 0 )
            {
                break;
            }

            runMax = std::max( runMax, sourcePsd[end] );

            realT rightVal = std::max( sourcePsd[end + 1], tiny );
            realT flankMax = std::max( leftVal, rightVal );
            bool canExtend = end + 1 < sourcePsd.size() - 1;
            if( canExtend && !repairMask.empty() )
            {
                canExtend = repairMask[end + 1] != 0;
            }

            if( sourcePsd[end] >= gapFactor * rightVal )
            {
                if( canExtend )
                {
                    continue;
                }

                break;
            }

            realT flankMin = std::min( leftVal, rightVal );
            if( runMax >= gapFactor * flankMin )
            {
                if( canExtend )
                {
                    continue;
                }

                break;
            }

            if( runMax > tinyFactor * flankMax )
            {
                if( canExtend )
                {
                    continue;
                }

                break;
            }

            realT xLeft = log10( std::max( freq[n - 1], refFreq ) );
            realT xRight = log10( std::max( freq[end + 1], refFreq ) );
            if( xRight <= xLeft )
            {
                xRight = xLeft + static_cast<realT>( 1 );
            }

            realT yLeft = log10( std::max( sourcePsd[n - 1], tiny ) );
            realT yRight = log10( std::max( sourcePsd[end + 1], tiny ) );

            for( size_t fill = n; fill <= end; ++fill )
            {
                realT xFill = log10( std::max( freq[fill], refFreq ) );
                realT alpha = ( xFill - xLeft ) / ( xRight - xLeft );
                if( alpha < static_cast<realT>( 0 ) )
                {
                    alpha = static_cast<realT>( 0 );
                }
                else if( alpha > static_cast<realT>( 1 ) )
                {
                    alpha = static_cast<realT>( 1 );
                }

                realT fillValue = pow( static_cast<realT>( 10 ), yLeft + alpha * ( yRight - yLeft ) );
                changed = changed || fillValue != updatedPsd[fill];
                updatedPsd[fill] = std::max( fillValue, tiny );
            }

            repaired = true;
            n = end + 1;
            break;
        }

        if( !repaired )
        {
            ++n;
        }
    }

    if( repairMask.empty() || repairMask.back() != 0 )
    {
        realT runMax = sourcePsd.back();
        for( size_t offset = 0; offset + 2 < sourcePsd.size(); ++offset )
        {
            size_t start = sourcePsd.size() - 1 - offset;
            if( !repairMask.empty() && repairMask[start] == 0 )
            {
                break;
            }

            runMax = std::max( runMax, sourcePsd[start] );

            realT left1 = std::max( sourcePsd[start - 1], tiny );
            realT left2 = std::max( sourcePsd[start - 2], tiny );
            realT flankMin = std::min( left1, left2 );
            realT flankMax = std::max( left1, left2 );
            if( runMax >= gapFactor * flankMin )
            {
                continue;
            }

            if( runMax > tinyFactor * flankMax )
            {
                continue;
            }

            realT yRight = log10( std::max( sourcePsd[start - 1], tiny ) );

            for( size_t fill = start; fill < sourcePsd.size(); ++fill )
            {
                realT useFreq = std::max( freq[fill], refFreq );
                realT lastGoodFreq = std::max( freq[start - 1], refFreq );
                realT fillValue =
                    pow( static_cast<realT>( 10 ), yRight ) * pow( lastGoodFreq / useFreq, powerLawIndex );
                changed = changed || fillValue != updatedPsd[fill];
                updatedPsd[fill] = std::max( fillValue, tiny );
            }

            sourcePsd = updatedPsd;
            break;
        }
    }

    if( changed )
    {
        processPsd.swap( updatedPsd );
    }

    return mx::error_t::noerror;
}

template <typename realT>
mx::error_t modalPsdProcessor<realT>::applyLpContinuum( std::vector<realT> &lpProcessPsd,
                                                        const std::vector<realT> &processPsd,
                                                        const std::vector<realT> &freq,
                                                        realT cutoffFreq,
                                                        realT continuumWidthHz )
{
    if( processPsd.size() != freq.size() )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr,
                                                 "Process PSD and frequency grid must be the same size" );
    }

    lpProcessPsd = processPsd;
    if( cutoffFreq <= static_cast<realT>( 0 ) )
    {
        return mx::error_t::noerror;
    }

    if( continuumWidthHz <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg, "LP continuum width must be positive" );
    }

    if( freq.size() < 2 )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::sizeerr, "At least two frequency bins are required" );
    }

    auto cutIt = std::lower_bound( freq.begin(), freq.end(), cutoffFreq );
    if( cutIt == freq.end() )
    {
        return mx::error_t::noerror;
    }

    size_t cutIndex = cutIt - freq.begin();
    if( cutIndex >= lpProcessPsd.size() )
    {
        return mx::error_t::noerror;
    }

    const realT df = freq[1] - freq[0];
    if( df <= static_cast<realT>( 0 ) )
    {
        return mx::error_report<mx::verbose::d>( mx::error_t::invalidarg, "Frequency spacing must be positive" );
    }

    int win = std::max( 3, static_cast<int>( std::lround( continuumWidthHz / df ) ) );
    if( win % 2 == 0 )
    {
        ++win;
    }

    const realT tiny = std::numeric_limits<realT>::min();
    std::vector<realT> logTail( lpProcessPsd.size() - cutIndex );
    for( size_t n = cutIndex; n < processPsd.size(); ++n )
    {
        logTail[n - cutIndex] = log10( std::max( processPsd[n], tiny ) );
    }

    std::vector<realT> logMedianTail;
    std::vector<realT> logContinuumTail;
    mx::math::vectorSmoothMedian( logMedianTail, logTail, win );
    mx::math::vectorSmoothMean( logContinuumTail, logMedianTail, win );

    for( size_t n = cutIndex; n < lpProcessPsd.size(); ++n )
    {
        lpProcessPsd[n] =
            std::max( static_cast<realT>( pow( static_cast<realT>( 10 ), logContinuumTail[n - cutIndex] ) ), tiny );
    }

    return mx::error_t::noerror;
}

} // namespace app
} // namespace MagAOX

#endif // modalPsdProcessor_hpp
