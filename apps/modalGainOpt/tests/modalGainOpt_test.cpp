/** \file modalGainOpt_test.cpp
 * \brief Catch2 tests for the modalGainOpt app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup modalGainOpt_files
 */

#include "../../../tests/testXWC.hpp"

#include "../modalGainOpt.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup modalGainOpt_unit_test modalGainOpt Unit Tests
 * \brief Unit tests for the modalGainOpt application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `modalGainOpt` unit tests.
/** \ingroup modalGainOpt_unit_test
 */
namespace modalGainOptTest
{

class processPsdProcessorHarness : public processPsdProcessorT
{
  public:
    using processPsdProcessorT::buildSmoothedProcessPsd;
    using processPsdProcessorT::estimatePowerLawContinuum;
    using processPsdProcessorT::estimateProcessPsdPowerLawOnly;
    using processPsdProcessorT::fillProcessPsdDropouts;
    using processPsdProcessorT::findAutoPowerLawCrossoverFreq;
    using processPsdProcessorT::resolvePowerLawCrossoverFrequencies;
};

/// \cond
class modalGainOptHarness : public modalGainOpt
{
  public:
    void readConfigFile( const std::string &path )
    {
        config.readConfig( path );
    }

    int shutdownState() const
    {
        return m_shutdown;
    }

    float gainGain() const
    {
        return m_gainGain;
    }

    float gainLeak() const
    {
        return m_gainLeak;
    }

    int extrapMethod() const
    {
        return m_extrapOL;
    }

    int extrapNoiseEstimateDomain() const
    {
        return m_extrapNoiseEstimateDomain;
    }

    int extrapNoiseEstimateRange() const
    {
        return m_extrapNoiseEstimateRange;
    }

    int extrapNoiseEstimateStatistic() const
    {
        return m_extrapNoiseEstimateStatistic;
    }

    int extrapClosedLoopOlEstimateMethod() const
    {
        return m_extrapClosedLoopOlEstimateMethod;
    }

    int extrapPowerLawCrossoverMode() const
    {
        return m_extrapPowerLawCrossoverMode;
    }

    void setExtrapMethodForTest( int method )
    {
        m_extrapOL = method;
    }

    void setExtrapNoiseEstimateDomainForTest( int domain )
    {
        m_extrapNoiseEstimateDomain = domain;
        m_extrapConfig.m_noiseEstimateDomain = extrapNoiseEstimateDomainName( domain );
    }

    void setExtrapNoiseEstimateRangeForTest( int range )
    {
        m_extrapNoiseEstimateRange = range;
        m_extrapConfig.m_noiseEstimateRange = extrapNoiseEstimateRangeName( range );
    }

    void setExtrapNoiseEstimateStatisticForTest( int statistic )
    {
        m_extrapNoiseEstimateStatistic = statistic;
        m_extrapConfig.m_noiseEstimateStatistic = extrapNoiseEstimateStatisticName( statistic );
    }

    void setExtrapClosedLoopOlEstimateMethodForTest( int method )
    {
        m_extrapClosedLoopOlEstimateMethod = method;
        m_extrapConfig.m_closedLoopOlEstimateMethod = extrapClosedLoopOlEstimateMethodName( method );
    }

    void setExtrapPowerLawCrossoverModeForTest( int mode )
    {
        m_extrapPowerLawCrossoverMode = mode;
        m_extrapConfig.m_powerLawCrossoverMode = extrapPowerLawCrossoverModeName( mode );
    }

    const processPsdProcessorT::processModelConfig &extrapConfig() const
    {
        return m_extrapConfig;
    }

    bool autoUpdate() const
    {
        return m_autoUpdate;
    }

    int modesOn() const
    {
        return m_modesOn;
    }

    int sinceChange() const
    {
        return m_sinceChange;
    }

    bool goptUpdated() const
    {
        return m_goptUpdated;
    }

    bool pcgoptUpdated() const
    {
        return m_pcgoptUpdated;
    }

    bool freqUpdated() const
    {
        return m_freqUpdated;
    }

    float fps() const
    {
        return m_fps;
    }

    const std::vector<float> &freq() const
    {
        return m_freq;
    }

    void setPcOnForTest( bool pcOn )
    {
        m_pcOn = pcOn;
    }

    void setLoopForTest( bool loop )
    {
        m_loop = loop;
    }

    void setModesOnForTest( int modesOn )
    {
        m_modesOn = modesOn;
    }

    void setSinceChangeForTest( int sinceChange )
    {
        m_sinceChange = sinceChange;
    }

    void setGoptUpdatedForTest( bool goptUpdated )
    {
        m_goptUpdated = goptUpdated;
    }

    void setPcgoptUpdatedForTest( bool pcgoptUpdated )
    {
        m_pcgoptUpdated = pcgoptUpdated;
    }

    void setFreqUpdatedForTest( bool freqUpdated )
    {
        m_freqUpdated = freqUpdated;
    }

    void setFpsForTest( float fps )
    {
        m_fps = fps;
    }

    void setFreqForTest( const std::vector<float> &freq )
    {
        m_freq = freq;
    }

    void configurePublishedGainState( const std::vector<float> &gainCalFacts,
                                      const std::vector<float> &gainSIRaw,
                                      const std::vector<float> &gainSI,
                                      const std::vector<float> &gainMaxSI,
                                      const std::vector<float> &gainLP,
                                      const std::vector<float> &gainMaxLP,
                                      const std::vector<float> &gainCals,
                                      const std::vector<float> &modeVarOL,
                                      const std::vector<float> &modeVarSI,
                                      const std::vector<float> &modeVarLP,
                                      float opticalGain )
    {
        m_gainCalFacts = gainCalFacts;
        m_optGainSIRaw = gainSIRaw;
        m_optGainSI = gainSI;
        m_gmaxSI = gainMaxSI;
        m_optGainLP = gainLP;
        m_gmaxLP = gainMaxLP;
        m_gainCals = gainCals;
        m_modeVarOL = modeVarOL;
        m_modeVarSI = modeVarSI;
        m_modeVarLP = modeVarLP;
        m_opticalGain = opticalGain;
    }

    void writePublishedGainArraysForTest( float *currentData,
                                          float *siRawData,
                                          float *siData,
                                          float *maxSiData,
                                          float *lpData,
                                          float *maxLpData,
                                          float *modeVarData )
    {
        writePublishedGainArrays( currentData, siRawData, siData, maxSiData, lpData, maxLpData, modeVarData );
    }

    const std::vector<float> &integratedSiGainsForTest() const
    {
        return m_optGainSI;
    }

    void configureSiIntegratorStateForTest( const std::vector<float> &gainSIRaw,
                                            const std::vector<float> &gainSI,
                                            float gainGain,
                                            float gainLeak )
    {
        m_optGainSIRaw = gainSIRaw;
        m_optGainSI = gainSI;
        m_gainGain = gainGain;
        m_gainLeak = gainLeak;
    }

    void updateIntegratedSiGainForTest( size_t modeIndex )
    {
        updateIntegratedSiGain( modeIndex );
    }

    int requestZeroGainsForTest( bool on = true )
    {
        if( on )
        {
            std::fill( m_optGainSI.begin(), m_optGainSI.end(), 0.0F );
            m_siGainStateNeedsSync = false;
            m_zeroGains = true;
        }
        else
        {
            m_zeroGains = false;
        }

        return 0;
    }

    void initZeroGainsPropertyForTest()
    {
        createStandardIndiRequestSw( m_indiP_zeroGains, "zero_gains" );
    }

    void configurePublishedPredictorState( const std::vector<float> &gainCalFacts,
                                           const std::vector<float> &gainLP,
                                           const std::vector<float> &gainCals,
                                           const std::vector<uint32_t> &Na,
                                           const std::vector<uint32_t> &Nb,
                                           const std::vector<std::vector<float>> &aCoeff,
                                           const std::vector<std::vector<float>> &bCoeff,
                                           float opticalGain,
                                           float gainGain )
    {
        m_gainCalFacts = gainCalFacts;
        m_optGainLP = gainLP;
        m_gainCals = gainCals;
        m_Na = Na;
        m_Nb = Nb;
        m_opticalGain = opticalGain;
        m_gainGain = gainGain;

        m_goptLP.resize( aCoeff.size() );
        for( size_t n = 0; n < aCoeff.size(); ++n )
        {
            m_goptLP[n].a( aCoeff[n] );
            m_goptLP[n].b( bCoeff[n] );
        }
    }

    void writePublishedPredictorArraysForTest(
        float *pcGainData, float *aCoeffData, uint32_t aWidth, float *bCoeffData, uint32_t bWidth, bool blend )
    {
        writePublishedPredictorArrays( pcGainData, aCoeffData, aWidth, bCoeffData, bWidth, blend );
    }

    int countEnabledGainFactorsForTest( const std::vector<float> &gainFacts ) const
    {
        return countEnabledGainFactors( gainFacts );
    }

    void updateAppliedModeCountForTest( const std::vector<float> &gainFacts, bool predictorPath )
    {
        updateAppliedModeCount( gainFacts, predictorPath );
    }

    bool applyGainFactorUpdateForTest( std::vector<float> &gainFacts,
                                       const std::vector<float> &incoming,
                                       bool predictorPath )
    {
        return applyGainFactorUpdate( gainFacts, incoming.data(), incoming.size(), predictorPath );
    }

    bool applyMultiplierUpdateForTest( std::vector<float> &multFacts,
                                       const std::vector<float> &incoming,
                                       bool predictorPath )
    {
        return applyMultiplierUpdate( multFacts, incoming.data(), incoming.size(), predictorPath );
    }

    bool applyFrequencyUpdateForTest( const std::vector<float> &incoming )
    {
        return applyFrequencyUpdate( incoming.data(), incoming.size() );
    }

    void configureGoptStructureInputsForTest( const std::vector<float> &gainFacts,
                                              const std::vector<float> &taus,
                                              const std::vector<float> &multFacts,
                                              const std::vector<float> &freq )
    {
        m_gainFacts = gainFacts;
        m_taus = taus;
        m_multFacts = multFacts;
        m_freq = freq;
        m_gmaxSI.resize( gainFacts.size(), 0.0F );
    }

    size_t goptCurrentSize() const
    {
        return m_goptCurrent.size();
    }

    bool refreshGoptStructuresForTest()
    {
        std::lock_guard<std::mutex> lock( m_goptMutex );
        return refreshGoptStructures();
    }

    void initExtrapSelectionPropertiesForTest()
    {
        createStandardIndiSelectionSw( m_indiP_extrapMethod,
                                       "extrap_method",
                                       { olProcessMethodElement( c_olProcessNone ),
                                         olProcessMethodElement( c_olProcessLegacy ),
                                         olProcessMethodElement( c_olProcessPowerLawOnly ),
                                         olProcessMethodElement( c_olProcessMoffatPeaks ) },
                                       { olProcessMethodLabel( c_olProcessNone ),
                                         olProcessMethodLabel( c_olProcessLegacy ),
                                         olProcessMethodLabel( c_olProcessPowerLawOnly ),
                                         olProcessMethodLabel( c_olProcessMoffatPeaks ) },
                                       "Extrapolation Method",
                                       "Extrapolation" );

        createStandardIndiSelectionSw( m_indiP_extrapNoiseEstimateDomain,
                                       "extrap_noiseEstimateDomain",
                                       { extrapNoiseEstimateDomainElement( c_extrapNoiseEstimateOpenLoop ),
                                         extrapNoiseEstimateDomainElement( c_extrapNoiseEstimateClosedLoopPreXfer ) },
                                       { extrapNoiseEstimateDomainLabel( c_extrapNoiseEstimateOpenLoop ),
                                         extrapNoiseEstimateDomainLabel( c_extrapNoiseEstimateClosedLoopPreXfer ) },
                                       "Noise Estimate Domain",
                                       "Extrapolation" );

        createStandardIndiSelectionSw( m_indiP_extrapNoiseEstimateRange,
                                       "extrap_noiseEstimateRange",
                                       { extrapNoiseEstimateRangeElement( c_extrapNoiseEstimateHighFreq ),
                                         extrapNoiseEstimateRangeElement( c_extrapNoiseEstimateLowFreq ) },
                                       { extrapNoiseEstimateRangeLabel( c_extrapNoiseEstimateHighFreq ),
                                         extrapNoiseEstimateRangeLabel( c_extrapNoiseEstimateLowFreq ) },
                                       "Noise Estimate Range",
                                       "Extrapolation" );

        createStandardIndiSelectionSw( m_indiP_extrapNoiseEstimateStatistic,
                                       "extrap_noiseEstimateStatistic",
                                       { extrapNoiseEstimateStatisticElement( c_extrapNoiseEstimatePercentile ),
                                         extrapNoiseEstimateStatisticElement( c_extrapNoiseEstimateMinimum ) },
                                       { extrapNoiseEstimateStatisticLabel( c_extrapNoiseEstimatePercentile ),
                                         extrapNoiseEstimateStatisticLabel( c_extrapNoiseEstimateMinimum ) },
                                       "Noise Estimate Statistic",
                                       "Extrapolation" );

        createStandardIndiSelectionSw(
            m_indiP_extrapClosedLoopOlEstimateMethod,
            "extrap_closedLoopOlEstimateMethod",
            { extrapClosedLoopOlEstimateMethodElement( c_extrapClosedLoopOlEstimateEtfOnly ),
              extrapClosedLoopOlEstimateMethodElement( c_extrapClosedLoopOlEstimateNtfAware ) },
            { extrapClosedLoopOlEstimateMethodLabel( c_extrapClosedLoopOlEstimateEtfOnly ),
              extrapClosedLoopOlEstimateMethodLabel( c_extrapClosedLoopOlEstimateNtfAware ) },
            "Closed Loop OL Estimate Method",
            "Extrapolation" );

        createStandardIndiSelectionSw(
            m_indiP_extrapPowerLawCrossoverMode,
            "extrap_powerLawCrossoverMode",
            { extrapPowerLawCrossoverModeElement( c_extrapPowerLawCrossoverManual ),
              extrapPowerLawCrossoverModeElement( c_extrapPowerLawCrossoverAutoSmoothedCrossing ) },
            { extrapPowerLawCrossoverModeLabel( c_extrapPowerLawCrossoverManual ),
              extrapPowerLawCrossoverModeLabel( c_extrapPowerLawCrossoverAutoSmoothedCrossing ) },
            "Power-Law Crossover Mode",
            "Extrapolation" );
    }

    int handleExtrapMethodPropertyForTest( const pcf::IndiProperty &ipRecv )
    {
        return handleExtrapMethodProperty( ipRecv );
    }

    int handleExtrapNoiseEstimateDomainPropertyForTest( const pcf::IndiProperty &ipRecv )
    {
        return handleExtrapNoiseEstimateDomainProperty( ipRecv );
    }

    int handleExtrapNoiseEstimateRangePropertyForTest( const pcf::IndiProperty &ipRecv )
    {
        return handleExtrapNoiseEstimateRangeProperty( ipRecv );
    }

    int handleExtrapNoiseEstimateStatisticPropertyForTest( const pcf::IndiProperty &ipRecv )
    {
        return handleExtrapNoiseEstimateStatisticProperty( ipRecv );
    }

    int handleExtrapClosedLoopOlEstimateMethodPropertyForTest( const pcf::IndiProperty &ipRecv )
    {
        return handleExtrapClosedLoopOlEstimateMethodProperty( ipRecv );
    }

    int handleExtrapPowerLawCrossoverModePropertyForTest( const pcf::IndiProperty &ipRecv )
    {
        return handleExtrapPowerLawCrossoverModeProperty( ipRecv );
    }

    pcf::IndiElement::SwitchStateType extrapMethodElementStateForTest( const std::string &element ) const
    {
        return m_indiP_extrapMethod[element].getSwitchState();
    }

    pcf::IndiElement::SwitchStateType extrapNoiseEstimateDomainElementStateForTest( const std::string &element ) const
    {
        return m_indiP_extrapNoiseEstimateDomain[element].getSwitchState();
    }

    pcf::IndiElement::SwitchStateType extrapNoiseEstimateRangeElementStateForTest( const std::string &element ) const
    {
        return m_indiP_extrapNoiseEstimateRange[element].getSwitchState();
    }

    pcf::IndiElement::SwitchStateType
    extrapNoiseEstimateStatisticElementStateForTest( const std::string &element ) const
    {
        return m_indiP_extrapNoiseEstimateStatistic[element].getSwitchState();
    }

    pcf::IndiElement::SwitchStateType
    extrapClosedLoopOlEstimateMethodElementStateForTest( const std::string &element ) const
    {
        return m_indiP_extrapClosedLoopOlEstimateMethod[element].getSwitchState();
    }

    pcf::IndiElement::SwitchStateType extrapPowerLawCrossoverModeElementStateForTest( const std::string &element ) const
    {
        return m_indiP_extrapPowerLawCrossoverMode[element].getSwitchState();
    }
};
/// \endcond

/// Verify the placeholder modalGainOpt test harness instantiates the app
/// cleanly.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt placeholder harness instantiates the app", "[modalGainOpt]" )
{
    // clang-format off
    #ifdef MODALGAINOPT_TEST_DOXYGEN_REF
    modalGainOpt();
    #endif
    // clang-format on

    SECTION( "default construction succeeds" )
    {
        modalGainOpt app;

        REQUIRE( true );
    }

    SECTION( "OL process-method helpers map consistently" )
    {
        REQUIRE( olProcessMethodName( c_olProcessNone ) == "none" );
        REQUIRE( olProcessMethodName( c_olProcessLegacy ) == "legacy" );
        REQUIRE( olProcessMethodName( c_olProcessPowerLawOnly ) == "power-law-only" );
        REQUIRE( olProcessMethodName( c_olProcessMoffatPeaks ) == "moffat-peaks" );

        REQUIRE( olProcessMethodFromElement( "none" ) == c_olProcessNone );
        REQUIRE( olProcessMethodFromElement( "legacy" ) == c_olProcessLegacy );
        REQUIRE( olProcessMethodFromElement( "power_law_only" ) == c_olProcessPowerLawOnly );
        REQUIRE( olProcessMethodFromElement( "moffat_peaks" ) == c_olProcessMoffatPeaks );

        REQUIRE( olProcessMethodFromName( "none" ) == c_olProcessNone );
        REQUIRE( olProcessMethodFromName( "legacy" ) == c_olProcessLegacy );
        REQUIRE( olProcessMethodFromName( "power_law_only" ) == c_olProcessPowerLawOnly );
        REQUIRE( olProcessMethodFromName( "power-law-only" ) == c_olProcessPowerLawOnly );
        REQUIRE( olProcessMethodFromName( "moffat_peaks" ) == c_olProcessMoffatPeaks );
        REQUIRE( olProcessMethodFromName( "moffat-peaks" ) == c_olProcessMoffatPeaks );

        REQUIRE( extrapNoiseEstimateDomainName( c_extrapNoiseEstimateOpenLoop ) == "open-loop" );
        REQUIRE( extrapNoiseEstimateDomainName( c_extrapNoiseEstimateClosedLoopPreXfer ) == "closed-loop-pre-xfer" );

        REQUIRE( extrapNoiseEstimateDomainFromElement( "open_loop" ) == c_extrapNoiseEstimateOpenLoop );
        REQUIRE( extrapNoiseEstimateDomainFromElement( "closed_loop_pre_xfer" ) ==
                 c_extrapNoiseEstimateClosedLoopPreXfer );

        REQUIRE( extrapNoiseEstimateDomainFromName( "open_loop" ) == c_extrapNoiseEstimateOpenLoop );
        REQUIRE( extrapNoiseEstimateDomainFromName( "open-loop" ) == c_extrapNoiseEstimateOpenLoop );
        REQUIRE( extrapNoiseEstimateDomainFromName( "closed_loop_pre_xfer" ) ==
                 c_extrapNoiseEstimateClosedLoopPreXfer );
        REQUIRE( extrapNoiseEstimateDomainFromName( "closed-loop-pre-xfer" ) ==
                 c_extrapNoiseEstimateClosedLoopPreXfer );

        REQUIRE( extrapNoiseEstimateRangeName( c_extrapNoiseEstimateHighFreq ) == "high-freq" );
        REQUIRE( extrapNoiseEstimateRangeName( c_extrapNoiseEstimateLowFreq ) == "low-freq" );
        REQUIRE( extrapNoiseEstimateRangeFromElement( "high_freq" ) == c_extrapNoiseEstimateHighFreq );
        REQUIRE( extrapNoiseEstimateRangeFromElement( "low_freq" ) == c_extrapNoiseEstimateLowFreq );
        REQUIRE( extrapNoiseEstimateRangeFromName( "high_freq" ) == c_extrapNoiseEstimateHighFreq );
        REQUIRE( extrapNoiseEstimateRangeFromName( "high-freq" ) == c_extrapNoiseEstimateHighFreq );
        REQUIRE( extrapNoiseEstimateRangeFromName( "low_freq" ) == c_extrapNoiseEstimateLowFreq );
        REQUIRE( extrapNoiseEstimateRangeFromName( "low-freq" ) == c_extrapNoiseEstimateLowFreq );

        REQUIRE( extrapNoiseEstimateStatisticName( c_extrapNoiseEstimatePercentile ) == "percentile" );
        REQUIRE( extrapNoiseEstimateStatisticName( c_extrapNoiseEstimateMinimum ) == "minimum" );
        REQUIRE( extrapNoiseEstimateStatisticFromElement( "percentile" ) == c_extrapNoiseEstimatePercentile );
        REQUIRE( extrapNoiseEstimateStatisticFromElement( "minimum" ) == c_extrapNoiseEstimateMinimum );
        REQUIRE( extrapNoiseEstimateStatisticFromName( "percentile" ) == c_extrapNoiseEstimatePercentile );
        REQUIRE( extrapNoiseEstimateStatisticFromName( "minimum" ) == c_extrapNoiseEstimateMinimum );

        REQUIRE( extrapClosedLoopOlEstimateMethodName( c_extrapClosedLoopOlEstimateEtfOnly ) == "etf-only" );
        REQUIRE( extrapClosedLoopOlEstimateMethodName( c_extrapClosedLoopOlEstimateNtfAware ) == "ntf-aware" );
        REQUIRE( extrapClosedLoopOlEstimateMethodFromElement( "etf_only" ) == c_extrapClosedLoopOlEstimateEtfOnly );
        REQUIRE( extrapClosedLoopOlEstimateMethodFromElement( "ntf_aware" ) == c_extrapClosedLoopOlEstimateNtfAware );
        REQUIRE( extrapClosedLoopOlEstimateMethodFromName( "etf_only" ) == c_extrapClosedLoopOlEstimateEtfOnly );
        REQUIRE( extrapClosedLoopOlEstimateMethodFromName( "etf-only" ) == c_extrapClosedLoopOlEstimateEtfOnly );
        REQUIRE( extrapClosedLoopOlEstimateMethodFromName( "ntf_aware" ) == c_extrapClosedLoopOlEstimateNtfAware );
        REQUIRE( extrapClosedLoopOlEstimateMethodFromName( "ntf-aware" ) == c_extrapClosedLoopOlEstimateNtfAware );
    }
}

/// Verify `modalGainOpt::loadConfig()` applies the configured gain update
/// factor.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt configuration loads PSD-processing settings without "
           "toggling autoUpdate",
           "[modalGainOpt]" )
{
    modalGainOptHarness app;

    app.setupConfig();

    mx::app::writeConfigFile( "/tmp/modalGainOpt_test.conf",
                              { "loop",          "loop",          "loop",          "loop",          "loop",
                                "loop",          "extrapolation", "extrapolation", "extrapolation", "extrapolation",
                                "extrapolation", "extrapolation", "extrapolation", "extrapolation", "extrapolation",
                                "extrapolation", "extrapolation", "extrapolation", "extrapolation", "extrapolation",
                                "extrapolation", "extrapolation", "extrapolation", "extrapolation", "extrapolation",
                                "extrapolation", "extrapolation", "extrapolation", "extrapolation", "extrapolation",
                                "extrapolation", "extrapolation", "extrapolation", "extrapolation", "extrapolation",
                                "extrapolation", "extrapolation" },
                              {
                                  "number",
                                  "name",
                                  "autoUpdate",
                                  "gainGain",
                                  "gainLeak",
                                  "psdDev",
                                  "method",
                                  "noiseEstimateDomain",
                                  "noiseEstimateRange",
                                  "noiseEstimateStatistic",
                                  "noiseEstimateLowFreqMaxHz",
                                  "closedLoopOlEstimateMethod",
                                  "powerLawIndex",
                                  "powerLawNormFreq",
                                  "powerLawMatchFreq",
                                  "powerLawMatchFallbackWindowHz",
                                  "powerLawCrossoverMode",
                                  "powerLawAutoSmoothWidthHz",
                                  "powerLawAutoMaxFreqFraction",
                                  "fitPowerLawIndex",
                                  "powerLawOnlyAboveFreq",
                                  "powerLawFitIncludesMatchPoint",
                                  "powerLawFitMinFreqHz",
                                  "powerLawFitMaxFreqHz",
                                  "powerLawFitBinWidthHz",
                                  "powerLawBlendBins",
                                  "peakDetectWidthHz",
                                  "peakDetectFactor",
                                  "peakDetectBroadFactor",
                                  "peakDetectMinWidthLog",
                                  "peakDetectPasses",
                                  "peakMoffatBeta",
                                  "dropoutGapFactor",
                                  "dropoutTinyFactor",
                                  "dropoutMaxBins",
                                  "clSignificanceThreshold",
                                  "clMinSignificantFraction",
                              },
                              { "2",
                                "aol2",
                                "false",
                                "0.35",
                                "0.65",
                                "psdDevice",
                                "moffat_peaks",
                                "closed_loop_pre_xfer",
                                "low_freq",
                                "minimum",
                                "123",
                                "ntf_aware",
                                "1.5",
                                "15",
                                "12.5",
                                "7.5",
                                "auto_smoothed_crossing",
                                "37.5",
                                "0.4",
                                "true",
                                "250",
                                "false",
                                "100",
                                "900",
                                "80",
                                "6",
                                "55",
                                "4",
                                "2.5",
                                "0.03",
                                "3",
                                "8",
                                "0.12",
                                "0.000001",
                                "6",
                                "1.25",
                                "0.07" } );
    app.readConfigFile( "/tmp/modalGainOpt_test.conf" );

    app.loadConfig();
    // clang-format off
    #ifdef MODALGAINOPT_TEST_DOXYGEN_REF
    modalGainOpt::setupConfig();
    modalGainOpt::loadConfig();
    #endif
    // clang-format on

    REQUIRE( app.shutdownState() == 0 );
    REQUIRE( app.autoUpdate() == false );
    REQUIRE( app.gainGain() == Approx( 0.35F ) );
    REQUIRE( app.gainLeak() == Approx( 0.65F ) );
    REQUIRE( app.extrapMethod() == c_olProcessMoffatPeaks );
    REQUIRE( app.extrapNoiseEstimateDomain() == c_extrapNoiseEstimateClosedLoopPreXfer );
    REQUIRE( app.extrapNoiseEstimateRange() == c_extrapNoiseEstimateLowFreq );
    REQUIRE( app.extrapNoiseEstimateStatistic() == c_extrapNoiseEstimateMinimum );
    REQUIRE( app.extrapClosedLoopOlEstimateMethod() == c_extrapClosedLoopOlEstimateNtfAware );
    REQUIRE( app.extrapConfig().m_noiseEstimateDomain == "closed-loop-pre-xfer" );
    REQUIRE( app.extrapConfig().m_noiseEstimateRange == "low-freq" );
    REQUIRE( app.extrapConfig().m_noiseEstimateStatistic == "minimum" );
    REQUIRE( app.extrapConfig().m_noiseEstimateLowFreqMaxHz == Approx( 123.0F ) );
    REQUIRE( app.extrapConfig().m_closedLoopOlEstimateMethod == "ntf-aware" );
    REQUIRE( app.extrapConfig().m_powerLawIndex == Approx( 1.5F ) );
    REQUIRE( app.extrapConfig().m_powerLawNormFreq == Approx( 15.0F ) );
    REQUIRE( app.extrapConfig().m_powerLawMatchFreq == Approx( 12.5F ) );
    REQUIRE( app.extrapConfig().m_powerLawMatchFallbackWindowHz == Approx( 7.5F ) );
    REQUIRE( app.extrapConfig().m_powerLawCrossoverMode == "auto-smoothed-crossing" );
    REQUIRE( app.extrapConfig().m_powerLawAutoSmoothWidthHz == Approx( 37.5F ) );
    REQUIRE( app.extrapConfig().m_powerLawAutoMaxFreqFraction == Approx( 0.4F ) );
    REQUIRE( app.extrapConfig().m_fitPowerLawIndex == true );
    REQUIRE( app.extrapConfig().m_powerLawOnlyAboveFreq == Approx( 250.0F ) );
    REQUIRE( app.extrapConfig().m_powerLawFitIncludesMatchPoint == false );
    REQUIRE( app.extrapConfig().m_powerLawFitMinFreqHz == Approx( 100.0F ) );
    REQUIRE( app.extrapConfig().m_powerLawFitMaxFreqHz == Approx( 900.0F ) );
    REQUIRE( app.extrapConfig().m_powerLawFitBinWidthHz == Approx( 80.0F ) );
    REQUIRE( app.extrapConfig().m_powerLawBlendBins == 6 );
    REQUIRE( app.extrapConfig().m_peakDetectWidthHz == Approx( 55.0F ) );
    REQUIRE( app.extrapConfig().m_peakDetectFactor == Approx( 4.0F ) );
    REQUIRE( app.extrapConfig().m_peakDetectBroadFactor == Approx( 2.5F ) );
    REQUIRE( app.extrapConfig().m_peakDetectMinWidthLog == Approx( 0.03F ) );
    REQUIRE( app.extrapConfig().m_peakDetectPasses == 3 );
    REQUIRE( app.extrapConfig().m_peakMoffatBeta == Approx( 8.0F ) );
    REQUIRE( app.extrapConfig().m_dropoutGapFactor == Approx( 0.12F ) );
    REQUIRE( app.extrapConfig().m_dropoutTinyFactor == Approx( 1.0e-6F ) );
    REQUIRE( app.extrapConfig().m_dropoutMaxBins == 6 );
    REQUIRE( app.extrapConfig().m_clSignificanceThreshold == Approx( 1.25F ) );
    REQUIRE( app.extrapConfig().m_clMinSignificantFraction == Approx( 0.07F ) );
}

TEST_CASE( "modalGainOpt restores current selection when extrapolation switches "
           "receive all-off updates",
           "[modalGainOpt]" )
{
    modalGainOptHarness app;
    app.initExtrapSelectionPropertiesForTest();

    SECTION( "extrapolation method is restored" )
    {
        app.setExtrapMethodForTest( c_olProcessMoffatPeaks );

        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.add( pcf::IndiElement( olProcessMethodElement( c_olProcessNone ), pcf::IndiElement::Off ) );
        ip.add( pcf::IndiElement( olProcessMethodElement( c_olProcessLegacy ), pcf::IndiElement::Off ) );
        ip.add( pcf::IndiElement( olProcessMethodElement( c_olProcessPowerLawOnly ), pcf::IndiElement::Off ) );
        ip.add( pcf::IndiElement( olProcessMethodElement( c_olProcessMoffatPeaks ), pcf::IndiElement::Off ) );

        REQUIRE( app.handleExtrapMethodPropertyForTest( ip ) == 0 );
        REQUIRE( app.extrapMethod() == c_olProcessMoffatPeaks );
        REQUIRE( app.extrapMethodElementStateForTest( olProcessMethodElement( c_olProcessMoffatPeaks ) ) ==
                 pcf::IndiElement::On );
        REQUIRE( app.extrapMethodElementStateForTest( olProcessMethodElement( c_olProcessLegacy ) ) ==
                 pcf::IndiElement::Off );
    }

    SECTION( "noise-estimate domain is restored" )
    {
        app.setExtrapNoiseEstimateDomainForTest( c_extrapNoiseEstimateClosedLoopPreXfer );

        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.add( pcf::IndiElement( extrapNoiseEstimateDomainElement( c_extrapNoiseEstimateOpenLoop ),
                                  pcf::IndiElement::Off ) );
        ip.add( pcf::IndiElement( extrapNoiseEstimateDomainElement( c_extrapNoiseEstimateClosedLoopPreXfer ),
                                  pcf::IndiElement::Off ) );

        REQUIRE( app.handleExtrapNoiseEstimateDomainPropertyForTest( ip ) == 0 );
        REQUIRE( app.extrapNoiseEstimateDomain() == c_extrapNoiseEstimateClosedLoopPreXfer );
        REQUIRE( app.extrapConfig().m_noiseEstimateDomain == "closed-loop-pre-xfer" );
        REQUIRE( app.extrapNoiseEstimateDomainElementStateForTest( extrapNoiseEstimateDomainElement(
                     c_extrapNoiseEstimateClosedLoopPreXfer ) ) == pcf::IndiElement::On );
        REQUIRE( app.extrapNoiseEstimateDomainElementStateForTest(
                     extrapNoiseEstimateDomainElement( c_extrapNoiseEstimateOpenLoop ) ) == pcf::IndiElement::Off );
    }

    SECTION( "noise-estimate range is restored" )
    {
        app.setExtrapNoiseEstimateRangeForTest( c_extrapNoiseEstimateLowFreq );

        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.add( pcf::IndiElement( extrapNoiseEstimateRangeElement( c_extrapNoiseEstimateHighFreq ),
                                  pcf::IndiElement::Off ) );
        ip.add( pcf::IndiElement( extrapNoiseEstimateRangeElement( c_extrapNoiseEstimateLowFreq ),
                                  pcf::IndiElement::Off ) );

        REQUIRE( app.handleExtrapNoiseEstimateRangePropertyForTest( ip ) == 0 );
        REQUIRE( app.extrapNoiseEstimateRange() == c_extrapNoiseEstimateLowFreq );
        REQUIRE( app.extrapConfig().m_noiseEstimateRange == "low-freq" );
        REQUIRE( app.extrapNoiseEstimateRangeElementStateForTest(
                     extrapNoiseEstimateRangeElement( c_extrapNoiseEstimateLowFreq ) ) == pcf::IndiElement::On );
        REQUIRE( app.extrapNoiseEstimateRangeElementStateForTest(
                     extrapNoiseEstimateRangeElement( c_extrapNoiseEstimateHighFreq ) ) == pcf::IndiElement::Off );
    }

    SECTION( "noise-estimate statistic is restored" )
    {
        app.setExtrapNoiseEstimateStatisticForTest( c_extrapNoiseEstimateMinimum );

        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.add( pcf::IndiElement( extrapNoiseEstimateStatisticElement( c_extrapNoiseEstimatePercentile ),
                                  pcf::IndiElement::Off ) );
        ip.add( pcf::IndiElement( extrapNoiseEstimateStatisticElement( c_extrapNoiseEstimateMinimum ),
                                  pcf::IndiElement::Off ) );

        REQUIRE( app.handleExtrapNoiseEstimateStatisticPropertyForTest( ip ) == 0 );
        REQUIRE( app.extrapNoiseEstimateStatistic() == c_extrapNoiseEstimateMinimum );
        REQUIRE( app.extrapConfig().m_noiseEstimateStatistic == "minimum" );
        REQUIRE( app.extrapNoiseEstimateStatisticElementStateForTest(
                     extrapNoiseEstimateStatisticElement( c_extrapNoiseEstimateMinimum ) ) == pcf::IndiElement::On );
        REQUIRE( app.extrapNoiseEstimateStatisticElementStateForTest( extrapNoiseEstimateStatisticElement(
                     c_extrapNoiseEstimatePercentile ) ) == pcf::IndiElement::Off );
    }

    SECTION( "closed-loop OL estimate method is restored" )
    {
        app.setExtrapClosedLoopOlEstimateMethodForTest( c_extrapClosedLoopOlEstimateNtfAware );

        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.add( pcf::IndiElement( extrapClosedLoopOlEstimateMethodElement( c_extrapClosedLoopOlEstimateEtfOnly ),
                                  pcf::IndiElement::Off ) );
        ip.add( pcf::IndiElement( extrapClosedLoopOlEstimateMethodElement( c_extrapClosedLoopOlEstimateNtfAware ),
                                  pcf::IndiElement::Off ) );

        REQUIRE( app.handleExtrapClosedLoopOlEstimateMethodPropertyForTest( ip ) == 0 );
        REQUIRE( app.extrapClosedLoopOlEstimateMethod() == c_extrapClosedLoopOlEstimateNtfAware );
        REQUIRE( app.extrapConfig().m_closedLoopOlEstimateMethod == "ntf-aware" );
        REQUIRE( app.extrapClosedLoopOlEstimateMethodElementStateForTest( extrapClosedLoopOlEstimateMethodElement(
                     c_extrapClosedLoopOlEstimateNtfAware ) ) == pcf::IndiElement::On );
        REQUIRE( app.extrapClosedLoopOlEstimateMethodElementStateForTest( extrapClosedLoopOlEstimateMethodElement(
                     c_extrapClosedLoopOlEstimateEtfOnly ) ) == pcf::IndiElement::Off );
    }

    SECTION( "power-law crossover mode is restored" )
    {
        app.setExtrapPowerLawCrossoverModeForTest( c_extrapPowerLawCrossoverAutoSmoothedCrossing );

        pcf::IndiProperty ip( pcf::IndiProperty::Switch );
        ip.add( pcf::IndiElement( extrapPowerLawCrossoverModeElement( c_extrapPowerLawCrossoverManual ),
                                  pcf::IndiElement::Off ) );
        ip.add( pcf::IndiElement( extrapPowerLawCrossoverModeElement( c_extrapPowerLawCrossoverAutoSmoothedCrossing ),
                                  pcf::IndiElement::Off ) );

        REQUIRE( app.handleExtrapPowerLawCrossoverModePropertyForTest( ip ) == 0 );
        REQUIRE( app.extrapPowerLawCrossoverMode() == c_extrapPowerLawCrossoverAutoSmoothedCrossing );
        REQUIRE( app.extrapConfig().m_powerLawCrossoverMode == "auto-smoothed-crossing" );
        REQUIRE( app.extrapPowerLawCrossoverModeElementStateForTest( extrapPowerLawCrossoverModeElement(
                     c_extrapPowerLawCrossoverAutoSmoothedCrossing ) ) == pcf::IndiElement::On );
        REQUIRE( app.extrapPowerLawCrossoverModeElementStateForTest(
                     extrapPowerLawCrossoverModeElement( c_extrapPowerLawCrossoverManual ) ) == pcf::IndiElement::Off );
    }
}

TEST_CASE( "modalPsdProcessor falls back when the requested power-law fit has "
           "too little frequency span",
           "[modalGainOpt]" )
{
    processPsdProcessorT::processModelConfig cfg;
    cfg.m_method = "moffat-peaks";
    cfg.m_powerLawIndex = 1.5F;
    cfg.m_powerLawMatchFreq = 25.0F;
    cfg.m_fitPowerLawIndex = true;
    cfg.m_powerLawFitMinFreqHz = 25.0F;
    cfg.m_powerLawFitMaxFreqHz = 1000.0F;
    cfg.m_powerLawFitBinWidthHz = 100.0F;
    cfg.m_powerLawBlendBins = 5;

    std::vector<float> measuredPsd{ 1.0e-6F, 9.0e-7F, 8.0e-7F, 7.0e-7F, 6.0e-7F };
    std::vector<float> freq{ 0.0F, 25.0F, 50.0F, 75.0F, 100.0F };

    processPsdProcessorT::processResults result;
    mx::error_t errc = processPsdProcessorT::analyzePsd( result, measuredPsd, freq, 10, cfg, 0.0F, 25.0F );

    REQUIRE( !errc );
    REQUIRE( result.m_powerLawIndex == Approx( cfg.m_powerLawIndex ) );
    REQUIRE( result.m_powerLawIndexFitSucceeded == false );
    REQUIRE( result.m_powerLawFitBinsUsed == 0 );
}

TEST_CASE( "modalPsdProcessor can estimate noise in closed-loop space before OL "
           "correction",
           "[modalGainOpt]" )
{
    processPsdProcessorT::processModelConfig cfg;
    cfg.m_method = "legacy";
    cfg.m_noiseEstimateDomain = "closed_loop_pre_xfer";

    std::vector<float> measuredPsd{ 0.0F, 5.0F, 5.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F };
    std::vector<float> freq{ 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F };
    std::vector<float> correctionPsd( measuredPsd.size(), 0.5F );
    correctionPsd[0] = 1.0F;

    processPsdProcessorT::processResults result;
    mx::error_t errc =
        processPsdProcessorT::analyzePsd( result, measuredPsd, freq, 10, cfg, 0.0F, 25.0F, &correctionPsd );

    REQUIRE( !errc );
    REQUIRE( result.m_noiseEstimateDomain == "closed-loop-pre-xfer" );
    REQUIRE( result.m_noiseFloor == Approx( 1.0F ) );
    REQUIRE( result.m_noisePsd[1] == Approx( 1.0F ) );
    REQUIRE( result.m_processPsd[1] == Approx( 8.0F ) );
}

TEST_CASE( "modalPsdProcessor can estimate noise from the low-frequency end", "[modalGainOpt]" )
{
    std::vector<float> measuredPsd{ 0.0F, 2.0F, 2.0F, 20.0F, 20.0F, 20.0F, 20.0F, 20.0F };
    std::vector<float> freq{ 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F };
    std::vector<float> noisePsd;
    float noiseFloor = 0.0F;

    mx::error_t errc =
        processPsdProcessorT::estimateNoisePsd( noisePsd, noiseFloor, measuredPsd, freq, 10, "low_freq" );

    REQUIRE( !errc );
    REQUIRE( noiseFloor == Approx( 2.0F ) );
    REQUIRE( noisePsd[1] == Approx( 2.0F ) );
}

TEST_CASE( "modalPsdProcessor can estimate noise using the minimum PSD in range", "[modalGainOpt]" )
{
    std::vector<float> measuredPsd{ 0.0F, 8.0F, 4.0F, 2.0F, 20.0F, 20.0F, 20.0F, 20.0F };
    std::vector<float> freq{ 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F };
    std::vector<float> noisePsd;
    float noiseFloor = 0.0F;

    mx::error_t errc =
        processPsdProcessorT::estimateNoisePsd( noisePsd, noiseFloor, measuredPsd, freq, 10, "low_freq", "minimum" );

    REQUIRE( !errc );
    REQUIRE( noiseFloor == Approx( 2.0F ) );
    REQUIRE( noisePsd[1] == Approx( 2.0F ) );
}

TEST_CASE( "modalPsdProcessor can limit low-frequency noise estimation to a max "
           "frequency",
           "[modalGainOpt]" )
{
    std::vector<float> measuredPsd{ 0.0F, 20.0F, 20.0F, 2.0F, 2.0F, 20.0F, 20.0F, 20.0F, 20.0F, 20.0F };
    std::vector<float> freq{ 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    std::vector<float> noisePsd;
    float noiseFloor = 0.0F;

    mx::error_t errc = processPsdProcessorT::estimateNoisePsd( noisePsd,
                                                               noiseFloor,
                                                               measuredPsd,
                                                               freq,
                                                               10,
                                                               "low_freq",
                                                               "percentile",
                                                               2.1F );

    REQUIRE( !errc );
    REQUIRE( noiseFloor == Approx( 20.0F ) );
    REQUIRE( noisePsd[1] == Approx( 20.0F ) );
}

TEST_CASE( "modalPsdProcessor can reconstruct OL PSD with NTF-aware closed-loop "
           "noise subtraction",
           "[modalGainOpt]" )
{
    processPsdProcessorT::processModelConfig cfg;
    cfg.m_method = "legacy";
    cfg.m_noiseEstimateDomain = "closed_loop_pre_xfer";
    cfg.m_closedLoopOlEstimateMethod = "ntf_aware";

    std::vector<float> measuredPsd{ 0.0F, 10.0F, 10.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F };
    std::vector<float> freq{ 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F };
    std::vector<float> etfPsd( measuredPsd.size(), 0.5F );
    std::vector<float> ntfPsd( measuredPsd.size(), 2.0F );
    etfPsd[0] = 1.0F;
    ntfPsd[0] = 1.0F;

    processPsdProcessorT::processResults result;
    mx::error_t errc =
        processPsdProcessorT::analyzePsd( result, measuredPsd, freq, 10, cfg, 0.0F, 25.0F, &etfPsd, &ntfPsd );

    REQUIRE( !errc );
    REQUIRE( result.m_noiseEstimateDomain == "closed-loop-pre-xfer" );
    REQUIRE( result.m_closedLoopOlEstimateMethod == "ntf-aware" );
    REQUIRE( result.m_noiseFloor == Approx( 2.0F ) );
    REQUIRE( result.m_noisePsd[1] == Approx( 2.0F ) );
    REQUIRE( result.m_processPsd[1] == Approx( 12.0F ) );
}

TEST_CASE( "modalPsdProcessor fits closed-loop noise on raw CL PSD even when OL "
           "reconstruction is NTF-aware",
           "[modalGainOpt]" )
{
    processPsdProcessorT::processModelConfig cfg;
    cfg.m_method = "legacy";
    cfg.m_noiseEstimateDomain = "closed_loop_pre_xfer";
    cfg.m_noiseEstimateRange = "low_freq";
    cfg.m_noiseEstimateStatistic = "minimum";
    cfg.m_noiseEstimateLowFreqMaxHz = 3.1F;
    cfg.m_closedLoopOlEstimateMethod = "ntf_aware";

    std::vector<float> measuredPsd{ 0.0F, 20.0F, 18.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F };
    std::vector<float> freq{ 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F };
    std::vector<float> etfPsd( measuredPsd.size(), 0.5F );
    std::vector<float> ntfPsd( measuredPsd.size(), 4.0F );
    etfPsd[0] = 1.0F;
    ntfPsd[0] = 1.0F;

    processPsdProcessorT::processResults result;
    mx::error_t errc =
        processPsdProcessorT::analyzePsd( result, measuredPsd, freq, 10, cfg, 0.0F, 25.0F, &etfPsd, &ntfPsd );

    REQUIRE( !errc );
    REQUIRE( result.m_noiseFloor == Approx( 2.0F ) );
    REQUIRE( result.m_noisePsd[1] == Approx( 2.0F ) );
    REQUIRE( result.m_processPsd[1] == Approx( 24.0F ) );
}

TEST_CASE( "modalPsdProcessor power-law-only matches moffat handoff when forced "
           "to pure power law above a cutoff",
           "[modalGainOpt]" )
{
    std::vector<float> measuredPsd{ 0.0F, 11.0F, 9.0F, 7.0F, 5.0F, 4.0F, 3.4F, 3.0F, 2.8F, 2.6F, 2.4F };
    std::vector<float> freq{ 0.0F, 20.0F, 40.0F, 60.0F, 80.0F, 100.0F, 120.0F, 140.0F, 160.0F, 180.0F, 200.0F };

    processPsdProcessorT::processModelConfig powerCfg;
    powerCfg.m_method = "power-law-only";
    powerCfg.m_powerLawIndex = 1.0F;
    powerCfg.m_powerLawMatchFreq = 20.0F;
    powerCfg.m_powerLawOnlyAboveFreq = 100.0F;
    powerCfg.m_noiseEstimateStatistic = "minimum";

    processPsdProcessorT::processModelConfig moffatCfg = powerCfg;
    moffatCfg.m_method = "moffat-peaks";
    moffatCfg.m_peakDetectFactor = 1.0e6F;
    moffatCfg.m_peakDetectBroadFactor = 1.0e6F;

    processPsdProcessorT::processResults powerResult;
    processPsdProcessorT::processResults moffatResult;

    mx::error_t errc = processPsdProcessorT::analyzePsd( powerResult, measuredPsd, freq, 10, powerCfg );
    REQUIRE( !errc );

    errc = processPsdProcessorT::analyzePsd( moffatResult, measuredPsd, freq, 10, moffatCfg );
    REQUIRE( !errc );
    REQUIRE( moffatResult.m_peaks.empty() );
    REQUIRE( powerResult.m_processPsd.size() == moffatResult.m_processPsd.size() );
    for( size_t n = 0; n < powerResult.m_processPsd.size(); ++n )
    {
        REQUIRE( powerResult.m_processPsd[n] == Approx( moffatResult.m_processPsd[n] ) );
    }
}

TEST_CASE( "modalPsdProcessor power-law-only repairs deep dropouts below the "
           "pure-power-law cutoff",
           "[modalGainOpt]" )
{
    std::vector<float> measuredPsd{ 0.0F, 11.0F, 10.0F, 2.5F, 9.0F, 8.0F, 7.5F, 7.0F };
    std::vector<float> freq{ 0.0F, 20.0F, 40.0F, 60.0F, 80.0F, 100.0F, 120.0F, 140.0F };

    processPsdProcessorT::processModelConfig cfg;
    cfg.m_method = "power-law-only";
    cfg.m_noiseEstimateRange = "low_freq";
    cfg.m_noiseEstimateStatistic = "minimum";
    cfg.m_powerLawIndex = 1.0F;
    cfg.m_powerLawMatchFreq = 20.0F;
    cfg.m_powerLawOnlyAboveFreq = 100.0F;

    processPsdProcessorT::processResults result;
    mx::error_t errc = processPsdProcessorT::analyzePsd( result, measuredPsd, freq, 10, cfg );

    REQUIRE( !errc );
    REQUIRE( result.m_noiseFloor == Approx( 2.5F ) );
    REQUIRE( result.m_processPsd[3] > 1.0F );
    REQUIRE( result.m_processPsd[5] == Approx( result.m_extrapolation * pow( 20.0F / freq[5], 1.0F ) ) );
    REQUIRE( result.m_processPsd[6] == Approx( result.m_extrapolation * pow( 20.0F / freq[6], 1.0F ) ) );
    REQUIRE( result.m_processPsd[7] == Approx( result.m_extrapolation * pow( 20.0F / freq[7], 1.0F ) ) );
}

TEST_CASE( "modalPsdProcessor can auto-select the power-law crossover from a "
           "smoothed noise crossing",
           "[modalGainOpt]" )
{
    std::vector<float> measuredPsd{ 0.0F, 10.0F, 8.5F, 6.5F, 4.5F, 3.1F, 2.7F, 2.55F, 4.6F, 2.51F, 2.5F };
    std::vector<float> freq{ 0.0F, 20.0F, 40.0F, 60.0F, 80.0F, 100.0F, 120.0F, 140.0F, 160.0F, 180.0F, 200.0F };

    processPsdProcessorT::processModelConfig cfg;
    cfg.m_method = "power-law-only";
    cfg.m_noiseEstimateStatistic = "minimum";
    cfg.m_powerLawIndex = 1.0F;
    cfg.m_powerLawNormFreq = 20.0F;
    cfg.m_powerLawMatchFreq = 0.0F;
    cfg.m_powerLawOnlyAboveFreq = 0.0F;
    cfg.m_powerLawCrossoverMode = "auto-smoothed-crossing";
    cfg.m_powerLawAutoSmoothWidthHz = 100.0F;
    cfg.m_powerLawAutoMaxFreqFraction = 0.0F;

    processPsdProcessorT::processResults result;
    mx::error_t errc = processPsdProcessorT::analyzePsd( result, measuredPsd, freq, 10, cfg );

    REQUIRE( !errc );
    REQUIRE( result.m_powerLawCrossoverMode == "auto-smoothed-crossing" );
    REQUIRE( result.m_powerLawMatchFreq == Approx( result.m_powerLawOnlyAboveFreq ) );
    REQUIRE( result.m_powerLawMatchFreq > 80.0F );
    REQUIRE( result.m_powerLawMatchFreq < 150.0F );
    REQUIRE( result.m_processPsd[8] < static_cast<float>( measuredPsd[8] - result.m_noiseFloor ) );
}

TEST_CASE( "modalPsdProcessor falls back to the highest-frequency smoothed "
           "minimum when no noise crossing exists",
           "[modalGainOpt]" )
{
    std::vector<float> smoothedProcessPsd{ 5.0F, 4.0F, 2.0F, 1.5F, 1.5F };
    std::vector<float> noisePsd{ 1.0F, 1.0F, 1.0F, 1.0F, 1.0F };
    std::vector<float> freq{ 0.0F, 25.0F, 50.0F, 75.0F, 100.0F };

    float crossoverFreq = 0.0F;
    mx::error_t errc = processPsdProcessorHarness::findAutoPowerLawCrossoverFreq( crossoverFreq,
                                                                                  smoothedProcessPsd,
                                                                                  noisePsd,
                                                                                  freq,
                                                                                  0.0F );

    REQUIRE( !errc );
    REQUIRE( crossoverFreq == Approx( 100.0F ) );
}

TEST_CASE( "modalPsdProcessor treats a below-to-above sign change as a valid "
           "smoothed noise crossing",
           "[modalGainOpt]" )
{
    std::vector<float> smoothedProcessPsd{ 0.5F, 0.8F, 1.0F, 1.2F, 1.4F };
    std::vector<float> noisePsd{ 1.0F, 1.0F, 1.0F, 1.0F, 1.0F };
    std::vector<float> freq{ 0.0F, 25.0F, 50.0F, 75.0F, 100.0F };

    float crossoverFreq = 0.0F;
    mx::error_t errc = processPsdProcessorHarness::findAutoPowerLawCrossoverFreq( crossoverFreq,
                                                                                  smoothedProcessPsd,
                                                                                  noisePsd,
                                                                                  freq,
                                                                                  0.0F );

    REQUIRE( !errc );
    REQUIRE( crossoverFreq == Approx( 50.0F ) );
}

TEST_CASE( "modalPsdProcessor auto crossover can cap the search to a fraction "
           "of the sampled maximum frequency",
           "[modalGainOpt]" )
{
    std::vector<float> smoothedProcessPsd{ 5.0F, 4.0F, 1.5F, 0.8F, 0.7F, 1.4F, 1.6F };
    std::vector<float> noisePsd{ 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F };
    std::vector<float> freq{ 0.0F, 100.0F, 200.0F, 300.0F, 400.0F, 900.0F, 1000.0F };

    float crossoverFreq = 0.0F;
    mx::error_t errc = processPsdProcessorHarness::findAutoPowerLawCrossoverFreq( crossoverFreq,
                                                                                  smoothedProcessPsd,
                                                                                  noisePsd,
                                                                                  freq,
                                                                                  0.4F );

    REQUIRE( !errc );
    REQUIRE( crossoverFreq == Approx( 271.42856F ) );
}

TEST_CASE( "modalPsdProcessor snaps the effective auto crossover to the next "
           "sampled frequency bin",
           "[modalGainOpt]" )
{
    std::vector<float> rawProcessPsd{ 5.0F, 4.0F, 2.0F, 0.8F, 0.7F };
    std::vector<float> smoothedProcessPsd{ 5.0F, 4.0F, 2.0F, 0.8F, 0.7F };
    std::vector<float> noisePsd{ 1.0F, 1.0F, 1.0F, 1.0F, 1.0F };
    std::vector<float> freq{ 0.0F, 100.0F, 200.0F, 300.0F, 400.0F };

    float matchFreq = 0.0F;
    float onlyAboveFreq = 0.0F;
    mx::error_t errc = processPsdProcessorHarness::resolvePowerLawCrossoverFrequencies( matchFreq,
                                                                                        onlyAboveFreq,
                                                                                        rawProcessPsd,
                                                                                        smoothedProcessPsd,
                                                                                        noisePsd,
                                                                                        freq,
                                                                                        "auto-smoothed-crossing",
                                                                                        0.0F );

    REQUIRE( !errc );
    REQUIRE( matchFreq == Approx( 300.0F ) );
    REQUIRE( onlyAboveFreq == Approx( 300.0F ) );
}

TEST_CASE( "modalPsdProcessor anchors the power-law match to the smoothed "
           "disturbance PSD",
           "[modalGainOpt]" )
{
    std::vector<float> rawProcessPsd{ 1.0F, 1.0F, 100.0F, 0.5F, 0.25F };
    std::vector<float> smoothedProcessPsd{ 1.0F, 1.0F, 10.0F, 0.5F, 0.25F };
    std::vector<float> noisePsd{ 0.1F, 0.1F, 0.1F, 0.1F, 0.1F };
    std::vector<float> freq{ 0.0F, 10.0F, 20.0F, 30.0F, 40.0F };

    std::vector<float> continuumPsd;
    float extrapolation = 0.0F;
    size_t anchorIndex = 0;
    mx::error_t errc = processPsdProcessorHarness::estimatePowerLawContinuum( continuumPsd,
                                                                              extrapolation,
                                                                              anchorIndex,
                                                                              rawProcessPsd,
                                                                              smoothedProcessPsd,
                                                                              noisePsd,
                                                                              freq,
                                                                              1.0F,
                                                                              10.0F,
                                                                              20.0F,
                                                                              0.0F );

    REQUIRE( !errc );
    REQUIRE( anchorIndex >= 1 );
    REQUIRE( extrapolation == Approx( 20.0F ) );
}

TEST_CASE( "modalPsdProcessor power-law-only auto handoff matches the "
           "smoothed crossover exactly without a blend ramp",
           "[modalGainOpt]" )
{
    std::vector<float> measuredPsd{ 1.0F, 1.2F, 41.0F, 31.0F, 3.0F, 2.0F };
    std::vector<float> smoothedProcessPsd{ 1.0F, 45.0F, 35.0F, 25.0F, 2.0F, 1.5F };
    std::vector<float> noisePsd{ 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F };
    std::vector<float> freq{ 0.0F, 10.0F, 20.0F, 30.0F, 40.0F, 50.0F };

    processPsdProcessorT::processModelConfig cfg;
    cfg.m_method = "power-law-only";
    cfg.m_powerLawIndex = 1.0F;
    cfg.m_powerLawNormFreq = 10.0F;
    cfg.m_powerLawMatchFreq = 40.0F;
    cfg.m_powerLawOnlyAboveFreq = 40.0F;
    cfg.m_powerLawCrossoverMode = "auto-smoothed-crossing";
    cfg.m_powerLawMatchFallbackWindowHz = 5.0F;
    cfg.m_powerLawBlendBins = 20;

    std::vector<float> processPsd;
    float extrapolation = 0.0F;
    size_t anchorIndex = 0;
    std::vector<unsigned char> repairMask;
    float usedPowerLawIndex = 0.0F;
    size_t fitBinsUsed = 0;

    mx::error_t errc = processPsdProcessorHarness::estimateProcessPsdPowerLawOnly( processPsd,
                                                                                   extrapolation,
                                                                                   anchorIndex,
                                                                                   repairMask,
                                                                                   measuredPsd,
                                                                                   smoothedProcessPsd,
                                                                                   noisePsd,
                                                                                   freq,
                                                                                   cfg,
                                                                                   &usedPowerLawIndex,
                                                                                   &fitBinsUsed );

    REQUIRE( !errc );
    REQUIRE( usedPowerLawIndex == Approx( 1.0F ) );
    REQUIRE( processPsd[1] == Approx( measuredPsd[1] - noisePsd[1] ) );
    REQUIRE( processPsd[3] == Approx( measuredPsd[3] - noisePsd[3] ) );
    REQUIRE( processPsd[4] == Approx( smoothedProcessPsd[4] ) );
    REQUIRE( processPsd[5] == Approx( 1.6F ) );
}

TEST_CASE( "modalPsdProcessor repairs raw disturbance dropouts before "
           "extrapolation",
           "[modalGainOpt]" )
{
    std::vector<float> measuredPsd{ 0.0F, 11.0F, 10.0F, 1.2F, 9.0F, 8.0F, 1.0F };
    std::vector<float> freq{ 0.0F, 10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F };

    processPsdProcessorT::processModelConfig cfg;
    cfg.m_method = "power-law-only";
    cfg.m_noiseEstimateStatistic = "minimum";
    cfg.m_powerLawIndex = 1.0F;
    cfg.m_powerLawNormFreq = 10.0F;
    cfg.m_powerLawMatchFreq = 0.0F;
    cfg.m_powerLawOnlyAboveFreq = 0.0F;

    processPsdProcessorT::processResults result;
    mx::error_t errc = processPsdProcessorT::analyzePsd( result, measuredPsd, freq, 10, cfg );

    REQUIRE( !errc );
    REQUIRE( result.m_noiseFloor == Approx( 1.0F ) );
    REQUIRE( result.m_rawProcessPsd[3] > 0.2F );
}

TEST_CASE( "modalPsdProcessor repairs trailing high-frequency dropout runs", "[modalGainOpt]" )
{
    std::vector<float> processPsd{ 10.0F, 9.0F, 8.0F, 1.0e-8F, 1.0e-8F, 1.0e-8F };
    std::vector<float> freq{ 0.0F, 10.0F, 20.0F, 30.0F, 40.0F, 50.0F };

    mx::error_t errc =
        processPsdProcessorHarness::fillProcessPsdDropouts( processPsd, freq, {}, 0.2F, 1.0e-6F, 4, 1.0F );

    REQUIRE( !errc );
    REQUIRE( processPsd[3] > 1.0F );
    REQUIRE( processPsd[4] > 1.0F );
    REQUIRE( processPsd[5] > 1.0F );
    REQUIRE( processPsd[3] == Approx( 16.0F / 3.0F ) );
    REQUIRE( processPsd[4] == Approx( 4.0F ) );
    REQUIRE( processPsd[5] == Approx( 3.2F ) );
}

TEST_CASE( "modalPsdProcessor keeps trailing dropout repair bounded when the "
           "last good bins rise",
           "[modalGainOpt]" )
{
    std::vector<float> processPsd{ 1.0F, 2.0F, 4.0F, 1.0e-8F, 1.0e-8F, 1.0e-8F };
    std::vector<float> freq{ 0.0F, 10.0F, 20.0F, 30.0F, 40.0F, 50.0F };

    mx::error_t errc =
        processPsdProcessorHarness::fillProcessPsdDropouts( processPsd, freq, {}, 0.2F, 1.0e-6F, 4, 1.0F );

    REQUIRE( !errc );
    REQUIRE( processPsd[3] == Approx( 8.0F / 3.0F ) );
    REQUIRE( processPsd[4] == Approx( 2.0F ) );
    REQUIRE( processPsd[5] == Approx( 1.6F ) );
}

TEST_CASE( "modalPsdProcessor does not treat a sharp post-peak decline as a "
           "dropout unless it is truly tiny",
           "[modalGainOpt]" )
{
    std::vector<float> processPsd{ 1.0e-4F, 1.0e-6F, 1.0e-8F, 1.0e-8F, 1.0e-8F };
    std::vector<float> freq{ 0.0F, 10.0F, 20.0F, 30.0F, 40.0F };

    mx::error_t errc =
        processPsdProcessorHarness::fillProcessPsdDropouts( processPsd, freq, {}, 0.2F, 1.0e-6F, 4, 1.0F );

    REQUIRE( !errc );
    REQUIRE( processPsd[2] == Approx( 1.0e-8F ) );
    REQUIRE( processPsd[3] == Approx( 1.0e-8F ) );
    REQUIRE( processPsd[4] == Approx( 1.0e-8F ) );
}

/// Verify `modalGainOpt` publishes LP and max-gain arrays into separate
/// buffers.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt published gain arrays keep LP and max LP outputs distinct", "[modalGainOpt]" )
{
    modalGainOptHarness app;

    app.configurePublishedGainState( { 2.0F, 4.0F },
                                     { 29.0F, 31.0F },
                                     { 3.0F, 5.0F },
                                     { 7.0F, 11.0F },
                                     { 13.0F, 17.0F },
                                     { 19.0F, 23.0F },
                                     { 1.0F, 2.0F },
                                     { 0.1F, 0.2F },
                                     { 1.1F, 1.2F },
                                     { 2.1F, 2.2F },
                                     2.0F );

    std::vector<float> currentData( 2, -1.0F );
    std::vector<float> siRawData( 2, -1.0F );
    std::vector<float> siData( 2, -1.0F );
    std::vector<float> maxSiData( 2, -1.0F );
    std::vector<float> lpData( 2, -1.0F );
    std::vector<float> maxLpData( 2, -1.0F );
    std::vector<float> modeVarData( 6, -1.0F );

    app.writePublishedGainArraysForTest( currentData.data(),
                                         siRawData.data(),
                                         siData.data(),
                                         maxSiData.data(),
                                         lpData.data(),
                                         maxLpData.data(),
                                         modeVarData.data() );

    mx::improc::eigenMap<float> modeVars( modeVarData.data(), 3, 2 );

    REQUIRE( currentData[0] == Approx( 3.0F ) );
    REQUIRE( currentData[1] == Approx( 5.0F ) );
    REQUIRE( siRawData[0] == Approx( 29.0F ) );
    REQUIRE( siRawData[1] == Approx( 31.0F ) );
    REQUIRE( siData[0] == Approx( 3.0F ) );
    REQUIRE( siData[1] == Approx( 5.0F ) );
    REQUIRE( maxSiData[0] == Approx( 7.0F ) );
    REQUIRE( maxSiData[1] == Approx( 11.0F ) );
    REQUIRE( lpData[0] == Approx( 13.0F ) );
    REQUIRE( lpData[1] == Approx( 17.0F ) );
    REQUIRE( maxLpData[0] == Approx( 19.0F ) );
    REQUIRE( maxLpData[1] == Approx( 23.0F ) );
    REQUIRE( modeVars( 0, 0 ) == Approx( 0.1F ) );
    REQUIRE( modeVars( 1, 0 ) == Approx( 1.1F ) );
    REQUIRE( modeVars( 2, 0 ) == Approx( 2.1F ) );
    REQUIRE( modeVars( 0, 1 ) == Approx( 0.2F ) );
    REQUIRE( modeVars( 1, 1 ) == Approx( 1.2F ) );
    REQUIRE( modeVars( 2, 1 ) == Approx( 2.2F ) );
}

/// Verify `modalGainOpt` applies gain calibration and optical-gain scaling when
/// publishing gains.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt published gain arrays apply calibration scaling", "[modalGainOpt]" )
{
    modalGainOptHarness app;

    app.configurePublishedGainState( { 6.0F, 3.0F },
                                     { 2.0F, 10.0F },
                                     { 4.0F, 12.0F },
                                     { 8.0F, 18.0F },
                                     { 10.0F, 20.0F },
                                     { 14.0F, 24.0F },
                                     { 3.0F, 6.0F },
                                     { 0.4F, 0.8F },
                                     { 1.4F, 1.8F },
                                     { 2.4F, 2.8F },
                                     4.0F );

    std::vector<float> currentData( 2, -1.0F );
    std::vector<float> siRawData( 2, -1.0F );
    std::vector<float> siData( 2, -1.0F );
    std::vector<float> maxSiData( 2, -1.0F );
    std::vector<float> lpData( 2, -1.0F );
    std::vector<float> maxLpData( 2, -1.0F );
    std::vector<float> modeVarData( 6, -1.0F );

    app.writePublishedGainArraysForTest( currentData.data(),
                                         siRawData.data(),
                                         siData.data(),
                                         maxSiData.data(),
                                         lpData.data(),
                                         maxLpData.data(),
                                         modeVarData.data() );

    REQUIRE( currentData[0] == Approx( 2.0F ) );
    REQUIRE( currentData[1] == Approx( 1.5F ) );
    REQUIRE( siRawData[0] == Approx( 1.0F ) );
    REQUIRE( siRawData[1] == Approx( 1.25F ) );
    REQUIRE( siData[0] == Approx( 2.0F ) );
    REQUIRE( siData[1] == Approx( 1.5F ) );
    REQUIRE( maxSiData[0] == Approx( 4.0F ) );
    REQUIRE( maxSiData[1] == Approx( 2.25F ) );
    REQUIRE( lpData[0] == Approx( 5.0F ) );
    REQUIRE( lpData[1] == Approx( 2.5F ) );
    REQUIRE( maxLpData[0] == Approx( 7.0F ) );
    REQUIRE( maxLpData[1] == Approx( 3.0F ) );
}

TEST_CASE( "modalGainOpt zero_gains request resets the integrated SI gains", "[modalGainOpt]" )
{
    modalGainOptHarness app;

    app.configurePublishedGainState( { 2.0F, 4.0F },
                                     { 29.0F, 31.0F },
                                     { 3.0F, 5.0F },
                                     { 7.0F, 11.0F },
                                     { 13.0F, 17.0F },
                                     { 19.0F, 23.0F },
                                     { 1.0F, 2.0F },
                                     { 0.1F, 0.2F },
                                     { 1.1F, 1.2F },
                                     { 2.1F, 2.2F },
                                     2.0F );
    app.initZeroGainsPropertyForTest();

    REQUIRE( app.integratedSiGainsForTest()[0] == Approx( 3.0F ) );
    REQUIRE( app.integratedSiGainsForTest()[1] == Approx( 5.0F ) );

    REQUIRE( app.requestZeroGainsForTest() == 0 );
    REQUIRE( app.integratedSiGainsForTest()[0] == Approx( 0.0F ) );
    REQUIRE( app.integratedSiGainsForTest()[1] == Approx( 0.0F ) );
}

TEST_CASE( "modalGainOpt SI gain integrator updates toward the raw optimum by delta", "[modalGainOpt]" )
{
    modalGainOptHarness app;

    app.configureSiIntegratorStateForTest( { 10.0F, 3.0F }, { 4.0F, 1.0F }, 0.2F, 0.9F );

    app.updateIntegratedSiGainForTest( 0 );
    app.updateIntegratedSiGainForTest( 1 );

    REQUIRE( app.integratedSiGainsForTest()[0] == Approx( 4.8F ) );
    REQUIRE( app.integratedSiGainsForTest()[1] == Approx( 1.3F ) );
}

/// Verify `modalGainOpt` counts enabled modes from positive gain factors.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt counts enabled gain-factor modes using positive entries", "[modalGainOpt]" )
{
    modalGainOptHarness app;

    // clang-format off
    #ifdef MODALGAINOPT_TEST_DOXYGEN_REF
    modalGainOpt::countEnabledGainFactors( std::vector<float>() );
    #endif
    // clang-format on

    REQUIRE( app.countEnabledGainFactorsForTest( {} ) == 0 );
    REQUIRE( app.countEnabledGainFactorsForTest( { -0.2F, 0.0F, 0.1F, 2.0F, -3.0F } ) == 2 );
    REQUIRE( app.countEnabledGainFactorsForTest( { 1.0F, 0.5F, 0.25F } ) == 3 );
}

/// Verify `modalGainOpt` ignores unchanged gain-factor frames.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt gain-factor updates leave state unchanged when the "
           "frame is identical",
           "[modalGainOpt]" )
{
    modalGainOptHarness app;

    std::vector<float> storedGainFacts( { 1.0F, 0.5F, 0.0F } );

    app.setLoopForTest( true );
    app.setPcOnForTest( false );
    app.setModesOnForTest( 7 );
    app.setSinceChangeForTest( 12 );

    // clang-format off
    #ifdef MODALGAINOPT_TEST_DOXYGEN_REF
    modalGainOpt::applyGainFactorUpdate( storedGainFacts, static_cast<const float *>( nullptr ), 0, false );
    #endif
    // clang-format on

    REQUIRE( app.applyGainFactorUpdateForTest( storedGainFacts, { 1.0F, 0.5F, 0.0F }, false ) == false );
    REQUIRE( storedGainFacts == std::vector<float>( { 1.0F, 0.5F, 0.0F } ) );
    REQUIRE( app.modesOn() == 7 );
    REQUIRE( app.sinceChange() == 12 );
}

/// Verify `modalGainOpt` resizes and copies SI gain-factor frames while
/// resetting the loop debounce timer.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt gain-factor updates resize SI state and reset "
           "sinceChange on change",
           "[modalGainOpt]" )
{
    modalGainOptHarness app;

    std::vector<float> storedGainFacts( { 9.0F } );

    app.setLoopForTest( true );
    app.setPcOnForTest( false );
    app.setModesOnForTest( 0 );
    app.setSinceChangeForTest( 5 );

    REQUIRE( app.applyGainFactorUpdateForTest( storedGainFacts, { 0.0F, 1.5F, -2.0F, 3.0F }, false ) == true );
    REQUIRE( storedGainFacts == std::vector<float>( { 0.0F, 1.5F, -2.0F, 3.0F } ) );
    REQUIRE( app.modesOn() == 2 );
    REQUIRE( app.sinceChange() == -1 );
}

/// Verify `modalGainOpt` ignores unchanged multiplier frames.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt multiplier updates leave state unchanged when the "
           "frame is identical",
           "[modalGainOpt]" )
{
    modalGainOptHarness app;

    std::vector<float> storedMultFacts( { 0.25F, 0.5F, 0.75F } );

    app.setLoopForTest( true );
    app.setSinceChangeForTest( 14 );
    app.setGoptUpdatedForTest( false );
    app.setPcgoptUpdatedForTest( false );

    // clang-format off
    #ifdef MODALGAINOPT_TEST_DOXYGEN_REF
    modalGainOpt::applyMultiplierUpdate( storedMultFacts, static_cast<const float *>( nullptr ), 0, false );
    #endif
    // clang-format on

    REQUIRE( app.applyMultiplierUpdateForTest( storedMultFacts, { 0.25F, 0.5F, 0.75F }, false ) == false );
    REQUIRE( storedMultFacts == std::vector<float>( { 0.25F, 0.5F, 0.75F } ) );
    REQUIRE( app.sinceChange() == 14 );
    REQUIRE( app.goptUpdated() == false );
    REQUIRE( app.pcgoptUpdated() == false );
}

/// Verify `modalGainOpt` marks SI optimizer state dirty when multiplier frames
/// change.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt SI multiplier updates set goptUpdated and reset sinceChange", "[modalGainOpt]" )
{
    modalGainOptHarness app;

    std::vector<float> storedMultFacts( { 1.0F } );

    app.setLoopForTest( true );
    app.setSinceChangeForTest( 6 );
    app.setGoptUpdatedForTest( false );
    app.setPcgoptUpdatedForTest( false );

    REQUIRE( app.applyMultiplierUpdateForTest( storedMultFacts, { 1.5F, 2.5F }, false ) == true );
    REQUIRE( storedMultFacts == std::vector<float>( { 1.5F, 2.5F } ) );
    REQUIRE( app.sinceChange() == -1 );
    REQUIRE( app.goptUpdated() == true );
    REQUIRE( app.pcgoptUpdated() == false );
}

/// Verify `modalGainOpt` ignores unchanged frequency frames.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt frequency updates leave state unchanged when the frame "
           "is identical",
           "[modalGainOpt]" )
{
    modalGainOptHarness app;

    app.setFreqForTest( { 10.0F, 20.0F, 30.0F } );
    app.setFpsForTest( 60.0F );
    app.setSinceChangeForTest( 8 );
    app.setGoptUpdatedForTest( false );
    app.setFreqUpdatedForTest( false );

    // clang-format off
    #ifdef MODALGAINOPT_TEST_DOXYGEN_REF
    modalGainOpt::applyFrequencyUpdate( static_cast<const float *>( nullptr ), 0 );
    #endif
    // clang-format on

    REQUIRE( app.applyFrequencyUpdateForTest( { 10.0F, 20.0F, 30.0F } ) == false );
    REQUIRE( app.freq() == std::vector<float>( { 10.0F, 20.0F, 30.0F } ) );
    REQUIRE( app.fps() == Approx( 60.0F ) );
    REQUIRE( app.sinceChange() == 8 );
    REQUIRE( app.goptUpdated() == false );
    REQUIRE( app.freqUpdated() == false );
}

/// Verify `modalGainOpt` resizes and copies changed frequency frames while
/// updating derived state.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt frequency updates resize state and refresh derived timing", "[modalGainOpt]" )
{
    modalGainOptHarness app;

    app.setFreqForTest( { 5.0F } );
    app.setFpsForTest( 10.0F );
    app.setSinceChangeForTest( 4 );
    app.setGoptUpdatedForTest( false );
    app.setFreqUpdatedForTest( false );

    REQUIRE( app.applyFrequencyUpdateForTest( { 12.5F, 25.0F, 40.0F } ) == true );
    REQUIRE( app.freq() == std::vector<float>( { 12.5F, 25.0F, 40.0F } ) );
    REQUIRE( app.fps() == Approx( 80.0F ) );
    REQUIRE( app.sinceChange() == -1 );
    REQUIRE( app.goptUpdated() == true );
    REQUIRE( app.freqUpdated() == true );
}

/// Verify `modalGainOpt` can refresh gain-optimization structures as soon as
/// metadata changes, without waiting for a PSD-triggered semaphore post.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt refreshes pending gopt structures without a PSD wakeup", "[modalGainOpt]" )
{
    modalGainOptHarness app;

    // clang-format off
    #ifdef MODALGAINOPT_TEST_DOXYGEN_REF
    modalGainOpt::refreshGoptStructures();
    #endif
    // clang-format on

    app.setFpsForTest( 1000.0F );
    app.setFreqForTest( { 100.0F, 200.0F, 300.0F } );
    app.configureGoptStructureInputsForTest( { 1.0F, 2.0F }, { 0.001F, 0.002F }, { 0.5F, 0.75F }, app.freq() );
    app.setFreqUpdatedForTest( true );
    app.setGoptUpdatedForTest( true );
    app.setPcgoptUpdatedForTest( false );
    app.setPcOnForTest( false );

    REQUIRE( app.goptCurrentSize() == 0 );
    REQUIRE( app.refreshGoptStructuresForTest() == true );
    REQUIRE( app.goptCurrentSize() == 2 );
    REQUIRE( app.goptUpdated() == false );
    REQUIRE( app.pcgoptUpdated() == false );
    REQUIRE( app.freqUpdated() == false );
}

/// Verify `modalGainOpt` writes predictive-control coefficients into per-mode
/// blocks.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt predictor publication preserves per-mode coefficient layout", "[modalGainOpt]" )
{
    modalGainOptHarness app;

    app.configurePublishedPredictorState( { 2.0F, 4.0F },
                                          { 3.0F, 5.0F },
                                          { 1.0F, 2.0F },
                                          { 2U, 1U },
                                          { 1U, 2U },
                                          { { 0.1F, 0.2F }, { 0.4F } },
                                          { { 0.3F }, { 0.5F, 0.6F } },
                                          2.0F,
                                          0.5F );

    std::vector<float> pcGainData( 2, -1.0F );
    std::vector<float> aCoeffData( 8, -1.0F );
    std::vector<float> bCoeffData( 8, -1.0F );

    app.writePublishedPredictorArraysForTest( pcGainData.data(), aCoeffData.data(), 4, bCoeffData.data(), 4, false );

    REQUIRE( pcGainData[0] == Approx( 3.0F ) );
    REQUIRE( pcGainData[1] == Approx( 5.0F ) );

    REQUIRE( aCoeffData[0] == Approx( 2.0F ) );
    REQUIRE( aCoeffData[1] == Approx( 0.1F ) );
    REQUIRE( aCoeffData[2] == Approx( 0.2F ) );
    REQUIRE( aCoeffData[3] == Approx( 0.0F ) );
    REQUIRE( aCoeffData[4] == Approx( 1.0F ) );
    REQUIRE( aCoeffData[5] == Approx( 0.4F ) );
    REQUIRE( aCoeffData[6] == Approx( 0.0F ) );
    REQUIRE( aCoeffData[7] == Approx( 0.0F ) );

    REQUIRE( bCoeffData[0] == Approx( 1.0F ) );
    REQUIRE( bCoeffData[1] == Approx( 0.3F ) );
    REQUIRE( bCoeffData[2] == Approx( 0.0F ) );
    REQUIRE( bCoeffData[3] == Approx( 0.0F ) );
    REQUIRE( bCoeffData[4] == Approx( 2.0F ) );
    REQUIRE( bCoeffData[5] == Approx( 0.5F ) );
    REQUIRE( bCoeffData[6] == Approx( 0.6F ) );
    REQUIRE( bCoeffData[7] == Approx( 0.0F ) );
}

/// Verify `modalGainOpt` blends predictive-control gains and coefficients
/// against existing outputs.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt predictor publication blends existing values and "
           "clears stale coefficients",
           "[modalGainOpt]" )
{
    modalGainOptHarness app;

    app.configurePublishedPredictorState( { 2.0F, 4.0F },
                                          { 3.0F, 5.0F },
                                          { 1.0F, 2.0F },
                                          { 2U, 1U },
                                          { 1U, 2U },
                                          { { 0.1F, 0.5F }, { 0.9F } },
                                          { { 0.2F }, { 0.6F, 1.0F } },
                                          2.0F,
                                          0.25F );

    std::vector<float> pcGainData( { 1.0F, 9.0F } );
    std::vector<float> aCoeffData( { 9.0F, 1.0F, 2.0F, 3.0F, 8.0F, 4.0F, 5.0F, 6.0F } );
    std::vector<float> bCoeffData( { 7.0F, 1.0F, 2.0F, 3.0F, 6.0F, 4.0F, 5.0F, 6.0F } );

    app.writePublishedPredictorArraysForTest( pcGainData.data(), aCoeffData.data(), 4, bCoeffData.data(), 4, true );

    REQUIRE( pcGainData[0] == Approx( 1.5F ) );
    REQUIRE( pcGainData[1] == Approx( 8.0F ) );

    REQUIRE( aCoeffData[0] == Approx( 2.0F ) );
    REQUIRE( aCoeffData[1] == Approx( 0.775F ) );
    REQUIRE( aCoeffData[2] == Approx( 1.625F ) );
    REQUIRE( aCoeffData[3] == Approx( 0.0F ) );
    REQUIRE( aCoeffData[4] == Approx( 1.0F ) );
    REQUIRE( aCoeffData[5] == Approx( 3.225F ) );
    REQUIRE( aCoeffData[6] == Approx( 0.0F ) );
    REQUIRE( aCoeffData[7] == Approx( 0.0F ) );

    REQUIRE( bCoeffData[0] == Approx( 1.0F ) );
    REQUIRE( bCoeffData[1] == Approx( 0.8F ) );
    REQUIRE( bCoeffData[2] == Approx( 0.0F ) );
    REQUIRE( bCoeffData[3] == Approx( 0.0F ) );
    REQUIRE( bCoeffData[4] == Approx( 2.0F ) );
    REQUIRE( bCoeffData[5] == Approx( 3.15F ) );
    REQUIRE( bCoeffData[6] == Approx( 4.0F ) );
    REQUIRE( bCoeffData[7] == Approx( 0.0F ) );
}

/// Verify `modalGainOpt` only applies SI gain-factor mode counts when the SI
/// path is active.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt SI mode counts update only while predictor control is off", "[modalGainOpt]" )
{
    modalGainOptHarness app;

    // clang-format off
    #ifdef MODALGAINOPT_TEST_DOXYGEN_REF
    modalGainOpt::updateAppliedModeCount( std::vector<float>(), false );
    #endif
    // clang-format on

    app.setPcOnForTest( false );
    app.updateAppliedModeCountForTest( { 1.0F, 0.0F, -1.0F, 2.0F }, false );
    REQUIRE( app.modesOn() == 2 );

    app.setPcOnForTest( true );
    app.updateAppliedModeCountForTest( { 5.0F, 4.0F, 3.0F }, false );
    REQUIRE( app.modesOn() == 2 );
}

/// Verify `modalGainOpt` clears stale predictor coefficients when a mode
/// publishes zero-order predictors.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt predictor publication clears stale coefficient blocks "
           "for zero-order modes",
           "[modalGainOpt]" )
{
    modalGainOptHarness app;

    app.configurePublishedPredictorState( { 8.0F, 2.0F },
                                          { 4.0F, 6.0F },
                                          { 4.0F, 2.0F },
                                          { 0U, 0U },
                                          { 0U, 0U },
                                          { {}, {} },
                                          { {}, {} },
                                          2.0F,
                                          0.5F );

    std::vector<float> pcGainData( { 9.0F, 10.0F } );
    std::vector<float> aCoeffData( { 7.0F, 1.0F, 2.0F, 3.0F, 6.0F, 4.0F, 5.0F, 6.0F } );
    std::vector<float> bCoeffData( { 5.0F, 7.0F, 8.0F, 9.0F, 4.0F, 10.0F, 11.0F, 12.0F } );

    app.writePublishedPredictorArraysForTest( pcGainData.data(), aCoeffData.data(), 4, bCoeffData.data(), 4, false );

    REQUIRE( pcGainData[0] == Approx( 4.0F ) );
    REQUIRE( pcGainData[1] == Approx( 3.0F ) );

    REQUIRE( aCoeffData[0] == Approx( 0.0F ) );
    REQUIRE( aCoeffData[1] == Approx( 0.0F ) );
    REQUIRE( aCoeffData[2] == Approx( 0.0F ) );
    REQUIRE( aCoeffData[3] == Approx( 0.0F ) );
    REQUIRE( aCoeffData[4] == Approx( 0.0F ) );
    REQUIRE( aCoeffData[5] == Approx( 0.0F ) );
    REQUIRE( aCoeffData[6] == Approx( 0.0F ) );
    REQUIRE( aCoeffData[7] == Approx( 0.0F ) );

    REQUIRE( bCoeffData[0] == Approx( 0.0F ) );
    REQUIRE( bCoeffData[1] == Approx( 0.0F ) );
    REQUIRE( bCoeffData[2] == Approx( 0.0F ) );
    REQUIRE( bCoeffData[3] == Approx( 0.0F ) );
    REQUIRE( bCoeffData[4] == Approx( 0.0F ) );
    REQUIRE( bCoeffData[5] == Approx( 0.0F ) );
    REQUIRE( bCoeffData[6] == Approx( 0.0F ) );
    REQUIRE( bCoeffData[7] == Approx( 0.0F ) );
}

/// Verify `modalGainOpt` only applies PC gain-factor mode counts when predictor
/// control is on.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt PC mode counts update only while predictor control is on", "[modalGainOpt]" )
{
    modalGainOptHarness app;

    app.setPcOnForTest( false );
    app.updateAppliedModeCountForTest( { 1.0F, 2.0F, 3.0F }, true );
    REQUIRE( app.modesOn() == 0 );

    app.setPcOnForTest( true );
    app.updateAppliedModeCountForTest( { -1.0F, 0.25F, 0.0F, 0.75F }, true );
    REQUIRE( app.modesOn() == 2 );
}

/// Verify `modalGainOpt` updates stored PC gain factors without disturbing
/// SI-applied mode counts when predictor control is off.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt PC gain-factor updates preserve applied mode counts "
           "while predictor control is off",
           "[modalGainOpt]" )
{
    modalGainOptHarness app;

    std::vector<float> storedPcGainFacts( { 0.5F, 0.5F } );

    app.setLoopForTest( false );
    app.setPcOnForTest( false );
    app.setModesOnForTest( 3 );
    app.setSinceChangeForTest( 9 );

    REQUIRE( app.applyGainFactorUpdateForTest( storedPcGainFacts, { 1.0F, 0.0F, 2.0F }, true ) == true );
    REQUIRE( storedPcGainFacts == std::vector<float>( { 1.0F, 0.0F, 2.0F } ) );
    REQUIRE( app.modesOn() == 3 );
    REQUIRE( app.sinceChange() == 9 );
}

/// Verify `modalGainOpt` marks predictive-control optimizer state dirty when PC
/// multiplier frames change.
/**
 * \ingroup modalGainOpt_unit_test
 */
TEST_CASE( "modalGainOpt PC multiplier updates set pcgoptUpdated without "
           "touching SI optimizer flags",
           "[modalGainOpt]" )
{
    modalGainOptHarness app;

    std::vector<float> storedPcMultFacts( { 0.1F, 0.2F } );

    app.setLoopForTest( false );
    app.setSinceChangeForTest( 11 );
    app.setGoptUpdatedForTest( false );
    app.setPcgoptUpdatedForTest( false );

    REQUIRE( app.applyMultiplierUpdateForTest( storedPcMultFacts, { 0.3F, 0.4F, 0.5F }, true ) == true );
    REQUIRE( storedPcMultFacts == std::vector<float>( { 0.3F, 0.4F, 0.5F } ) );
    REQUIRE( app.sinceChange() == 11 );
    REQUIRE( app.goptUpdated() == false );
    REQUIRE( app.pcgoptUpdated() == true );
}

} // namespace modalGainOptTest

} // namespace libXWCTest
