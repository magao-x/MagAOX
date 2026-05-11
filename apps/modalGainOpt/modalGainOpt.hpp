/** \file modalGainOpt.hpp
 * \brief The MagAO-X PSD-based gain optimizer header file
 *
 * \ingroup modalGainOpt_files
 */

#ifndef modalGainOpt_hpp
#define modalGainOpt_hpp

#include <algorithm>
#include <atomic>
#include <cctype>

#include <mx/ao/analysis/clAOLinearPredictor.hpp>
#include <mx/ao/analysis/clGainOpt.hpp>
#include <mx/mxException.hpp>

#include "modalPsdProcessor.hpp"

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup modalGainOpt
 * \brief The MagAO-X application to perform PSD-based gain optimization
 *
 * <a href="../handbook/operating/software/apps/modalGainOpt.html">Application
 * Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup modalGainOpt_files
 * \ingroup modalGainOpt
 */

namespace MagAOX
{
namespace app
{

#define MGO_BREADCRUMB

// #define MGO_BREADCRUMB  std::cerr << __FILE__ << ' ' << __LINE__ << '\n';

typedef modalPsdProcessor<float> processPsdProcessorT;

static constexpr int c_olProcessNone = 0;
static constexpr int c_olProcessLegacy = 1;
static constexpr int c_olProcessPowerLawOnly = 2;
static constexpr int c_olProcessMoffatPeaks = 3;

static constexpr int c_extrapNoiseEstimateOpenLoop = 0;
static constexpr int c_extrapNoiseEstimateClosedLoopPreXfer = 1;
static constexpr int c_extrapNoiseEstimateHighFreq = 0;
static constexpr int c_extrapNoiseEstimateLowFreq = 1;
static constexpr int c_extrapNoiseEstimatePercentile = 0;
static constexpr int c_extrapNoiseEstimateMinimum = 1;
static constexpr int c_extrapClosedLoopOlEstimateEtfOnly = 0;
static constexpr int c_extrapClosedLoopOlEstimateNtfAware = 1;
static constexpr int c_extrapPowerLawCrossoverManual = 0;
static constexpr int c_extrapPowerLawCrossoverAutoSmoothedCrossing = 1;

inline std::string olProcessMethodElement( int method )
{
    switch( method )
    {
        case c_olProcessLegacy:
            return "legacy";
        case c_olProcessPowerLawOnly:
            return "power_law_only";
        case c_olProcessMoffatPeaks:
            return "moffat_peaks";
        case c_olProcessNone:
        default:
            return "none";
    }
}

inline std::string olProcessMethodLabel( int method )
{
    switch( method )
    {
        case c_olProcessLegacy:
            return "Legacy";
        case c_olProcessPowerLawOnly:
            return "Power Law Only";
        case c_olProcessMoffatPeaks:
            return "Moffat Peaks";
        case c_olProcessNone:
        default:
            return "None";
    }
}

inline std::string olProcessMethodName( int method )
{
    switch( method )
    {
        case c_olProcessLegacy:
            return "legacy";
        case c_olProcessPowerLawOnly:
            return "power-law-only";
        case c_olProcessMoffatPeaks:
            return "moffat-peaks";
        case c_olProcessNone:
        default:
            return "none";
    }
}

inline int olProcessMethodFromElement( const std::string &element )
{
    if( element == "legacy" )
    {
        return c_olProcessLegacy;
    }

    if( element == "power_law_only" )
    {
        return c_olProcessPowerLawOnly;
    }

    if( element == "moffat_peaks" )
    {
        return c_olProcessMoffatPeaks;
    }

    return c_olProcessNone;
}

inline int olProcessMethodFromName( std::string method )
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

    if( method == "legacy" )
    {
        return c_olProcessLegacy;
    }

    if( method == "power-law-only" )
    {
        return c_olProcessPowerLawOnly;
    }

    if( method == "moffat-peaks" )
    {
        return c_olProcessMoffatPeaks;
    }

    return c_olProcessNone;
}

inline std::string extrapBoolString( bool value )
{
    return value ? "true" : "false";
}

inline std::string extrapNoiseEstimateDomainElement( int domain )
{
    switch( domain )
    {
        case c_extrapNoiseEstimateClosedLoopPreXfer:
            return "closed_loop_pre_xfer";
        case c_extrapNoiseEstimateOpenLoop:
        default:
            return "open_loop";
    }
}

inline std::string extrapNoiseEstimateDomainLabel( int domain )
{
    switch( domain )
    {
        case c_extrapNoiseEstimateClosedLoopPreXfer:
            return "Closed Loop Pre-Xfer";
        case c_extrapNoiseEstimateOpenLoop:
        default:
            return "Open Loop";
    }
}

inline std::string extrapNoiseEstimateDomainName( int domain )
{
    switch( domain )
    {
        case c_extrapNoiseEstimateClosedLoopPreXfer:
            return "closed-loop-pre-xfer";
        case c_extrapNoiseEstimateOpenLoop:
        default:
            return "open-loop";
    }
}

inline int extrapNoiseEstimateDomainFromElement( const std::string &element )
{
    if( element == "closed_loop_pre_xfer" )
    {
        return c_extrapNoiseEstimateClosedLoopPreXfer;
    }

    return c_extrapNoiseEstimateOpenLoop;
}

inline int extrapNoiseEstimateDomainFromName( std::string domain )
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

    if( domain == "closed-loop-pre-xfer" )
    {
        return c_extrapNoiseEstimateClosedLoopPreXfer;
    }

    return c_extrapNoiseEstimateOpenLoop;
}

inline std::string extrapNoiseEstimateRangeElement( int range )
{
    switch( range )
    {
        case c_extrapNoiseEstimateLowFreq:
            return "low_freq";
        case c_extrapNoiseEstimateHighFreq:
        default:
            return "high_freq";
    }
}

inline std::string extrapNoiseEstimateRangeLabel( int range )
{
    switch( range )
    {
        case c_extrapNoiseEstimateLowFreq:
            return "Low Frequency";
        case c_extrapNoiseEstimateHighFreq:
        default:
            return "High Frequency";
    }
}

inline std::string extrapNoiseEstimateRangeName( int range )
{
    switch( range )
    {
        case c_extrapNoiseEstimateLowFreq:
            return "low-freq";
        case c_extrapNoiseEstimateHighFreq:
        default:
            return "high-freq";
    }
}

inline int extrapNoiseEstimateRangeFromElement( const std::string &element )
{
    if( element == "low_freq" )
    {
        return c_extrapNoiseEstimateLowFreq;
    }

    return c_extrapNoiseEstimateHighFreq;
}

inline int extrapNoiseEstimateRangeFromName( std::string range )
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

    if( range == "low-freq" )
    {
        return c_extrapNoiseEstimateLowFreq;
    }

    return c_extrapNoiseEstimateHighFreq;
}

inline std::string extrapNoiseEstimateStatisticElement( int statistic )
{
    switch( statistic )
    {
        case c_extrapNoiseEstimateMinimum:
            return "minimum";
        case c_extrapNoiseEstimatePercentile:
        default:
            return "percentile";
    }
}

inline std::string extrapNoiseEstimateStatisticLabel( int statistic )
{
    switch( statistic )
    {
        case c_extrapNoiseEstimateMinimum:
            return "Minimum";
        case c_extrapNoiseEstimatePercentile:
        default:
            return "Percentile";
    }
}

inline std::string extrapNoiseEstimateStatisticName( int statistic )
{
    switch( statistic )
    {
        case c_extrapNoiseEstimateMinimum:
            return "minimum";
        case c_extrapNoiseEstimatePercentile:
        default:
            return "percentile";
    }
}

inline int extrapNoiseEstimateStatisticFromElement( const std::string &element )
{
    if( element == "minimum" )
    {
        return c_extrapNoiseEstimateMinimum;
    }

    return c_extrapNoiseEstimatePercentile;
}

inline int extrapNoiseEstimateStatisticFromName( std::string statistic )
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

    if( statistic == "minimum" )
    {
        return c_extrapNoiseEstimateMinimum;
    }

    return c_extrapNoiseEstimatePercentile;
}

inline std::string extrapClosedLoopOlEstimateMethodElement( int method )
{
    switch( method )
    {
        case c_extrapClosedLoopOlEstimateNtfAware:
            return "ntf_aware";
        case c_extrapClosedLoopOlEstimateEtfOnly:
        default:
            return "etf_only";
    }
}

inline std::string extrapClosedLoopOlEstimateMethodLabel( int method )
{
    switch( method )
    {
        case c_extrapClosedLoopOlEstimateNtfAware:
            return "NTF Aware";
        case c_extrapClosedLoopOlEstimateEtfOnly:
        default:
            return "ETF Only";
    }
}

inline std::string extrapClosedLoopOlEstimateMethodName( int method )
{
    switch( method )
    {
        case c_extrapClosedLoopOlEstimateNtfAware:
            return "ntf-aware";
        case c_extrapClosedLoopOlEstimateEtfOnly:
        default:
            return "etf-only";
    }
}

inline int extrapClosedLoopOlEstimateMethodFromElement( const std::string &element )
{
    if( element == "ntf_aware" )
    {
        return c_extrapClosedLoopOlEstimateNtfAware;
    }

    return c_extrapClosedLoopOlEstimateEtfOnly;
}

inline int extrapClosedLoopOlEstimateMethodFromName( std::string method )
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

    if( method == "ntf-aware" )
    {
        return c_extrapClosedLoopOlEstimateNtfAware;
    }

    return c_extrapClosedLoopOlEstimateEtfOnly;
}

inline std::string extrapPowerLawCrossoverModeElement( int mode )
{
    switch( mode )
    {
        case c_extrapPowerLawCrossoverAutoSmoothedCrossing:
            return "auto_smoothed_crossing";
        case c_extrapPowerLawCrossoverManual:
        default:
            return "manual";
    }
}

inline std::string extrapPowerLawCrossoverModeLabel( int mode )
{
    switch( mode )
    {
        case c_extrapPowerLawCrossoverAutoSmoothedCrossing:
            return "Auto Smoothed Crossing";
        case c_extrapPowerLawCrossoverManual:
        default:
            return "Manual";
    }
}

inline std::string extrapPowerLawCrossoverModeName( int mode )
{
    switch( mode )
    {
        case c_extrapPowerLawCrossoverAutoSmoothedCrossing:
            return "auto-smoothed-crossing";
        case c_extrapPowerLawCrossoverManual:
        default:
            return "manual";
    }
}

inline int extrapPowerLawCrossoverModeFromElement( const std::string &element )
{
    if( element == "auto_smoothed_crossing" )
    {
        return c_extrapPowerLawCrossoverAutoSmoothedCrossing;
    }

    return c_extrapPowerLawCrossoverManual;
}

inline int extrapPowerLawCrossoverModeFromName( std::string mode )
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

    if( mode == "auto" || mode == "automatic" || mode == "auto-crossing" || mode == "auto-smoothed-crossing" )
    {
        return c_extrapPowerLawCrossoverAutoSmoothedCrossing;
    }

    return c_extrapPowerLawCrossoverManual;
}

struct psdShmimT
{
    static std::string configSection()
    {
        return "psdShmim";
    };

    static std::string indiPrefix()
    {
        return "psd";
    };
};

struct freqShmimT
{
    static std::string configSection()
    {
        return "freqShmim";
    };

    static std::string indiPrefix()
    {
        return "freq";
    };
};

struct gainFactShmimT
{
    static std::string configSection()
    {
        return "gainFactShmim";
    };

    static std::string indiPrefix()
    {
        return "gainFact";
    };
};

struct multFactShmimT
{
    static std::string configSection()
    {
        return "multFactShmim";
    };

    static std::string indiPrefix()
    {
        return "multFact";
    };
};

struct pcGainFactShmimT
{
    static std::string configSection()
    {
        return "pcGainFactShmim";
    };

    static std::string indiPrefix()
    {
        return "pcGainFact";
    };
};

struct pcMultFactShmimT
{
    static std::string configSection()
    {
        return "pcMultFactShmim";
    };

    static std::string indiPrefix()
    {
        return "pcMultFact";
    };
};

struct numpccoeffShmimT
{
    static std::string configSection()
    {
        return "numpccoeffShmim";
    };

    static std::string indiPrefix()
    {
        return "numpccoeff";
    };
};

struct acoeffShmimT
{
    static std::string configSection()
    {
        return "acoeffShmim";
    };

    static std::string indiPrefix()
    {
        return "acoeff";
    };
};

struct bcoeffShmimT
{
    static std::string configSection()
    {
        return "bcoeffShmim";
    };

    static std::string indiPrefix()
    {
        return "bcoeff";
    };
};

struct gainCalShmimT
{
    static std::string configSection()
    {
        return "gainCalShmim";
    };

    static std::string indiPrefix()
    {
        return "gainCal";
    };
};

struct gainCalFactShmimT
{
    static std::string configSection()
    {
        return "gainCalFactShmim";
    };

    static std::string indiPrefix()
    {
        return "gainCalFact";
    };
};

struct tauShmimT
{
    static std::string configSection()
    {
        return "tauShmim";
    };

    static std::string indiPrefix()
    {
        return "tau";
    };
};

struct noiseShmimT
{
    static std::string configSection()
    {
        return "noiseShmim";
    };

    static std::string indiPrefix()
    {
        return "noise";
    };
};

struct wfsavgShmimT
{
    static std::string configSection()
    {
        return "wfsavgShmim";
    };

    static std::string indiPrefix()
    {
        return "wfsavg";
    };
};

struct wfsmaskShmimT
{
    static std::string configSection()
    {
        return "wfsmaskShmim";
    };

    static std::string indiPrefix()
    {
        return "wfsmask";
    };
};

/// The MagAO-X PSD-based gain optimizer
/**
 * \ingroup modalGainOpt
 */
class modalGainOpt : public MagAOXApp<true>,
                     public dev::telemeter<modalGainOpt>,
                     dev::shmimMonitor<modalGainOpt, psdShmimT>,
                     dev::shmimMonitor<modalGainOpt, freqShmimT>,
                     dev::shmimMonitor<modalGainOpt, gainFactShmimT>,
                     dev::shmimMonitor<modalGainOpt, multFactShmimT>,
                     dev::shmimMonitor<modalGainOpt, pcGainFactShmimT>,
                     dev::shmimMonitor<modalGainOpt, pcMultFactShmimT>,
                     dev::shmimMonitor<modalGainOpt, numpccoeffShmimT>,
                     dev::shmimMonitor<modalGainOpt, acoeffShmimT>,
                     dev::shmimMonitor<modalGainOpt, bcoeffShmimT>,
                     dev::shmimMonitor<modalGainOpt, gainCalShmimT>,
                     dev::shmimMonitor<modalGainOpt, gainCalFactShmimT>,
                     dev::shmimMonitor<modalGainOpt, tauShmimT>,
                     dev::shmimMonitor<modalGainOpt, noiseShmimT>,
                     dev::shmimMonitor<modalGainOpt, wfsavgShmimT>,
                     dev::shmimMonitor<modalGainOpt, wfsmaskShmimT>

{

    // Give the test harness access.
    friend class modalGainOpt_test;
    typedef dev::telemeter<modalGainOpt> telemeterT;

    friend class dev::telemeter<modalGainOpt>;

    friend class dev::shmimMonitor<modalGainOpt, psdShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, freqShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, gainFactShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, multFactShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, pcGainFactShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, pcMultFactShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, numpccoeffShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, acoeffShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, bcoeffShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, gainCalShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, gainCalFactShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, tauShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, noiseShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, wfsavgShmimT>;
    friend class dev::shmimMonitor<modalGainOpt, wfsmaskShmimT>;

  public:
    typedef dev::shmimMonitor<modalGainOpt, psdShmimT> psdShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, freqShmimT> freqShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, gainFactShmimT> gainFactShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, multFactShmimT> multFactShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, pcGainFactShmimT> pcGainFactShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, pcMultFactShmimT> pcMultFactShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, numpccoeffShmimT> numpccoeffShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, acoeffShmimT> acoeffShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, bcoeffShmimT> bcoeffShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, gainCalShmimT> gainCalShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, gainCalFactShmimT> gainCalFactShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, tauShmimT> tauShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, noiseShmimT> noiseShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, wfsavgShmimT> wfsavgShmimMonitorT;
    typedef dev::shmimMonitor<modalGainOpt, wfsmaskShmimT> wfsmaskShmimMonitorT;

    typedef std::chrono::time_point<std::chrono::steady_clock> timePointT;
    typedef std::chrono::duration<double> durationT;

  protected:
    /** \name Configurable Parameters
     *@{
     */

    int m_loopNum{ 1 };     ///< The number of the loop. Used to set shmim names, as in
                            ///< aolN_mgainfact.

    std::string m_loopName; ///< The name of the loop control INDI device name.

    std::string m_wfsDevice{ "camwfs" };

    std::string m_psdDevice{ "hopsds" }; /**< The INDI device name of the PSD calculator.  Defaults to
                                   aolN_modevalPSDs where N is m_loopNum.*/

    std::string m_opticalGainDevice{ "strehl" };
    std::string m_opticalGainProperty{ "strehl_optimal" };
    std::string m_opticalGainElement{ "pyramid" };

    bool m_autoUpdate{ false };        ///< Flag controlling whether gains are automatically updated
    bool m_opticalGainUpdate{ false }; ///< Flag controlling whether optical gain is
                                       ///< automatically updated;

    float m_gainGain{ 0.1 };           ///< The gain to use for SI gain correction updates.  Default is 0.1.
    float m_gainLeak{ 0.9 };           ///< The leak factor used for SI gain integration. Default is 0.9.
    processPsdProcessorT::processModelConfig m_extrapConfig; ///< Configuration of the OL PSD extrapolation model.

    uint32_t m_maxNCoeff{ 1000 };

    uint32_t m_defaultNCoeff{ 25 };

    int m_extrapOL{ c_olProcessNone }; ///< Which extrapolation method to use for the OL PSD.
    int m_extrapNoiseEstimateDomain{ c_extrapNoiseEstimateOpenLoop };      ///< Where to estimate the modal noise
                                                                           ///< floor.
    int m_extrapNoiseEstimateRange{ c_extrapNoiseEstimateHighFreq };       ///< Which PSD end to use for the noise
                                                                           ///< fit.
    int m_extrapNoiseEstimateStatistic{ c_extrapNoiseEstimatePercentile }; ///< How the selected PSD bins are
                                                                           ///< summarized into a noise floor.
    int m_extrapClosedLoopOlEstimateMethod{ c_extrapClosedLoopOlEstimateEtfOnly }; ///< Which CL-to-OL reconstruction
                                                                                   ///< to use.
    int m_extrapPowerLawCrossoverMode{ c_extrapPowerLawCrossoverManual };          ///< How the power-law match/cutoff
                                                                                   ///< frequencies are chosen.

    ///@}

    uint32_t m_nFreq{ 0 };
    uint32_t m_nModes{ 0 };

    bool m_updateOnce{ false }; ///< Flag to trigger a single update with gain.

    bool m_dump{ false };       ///< Flag to trigger a single update with no gain.
    bool m_zeroGains{ false };  ///< Flag requesting the SI gain integrator state be zeroed.

    float m_fps{ 0 };

    /// Each mode gets its own gain optimizer
    std::vector<mx::AO::analysis::clGainOpt<float>> m_goptCurrent;
    std::vector<mx::AO::analysis::clGainOpt<float>> m_goptSI;
    std::vector<mx::AO::analysis::clGainOpt<float>> m_goptLP;
    std::vector<mx::AO::analysis::clAOLinearPredictor<float>> m_linPred;

    bool m_goptUpdated{ true };   ///< Tracks if a parameter has updated requiring
                                  ///< updates to the m_gopt entries.
    bool m_pcgoptUpdated{ true }; ///< Tracks if a parameter has updated requiring
                                  ///< updates to the m_gopt entries.

    bool m_freqUpdated{ true };   /**< Tracks if the frequency scale has updated, which necessitates
                                     additional calcs. If true, implies m_goptUpdate == true.*/
    float m_psdTime{ 1 };
    float m_psdAvgTime{ 10 };
    float m_psdOverlapFraction{ 0.5 };

    std::vector<float> m_freq;

    mx::improc::eigenImage<float> m_clPSDs;
    mx::improc::eigenImage<float> m_clXferCurrent; ///< Published current closed-loop error transfer
                                                   ///< function.
    mx::improc::eigenImage<float> m_clNtfCurrent;  ///< Published current closed-loop noise transfer
                                                   ///< function.
    mx::improc::eigenImage<float> m_clXferSI;      ///< Published simple-integrator closed-loop error transfer
                                                   ///< function.
    mx::improc::eigenImage<float> m_clNtfSI;       ///< Published simple-integrator closed-loop noise transfer
                                                   ///< function.
    mx::improc::eigenImage<float> m_clXferLP;      ///< Published predictive closed-loop error transfer function.
    mx::improc::eigenImage<float> m_clNtfLP;       ///< Published predictive closed-loop noise transfer function.

    std::vector<std::vector<float>> m_olPSDs;
    std::vector<std::vector<float>> m_rawOlPSDs;
    std::vector<std::vector<float>> m_smoothOlPSDs;
    std::vector<std::vector<float>> m_nPSDs;
    std::vector<float> m_modeVarCL;
    std::vector<float> m_modeVarOL;

    int m_modesOn;

    std::vector<float> m_optGainSIRaw; ///< The raw SI optimal gains before leaky integration.
    std::vector<float> m_optGainSI;    ///< The leaky-integrated SI optimal gains.
    std::vector<float> m_gmaxSI;       ///< The previously calculated maximum gains for SI.
    std::vector<float> m_modeVarSI;
    std::vector<int> m_timesOnSI;
    int m_modesOnSI;
    bool m_siGainStateNeedsSync{ true }; ///< Tracks whether the SI gain integrator state should be synced from the
                                         ///< applied gain factors.

    std::vector<float> m_optGainLP;
    std::vector<float> m_gmaxLP; ///< The previously calculated maximum gains for LP.
    std::vector<float> m_modeVarLP;
    std::vector<int> m_timesOnLP;
    int m_modesOnLP;

    bool m_loop{ false };

    float m_opticalGain{ 1 };

    float m_opticalGainSource{ 1 };

    float m_gain{ 0 };

    float m_mult{ 1 };

    float m_siGain{ 0 };

    float m_siMult{ 1 };

    bool m_doPCCalcs{ true };

    float m_pcGain{ 0 };

    float m_pcMult{ 0 };

    bool m_pcOn{ false };

    std::vector<float> m_gainFacts;

    std::vector<float> m_multFacts;

    std::vector<float> m_pcGainFacts;

    std::vector<float> m_pcMultFacts;

    std::vector<uint32_t> m_Na;        // The latest user specified number of a coefficients

    std::vector<uint32_t> m_NaCurrent; // The current number of a coefficients

    std::vector<uint32_t> m_Nb;        // The latest user specified number of b coefficients

    std::vector<uint32_t> m_NbCurrent; // The current number of b coefficients

    eigenImage<float> m_as;

    eigenImage<float> m_bs;

    int m_nRegCycles{ 60 };        ///< How often to regularize each mode

    std::vector<int> m_regCounter; ///< Counters to track when this mode was last regularized

    std::vector<float> m_regScale; ///< The regularization scale factors for each mode

    std::vector<float> m_gainCals;

    std::vector<float> m_gainCalFacts;

    std::vector<float> m_taus;

    eigenImage<float> m_noiseParams;

    eigenImage<float> m_wfsavg;
    eigenImage<float> m_wfsmask;
    float m_counts{ 0 };
    float m_emg{ 1 };
    int m_npix{ 0 };

    int m_sinceChange{ -1 };

    std::string m_olPSDShmimName;
    std::string m_rawOlPSDShmimName;
    std::string m_smoothOlPSDShmimName;
    std::string m_noisePSDShmimName;
    std::string m_clXferCurrentShmimName;
    std::string m_clNtfCurrentShmimName;
    std::string m_clXferSIShmimName;
    std::string m_clNtfSIShmimName;
    std::string m_clXferLPShmimName;
    std::string m_clNtfLPShmimName;

    std::string m_optGainShmimName;
    std::string m_optGainSIRawShmimName;
    std::string m_optGainSIShmimName;
    std::string m_maxGainSIShmimName;

    std::string m_optGainLPShmimName;
    std::string m_maxGainLPShmimName;

    std::string m_modevarShmimName;

    IMAGE *m_olPSDStream{ nullptr };         ///< The ImageStreamIO shared memory buffer to
                                             ///< publish the open loop PSDs
    IMAGE *m_rawOlPSDStream{ nullptr };      ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the raw OL PSDs
    IMAGE *m_smoothOlPSDStream{ nullptr };   ///< The ImageStreamIO shared memory buffer to publish the
                                             ///< smoothed OL PSDs
    IMAGE *m_noisePSDStream{ nullptr };      ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the noise PSDs
    IMAGE *m_clXferCurrentStream{ nullptr }; ///< The ImageStreamIO shared memory
                                             ///< buffer to publish the current ETF
    IMAGE *m_clNtfCurrentStream{ nullptr };  ///< The ImageStreamIO shared memory
                                             ///< buffer to publish the current NTF
    IMAGE *m_clXferSIStream{ nullptr };      ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the SI ETF
    IMAGE *m_clNtfSIStream{ nullptr };       ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the SI NTF
    IMAGE *m_clXferLPStream{ nullptr };      ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the LP ETF
    IMAGE *m_clNtfLPStream{ nullptr };       ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the LP NTF

    IMAGE *m_optGainStream{ nullptr };       ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the current optimal gains

    IMAGE *m_optGainSIRawStream{ nullptr };  ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the raw SI optimal gains
    IMAGE *m_optGainSIStream{ nullptr };     ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the integrated SI optimal gains
    IMAGE *m_maxGainSIStream{ nullptr };     ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the SI max gains

    IMAGE *m_optGainLPStream{ nullptr };     ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the LP optimal gains
    IMAGE *m_maxGainLPStream{ nullptr };     ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the LP max gains

    IMAGE *m_modevarStream{ nullptr };       ///< The ImageStreamIO shared memory buffer
                                             ///< to publish the mode variances

    /// Destroy an owned ImageStreamIO output stream and clear its pointer.
    void destroyImageStream( IMAGE *&stream /**< [in.out] stream pointer to destroy and clear */ );

    /// Allocate and create an owned ImageStreamIO output stream.
    int createImageStream( IMAGE *&stream,          /**< [in.out] stream pointer to allocate and create */
                           const std::string &name, /**< [in] shmim name for the output stream */
                           uint32_t size0,          /**< [in] first axis size */
                           uint32_t size1,          /**< [in] second axis size */
                           uint32_t size2,          /**< [in] third axis size */
                           uint8_t dataType         /**< [in] ImageStreamIO datatype for the stream */
    );

    /// Populate the published gain and variance arrays from the current
    /// optimization state.
    void writePublishedGainArrays( float *currentData, /**< [out] current optimal-gain stream buffer */
                                   float *siRawData,   /**< [out] raw SI optimal-gain stream buffer */
                                   float *siData,      /**< [out] integrated SI optimal-gain stream buffer */
                                   float *maxSiData,   /**< [out] SI max-gain stream buffer */
                                   float *lpData,      /**< [out] LP optimal-gain stream buffer */
                                   float *maxLpData,   /**< [out] LP max-gain stream buffer */
                                   float *modeVarData  /**< [out] mode-variance buffer, laid out as 3 x N */
    );

    /// Populate the published predictive-control gain and coefficient arrays.
    void writePublishedPredictorArrays( float *pcGainData, /**< [in.out] PC gain-factor stream buffer */
                                        float *aCoeffData, /**< [in.out] predictor a-coefficient stream buffer */
                                        uint32_t aWidth,   /**< [in] entries stored per mode in aCoeffData */
                                        float *bCoeffData, /**< [in.out] predictor b-coefficient stream buffer */
                                        uint32_t bWidth,   /**< [in] entries stored per mode in bCoeffData */
                                        bool blend         /**< [in] when true, blend against existing values */
    );

    /// Count how many modes are enabled by a gain-factor vector.
    int countEnabledGainFactors( const std::vector<float> &gainFacts /**< [in] gain factors to inspect */ ) const;

    /// Update `m_modesOn` when a changed gain-factor stream matches the active
    /// control path.
    void updateAppliedModeCount( const std::vector<float> &gainFacts, /**< [in] gain factors from the changed stream */
                                 bool predictorPath /**< [in] true when the values came from the predictor
                                                       path */
    );

    /// Apply an incoming gain-factor frame to one of the stored gain vectors.
    bool applyGainFactorUpdate( std::vector<float> &gainFacts, /**< [in.out] stored gain factors to resize and update */
                                const float *incoming,         /**< [in] incoming gain-factor frame */
                                uint32_t width,                /**< [in] number of gain factors in `incoming` */
                                bool predictorPath             /**< [in] true when the values came from the predictor
                                                                  path */
    );

    /// Apply an incoming multiplier frame to one of the stored multiplier
    /// vectors.
    bool applyMultiplierUpdate( std::vector<float> &multFacts, /**< [in.out] stored multiplier factors to
                                                                  resize and update */
                                const float *incoming,         /**< [in] incoming multiplier frame */
                                uint32_t width,                /**< [in] number of multiplier factors in `incoming` */
                                bool predictorPath             /**< [in] true when the values came from the predictor
                                                                  path */
    );

    /// Apply an incoming frequency frame to the stored frequency scale.
    bool applyFrequencyUpdate( const float *incoming, /**< [in] incoming frequency frame */
                               size_t size            /**< [in] number of frequency samples in `incoming` */
    );

    /// Refresh gain-optimization structures after coefficient, multiplier, or
    /// frequency changes.
    /** The gain-optimization mutex must be locked before calling this helper.
     *
     * \returns true when a structure refresh was performed
     * \returns false when no refresh was needed
     */
    bool refreshGoptStructures();

    /// Synchronize the integrated SI gain state from the applied gain-factor stream.
    void syncSiGainStateFromAppliedGains();

    /// Apply one SI leaky-integrator update from the raw optimal gain.
    void updateIntegratedSiGain( size_t modeIndex );

    /// Handle a standard target/current numeric extrapolation property update.
    template <typename valueT>
    int handleExtrapNumberProperty( pcf::IndiProperty &localProperty,
                                    valueT &localTarget,
                                    const pcf::IndiProperty &ipRecv,
                                    const std::string &label );

    /// Handle a boolean extrapolation toggle property update.
    int handleExtrapToggleProperty( pcf::IndiProperty &localProperty,
                                    bool &localTarget,
                                    const pcf::IndiProperty &ipRecv,
                                    const std::string &label );

    /// Handle the extrapolation-method selection switch property.
    int handleExtrapMethodProperty( const pcf::IndiProperty &ipRecv );

    /// Handle the noise-estimation-domain selection switch property.
    int handleExtrapNoiseEstimateDomainProperty( const pcf::IndiProperty &ipRecv );

    /// Handle the noise-estimation-range selection switch property.
    int handleExtrapNoiseEstimateRangeProperty( const pcf::IndiProperty &ipRecv );

    /// Handle the noise-estimation-statistic selection switch property.
    int handleExtrapNoiseEstimateStatisticProperty( const pcf::IndiProperty &ipRecv );

    /// Handle the closed-loop OL-estimation-method selection switch property.
    int handleExtrapClosedLoopOlEstimateMethodProperty( const pcf::IndiProperty &ipRecv );

    /// Handle the power-law crossover-mode selection switch property.
    int handleExtrapPowerLawCrossoverModeProperty( const pcf::IndiProperty &ipRecv );

  public:
    /// Default c'tor.
    modalGainOpt();

    /// D'tor, declared and defined for noexcept.
    ~modalGainOpt() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] an application configuration
                                                                    from which to load values*/ );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for modalGainOpt.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     *
     */
    virtual int appShutdown();

    int allocatePCShmims();

    int allocate( const psdShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,   ///< [in] pointer to the start of the current frame
                      const psdShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const freqShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,    ///< [in] pointer to the start of the current frame
                      const freqShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const gainFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,        ///< [in] pointer to the start of the current frame
                      const gainFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const multFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,        ///< [in] pointer to the start of the current frame
                      const multFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const pcGainFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,          ///< [in] pointer to the start of the current frame
                      const pcGainFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const pcMultFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,          ///< [in] pointer to the start of the current frame
                      const pcMultFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const numpccoeffShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,          ///< [in] pointer to the start of the current frame
                      const numpccoeffShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const acoeffShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,      ///< [in] pointer to the start of the current frame
                      const acoeffShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const bcoeffShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,      ///< [in] pointer to the start of the current frame
                      const bcoeffShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const gainCalShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,       ///< [in] pointer to the start of the current frame
                      const gainCalShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const gainCalFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,           ///< [in] pointer to the start of the current frame
                      const gainCalFactShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const tauShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,   ///< [in] pointer to the start of the current frame
                      const tauShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const noiseShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,     ///< [in] pointer to the start of the current frame
                      const noiseShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const wfsavgShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,      ///< [in] pointer to the start of the current frame
                      const wfsavgShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int allocate( const wfsmaskShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    int processImage( void *curr_src,       ///< [in] pointer to the start of the current frame
                      const wfsmaskShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    /// Check that all sizes and allocations have occurred
    int checkSizes();

  protected:
    /// Mutex for synchronizing updates.
    std::mutex m_goptMutex;

    /// Flag used to indicate to the goptThread that it should stop calculations
    /// ASAP
    std::atomic<bool> m_updating{ false };

    /** \name Gain Optimization Thread
     *
     * @{
     */
    int m_goptThreadPrio{ 0 };          ///< Priority of the gain optimization thread.

    std::string m_goptThreadCpuset;     ///< The cpuset to use for the gain
                                        ///< optimization thread.

    std::thread m_goptThread;           ///< The gain optimization thread.

    bool m_goptThreadInit{ true };      ///< Initialization flag for the gain optimization thread.

    pid_t m_goptThreadID{ 0 };          ///< gain optimization thread PID.

    pcf::IndiProperty m_goptThreadProp; ///< The property to hold the gain
                                        ///< optimization thread details.

    sem_t m_goptSemaphore;              ///< Semaphore used to synchronize the psdShmim thread
                                        ///< and the gopt thread.
    bool m_goptSemaphoreInit{ false };  ///< Tracks whether the gain optimization semaphore needs cleanup.

    float noisePSD( int n );

    /// Gain Optimization thread starter function
    static void goptThreadStart( modalGainOpt *p /**< [in] pointer to this */ );

    /// Gain optimization thread function
    /** Runs until m_shutdown is true.
     */
    void goptThreadExec();

    ///@}

  public:
    /** \name INDI
     * @{
     */

    pcf::IndiProperty m_indiP_autoUpdate;
    pcf::IndiProperty m_indiP_updateOnce;
    pcf::IndiProperty m_indiP_dump;

    pcf::IndiProperty m_indiP_opticalGain;

    pcf::IndiProperty m_indiP_gainGain;
    pcf::IndiProperty m_indiP_gainLeak;
    pcf::IndiProperty m_indiP_zeroGains;
    pcf::IndiProperty m_indiP_extrapMethod;
    pcf::IndiProperty m_indiP_extrapNoiseEstimateDomain;
    pcf::IndiProperty m_indiP_extrapNoiseEstimateRange;
    pcf::IndiProperty m_indiP_extrapNoiseEstimateStatistic;
    pcf::IndiProperty m_indiP_extrapNoiseEstimateLowFreqMaxHz;
    pcf::IndiProperty m_indiP_extrapClosedLoopOlEstimateMethod;
    pcf::IndiProperty m_indiP_extrapPowerLawIndex;
    pcf::IndiProperty m_indiP_extrapPowerLawNormFreq;
    pcf::IndiProperty m_indiP_extrapPowerLawMatchFreq;
    pcf::IndiProperty m_indiP_extrapPowerLawMatchFallbackWindowHz;
    pcf::IndiProperty m_indiP_extrapPowerLawCrossoverMode;
    pcf::IndiProperty m_indiP_extrapPowerLawAutoSmoothWidthHz;
    pcf::IndiProperty m_indiP_extrapPowerLawAutoMaxFreqFraction;
    pcf::IndiProperty m_indiP_extrapFitPowerLawIndex;
    pcf::IndiProperty m_indiP_extrapPowerLawOnlyAboveFreq;
    pcf::IndiProperty m_indiP_extrapPowerLawFitIncludesMatchPoint;
    pcf::IndiProperty m_indiP_extrapPowerLawFitMinFreqHz;
    pcf::IndiProperty m_indiP_extrapPowerLawFitMaxFreqHz;
    pcf::IndiProperty m_indiP_extrapPowerLawFitBinWidthHz;
    pcf::IndiProperty m_indiP_extrapPowerLawBlendBins;
    pcf::IndiProperty m_indiP_extrapDropoutGapFactor;
    pcf::IndiProperty m_indiP_extrapDropoutTinyFactor;
    pcf::IndiProperty m_indiP_extrapDropoutMaxBins;
    pcf::IndiProperty m_indiP_extrapClSignificanceThreshold;
    pcf::IndiProperty m_indiP_extrapClMinSignificantFraction;

    pcf::IndiProperty m_indiP_emg;
    pcf::IndiProperty m_indiP_psdTime;
    pcf::IndiProperty m_indiP_psdAvgTime;
    pcf::IndiProperty m_indiP_loop;
    pcf::IndiProperty m_indiP_siGain;
    pcf::IndiProperty m_indiP_siMult;
    pcf::IndiProperty m_indiP_pcGain;
    pcf::IndiProperty m_indiP_pcMult;
    pcf::IndiProperty m_indiP_pcOn;

    pcf::IndiProperty m_indiP_modesOn;

    pcf::IndiProperty m_indiP_opticalGainSource;
    pcf::IndiProperty m_indiP_opticalGainUpdate;

    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_autoUpdate );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_updateOnce );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_dump );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_opticalGain );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_gainGain );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_gainLeak );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_zeroGains );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapMethod );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapNoiseEstimateDomain );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapNoiseEstimateRange );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapNoiseEstimateStatistic );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapNoiseEstimateLowFreqMaxHz );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapClosedLoopOlEstimateMethod );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawIndex );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawNormFreq );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawMatchFreq );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawMatchFallbackWindowHz );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawCrossoverMode );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawAutoSmoothWidthHz );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawAutoMaxFreqFraction );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapFitPowerLawIndex );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawOnlyAboveFreq );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawFitIncludesMatchPoint );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawFitMinFreqHz );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawFitMaxFreqHz );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawFitBinWidthHz );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapPowerLawBlendBins );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapDropoutGapFactor );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapDropoutTinyFactor );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapDropoutMaxBins );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapClSignificanceThreshold );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_extrapClMinSignificantFraction );
    INDI_SETCALLBACK_DECL( modalGainOpt, m_indiP_emg );
    INDI_SETCALLBACK_DECL( modalGainOpt, m_indiP_psdTime );
    INDI_SETCALLBACK_DECL( modalGainOpt, m_indiP_psdAvgTime );
    INDI_SETCALLBACK_DECL( modalGainOpt, m_indiP_loop );
    INDI_SETCALLBACK_DECL( modalGainOpt, m_indiP_siGain );
    INDI_SETCALLBACK_DECL( modalGainOpt, m_indiP_siMult );
    INDI_SETCALLBACK_DECL( modalGainOpt, m_indiP_pcGain );
    INDI_SETCALLBACK_DECL( modalGainOpt, m_indiP_pcMult );
    INDI_SETCALLBACK_DECL( modalGainOpt, m_indiP_pcOn );

    INDI_SETCALLBACK_DECL( modalGainOpt, m_indiP_opticalGainSource );
    INDI_NEWCALLBACK_DECL( modalGainOpt, m_indiP_opticalGainUpdate );

    /** \name Telemeter Interface
     *
     * @{
     */
    int checkRecordTimes();

    int recordTelem( const telem_modalgainopt * );

    int recordModalGainOpt( bool force = false );

    ///@}

    ///@}
};

modalGainOpt::modalGainOpt() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    psdShmimMonitorT::m_getExistingFirst = true;
    freqShmimMonitorT::m_getExistingFirst = true;
    gainFactShmimMonitorT::m_getExistingFirst = true;
    multFactShmimMonitorT::m_getExistingFirst = true;
    pcGainFactShmimMonitorT::m_getExistingFirst = true;
    pcMultFactShmimMonitorT::m_getExistingFirst = true;
    numpccoeffShmimMonitorT::m_getExistingFirst = true;
    acoeffShmimMonitorT::m_getExistingFirst = true;
    bcoeffShmimMonitorT::m_getExistingFirst = true;
    gainCalShmimMonitorT::m_getExistingFirst = true;
    gainCalFactShmimMonitorT::m_getExistingFirst = true;
    tauShmimMonitorT::m_getExistingFirst = true;
    noiseShmimMonitorT::m_getExistingFirst = true;
    wfsavgShmimMonitorT::m_getExistingFirst = true;
    wfsmaskShmimMonitorT::m_getExistingFirst = true;

    return;
}

void modalGainOpt::setupConfig()
{
    config.add( "loop.number",
                "",
                "loop.number",
                argType::Required,
                "loop",
                "number",
                false,
                "int",
                "The number of the loop. Used to set shmim names, as in aolN_mgainfact." );

    config.add( "loop.name",
                "",
                "loop.name",
                argType::Required,
                "loop",
                "name",
                false,
                "string",
                "The name of the loop control INDI device name." );

    config.add( "loop.psdDev",
                "",
                "loop.psdDev",
                argType::Required,
                "loop",
                "psdDev",
                false,
                "string",
                "The INDI device name of the PSD calculator.  Defaults to "
                "aolN_modevalPSDs where N is loop.number." );

    config.add( "loop.autoUpdate",
                "",
                "loop.autoUpdate",
                argType::Required,
                "loop",
                "autoUpdate",
                false,
                "bool",
                "Flag controlling whether the gains are auto updated.  Also "
                "settable via INDI." );

    config.add( "loop.gainGain",
                "",
                "loop.gainGain",
                argType::Required,
                "loop",
                "gainGain",
                false,
                "float",
                "The gain to use for closed-loop gain updates.  Default is 0.1" );

    config.add( "loop.gainLeak",
                "",
                "loop.gainLeak",
                argType::Required,
                "loop",
                "gainLeak",
                false,
                "float",
                "The leak factor to use for SI optimal-gain integration.  Default is 0.9" );

    config.add( "extrapolation.method",
                "",
                "extrapolation.method",
                argType::Required,
                "extrapolation",
                "method",
                false,
                "string",
                "The OL PSD extrapolation method: none, legacy, power_law_only, "
                "or moffat_peaks." );

    config.add( "extrapolation.noiseEstimateDomain",
                "",
                "extrapolation.noiseEstimateDomain",
                argType::Required,
                "extrapolation",
                "noiseEstimateDomain",
                false,
                "string",
                "Where to estimate the flat noise floor: open_loop or "
                "closed_loop_pre_xfer." );

    config.add( "extrapolation.noiseEstimateRange",
                "",
                "extrapolation.noiseEstimateRange",
                argType::Required,
                "extrapolation",
                "noiseEstimateRange",
                false,
                "string",
                "Which end of the PSD is used for noise estimation: high_freq or "
                "low_freq." );

    config.add( "extrapolation.noiseEstimateStatistic",
                "",
                "extrapolation.noiseEstimateStatistic",
                argType::Required,
                "extrapolation",
                "noiseEstimateStatistic",
                false,
                "string",
                "How to summarize the selected noise-estimation bins: percentile "
                "or minimum." );

    config.add( "extrapolation.noiseEstimateLowFreqMaxHz",
                "",
                "extrapolation.noiseEstimateLowFreqMaxHz",
                argType::Required,
                "extrapolation",
                "noiseEstimateLowFreqMaxHz",
                false,
                "float",
                "For low_freq noise estimation, the maximum frequency in Hz to "
                "include. Set to 0 to disable." );

    config.add( "extrapolation.closedLoopOlEstimateMethod",
                "",
                "extrapolation.closedLoopOlEstimateMethod",
                argType::Required,
                "extrapolation",
                "closedLoopOlEstimateMethod",
                false,
                "string",
                "How to reconstruct OL PSD from CL PSD: etf_only or ntf_aware." );

    config.add( "extrapolation.powerLawIndex",
                "",
                "extrapolation.powerLawIndex",
                argType::Required,
                "extrapolation",
                "powerLawIndex",
                false,
                "float",
                "The power-law exponent a in the 1/f^a continuum model." );

    config.add( "extrapolation.powerLawNormFreq",
                "",
                "extrapolation.powerLawNormFreq",
                argType::Required,
                "extrapolation",
                "powerLawNormFreq",
                false,
                "float",
                "The power-law normalization frequency in Hz. Set to 0 to use the "
                "first positive bin." );

    config.add( "extrapolation.powerLawMatchFreq",
                "",
                "extrapolation.powerLawMatchFreq",
                argType::Required,
                "extrapolation",
                "powerLawMatchFreq",
                false,
                "float",
                "The frequency in Hz where the extrapolated power law is forced "
                "to match the measured PSD." );

    config.add( "extrapolation.powerLawMatchFallbackWindowHz",
                "",
                "extrapolation.powerLawMatchFallbackWindowHz",
                argType::Required,
                "extrapolation",
                "powerLawMatchFallbackWindowHz",
                false,
                "float",
                "Half-width in Hz of the local fallback window used when the "
                "match bin falls in a trough." );

    config.add( "extrapolation.powerLawCrossoverMode",
                "",
                "extrapolation.powerLawCrossoverMode",
                argType::Required,
                "extrapolation",
                "powerLawCrossoverMode",
                false,
                "string",
                "How the power-law match/cutoff frequencies are chosen: manual or "
                "auto_smoothed_crossing." );

    config.add( "extrapolation.powerLawAutoSmoothWidthHz",
                "",
                "extrapolation.powerLawAutoSmoothWidthHz",
                argType::Required,
                "extrapolation",
                "powerLawAutoSmoothWidthHz",
                false,
                "float",
                "Median-smoothing width in Hz used when auto power-law crossover "
                "selection is enabled." );

    config.add( "extrapolation.powerLawAutoMaxFreqFraction",
                "",
                "extrapolation.powerLawAutoMaxFreqFraction",
                argType::Required,
                "extrapolation",
                "powerLawAutoMaxFreqFraction",
                false,
                "float",
                "Maximum searched frequency for auto power-law crossover as a "
                "fraction of the sampled maximum frequency. Set to 0 to disable "
                "the cap." );

    config.add( "extrapolation.fitPowerLawIndex",
                "",
                "extrapolation.fitPowerLawIndex",
                argType::Required,
                "extrapolation",
                "fitPowerLawIndex",
                false,
                "bool",
                "Whether to fit the power-law index from the high-frequency "
                "disturbance PSD bins." );

    config.add( "extrapolation.powerLawOnlyAboveFreq",
                "",
                "extrapolation.powerLawOnlyAboveFreq",
                argType::Required,
                "extrapolation",
                "powerLawOnlyAboveFreq",
                false,
                "float",
                "Above this frequency in Hz, force the extrapolation to be "
                "power-law only." );

    config.add( "extrapolation.powerLawFitIncludesMatchPoint",
                "",
                "extrapolation.powerLawFitIncludesMatchPoint",
                argType::Required,
                "extrapolation",
                "powerLawFitIncludesMatchPoint",
                false,
                "bool",
                "Whether to include the explicit match point directly in the "
                "power-law exponent fit." );

    config.add( "extrapolation.powerLawFitMinFreqHz",
                "",
                "extrapolation.powerLawFitMinFreqHz",
                argType::Required,
                "extrapolation",
                "powerLawFitMinFreqHz",
                false,
                "float",
                "The low edge in Hz of the power-law exponent fit range." );

    config.add( "extrapolation.powerLawFitMaxFreqHz",
                "",
                "extrapolation.powerLawFitMaxFreqHz",
                argType::Required,
                "extrapolation",
                "powerLawFitMaxFreqHz",
                false,
                "float",
                "The high edge in Hz of the power-law exponent fit range." );

    config.add( "extrapolation.powerLawFitBinWidthHz",
                "",
                "extrapolation.powerLawFitBinWidthHz",
                argType::Required,
                "extrapolation",
                "powerLawFitBinWidthHz",
                false,
                "float",
                "The width in Hz of the median bins used in the power-law exponent fit." );

    config.add( "extrapolation.powerLawBlendBins",
                "",
                "extrapolation.powerLawBlendBins",
                argType::Required,
                "extrapolation",
                "powerLawBlendBins",
                false,
                "int",
                "The number of bins used to blend between the measured PSD and "
                "the extrapolated continuum." );

    config.add( "extrapolation.peakDetectWidthHz",
                "",
                "extrapolation.peakDetectWidthHz",
                argType::Required,
                "extrapolation",
                "peakDetectWidthHz",
                false,
                "float",
                "The wide smoothing width in Hz used for peak detection." );

    config.add( "extrapolation.peakDetectFactor",
                "",
                "extrapolation.peakDetectFactor",
                argType::Required,
                "extrapolation",
                "peakDetectFactor",
                false,
                "float",
                "The factor above the smoothed PSD required for a strong peak "
                "detection." );

    config.add( "extrapolation.peakDetectBroadFactor",
                "",
                "extrapolation.peakDetectBroadFactor",
                argType::Required,
                "extrapolation",
                "peakDetectBroadFactor",
                false,
                "float",
                "The lower factor above the smoothed PSD used for broad-peak "
                "candidates." );

    config.add( "extrapolation.peakDetectMinWidthLog",
                "",
                "extrapolation.peakDetectMinWidthLog",
                argType::Required,
                "extrapolation",
                "peakDetectMinWidthLog",
                false,
                "float",
                "The minimum accepted broad-peak width in log-frequency." );

    config.add( "extrapolation.peakDetectPasses",
                "",
                "extrapolation.peakDetectPasses",
                argType::Required,
                "extrapolation",
                "peakDetectPasses",
                false,
                "int",
                "The number of iterative subtract-and-redetect peak-detection passes." );

    config.add( "extrapolation.peakMoffatBeta",
                "",
                "extrapolation.peakMoffatBeta",
                argType::Required,
                "extrapolation",
                "peakMoffatBeta",
                false,
                "float",
                "The minimum Moffat beta used when synthesizing extrapolated peaks." );

    config.add( "extrapolation.dropoutGapFactor",
                "",
                "extrapolation.dropoutGapFactor",
                argType::Required,
                "extrapolation",
                "dropoutGapFactor",
                false,
                "float",
                "The relative depth threshold used to identify PSD dropouts for repair." );

    config.add( "extrapolation.dropoutTinyFactor",
                "",
                "extrapolation.dropoutTinyFactor",
                argType::Required,
                "extrapolation",
                "dropoutTinyFactor",
                false,
                "float",
                "The maximum fraction of the local good-bin scale allowed for a "
                "candidate dropout run to be considered truly tiny." );

    config.add( "extrapolation.dropoutMaxBins",
                "",
                "extrapolation.dropoutMaxBins",
                argType::Required,
                "extrapolation",
                "dropoutMaxBins",
                false,
                "int",
                "The maximum consecutive dropout-run length that will be repaired." );

    config.add( "extrapolation.clSignificanceThreshold",
                "",
                "extrapolation.clSignificanceThreshold",
                argType::Required,
                "extrapolation",
                "clSignificanceThreshold",
                false,
                "float",
                "The multiplier above the fitted raw CL noise floor required "
                "for a PSD bin to be considered significant." );

    config.add( "extrapolation.clMinSignificantFraction",
                "",
                "extrapolation.clMinSignificantFraction",
                argType::Required,
                "extrapolation",
                "clMinSignificantFraction",
                false,
                "float",
                "The minimum fraction of raw CL PSD bins that must exceed the "
                "significance threshold for a mode to remain active." );

    SHMIMMONITORT_SETUP_CONFIG( psdShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( freqShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( gainFactShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( multFactShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( pcGainFactShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( pcMultFactShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( numpccoeffShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( acoeffShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( bcoeffShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( gainCalShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( gainCalFactShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( tauShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( noiseShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( wfsavgShmimMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( wfsmaskShmimMonitorT, config );

    telemeterT::setupConfig( config );
}

int modalGainOpt::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_loopNum, "loop.number" );
    _config( m_loopName, "loop.name" );
    _config( m_autoUpdate, "loop.autoUpdate" );
    _config( m_gainGain, "loop.gainGain" );
    _config( m_gainLeak, "loop.gainLeak" );

    std::string extrapMethod = olProcessMethodName( m_extrapOL );
    _config( extrapMethod, "extrapolation.method" );
    m_extrapOL = olProcessMethodFromName( extrapMethod );
    std::string noiseEstimateDomain = extrapNoiseEstimateDomainName( m_extrapNoiseEstimateDomain );
    _config( noiseEstimateDomain, "extrapolation.noiseEstimateDomain" );
    m_extrapNoiseEstimateDomain = extrapNoiseEstimateDomainFromName( noiseEstimateDomain );
    m_extrapConfig.m_noiseEstimateDomain = extrapNoiseEstimateDomainName( m_extrapNoiseEstimateDomain );
    std::string noiseEstimateRange = extrapNoiseEstimateRangeName( m_extrapNoiseEstimateRange );
    _config( noiseEstimateRange, "extrapolation.noiseEstimateRange" );
    m_extrapNoiseEstimateRange = extrapNoiseEstimateRangeFromName( noiseEstimateRange );
    m_extrapConfig.m_noiseEstimateRange = extrapNoiseEstimateRangeName( m_extrapNoiseEstimateRange );
    std::string noiseEstimateStatistic = extrapNoiseEstimateStatisticName( m_extrapNoiseEstimateStatistic );
    _config( noiseEstimateStatistic, "extrapolation.noiseEstimateStatistic" );
    m_extrapNoiseEstimateStatistic = extrapNoiseEstimateStatisticFromName( noiseEstimateStatistic );
    m_extrapConfig.m_noiseEstimateStatistic = extrapNoiseEstimateStatisticName( m_extrapNoiseEstimateStatistic );
    _config( m_extrapConfig.m_noiseEstimateLowFreqMaxHz, "extrapolation.noiseEstimateLowFreqMaxHz" );
    std::string closedLoopOlEstimateMethod = extrapClosedLoopOlEstimateMethodName( m_extrapClosedLoopOlEstimateMethod );
    _config( closedLoopOlEstimateMethod, "extrapolation.closedLoopOlEstimateMethod" );
    m_extrapClosedLoopOlEstimateMethod = extrapClosedLoopOlEstimateMethodFromName( closedLoopOlEstimateMethod );
    m_extrapConfig.m_closedLoopOlEstimateMethod =
        extrapClosedLoopOlEstimateMethodName( m_extrapClosedLoopOlEstimateMethod );

    _config( m_extrapConfig.m_powerLawIndex, "extrapolation.powerLawIndex" );
    _config( m_extrapConfig.m_powerLawNormFreq, "extrapolation.powerLawNormFreq" );
    _config( m_extrapConfig.m_powerLawMatchFreq, "extrapolation.powerLawMatchFreq" );
    _config( m_extrapConfig.m_powerLawMatchFallbackWindowHz, "extrapolation.powerLawMatchFallbackWindowHz" );
    std::string powerLawCrossoverMode = extrapPowerLawCrossoverModeName( m_extrapPowerLawCrossoverMode );
    _config( powerLawCrossoverMode, "extrapolation.powerLawCrossoverMode" );
    m_extrapPowerLawCrossoverMode = extrapPowerLawCrossoverModeFromName( powerLawCrossoverMode );
    m_extrapConfig.m_powerLawCrossoverMode = extrapPowerLawCrossoverModeName( m_extrapPowerLawCrossoverMode );
    _config( m_extrapConfig.m_powerLawAutoSmoothWidthHz, "extrapolation.powerLawAutoSmoothWidthHz" );
    _config( m_extrapConfig.m_powerLawAutoMaxFreqFraction, "extrapolation.powerLawAutoMaxFreqFraction" );
    _config( m_extrapConfig.m_fitPowerLawIndex, "extrapolation.fitPowerLawIndex" );
    _config( m_extrapConfig.m_powerLawOnlyAboveFreq, "extrapolation.powerLawOnlyAboveFreq" );
    _config( m_extrapConfig.m_powerLawFitIncludesMatchPoint, "extrapolation.powerLawFitIncludesMatchPoint" );
    _config( m_extrapConfig.m_powerLawFitMinFreqHz, "extrapolation.powerLawFitMinFreqHz" );
    _config( m_extrapConfig.m_powerLawFitMaxFreqHz, "extrapolation.powerLawFitMaxFreqHz" );
    _config( m_extrapConfig.m_powerLawFitBinWidthHz, "extrapolation.powerLawFitBinWidthHz" );
    _config( m_extrapConfig.m_powerLawBlendBins, "extrapolation.powerLawBlendBins" );
    _config( m_extrapConfig.m_peakDetectWidthHz, "extrapolation.peakDetectWidthHz" );
    _config( m_extrapConfig.m_peakDetectFactor, "extrapolation.peakDetectFactor" );
    _config( m_extrapConfig.m_peakDetectBroadFactor, "extrapolation.peakDetectBroadFactor" );
    _config( m_extrapConfig.m_peakDetectMinWidthLog, "extrapolation.peakDetectMinWidthLog" );
    _config( m_extrapConfig.m_peakDetectPasses, "extrapolation.peakDetectPasses" );
    _config( m_extrapConfig.m_peakMoffatBeta, "extrapolation.peakMoffatBeta" );
    _config( m_extrapConfig.m_dropoutGapFactor, "extrapolation.dropoutGapFactor" );
    _config( m_extrapConfig.m_dropoutTinyFactor, "extrapolation.dropoutTinyFactor" );
    _config( m_extrapConfig.m_dropoutMaxBins, "extrapolation.dropoutMaxBins" );
    _config( m_extrapConfig.m_clSignificanceThreshold, "extrapolation.clSignificanceThreshold" );
    _config( m_extrapConfig.m_clMinSignificantFraction, "extrapolation.clMinSignificantFraction" );

    char shmim[1024];

    _config( m_psdDevice, "loop.psdDev" );

    snprintf( shmim, sizeof( shmim ), "aol%d_clpsds", m_loopNum );
    psdShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( psdShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_freq", m_loopNum );
    freqShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( freqShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_mgainfact", m_loopNum );
    gainFactShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( gainFactShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_mmultfact", m_loopNum );
    multFactShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( multFactShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_mpcgainfact", m_loopNum );
    pcGainFactShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( pcGainFactShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_mpcmultfact", m_loopNum );
    pcMultFactShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( pcMultFactShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_numpccoeff", m_loopNum );
    numpccoeffShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( numpccoeffShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_acoeff", m_loopNum );
    acoeffShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( acoeffShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_bcoeff", m_loopNum );
    bcoeffShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( bcoeffShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_mgaincal", m_loopNum );
    gainCalShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( gainCalShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_mgaincalfact", m_loopNum );
    gainCalFactShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( gainCalFactShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_looptau", m_loopNum );
    tauShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( tauShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_mnoiseparam", m_loopNum );
    noiseShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( noiseShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_wfsavg", m_loopNum );
    wfsavgShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( wfsavgShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_wfsmask", m_loopNum );
    wfsmaskShmimMonitorT::m_shmimName = shmim;
    SHMIMMONITORT_LOAD_CONFIG( wfsmaskShmimMonitorT, _config );

    snprintf( shmim, sizeof( shmim ), "aol%d_olpsds", m_loopNum );
    m_olPSDShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_olpsds_raw", m_loopNum );
    m_rawOlPSDShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_olpsds_smooth", m_loopNum );
    m_smoothOlPSDShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_noisepsds", m_loopNum );
    m_noisePSDShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_clxferCurrent", m_loopNum );
    m_clXferCurrentShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_clntfCurrent", m_loopNum );
    m_clNtfCurrentShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_clxferSI", m_loopNum );
    m_clXferSIShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_clntfSI", m_loopNum );
    m_clNtfSIShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_clxferLP", m_loopNum );
    m_clXferLPShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_clntfLP", m_loopNum );
    m_clNtfLPShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_mgainoptimal", m_loopNum );
    m_optGainShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_mgainoptimalSI_raw", m_loopNum );
    m_optGainSIRawShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_mgainoptimalSI", m_loopNum );
    m_optGainSIShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_mgainmaxSI", m_loopNum );
    m_maxGainSIShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_mgainoptimalLP", m_loopNum );
    m_optGainLPShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_mgainmaxLP", m_loopNum );
    m_maxGainLPShmimName = shmim;

    snprintf( shmim, sizeof( shmim ), "aol%d_mmodevar", m_loopNum );
    m_modevarShmimName = shmim;

    if( telemeterT::loadConfig( _config ) < 0 )
    {
        log<text_log>( "Error during telemeter config", logPrio::LOG_CRITICAL );
        m_shutdown = true;
    }

    return 0;
}

void modalGainOpt::loadConfig()
{
    loadConfigImpl( config );
}

int modalGainOpt::appStartup()
{
    if( telemeterT::appStartup() < 0 )
    {
        return log<software_error, -1>( { "error from telemeter appStartup" } );
    }

    SHMIMMONITORT_APP_STARTUP( psdShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( freqShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( gainFactShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( multFactShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( pcGainFactShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( pcMultFactShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( numpccoeffShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( acoeffShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( bcoeffShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( gainCalShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( gainCalFactShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( tauShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( noiseShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( wfsavgShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( wfsmaskShmimMonitorT );

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_autoUpdate, "update_auto" );
    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_updateOnce, "update_once" );
    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_dump, "update_dump" );
    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_zeroGains, "zero_gains" );

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_opticalGain,
                                 "opticalGain",
                                 0,
                                 1,
                                 0.01,
                                 "%0.01f",
                                 "Optical Gain",
                                 "Gain Opt." );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_gainGain, "gainGain", 0, 1, 0.01, "%0.01f", "Gain Gain", "Gain Opt." );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_gainLeak, "gainLeak", 0, 1, 0.01, "%0.02f", "Gain Leak", "Gain Opt." );
    if( createStandardIndiSelectionSw( m_indiP_extrapMethod,
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
                                       "Extrapolation" ) < 0 )
    {
        log<software_error>( { "error from createStandardIndiSelectionSw" } );
        return -1;
    }
    if( registerIndiPropertyNew( m_indiP_extrapMethod, INDI_NEWCALLBACK( m_indiP_extrapMethod ) ) < 0 )
    {
        log<software_error>( { "error from registerIndiPropertyNew" } );
        return -1;
    }
    if( createStandardIndiSelectionSw( m_indiP_extrapNoiseEstimateDomain,
                                       "extrap_noiseEstimateDomain",
                                       { extrapNoiseEstimateDomainElement( c_extrapNoiseEstimateOpenLoop ),
                                         extrapNoiseEstimateDomainElement( c_extrapNoiseEstimateClosedLoopPreXfer ) },
                                       { extrapNoiseEstimateDomainLabel( c_extrapNoiseEstimateOpenLoop ),
                                         extrapNoiseEstimateDomainLabel( c_extrapNoiseEstimateClosedLoopPreXfer ) },
                                       "Noise Estimate Domain",
                                       "Extrapolation" ) < 0 )
    {
        log<software_error>( { "error from createStandardIndiSelectionSw" } );
        return -1;
    }
    if( registerIndiPropertyNew( m_indiP_extrapNoiseEstimateDomain,
                                 INDI_NEWCALLBACK( m_indiP_extrapNoiseEstimateDomain ) ) < 0 )
    {
        log<software_error>( { "error from registerIndiPropertyNew" } );
        return -1;
    }
    if( createStandardIndiSelectionSw( m_indiP_extrapNoiseEstimateRange,
                                       "extrap_noiseEstimateRange",
                                       { extrapNoiseEstimateRangeElement( c_extrapNoiseEstimateHighFreq ),
                                         extrapNoiseEstimateRangeElement( c_extrapNoiseEstimateLowFreq ) },
                                       { extrapNoiseEstimateRangeLabel( c_extrapNoiseEstimateHighFreq ),
                                         extrapNoiseEstimateRangeLabel( c_extrapNoiseEstimateLowFreq ) },
                                       "Noise Estimate Range",
                                       "Extrapolation" ) < 0 )
    {
        log<software_error>( { "error from createStandardIndiSelectionSw" } );
        return -1;
    }
    if( registerIndiPropertyNew( m_indiP_extrapNoiseEstimateRange,
                                 INDI_NEWCALLBACK( m_indiP_extrapNoiseEstimateRange ) ) < 0 )
    {
        log<software_error>( { "error from registerIndiPropertyNew" } );
        return -1;
    }
    if( createStandardIndiSelectionSw( m_indiP_extrapNoiseEstimateStatistic,
                                       "extrap_noiseEstimateStatistic",
                                       { extrapNoiseEstimateStatisticElement( c_extrapNoiseEstimatePercentile ),
                                         extrapNoiseEstimateStatisticElement( c_extrapNoiseEstimateMinimum ) },
                                       { extrapNoiseEstimateStatisticLabel( c_extrapNoiseEstimatePercentile ),
                                         extrapNoiseEstimateStatisticLabel( c_extrapNoiseEstimateMinimum ) },
                                       "Noise Estimate Statistic",
                                       "Extrapolation" ) < 0 )
    {
        log<software_error>( { "error from createStandardIndiSelectionSw" } );
        return -1;
    }
    if( registerIndiPropertyNew( m_indiP_extrapNoiseEstimateStatistic,
                                 INDI_NEWCALLBACK( m_indiP_extrapNoiseEstimateStatistic ) ) < 0 )
    {
        log<software_error>( { "error from registerIndiPropertyNew" } );
        return -1;
    }
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapNoiseEstimateLowFreqMaxHz,
                                 "extrap_noiseEstimateLowFreqMaxHz",
                                 0,
                                 10000,
                                 0.1,
                                 "%0.2f",
                                 "Noise Estimate Low-Freq Max",
                                 "Extrapolation" );
    if( createStandardIndiSelectionSw(
            m_indiP_extrapClosedLoopOlEstimateMethod,
            "extrap_closedLoopOlEstimateMethod",
            { extrapClosedLoopOlEstimateMethodElement( c_extrapClosedLoopOlEstimateEtfOnly ),
              extrapClosedLoopOlEstimateMethodElement( c_extrapClosedLoopOlEstimateNtfAware ) },
            { extrapClosedLoopOlEstimateMethodLabel( c_extrapClosedLoopOlEstimateEtfOnly ),
              extrapClosedLoopOlEstimateMethodLabel( c_extrapClosedLoopOlEstimateNtfAware ) },
            "Closed Loop OL Estimate Method",
            "Extrapolation" ) < 0 )
    {
        log<software_error>( { "error from createStandardIndiSelectionSw" } );
        return -1;
    }
    if( registerIndiPropertyNew( m_indiP_extrapClosedLoopOlEstimateMethod,
                                 INDI_NEWCALLBACK( m_indiP_extrapClosedLoopOlEstimateMethod ) ) < 0 )
    {
        log<software_error>( { "error from registerIndiPropertyNew" } );
        return -1;
    }
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapPowerLawIndex,
                                 "extrap_powerLawIndex",
                                 0,
                                 10,
                                 0.01,
                                 "%0.3f",
                                 "Power-Law Index",
                                 "Extrapolation" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapPowerLawNormFreq,
                                 "extrap_powerLawNormFreq",
                                 0,
                                 10000,
                                 0.1,
                                 "%0.2f",
                                 "Power-Law Norm Freq",
                                 "Extrapolation" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapPowerLawMatchFreq,
                                 "extrap_powerLawMatchFreq",
                                 0,
                                 10000,
                                 0.1,
                                 "%0.2f",
                                 "Power-Law Match Freq",
                                 "Extrapolation" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapPowerLawMatchFallbackWindowHz,
                                 "extrap_powerLawMatchFallbackWindowHz",
                                 0,
                                 1000,
                                 0.1,
                                 "%0.2f",
                                 "Power-Law Match Window",
                                 "Extrapolation" );
    if( createStandardIndiSelectionSw(
            m_indiP_extrapPowerLawCrossoverMode,
            "extrap_powerLawCrossoverMode",
            { extrapPowerLawCrossoverModeElement( c_extrapPowerLawCrossoverManual ),
              extrapPowerLawCrossoverModeElement( c_extrapPowerLawCrossoverAutoSmoothedCrossing ) },
            { extrapPowerLawCrossoverModeLabel( c_extrapPowerLawCrossoverManual ),
              extrapPowerLawCrossoverModeLabel( c_extrapPowerLawCrossoverAutoSmoothedCrossing ) },
            "Power-Law Crossover Mode",
            "Extrapolation" ) < 0 )
    {
        log<software_error>( { "error from createStandardIndiSelectionSw" } );
        return -1;
    }
    if( registerIndiPropertyNew( m_indiP_extrapPowerLawCrossoverMode,
                                 INDI_NEWCALLBACK( m_indiP_extrapPowerLawCrossoverMode ) ) < 0 )
    {
        log<software_error>( { "error from registerIndiPropertyNew" } );
        return -1;
    }
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapPowerLawAutoSmoothWidthHz,
                                 "extrap_powerLawAutoSmoothWidthHz",
                                 0,
                                 10000,
                                 0.1,
                                 "%0.2f",
                                 "Power-Law Auto Smooth Width",
                                 "Extrapolation" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapPowerLawAutoMaxFreqFraction,
                                 "extrap_powerLawAutoMaxFreqFraction",
                                 0,
                                 1,
                                 0.01,
                                 "%0.3f",
                                 "Power-Law Auto Max Freq Fraction",
                                 "Extrapolation" );
    if( createStandardIndiToggleSw( m_indiP_extrapFitPowerLawIndex,
                                    "extrap_fitPowerLawIndex",
                                    "Fit Power-Law Index",
                                    "Extrapolation" ) < 0 )
    {
        log<software_error>( { "error from createStandardIndiToggleSw" } );
        return -1;
    }
    if( registerIndiPropertyNew( m_indiP_extrapFitPowerLawIndex, INDI_NEWCALLBACK( m_indiP_extrapFitPowerLawIndex ) ) <
        0 )
    {
        log<software_error>( { "error from registerIndiPropertyNew" } );
        return -1;
    }
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapPowerLawOnlyAboveFreq,
                                 "extrap_powerLawOnlyAboveFreq",
                                 0,
                                 10000,
                                 0.1,
                                 "%0.2f",
                                 "Power-Law Only Above",
                                 "Extrapolation" );
    if( createStandardIndiToggleSw( m_indiP_extrapPowerLawFitIncludesMatchPoint,
                                    "extrap_powerLawFitIncludesMatchPoint",
                                    "Fit Includes Match Point",
                                    "Extrapolation" ) < 0 )
    {
        log<software_error>( { "error from createStandardIndiToggleSw" } );
        return -1;
    }
    if( registerIndiPropertyNew( m_indiP_extrapPowerLawFitIncludesMatchPoint,
                                 INDI_NEWCALLBACK( m_indiP_extrapPowerLawFitIncludesMatchPoint ) ) < 0 )
    {
        log<software_error>( { "error from registerIndiPropertyNew" } );
        return -1;
    }
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapPowerLawFitMinFreqHz,
                                 "extrap_powerLawFitMinFreqHz",
                                 0,
                                 10000,
                                 0.1,
                                 "%0.2f",
                                 "Fit Min Freq",
                                 "Extrapolation" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapPowerLawFitMaxFreqHz,
                                 "extrap_powerLawFitMaxFreqHz",
                                 0,
                                 10000,
                                 0.1,
                                 "%0.2f",
                                 "Fit Max Freq",
                                 "Extrapolation" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapPowerLawFitBinWidthHz,
                                 "extrap_powerLawFitBinWidthHz",
                                 0,
                                 10000,
                                 0.1,
                                 "%0.2f",
                                 "Fit Bin Width",
                                 "Extrapolation" );
    CREATE_REG_INDI_NEW_NUMBERI( m_indiP_extrapPowerLawBlendBins,
                                 "extrap_powerLawBlendBins",
                                 0,
                                 100,
                                 1,
                                 "%d",
                                 "Blend Bins",
                                 "Extrapolation" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapDropoutGapFactor,
                                 "extrap_dropoutGapFactor",
                                 0,
                                 1,
                                 0.01,
                                 "%0.3f",
                                 "Dropout Gap Factor",
                                 "Extrapolation" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapDropoutTinyFactor,
                                 "extrap_dropoutTinyFactor",
                                 0,
                                 1,
                                 1e-7,
                                 "%0.3e",
                                 "Dropout Tiny Factor",
                                 "Extrapolation" );
    CREATE_REG_INDI_NEW_NUMBERI( m_indiP_extrapDropoutMaxBins,
                                 "extrap_dropoutMaxBins",
                                 1,
                                 1000,
                                 1,
                                 "%d",
                                 "Dropout Max Bins",
                                 "Extrapolation" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapClSignificanceThreshold,
                                 "extrap_clSignificanceThreshold",
                                 0,
                                 1000,
                                 0.01,
                                 "%0.3f",
                                 "CL Significance Threshold",
                                 "Extrapolation" );
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_extrapClMinSignificantFraction,
                                 "extrap_clMinSignificantFraction",
                                 0,
                                 1,
                                 0.01,
                                 "%0.3f",
                                 "CL Min Significant Fraction",
                                 "Extrapolation" );

    REG_INDI_SETPROP( m_indiP_emg, m_wfsDevice, "emgain" );
    REG_INDI_SETPROP( m_indiP_psdTime, m_psdDevice, "psdTime" );
    REG_INDI_SETPROP( m_indiP_psdAvgTime, m_psdDevice, "psdAvgTime" );
    REG_INDI_SETPROP( m_indiP_loop, m_loopName, "loop_state" );
    REG_INDI_SETPROP( m_indiP_siGain, m_loopName, "loop_gain" );
    REG_INDI_SETPROP( m_indiP_siMult, m_loopName, "loop_multcoeff" );
    REG_INDI_SETPROP( m_indiP_pcGain, m_loopName, "loop_pcgain" );
    REG_INDI_SETPROP( m_indiP_pcMult, m_loopName, "loop_pcmultcoeff" );
    REG_INDI_SETPROP( m_indiP_pcOn, m_loopName, "loop_pcOn" );

    CREATE_REG_INDI_RO_NUMBER( m_indiP_modesOn, "num_modes", "number of modes", "Gain Opt." );
    indi::addNumberElement( m_indiP_modesOn, "current", 0, 2400, 1, "%d", "Applied Modes" );
    indi::addNumberElement( m_indiP_modesOn, "integrator", 0, 2400, 1, "%d", "SI optimal" );
    indi::addNumberElement( m_indiP_modesOn, "predictor", 0, 2400, 1, "%d", "LP optimal" );

    REG_INDI_SETPROP( m_indiP_opticalGainSource, m_opticalGainDevice, m_opticalGainProperty );

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_opticalGainUpdate, "track_optical_gain" );

    if( sem_init( &m_goptSemaphore, 0, 0 ) < 0 )
    {
        return log<software_critical, -1>( { errno, 0, "Initializing gopt semaphore" } );
    }
    m_goptSemaphoreInit = true;

    XWCAPP_THREAD_START( m_goptThread,
                         m_goptThreadInit,
                         m_goptThreadID,
                         m_goptThreadProp,
                         m_goptThreadPrio,
                         m_goptThreadCpuset,
                         "gainopt",
                         goptThreadStart );

    state( stateCodes::OPERATING );
    return 0;
}

int modalGainOpt::appLogic()
{
    if( telemeterT::appLogic() < 0 )
    {
        return log<software_error, -1>( { "error from telemeter appLogic" } );
    }

    SHMIMMONITORT_APP_LOGIC( psdShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( freqShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( gainFactShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( multFactShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( pcGainFactShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( pcMultFactShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( numpccoeffShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( acoeffShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( bcoeffShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( gainCalShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( gainCalFactShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( tauShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( noiseShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( wfsavgShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( wfsmaskShmimMonitorT );

    XWCAPP_THREAD_CHECK( m_goptThread, "gainopt" );

    SHMIMMONITORT_UPDATE_INDI( psdShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( freqShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( gainFactShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( multFactShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( pcGainFactShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( pcMultFactShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( numpccoeffShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( acoeffShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( bcoeffShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( gainCalShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( gainCalFactShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( tauShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( noiseShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( wfsavgShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( wfsmaskShmimMonitorT );

    bool autoUpdate = false;
    bool updateOnce = false;
    bool dump = false;
    bool zeroGains = false;
    bool opticalGainUpdate = false;
    float opticalGain = 0;
    float gainGain = 0;
    float gainLeak = 0;
    processPsdProcessorT::processModelConfig extrapConfig;
    int extrapOL = 0;
    int extrapNoiseEstimateDomain = 0;
    int extrapNoiseEstimateRange = 0;
    int extrapNoiseEstimateStatistic = 0;
    int extrapClosedLoopOlEstimateMethod = 0;
    int modesOn = 0;
    int modesOnSI = 0;
    int modesOnLP = 0;

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_goptMutex );

        autoUpdate = m_autoUpdate;
        updateOnce = m_updateOnce;
        dump = m_dump;
        zeroGains = m_zeroGains;
        opticalGainUpdate = m_opticalGainUpdate;
        opticalGain = m_opticalGain;
        gainGain = m_gainGain;
        gainLeak = m_gainLeak;
        extrapConfig = m_extrapConfig;
        extrapOL = m_extrapOL;
        extrapNoiseEstimateDomain = m_extrapNoiseEstimateDomain;
        extrapNoiseEstimateRange = m_extrapNoiseEstimateRange;
        extrapNoiseEstimateStatistic = m_extrapNoiseEstimateStatistic;
        extrapClosedLoopOlEstimateMethod = m_extrapClosedLoopOlEstimateMethod;
        modesOn = m_modesOn;
        modesOnSI = m_modesOnSI;
        modesOnLP = m_modesOnLP;
    }

    if( autoUpdate )
    {
        updateSwitchIfChanged( m_indiP_autoUpdate, "toggle", pcf::IndiElement::On, INDI_OK );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_autoUpdate, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    }

    if( updateOnce )
    {
        updateSwitchIfChanged( m_indiP_updateOnce, "request", pcf::IndiElement::On, INDI_OK );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_updateOnce, "request", pcf::IndiElement::Off, INDI_IDLE );
    }

    if( dump )
    {
        updateSwitchIfChanged( m_indiP_dump, "request", pcf::IndiElement::On, INDI_OK );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_dump, "request", pcf::IndiElement::Off, INDI_IDLE );
    }

    if( zeroGains )
    {
        updateSwitchIfChanged( m_indiP_zeroGains, "request", pcf::IndiElement::On, INDI_OK );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_zeroGains, "request", pcf::IndiElement::Off, INDI_IDLE );
    }

    if( opticalGainUpdate )
    {
        updateSwitchIfChanged( m_indiP_opticalGainUpdate, "toggle", pcf::IndiElement::On, INDI_OK );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_opticalGainUpdate, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    }

    updatesIfChanged<float>( m_indiP_opticalGain, { "current", "target" }, { opticalGain, opticalGain } );

    updatesIfChanged<float>( m_indiP_gainGain, { "current", "target" }, { gainGain, gainGain } );
    updatesIfChanged<float>( m_indiP_gainLeak, { "current", "target" }, { gainLeak, gainLeak } );
    indi::updateSelectionSwitchIfChanged( m_indiP_extrapMethod,
                                          olProcessMethodElement( extrapOL ),
                                          m_indiDriver,
                                          INDI_OK );
    indi::updateSelectionSwitchIfChanged( m_indiP_extrapNoiseEstimateDomain,
                                          extrapNoiseEstimateDomainElement( extrapNoiseEstimateDomain ),
                                          m_indiDriver,
                                          INDI_OK );
    indi::updateSelectionSwitchIfChanged( m_indiP_extrapNoiseEstimateRange,
                                          extrapNoiseEstimateRangeElement( extrapNoiseEstimateRange ),
                                          m_indiDriver,
                                          INDI_OK );
    indi::updateSelectionSwitchIfChanged( m_indiP_extrapNoiseEstimateStatistic,
                                          extrapNoiseEstimateStatisticElement( extrapNoiseEstimateStatistic ),
                                          m_indiDriver,
                                          INDI_OK );
    updatesIfChanged<float>( m_indiP_extrapNoiseEstimateLowFreqMaxHz,
                             { "current", "target" },
                             { extrapConfig.m_noiseEstimateLowFreqMaxHz, extrapConfig.m_noiseEstimateLowFreqMaxHz } );
    indi::updateSelectionSwitchIfChanged( m_indiP_extrapClosedLoopOlEstimateMethod,
                                          extrapClosedLoopOlEstimateMethodElement( extrapClosedLoopOlEstimateMethod ),
                                          m_indiDriver,
                                          INDI_OK );
    updatesIfChanged<float>( m_indiP_extrapPowerLawIndex,
                             { "current", "target" },
                             { extrapConfig.m_powerLawIndex, extrapConfig.m_powerLawIndex } );
    updatesIfChanged<float>( m_indiP_extrapPowerLawNormFreq,
                             { "current", "target" },
                             { extrapConfig.m_powerLawNormFreq, extrapConfig.m_powerLawNormFreq } );
    updatesIfChanged<float>( m_indiP_extrapPowerLawMatchFreq,
                             { "current", "target" },
                             { extrapConfig.m_powerLawMatchFreq, extrapConfig.m_powerLawMatchFreq } );
    updatesIfChanged<float>(
        m_indiP_extrapPowerLawMatchFallbackWindowHz,
        { "current", "target" },
        { extrapConfig.m_powerLawMatchFallbackWindowHz, extrapConfig.m_powerLawMatchFallbackWindowHz } );
    indi::updateSelectionSwitchIfChanged( m_indiP_extrapPowerLawCrossoverMode,
                                          extrapPowerLawCrossoverModeElement( m_extrapPowerLawCrossoverMode ),
                                          m_indiDriver,
                                          INDI_OK );
    updatesIfChanged<float>( m_indiP_extrapPowerLawAutoSmoothWidthHz,
                             { "current", "target" },
                             { extrapConfig.m_powerLawAutoSmoothWidthHz, extrapConfig.m_powerLawAutoSmoothWidthHz } );
    updatesIfChanged<float>(
        m_indiP_extrapPowerLawAutoMaxFreqFraction,
        { "current", "target" },
        { extrapConfig.m_powerLawAutoMaxFreqFraction, extrapConfig.m_powerLawAutoMaxFreqFraction } );
    updateSwitchIfChanged( m_indiP_extrapFitPowerLawIndex,
                           "toggle",
                           extrapConfig.m_fitPowerLawIndex ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           INDI_OK );
    updatesIfChanged<float>( m_indiP_extrapPowerLawOnlyAboveFreq,
                             { "current", "target" },
                             { extrapConfig.m_powerLawOnlyAboveFreq, extrapConfig.m_powerLawOnlyAboveFreq } );
    updateSwitchIfChanged( m_indiP_extrapPowerLawFitIncludesMatchPoint,
                           "toggle",
                           extrapConfig.m_powerLawFitIncludesMatchPoint ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           INDI_OK );
    updatesIfChanged<float>( m_indiP_extrapPowerLawFitMinFreqHz,
                             { "current", "target" },
                             { extrapConfig.m_powerLawFitMinFreqHz, extrapConfig.m_powerLawFitMinFreqHz } );
    updatesIfChanged<float>( m_indiP_extrapPowerLawFitMaxFreqHz,
                             { "current", "target" },
                             { extrapConfig.m_powerLawFitMaxFreqHz, extrapConfig.m_powerLawFitMaxFreqHz } );
    updatesIfChanged<float>( m_indiP_extrapPowerLawFitBinWidthHz,
                             { "current", "target" },
                             { extrapConfig.m_powerLawFitBinWidthHz, extrapConfig.m_powerLawFitBinWidthHz } );
    updatesIfChanged<int>( m_indiP_extrapPowerLawBlendBins,
                           { "current", "target" },
                           { extrapConfig.m_powerLawBlendBins, extrapConfig.m_powerLawBlendBins } );
    updatesIfChanged<float>( m_indiP_extrapDropoutGapFactor,
                             { "current", "target" },
                             { extrapConfig.m_dropoutGapFactor, extrapConfig.m_dropoutGapFactor } );
    updatesIfChanged<float>( m_indiP_extrapDropoutTinyFactor,
                             { "current", "target" },
                             { extrapConfig.m_dropoutTinyFactor, extrapConfig.m_dropoutTinyFactor } );
    updatesIfChanged<int>(
        m_indiP_extrapDropoutMaxBins,
        { "current", "target" },
        { static_cast<int>( extrapConfig.m_dropoutMaxBins ), static_cast<int>( extrapConfig.m_dropoutMaxBins ) } );
    updatesIfChanged<float>( m_indiP_extrapClSignificanceThreshold,
                             { "current", "target" },
                             { extrapConfig.m_clSignificanceThreshold, extrapConfig.m_clSignificanceThreshold } );
    updatesIfChanged<float>( m_indiP_extrapClMinSignificantFraction,
                             { "current", "target" },
                             { extrapConfig.m_clMinSignificantFraction, extrapConfig.m_clMinSignificantFraction } );

    updatesIfChanged<int>( m_indiP_modesOn,
                           { "current", "integrator", "predictor" },
                           { modesOn, modesOnSI, modesOnLP } );

    return 0;
}

int modalGainOpt::appShutdown()
{
    XWCAPP_THREAD_STOP( m_goptThread );

    SHMIMMONITORT_APP_SHUTDOWN( psdShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( freqShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( gainFactShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( multFactShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( pcGainFactShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( pcMultFactShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( numpccoeffShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( acoeffShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( bcoeffShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( gainCalShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( gainCalFactShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( tauShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( noiseShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( wfsavgShmimMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( wfsmaskShmimMonitorT );

    destroyImageStream( m_olPSDStream );
    destroyImageStream( m_rawOlPSDStream );
    destroyImageStream( m_smoothOlPSDStream );
    destroyImageStream( m_noisePSDStream );
    destroyImageStream( m_clXferCurrentStream );
    destroyImageStream( m_clNtfCurrentStream );
    destroyImageStream( m_clXferSIStream );
    destroyImageStream( m_clNtfSIStream );
    destroyImageStream( m_clXferLPStream );
    destroyImageStream( m_clNtfLPStream );

    destroyImageStream( m_optGainStream );
    destroyImageStream( m_optGainSIRawStream );
    destroyImageStream( m_optGainSIStream );
    destroyImageStream( m_maxGainSIStream );
    destroyImageStream( m_optGainLPStream );
    destroyImageStream( m_maxGainLPStream );
    destroyImageStream( m_modevarStream );

    if( m_goptSemaphoreInit )
    {
        sem_destroy( &m_goptSemaphore );
        m_goptSemaphoreInit = false;
    }

    telemeterT::appShutdown();

    return 0;
}

inline int modalGainOpt::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_modalgainopt() );
}

inline int modalGainOpt::recordTelem( const telem_modalgainopt * )
{
    return recordModalGainOpt( true );
}

inline int modalGainOpt::recordModalGainOpt( bool force )
{
    static bool lastAutoUpdate{ false };
    static bool lastOpticalGainUpdate{ false };
    static float lastOpticalGain{ -1e6F };
    static float lastGainGain{ -1e6F };
    static float lastGainLeak{ -1e6F };

    bool autoUpdate = false;
    bool opticalGainUpdate = false;
    float opticalGain = 0;
    float gainGain = 0;
    float gainLeak = 0;

    {
        std::lock_guard<std::mutex> lock( m_goptMutex );
        autoUpdate = m_autoUpdate;
        opticalGainUpdate = m_opticalGainUpdate;
        opticalGain = m_opticalGain;
        gainGain = m_gainGain;
        gainLeak = m_gainLeak;
    }

    if( force || autoUpdate != lastAutoUpdate || opticalGainUpdate != lastOpticalGainUpdate ||
        opticalGain != lastOpticalGain || gainGain != lastGainGain || gainLeak != lastGainLeak )
    {
        lastAutoUpdate = autoUpdate;
        lastOpticalGainUpdate = opticalGainUpdate;
        lastOpticalGain = opticalGain;
        lastGainGain = gainGain;
        lastGainLeak = gainLeak;

        telem<telem_modalgainopt>( { autoUpdate, opticalGainUpdate, opticalGain, gainGain, gainLeak } );
    }

    return 0;
}

void modalGainOpt::destroyImageStream( IMAGE *&stream )
{
    if( stream == nullptr )
    {
        return;
    }

    ImageStreamIO_destroyIm( stream );
    free( stream );
    stream = nullptr;
}

int modalGainOpt::createImageStream(
    IMAGE *&stream, const std::string &name, uint32_t size0, uint32_t size1, uint32_t size2, uint8_t dataType )
{
    destroyImageStream( stream );

    stream = static_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );
    if( stream == nullptr )
    {
        return log<software_error, -1>( { "error allocating stream for " + name } );
    }

    uint32_t imsize[3];
    imsize[0] = size0;
    imsize[1] = size1;
    imsize[2] = size2;

    if( ImageStreamIO_createIm_gpu( stream,
                                    name.c_str(),
                                    3,
                                    imsize,
                                    dataType,
                                    -1,
                                    1,
                                    IMAGE_NB_SEMAPHORE,
                                    0,
                                    CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                    0 ) != 0 )
    {
        free( stream );
        stream = nullptr;
        return log<software_error, -1>( { "error creating stream for " + name } );
    }

    if( stream->md == nullptr )
    {
        destroyImageStream( stream );
        return log<software_error, -1>( { "stream metadata not initialized for " + name } );
    }

    stream->md->cnt0 = 0;
    stream->md->cnt1 = 0;

    return 0;
}

void modalGainOpt::writePublishedGainArrays( float *currentData,
                                             float *siRawData,
                                             float *siData,
                                             float *maxSiData,
                                             float *lpData,
                                             float *maxLpData,
                                             float *modeVarData )
{
    mx::improc::eigenMap<float> modeVars( modeVarData, 3, m_modeVarSI.size() );

    for( size_t n = 0; n < m_optGainSI.size(); ++n )
    {
        currentData[n] = ( m_gainCalFacts[n] * m_optGainSI[n] / m_gainCals[n] ) / m_opticalGain;
        siRawData[n] = ( m_gainCalFacts[n] * m_optGainSIRaw[n] / m_gainCals[n] ) / m_opticalGain;
        siData[n] = currentData[n];
        maxSiData[n] = ( m_gainCalFacts[n] * m_gmaxSI[n] / m_gainCals[n] ) / m_opticalGain;

        lpData[n] = ( m_gainCalFacts[n] * m_optGainLP[n] / m_gainCals[n] ) / m_opticalGain;
        maxLpData[n] = ( m_gainCalFacts[n] * m_gmaxLP[n] / m_gainCals[n] ) / m_opticalGain;

        modeVars( 0, n ) = m_modeVarOL[n];
        modeVars( 1, n ) = m_modeVarSI[n];
        modeVars( 2, n ) = m_modeVarLP[n];
    }
}

void modalGainOpt::writePublishedPredictorArrays(
    float *pcGainData, float *aCoeffData, uint32_t aWidth, float *bCoeffData, uint32_t bWidth, bool blend )
{
    for( size_t n = 0; n < m_optGainLP.size(); ++n )
    {
        const float targetGain = ( m_gainCalFacts[n] * m_optGainLP[n] / m_gainCals[n] ) / m_opticalGain;

        if( blend )
        {
            pcGainData[n] = pcGainData[n] + m_gainGain * ( targetGain - pcGainData[n] );
        }
        else
        {
            pcGainData[n] = targetGain;
        }

        const size_t aBase = n * aWidth;
        const size_t bBase = n * bWidth;

        aCoeffData[aBase] = m_Na[n];
        bCoeffData[bBase] = m_Nb[n];

        for( uint32_t k = 0; k < m_Na[n]; ++k )
        {
            if( blend )
            {
                aCoeffData[aBase + 1 + k] =
                    aCoeffData[aBase + 1 + k] + m_gainGain * ( m_goptLP[n].a()[k] - aCoeffData[aBase + 1 + k] );
            }
            else
            {
                aCoeffData[aBase + 1 + k] = m_goptLP[n].a()[k];
            }
        }
        for( uint32_t k = m_Na[n]; k < aWidth - 1; ++k )
        {
            aCoeffData[aBase + 1 + k] = 0;
        }

        for( uint32_t k = 0; k < m_Nb[n]; ++k )
        {
            if( blend )
            {
                bCoeffData[bBase + 1 + k] =
                    bCoeffData[bBase + 1 + k] + m_gainGain * ( m_goptLP[n].b()[k] - bCoeffData[bBase + 1 + k] );
            }
            else
            {
                bCoeffData[bBase + 1 + k] = m_goptLP[n].b()[k];
            }
        }
        for( uint32_t k = m_Nb[n]; k < bWidth - 1; ++k )
        {
            bCoeffData[bBase + 1 + k] = 0;
        }
    }
}

int modalGainOpt::countEnabledGainFactors( const std::vector<float> &gainFacts ) const
{
    int modesOn = 0;

    for( size_t n = 0; n < gainFacts.size(); ++n )
    {
        if( gainFacts[n] > 0 )
        {
            ++modesOn;
        }
    }

    return modesOn;
}

void modalGainOpt::updateAppliedModeCount( const std::vector<float> &gainFacts, bool predictorPath )
{
    if( predictorPath != m_pcOn )
    {
        return;
    }

    m_modesOn = countEnabledGainFactors( gainFacts );
}

bool modalGainOpt::applyGainFactorUpdate( std::vector<float> &gainFacts,
                                          const float *incoming,
                                          uint32_t width,
                                          bool predictorPath )
{
    bool change = false;

    if( width != gainFacts.size() )
    {
        gainFacts.resize( width );
        change = true;
    }

    for( uint32_t n = 0; n < width; ++n )
    {
        if( change || gainFacts[n] != incoming[n] )
        {
            gainFacts[n] = incoming[n];
            change = true;
        }
    }

    if( !change )
    {
        return false;
    }

    if( m_loop )
    {
        m_sinceChange = -1;
    }

    if( !predictorPath )
    {
        m_siGainStateNeedsSync = true;
    }

    updateAppliedModeCount( gainFacts, predictorPath );

    return true;
}

bool modalGainOpt::applyMultiplierUpdate( std::vector<float> &multFacts,
                                          const float *incoming,
                                          uint32_t width,
                                          bool predictorPath )
{
    bool change = false;

    if( width != multFacts.size() )
    {
        multFacts.resize( width );
        change = true;
    }

    for( uint32_t n = 0; n < width; ++n )
    {
        if( change || multFacts[n] != incoming[n] )
        {
            multFacts[n] = incoming[n];
            change = true;
        }
    }

    if( !change )
    {
        return false;
    }

    if( m_loop )
    {
        m_sinceChange = -1;
    }

    if( predictorPath )
    {
        m_pcgoptUpdated = true;
    }
    else
    {
        m_goptUpdated = true;
    }

    return true;
}

bool modalGainOpt::applyFrequencyUpdate( const float *incoming, size_t size )
{
    bool change = false;

    if( size != m_freq.size() )
    {
        m_freq.resize( size );
        change = true;
    }

    for( size_t n = 0; n < size; ++n )
    {
        if( change || m_freq[n] != incoming[n] )
        {
            m_freq[n] = incoming[n];
            change = true;
        }
    }

    if( !change )
    {
        return false;
    }

    m_fps = 2 * m_freq.back();

    m_sinceChange = -1;
    m_goptUpdated = true;
    m_freqUpdated = true;

    return true;
}

bool modalGainOpt::refreshGoptStructures()
{
    if( !( m_goptUpdated || m_pcgoptUpdated || m_freqUpdated || m_goptCurrent.size() != m_gainFacts.size() ) )
    {
        return false;
    }

    if( m_goptCurrent.size() != m_gainFacts.size() )
    {
        m_freqUpdated = true; // force freq update in this case
    }

    std::cerr << "updating gopt structures\n";

    m_goptCurrent.resize( m_gainFacts.size() );
    m_goptSI.resize( m_gainFacts.size() );
    m_goptLP.resize( m_gainFacts.size() );
    m_linPred.resize( m_gainFacts.size() );

    for( size_t n = 0; n < m_goptCurrent.size(); ++n )
    {
        m_goptCurrent[n].Ti( 1.0 / m_fps );
        m_goptCurrent[n].tau( m_taus[n] );

        m_goptSI[n].Ti( 1.0 / m_fps );
        m_goptSI[n].tau( m_taus[n] );

        m_goptLP[n].Ti( 1.0 / m_fps );
        m_goptLP[n].tau( m_taus[n] );

        if( !m_pcOn )
        {
            m_goptCurrent[n].setLeakyIntegrator( m_mult * m_multFacts[n] );
        }
        else
        {
            std::vector<float> ta( m_NaCurrent[n] );
            for( size_t m = 0; m < ta.size(); ++m )
            {
                ta[m] = m_as( m, n );
            }
            m_goptCurrent[n].a( ta );

            std::vector<float> tb( m_NbCurrent[n] );
            for( size_t m = 0; m < tb.size(); ++m )
            {
                tb[m] = m_bs( m, n );
            }
            m_goptCurrent[n].b( tb );

            m_goptCurrent[n].remember( m_pcMult * m_pcMultFacts[n] );
        }

        m_goptSI[n].setLeakyIntegrator( m_mult * m_multFacts[n] );

        if( m_freqUpdated )
        {
            m_goptCurrent[n].f( m_freq );
            m_goptSI[n].f( m_freq );
            m_goptLP[n].f( m_freq );
        }

        m_gmaxSI[n] = m_goptSI[n].maxStableGain();
    }

    m_goptUpdated = false;
    m_pcgoptUpdated = false;
    m_freqUpdated = false;

    std::cerr << "done.\n";
    return true;
}

void modalGainOpt::syncSiGainStateFromAppliedGains()
{
    const float tiny = std::numeric_limits<float>::min();

    if( m_optGainSI.size() != m_gainFacts.size() )
    {
        m_optGainSI.resize( m_gainFacts.size(), 0.0F );
    }

    if( m_optGainSIRaw.size() != m_gainFacts.size() )
    {
        m_optGainSIRaw.resize( m_gainFacts.size(), 0.0F );
    }

    if( m_gainFacts.size() != m_gainCals.size() || m_gainFacts.size() != m_gainCalFacts.size() ||
        std::abs( m_opticalGain ) <= tiny )
    {
        std::fill( m_optGainSI.begin(), m_optGainSI.end(), 0.0F );
        m_siGainStateNeedsSync = false;
        return;
    }

    for( size_t n = 0; n < m_gainFacts.size(); ++n )
    {
        if( std::abs( m_gainCalFacts[n] ) <= tiny )
        {
            m_optGainSI[n] = 0.0F;
            continue;
        }

        m_optGainSI[n] = m_gainFacts[n] * m_gainCals[n] * m_opticalGain / m_gainCalFacts[n];
    }

    m_siGainStateNeedsSync = false;
}

void modalGainOpt::updateIntegratedSiGain( size_t modeIndex )
{
    m_optGainSI[modeIndex] =
        m_gainGain * ( m_optGainSIRaw[modeIndex] - m_optGainSI[modeIndex] ) + m_gainLeak * m_optGainSI[modeIndex];
}

int modalGainOpt::allocatePCShmims()
{
    // mutex should be locked before calling this

    if( numpccoeffShmimMonitorT::m_smState != dev::shmimMonitorState::connected ||
        numpccoeffShmimMonitorT::m_width != m_nModes || numpccoeffShmimMonitorT::m_height != 2 )
    {
        mx::improc::eigenImage<uint32_t> Npc( m_nModes, 2 );
        Npc.setConstant( m_defaultNCoeff );

        if( numpccoeffShmimMonitorT::create( m_nModes, 2, 1, _DATATYPE_UINT32, Npc.data() ) != 0 )
        {
            return log<software_error, -1>( { "error creating numpccoeffShmim" } );
        }

        MGO_BREADCRUMB;

        std::cerr << "created numppccoeff shmim\n";
    }

    if( acoeffShmimMonitorT::m_smState != dev::shmimMonitorState::connected ||
        acoeffShmimMonitorT::m_width != m_maxNCoeff + 1 || acoeffShmimMonitorT::m_height != m_nModes )
    {
        if( acoeffShmimMonitorT::create( m_maxNCoeff + 1, m_nModes, 1, _DATATYPE_FLOAT ) != 0 )
        {
            return log<software_error, -1>( { "error creating acoeffShmim" } );
        }

        std::cerr << "created acoeff shmim\n";
    }

    if( bcoeffShmimMonitorT::m_smState != dev::shmimMonitorState::connected ||
        bcoeffShmimMonitorT::m_width != m_maxNCoeff || bcoeffShmimMonitorT::m_height != m_nModes )
    {
        if( bcoeffShmimMonitorT::create( m_maxNCoeff + 1, m_nModes, 1, _DATATYPE_FLOAT ) != 0 )
        {
            return log<software_error, -1>( { "error creating bcoeffShmim" } );
        }

        std::cerr << "created bcoeff shmim\n";
    }

    if( pcGainFactShmimMonitorT::m_smState != dev::shmimMonitorState::connected ||
        pcGainFactShmimMonitorT::m_width != m_nModes || pcGainFactShmimMonitorT::m_height != 1 )
    {
        if( pcGainFactShmimMonitorT::create( m_nModes, 1, 1, _DATATYPE_FLOAT ) != 0 )
        {
            return log<software_error, -1>( { "error creating pcGainFactShmim" } );
        }

        std::cerr << "created pcGainFact shmim\n";
    }

    if( pcMultFactShmimMonitorT::m_smState != dev::shmimMonitorState::connected ||
        pcMultFactShmimMonitorT::m_width != m_nModes || pcMultFactShmimMonitorT::m_height != 1 )
    {
        if( pcMultFactShmimMonitorT::create( m_nModes, 1, 1, _DATATYPE_FLOAT ) != 0 )
        {
            return log<software_error, -1>( { "error creating pcMultFactShmim" } );
        }

        std::cerr << "created pcMultFact shmim\n";
    }

    return 0;
}

int modalGainOpt::allocate( const psdShmimT &dummy )
{
    static_cast<void>( dummy );

    m_updating = true;
    std::lock_guard<std::mutex> lock( m_goptMutex );

    m_updating = true;

    m_nFreq = psdShmimMonitorT::m_width;
    m_nModes = psdShmimMonitorT::m_height;

    m_clPSDs.resize( m_nFreq, m_nModes );

    m_clXferCurrent.resize( m_nFreq, m_nModes );
    m_clNtfCurrent.resize( m_nFreq, m_nModes );
    m_clXferSI.resize( m_nFreq, m_nModes );
    m_clNtfSI.resize( m_nFreq, m_nModes );
    m_clXferLP.resize( m_nFreq, m_nModes );
    m_clNtfLP.resize( m_nFreq, m_nModes );

    m_olPSDs.resize( m_nModes );
    m_rawOlPSDs.resize( m_nModes );
    m_smoothOlPSDs.resize( m_nModes );
    m_nPSDs.resize( m_nModes );
    for( size_t n = 0; n < m_olPSDs.size(); ++n )
    {
        m_olPSDs[n].resize( m_nFreq );
        m_rawOlPSDs[n].resize( m_nFreq );
        m_smoothOlPSDs[n].resize( m_nFreq );
        m_nPSDs[n].resize( m_nFreq );
    }

    m_modeVarOL.resize( m_nModes );

    m_optGainSIRaw.resize( m_nModes );
    m_optGainSI.resize( m_nModes );
    m_gmaxSI.resize( m_nModes );
    m_modeVarSI.resize( m_nModes );
    m_timesOnSI.resize( m_nModes, 5 );

    m_optGainLP.resize( m_nModes );
    m_modeVarLP.resize( m_nModes );
    m_timesOnLP.resize( m_nModes, 5 );
    m_siGainStateNeedsSync = true;

    if( m_olPSDStream != nullptr &&
        ( m_olPSDStream->md->size[0] != m_nFreq || m_olPSDStream->md->size[1] != m_nModes ) )
    {
        destroyImageStream( m_olPSDStream );
        destroyImageStream( m_rawOlPSDStream );
        destroyImageStream( m_smoothOlPSDStream );
        destroyImageStream( m_noisePSDStream );
        destroyImageStream( m_clXferCurrentStream );
        destroyImageStream( m_clNtfCurrentStream );
        destroyImageStream( m_clXferSIStream );
        destroyImageStream( m_clNtfSIStream );
        destroyImageStream( m_clXferLPStream );
        destroyImageStream( m_clNtfLPStream );
    }

    if( m_olPSDStream == nullptr )
    {
        if( createImageStream( m_olPSDStream, m_olPSDShmimName, m_nFreq, m_nModes, 1, psdShmimMonitorT::m_dataType ) <
            0 )
        {
            return -1;
        }

        if( createImageStream( m_rawOlPSDStream,
                               m_rawOlPSDShmimName,
                               m_nFreq,
                               m_nModes,
                               1,
                               psdShmimMonitorT::m_dataType ) < 0 )
        {
            return -1;
        }

        if( createImageStream( m_smoothOlPSDStream,
                               m_smoothOlPSDShmimName,
                               m_nFreq,
                               m_nModes,
                               1,
                               psdShmimMonitorT::m_dataType ) < 0 )
        {
            return -1;
        }

        if( createImageStream( m_noisePSDStream,
                               m_noisePSDShmimName,
                               m_nFreq,
                               m_nModes,
                               1,
                               psdShmimMonitorT::m_dataType ) < 0 )
        {
            return -1;
        }

        if( createImageStream( m_clXferCurrentStream,
                               m_clXferCurrentShmimName,
                               m_nFreq,
                               m_nModes,
                               1,
                               psdShmimMonitorT::m_dataType ) < 0 )
        {
            return -1;
        }

        if( createImageStream( m_clNtfCurrentStream,
                               m_clNtfCurrentShmimName,
                               m_nFreq,
                               m_nModes,
                               1,
                               psdShmimMonitorT::m_dataType ) < 0 )
        {
            return -1;
        }

        if( createImageStream( m_clXferSIStream,
                               m_clXferSIShmimName,
                               m_nFreq,
                               m_nModes,
                               1,
                               psdShmimMonitorT::m_dataType ) < 0 )
        {
            return -1;
        }

        if( createImageStream( m_clNtfSIStream,
                               m_clNtfSIShmimName,
                               m_nFreq,
                               m_nModes,
                               1,
                               psdShmimMonitorT::m_dataType ) < 0 )
        {
            return -1;
        }

        if( createImageStream( m_clXferLPStream,
                               m_clXferLPShmimName,
                               m_nFreq,
                               m_nModes,
                               1,
                               psdShmimMonitorT::m_dataType ) < 0 )
        {
            return -1;
        }

        if( createImageStream( m_clNtfLPStream,
                               m_clNtfLPShmimName,
                               m_nFreq,
                               m_nModes,
                               1,
                               psdShmimMonitorT::m_dataType ) < 0 )
        {
            return -1;
        }
    }

    if( m_optGainStream != nullptr &&
        ( m_optGainStream->md->size[0] != psdShmimMonitorT::m_height || m_optGainStream->md->size[1] != 1 ) )
    {
        destroyImageStream( m_optGainStream );
        destroyImageStream( m_optGainSIRawStream );
        destroyImageStream( m_optGainSIStream );
        destroyImageStream( m_maxGainSIStream );
        destroyImageStream( m_optGainLPStream );
        destroyImageStream( m_maxGainLPStream );
        destroyImageStream( m_modevarStream );
    }

    if( m_optGainStream == nullptr )
    {
        if( createImageStream( m_optGainStream, m_optGainShmimName, m_nModes, 1, 1, psdShmimMonitorT::m_dataType ) < 0 )
        {
            return -1;
        }

        if( createImageStream( m_optGainSIStream, m_optGainSIShmimName, m_nModes, 1, 1, psdShmimMonitorT::m_dataType ) <
            0 )
        {
            return -1;
        }

        if( createImageStream( m_optGainSIRawStream,
                               m_optGainSIRawShmimName,
                               m_nModes,
                               1,
                               1,
                               psdShmimMonitorT::m_dataType ) < 0 )
        {
            return -1;
        }

        if( createImageStream( m_maxGainSIStream, m_maxGainSIShmimName, m_nModes, 1, 1, psdShmimMonitorT::m_dataType ) <
            0 )
        {
            return -1;
        }

        if( createImageStream( m_optGainLPStream, m_optGainLPShmimName, m_nModes, 1, 1, psdShmimMonitorT::m_dataType ) <
            0 )
        {
            return -1;
        }

        if( createImageStream( m_maxGainLPStream, m_maxGainLPShmimName, m_nModes, 1, 1, psdShmimMonitorT::m_dataType ) <
            0 )
        {
            return -1;
        }

        if( createImageStream( m_modevarStream, m_modevarShmimName, 3, m_nModes, 1, psdShmimMonitorT::m_dataType ) < 0 )
        {
            return -1;
        }
    }

    m_sinceChange = -1;

    m_updating = false;
    return 0;
}

int modalGainOpt::processImage( void *curr_src, const psdShmimT &dummy )
{
    static_cast<void>( dummy );

    ++m_sinceChange;

    if( m_psdAvgTime <= 0 || m_psdTime <= 0 ) // Safety check, shouldn't happen but means we need to wait.
    {
        return 0;
    }

    int deadTime = ( m_psdAvgTime / m_psdTime ) / m_psdOverlapFraction;

    if( m_sinceChange < deadTime )
    {
        return 0;
    }

    // Here we would update psds, but don't do that if we're in the middle of
    // calculating
    std::unique_lock<std::mutex> lock( m_goptMutex, std::try_to_lock );
    if( !lock.owns_lock() )
    {
        ///\todo update a frame-missed counter
        return 0;
    }

    m_updating = true;

    m_clPSDs = Eigen::Map<Eigen::Array<float, -1, -1>>( static_cast<float *>( curr_src ),
                                                        psdShmimMonitorT::m_width,
                                                        psdShmimMonitorT::m_height );

    m_updating = false;

    lock.unlock();

    if( sem_post( &m_goptSemaphore ) < 0 )
    {
        return log<software_critical, -1>( { errno, 0, "Error posting to semaphore" } );
    }

    return 0;
}

int modalGainOpt::allocate( const freqShmimT &dummy )
{
    static_cast<void>( dummy );

    return 0;
}

int modalGainOpt::processImage( void *curr_src, const freqShmimT &dummy )
{
    static_cast<void>( dummy );

    if( freqShmimMonitorT::m_width != 1 )
    {
        return log<software_error, -1>( { "got freq with width not 1" } );
    }

    float *f = static_cast<float *>( curr_src );
    size_t sz = freqShmimMonitorT::m_height;

    bool sizeChange = ( sz != m_freq.size() );
    bool dataChange = false;

    if( !sizeChange )
    {
        for( size_t n = 0; n < sz; ++n )
        {
            if( f[n] != m_freq[n] )
            {
                dataChange = true;
                break;
            }
        }
    }

    if( sizeChange || dataChange )
    {
        m_updating = true;
        std::lock_guard<std::mutex> lock( m_goptMutex );

        m_updating = true; // Make sure it didn't get set to false by thread that
                           // had the lock

        if( applyFrequencyUpdate( f, sz ) )
        {
            std::cerr << "got freq: " << sz << '\n';
            std::cerr << "     fps: " << m_fps << '\n';
        }

        m_updating = false;
    }

    return 0;
}

int modalGainOpt::allocate( const gainFactShmimT &dummy )
{
    static_cast<void>( dummy );

    if( gainFactShmimMonitorT::m_height != 1 )
    {
        return log<software_error, -1>( { "got gains with height not 1" } );
    }

    return 0;
}

int modalGainOpt::processImage( void *curr_src, const gainFactShmimT &dummy )
{
    static_cast<void>( dummy );

    uint32_t w = gainFactShmimMonitorT::m_width;
    float *g = static_cast<float *>( curr_src );

    std::unique_lock<std::mutex> lock( m_goptMutex, std::defer_lock );

    bool sizeChange = ( w != m_gainFacts.size() );
    bool dataChange = false;

    if( !sizeChange )
    {
        for( uint32_t n = 0; n < w; ++n )
        {
            if( m_gainFacts[n] != g[n] )
            {
                dataChange = true;
                break;
            }
        }
    }

    if( sizeChange || dataChange )
    {
        m_updating = true;
        lock.lock();
        m_updating = true; // Make sure it didn't get set to false by thread that
                           // had the lock

        if( applyGainFactorUpdate( m_gainFacts, g, w, false ) )
        {
            m_updating = false;
            std::cerr << "got gains: " << m_gainFacts.size() << "\n";

            lock.unlock();
        }
        else
        {
            m_updating = false;
        }
    }

    return 0;
}

int modalGainOpt::allocate( const multFactShmimT &dummy )
{
    static_cast<void>( dummy );

    if( multFactShmimMonitorT::m_height != 1 )
    {
        return log<software_error, -1>( { "got multcoeffs with height not 1" } );
    }

    return 0;
}

int modalGainOpt::processImage( void *curr_src, const multFactShmimT &dummy )
{
    static_cast<void>( dummy );

    uint32_t w = multFactShmimMonitorT::m_width;
    float *m = static_cast<float *>( curr_src );

    std::unique_lock<std::mutex> lock( m_goptMutex, std::defer_lock );

    bool sizeChange = ( w != m_multFacts.size() );
    bool dataChange = false;

    if( !sizeChange )
    {
        for( uint32_t n = 0; n < w; ++n )
        {
            if( m_multFacts[n] != m[n] )
            {
                dataChange = true;
                break;
            }
        }
    }

    if( sizeChange || dataChange )
    {
        m_updating = true;
        lock.lock();
        m_updating = true; // Make sure it didn't get set to false by thread that
                           // had the lock

        if( applyMultiplierUpdate( m_multFacts, m, w, false ) )
        {
            m_updating = false;
            std::cerr << "got mcs: " << m_multFacts.size() << " " << w << "\n";

            lock.unlock();
        }
        else
        {
            m_updating = false;
        }
    }

    return 0;
}

int modalGainOpt::allocate( const pcGainFactShmimT &dummy )
{
    static_cast<void>( dummy );

    if( pcGainFactShmimMonitorT::m_height != 1 )
    {
        return log<software_error, -1>( { "got pc gains with height not 1" } );
    }

    return 0;
}

int modalGainOpt::processImage( void *curr_src, const pcGainFactShmimT &dummy )
{
    static_cast<void>( dummy );

    uint32_t w = pcGainFactShmimMonitorT::m_width;
    float *g = static_cast<float *>( curr_src );

    std::unique_lock<std::mutex> lock( m_goptMutex, std::defer_lock );

    bool sizeChange = ( w != m_pcGainFacts.size() );
    bool dataChange = false;

    if( !sizeChange )
    {
        for( uint32_t n = 0; n < w; ++n )
        {
            if( m_pcGainFacts[n] != g[n] )
            {
                dataChange = true;
                break;
            }
        }
    }

    if( sizeChange || dataChange )
    {
        m_updating = true;
        lock.lock();
        m_updating = true; // Make sure it didn't get set to false by thread that
                           // had the lock

        if( applyGainFactorUpdate( m_pcGainFacts, g, w, true ) )
        {
            m_updating = false;
            std::cerr << "got pc gains: " << m_pcGainFacts.size() << "\n";

            lock.unlock();
        }
        else
        {
            m_updating = false;
        }
    }

    return 0;
}

int modalGainOpt::allocate( const pcMultFactShmimT &dummy )
{
    static_cast<void>( dummy );

    if( pcMultFactShmimMonitorT::m_height != 1 )
    {
        return log<software_error, -1>( { "got pcMultcoeffs with height not 1" } );
    }

    return 0;
}

int modalGainOpt::processImage( void *curr_src, const pcMultFactShmimT &dummy )
{
    static_cast<void>( dummy );

    uint32_t w = pcMultFactShmimMonitorT::m_width;
    float *m = static_cast<float *>( curr_src );

    std::unique_lock<std::mutex> lock( m_goptMutex, std::defer_lock );

    bool sizeChange = ( w != m_pcMultFacts.size() );
    bool dataChange = false;

    if( !sizeChange )
    {
        for( uint32_t n = 0; n < w; ++n )
        {
            if( m_pcMultFacts[n] != m[n] )
            {
                dataChange = true;
                break;
            }
        }
    }

    if( sizeChange || dataChange )
    {
        m_updating = true;
        lock.lock();
        m_updating = true; // Make sure it didn't get set to false by thread that
                           // had the lock

        if( applyMultiplierUpdate( m_pcMultFacts, m, w, true ) )
        {
            m_updating = false;

            lock.unlock();
            std::cerr << "got mcs: " << m_pcMultFacts.size() << "\n";
        }
        else
        {
            m_updating = false;
        }
    }

    return 0;
}

int modalGainOpt::allocate( const numpccoeffShmimT &dummy )
{
    static_cast<void>( dummy );

    if( numpccoeffShmimMonitorT::m_height != 2 )
    {
        return log<software_error, -1>( { "got numpccoeff's with height not 2" } );
    }

    std::cerr << "numpccoeffShmimMonitorT::allocate\n";
    return 0;
}

int modalGainOpt::processImage( void *curr_src, const numpccoeffShmimT &dummy )
{
    static_cast<void>( dummy );

    bool change = false;

    uint32_t w = numpccoeffShmimMonitorT::m_width;

    std::unique_lock<std::mutex> lock( m_goptMutex, std::defer_lock );

    if( w != m_Na.size() || w != m_Nb.size() || w != m_regCounter.size() || w != m_regScale.size() )
    {
        m_updating = true;
        lock.lock();
        m_updating = true; // Make sure it didn't get set to false by thread that
                           // had the lock

        change = true;
        m_Na.resize( w );
        m_Nb.resize( w );

        m_regCounter.resize( w );

        // Initialize the regCounter
        int nc = 0;

        for( size_t n = 0; n < m_regCounter.size(); ++n )
        {
            m_regCounter[n] = nc;
            ++nc;

            if( nc >= m_nRegCycles )
            {
                nc = 0;
            }
        }

        m_regScale.resize( w, -999 );
        m_gmaxLP.resize( w, 0 );
    }

    mx::improc::eigenMap<uint32_t> Npc( reinterpret_cast<uint32_t *>( curr_src ), w, 2 );

    for( uint32_t n = 0; n < w; ++n )
    {
        if( change || m_Na[n] != Npc( n, 0 ) || m_Nb[n] != Npc( n, 1 ) )
        {
            if( !change )
            {
                m_updating = true;
                lock.lock();
                m_updating = true; // Make sure it didn't get set to false by thread
                                   // that had the lock

                change = true;
            }

            m_Na[n] = Npc( n, 0 );
            m_Nb[n] = Npc( n, 1 );
        }
    }

    if( change )
    {
        if( m_loop )
        {
            m_sinceChange = -1;
        }

        m_updating = false;
        m_goptUpdated = true;

        lock.unlock();
        std::cerr << "got num pc coeffs: " << m_Na.size() << "\n";
    }

    return 0;
}

int modalGainOpt::allocate( const acoeffShmimT &dummy )
{
    static_cast<void>( dummy );

    return 0;
}

int modalGainOpt::processImage( void *curr_src, const acoeffShmimT &dummy )
{
    static_cast<void>( dummy );

    bool change = false;

    std::unique_lock<std::mutex> lock( m_goptMutex, std::defer_lock );

    uint32_t w = acoeffShmimMonitorT::m_width;
    uint32_t h = acoeffShmimMonitorT::m_height;

    // If there's a size change we lock
    if( w - 1 != m_as.rows() || h != m_as.cols() || h != m_NaCurrent.size() )
    {
        m_updating = true;
        lock.lock();
        m_updating = true; // Make sure it didn't get set to false by thread that
                           // had the lock

        change = true;
        m_NaCurrent.resize( h );
        m_as.resize( w - 1, h );
    }

    eigenMap<float> ac( reinterpret_cast<float *>( curr_src ), w, h );

    for( uint32_t cc = 0; cc < h; ++cc )
    {
        if( change || m_NaCurrent[cc] != ac( 0, cc ) )
        {
            if( !change )
            {
                m_updating = true;
                lock.lock();
                m_updating = true; // Make sure it didn't get set to false by thread
                                   // that had the lock

                change = true;
            }

            m_NaCurrent[cc] = ac( 0, cc );
        }

        for( uint32_t rr = 1; rr < w; ++rr )
        {
            if( change || m_as( rr - 1, cc ) != ac( rr, cc ) )
            {
                if( !change )
                {
                    m_updating = true;
                    lock.lock();
                    m_updating = true; // Make sure it didn't get set to false by thread
                                       // that had the lock

                    change = true;
                }

                m_as( rr - 1, cc ) = ac( rr, cc );
            }
        }
    }

    if( change )
    {
        if( m_loop && m_pcOn )
        {
            m_sinceChange = -1;
        }

        m_updating = false;
        m_pcgoptUpdated = true;

        lock.unlock();
        std::cerr << "got a coeffs: " << w << ' ' << h << ' ' << m_NaCurrent.size() << "\n";
    }

    return 0;
}

int modalGainOpt::allocate( const bcoeffShmimT &dummy )
{
    static_cast<void>( dummy );

    return 0;
}

int modalGainOpt::processImage( void *curr_src, const bcoeffShmimT &dummy )
{
    static_cast<void>( dummy );

    bool change = false;

    std::unique_lock<std::mutex> lock( m_goptMutex, std::defer_lock );

    uint32_t w = bcoeffShmimMonitorT::m_width;
    uint32_t h = bcoeffShmimMonitorT::m_height;

    // If there's a size change we lock
    if( w - 1 != m_bs.rows() || h != m_bs.cols() || h != m_NbCurrent.size() )
    {
        m_updating = true;
        lock.lock();
        m_updating = true; // Make sure it didn't get set to false by thread that
                           // had the lock

        change = true;

        m_NbCurrent.resize( h );
        m_bs.resize( w - 1, h );
    }

    eigenMap<float> bc( reinterpret_cast<float *>( curr_src ), w, h );

    for( uint32_t cc = 0; cc < h; ++cc )
    {
        if( change || m_NbCurrent[cc] != bc( 0, cc ) )
        {
            if( !change )
            {
                m_updating = true;
                lock.lock();
                m_updating = true; // Make sure it didn't get set to false by thread
                                   // that had the lock

                change = true;
            }

            m_NbCurrent[cc] = bc( 0, cc );
        }

        for( uint32_t rr = 1; rr < w; ++rr )
        {
            if( change || m_bs( rr - 1, cc ) != bc( rr, cc ) )
            {
                if( !change )
                {
                    m_updating = true;
                    lock.lock();
                    m_updating = true; // Make sure it didn't get set to false by thread
                                       // that had the lock

                    change = true;
                }

                m_bs( rr - 1, cc ) = bc( rr, cc );
            }
        }
    }

    if( change )
    {
        if( m_loop && m_pcOn )
        {
            m_sinceChange = -1;
        }

        m_updating = false;
        m_pcgoptUpdated = true;

        std::cerr << "got b coeffs: " << w << ' ' << h << ' ' << m_NbCurrent.size() << "\n";

        lock.unlock();
    }

    return 0;
}

int modalGainOpt::allocate( const gainCalShmimT &dummy )
{
    static_cast<void>( dummy );

    return 0;
}

int modalGainOpt::processImage( void *curr_src, const gainCalShmimT &dummy )
{
    static_cast<void>( dummy );

    if( gainCalShmimMonitorT::m_height != 1 )
    {
        return log<software_error, -1>( { "got gainCals with height not 1" } );
    }

    bool change = false;

    uint32_t w = gainCalShmimMonitorT::m_width;

    std::unique_lock<std::mutex> lock( m_goptMutex, std::defer_lock );

    if( w != m_gainCals.size() )
    {
        m_updating = true;
        lock.lock();

        m_updating = true; // Make sure it didn't get set to false by thread that
                           // had the lock

        change = true;
        m_gainCals.resize( w );
    }

    float *g = static_cast<float *>( curr_src );

    for( uint32_t n = 0; n < w; ++n )
    {
        if( change || m_gainCals[n] != g[n] )
        {
            if( !change )
            {
                m_updating = true;
                lock.lock();
                m_updating = true; // Make sure it didn't get set to false by thread
                                   // that had the lock

                change = true;
            }

            m_gainCals[n] = g[n];
        }
    }

    if( change )
    {
        m_sinceChange = -1;
        m_updating = false;
        std::cerr << "got gainCals: " << m_gainCals.size() << "\n";
        lock.unlock();
    }

    return 0;
}

int modalGainOpt::allocate( const gainCalFactShmimT &dummy )
{
    static_cast<void>( dummy );

    return 0;
}

int modalGainOpt::processImage( void *curr_src, const gainCalFactShmimT &dummy )
{
    static_cast<void>( dummy );

    if( gainCalFactShmimMonitorT::m_height != 1 )
    {
        return log<software_error, -1>( { "got gainCalFacts with height not 1" } );
    }

    bool change = false;

    uint32_t w = gainCalFactShmimMonitorT::m_width;

    std::unique_lock<std::mutex> lock( m_goptMutex, std::defer_lock );

    if( w != m_gainCalFacts.size() )
    {
        m_updating = true;
        lock.lock();
        m_updating = true; // Make sure it didn't get set to false by thread that
                           // had the lock

        change = true;
        m_gainCalFacts.resize( w );
    }

    float *g = static_cast<float *>( curr_src );

    for( uint32_t n = 0; n < w; ++n )
    {
        if( change || m_gainCalFacts[n] != g[n] )
        {
            if( !change )
            {
                m_updating = true;
                lock.lock();
                m_updating = true; // Make sure it didn't get set to false by thread
                                   // that had the lock

                change = true;
            }

            m_gainCalFacts[n] = g[n];
        }
    }

    if( change )
    {
        m_sinceChange = -1;
        m_updating = false;
        std::cerr << "got gainCalsFacts: " << m_gainCalFacts.size() << "\n";
        lock.unlock();
    }

    return 0;
}

int modalGainOpt::allocate( const tauShmimT &dummy )
{
    static_cast<void>( dummy );

    return 0;
}

int modalGainOpt::processImage( void *curr_src, const tauShmimT &dummy )
{
    static_cast<void>( dummy );

    if( tauShmimMonitorT::m_height != 1 )
    {
        return log<software_error, -1>( { "got tau with height not 1" } );
    }

    bool change = false;

    uint32_t w = tauShmimMonitorT::m_width;

    std::unique_lock<std::mutex> lock( m_goptMutex, std::defer_lock );

    if( w != m_taus.size() )
    {
        m_updating = true;
        lock.lock();
        m_updating = true; // Make sure it didn't get set to false by thread that
                           // had the lock

        change = true;
        m_taus.resize( w );
    }

    float *t = static_cast<float *>( curr_src );

    for( uint32_t n = 0; n < w; ++n )
    {
        if( change || m_taus[n] != t[n] )
        {
            if( !change )
            {
                m_updating = true;
                lock.lock();
                m_updating = true; // Make sure it didn't get set to false by thread
                                   // that had the lock

                change = true;
            }

            m_taus[n] = t[n];
        }
    }

    if( change )
    {
        m_sinceChange = -1;
        m_updating = false;
        m_goptUpdated = true;
        std::cerr << "got taus: " << m_taus.size() << "\n";
        lock.unlock();
    }
    return 0;
}

int modalGainOpt::allocate( const noiseShmimT &dummy )
{
    static_cast<void>( dummy );

    return 0;
}

int modalGainOpt::processImage( void *curr_src, const noiseShmimT &dummy )
{
    static_cast<void>( dummy );

    if( noiseShmimMonitorT::m_width != 3 )
    {
        return log<software_error, -1>( { "got tau with width not 3" } );
    }

    bool change = false;

    uint32_t h = noiseShmimMonitorT::m_height;

    std::unique_lock<std::mutex> lock( m_goptMutex, std::defer_lock );

    if( h != m_noiseParams.cols() )
    {
        m_updating = true;
        lock.lock();
        m_updating = true; // Make sure it didn't get set to false by thread that
                           // had the lock

        change = true;
        m_noiseParams.resize( 3, h );
    }

    mx::improc::eigenMap<float> np( static_cast<float *>( curr_src ), 3, h );

    for( uint32_t n = 0; n < h; ++n )
    {
        if( change || m_noiseParams( 0, n ) != np( 0, n ) || m_noiseParams( 1, n ) != np( 1, n ) ||
            m_noiseParams( 2, n ) != np( 2, n ) )
        {
            if( !change )
            {
                m_updating = true;
                lock.lock();
                m_updating = true; // Make sure it didn't get set to false by thread
                                   // that had the lock

                change = true;
            }

            m_noiseParams.col( n ) = np.col( n );
        }
    }

    if( change )
    {
        m_sinceChange = -1;
        m_updating = false;
        m_goptUpdated = true;
        std::cerr << "got noise params: " << m_noiseParams.rows() << " x " << m_noiseParams.cols() << "\n";
        lock.unlock();
    }
    return 0;
}

int modalGainOpt::allocate( const wfsavgShmimT &dummy )
{
    static_cast<void>( dummy );

    std::cerr << "Got WFS avg: " << wfsavgShmimMonitorT::m_width << " x " << wfsavgShmimMonitorT::m_height << '\n';
    return 0;
}

int modalGainOpt::processImage( void *curr_src, const wfsavgShmimT &dummy )
{
    static_cast<void>( dummy );

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_goptMutex );

        m_wfsavg = mx::improc::eigenMap<float>( reinterpret_cast<float *>( curr_src ),
                                                wfsavgShmimMonitorT::m_width,
                                                wfsavgShmimMonitorT::m_height );

        if( m_wfsavg.rows() == m_wfsmask.rows() && m_wfsavg.cols() == m_wfsmask.cols() )
        {
            m_counts = ( m_wfsavg * m_wfsmask ).sum();
        }
    }

    return 0;
}

int modalGainOpt::allocate( const wfsmaskShmimT &dummy )
{
    static_cast<void>( dummy );

    std::cerr << "Got WFS mask: " << wfsmaskShmimMonitorT::m_width << " x " << wfsmaskShmimMonitorT::m_height << '\n';
    return 0;
}

int modalGainOpt::processImage( void *curr_src, const wfsmaskShmimT &dummy )
{
    static_cast<void>( dummy );

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_goptMutex );

        m_wfsmask = mx::improc::eigenMap<float>( reinterpret_cast<float *>( curr_src ),
                                                 wfsmaskShmimMonitorT::m_width,
                                                 wfsmaskShmimMonitorT::m_height );

        m_npix = m_wfsmask.sum();

        if( m_wfsavg.rows() == m_wfsmask.rows() && m_wfsavg.cols() == m_wfsmask.cols() )
        {
            m_counts = ( m_wfsavg * m_wfsmask ).sum();
        }
    }

    return 0;
}

int modalGainOpt::checkSizes()
{
    static std::vector<bool> logged( 50, false );

    int L = 0;
    if( m_clPSDs.rows() == 0 || m_clPSDs.cols() == 0 ) // somehow here without any data
    {
        if( !logged[L] )
        {
            log<software_error>( { "PSDs have not been updated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( static_cast<size_t>( m_clPSDs.rows() ) != m_freq.size() )
    {
        if( !logged[L] )
        {
            log<software_error>( { "PSDs and freq size mismatch" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_nModes != m_gainFacts.size() )
    {
        if( !logged[L] )
        {
            log<software_error>( { "PSDs and gains number of modes mismatch" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_nModes != m_multFacts.size() )
    {
        if( !logged[L] )
        {
            log<software_error>( { "PSDs and mult coeffs number of modes mismatch" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_nModes != m_gainCals.size() )
    {
        if( !logged[L] )
        {
            log<software_error>( { "PSDs and gain cals number of modes mismatch" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_nModes != m_gainCalFacts.size() )
    {
        if( !logged[L] )
        {
            log<software_error>( { "PSDs and gain cal facts number of modes mismatch" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_nModes != m_taus.size() )
    {
        if( !logged[L] )
        {
            log<software_error>( { "Loop taus have not been set" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    /*if( m_clPSDs.cols() != m_noiseParams.cols() || m_noiseParams.rows() != 3 )
    {
        if( !logged[L] )
        {
            log<software_error>( { "noise params have not been set" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;*/

    if( m_fps <= 0 )
    {
        if( !logged[L] )
        {
            log<software_error>( { "Loop fps has not been set" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_olPSDStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "m_olPSDStream is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_rawOlPSDStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "m_rawOlPSDStream is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_smoothOlPSDStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "m_smoothOlPSDStream is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_noisePSDStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "m_noisePSDStream is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_clXferCurrentStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "m_clXferCurrentStream is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_clNtfCurrentStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "m_clNtfCurrentStream is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_clXferSIStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "m_clXferSIStream is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_clNtfSIStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "m_clNtfSIStream is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_optGainStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "optGainsStream is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_optGainSIStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "optGainsStream SI is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_optGainSIRawStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "optGainsStream SI raw is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    // Check the PC shmims that we don't automatically create
    if( m_Na.size() != m_nModes || m_NaCurrent.size() != m_nModes || (size_t)m_as.cols() != m_nModes ||
        m_Nb.size() != m_nModes || m_NbCurrent.size() != m_nModes || (size_t)m_bs.cols() != m_nModes ||
        m_pcGainFacts.size() != m_nModes || m_pcMultFacts.size() != m_nModes )
    {

        if( allocatePCShmims() < 0 )
        {
            log<software_error>( { "error allocating PC shmims" } );
        }

        if( m_Na.size() != m_nModes || m_NaCurrent.size() != m_nModes || (size_t)m_as.cols() != m_nModes ||
            m_Nb.size() != m_nModes || m_NbCurrent.size() != m_nModes || (size_t)m_bs.cols() != m_nModes ||
            m_pcGainFacts.size() != m_nModes || m_pcMultFacts.size() != m_nModes )
        {

            if( !logged[L] )
            {
                log<software_error>( { "PC shmims not allcoated" } );
            }
            logged[L] = true;
            return -1;
        }
        else
        {
            logged[L] = false;
        }
    }
    else
    {
        logged[L] = false;
    }
    ++L;

    if( m_clXferLPStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "m_clXferLPStream is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_clNtfLPStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "m_clNtfLPStream is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_optGainLPStream == nullptr )
    {
        if( !logged[L] )
        {
            log<software_error>( { "m_optGainLPStream is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    if( m_nModes != m_optGainLP.size() )
    {
        if( !logged[L] )
        {
            log<software_error>( { "m_optGainLP is not allocated" } );
        }
        logged[L] = true;
        return -1;
    }
    logged[L++] = false;

    return 0;
}

float modalGainOpt::noisePSD( int n )
{
    float mc = m_noiseParams( 1, n ) * m_counts / m_emg;

    float snr2 = ( mc * mc ) / ( mc + m_npix * pow( m_noiseParams( 2, n ) / m_emg, 2 ) );

    return 0.05 * m_noiseParams( 0, n ) / snr2;
}

void modalGainOpt::goptThreadStart( modalGainOpt *p )
{
    p->goptThreadExec();
}

void modalGainOpt::goptThreadExec()
{
    m_goptThreadID = syscall( SYS_gettid );

    while( m_goptThreadInit == true && shutdown() == 0 )
    {
        sleep( 1 );
    }

    while( shutdown() == 0 )
    {
        timespec ts;
        XWC_SEM_WAIT_TS_RETVOID( ts, 1, 0 );

        if( sem_timedwait( &m_goptSemaphore, &ts ) == 0 )
        {
            std::lock_guard<std::mutex> lock( m_goptMutex );

            if( checkSizes() < 0 )
            {
                std::cerr << "check sizes failed\n";
                continue; // we just wait
            }

            refreshGoptStructures();

            MGO_BREADCRUMB;
            if( m_updating )
            {
                continue;
            }

            if( m_siGainStateNeedsSync )
            {
                syncSiGainStateFromAppliedGains();
            }

            if( m_zeroGains )
            {
                std::fill( m_optGainSI.begin(), m_optGainSI.end(), 0.0F );
                m_zeroGains = false;
            }

            timePointT t0 = std::chrono::steady_clock::now();

            MGO_BREADCRUMB;

            int off = 0;
            int offLP = 0;

#pragma omp parallel for num_threads( 15 )
            for( size_t n = 0; n < m_goptCurrent.size(); ++n )
            {
                if( m_updating || m_shutdown )
                {
                    continue; // don't break b/c of omp
                }

                if( !m_pcOn )
                {
                    MGO_BREADCRUMB;
                    float currGain = m_gain * m_gainFacts[n] * m_gainCals[n] * m_opticalGain;

                    for( size_t f = 0; f < m_goptCurrent[n].f_size(); ++f )
                    {
                        m_goptCurrent[n].clTF2( m_clXferCurrent( f, n ), m_clNtfCurrent( f, n ), f, currGain );
                    }
                }
                else
                {
                    float currGain = m_pcGain * m_pcGainFacts[n] * m_gainCals[n] * m_opticalGain;
                    for( size_t f = 0; f < m_goptCurrent[n].f_size(); ++f )
                    {
                        m_goptCurrent[n].clTF2( m_clXferCurrent( f, n ), m_clNtfCurrent( f, n ), f, currGain );
                    }
                }

                MGO_BREADCRUMB;
                // Calculate the OL PSD with the current gopt (PC or SI)
                float og2 = m_opticalGain * m_opticalGain;
                std::vector<float> clMeasuredPsd( m_goptCurrent[n].f_size(), 0.0F );
                std::vector<float> etfCorrectionPsd( m_goptCurrent[n].f_size(), 1.0F );
                std::vector<float> ntfCorrectionPsd( m_goptCurrent[n].f_size(), 1.0F );
                if( !m_loop )
                {
                    MGO_BREADCRUMB;
                    for( size_t f = 1; f < m_goptCurrent[n].f_size(); ++f )
                    {
                        clMeasuredPsd[f] = m_clPSDs( f, n ) / og2;
                        m_olPSDs[n][f] = clMeasuredPsd[f];
                    }
                }
                else
                {
                    MGO_BREADCRUMB;
                    for( size_t f = 1; f < m_goptCurrent[n].f_size(); ++f )
                    {
                        clMeasuredPsd[f] = m_clPSDs( f, n ) / og2;
                        etfCorrectionPsd[f] = m_clXferCurrent( f, n );
                        ntfCorrectionPsd[f] = m_clNtfCurrent( f, n );
                        m_olPSDs[n][f] = clMeasuredPsd[f] / etfCorrectionPsd[f];
                    }
                }

                MGO_BREADCRUMB;

                clMeasuredPsd[0] = clMeasuredPsd[1];
                etfCorrectionPsd[0] = etfCorrectionPsd[1];
                ntfCorrectionPsd[0] = ntfCorrectionPsd[1];
                m_olPSDs[n][0] = m_olPSDs[n][1];

                bool flagOff = false;
                std::vector<float> lpProcessPsd = m_olPSDs[n];
                bool useClosedLoopNoiseEstimate =
                    m_extrapNoiseEstimateDomain == c_extrapNoiseEstimateClosedLoopPreXfer && m_loop;
                bool useNtfAwareClosedLoopEstimate =
                    useClosedLoopNoiseEstimate &&
                    m_extrapClosedLoopOlEstimateMethod == c_extrapClosedLoopOlEstimateNtfAware;
                std::vector<float> noiseEstimateWorkPsd;
                std::vector<float> clSignificanceNoisePsd;
                float clSignificanceNoiseFloor = 0;
                bool doClSignificanceCheck =
                    m_extrapConfig.m_clSignificanceThreshold > 0 && m_extrapConfig.m_clMinSignificantFraction > 0;
                const std::vector<float> *noiseEstimateSourcePsd =
                    useClosedLoopNoiseEstimate ? &clMeasuredPsd : &m_olPSDs[n];
                if( useNtfAwareClosedLoopEstimate )
                {
                    const float tiny = std::numeric_limits<float>::min();
                    noiseEstimateWorkPsd.resize( clMeasuredPsd.size(), 0.0F );
                    for( size_t f = 0; f < clMeasuredPsd.size(); ++f )
                    {
                        noiseEstimateWorkPsd[f] = clMeasuredPsd[f] / std::max( ntfCorrectionPsd[f], tiny );
                    }

                    noiseEstimateSourcePsd = &noiseEstimateWorkPsd;
                }

                if( doClSignificanceCheck )
                {
                    mx::error_t clSignificanceErr =
                        processPsdProcessorT::estimateNoisePsd( clSignificanceNoisePsd,
                                                                clSignificanceNoiseFloor,
                                                                clMeasuredPsd,
                                                                m_freq,
                                                                n,
                                                                m_extrapConfig.m_noiseEstimateRange,
                                                                m_extrapConfig.m_noiseEstimateStatistic,
                                                                m_extrapConfig.m_noiseEstimateLowFreqMaxHz );
                    if( !!clSignificanceErr )
                    {
#pragma omp critical
                        {
                            log<software_error>( { "error estimating raw CL modal significance noise PSD" } );
                        }

                        continue;
                    }
                }

                if( m_extrapOL == c_olProcessNone )
                {
                    float noiseFloor = 0;
                    mx::error_t errc =
                        processPsdProcessorT::estimateNoisePsd( m_nPSDs[n],
                                                                noiseFloor,
                                                                *noiseEstimateSourcePsd,
                                                                m_freq,
                                                                n,
                                                                m_extrapConfig.m_noiseEstimateRange,
                                                                m_extrapConfig.m_noiseEstimateStatistic,
                                                                m_extrapConfig.m_noiseEstimateLowFreqMaxHz );
                    if( !!errc )
                    {
#pragma omp critical
                        {
                            log<software_error>( { "error estimating modal noise PSD" } );
                        }

                        continue;
                    }

                    if( useClosedLoopNoiseEstimate )
                    {
                        const float tiny = std::numeric_limits<float>::min();
                        for( size_t f = 1; f < m_goptCurrent[n].f_size(); ++f )
                        {
                            float closedLoopNoise = m_nPSDs[n][f];
                            if( useNtfAwareClosedLoopEstimate )
                            {
                                closedLoopNoise *= std::max( ntfCorrectionPsd[f], tiny );
                            }

                            m_olPSDs[n][f] = std::max( clMeasuredPsd[f] - closedLoopNoise, tiny ) /
                                             std::max( etfCorrectionPsd[f], tiny );
                        }

                        m_olPSDs[n][0] = m_olPSDs[n][1];
                    }

                    m_rawOlPSDs[n] = m_olPSDs[n];
                    m_smoothOlPSDs[n] = m_rawOlPSDs[n];
                }
                else
                {
                    processPsdProcessorT::processModelConfig processConfig = m_extrapConfig;
                    processConfig.m_method = olProcessMethodName( m_extrapOL );
                    processConfig.m_noiseEstimateDomain = extrapNoiseEstimateDomainName(
                        useClosedLoopNoiseEstimate ? c_extrapNoiseEstimateClosedLoopPreXfer
                                                   : c_extrapNoiseEstimateOpenLoop );
                    processConfig.m_noiseEstimateRange = extrapNoiseEstimateRangeName( m_extrapNoiseEstimateRange );
                    processConfig.m_noiseEstimateStatistic =
                        extrapNoiseEstimateStatisticName( m_extrapNoiseEstimateStatistic );
                    processConfig.m_noiseEstimateLowFreqMaxHz = m_extrapConfig.m_noiseEstimateLowFreqMaxHz;
                    processConfig.m_closedLoopOlEstimateMethod =
                        extrapClosedLoopOlEstimateMethodName( m_extrapClosedLoopOlEstimateMethod );

                    processPsdProcessorT::processResults processResult;
                    const std::vector<float> &processMeasuredPsd =
                        useClosedLoopNoiseEstimate ? clMeasuredPsd : m_olPSDs[n];
                    const std::vector<float> *processEtfPsd = useClosedLoopNoiseEstimate ? &etfCorrectionPsd : nullptr;
                    const std::vector<float> *processNtfPsd = useClosedLoopNoiseEstimate ? &ntfCorrectionPsd : nullptr;
                    mx::error_t errc =
                        processPsdProcessorT::analyzePsd( processResult,
                                                          processMeasuredPsd,
                                                          m_freq,
                                                          n,
                                                          processConfig,
                                                          0,
                                                          processPsdProcessorT::c_defaultLpContinuumWidthHz,
                                                          processEtfPsd,
                                                          processNtfPsd );
                    if( !!errc )
                    {
#pragma omp critical
                        {
                            log<software_error>( { __FILE__,
                                                   __LINE__,
                                                   "error processing modal PSD with method " +
                                                       olProcessMethodName( m_extrapOL ) + ": [" +
                                                       std::string( mx::errorName( errc ) ) + "] " +
                                                       mx::errorMessage( errc ) } );
                            log<text_log>(
                                "extrapolation settings: noiseDomain=" + processConfig.m_noiseEstimateDomain +
                                    " noiseRange=" + processConfig.m_noiseEstimateRange +
                                    " noiseStatistic=" + processConfig.m_noiseEstimateStatistic +
                                    " noiseLowMax=" + std::to_string( processConfig.m_noiseEstimateLowFreqMaxHz ) +
                                    " olEstimateMethod=" + processConfig.m_closedLoopOlEstimateMethod +
                                    " match=" + std::to_string( processConfig.m_powerLawMatchFreq ) +
                                    " matchWindow=" + std::to_string( processConfig.m_powerLawMatchFallbackWindowHz ) +
                                    " crossoverMode=" + processConfig.m_powerLawCrossoverMode +
                                    " autoSmooth=" + std::to_string( processConfig.m_powerLawAutoSmoothWidthHz ) +
                                    " autoMaxFrac=" + std::to_string( processConfig.m_powerLawAutoMaxFreqFraction ) +
                                    " fitIndex=" + std::string( processConfig.m_fitPowerLawIndex ? "true" : "false" ) +
                                    " fitMin=" + std::to_string( processConfig.m_powerLawFitMinFreqHz ) +
                                    " fitMax=" + std::to_string( processConfig.m_powerLawFitMaxFreqHz ) +
                                    " fitBin=" + std::to_string( processConfig.m_powerLawFitBinWidthHz ) +
                                    " blendBins=" + std::to_string( processConfig.m_powerLawBlendBins ) +
                                    " peakWidth=" + std::to_string( processConfig.m_peakDetectWidthHz ) +
                                    " peakFactor=" + std::to_string( processConfig.m_peakDetectFactor ) +
                                    " peakBroadFactor=" + std::to_string( processConfig.m_peakDetectBroadFactor ) +
                                    " peakMinWidthLog=" + std::to_string( processConfig.m_peakDetectMinWidthLog ) +
                                    " peakPasses=" + std::to_string( processConfig.m_peakDetectPasses ) +
                                    " peakBeta=" + std::to_string( processConfig.m_peakMoffatBeta ) +
                                    " dropoutGap=" + std::to_string( processConfig.m_dropoutGapFactor ) +
                                    " dropoutTiny=" + std::to_string( processConfig.m_dropoutTinyFactor ) +
                                    " dropoutMaxBins=" + std::to_string( processConfig.m_dropoutMaxBins ) +
                                    " powerLawOnlyAbove=" + std::to_string( processConfig.m_powerLawOnlyAboveFreq ),
                                logPrio::LOG_NOTICE );
                        }

                        float noiseFloor = 0;
                        mx::error_t noiseErr =
                            processPsdProcessorT::estimateNoisePsd( m_nPSDs[n],
                                                                    noiseFloor,
                                                                    *noiseEstimateSourcePsd,
                                                                    m_freq,
                                                                    n,
                                                                    processConfig.m_noiseEstimateRange,
                                                                    processConfig.m_noiseEstimateStatistic,
                                                                    processConfig.m_noiseEstimateLowFreqMaxHz );
                        if( !!noiseErr )
                        {
#pragma omp critical
                            {
                                log<software_error>( { "error estimating fallback modal noise PSD" } );
                            }

                            continue;
                        }

                        if( useClosedLoopNoiseEstimate )
                        {
                            const float tiny = std::numeric_limits<float>::min();
                            for( size_t f = 1; f < m_goptCurrent[n].f_size(); ++f )
                            {
                                float closedLoopNoise = m_nPSDs[n][f];
                                if( useNtfAwareClosedLoopEstimate )
                                {
                                    closedLoopNoise *= std::max( ntfCorrectionPsd[f], tiny );
                                }

                                m_olPSDs[n][f] = std::max( clMeasuredPsd[f] - closedLoopNoise, tiny ) /
                                                 std::max( etfCorrectionPsd[f], tiny );
                            }

                            m_olPSDs[n][0] = m_olPSDs[n][1];
                        }

                        m_rawOlPSDs[n] = m_olPSDs[n];
                        m_smoothOlPSDs[n] = m_rawOlPSDs[n];
                    }
                    else
                    {
                        m_nPSDs[n] = processResult.m_noisePsd;
                        m_olPSDs[n] = processResult.m_processPsd;
                        m_rawOlPSDs[n] = processResult.m_rawProcessPsd;
                        m_smoothOlPSDs[n] = processResult.m_smoothedProcessPsd;
                        lpProcessPsd = processResult.m_lpProcessPsd;
                    }
                }

                if( !flagOff && doClSignificanceCheck && clMeasuredPsd.size() > 1 )
                {
                    size_t significantCount = 0;
                    size_t totalCount = 0;
                    for( size_t f = 1; f < clMeasuredPsd.size(); ++f )
                    {
                        ++totalCount;
                        if( clMeasuredPsd[f] > m_extrapConfig.m_clSignificanceThreshold * clSignificanceNoisePsd[f] )
                        {
                            ++significantCount;
                        }
                    }

                    if( totalCount == 0 || static_cast<float>( significantCount ) / static_cast<float>( totalCount ) <
                                               m_extrapConfig.m_clMinSignificantFraction )
                    {
                        flagOff = true;
                    }
                }

                MGO_BREADCRUMB;

                if( flagOff )
                {
                    MGO_BREADCRUMB;
#pragma omp critical
                    {
                        ++off;
                    }

                    for( size_t f = 0; f < m_goptCurrent[n].f_size(); ++f )
                    {
                        m_olPSDs[n][f] = 0;
                    }

                    m_modeVarOL[n] = mx::sigproc::psdVar( m_freq, m_olPSDs[n] );

                    m_optGainSIRaw[n] = 0;
                    m_modeVarSI[n] = m_modeVarOL[n];

                    for( size_t f = 0; f < m_goptCurrent[n].f_size(); ++f )
                    {
                        m_clXferSI( f, n ) = 1;
                        m_clNtfSI( f, n ) = 0;
                    }

                    m_timesOnSI[n] = 0;
                }
                else
                {
                    MGO_BREADCRUMB;

                    MGO_BREADCRUMB;
                    m_modeVarOL[n] = mx::sigproc::psdVar( m_freq, m_olPSDs[n] );

                    m_optGainSIRaw[n] =
                        m_goptSI[n].optGainOpenLoop( m_modeVarSI[n], m_olPSDs[n], m_nPSDs[n], m_gmaxSI[n], false );

                    if( ( m_modeVarSI[n] - m_modeVarOL[n] ) / m_modeVarOL[n] > -0.001 )
                    {
#pragma omp critical
                        {
                            ++off;
                        }

                        MGO_BREADCRUMB;
                        m_optGainSIRaw[n] = 0;
                        m_modeVarSI[n] = m_modeVarOL[n];

                        for( size_t f = 0; f < m_goptCurrent[n].f_size(); ++f )
                        {
                            m_clXferSI( f, n ) = 1;
                            m_clNtfSI( f, n ) = 0;
                        }

                        m_timesOnSI[n] = 0;
                    }
                    else if( m_timesOnSI[n] < 5 )
                    {
#pragma omp critical
                        {
                            ++off;
                        }

                        MGO_BREADCRUMB;
                        // Would be on, but we debounce
                        m_optGainSIRaw[n] = 0;
                        m_modeVarSI[n] = m_modeVarOL[n];

                        for( size_t f = 0; f < m_goptCurrent[n].f_size(); ++f )
                        {
                            m_clXferSI( f, n ) = 1;
                            m_clNtfSI( f, n ) = 0;
                        }

                        ++m_timesOnSI[n];
                    }
                    else
                    {
                        MGO_BREADCRUMB;
                        for( size_t f = 0; f < m_goptCurrent[n].f_size(); ++f )
                        {
                            m_goptSI[n].clTF2( m_clXferSI( f, n ), m_clNtfSI( f, n ), f, m_optGainSIRaw[n] );
                        }

                        ++m_timesOnSI[n];
                    }
                }

                updateIntegratedSiGain( n );

                if( m_doPCCalcs && !flagOff )
                {
                    MGO_BREADCRUMB;
                    if( m_regScale[n] == -999 || m_regCounter[n] >= m_nRegCycles )
                    {
                        MGO_BREADCRUMB;
                        float min_sc;
                        float gmax_lp = 0;

                        if( m_linPred[n].regularizeCoefficients( gmax_lp,
                                                                 m_optGainLP[n],
                                                                 m_modeVarLP[n],
                                                                 min_sc,
                                                                 m_goptLP[n],
                                                                 lpProcessPsd,
                                                                 m_nPSDs[n],
                                                                 m_Na[n] ) < 0 )
                        {
                            MGO_BREADCRUMB;

                            m_optGainLP[n] = 0;
                            m_modeVarLP[n] = m_modeVarOL[n];

                            ///\todo what to do about coeffs?
                        }

                        MGO_BREADCRUMB;

                        if( m_regScale[n] == -999 )
                        {
                            MGO_BREADCRUMB;
                            m_regCounter[n] = n % m_nRegCycles;
                        }
                        else
                        {
                            MGO_BREADCRUMB;
                            m_regCounter[n] = 0;
                        }

                        m_regScale[n] = min_sc;
                        m_gmaxLP[n] = gmax_lp;
                    }
                    else
                    {
                        MGO_BREADCRUMB;
                        // use pre-regularized version
                        float psdReg = lpProcessPsd[0];
                        if( m_linPred[n].calcCoefficients( lpProcessPsd,
                                                           m_nPSDs[n],
                                                           psdReg * pow( 10, -m_regScale[n] / 10 ),
                                                           m_Na[n] ) < 0 )
                        {
                            m_optGainLP[n] = 0;
                            m_modeVarLP[n] = m_modeVarOL[n];

                            ///\todo what to do about coeffs?
                        }
                        else
                        {
                            m_goptLP[n].a( m_linPred[n].m_lp.m_c );
                            m_goptLP[n].b( m_linPred[n].m_lp.m_c );

                            m_optGainLP[n] = m_goptLP[n].optGainOpenLoop( m_modeVarLP[n],
                                                                          m_olPSDs[n],
                                                                          m_nPSDs[n],
                                                                          m_gmaxLP[n],
                                                                          false );
                        }
                        ++m_regCounter[n];
                    }

                    MGO_BREADCRUMB;

                    /*if( ( m_modeVarLP[n] - m_modeVarOL[n] ) / m_modeVarOL[n] > -0.001 )
                    {
                        m_optGainLP[n] = 0;
                        m_modeVarLP[n] = m_modeVarOL[n];

                        for( size_t f = 0; f < m_goptCurrent[n].f_size(); ++f )
                        {
                            m_clXferLP( f, n ) = 1;
                        }

                        m_timesOnLP[n] = 0;
                    }
                    else*/
                    if( ( m_modeVarLP[n] - m_modeVarSI[n] ) / m_modeVarSI[n] > -0.001 )
                    {
#pragma omp critical
                        {
                            ++offLP;
                        }

                        MGO_BREADCRUMB;
                        m_optGainLP[n] = m_optGainSI[n];
                        m_modeVarLP[n] = m_modeVarSI[n];

                        for( size_t f = 0; f < m_goptCurrent[n].f_size(); ++f )
                        {
                            m_clXferLP( f, n ) = m_clXferSI( f, n );
                            m_clNtfLP( f, n ) = m_clNtfSI( f, n );
                        }

                        m_timesOnLP[n] = 0;
                    }
                    else if( m_timesOnLP[n] < 5 )
                    {
#pragma omp critical
                        {
                            ++offLP;
                        }

                        MGO_BREADCRUMB;
                        m_optGainLP[n] = m_optGainSI[n];
                        m_modeVarLP[n] = m_modeVarSI[n];

                        for( size_t f = 0; f < m_goptCurrent[n].f_size(); ++f )
                        {
                            m_clXferLP( f, n ) = m_clXferSI( f, n );
                            m_clNtfLP( f, n ) = m_clNtfSI( f, n );
                        }

                        ++m_timesOnLP[n];
                    }
                    else
                    {
                        MGO_BREADCRUMB;
                        for( size_t f = 0; f < m_goptCurrent[n].f_size(); ++f )
                        {
                            m_goptLP[n].clTF2( m_clXferLP( f, n ), m_clNtfLP( f, n ), f, m_optGainLP[n] );
                        }

                        ++m_timesOnLP[n];
                    }
                }
                else
                {
#pragma omp critical
                    {
                        ++offLP;
                    }

                    MGO_BREADCRUMB;
                    m_optGainLP[n] = 0;

                    ++m_regCounter[n];

                    for( size_t f = 0; f < m_goptCurrent[n].f_size(); ++f )
                    {
                        m_clXferLP( f, n ) = 1;
                        m_clNtfLP( f, n ) = 0;
                    }

                    m_timesOnLP[n] = 0;
                }

                MGO_BREADCRUMB;
            }

            MGO_BREADCRUMB;

            m_modesOnSI = m_nModes - off;
            m_modesOnLP = m_nModes - offLP;

            if( m_updating )
            {
                continue; // don't break b/c of omp
            }

            timePointT t1 = std::chrono::steady_clock::now();
            durationT dt = t1 - t0;

            std::cerr << "Optimization took " << dt.count() << " seconds\n";

            /*float totVar = 0;
            for( size_t n = 0; n < m_modeVarSI.size(); ++n )
            {
                totVar += m_modeVarSI[n];
            }*/

            // std::cerr << "total variance: " << totVar << '\n';

            float *f = m_optGainStream->array.F;
            float *fSIRaw = m_optGainSIRawStream->array.F;
            float *fSI = m_optGainSIStream->array.F;
            float *fmaxSI = m_maxGainSIStream->array.F;
            float *fLP = m_optGainLPStream->array.F;
            float *fmaxLP = m_maxGainLPStream->array.F;

            mx::improc::eigenMap<float> mvs( m_modevarStream->array.F, 3, m_modeVarSI.size() );

            m_optGainStream->md->write = 1;
            m_optGainSIRawStream->md->write = 1;
            m_optGainSIStream->md->write = 1;
            m_maxGainSIStream->md->write = 1;
            m_optGainLPStream->md->write = 1;
            m_maxGainLPStream->md->write = 1;
            m_modevarStream->md->write = 1;

            writePublishedGainArrays( f, fSIRaw, fSI, fmaxSI, fLP, fmaxLP, mvs.data() );

            ImageStreamIO_UpdateIm( m_optGainStream );
            ImageStreamIO_UpdateIm( m_optGainSIRawStream );
            ImageStreamIO_UpdateIm( m_optGainSIStream );
            ImageStreamIO_UpdateIm( m_maxGainSIStream );
            ImageStreamIO_UpdateIm( m_optGainLPStream );
            ImageStreamIO_UpdateIm( m_maxGainLPStream );
            ImageStreamIO_UpdateIm( m_modevarStream );

            if( m_autoUpdate || m_updateOnce || m_dump )
            {
                float *f = gainFactShmimMonitorT::m_imageStream.array.F;

                gainFactShmimMonitorT::m_imageStream.md->write = 1;
                for( size_t n = 0; n < m_nModes; ++n )
                {
                    f[n] = ( m_gainCalFacts[n] * m_optGainSI[n] / m_gainCals[n] ) / m_opticalGain;
                }

                ImageStreamIO_UpdateIm( &( gainFactShmimMonitorT::m_imageStream ) );

                if( m_doPCCalcs )
                {
                    float *fpc = pcGainFactShmimMonitorT::m_imageStream.array.F;
                    float *fa = acoeffShmimMonitorT::m_imageStream.array.F;
                    float *fb = bcoeffShmimMonitorT::m_imageStream.array.F;

                    pcGainFactShmimMonitorT::m_imageStream.md->write = 1;
                    acoeffShmimMonitorT::m_imageStream.md->write = 1;
                    bcoeffShmimMonitorT::m_imageStream.md->write = 1;

                    writePublishedPredictorArrays( fpc,
                                                   fa,
                                                   acoeffShmimMonitorT::m_width,
                                                   fb,
                                                   bcoeffShmimMonitorT::m_width,
                                                   !m_dump );

                    ImageStreamIO_UpdateIm( &( pcGainFactShmimMonitorT::m_imageStream ) );
                    ImageStreamIO_UpdateIm( &( acoeffShmimMonitorT::m_imageStream ) );
                    ImageStreamIO_UpdateIm( &( bcoeffShmimMonitorT::m_imageStream ) );
                }

                if( m_dump )
                {
                    log<text_log>( "gains updated by dump", logPrio::LOG_NOTICE );
                    m_dump = false;
                }
                else if( m_updateOnce && !m_autoUpdate )
                {
                    log<text_log>( "gains updated once", logPrio::LOG_NOTICE );
                }

                m_updateOnce = false;
            }

            if( m_loop & m_autoUpdate )
            {
                m_sinceChange = -1;
            }

            // Update OL PSDs and Transfer Functions
            m_olPSDStream->md->write = 1;
            m_rawOlPSDStream->md->write = 1;
            m_smoothOlPSDStream->md->write = 1;
            m_noisePSDStream->md->write = 1;
            m_clXferCurrentStream->md->write = 1;
            m_clNtfCurrentStream->md->write = 1;
            m_clXferSIStream->md->write = 1;
            m_clNtfSIStream->md->write = 1;
            m_clXferLPStream->md->write = 1;
            m_clNtfLPStream->md->write = 1;

            for( size_t q = 0; q < m_olPSDs.size(); ++q )
            {
                memcpy( m_olPSDStream->array.F + q * m_olPSDs[0].size(),
                        m_olPSDs[q].data(),
                        m_olPSDs[0].size() * sizeof( float ) );
                memcpy( m_rawOlPSDStream->array.F + q * m_rawOlPSDs[0].size(),
                        m_rawOlPSDs[q].data(),
                        m_rawOlPSDs[0].size() * sizeof( float ) );
                memcpy( m_smoothOlPSDStream->array.F + q * m_smoothOlPSDs[0].size(),
                        m_smoothOlPSDs[q].data(),
                        m_smoothOlPSDs[0].size() * sizeof( float ) );

                // m_noisePSDStream->array.F[q] = m_nPSDs[q][0];
                memcpy( m_noisePSDStream->array.F + q * m_nPSDs[0].size(),
                        m_nPSDs[q].data(),
                        m_nPSDs[0].size() * sizeof( float ) );
            }

            memcpy( m_clXferCurrentStream->array.F,
                    m_clXferCurrent.data(),
                    m_clXferCurrent.rows() * m_clXferCurrent.cols() * sizeof( float ) );
            memcpy( m_clNtfCurrentStream->array.F,
                    m_clNtfCurrent.data(),
                    m_clNtfCurrent.rows() * m_clNtfCurrent.cols() * sizeof( float ) );
            memcpy( m_clXferSIStream->array.F,
                    m_clXferSI.data(),
                    m_clXferSI.rows() * m_clXferSI.cols() * sizeof( float ) );
            memcpy( m_clNtfSIStream->array.F, m_clNtfSI.data(), m_clNtfSI.rows() * m_clNtfSI.cols() * sizeof( float ) );
            memcpy( m_clXferLPStream->array.F,
                    m_clXferLP.data(),
                    m_clXferLP.rows() * m_clXferLP.cols() * sizeof( float ) );
            memcpy( m_clNtfLPStream->array.F, m_clNtfLP.data(), m_clNtfLP.rows() * m_clNtfLP.cols() * sizeof( float ) );

            ImageStreamIO_UpdateIm( m_olPSDStream );
            ImageStreamIO_UpdateIm( m_rawOlPSDStream );
            ImageStreamIO_UpdateIm( m_smoothOlPSDStream );
            ImageStreamIO_UpdateIm( m_noisePSDStream );
            ImageStreamIO_UpdateIm( m_clXferCurrentStream );
            ImageStreamIO_UpdateIm( m_clNtfCurrentStream );
            ImageStreamIO_UpdateIm( m_clXferSIStream );
            ImageStreamIO_UpdateIm( m_clNtfSIStream );
            ImageStreamIO_UpdateIm( m_clXferLPStream );
            ImageStreamIO_UpdateIm( m_clNtfLPStream );
        }
        else
        {
            /* Check for why we timed out */
            /* ETIMEDOUT just means keep waiting */
            if( errno == ETIMEDOUT )
            {
                std::lock_guard<std::mutex> lock( m_goptMutex );

                if( checkSizes() >= 0 )
                {
                    refreshGoptStructures();
                }
                continue;
            }

            /* EINTER probably indicates time to shutdown, loop wil exit if m_shutdown
             * is set */
            if( errno == EINTR )
            {
                continue;
            }

            /*Otherwise, report an error.*/
            log<software_error>( { errno, "sem_timedwait" } );
            break;
        }
    }
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_autoUpdate )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_autoUpdate, ipRecv );

    if( ipRecv.find( "toggle" ) )
    {
        recordModalGainOpt( false );
        {
            std::lock_guard<std::mutex> lock( m_goptMutex );

            if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
            {
                if( !m_autoUpdate )
                {
                    log<text_log>( "updating gains", logPrio::LOG_NOTICE );
                }
                m_autoUpdate = true;
            }
            else
            {
                if( m_autoUpdate )
                {
                    log<text_log>( "stopped updating gains", logPrio::LOG_NOTICE );
                }
                m_autoUpdate = false;
            }
        }

        recordModalGainOpt( false );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_updateOnce )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_updateOnce, ipRecv );

    if( ipRecv.find( "request" ) )
    {
        std::lock_guard<std::mutex> lock( m_goptMutex );

        if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
        {
            m_updateOnce = true;
        }
        else
        {
            m_updateOnce = false;
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_dump )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_dump, ipRecv );

    if( ipRecv.find( "request" ) )
    {
        std::lock_guard<std::mutex> lock( m_goptMutex );

        if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
        {
            m_dump = true;
        }
        else
        {
            m_dump = false;
        }
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_opticalGain )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_opticalGain, ipRecv );

    float target;
    if( indiTargetUpdate( m_indiP_opticalGain, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    { // mutex scope
        recordModalGainOpt( false );

        std::lock_guard<std::mutex> lock( m_goptMutex );
        m_opticalGain = sqrt( target );
    }

    recordModalGainOpt( false );

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_opticalGainUpdate )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_opticalGainUpdate, ipRecv );

    if( ipRecv.find( "toggle" ) )
    {
        recordModalGainOpt( false );
        {
            std::lock_guard<std::mutex> lock( m_goptMutex );

            if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
            {
                m_opticalGainUpdate = false;
            }
            else
            {
                m_opticalGainUpdate = true;

                if( m_opticalGainSource > 0 && m_opticalGainSource < 1 )
                {
                    m_opticalGain = m_opticalGainSource;
                }
            }
        }

        recordModalGainOpt( false );
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( modalGainOpt, m_indiP_opticalGainSource )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_opticalGainSource, ipRecv );

    if( ipRecv.find( m_opticalGainElement ) )
    {
        recordModalGainOpt( false );
        {
            std::lock_guard<std::mutex> lock( m_goptMutex );

            float opticalg = ipRecv[m_opticalGainElement].get<float>();

            opticalg = ( floor( opticalg * 100 + 0.5 ) ) / 100.;

            if( opticalg > 0 && opticalg < 1 )
            {
                m_opticalGainSource = opticalg;
            }

            if( m_opticalGainUpdate )
            {
                m_opticalGain = m_opticalGainSource;
            }
        }

        recordModalGainOpt( false );
    }
    return 0;
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_gainGain )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_gainGain, ipRecv );

    float target;
    if( indiTargetUpdate( m_indiP_gainGain, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    { // mutex scope
        recordModalGainOpt( false );

        std::lock_guard<std::mutex> lock( m_goptMutex );
        m_gainGain = target;
    }

    recordModalGainOpt( false );

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_gainLeak )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_gainLeak, ipRecv );

    float target;
    if( indiTargetUpdate( m_indiP_gainLeak, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    { // mutex scope
        recordModalGainOpt( false );

        std::lock_guard<std::mutex> lock( m_goptMutex );
        m_gainLeak = target;
    }

    recordModalGainOpt( false );

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_zeroGains )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_zeroGains, ipRecv );

    if( ipRecv.find( "request" ) )
    {
        std::lock_guard<std::mutex> lock( m_goptMutex );

        if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
        {
            std::fill( m_optGainSI.begin(), m_optGainSI.end(), 0.0F );
            m_siGainStateNeedsSync = false;
            m_zeroGains = true;
        }
        else
        {
            m_zeroGains = false;
        }
    }

    return 0;
}

template <typename valueT>
int modalGainOpt::handleExtrapNumberProperty( pcf::IndiProperty &localProperty,
                                              valueT &localTarget,
                                              const pcf::IndiProperty &ipRecv,
                                              const std::string &label )
{
    valueT target;
    if( indiTargetUpdate( localProperty, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    if( target != localTarget )
    {
        m_updating = true;
        std::lock_guard<std::mutex> lock( m_goptMutex );
        m_updating = true;

        localTarget = target;

        m_sinceChange = -1;
        m_updating = false;
        std::cerr << "Got " << label << ": " << localTarget << '\n';
    }

    return 0;
}

int modalGainOpt::handleExtrapToggleProperty( pcf::IndiProperty &localProperty,
                                              bool &localTarget,
                                              const pcf::IndiProperty &ipRecv,
                                              const std::string &label )
{
    static_cast<void>( localProperty );

    if( !ipRecv.find( "toggle" ) )
    {
        return log<software_error, -1>( { "Missing toggle element for " + label } );
    }

    bool target = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;

    if( target != localTarget )
    {
        m_updating = true;
        std::lock_guard<std::mutex> lock( m_goptMutex );
        m_updating = true;

        localTarget = target;

        m_sinceChange = -1;
        m_updating = false;
        std::cerr << "Got " << label << ": " << extrapBoolString( localTarget ) << '\n';
    }

    return 0;
}

int modalGainOpt::handleExtrapMethodProperty( const pcf::IndiProperty &ipRecv )
{
    int target = c_olProcessNone;
    bool found = false;
    for( auto elit = ipRecv.getElements().begin(); elit != ipRecv.getElements().end(); ++elit )
    {
        if( elit->second.getSwitchState() != pcf::IndiElement::On )
        {
            continue;
        }

        if( found )
        {
            return log<software_error, -1>( { "Multiple extrapolation methods selected in one update" } );
        }

        target = olProcessMethodFromElement( elit->first );
        if( target == c_olProcessNone && elit->first != olProcessMethodElement( c_olProcessNone ) )
        {
            return log<software_error, -1>( { "Invalid extrapolation method element: " + elit->first } );
        }

        found = true;
    }

    if( !found )
    {
        int current = c_olProcessNone;
        {
            std::lock_guard<std::mutex> lock( m_goptMutex );
            current = m_extrapOL;
        }

        const std::string currentElement = olProcessMethodElement( current );
        if( m_indiP_extrapMethod.find( currentElement ) )
        {
            for( auto elit = m_indiP_extrapMethod.getElements().begin();
                 elit != m_indiP_extrapMethod.getElements().end();
                 ++elit )
            {
                m_indiP_extrapMethod[elit->first].setSwitchState(
                    elit->first == currentElement ? pcf::IndiElement::On : pcf::IndiElement::Off );
            }
            m_indiP_extrapMethod.setState( pcf::IndiProperty::Ok );
        }

        indi::updateSelectionSwitchIfChanged( m_indiP_extrapMethod, currentElement, m_indiDriver, INDI_OK );
        return 0;
    }

    if( target != m_extrapOL )
    {
        m_updating = true;
        std::lock_guard<std::mutex> lock( m_goptMutex );
        m_updating = true;

        m_extrapOL = target;

        m_sinceChange = -1;
        m_updating = false;
        std::cerr << "Got extrapolation method: " << olProcessMethodName( m_extrapOL ) << '\n';
    }

    return 0;
}

int modalGainOpt::handleExtrapNoiseEstimateDomainProperty( const pcf::IndiProperty &ipRecv )
{
    int target = c_extrapNoiseEstimateOpenLoop;
    bool found = false;
    for( auto elit = ipRecv.getElements().begin(); elit != ipRecv.getElements().end(); ++elit )
    {
        if( elit->second.getSwitchState() != pcf::IndiElement::On )
        {
            continue;
        }

        if( found )
        {
            return log<software_error, -1>( { "Multiple noise-estimate domains selected in one update" } );
        }

        target = extrapNoiseEstimateDomainFromElement( elit->first );
        if( target == c_extrapNoiseEstimateOpenLoop &&
            elit->first != extrapNoiseEstimateDomainElement( c_extrapNoiseEstimateOpenLoop ) )
        {
            return log<software_error, -1>( { "Invalid noise-estimate-domain element: " + elit->first } );
        }

        found = true;
    }

    if( !found )
    {
        int current = c_extrapNoiseEstimateOpenLoop;
        {
            std::lock_guard<std::mutex> lock( m_goptMutex );
            current = m_extrapNoiseEstimateDomain;
        }

        const std::string currentElement = extrapNoiseEstimateDomainElement( current );
        if( m_indiP_extrapNoiseEstimateDomain.find( currentElement ) )
        {
            for( auto elit = m_indiP_extrapNoiseEstimateDomain.getElements().begin();
                 elit != m_indiP_extrapNoiseEstimateDomain.getElements().end();
                 ++elit )
            {
                m_indiP_extrapNoiseEstimateDomain[elit->first].setSwitchState(
                    elit->first == currentElement ? pcf::IndiElement::On : pcf::IndiElement::Off );
            }
            m_indiP_extrapNoiseEstimateDomain.setState( pcf::IndiProperty::Ok );
        }

        indi::updateSelectionSwitchIfChanged( m_indiP_extrapNoiseEstimateDomain,
                                              currentElement,
                                              m_indiDriver,
                                              INDI_OK );
        return 0;
    }

    if( target != m_extrapNoiseEstimateDomain )
    {
        m_updating = true;
        std::lock_guard<std::mutex> lock( m_goptMutex );
        m_updating = true;

        m_extrapNoiseEstimateDomain = target;
        m_extrapConfig.m_noiseEstimateDomain = extrapNoiseEstimateDomainName( target );

        m_sinceChange = -1;
        m_updating = false;
        std::cerr << "Got noise-estimate domain: " << extrapNoiseEstimateDomainName( target ) << '\n';
    }

    return 0;
}

int modalGainOpt::handleExtrapNoiseEstimateRangeProperty( const pcf::IndiProperty &ipRecv )
{
    int target = c_extrapNoiseEstimateHighFreq;
    bool found = false;
    for( auto elit = ipRecv.getElements().begin(); elit != ipRecv.getElements().end(); ++elit )
    {
        if( elit->second.getSwitchState() != pcf::IndiElement::On )
        {
            continue;
        }

        if( found )
        {
            return log<software_error, -1>( { "Multiple noise-estimate ranges selected in one update" } );
        }

        target = extrapNoiseEstimateRangeFromElement( elit->first );
        if( target == c_extrapNoiseEstimateHighFreq &&
            elit->first != extrapNoiseEstimateRangeElement( c_extrapNoiseEstimateHighFreq ) )
        {
            return log<software_error, -1>( { "Invalid noise-estimate-range element: " + elit->first } );
        }

        found = true;
    }

    if( !found )
    {
        int current = c_extrapNoiseEstimateHighFreq;
        {
            std::lock_guard<std::mutex> lock( m_goptMutex );
            current = m_extrapNoiseEstimateRange;
        }

        const std::string currentElement = extrapNoiseEstimateRangeElement( current );
        if( m_indiP_extrapNoiseEstimateRange.find( currentElement ) )
        {
            for( auto elit = m_indiP_extrapNoiseEstimateRange.getElements().begin();
                 elit != m_indiP_extrapNoiseEstimateRange.getElements().end();
                 ++elit )
            {
                m_indiP_extrapNoiseEstimateRange[elit->first].setSwitchState(
                    elit->first == currentElement ? pcf::IndiElement::On : pcf::IndiElement::Off );
            }
            m_indiP_extrapNoiseEstimateRange.setState( pcf::IndiProperty::Ok );
        }

        indi::updateSelectionSwitchIfChanged( m_indiP_extrapNoiseEstimateRange, currentElement, m_indiDriver, INDI_OK );
        return 0;
    }

    if( target != m_extrapNoiseEstimateRange )
    {
        m_updating = true;
        std::lock_guard<std::mutex> lock( m_goptMutex );
        m_updating = true;

        m_extrapNoiseEstimateRange = target;
        m_extrapConfig.m_noiseEstimateRange = extrapNoiseEstimateRangeName( target );

        m_sinceChange = -1;
        m_updating = false;
        std::cerr << "Got noise-estimate range: " << extrapNoiseEstimateRangeName( target ) << '\n';
    }

    return 0;
}

int modalGainOpt::handleExtrapNoiseEstimateStatisticProperty( const pcf::IndiProperty &ipRecv )
{
    int target = c_extrapNoiseEstimatePercentile;
    bool found = false;
    for( auto elit = ipRecv.getElements().begin(); elit != ipRecv.getElements().end(); ++elit )
    {
        if( elit->second.getSwitchState() != pcf::IndiElement::On )
        {
            continue;
        }

        if( found )
        {
            return log<software_error, -1>( { "Multiple noise-estimate statistics selected in one update" } );
        }

        target = extrapNoiseEstimateStatisticFromElement( elit->first );
        if( target == c_extrapNoiseEstimatePercentile &&
            elit->first != extrapNoiseEstimateStatisticElement( c_extrapNoiseEstimatePercentile ) )
        {
            return log<software_error, -1>( { "Invalid noise-estimate-statistic element: " + elit->first } );
        }

        found = true;
    }

    if( !found )
    {
        int current = c_extrapNoiseEstimatePercentile;
        {
            std::lock_guard<std::mutex> lock( m_goptMutex );
            current = m_extrapNoiseEstimateStatistic;
        }

        const std::string currentElement = extrapNoiseEstimateStatisticElement( current );
        if( m_indiP_extrapNoiseEstimateStatistic.find( currentElement ) )
        {
            for( auto elit = m_indiP_extrapNoiseEstimateStatistic.getElements().begin();
                 elit != m_indiP_extrapNoiseEstimateStatistic.getElements().end();
                 ++elit )
            {
                m_indiP_extrapNoiseEstimateStatistic[elit->first].setSwitchState(
                    elit->first == currentElement ? pcf::IndiElement::On : pcf::IndiElement::Off );
            }
            m_indiP_extrapNoiseEstimateStatistic.setState( pcf::IndiProperty::Ok );
        }

        indi::updateSelectionSwitchIfChanged( m_indiP_extrapNoiseEstimateStatistic,
                                              currentElement,
                                              m_indiDriver,
                                              INDI_OK );
        return 0;
    }

    if( target != m_extrapNoiseEstimateStatistic )
    {
        m_updating = true;
        std::lock_guard<std::mutex> lock( m_goptMutex );
        m_updating = true;

        m_extrapNoiseEstimateStatistic = target;
        m_extrapConfig.m_noiseEstimateStatistic = extrapNoiseEstimateStatisticName( target );

        m_sinceChange = -1;
        m_updating = false;
        std::cerr << "Got noise-estimate statistic: " << extrapNoiseEstimateStatisticName( target ) << '\n';
    }

    return 0;
}

int modalGainOpt::handleExtrapClosedLoopOlEstimateMethodProperty( const pcf::IndiProperty &ipRecv )
{
    int target = c_extrapClosedLoopOlEstimateEtfOnly;
    bool found = false;
    for( auto elit = ipRecv.getElements().begin(); elit != ipRecv.getElements().end(); ++elit )
    {
        if( elit->second.getSwitchState() != pcf::IndiElement::On )
        {
            continue;
        }

        if( found )
        {
            return log<software_error, -1>( { "Multiple closed-loop OL estimate methods selected in one update" } );
        }

        target = extrapClosedLoopOlEstimateMethodFromElement( elit->first );
        if( target == c_extrapClosedLoopOlEstimateEtfOnly &&
            elit->first != extrapClosedLoopOlEstimateMethodElement( c_extrapClosedLoopOlEstimateEtfOnly ) )
        {
            return log<software_error, -1>( { "Invalid closed-loop-OL-estimate-method element: " + elit->first } );
        }

        found = true;
    }

    if( !found )
    {
        int current = c_extrapClosedLoopOlEstimateEtfOnly;
        {
            std::lock_guard<std::mutex> lock( m_goptMutex );
            current = m_extrapClosedLoopOlEstimateMethod;
        }

        const std::string currentElement = extrapClosedLoopOlEstimateMethodElement( current );
        if( m_indiP_extrapClosedLoopOlEstimateMethod.find( currentElement ) )
        {
            for( auto elit = m_indiP_extrapClosedLoopOlEstimateMethod.getElements().begin();
                 elit != m_indiP_extrapClosedLoopOlEstimateMethod.getElements().end();
                 ++elit )
            {
                m_indiP_extrapClosedLoopOlEstimateMethod[elit->first].setSwitchState(
                    elit->first == currentElement ? pcf::IndiElement::On : pcf::IndiElement::Off );
            }
            m_indiP_extrapClosedLoopOlEstimateMethod.setState( pcf::IndiProperty::Ok );
        }

        indi::updateSelectionSwitchIfChanged( m_indiP_extrapClosedLoopOlEstimateMethod,
                                              currentElement,
                                              m_indiDriver,
                                              INDI_OK );
        return 0;
    }

    if( target != m_extrapClosedLoopOlEstimateMethod )
    {
        m_updating = true;
        std::lock_guard<std::mutex> lock( m_goptMutex );
        m_updating = true;

        m_extrapClosedLoopOlEstimateMethod = target;
        m_extrapConfig.m_closedLoopOlEstimateMethod = extrapClosedLoopOlEstimateMethodName( target );

        m_sinceChange = -1;
        m_updating = false;
        std::cerr << "Got closed-loop OL estimate method: " << extrapClosedLoopOlEstimateMethodName( target ) << '\n';
    }

    return 0;
}

int modalGainOpt::handleExtrapPowerLawCrossoverModeProperty( const pcf::IndiProperty &ipRecv )
{
    int target = c_extrapPowerLawCrossoverManual;
    bool found = false;
    for( auto elit = ipRecv.getElements().begin(); elit != ipRecv.getElements().end(); ++elit )
    {
        if( elit->second.getSwitchState() != pcf::IndiElement::On )
        {
            continue;
        }

        if( found )
        {
            return log<software_error, -1>( { "Multiple power-law crossover modes selected in one update" } );
        }

        target = extrapPowerLawCrossoverModeFromElement( elit->first );
        if( target == c_extrapPowerLawCrossoverManual &&
            elit->first != extrapPowerLawCrossoverModeElement( c_extrapPowerLawCrossoverManual ) )
        {
            return log<software_error, -1>( { "Invalid power-law-crossover-mode element: " + elit->first } );
        }

        found = true;
    }

    if( !found )
    {
        int current = c_extrapPowerLawCrossoverManual;
        {
            std::lock_guard<std::mutex> lock( m_goptMutex );
            current = m_extrapPowerLawCrossoverMode;
        }

        const std::string currentElement = extrapPowerLawCrossoverModeElement( current );
        if( m_indiP_extrapPowerLawCrossoverMode.find( currentElement ) )
        {
            for( auto elit = m_indiP_extrapPowerLawCrossoverMode.getElements().begin();
                 elit != m_indiP_extrapPowerLawCrossoverMode.getElements().end();
                 ++elit )
            {
                m_indiP_extrapPowerLawCrossoverMode[elit->first].setSwitchState(
                    elit->first == currentElement ? pcf::IndiElement::On : pcf::IndiElement::Off );
            }
            m_indiP_extrapPowerLawCrossoverMode.setState( pcf::IndiProperty::Ok );
        }

        indi::updateSelectionSwitchIfChanged( m_indiP_extrapPowerLawCrossoverMode,
                                              currentElement,
                                              m_indiDriver,
                                              INDI_OK );
        return 0;
    }

    if( target != m_extrapPowerLawCrossoverMode )
    {
        m_updating = true;
        std::lock_guard<std::mutex> lock( m_goptMutex );
        m_updating = true;

        m_extrapPowerLawCrossoverMode = target;
        m_extrapConfig.m_powerLawCrossoverMode = extrapPowerLawCrossoverModeName( target );

        m_sinceChange = -1;
        m_updating = false;
        std::cerr << "Got power-law crossover mode: " << extrapPowerLawCrossoverModeName( target ) << '\n';
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapMethod )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapMethod, ipRecv );
    return handleExtrapMethodProperty( ipRecv );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapNoiseEstimateDomain )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapNoiseEstimateDomain, ipRecv );
    return handleExtrapNoiseEstimateDomainProperty( ipRecv );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapNoiseEstimateRange )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapNoiseEstimateRange, ipRecv );
    return handleExtrapNoiseEstimateRangeProperty( ipRecv );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapNoiseEstimateStatistic )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapNoiseEstimateStatistic, ipRecv );
    return handleExtrapNoiseEstimateStatisticProperty( ipRecv );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapNoiseEstimateLowFreqMaxHz )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapNoiseEstimateLowFreqMaxHz, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapNoiseEstimateLowFreqMaxHz,
                                       m_extrapConfig.m_noiseEstimateLowFreqMaxHz,
                                       ipRecv,
                                       "extrap noise-estimate low-freq max hz" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapClosedLoopOlEstimateMethod )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapClosedLoopOlEstimateMethod, ipRecv );
    return handleExtrapClosedLoopOlEstimateMethodProperty( ipRecv );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawIndex )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawIndex, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapPowerLawIndex,
                                       m_extrapConfig.m_powerLawIndex,
                                       ipRecv,
                                       "extrap power-law index" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawNormFreq )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawNormFreq, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapPowerLawNormFreq,
                                       m_extrapConfig.m_powerLawNormFreq,
                                       ipRecv,
                                       "extrap power-law norm freq" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawMatchFreq )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawMatchFreq, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapPowerLawMatchFreq,
                                       m_extrapConfig.m_powerLawMatchFreq,
                                       ipRecv,
                                       "extrap power-law match freq" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawMatchFallbackWindowHz )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawMatchFallbackWindowHz, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapPowerLawMatchFallbackWindowHz,
                                       m_extrapConfig.m_powerLawMatchFallbackWindowHz,
                                       ipRecv,
                                       "extrap power-law match fallback window" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawCrossoverMode )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawCrossoverMode, ipRecv );
    return handleExtrapPowerLawCrossoverModeProperty( ipRecv );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawAutoSmoothWidthHz )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawAutoSmoothWidthHz, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapPowerLawAutoSmoothWidthHz,
                                       m_extrapConfig.m_powerLawAutoSmoothWidthHz,
                                       ipRecv,
                                       "extrap power-law auto smooth width" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawAutoMaxFreqFraction )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawAutoMaxFreqFraction, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapPowerLawAutoMaxFreqFraction,
                                       m_extrapConfig.m_powerLawAutoMaxFreqFraction,
                                       ipRecv,
                                       "extrap power-law auto max freq fraction" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapFitPowerLawIndex )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapFitPowerLawIndex, ipRecv );
    return handleExtrapToggleProperty( m_indiP_extrapFitPowerLawIndex,
                                       m_extrapConfig.m_fitPowerLawIndex,
                                       ipRecv,
                                       "extrap fit power-law index" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawOnlyAboveFreq )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawOnlyAboveFreq, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapPowerLawOnlyAboveFreq,
                                       m_extrapConfig.m_powerLawOnlyAboveFreq,
                                       ipRecv,
                                       "extrap power-law only above freq" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawFitIncludesMatchPoint )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawFitIncludesMatchPoint, ipRecv );
    return handleExtrapToggleProperty( m_indiP_extrapPowerLawFitIncludesMatchPoint,
                                       m_extrapConfig.m_powerLawFitIncludesMatchPoint,
                                       ipRecv,
                                       "extrap fit includes match point" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawFitMinFreqHz )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawFitMinFreqHz, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapPowerLawFitMinFreqHz,
                                       m_extrapConfig.m_powerLawFitMinFreqHz,
                                       ipRecv,
                                       "extrap fit min freq" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawFitMaxFreqHz )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawFitMaxFreqHz, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapPowerLawFitMaxFreqHz,
                                       m_extrapConfig.m_powerLawFitMaxFreqHz,
                                       ipRecv,
                                       "extrap fit max freq" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawFitBinWidthHz )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawFitBinWidthHz, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapPowerLawFitBinWidthHz,
                                       m_extrapConfig.m_powerLawFitBinWidthHz,
                                       ipRecv,
                                       "extrap fit bin width" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapPowerLawBlendBins )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapPowerLawBlendBins, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapPowerLawBlendBins,
                                       m_extrapConfig.m_powerLawBlendBins,
                                       ipRecv,
                                       "extrap power-law blend bins" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapDropoutGapFactor )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapDropoutGapFactor, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapDropoutGapFactor,
                                       m_extrapConfig.m_dropoutGapFactor,
                                       ipRecv,
                                       "extrap dropout gap factor" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapDropoutTinyFactor )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapDropoutTinyFactor, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapDropoutTinyFactor,
                                       m_extrapConfig.m_dropoutTinyFactor,
                                       ipRecv,
                                       "extrap dropout tiny factor" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapDropoutMaxBins )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapDropoutMaxBins, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapDropoutMaxBins,
                                       m_extrapConfig.m_dropoutMaxBins,
                                       ipRecv,
                                       "extrap dropout max bins" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapClSignificanceThreshold )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapClSignificanceThreshold, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapClSignificanceThreshold,
                                       m_extrapConfig.m_clSignificanceThreshold,
                                       ipRecv,
                                       "extrap CL significance threshold" );
}

INDI_NEWCALLBACK_DEFN( modalGainOpt, m_indiP_extrapClMinSignificantFraction )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_extrapClMinSignificantFraction, ipRecv );
    return handleExtrapNumberProperty( m_indiP_extrapClMinSignificantFraction,
                                       m_extrapConfig.m_clMinSignificantFraction,
                                       ipRecv,
                                       "extrap CL minimum significant fraction" );
}

INDI_SETCALLBACK_DEFN( modalGainOpt, m_indiP_emg )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_emg, ipRecv );

    if( ipRecv.find( "current" ) )
    {
        std::lock_guard<std::mutex> lock( m_goptMutex );

        float emg = ipRecv["current"].get<float>();

        if( emg != m_emg )
        {
            m_emg = emg;
            std::cerr << "Got EMG: " << m_emg << '\n';
        }
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( modalGainOpt, m_indiP_psdTime )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_psdTime, ipRecv );

    if( ipRecv.find( "current" ) )
    {
        float psdTime = ipRecv["current"].get<float>();

        if( psdTime != m_psdTime )
        {
            m_updating = true;
            std::lock_guard<std::mutex> lock( m_goptMutex );
            m_updating = true;

            m_psdTime = psdTime;

            m_sinceChange = -1;
            m_updating = false;

            std::cerr << "Got psdTime: " << m_psdTime << '\n';
        }
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( modalGainOpt, m_indiP_psdAvgTime )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_psdAvgTime, ipRecv );

    if( ipRecv.find( "current" ) )
    {
        float psdAvgTime = ipRecv["current"].get<float>();

        if( psdAvgTime != m_psdAvgTime )
        {
            m_updating = true;
            std::lock_guard<std::mutex> lock( m_goptMutex );
            m_updating = true;

            m_psdAvgTime = psdAvgTime;

            m_sinceChange = -1;
            m_updating = false;

            std::cerr << "Got psdAvgTime: " << m_psdAvgTime << '\n';
        }
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( modalGainOpt, m_indiP_loop )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_loop, ipRecv );

    if( ipRecv.find( "toggle" ) )
    {
        bool state;
        bool changed{ false };

        if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
        {
            state = true;
        }
        else
        {
            state = false;
        }

        if( state != m_loop )
        {
            changed = true;
            recordModalGainOpt( false );
            m_updating = true;
            std::lock_guard<std::mutex> lock( m_goptMutex );
            m_updating = true;

            m_loop = state;

            if( !m_loop )
            {
                m_autoUpdate = false;
                m_dump = false;
            }

            m_sinceChange = -1;
            m_updating = false;
            std::cerr << "Got loop: " << m_loop << '\n';
        }

        if( changed )
        {
            recordModalGainOpt( false );
        }
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( modalGainOpt, m_indiP_siGain )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_siGain, ipRecv );

    if( ipRecv.find( "current" ) )
    {
        float gain = ipRecv["current"].get<float>();

        if( gain != m_gain && !m_pcOn )
        {
            m_updating = true;
            std::lock_guard<std::mutex> lock( m_goptMutex );
            m_updating = true;

            m_gain = gain;

            if( m_loop )
            {
                m_sinceChange = -1;
            }

            m_updating = false;

            std::cerr << "Got gain: " << m_gain << '\n';
        }
        else
        {
            m_gain = gain; // for the m_pcOn case
        }
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( modalGainOpt, m_indiP_siMult )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_siMult, ipRecv );

    if( ipRecv.find( "current" ) )
    {
        float mc = ipRecv["current"].get<float>();

        if( mc != m_mult && !m_pcOn )
        {
            m_updating = true;
            std::lock_guard<std::mutex> lock( m_goptMutex );
            m_updating = true;

            m_mult = mc;

            if( m_loop )
            {
                m_sinceChange = -1;
            }

            m_goptUpdated = true;
            m_updating = false;
            std::cerr << "Got mc: " << m_mult << '\n';
        }
        else
        {
            m_mult = mc; // for the m_pcOn case
        }
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( modalGainOpt, m_indiP_pcGain )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pcGain, ipRecv );

    if( ipRecv.find( "current" ) )
    {
        float gain = ipRecv["current"].get<float>();

        if( gain != m_pcGain && m_pcOn )
        {
            m_updating = true;
            std::lock_guard<std::mutex> lock( m_goptMutex );
            m_updating = true;

            m_pcGain = gain;

            if( m_loop )
            {
                m_sinceChange = -1;
            }

            m_updating = false;

            std::cerr << "Got pc gain: " << m_pcGain << '\n';
        }
        else
        {
            m_pcGain = gain; // for the !m_pcOn case
        }
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( modalGainOpt, m_indiP_pcMult )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pcMult, ipRecv );

    if( ipRecv.find( "current" ) )
    {
        float mc = ipRecv["current"].get<float>();

        if( mc != m_pcMult && m_pcOn )
        {
            m_updating = true;
            std::lock_guard<std::mutex> lock( m_goptMutex );
            m_updating = true;

            m_pcMult = mc;

            if( m_loop )
            {
                m_sinceChange = -1;
            }

            m_pcgoptUpdated = true;
            m_updating = false;
            std::cerr << "Got pc mc: " << m_pcMult << '\n';
        }
        else
        {
            m_pcMult = mc; // for the !m_pcOn case
        }
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( modalGainOpt, m_indiP_pcOn )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pcOn, ipRecv );

    if( ipRecv.find( "toggle" ) )
    {
        bool state;

        if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
        {
            state = true;
        }
        else
        {
            state = false;
        }

        if( state != m_pcOn )
        {
            m_updating = true;
            std::lock_guard<std::mutex> lock( m_goptMutex );
            m_updating = true;

            m_pcOn = state;

            m_sinceChange = -1;
            m_updating = false;
            std::cerr << "Got pcOn: " << std::boolalpha << m_pcOn << '\n';
        }
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // modalGainOpt_hpp
