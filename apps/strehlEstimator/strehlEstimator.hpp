/** \file strehlEstimator.hpp
 * \brief The MagAO-X XXXXXX header file
 *
 * \ingroup strehlEstimator_files
 */

#ifndef strehlEstimator_hpp
#define strehlEstimator_hpp

#include <mx/ao/analysis/aoSystem.hpp>
using namespace mx::math;

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup strehlEstimator
 * \brief The XXXXXX application to do YYYYYYY
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

/// The MagAO-X xxxxxxxx
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
    typedef dev::shmimMonitor<strehlEstimator, wfsavgShmimT>  wfsavgShmimMonitorT;
    typedef dev::shmimMonitor<strehlEstimator, wfsmaskShmimT> wfsmaskShmimMonitorT;

  protected:
    /** \name Configurable Parameters
     *@{
     */

    int m_loopNum{ 1 }; ///< The number of the loop. Used to set shmim names, as in aolN_wfsmask.

    std::string m_wfsDevice{ "camwfs" };

    std::string m_stagebsDevice{ "stagebs" };

    float m_again{ 28.547 };

    float m_qe{ 0.53 };

    float m_F0_6535{ 4.2e10 };

    float m_F0_HaIR{ 5.3e10 };

    float m_lam0_6535{ 0.791 };

    float m_lam0_HaIR{ 0.837 };

    float m_qe_6535{ 0.53 };

    float m_qe_HaIR{ 0.53 };

    ///@}

    float m_fps{ 2000 };

    float m_emg{ 1 };

    float m_F0{ m_F0_6535 };

    float m_lam0{ m_lam0_6535 };

    float m_seeing{ 0.64 };

    float m_r0{ 0.16 };

    float m_elevation{ 90 };

    int m_npix;

    float m_counts{ 0 };

    float m_mag{ 0 };

    mx::improc::eigenImage<float> m_wfsmask;
    mx::improc::eigenImage<float> m_wfsavg;

    mx::AO::analysis::aoSystem<float, mx::AO::analysis::vonKarmanSpectrum<float>> m_aosys;

    double m_dimm_fwhm_corr{ 0 }; ///< DIMM elevation corrected FWHM
    int    m_dimm_time{ 0 };      ///< Seconds since midnight of DIMM measurement.

    double m_mag1_fwhm_corr{ 0 }; ///< MAG1 elevation corrected FWHM
    int    m_mag1_time{ 0 };      ///< Seconds since midnight of MAG1 measurement.

    double m_mag2_fwhm_corr{ 0 }; ///< MAG2 elevation corrected FWHM
    int    m_mag2_time{ 0 };      ///< Seconds since midnight of MAG2 measurement.

  public:
    /// Default c'tor.
    strehlEstimator();

    /// D'tor, declared and defined for noexcept.
    ~strehlEstimator() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for strehlEstimator.
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

    void calcMag();

    /** INDI
     * @{
     */

    pcf::IndiProperty m_indiP_fps;
    pcf::IndiProperty m_indiP_emg;

    pcf::IndiProperty m_indiP_stage;

    pcf::IndiProperty m_indiP_tcsi_seeing;
    pcf::IndiProperty m_indiP_tcsi_telpos;
    pcf::IndiProperty m_indiP_seeing_magaox;

    pcf::IndiProperty m_indiP_mag;
    pcf::IndiProperty m_indiP_strehl;
    pcf::IndiProperty m_indiP_wfe;

    INDI_SETCALLBACK_DECL( strehlEstimator, m_indiP_fps );
    INDI_SETCALLBACK_DECL( strehlEstimator, m_indiP_emg );
    INDI_SETCALLBACK_DECL( strehlEstimator, m_indiP_stage );
    INDI_SETCALLBACK_DECL( strehlEstimator, m_indiP_tcsi_seeing );
    INDI_SETCALLBACK_DECL( strehlEstimator, m_indiP_tcsi_telpos );

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

int strehlEstimator::appStartup()
{

    SHMIMMONITORT_APP_STARTUP( wfsavgShmimMonitorT );
    SHMIMMONITORT_APP_STARTUP( wfsmaskShmimMonitorT );

    REG_INDI_SETPROP( m_indiP_fps, m_wfsDevice, "fps" );

    REG_INDI_SETPROP( m_indiP_emg, m_wfsDevice, "emgain" );

    REG_INDI_SETPROP( m_indiP_stage, m_stagebsDevice, "presetName" );

    REG_INDI_SETPROP( m_indiP_tcsi_seeing, "tcsi", "seeing" );
    REG_INDI_SETPROP( m_indiP_tcsi_telpos, "tcsi", "telpos" );

    CREATE_REG_INDI_RO_NUMBER( m_indiP_mag, "star_mag", "Star Magnitude", "Error Budget" );
    m_indiP_mag.add( pcf::IndiElement( "current", 0 ) );

    CREATE_REG_INDI_RO_NUMBER( m_indiP_strehl, "strehl_optimal", "Strehl", "Error Budget" );
    m_indiP_strehl.add( pcf::IndiElement( "pyramid", 0 ) );

    CREATE_REG_INDI_RO_NUMBER( m_indiP_wfe, "wfe_predicted", "WFE", "Error Budget" );
    m_indiP_wfe.add( pcf::IndiElement( "total", 0 ) );
    m_indiP_wfe.add( pcf::IndiElement( "measurement", 0 ) );
    m_indiP_wfe.add( pcf::IndiElement( "time_delay", 0 ) );
    m_indiP_wfe.add( pcf::IndiElement( "fitting", 0 ) );

    state( stateCodes::OPERATING );

    return 0;
}

int strehlEstimator::appLogic()
{
    SHMIMMONITORT_APP_LOGIC( wfsavgShmimMonitorT );
    SHMIMMONITORT_APP_LOGIC( wfsmaskShmimMonitorT );

    SHMIMMONITORT_UPDATE_INDI( wfsavgShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( wfsmaskShmimMonitorT );

    m_aosys.starMag( m_mag );
    m_aosys.F0( m_qe * m_F0 );
    m_aosys.lam_wfs( m_lam0 * 1e-6 );
    m_aosys.lam_sci( m_lam0 * 1e-6 );
    m_aosys.ron_wfs( std::vector<float>( { 245.0f / m_emg } ) );
    m_aosys.npix_wfs( std::vector<float>( { 1.0f * m_npix } ) );
    m_aosys.minTauWFS( std::vector<float>( { 1.0f / m_fps } ) );
    m_aosys.tauWFS( 1. / m_fps );
    m_aosys.atm.r_0( m_r0, 0.5e-6 );
    m_aosys.zeta( ( 90. - m_elevation ) * 3.14159 / 180. );

    updateIfChanged( m_indiP_strehl, "pyramid", m_aosys.strehl() );
    updatesIfChanged<double>( m_indiP_wfe,
                              { "total", "measurement", "time_delay", "fitting" },
                              { sqrt( m_aosys.wfeVar() ) * ( 1000 * m_lam0 / two_pi<float>() ),
                                sqrt( m_aosys.measurementErrorTotal() ) * ( 1000 * m_lam0 / two_pi<float>() ),
                                sqrt( m_aosys.timeDelayErrorTotal() ) * ( 1000 * m_lam0 / two_pi<float>() ),
                                sqrt( m_aosys.fittingErrorTotal() ) * ( 1000 * m_lam0 / two_pi<float>() ) } );
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

    m_npix = m_wfsmask.sum();

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

    m_mag = -2.5 * log10( m_counts * m_again / m_emg * m_fps / ( m_qe * m_F0 ) );

    updateIfChanged( m_indiP_mag, "current", m_mag );
}

INDI_SETCALLBACK_DEFN( strehlEstimator, m_indiP_fps )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fps, ipRecv );

    if( ipRecv.find( "current" ) )
    {
        float fps = ipRecv["current"].get<float>();

        if( fps != m_fps )
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

        if( emg != m_emg )
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

    // if( ipRecv.find( "presetName" ) )
    {
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
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( strehlEstimator, m_indiP_tcsi_seeing )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_tcsi_seeing, ipRecv );

    if( ipRecv.find( "dimm_fwhm" ) )
    {
        float seeing = ipRecv["dimm_fwhm"].get<float>();

        if( seeing != m_seeing )
        {
            m_seeing = seeing;
            std::cerr << "Got seeing: " << m_seeing << '\n';

            m_r0 = 0.2063 * 0.5 / m_seeing;
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

        if( elevation != m_elevation )
        {
            m_elevation = elevation;
            std::cerr << "Got elevation: " << m_elevation << '\n';
        }
    }
    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // strehlEstimator_hpp
