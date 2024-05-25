/** \file dmPokeXCorr.hpp
 * \brief The MagAO-X DM Poke Centering header file
 *
 * \ingroup dmPokeXCorr_files
 */

#ifndef dmPokeXCorr_hpp
#define dmPokeXCorr_hpp

#include <mx/improc/imageFilters.hpp>
#include <mx/improc/imageXCorrFFT.hpp>
using namespace mx::improc;

#include <mx/math/fit/fitGaussian.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup dmPokeXCorr
 * \brief The MagAO-X application to center a DM pupil by poking actuators
 *
 * <a href="../handbook/operating/software/apps/dmPokeXCorr.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup dmPokeXCorr_files
 * \ingroup dmPokeXCorr
 */

namespace MagAOX
{
namespace app
{

struct zrespShmimT
{
    static std::string configSection()
    {
        return "zrespM";
    };
    static std::string indiPrefix()
    {
        return "zrespM";
    };
};

/// The MagAO-X DM to PWFS alignment Application
/**
 * \ingroup dmPokeXCorr
 */
class dmPokeXCorr : public MagAOXApp<true>,
                    public dev::dmPokeWFS<dmPokeXCorr>,
                    public dev::shmimMonitor<dmPokeXCorr, dev::dmPokeWFS<dmPokeXCorr>::wfsShmimT>,
                    public dev::shmimMonitor<dmPokeXCorr, dev::dmPokeWFS<dmPokeXCorr>::darkShmimT>,
                    public dev::shmimMonitor<dmPokeXCorr, zrespShmimT>,
                    public dev::telemeter<dmPokeXCorr>
{
    // Give the test harness access.
    friend class dmPokeXCorr_test;

    friend class dev::dmPokeWFS<dmPokeXCorr>;
    
    typedef dev::dmPokeWFS<dmPokeXCorr> dmPokeWFST;

    friend class dev::shmimMonitor<dmPokeXCorr, dev::dmPokeWFS<dmPokeXCorr>::wfsShmimT>;

    typedef dev::shmimMonitor<dmPokeXCorr, dev::dmPokeWFS<dmPokeXCorr>::wfsShmimT> shmimMonitorT;

    friend class dev::shmimMonitor<dmPokeXCorr, dev::dmPokeWFS<dmPokeXCorr>::darkShmimT>;

    typedef dev::shmimMonitor<dmPokeXCorr, dev::dmPokeWFS<dmPokeXCorr>::darkShmimT> darkShmimMonitorT;

    friend class dev::shmimMonitor<dmPokeXCorr, zrespShmimT>;

    typedef dev::shmimMonitor<dmPokeXCorr, zrespShmimT> zrespShmimMonitorT;

    friend class dev::telemeter<dmPokeXCorr>;

    typedef dev::telemeter<dmPokeXCorr> telemeterT;

protected:
    /** \name Configurable Parameters
     *@{
     */

    std::string m_zRespMFile;

    ///@}

    mx::improc::imageXCorrFFT<eigenImage<float>> m_xcorr;

    mx::improc::milkImage<float> m_refIm;


public:
    /// Default c'tor.
    dmPokeXCorr();

    /// D'tor, declared and defined for noexcept.
    ~dmPokeXCorr() noexcept
    {
    }

    /**\name MagAOX Interface
     *
     * @{
     */
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/);

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for dmPokeXCorr.
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

    ///@}

    shmimMonitorT & shmimMonitor()
    {
        return *dynamic_cast<shmimMonitorT *>(this);
    }

    darkShmimMonitorT & darkShmimMonitor()
    {
        return *static_cast<darkShmimMonitorT *>(this);
    }

    /** \name zrespShmimMonitorT inteface
      * @{
      */

    int allocate(const zrespShmimT & dummy );

    int processImage( void * curr_src,    
                      const zrespShmimT &  dummy
                    );

    using dmPokeWFST::allocate;

    using dmPokeWFST::processImage;

    ///@}

    /** \name dmPokeWFS Interface
      * @{
      */

    /// Run the sensor steps
    /** Coordinates the actions of poking and collecting images.
     * Upon completion this calls runSensor.  If \p firstRun == true, one time
     * actions such as taking a dark can be executed.
     *
     * \returns 0 on success
     * \returns \< 0 on an error
     */
    int runSensor(bool firstRun /**< [in] flag indicating this is the first call.  triggers taking a dark if true.*/);

    /// Analyze the poke image
    /** This analyzes the resulting poke image and reports the results.
     *
     * \returns 0 on success
     * \returns \< 0 on an error
     */
    int analyzeSensor();

    ///@} 

    /** \name INDI Interface
     * @{
     */
protected:
    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    int checkRecordTimes();

    

    ///@}
};

dmPokeXCorr::dmPokeXCorr() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
    return;
}

void dmPokeXCorr::setupConfig()
{
    DMPOKEWFS_SETUP_CONFIG(config);

    SHMIMMONITORT_SETUP_CONFIG( zrespShmimMonitorT, config ); 

    TELEMETER_SETUP_CONFIG(config);
}

int dmPokeXCorr::loadConfigImpl(mx::app::appConfigurator &_config)
{
    DMPOKEWFS_LOAD_CONFIG(_config);

    SHMIMMONITORT_LOAD_CONFIG( zrespShmimMonitorT, _config);

    TELEMETER_LOAD_CONFIG(_config);

    return 0;
}

void dmPokeXCorr::loadConfig()
{
    if (loadConfigImpl(config) < 0)
    {
        m_shutdown = true;
    }
}

int dmPokeXCorr::appStartup()
{
    DMPOKEWFS_APP_STARTUP;

    SHMIMMONITORT_APP_STARTUP(zrespShmimMonitorT);

    TELEMETER_APP_STARTUP;

    //Gotta connect to the DM stream to find out its size
    mx::improc::milkImage<float> mdm;
    try
    {
        mdm.open(m_dmChan);    
    }
    catch(const std::exception& e) //this can check for invalid_argument and distinguish not existing
    {
        return log<software_error,-1>({__FILE__, __LINE__, std::string("exception opening DM: ") + e.what()});
    }

    
    state(stateCodes::READY);

    return 0;
}

int dmPokeXCorr::appLogic()
{
    DMPOKEWFS_APP_LOGIC;

    SHMIMMONITORT_APP_LOGIC( zrespShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( zrespShmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( shmimMonitorT );
    SHMIMMONITORT_UPDATE_INDI( darkShmimMonitorT );

    TELEMETER_APP_LOGIC;

    return 0;
}

int dmPokeXCorr::appShutdown()
{
    DMPOKEWFS_APP_SHUTDOWN;

    SHMIMMONITORT_APP_SHUTDOWN( zrespShmimMonitorT );

    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int dmPokeXCorr::allocate(const zrespShmimT & dummy )
{
    static_cast<void>(dummy);

    if(m_refIm.rows() != zrespShmimMonitorT::m_width || m_refIm.cols() != zrespShmimMonitorT::m_height)
    {
        m_refIm.create( m_configName + "_refIm", zrespShmimMonitorT::m_width, zrespShmimMonitorT::m_height);
    }

    return 0;
}

int dmPokeXCorr::processImage( void * curr_src,    
                               const zrespShmimT &  dummy
                             )
{
    static_cast<void>(dummy);

    //Gotta connect to the DM stream to find out its size
    //can't assume that we have connected anywhere else.
    mx::improc::milkImage<float> mdm;
    try
    {
        mdm.open(m_dmChan);    
    }
    catch(const std::exception& e) 
    {
        return log<software_error,-1>({__FILE__, __LINE__, std::string("exception opening DM: ") + e.what()});
    }

    mx::improc::eigenCube<float> zRespM ((float *) curr_src, zrespShmimMonitorT::m_width, zrespShmimMonitorT::m_height, zrespShmimMonitorT::m_depth);

    mx::improc::eigenImage<float> refIm;
    refIm.resize(zRespM.rows(), zRespM.cols());
    refIm.setZero();

    for(size_t n = 0; n < m_poke_x.size(); ++n)
    {
        int actno = m_poke_y[n]*mdm.rows() + m_poke_x[n];

        refIm += zRespM.image(actno);
    }

    m_refIm = refIm;
    
    m_xcorr.refIm(m_refIm());

    return 0;
}

int dmPokeXCorr::runSensor(bool firstRun)
{
    static_cast<void>(firstRun);

    int rv = dmPokeWFST::basicRunSensor();

    if (rv > 0)
    {
        return 0;
    }
    else if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__});
        return rv;
    }

    return 0;
}

int dmPokeXCorr::analyzeSensor()
{
    if(m_xcorr.refIm().rows() != m_pokeImage().rows() || m_xcorr.refIm().rows() != m_pokeImage().cols() )
    {
        return 0;
    }

    float xs, ys;

    m_xcorr(xs, ys, m_pokeImage());

    std::cerr << "dmPokeXCorr::analyzeSensor: " << xs << " " << ys << "\n";
    
    if(updateMeasurement(xs, ys) < 0)
    {
        return log<software_error,-1>({__FILE__, __LINE__, "error from dmPokeWFS::updateMeasurement"});
    }

    return 0;
}

int dmPokeXCorr::checkRecordTimes()
{
    return telemeterT::checkRecordTimes(telem_pokeloop());
}



} // namespace app
} // namespace MagAOX

#endif // dmPokeXCorr_hpp
