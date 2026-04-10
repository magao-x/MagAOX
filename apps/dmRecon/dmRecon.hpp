/** \file dmRecon.hpp
 * \brief The MagAO-X DM shape reconstructor
 *
 * \ingroup dmRecon_files
 */

#ifndef dmRecon_hpp
#define dmRecon_hpp

#include <atomic>
#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/sigproc/gramSchmidt.hpp>
#include <mx/math/templateBLAS.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <mx/math/cuda/cudaPtr.hpp>
#include <mx/math/cuda/cublasHandle.hpp>
#include <mx/math/cuda/templateCublas.hpp>
#include <mx/math/eigenLapack.hpp>
#include <mx/sigproc/basisUtils2D.hpp>

namespace MagAOX
{
namespace app
{

/** \defgroup dmRecon DM Shape Reconstructor
 * \brief Reconstruct the wavefront corresponding to a DM shape
 *
 * <a href="../handbook/operating/software/apps/dmRecon.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup dmRecon_files DM Shape Reconstructor Files
 * \ingroup dmRecon
 */

struct dmModesShmimT
{
    static std::string configSection()
    {
        return "dmModes";
    };

    static std::string indiPrefix()
    {
        return "dmModes";
    };
};

struct dmMaskShmimT
{
    static std::string configSection()
    {
        return "dmMask";
    };

    static std::string indiPrefix()
    {
        return "dmMask";
    };
};

struct dmCommandShmimT
{
    static std::string configSection()
    {
        return "dmCommand";
    };

    static std::string indiPrefix()
    {
        return "dmCommand";
    };
};

/** MagAO-X application to perform wavefront reconstruction from a DM surface
 *
 * \ingroup dmRecon
 *
 */
class dmRecon : public MagAOXApp<true>,
                public dev::shmimMonitor<dmRecon, dmModesShmimT>,
                public dev::shmimMonitor<dmRecon, dmMaskShmimT>,
                public dev::shmimMonitor<dmRecon, dmCommandShmimT>,
                public dev::frameGrabber<dmRecon>,
                public dev::telemeter<dmRecon>
{
    // Give the test harness access.
    friend class dmRecon_test;

    friend class dev::shmimMonitor<dmRecon, dmModesShmimT>;
    typedef dev::shmimMonitor<dmRecon, dmModesShmimT> dmModesSMT;

    friend class dev::shmimMonitor<dmRecon, dmMaskShmimT>;
    typedef dev::shmimMonitor<dmRecon, dmMaskShmimT> dmMaskSMT;

    friend class dev::shmimMonitor<dmRecon, dmCommandShmimT>;
    typedef dev::shmimMonitor<dmRecon, dmCommandShmimT> dmCommandSMT;

    friend class dev::frameGrabber<dmRecon>;
    typedef dev::frameGrabber<dmRecon> frameGrabberT;

    static constexpr bool c_frameGrabber_flippable = false;

    friend class dev::telemeter<dmRecon>;
    typedef dev::telemeter<dmRecon> telemeterT;

    /// Floating point type in which to do all calculations.
    typedef float realT;

  protected:
    /** \name Configurable Parameters
     *@{
     */

    int m_loopNumber{ 1 }; ///< The loop number.  Default is 1 as in aol1.

    std::string m_respMPath; ///< Optional response matrix.  If set then CM modes are converted by this response.

    std::string m_fpsSource{ "camwfs" }; /**< Device name for getting fps of the loop.
                                              This device must have *.fps.current.  Default is camwfs*/

    int m_numModes{ 0 }; ///< Number of modes to reconstruct.  If 0 (default) all modes in CM are used.

    int m_inverseNumModes{
        0 }; /**< Number of modes to use for pseudo-inverse truncation.  If 0 (default) all modes are used.*/

    uint16_t m_gpuIndex{ 0 }; /**< Index of the GPU to use for calculations */

    bool m_useGPU{ false }; /**< Flag controlling whether the GPU is used for calculations */

    ///@}

    mx::improc::eigenImage<float> m_respM;

    uint32_t m_width{ 0 };

    uint32_t m_height{ 0 };

    uint32_t m_depth{ 0 };

    float m_fps{ 0 }; ///< Current FPS from the FPS source.

    mx::improc::eigenImage<realT> m_PInv; ///< The pseudo-inverse

    // clang-format off
    #ifdef MXLIB_CUDA

    mx::cuda::cudaPtr<realT> m_PInv_GPU; ///< The pseudo-inverse on the GPU

    #endif
    // clang-format on

    std::atomic<bool> m_dmModesReady{ false }; ///< Flag indicating that the DM modes are ready for processing

    mx::improc::eigenImage<float> m_mask;

    std::vector<size_t> m_maskIDX; ///< The index of masked pixels

    std::atomic<bool> m_dmMaskReady{ false }; ///< Flag indicating that the DM mask is ready for processing

    std::atomic<bool> m_commandReady{ false }; ///< Flag indicating that all sizes match and arrays are ready for processing

    std::atomic<bool> m_fgWaiting{ false }; ///< Flag indicating that the FG thread is waiting for the command thread

    mx::improc::eigenImage<float> m_command; ///< The DM command, copied out of the incoming shmim

    mx::improc::eigenImage<float> m_modevals; ///< The calculated mode amplitudes

    std::atomic<bool> m_writeDMf{ false };

    std::string                  m_monShmimName;
    mx::improc::milkImage<float> m_modevalMon; ///< The actual calculated modevals.

    mx::improc::milkImage<float> m_modeval;
    mx::improc::milkImage<float> m_modevalDiff;

    // clang-format off
    #ifdef MXLIB_CUDA

    mx::cuda::cudaPtr<float> m_command_GPU;

    mx::cuda::cudaPtr<float> m_modevals_GPU;

    #endif
    // clang-format on

    sem_t m_smSemaphore{ 0 }; ///< Semaphore used to synchronize the fg thread and the dm command thread.

    std::atomic<bool> m_updated{ false }; ///< Flag indicating that the mode vals have been updated

    std::mutex m_modevalsMutex; ///< Guards the modevals buffer during producer/consumer handoff.

    mx::cuda::cublasHandle m_cublas; ///< Handle for the cuBLAS library

  public:
    /// Default c'tor.
    dmRecon();

    /// D'tor, declared and defined for noexcept.
    ~dmRecon() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] an application configuration
                        from which to load values*/
    );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for dmRecon.
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

    /// Set the GPU index
    /** Uses m_gpuIndex.  On errors it sets m_useGPU to false.
     *
     */
    int setGPU();

    /// Allocate method for the dm modes shmimMonitor
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    int allocate( const dmModesShmimT & /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Process images for the dm modes shmimMonitor
    /**
     * \returns 0 on sucess
     * \returns -1 on an error
     */
    int processImage( void *curr_src,       ///< [in] pointer to start of current frame.
                      const dmModesShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    /// Allocate method for the dm mask shmimMonitor
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    int allocate( const dmMaskShmimT & /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Process images for the dm mask shmimMonitor
    /**
     * \returns 0 on sucess
     * \returns -1 on an error
     */
    int processImage( void *curr_src,      ///< [in] pointer to start of current frame.
                      const dmMaskShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    /// Allocate method for the dm command shmimMonitor
    /**
     * \returns 0 on success
     * \returns -1 on an error
     */
    int allocate( const dmCommandShmimT & /**< [in] tag to differentiate shmimMonitor parents.*/ );

    /// Process images for the dm command shmimMonitor
    /**
     * \returns 0 on sucess
     * \returns -1 on an error
     */
    int processImage( void *curr_src,         ///< [in] pointer to start of current frame.
                      const dmCommandShmimT & ///< [in] tag to differentiate shmimMonitor parents.
    );

    /** \name Framegrabber Interface */
    /**
     * @{
     */

    int configureAcquisition();

    float fps();

    int startAcquisition();

    int acquireAndCheckValid();

    int loadImageIntoStream( void *dest );

    int reconfig();

    ///@}
  protected:
    /** \name INDI Interface
     *
     * @{
     */
    pcf::IndiProperty m_indiP_fpsSource;
    INDI_SETCALLBACK_DECL( dmRecon, m_indiP_fpsSource );

    pcf::IndiProperty m_indiP_fps;

    pcf::IndiProperty m_indiP_writeDMf;
    INDI_NEWCALLBACK_DECL( dmRecon, m_indiP_writeDMf );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */

    int checkRecordTimes();

    int recordTelem( const telem_fgtimings * );

    ///@}
};

inline dmRecon::dmRecon() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    frameGrabberT::m_ownShmim = false;

    return;
}

inline void dmRecon::setupConfig()
{
    config.add( "recon.loopNumber",
                "",
                "recon.loopNumber",
                argType::Required,
                "recon",
                "loopNumber",
                false,
                "int",
                "The loop number.  Default is 1 as in aol1." );

    config.add( "recon.respMPath",
                "",
                "recon.respMPath",
                argType::Required,
                "recon",
                "respMPath",
                false,
                "int",
                "Optional response matrix.  If set then CM modes are converted by this response." );

    config.add( "recon.numModes",
                "",
                "recon.numModes",
                argType::Required,
                "recon",
                "numModes",
                false,
                "int",
                "Number of modes to reconstruct.  If 0 (default) all modes in CM are used." );

    config.add( "recon.inverseNumModes",
                "",
                "recon.inverseNumModes",
                argType::Required,
                "recon",
                "inverseNumModes",
                false,
                "int",
                "Number of modes to use for pseudo-inverse truncation.  If 0 (default) all modes are used." );

    config.add( "recon.fpsSource",
                "",
                "recon.fpsSource",
                argType::Required,
                "recon",
                "fpsSource",
                false,
                "string",
                "Device name for getting fps of the loop.  This device should have *.fps.current.  "
                "Default is camwfs" );

    config.add( "recon.gpuIndex",
                "",
                "recon.gpuIndex",
                argType::Required,
                "recon",
                "gpuIndex",
                false,
                "int",
                "Index of the GPU to use for calculations.  Default is 0." );

    config.add( "recon.useGPU",
                "",
                "recon.useGPU",
                argType::Required,
                "recon",
                "useGPU",
                false,
                "bool",
                "Flag controlling whether the GPU is used for calculations. Default is false." );

    SHMIMMONITORT_SETUP_CONFIG( dmModesSMT, config );

    SHMIMMONITORT_SETUP_CONFIG( dmMaskSMT, config );

    SHMIMMONITORT_SETUP_CONFIG( dmCommandSMT, config );

    FRAMEGRABBER_SETUP_CONFIG( config );

    TELEMETER_SETUP_CONFIG( config );
}

inline int dmRecon::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_loopNumber, "recon.loopNumber" );
    _config( m_respMPath, "recon.respMPath" );
    _config( m_numModes, "recon.numModes" );
    _config( m_inverseNumModes, "recon.inverseNumModes" );
    _config( m_fpsSource, "recon.fpsSource" );
    _config( m_gpuIndex, "recon.gpuIndex" );
    _config( m_useGPU, "recon.useGPU" );

    std::string loopName = std::format( "aol{}", m_loopNumber );

    dmModesSMT::m_shmimName        = loopName + "_CMmodesDM";
    dmModesSMT::m_getExistingFirst = true;
    SHMIMMONITORT_LOAD_CONFIG( dmModesSMT, _config );

    dmCommandSMT::m_shmimName        = std::format( "dm{:02}disp_delta", m_loopNumber );
    dmCommandSMT::m_getExistingFirst = true;
    SHMIMMONITORT_LOAD_CONFIG( dmCommandSMT, _config );

    dmMaskSMT::m_shmimName        = std::format( "dm{:02}disp_actmask", m_loopNumber );
    dmMaskSMT::m_getExistingFirst = true;
    SHMIMMONITORT_LOAD_CONFIG( dmMaskSMT, _config );

    frameGrabberT::m_shmimName = loopName + "_modevalDMf";
    FRAMEGRABBER_LOAD_CONFIG( _config );

    m_monShmimName = frameGrabberT::m_shmimName + "_mon";

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

inline void dmRecon::loadConfig()
{
    loadConfigImpl( config );
}

inline int dmRecon::appStartup()
{
    REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsSource, std::string( "fps" ) );

    createROIndiNumber( m_indiP_fps, "fps" );
    m_indiP_fps.add( pcf::IndiElement( "current" ) );
    if( registerIndiPropertyReadOnly( m_indiP_fps ) < 0 )
    {
        log<software_error>( { "" } );
        return -1;
    }

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_writeDMf, "writeDMf" );

    if( sem_init( &m_smSemaphore, 0, 0 ) < 0 )
    {
        log<software_critical>( { errno, "Initializing S.M. semaphore" } );
        return -1;
    }

    if( m_respMPath != "" )
    {
        mx::fits::fitsFile<float> ff;
        ff.read( m_respM, m_respMPath );

        m_width  = sqrt( m_respM.rows() );
        m_height = m_width;
    }

    dmModesSMT::m_getExistingFirst = true;
    SHMIMMONITORT_APP_STARTUP( dmModesSMT );
    dmMaskSMT::m_getExistingFirst = true;
    SHMIMMONITORT_APP_STARTUP( dmMaskSMT );
    SHMIMMONITORT_APP_STARTUP( dmCommandSMT );

    FRAMEGRABBER_APP_STARTUP;

    TELEMETER_APP_STARTUP;

    state( stateCodes::OPERATING );

    return 0;
}

int dmRecon::appLogic()
{
    SHMIMMONITORT_APP_LOGIC( dmModesSMT );
    SHMIMMONITORT_APP_LOGIC( dmMaskSMT );
    SHMIMMONITORT_APP_LOGIC( dmCommandSMT );

    FRAMEGRABBER_APP_LOGIC;

    TELEMETER_APP_LOGIC;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( m_writeDMf.load() )
    {
        updateSwitchIfChanged( m_indiP_writeDMf, "toggle", pcf::IndiElement::On );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_writeDMf, "toggle", pcf::IndiElement::Off );
    }

    SHMIMMONITORT_UPDATE_INDI( dmModesSMT );
    SHMIMMONITORT_UPDATE_INDI( dmMaskSMT );
    SHMIMMONITORT_UPDATE_INDI( dmCommandSMT );

    FRAMEGRABBER_UPDATE_INDI;

    return 0;
}

inline int dmRecon::appShutdown()
{
    SHMIMMONITORT_APP_SHUTDOWN( dmModesSMT );
    SHMIMMONITORT_APP_SHUTDOWN( dmMaskSMT );
    SHMIMMONITORT_APP_SHUTDOWN( dmCommandSMT );

    FRAMEGRABBER_APP_SHUTDOWN;

    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int dmRecon::setGPU()
{
    // clang-format off
    #ifdef MXLIB_CUDA // clang-format on

    if( !m_useGPU )
    {
        return 0;
    }

    int deviceCount;
    int devicecntMax = 100;

    cudaError_t ce = cudaGetDeviceCount( &deviceCount );

    if( ce != cudaSuccess )
    {

        log<software_error>( { std::format( "cudaGetDeviceCount returned error: "
                                            "[{}] {}\nNOT USING GPU",
                                            cudaGetErrorName( ce ),
                                            cudaGetErrorString( ce ) ) } );

        m_useGPU = false;
        state( state(), true );
        return -1;
    }

    std::string msg = std::format( "CUDA: found {} devices\n", deviceCount );

    if( deviceCount > devicecntMax )
    {
        deviceCount = 0;
        msg += "      greater than devicecntMax\n";
    }
    if( deviceCount < 0 )
    {
        msg += "      less than zero\n";
    }

    if( deviceCount == 0 )
    {
        msg += "      no devices found!\nNOT USING GPU";
        log<software_error>( { msg } );
        m_useGPU = false;
        state( state(), true );
        return -1;
    }

    for( int k = 0; k < deviceCount; k++ )
    {
        cudaDeviceProp deviceProp;
        ce = cudaGetDeviceProperties( &deviceProp, k );

        if( ce != cudaSuccess )
        {
            msg += std::format( "cudaGetDeviceProperties returned error: "
                                "[{}] {}\nNOT USING GPU",
                                cudaGetErrorName( ce ),
                                cudaGetErrorString( ce ) );
            log<software_error>( { msg } );
            m_useGPU = false;
            state( state(), true );
            return -1;
        }

        int clockRate;
        ce = cudaDeviceGetAttribute( &clockRate, cudaDevAttrClockRate, k );

        if( ce != cudaSuccess )
        {
            msg += std::format( "cudaGetDeviceAttribute returned error: "
                                "[{}] {}\nNOT USING GPU",
                                cudaGetErrorName( ce ),
                                cudaGetErrorString( ce ) );
            log<software_error>( { msg } );
            m_useGPU = false;
            state( state(), true );
            return -1;
        }

        msg += std::format( "      Device {} / {} [ {} ] has compute capability {}.{}.\n",
                            k + 1,
                            deviceCount,
                            deviceProp.name,
                            deviceProp.major,
                            deviceProp.minor );

        msg += std::format( "          Total amount of global memory: {} MBytes\n",
                            (float)deviceProp.totalGlobalMem / 1048576.0f );

        msg += std::format( "          Multiprocessors: {}\n", deviceProp.multiProcessorCount );
        msg += std::format( "          Clock rate: {} MHz ({} GHz)\n", clockRate * 1e-3f, clockRate * 1e-6f );
    }

    if( m_gpuIndex >= deviceCount )
    {
        msg += std::format( "gpuIndex = {} is not valid for {} devices\nNOT USING GPU", m_gpuIndex, deviceCount );
        log<software_error>( { msg } );
        m_useGPU = false;
        state( state(), true );
        return -1;
    }

    ce = cudaSetDevice( m_gpuIndex );

    if( ce != cudaSuccess )
    {
        msg += std::format( "cudaSetDevice returned error: "
                            "[{}] {}\nNOT USING GPU",
                            cudaGetErrorName( ce ),
                            cudaGetErrorString( ce ) );
        log<software_error>( { msg } );
        m_useGPU = false;
        state( state(), true );
        return -1;
    }

    msg += std::format( "Set GPU Index to device {} ( {} / {})\n", m_gpuIndex, m_gpuIndex + 1, deviceCount );

    cublasStatus_t cbs = m_cublas.create();

    if( cbs != CUBLAS_STATUS_SUCCESS )
    {
        msg += std::format( "cublasHandle create returned error: "
                            "[{}] {}\nNOT USING GPU",
                            cublasGetStatusName( cbs ),
                            cublasGetStatusString( cbs ) );
        log<software_error>( { msg } );
        m_useGPU = false;
        state( state(), true );
        return -1;
    }

    msg += "      cuBLAS initialized";

    log<text_log>( msg );

    return 0;

        // clang-format off
    #else // MXLIB_CUDA
    // clang-format on

    if( m_useGPU )
    {
        log<software_error>( { "mxlib was compiled without CUDA support. NOT USING GPU" } );

        m_useGPU = false;
        state( state(), true );
        return -1;
    }

    return 0;

        // clang-format off
    #endif // MXLIB_CUDA
    // clang-format on
}

int dmRecon::allocate( const dmModesShmimT & )
{
    m_dmModesReady = false;

    std::cerr << "modes not ready\n";

    dmCommandSMT::m_restart = true;

    if( m_respM.rows() == 0 )
    {
        m_width  = dmModesSMT::m_width;
        m_height = dmModesSMT::m_height;
    }

    if( m_numModes == 0 )
    {
        m_depth = dmModesSMT::m_depth;
    }
    else
    {
        m_depth = m_numModes;
        if( m_depth > dmModesSMT::m_depth )
        {
            m_depth = dmModesSMT::m_depth;
        }
    }

    // Can't process modes until mask is ready.
    if( m_width != dmMaskSMT::m_width || m_height != dmMaskSMT::m_height || !m_dmMaskReady.load() )
    {
        mx::sys::milliSleep( 1000 );
        dmModesSMT::m_restart = true;
        return 0; // This won't log an error, but setting m_restart will cause it to loop again until sizes match
    }

    // This will let us go on to processImage

    // Do a type check for float
    return 0;
}

int dmRecon::processImage( void *curr_src, const dmModesShmimT & )
{
    if( m_dmModesReady.load() == true )
    {
        // This means new image has come in.  We need to reset and restart everything.
        dmModesSMT::m_restart   = true;
        dmMaskSMT::m_restart    = true;
        dmCommandSMT::m_restart = true;
        return 0;
    }

    mx::improc::eigenCube<float> dmModes( dmModesSMT::m_width, dmModesSMT::m_height, m_depth );

    for( size_t n = 0; n < dmModesSMT::m_width * dmModesSMT::m_height * m_depth; ++n )
    {
        dmModes.data()[n] = reinterpret_cast<float *>( curr_src )[n];
    }

    // Wait for m_commandReady to become false
    while( m_commandReady.load() == true && !m_shutdown && dmModesSMT::m_restart == false )
    {
        mx::sys::milliSleep( 1000 );
    }

    if( m_respM.rows() > 0 )
    {
        mx::improc::eigenCube<realT> tmpc;

        int nr = sqrt( m_respM.rows() );

        tmpc.resize( nr, nr, dmModes.planes() );

        std::cerr << __LINE__ << '\n';

        for( int p = 0; p < tmpc.planes(); ++p )
        {
            // cast to matrices for math
            Eigen::Map<Eigen::Matrix<float,-1,-1>> outim(tmpc.image(p).data(), nr*nr,1);
            Eigen::Map<Eigen::Matrix<float,-1,-1>> inim(dmModes.image(p).data(), dmModes.rows()*dmModes.cols(),1);

            outim = (m_respM.matrix() * inim);

            float norm = sqrt(tmpc.image(p).square().sum()/m_maskIDX.size());
            float scale = sqrt(dmModes.image(p).square().sum()/ (dmModes.rows()*dmModes.cols()));

            tmpc.image(p) *= scale/norm;
        }

        dmModes = tmpc;

        mx::fits::fitsFile<float> ff;
        ff.write( "wmodes.fits", dmModes );
    }

    mx::improc::eigenImage<float> maskedDMModes;

    maskedDMModes.resize( m_maskIDX.size(), dmModes.planes() );

    // Load only the unmasked pixels
    for( int rr = 0; rr < maskedDMModes.cols(); ++rr )
    {
        for( size_t n = 0; n < m_maskIDX.size(); ++n )
        {
            maskedDMModes( n, rr ) = dmModes.image( rr ).data()[m_maskIDX[n]];
        }
    }

    realT condition;

    int nRejected;

    realT maxCondition = -1 * m_inverseNumModes; // Specify number of modes to keep.  If 0 it's all.

    int rv = mx::math::eigenPseudoInverse( m_PInv, condition, nRejected, maskedDMModes, maxCondition );

    if( rv < 0 )
    {
        log<software_error>( { 0, rv, "error in eigenPseudoInverse " } );
        m_shutdown = 1;
        return -1;
    }

    std::cerr << "PInv: " << m_PInv.rows() << ' ' << m_PInv.cols() << '\n';

    mx::fits::fitsFile<float> ff;
    ff.write( "PInv.fits", m_PInv );

    log<text_log>( std::format( "Inverted CMmodesDM. Rejected {} "
                                "of {} modes, condition numer = {}",
                                nRejected,
                                dmModes.planes(),
                                condition ) );

    m_dmModesReady = true;

    std::cerr << "modes ready\n";
    return 0;
}

int dmRecon::allocate( const dmMaskShmimT & )
{
    m_dmMaskReady = false;

    std::cerr << "mask not ready\n";

    dmCommandSMT::m_restart = true;

    // Do a type check for float
    return 0;
}

int dmRecon::processImage( void *curr_src, const dmMaskShmimT & )
{
    if( m_dmMaskReady.load() == true )
    {
        // This means an new image has come in.  We need to reset and restart everything.
        dmModesSMT::m_restart = true;
        dmMaskSMT::m_restart  = true;

        dmCommandSMT::m_restart = true;

        return 0;
    }

    // Wait for m_commandReady to become false
    while( m_commandReady.load() == true && !m_shutdown && dmMaskSMT::m_restart == false )
    {
        mx::sys::milliSleep( 1000 );
    }

    m_mask =
        mx::improc::eigenMap<float>( reinterpret_cast<float *>( curr_src ), dmMaskSMT::m_width, dmMaskSMT::m_height );

    m_maskIDX.clear();

    size_t n = 0;

    int nmax = 0;
    for( int rr = 0; rr < m_mask.rows(); ++rr )
    {
        for( int cc = 0; cc < m_mask.cols(); ++cc )
        {
            if( m_mask( rr, cc ) == 1 )
            {
                m_maskIDX.push_back( n );
                nmax = n;
            }

            ++n;
        }
    }

    std::cerr << n << ' ' << nmax << '\n';

    std::cerr << "Got mask of size " << m_mask.rows() << " x " << m_mask.cols() << " with " << m_maskIDX.size()
              << " good pixels.\n";

    m_dmMaskReady = true;
    std::cerr << "mask ready\n";
    return 0;
}

int dmRecon::allocate( const dmCommandShmimT & )
{
    // This is the only place that m_commandReady can be changed
    m_commandReady = false;
    std::cerr << "command not ready\n";

    if( !m_dmModesReady.load() || !m_dmMaskReady.load() || dmCommandSMT::m_width != m_width || dmCommandSMT::m_height != m_height ||
        dmCommandSMT::m_width != dmMaskSMT::m_width || dmCommandSMT::m_height != dmMaskSMT::m_height )
    {
        dmCommandSMT::m_restart   = true;
        frameGrabberT::m_reconfig = true;

        mx::sys::milliSleep( 1000 );

        return 0; // This won't log an error, but setting m_restart will cause it to reconnect again until sizes match
    }

    if( !m_fgWaiting.load() )
    {
        frameGrabberT::m_reconfig = true;
        mx::sys::milliSleep( 1000 );

        dmCommandSMT::m_restart = true;
        return 0; // This won't log an error, but setting m_restart will cause it to reconnect again until sizes match
    }

    m_command.resize( m_maskIDX.size(), 1 );

    m_modevals.resize( m_PInv.rows(), 1 );

    m_modevalMon.create( m_monShmimName, m_PInv.rows(), 1 );

    m_modeval.open(std::format("aol{}_modevalDM", m_loopNumber));

    m_modevalDiff.create( std::format("aol{}_modevalDMf_diff", m_loopNumber), m_PInv.rows(), 1 );

    // clang-format off
    #ifdef MXLIB_CUDA
    // clang-format on

    if( m_useGPU )
    {
        // Do all initializations and uploads here so it's in the right thread on the right device
        if( setGPU() < 0 )
        {
            log<software_error>( { "setting GPU device failed." } );
            m_useGPU = false;
            state( state(), true );
            return -1;
        }

        mx::error_t ec = m_PInv_GPU.upload( m_PInv.data(), m_PInv.rows(), m_PInv.cols() );

        if( ec != mx::error_t::noerror )
        {
            return log<software_error, -1>( { std::format(
                "error uploading PInv to GPU: [{}] {}", mx::errorName( ec ), mx::errorMessage( ec ) ) } );
        }

        ec = m_command_GPU.resize( m_command.rows() * m_command.cols() );
        if( ec != mx::error_t::noerror )
        {
            return log<software_error, -1>( { std::format(
                "error allocating command on GPU: [{}] {}", mx::errorName( ec ), mx::errorMessage( ec ) ) } );
        }

        ec = m_modevals_GPU.resize( m_modevals.rows() * m_modevals.cols() );
        if( ec != mx::error_t::noerror )
        {
            return log<software_error, -1>( { std::format(
                "error allocating modevals on GPU: [{}] {}", mx::errorName( ec ), mx::errorMessage( ec ) ) } );
        }
    }

        // clang-format off
    #endif // MXLIB_CUDA
    // clang-format on

    m_updated      = false;
    m_commandReady = true;

    std::cerr << "command ready\n";

    return 0;
}

int dmRecon::processImage( void *curr_src, const dmCommandShmimT & )
{
    if( !m_commandReady.load() )
    {
        dmCommandSMT::m_restart = true;
        return 0;
    }

    // Set atime to now
    clock_gettime( CLOCK_REALTIME, &m_currImageTimestamp );

    // extract masked pixels
    for( size_t n = 0; n < m_maskIDX.size(); ++n )
    {
        m_command( n, 0 ) = reinterpret_cast<float *>( curr_src )[m_maskIDX[n]];
    }

    // clang-format off
    #ifdef MXLIB_CUDA // clang-format on
    if( !m_useGPU )
    {
        //  CPU:
        std::lock_guard<std::mutex> guard( m_modevalsMutex );
        m_modevals = ( m_PInv.matrix() * m_command.matrix() ).array();
    }
    else
    {
        // GPU:
        mx::error_t ec = m_command_GPU.upload( m_command.data() );
        if( ec != mx::error_t::noerror )
        {
            return log<software_error, -1>( { std::format(
                "error uploading command to GPU: [{}] {}", mx::errorName( ec ), mx::errorMessage( ec ) ) } );
        }

        float alpha = 1;
        float beta  = 0;

        cublasStatus_t cbs = mx::cuda::cublasTgemv( m_cublas,
                                                    CUBLAS_OP_N,
                                                    m_PInv_GPU.rows(),
                                                    m_PInv_GPU.cols(),
                                                    &alpha,
                                                    m_PInv_GPU.data(),
                                                    m_PInv_GPU.rows(),
                                                    m_command_GPU.data(),
                                                    1,
                                                    &beta,
                                                    m_modevals_GPU.data(),
                                                    1 );

        if( cbs != CUBLAS_STATUS_SUCCESS )
        {
            return log<software_error, -1>( { std::format( "error downloading modevals from GPU: [{}] {}",
                                                           cublasGetStatusName( cbs ),
                                                           cublasGetStatusString( cbs ) ) } );
        }

        std::lock_guard<std::mutex> guard( m_modevalsMutex );
        ec = m_modevals_GPU.download( m_modevals.data() );
        if( ec != mx::error_t::noerror )
        {
            return log<software_error, -1>( { std::format(
                "error downloading modevals from GPU: [{}] {}", mx::errorName( ec ), mx::errorMessage( ec ) ) } );
        }
    }

        // clang-format off
    #else // MXLIB_CUDA

    // CPU:
    m_modevals = (m_PInv.matrix() * m_command.matrix()).array()

    #endif // MXLIB_CUDA
    // clang-format on

    m_updated = true;

    if( m_writeDMf.load() )
    {
        // trigger framegrabber
        if( sem_post( &m_smSemaphore ) < 0 )
        {
            log<software_critical>( { errno, 0, "Error posting to semaphore" } );
            return -1;
        }
    }

    // write to the monitor stream
    m_modevalMon.setWrite(1);
    m_modevalDiff.setWrite();
    for(uint32_t r = 0; r < m_modevalMon.rows(); ++r)
    {
        m_modevalMon(r,0) = m_modevals(r,0);
        m_modevalDiff(r,0) = m_modevals(r,0) - m_modeval(r,0);
    }

    m_modevalMon.post();
    m_modevalDiff.post();

    return 0;
}

int dmRecon::configureAcquisition()
{
    if( !m_commandReady.load() )
    {
        m_fgWaiting = true;
        mx::sys::milliSleep( 100 );
        return -1;
    }

    m_fgWaiting = false;

    frameGrabberT::m_width    = m_modevals.rows();
    frameGrabberT::m_height   = m_modevals.cols();
    frameGrabberT::m_dataType = _DATATYPE_FLOAT;

    static int logged = 0;

    if( frameGrabberT::m_imageStream != nullptr )
    {
        ImageStreamIO_closeIm( frameGrabberT::m_imageStream );
        free( frameGrabberT::m_imageStream );
        frameGrabberT::m_imageStream = nullptr;
    }

    // b/c ImageStreamIO prints every single time, and latest version don't support stopping it yet, and that
    // isn't thread-safe-able anyway we do our own checks.  This is the same code in ImageStreamIO_openIm...
    int  SM_fd;
    char SM_fname[200];
    ImageStreamIO_filename( SM_fname, sizeof( SM_fname ), frameGrabberT::m_shmimName.c_str() );
    SM_fd = open( SM_fname, O_RDWR );

    if( SM_fd == -1 )
    {
        if( !logged )
        {
            log<text_log>( "ImageStream " + frameGrabberT::m_shmimName + " not found (yet).  Retrying . . .",
                           logPrio::LOG_NOTICE );
            logged = 1;
        }

        return 1;
    }

    // Found and opened,  close it and then use ImageStreamIO
    logged = 0;
    close( SM_fd );

    frameGrabberT::m_imageStream = reinterpret_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );

    if( ImageStreamIO_openIm( frameGrabberT::m_imageStream, frameGrabberT::m_shmimName.c_str() ) == 0 )
    {
        if( frameGrabberT::m_imageStream->md[0].sem < SEMAPHORE_MAXVAL )
        {
            ImageStreamIO_closeIm( frameGrabberT::m_imageStream );
            free( frameGrabberT::m_imageStream );
            frameGrabberT::m_imageStream = nullptr;

            return 1; // We just need to wait for the server process to finish startup.
        }
        else
        {
            char SM_fname[200];
            ImageStreamIO_filename( SM_fname, sizeof( SM_fname ), frameGrabberT::m_shmimName.c_str() );

            struct stat buffer;
            int         rv = stat( SM_fname, &buffer );

            if( rv != 0 )
            {
                log<software_critical>( { errno,
                                          "Could not get inode for " + frameGrabberT::m_shmimName +
                                              ". Source process will need to be restarted." } );

                ImageStreamIO_closeIm( frameGrabberT::m_imageStream );

                free( frameGrabberT::m_imageStream );

                frameGrabberT::m_imageStream = nullptr;

                m_shutdown = true;

                return -1;
            }

            frameGrabberT::m_inode = buffer.st_ino;
        }
    }
    else
    {
        free( frameGrabberT::m_imageStream );
        frameGrabberT::m_imageStream = nullptr;

        return 1; // be patient
    }

    return 0;
}

float dmRecon::fps()
{
    return m_fps;
}

int dmRecon::startAcquisition()
{

    std::cerr << "startAcquisition\n";
    return 0;
}

int dmRecon::acquireAndCheckValid()
{
    timespec ts;

    errno = 0;
    if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
    {
        log<software_critical>( { errno, "clock_gettime" } );
        return -1;
    }

    ts.tv_sec += 1;

    if( !m_commandReady.load() )
    {
        return 1;
    }

    if( sem_timedwait( &m_smSemaphore, &ts ) == 0 )
    {
        if( m_updated.load() && m_commandReady.load() )
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }
    else
    {
        return 1;
    }

    return 0;
}

int dmRecon::loadImageIntoStream( void *dest )
{
    std::lock_guard<std::mutex> guard( m_modevalsMutex );
    memcpy( dest, m_modevals.data(), m_modevals.rows() * m_modevals.cols() * sizeof( float ) );

    return 0;
}

int dmRecon::reconfig()
{
    return 0;
}

INDI_SETCALLBACK_DEFN( dmRecon, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( ipRecv.find( "current" ) != true ) // this isn't valid
    {
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    realT fps = ipRecv["current"].get<float>();

    if( fps != m_fps )
    {
        m_fps = fps;
        updateIfChanged( m_indiP_fps, "current", m_fps );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmRecon, m_indiP_writeDMf )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_writeDMf, ipRecv );

    if( ipRecv.find( "toggle" ) != true ) // this isn't valid
    {
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        m_writeDMf = true;
        log<text_log>( "writing modevalDMf", logPrio::LOG_INFO );
        updateSwitchIfChanged( m_indiP_writeDMf, "toggle", pcf::IndiElement::On );
    }
    else
    {
        m_writeDMf = false;
        log<text_log>( "not writing modevalDMf", logPrio::LOG_INFO );
        updateSwitchIfChanged( m_indiP_writeDMf, "toggle", pcf::IndiElement::Off );
    }

    return 0;
}

int dmRecon::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

int dmRecon::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

} // namespace app
} // namespace MagAOX

#endif // dmRecon_hpp
