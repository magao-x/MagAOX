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

    std::string m_zRespMFile; ///< Unused.  The zonal response is read from the zrespM shmim.

    float m_minXCorr{ 0.3 }; /**< Minimum normalized cross-correlation peak for a valid measurement.  Lower values are
                                  rejected as no poke response.  <= 0 disables the check.  Default is 0.3. */

    ///@}

    mx::improc::imageXCorrFFT<eigenImage<float>> m_xcorr; ///< Cross-correlator holding the poke reference image.

    mx::improc::milkImage<float> m_refIm; ///< Stream publishing the poke reference image built from zrespM.

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
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

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

    shmimMonitorT &shmimMonitor()
    {
        return *dynamic_cast<shmimMonitorT *>( this );
    }

    darkShmimMonitorT &darkShmimMonitor()
    {
        return *static_cast<darkShmimMonitorT *>( this );
    }

    /** \name zrespShmimMonitorT inteface
     * @{
     */

    int allocate( const zrespShmimT &dummy );

    int processImage( void *curr_src, const zrespShmimT &dummy );

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
    int runSensor( bool firstRun /**< [in] flag indicating this is the first call.  triggers taking a dark if true.*/ );

    /// Analyze the poke image
    /** This analyzes the resulting poke image and reports the results.
     *
     * \returns 0 on success
     * \returns \< 0 on an error
     */
    int analyzeSensor();

    /// Convert a cross-correlation peak to a correlation coefficient
    /** The cross-correlator normalizes both images to zero mean and unit variance, so the peak divided by the number of
     * pixels is the correlation coefficient at the peak.
     *
     * \returns the normalized peak, or 0 if the poke image is empty
     */
    float normalizedPeak( float peak /**< [in] the cross-correlation peak */ );

    /// Check whether a normalized cross-correlation peak indicates a poke response
    /**
     * \returns true if \p normPeak is at least m_minXCorr, or if m_minXCorr <= 0
     * \returns false otherwise
     */
    bool peakValid( float normPeak /**< [in] the normalized cross-correlation peak */ ) const;

    /// Cross-correlate the negated poke image with the reference
    /** A sign-inverted poke response gives an in-range delta here.
     *
     * \returns 0 on success
     * \returns -1 on error, which is logged
     */
    int xcorrInverted( float &deltaX, /**< [out] the x delta of the negated poke image, in pixels */
                       float &deltaY, /**< [out] the y delta of the negated poke image, in pixels */
                       float &peak    /**< [out] the cross-correlation peak of the negated poke image */
    );

    /// Save diagnostics for the current measurement
    /** Adds the cross-correlation results to the header, then writes the base diagnostics and the reference image.
     *
     * \returns 0 on success
     * \returns -1 on error, which is logged
     */
    int saveDiagnostics( std::string       &prefix,    /**< [out] the path and file name prefix of the files written */
                         const std::string &reason,    /**< [in] why diagnostics are being saved */
                         float              deltaX,    /**< [in] the measured x delta, in pixels */
                         float              deltaY,    /**< [in] the measured y delta, in pixels */
                         float              peak,      /**< [in] the cross-correlation peak */
                         float              invDeltaX, /**< [in] the x delta of the negated poke image, in pixels */
                         float              invDeltaY, /**< [in] the y delta of the negated poke image, in pixels */
                         float              invPeak    /**< [in] the cross-correlation peak of the negated poke image */
    );

    /// Reject an out-of-range or low-correlation measurement
    /** Checks the negated poke image to distinguish a sign inversion from a missing poke response, saves diagnostics,
     * and logs an error with a hint to the operator.  The measurement is not published.
     *
     * \returns -1 so that the caller can stop measuring
     */
    int rejectMeasurement( float deltaX, /**< [in] the measured x delta, in pixels */
                           float deltaY, /**< [in] the measured y delta, in pixels */
                           float peak    /**< [in] the cross-correlation peak */
    );

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

dmPokeXCorr::dmPokeXCorr() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    darkShmimMonitorT::m_getExistingFirst  = true;
    zrespShmimMonitorT::m_getExistingFirst = true;

    return;
}

void dmPokeXCorr::setupConfig()
{
    DMPOKEWFS_SETUP_CONFIG( config );

    SHMIMMONITORT_SETUP_CONFIG( zrespShmimMonitorT, config );

    config.add( "xcorr.minPeak",
                "",
                "xcorr.minPeak",
                argType::Required,
                "xcorr",
                "minPeak",
                false,
                "float",
                "Minimum normalized cross-correlation peak for a valid measurement.  Lower values are rejected as no "
                "poke response.  <= 0 disables the check.  Default 0.3." );

    TELEMETER_SETUP_CONFIG( config );
}

int dmPokeXCorr::loadConfigImpl( mx::app::appConfigurator &_config )
{
    DMPOKEWFS_LOAD_CONFIG( _config );

    SHMIMMONITORT_LOAD_CONFIG( zrespShmimMonitorT, _config );

    _config( m_minXCorr, "xcorr.minPeak" );

    TELEMETER_LOAD_CONFIG( _config );

    return 0;
}

void dmPokeXCorr::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = true;
    }
}

int dmPokeXCorr::appStartup()
{
    DMPOKEWFS_APP_STARTUP;

    SHMIMMONITORT_APP_STARTUP( zrespShmimMonitorT );

    TELEMETER_APP_STARTUP;

    // Gotta connect to the DM stream to find out its size
    mx::improc::milkImage<float> mdm;
    try
    {
        mdm.open( m_dmChan );
    }
    catch( const std::exception &e ) // this can check for invalid_argument and distinguish not existing
    {
        return log<software_error, -1>( { __FILE__, __LINE__, std::string( "exception opening DM: " ) + e.what() } );
    }

    state( stateCodes::READY );

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

int dmPokeXCorr::allocate( const zrespShmimT &dummy )
{
    static_cast<void>( dummy );

    if( m_refIm.rows() != zrespShmimMonitorT::m_width || m_refIm.cols() != zrespShmimMonitorT::m_height )
    {
        m_refIm.create( m_configName + "_refIm", zrespShmimMonitorT::m_width, zrespShmimMonitorT::m_height );
    }

    return 0;
}

int dmPokeXCorr::processImage( void *curr_src, const zrespShmimT &dummy )
{
    static_cast<void>( dummy );

    // Gotta connect to the DM stream to find out its size
    // can't assume that we have connected anywhere else.
    mx::improc::milkImage<float> mdm;
    try
    {
        mdm.open( m_dmChan );
    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, std::string( "exception opening DM: " ) + e.what() } );
    }

    mx::improc::eigenCube<float> zRespM(
        (float *)curr_src, zrespShmimMonitorT::m_width, zrespShmimMonitorT::m_height, zrespShmimMonitorT::m_depth );

    mx::improc::eigenImage<float> refIm;
    refIm.resize( zRespM.rows(), zRespM.cols() );
    refIm.setZero();

    for( size_t n = 0; n < m_poke_x.size(); ++n )
    {
        int actno = m_poke_y[n] * mdm.rows() + m_poke_x[n];

        refIm += zRespM.image( actno );
    }

    m_refIm = refIm;

    std::cerr << "Got reference: " << refIm.rows() << " x " << refIm.cols() << '\n';
    try
    {
        m_xcorr.peakMethod( xcorrPeakMethod::mftOversamp );
        m_xcorr.normalize( true );
        m_xcorr.maxLag( 10 );
        m_xcorr.tol( 0.001 );

        m_xcorr.refIm( refIm, 1 );

    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, std::string( "exception caught: \n" ) + e.what() } );
    }
    return 0;
}

int dmPokeXCorr::runSensor( bool firstRun )
{
    static_cast<void>( firstRun );

    int rv = dmPokeWFST::basicRunSensor();

    if( rv > 0 )
    {
        return 0;
    }
    else if( rv < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return rv;
    }

    return 0;
}

int dmPokeXCorr::analyzeSensor()
{
    if( m_xcorr.refIm().rows() != m_pokeImage().rows() || m_xcorr.refIm().cols() != m_pokeImage().cols() )
    {
        std::cerr << "refIm:  " << m_xcorr.refIm().rows() << " x " << m_xcorr.refIm().cols() << '\n';
        std::cerr << "pokeIm: " << m_pokeImage().rows() << " x " << m_pokeImage().cols() << '\n';
        return log<software_error, -1>( { __FILE__, __LINE__, "reference is not valid" } );
    }

    float xs, ys, pk;

    try
    {
        m_xcorr( xs, ys, pk, m_pokeImage() );
    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, std::string( "exception caught: \n" ) + e.what() } );
    }

    std::cerr.precision( 5 );
    std::cerr << "dmPokeXCorr::analyzeSensor: " << xs << " " << ys << "\n";

    if( !measurementInRange( xs, ys ) || !peakValid( normalizedPeak( pk ) ) )
    {
        return rejectMeasurement( xs, ys, pk );
    }

    if( m_saveDiagnostics.exchange( false ) )
    {
        float       xi = 0, yi = 0, pki = 0;
        std::string prefix;
        if( xcorrInverted( xi, yi, pki ) == 0 && saveDiagnostics( prefix, "requested", xs, ys, pk, xi, yi, pki ) == 0 )
        {
            log<text_log>( "diagnostics saved to " + prefix + "_*.fits" );
        }
    }

    if( updateMeasurement( xs, ys ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from dmPokeWFS::updateMeasurement" } );
    }

    return 0;
}

float dmPokeXCorr::normalizedPeak( float peak )
{
    float npix = m_pokeImage().rows() * m_pokeImage().cols();

    return ( npix > 0 ) ? peak / npix : 0;
}

bool dmPokeXCorr::peakValid( float normPeak ) const
{
    return ( m_minXCorr <= 0 || normPeak >= m_minXCorr );
}

int dmPokeXCorr::xcorrInverted( float &deltaX, float &deltaY, float &peak )
{
    eigenImage<float> inv = -m_pokeImage();

    try
    {
        m_xcorr( deltaX, deltaY, peak, inv );
    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, std::string( "exception caught: \n" ) + e.what() } );
    }

    return 0;
}

int dmPokeXCorr::saveDiagnostics( std::string       &prefix,
                                  const std::string &reason,
                                  float              deltaX,
                                  float              deltaY,
                                  float              peak,
                                  float              invDeltaX,
                                  float              invDeltaY,
                                  float              invPeak )
{
    mx::fits::fitsHeader<> head;
    head.append( "REASON", reason, "why diagnostics were saved" );
    head.append( "DELTAX", deltaX, "measured x delta [pix]" );
    head.append( "DELTAY", deltaY, "measured y delta [pix]" );
    head.append( "XCPEAK", peak, "cross-correlation peak" );
    head.append( "XCNORM", normalizedPeak( peak ), "normalized cross-correlation peak" );
    head.append( "MINXCORR", m_minXCorr, "minimum normalized peak" );
    head.append( "INVDX", invDeltaX, "x delta of negated poke image [pix]" );
    head.append( "INVDY", invDeltaY, "y delta of negated poke image [pix]" );
    head.append( "INVPEAK", invPeak, "cross-correlation peak of negated poke image" );
    head.append( "INVNORM", normalizedPeak( invPeak ), "normalized peak of negated poke image" );
    uint64_t wfsCnt0 = ( shmimMonitorT::m_imageStream.md != nullptr ) ? shmimMonitorT::m_imageStream.md->cnt0 : 0;
    head.append( "WFSCNT0", wfsCnt0, "WFS stream cnt0 at write" );

    if( writeDiagnostics( prefix, head ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from writeDiagnostics" } );
    }

    mx::fits::fitsFile<float> ff;
    eigenImage<float>         ref = m_xcorr.refIm();
    if( !!ff.write( prefix + "_refIm.fits", ref, head ) )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error writing " + prefix + "_refIm.fits" } );
    }

    return 0;
}

int dmPokeXCorr::rejectMeasurement( float deltaX, float deltaY, float peak )
{
    float xi = 0, yi = 0, pki = 0;
    bool  invOK = ( xcorrInverted( xi, yi, pki ) == 0 );

    float norm    = normalizedPeak( peak );
    float invNorm = normalizedPeak( pki );

    std::ostringstream msg;
    msg << "measurement rejected: delta (" << deltaX << ", " << deltaY << "), normalized xcorr peak " << norm << ". ";

    if( !measurementInRange( deltaX, deltaY ) )
    {
        msg << "Delta exceeds maxDelta " << m_maxDelta << " pix. ";
    }

    if( !peakValid( norm ) )
    {
        msg << "Peak is below minPeak " << m_minXCorr << ". ";
    }

    if( invOK && measurementInRange( xi, yi ) && peakValid( invNorm ) && invNorm > norm )
    {
        msg << "Inverted response confirmed: the negated poke image gives delta (" << xi << ", " << yi
            << ") with normalized peak " << invNorm << ". Change the sign of poke_amp. ";
    }
    else if( invOK && !peakValid( norm ) && !peakValid( invNorm ) )
    {
        msg << "No poke response detected (negated peak " << invNorm << "); check that the DM is responding. ";
    }
    else if( invOK )
    {
        msg << "This is not a simple sign inversion (negated poke image gives delta (" << xi << ", " << yi
            << ") with normalized peak " << invNorm << "). ";
    }

    std::string prefix;
    if( saveDiagnostics( prefix, "rejected", deltaX, deltaY, peak, xi, yi, pki ) == 0 )
    {
        msg << "Measurement stopped; diagnostics written to " << prefix << "_*.fits";
    }
    else
    {
        msg << "Measurement stopped; diagnostics could not be written";
    }

    log<text_log>( msg.str(), logPrio::LOG_ERROR );

    return -1;
}

int dmPokeXCorr::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_pokeloop() );
}

} // namespace app
} // namespace MagAOX

#endif // dmPokeXCorr_hpp
