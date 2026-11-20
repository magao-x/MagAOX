/** \file xrif2fits.hpp
 * \brief The xrif2fits class declaration and definition.
 *
 * \ingroup xrif2fits_files
 */

#ifndef xrif2fits_hpp
#define xrif2fits_hpp

#include <ImageStreamIO/ImageStreamIO.h>

#include <xrif/xrif.h>

#include <sstream>
#include <iomanip>
#include <set>

#include <mx/ioutils/fileUtils.hpp>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include <mx/ioutils/fits/fitsFile.hpp>

#include <mx/sys/timeUtils.hpp>
using namespace mx::sys::tscomp;
using namespace mx::sys::tsop;

#include "../../libMagAOX/libMagAOX.hpp"

#ifndef DEBUG_CRUMB
    #ifdef DEBUG
        #define DEBUG_CRUMB( msg )                                                                                     \
            {                                                                                                          \
                std::cerr << msg << '(' << __FILE__ << ' ' << __LINE__ << "\n";                                        \
            }
    #else
        #define DEBUG_CRUMB( msg )
    #endif
#endif

#ifndef XRIF2FITS_DEBUG_CRUMB
    #ifdef XRIF2FITS_DEBUG
        #define XRIF2FITS_DEBUG_CRUMB( msg )                                                                           \
            {                                                                                                          \
                std::cerr << msg << " (" << __FILE__ << ' ' << __LINE__ << ")\n";                                      \
            }
    #else
        #define XRIF2FITS_DEBUG_CRUMB( msg )
    #endif
#endif

#define ERR_INVOKED_NAME( msg )                                                                                        \
    std::cerr << invokedName + ": " << msg << "\n  at:" << __FILE__ << ' ' << __LINE__ << '\n';

#define ERR_INVOKED_NAME_ERRNO( msg )                                                                                  \
    std::cerr << invokedName + ": " << msg << "\n  errno says:" << strerror( errno ) << "\n  at: " << __FILE__ << ' '  \
              << __LINE__ << '\n'

/** \defgroup xrif2fits xrif2fits: xrif-archive to FITS cube converter
 * \brief Read images from an xrif archive and write to FITS
 *
 * <a href="../handbook/utils/xrif2fits.html">Utility Documentation</a>
 *
 * \ingroup utils
 *
 */

/** \defgroup xrif2fits_files xrif2fits Files
 * \ingroup xrif2fits
 */

bool g_timeToDie = false;

void sigTermHandler( int signum, siginfo_t *siginf, void *ucont )
{
    // Suppress those warnings . . .
    static_cast<void>( signum );
    static_cast<void>( siginf );
    static_cast<void>( ucont );

    std::cerr << "\n"; // clear out the ^C char

    g_timeToDie = true;
}

/// A utility to stream MagaO-X images from xrif compressed archives to an ImageStreamIO stream.
/**
 * \todo finish md doc for xrif2fits
 *
 * \ingroup xrif2fits
 */
class xrif2fits : public mx::app::application
{
    typedef XWC_DEFAULT_VERBOSITY verboseT;

    typedef MagAOX::file::stdFileName<verboseT> stdFileNameT;

  protected:
    /** \name Configurable Parameters
     * @{
     */
    std::string m_camera; /**< The INDI device name of the camera to process.
                               Sets m_cameraHeader to `<m_camera>_header.conf` */

    std::string m_cameraHeader; /**< The filename of the config file containing the camera header specification.
                                     Setting this overrides the setting from m_camera.
                                     The path specified by $MagAOX_PATH/$MagAOX_CONFIG is searched,
                                     unless XRIF2FITS_CONFIGPATH is set in the environment.*/

    bool m_noHeader{ false }; /**< if true then no camera header is generated */

    std::string m_dir; /**< The directory to search for files.  Can be empty if full path given in files.
                            If files is empty, all archives in dir will be used.  Defaults to `./`.*/

    bool m_overWriteDir{ false }; ///< Overwrite an existing directory.  Default is to stop if directory exists.

    std::vector<std::string> m_files; /**< List of files to use.  If dir is not empty,
                                           it will be pre-pended to each name.*/

    std::vector<MagAOX::file::stdFileName<verboseT>> m_fileNames; /**< The decoded file names broken down into
                                                              constituent parts */

    std::vector<std::string> m_logDir;

    std::vector<std::string> m_telDir;

    std::string m_outDir = "fits/";

    bool m_noMeta{ false };

    bool m_metaOnly{ false };

    bool m_timesOnly{ false };

    bool m_cubeMode{ false };

    bool m_strict{ false };

    bool m_showDetails{ false };

    bool m_quiet{ false };

    double m_maxMetadataGap{ 25.0 };

    bool m_strictAbort{ false };

    size_t m_recoverableErrors{ 0 };

    logMap<verboseT> m_logs;

    logMap<verboseT> m_tels;

    /// Warning keys already printed, used to avoid repeating metadata-availability notices per frame.
    std::set<std::string> m_warnedMetadata;

  protected:
    ///@}

    std::string MagAOXPath;
    std::string ConfigRelPath;

    std::vector<logMeta> m_logMetas;

    xrif_t m_xrif{ nullptr };
    xrif_t m_xrif_timing{ nullptr };

  public:
    /// Construct an xrif archive to FITS converter.
    /** Sets up the default config paths by reading from the environment
     *
     */
    xrif2fits();

    /// Destroy allocated xrif decoder handles.
    ~xrif2fits();

    /// Configure command-line and file configuration options.
    virtual void setupConfig();

    /// Load command-line and file configuration results.
    virtual void loadConfig();

    /// Read a camera FITS header metadata configuration file.
    virtual mx::error_t readHeaderConfig( const std::string &hcfile /**< [in] header configuration file path */ );

    /// Execute the conversion.
    virtual int execute();

    /// Prepare the file list and output directory
    /** Based on loaded configuration
     *
     * \returns mx::error_t::noerror on success
     * \returns error code on an error
     *
     */
    mx::error_t prepareFiles();

    /// Confirm a configured metadata source and app subdirectory exist.
    mx::error_t validateMetaSourceDir( const std::string &dir,   /**< [in] configured metadata source directory */
                                       const std::string &app,   /**< [in] app/device subdirectory name */
                                       const std::string &source /**< [in] user-facing source label */
    );

    /// Load metadata file maps from configured source directories.
    mx::error_t loadMetaFileMaps( logMap<verboseT>               &logMap, /**< [in,out] metadata map to populate */
                                  const std::vector<std::string> &dirs,   /**< [in] metadata source directories */
                                  const std::string              &app,    /**< [in] app/device name to load */
                                  const std::string              &ext,    /**< [in] metadata file extension */
                                  const stdFileNameT &firstFile,          /**< [in] first xrif file needing coverage */
                                  const stdFileNameT &lastFile,           /**< [in] last xrif file needing coverage */
                                  const std::string  &source              /**< [in] user-facing source label */
    );

    /// Check whether telemetry files are available for an app.
    bool hasTelemetry( const std::string &app /**< [in] app/device name */ ) const;

    /// Count and report a recoverable metadata error.
    void recoverableError( const std::string &key, /**< [in] stable error key */
                           const std::string &msg  /**< [in] user-facing error message */
    );

    /// Count recoverable metadata and log errors seen so far.
    size_t recoverableErrorCount() const;

    /// Fail in strict mode if any recoverable errors have been seen.
    bool strictOkay( const std::string &context /**< [in] operation about to proceed */ );

    /// Report a writeImages failure unless strict mode already reported the real cause.
    void reportWriteImagesFailure( const std::string &inputFile /**< [in] xrif archive being processed */ ) const;

    /// Compute an exposure interval from camera telemetry.
    bool exposureTime( timespec          &stime,   /**< [out] exposure start time */
                       double            &exptime, /**< [out] exposure duration in seconds */
                       const std::string &app,     /**< [in] camera app/device name */
                       const timespec    &atime    /**< [in] acquisition/end time */
    );

    /// Write FITS header and text metadata for one configured metadata item.
    bool appendMetadata( mx::fits::fitsHeader<verboseT> &fh,        /**< [in,out] FITS header being built */
                         std::ofstream                  &metaOut,   /**< [in,out] metadata text stream */
                         logMeta                        &meta,      /**< [in,out] metadata item with lookup hints */
                         bool                            writeMeta, /**< [in] true to write metadata text */
                         bool                       canLookup, /**< [in] true if interval metadata lookup is valid */
                         const flatlogs::timespecX &stime,     /**< [in] exposure start time */
                         const flatlogs::timespecX &atime      /**< [in] acquisition/end time */
    );

    /// Write decoded images from one archive to FITS files.
    template <typename dataT>
    int writeImages( int           n,  /**< [in] index of the xrif file being written */
                     stdFileNameT &lfn /**< [in] parsed standard file name */
    );

    /// Format a nanosecond count as a fixed-width 9-digit string.
    std::string format_nano( uint64_t n /**< [in] nanoseconds */ );
};

inline xrif2fits::xrif2fits()
{
    m_logs.m_reportPrefix = "xrif2fits: ";
    m_tels.m_reportPrefix = "xrif2fits: ";

    // setup the default config path
    MagAOXPath = mx::sys::getEnv( MAGAOX_env_path );

    if( MagAOXPath == "" )
    {
        MagAOXPath = MAGAOX_path;
    }

    if( MagAOXPath.size() > 0 )
    {
        if( MagAOXPath.back() != '/' )
        {
            MagAOXPath += '/';
        }
    }

    ConfigRelPath = mx::sys::getEnv( MAGAOX_env_config );

    if( ConfigRelPath == "" )
    {
        ConfigRelPath = MAGAOX_configRelPath;
    }

    if( ConfigRelPath.size() > 0 )
    {
        if( ConfigRelPath.back() != '/' )
        {
            ConfigRelPath += '/';
        }

        mx::app::application::m_configPathCLBase = MagAOXPath + ConfigRelPath + '/';
    }

    // Allow overriding the config path
    mx::app::application::m_configPathCLBase_env = "XRIF2FITS_CONFIGPATH";
}

inline xrif2fits::~xrif2fits()
{
    if( m_xrif )
    {
        xrif_delete( m_xrif );
    }

    if( m_xrif_timing )
    {
        xrif_delete( m_xrif_timing );
    }
}

inline void xrif2fits::setupConfig()
{
    config.add( "camera",
                "",
                "camera",
                argType::Required,
                "",
                "camera",
                false,
                "string",
                "The device name of the camera.  Sets the header.camera config to <camera>_header.conf" );

    config.add( "header.camera",
                "",
                "header.camera",
                argType::Required,
                "header",
                "camera",
                false,
                "string",
                "The name of a config file defining a camera header.  Overrides the default for `camera`."
                "Searches $MagAOX_PATH/$MagAOX_CONFIG, unless XRIF2FITS_CONFIGPATH is set in the environment." );

    config.add( "noHeader",
                "N",
                "noHeader",
                argType::True,
                "",
                "noHeader",
                false,
                "bool",
                "If true, then no camera header is generated" );

    config.add( "dir",
                "d",
                "dir",
                argType::Required,
                "",
                "dir",
                false,
                "string",
                "The directory to search for files. Can be empty if full path given in files." );

    config.add( "overwrite",
                "O",
                "overwrite",
                argType::True,
                "",
                "overwrite",
                false,
                "bool",
                "Overwrite an existing directory.  Default is to stop if directory exists." );

    config.add( "files",
                "f",
                "files",
                argType::Required,
                "",
                "files",
                false,
                "vector<string>",
                "List of files to use. If dir is not empty, it will be pre-pended to each name." );

    config.add( "logdir",
                "l",
                "logdir",
                argType::Required,
                "",
                "logdir",
                false,
                "vector<string>",
                "Directories for log files." );

    config.add( "teldir",
                "t",
                "teldir",
                argType::Required,
                "",
                "teldir",
                false,
                "vector<string>",
                "Directories for telemetry files." );

    config.add( "outDir",
                "D",
                "outDir",
                argType::Required,
                "",
                "outDir",
                false,
                "string",
                "The directory in which to write output files.  Default is ./fits/." );

    config.add( "metaOnly",
                "",
                "metaOnly",
                argType::True,
                "",
                "metaOnly",
                false,
                "bool",
                "If true, output only meta data, without decoding images.  Default is false." );

    config.add( "time",
                "T",
                "time",
                argType::True,
                "",
                "time",
                false,
                "bool",
                "time span mode: output one line per input file in the format [filename] [start time] [end time] "
                "[number of frames], with ISO 8601 timestamps" );

    config.add( "noMeta",
                "",
                "noMeta",
                argType::True,
                "",
                "noMeta",
                false,
                "bool",
                "If true, the meta data file is not written (FITS headers will still be).  Default is false." );

    config.add( "cubeMode",
                "C",
                "cubeMode",
                argType::True,
                "",
                "cubeMode",
                false,
                "bool",
                "If true, the archive is written as a FITS cube with minimal header.  Default is false." );

    config.add( "strict",
                "",
                "strict",
                argType::True,
                "",
                "strict",
                false,
                "bool",
                "If true, recoverable metadata/log errors stop processing before FITS files are written.  Default is "
                "false." );

    config.add( "showDetails",
                "",
                "show-details",
                argType::True,
                "",
                "showDetails",
                false,
                "bool",
                "If true, print xrif compression details while processing.  Default is false." );

    config.add( "quiet",
                "",
                "quiet",
                argType::True,
                "",
                "quiet",
                false,
                "bool",
                "If true, suppress non-error status output.  Default is false." );

    config.add( "maxMetadataGap",
                "",
                "maxMetadataGap",
                argType::Required,
                "",
                "maxMetadataGap",
                false,
                "double",
                "Maximum permitted time gap in seconds for metadata telemetry coverage.  Set negative to disable.  "
                "Default is 25 seconds." );
}

inline void xrif2fits::loadConfig()
{
    config( m_camera, "camera" );

    if( m_camera != "" )
    {
        m_cameraHeader = m_camera + "_header.conf";
    }

    config( m_cameraHeader, "header.camera" );

    config( m_noHeader, "noHeader" );

    config( m_dir, "dir" );
    config( m_overWriteDir, "overwrite" );
    config( m_files, "files" );
    config( m_outDir, "outDir" );
    config( m_logDir, "logdir" );
    config( m_telDir, "teldir" );
    config( m_metaOnly, "metaOnly" );
    config( m_timesOnly, "time" );
    config( m_noMeta, "noMeta" );
    config( m_cubeMode, "cubeMode" );
    config( m_strict, "strict" );
    config( m_showDetails, "showDetails" );
    config( m_quiet, "quiet" );
    config( m_maxMetadataGap, "maxMetadataGap" );

    if( m_configPathCLBase.size() > 0 )
    {
        if( mx::app::application::m_configPathCLBase.back() != '/' )
        {
            mx::app::application::m_configPathCLBase += '/';
        }
    }
}

inline mx::error_t xrif2fits::readHeaderConfig( const std::string &hcfile )
{
    if( hcfile == "" )
    {
        return mx::error_t::noerror;
    }

    mx::app::appConfigurator hconfig;

    hconfig.add( "include", "", "include", argType::Required, "", "include", false, "string", "" );

    try
    {
        XRIF2FITS_DEBUG_CRUMB( "reading: " + hcfile );

        if( hconfig.readConfig( hcfile, true ) != 0 )
        {
            return mx::error_report<verboseT>( mx::error_t::error, "Error reading header config: " + hcfile );
        }
    }
    catch( const std::exception &e )
    {
        return mx::error_report<verboseT>( mx::error_t::std_exception,
                                           "Exception reading header config: " + hcfile + ". " + e.what() );
    }

    std::vector<std::string> includes;
    hconfig( includes, "include" );

    for( auto &include : includes )
    {
        if( include.size() > 4 )
        {
            if( include.substr( include.size() - 5, 4 ) != ".conf" )
            {
                include += ".conf";
            }
        }

        XRIF2FITS_DEBUG_CRUMB( "reading include: " + mx::app::application::m_configPathCLBase + include );

        mx_error_check( readHeaderConfig( mx::app::application::m_configPathCLBase + include ) );
    }

    std::vector<std::string> devices;

    hconfig.unusedSections( devices );

    if( devices.size() == 0 && includes.size() == 0 ) // this allows include-only
    {
        return mx::error_report<verboseT>( mx::error_t::notfound, "No device sections in header config:" + hcfile );
    }

    for( auto &device : devices )
    {
        // Wind through all the unused targets
        for( auto it = hconfig.m_unusedConfigs.begin(); it != hconfig.m_unusedConfigs.end(); ++it )
        {
            if( device == it->second.section )
            {
                std::string eventCode = it->second.keyword;

                // Check if this keyword is a valid flatlogs eventCode
                flatlogs::eventCodeT ec = MagAOX::logger::eventCode( eventCode );
                if( ec != eventCodes::UNKNOWN )
                {
                    std::vector<std::string> fields;
                    hconfig.configUnused( fields, mx::app::iniFile::makeKey( device, eventCode ) );

                    for( auto &field : fields )
                    {
                        m_logMetas.push_back( logMetaSpec( { device, ec, field } ) );
                    }
                }
            }
        }
    }

    return mx::error_t::noerror;
}

inline mx::error_t
xrif2fits::validateMetaSourceDir( const std::string &dir, const std::string &app, const std::string &source )
{
    mx::error_t errc;
    bool        isdir = mx::ioutils::dir_exists_is( dir, errc );

    if( !!errc )
    {
        return mx::error_report<verboseT>( errc, "checking " + source + " directory: " + dir );
    }

    if( !isdir )
    {
        return mx::error_report<verboseT>( mx::error_t::dirnotfound, source + " directory does not exist: " + dir );
    }

    std::string appDir = dir;
    if( appDir.size() > 0 && appDir.back() != '/' )
    {
        appDir += '/';
    }
    appDir += app;

    isdir = mx::ioutils::dir_exists_is( appDir, errc );

    if( !!errc )
    {
        return mx::error_report<verboseT>( errc, "checking " + source + " app directory: " + appDir );
    }

    if( !isdir )
    {
        return mx::error_report<verboseT>( mx::error_t::dirnotfound,
                                           source + " app directory does not exist: " + appDir );
    }

    return mx::error_t::noerror;
}

inline mx::error_t xrif2fits::loadMetaFileMaps( logMap<verboseT>               &logMap,
                                                const std::vector<std::string> &dirs,
                                                const std::string              &app,
                                                const std::string              &ext,
                                                const stdFileNameT             &firstFile,
                                                const stdFileNameT             &lastFile,
                                                const std::string              &source )
{
    if( dirs.size() == 0 )
    {
        recoverableError( source + ":" + app + ":not-configured",
                          "No " + source + " directories configured for " + app +
                              "; metadata from this source will be marked " + logMeta::unavailableValue() + "." );
        return mx::error_t::noerror;
    }

    bool        foundAppDir = false;
    mx::error_t errc;
    bool        hasExplicitDir = false;

    for( size_t n = 0; n < dirs.size(); ++n )
    {
        if( dirs[n] != "" && dirs[n] != "." && dirs[n] != "./" )
        {
            hasExplicitDir = true;
        }
    }

    for( size_t n = 0; n < dirs.size(); ++n )
    {
        if( dirs[n] == "" || ( hasExplicitDir && ( dirs[n] == "." || dirs[n] == "./" ) ) )
        {
            continue;
        }

        bool isdir = mx::ioutils::dir_exists_is( dirs[n], errc );
        if( !!errc )
        {
            return mx::error_report<verboseT>( errc, "checking " + source + " directory: " + dirs[n] );
        }

        if( !isdir )
        {
            return mx::error_report<verboseT>( mx::error_t::dirnotfound,
                                               source + " directory does not exist: " + dirs[n] );
        }

        std::string appDir = dirs[n];
        if( appDir.size() > 0 && appDir.back() != '/' )
        {
            appDir += '/';
        }
        appDir += app;

        isdir = mx::ioutils::dir_exists_is( appDir, errc );
        if( !!errc )
        {
            return mx::error_report<verboseT>( errc, "checking " + source + " app directory: " + appDir );
        }

        if( !isdir )
        {
            continue;
        }

        foundAppDir = true;
        mx_error_check( logMap.loadAppToFileMap( dirs[n], app, ext, firstFile, lastFile ) );
        XRIF2FITS_DEBUG_CRUMB( "metadata map source loaded: source=" + source + " app=" + app + " dir=" + dirs[n] +
                               " mappedFiles=" + std::to_string( logMap.m_appToFileMap[app].size() ) );
    }

    if( !foundAppDir )
    {
        return mx::error_report<verboseT>(
            mx::error_t::dirnotfound, source + " app directory does not exist in any configured source for " + app );
    }

    if( logMap.m_appToFileMap[app].size() == 0 )
    {
        recoverableError( source + ":" + app + ":no-files",
                          "No " + source + " files found in the requested time range for " + app +
                              "; metadata from this source will be marked " + logMeta::unavailableValue() + "." );
    }

    return mx::error_t::noerror;
}

inline bool xrif2fits::hasTelemetry( const std::string &app ) const
{
    logMap<verboseT>::appToFileMapT::const_iterator it = m_tels.m_appToFileMap.find( app );

    return it != m_tels.m_appToFileMap.end() && it->second.size() > 0;
}

inline void xrif2fits::recoverableError( const std::string &key, const std::string &msg )
{
    if( m_warnedMetadata.insert( key ).second )
    {
        ++m_recoverableErrors;
        std::cerr << invokedName << ": " << msg << "\n";
    }
}

inline size_t xrif2fits::recoverableErrorCount() const
{
    return m_recoverableErrors + m_logs.recoverableErrors() + m_tels.recoverableErrors();
}

inline bool xrif2fits::strictOkay( const std::string &context )
{
    if( !m_strict )
    {
        return true;
    }

    size_t nErrors = recoverableErrorCount();
    if( nErrors == 0 )
    {
        return true;
    }

    m_strictAbort = true;
    std::cerr << invokedName << ": strict mode aborting before " << context << " after " << nErrors
              << " recoverable error(s).\n";
    return false;
}

inline void xrif2fits::reportWriteImagesFailure( const std::string &inputFile ) const
{
    if( !m_strictAbort )
    {
        ERR_INVOKED_NAME( "error writing to file: " + inputFile );
    }
}

inline bool xrif2fits::exposureTime( timespec &stime, double &exptime, const std::string &app, const timespec &atime )
{
    if( !hasTelemetry( app ) )
    {
        recoverableError( "exptime:" + app + ":no-telemetry",
                          "No telemetry files are available for " + app +
                              "; exposure-dependent metadata will be marked " + logMeta::unavailableValue() + "." );
        return false;
    }

    char *prior = nullptr;
    if( m_tels.getPriorLog( prior, app, eventCodes::TELEM_STDCAM, atime ) != 0 || prior == nullptr )
    {
        recoverableError( "exptime:" + app + ":no-prior",
                          "No prior exposure-time telemetry is available for " + app +
                              "; exposure-dependent metadata will be marked " + logMeta::unavailableValue() + "." );
        return false;
    }

    exptime = telem_stdcam::exptime( logHeader::messageBuffer( prior ) );
    if( !logMetaGapValid( logHeader::timespec( prior ), atime, m_maxMetadataGap ) )
    {
        recoverableError( "exptime:" + app + ":prior-gap",
                          "Prior exposure-time telemetry for " + app +
                              " is outside the maximum metadata gap; exposure-dependent metadata will be marked " +
                              logMeta::unavailableValue() + "." );
        return false;
    }

    stime = atime - exptime;

    XRIF2FITS_DEBUG_CRUMB( "exposureTime app=" + app + " atime=" + std::to_string( atime.tv_sec ) + "." +
                           std::to_string( atime.tv_nsec ) + " exptime=" + std::to_string( exptime ) +
                           " stime=" + std::to_string( stime.tv_sec ) + "." + std::to_string( stime.tv_nsec ) );

    char *priorprior = nullptr;
    if( m_tels.getPriorLog( priorprior, app, eventCodes::TELEM_STDCAM, stime ) == 0 && priorprior != nullptr )
    {
        if( !logMetaGapValid( logHeader::timespec( priorprior ), stime, m_maxMetadataGap ) )
        {
            recoverableError( "exptime:" + app + ":start-prior-gap",
                              "Exposure-start telemetry for " + app +
                                  " is outside the maximum metadata gap; exposure-dependent metadata will be marked " +
                                  logMeta::unavailableValue() + "." );
            return false;
        }

        /// \todo this needs to check for any log entries between end and start
        if( telem_stdcam::exptime( logHeader::messageBuffer( priorprior ) ) != exptime )
        {
            std::cerr << invokedName << ": Change in exposure time mid-exposure\n";
        }
    }
    else
    {
        recoverableError( "exptime:" + app + ":no-start-prior",
                          "No exposure-time telemetry was found before the exposure start for " + app + "." );
    }

    return true;
}

inline bool xrif2fits::appendMetadata( mx::fits::fitsHeader<verboseT> &fh,
                                       std::ofstream                  &metaOut,
                                       logMeta                        &meta,
                                       bool                            writeMeta,
                                       bool                            canLookup,
                                       const flatlogs::timespecX      &stime,
                                       const flatlogs::timespecX      &atime )
{
    XRIF2FITS_DEBUG_CRUMB( "metadata begin: " + meta.device() + " " + meta.keyword() );

    if( canLookup && hasTelemetry( meta.device() ) )
    {
        XRIF2FITS_DEBUG_CRUMB( "metadata build card: " + meta.device() + " " + meta.keyword() );
        std::string value = meta.value( m_tels, stime, atime, m_maxMetadataGap );
        if( value == logMeta::unavailableValue() )
        {
            std::string reason = meta.unavailableReason();
            if( reason.size() > 0 )
            {
                reason = " " + reason;
            }

            recoverableError( "metadata:" + meta.device() + ":" + meta.keyword() + ":unavailable",
                              "Metadata " + meta.device() + " " + meta.keyword() + " is " +
                                  logMeta::unavailableValue() + reason + "." );
            fh.append( meta.unavailableCard() );
            if( writeMeta )
            {
                metaOut << " " << logMeta::unavailableValue();
            }

            XRIF2FITS_DEBUG_CRUMB( "metadata end: " + meta.device() + " " + meta.keyword() );
            return strictOkay( "FITS header metadata: " + meta.device() + " " + meta.keyword() );
        }

        mx::fits::fitsHeaderCard<verboseT> fc = meta.card( m_tels, stime, atime, m_maxMetadataGap );
        XRIF2FITS_DEBUG_CRUMB( "metadata append card: " + meta.device() + " " + meta.keyword() );
        fh.append( fc );
        if( writeMeta )
        {
            XRIF2FITS_DEBUG_CRUMB( "metadata write text: " + meta.device() + " " + meta.keyword() );
            metaOut << " " << value;
        }
    }
    else
    {
        XRIF2FITS_DEBUG_CRUMB( "metadata unavailable: " + meta.device() + " " + meta.keyword() );
        recoverableError( "metadata:" + meta.device() + ":" + meta.keyword() + ":no-lookup",
                          "Metadata " + meta.device() + " " + meta.keyword() + " is " + logMeta::unavailableValue() +
                              "." );
        fh.append( meta.unavailableCard() );
        if( writeMeta )
        {
            metaOut << " " << logMeta::unavailableValue();
        }

        XRIF2FITS_DEBUG_CRUMB( "metadata end: " + meta.device() + " " + meta.keyword() );
        return strictOkay( "FITS header metadata: " + meta.device() + " " + meta.keyword() );
    }

    XRIF2FITS_DEBUG_CRUMB( "metadata end: " + meta.device() + " " + meta.keyword() );
    return true;
}

inline int xrif2fits::execute()
{
    XRIF2FITS_DEBUG_CRUMB( "execute begin" );

    // Install signal handling
    struct sigaction act;
    sigset_t         set;

    act.sa_sigaction = sigTermHandler;
    act.sa_flags     = SA_SIGINFO;
    sigemptyset( &set );
    act.sa_mask = set;

    errno = 0;
    if( sigaction( SIGTERM, &act, 0 ) < 0 )
    {
        std::cerr << invokedName << ": error setting SIGTERM handler: " << strerror( errno ) << "\n";
        return -1;
    }

    errno = 0;
    if( sigaction( SIGQUIT, &act, 0 ) < 0 )
    {
        std::cerr << invokedName << ": error setting SIGQUIT handler: " << strerror( errno ) << "\n";
        return -1;
    }

    errno = 0;
    if( sigaction( SIGINT, &act, 0 ) < 0 )
    {
        std::cerr << invokedName << ": error setting SIGINT handler: " << strerror( errno ) << "\n";
        return -1;
    }

    try
    {
        XRIF2FITS_DEBUG_CRUMB( "prepareFiles begin" );
        mx::error_t errc = prepareFiles();
        if( !!errc )
        {
            mx::error_report<verboseT>( errc, "error from prepareFiles" );
            return -1;
        }
        XRIF2FITS_DEBUG_CRUMB( "prepareFiles end" );
    }
    catch( ... )
    {
        std::throw_with_nested( MagAOX::xwcException( "error from prepareFiles" ) );
    }

    // this has to be here
    stdFileNameT &firstFile = m_fileNames[0];
    stdFileNameT &lastFile  = m_fileNames.back();

    xrif_error_t rv;
    XRIF2FITS_DEBUG_CRUMB( "xrif_new image begin" );
    rv = xrif_new( &m_xrif );

    if( rv < 0 )
    {
        std::cerr << invokedName << ": Error allocating xrif.\n";
        return -1;
    }
    XRIF2FITS_DEBUG_CRUMB( "xrif_new image end" );

    XRIF2FITS_DEBUG_CRUMB( "xrif_new timing begin" );
    rv = xrif_new( &m_xrif_timing );

    if( rv < 0 )
    {
        std::cerr << invokedName << ": Error allocating xrif_timing.\n";
        return -1;
    }
    XRIF2FITS_DEBUG_CRUMB( "xrif_new timing end" );

    if( !m_noHeader )
    {
        XRIF2FITS_DEBUG_CRUMB( "metadata map load begin" );
        m_logMetas.push_back( logMetaSpec( { firstFile.appName(), telem_stdcam::eventCode, "exptime" } ) );

        // Build list of apps, this will be automagic as part of config
        std::set<std::string> logApps;

        logApps.insert( m_fileNames[0].appName() );

        for( auto &meta : m_logMetas )
        {
            logApps.insert( meta.device() );
        }

        for( auto &app : logApps )
        {
            mx::error_t errc = loadMetaFileMaps( m_logs, m_logDir, app, ".binlog", firstFile, lastFile, "log" );
            if( !!errc )
            {
                mx::error_report<verboseT>( errc, "error loading log file map for " + app );
                return -1;
            }

            errc = loadMetaFileMaps( m_tels, m_telDir, app, ".bintel", firstFile, lastFile, "telemetry" );
            if( !!errc )
            {
                mx::error_report<verboseT>( errc, "error loading telemetry file map for " + app );
                return -1;
            }
        }
        XRIF2FITS_DEBUG_CRUMB( "metadata map load end" );
        if( !strictOkay( "archive decoding" ) )
        {
            return -1;
        }
    }

    // Now de-compress and load the frames
    // Only decompressing the number of files needed, and only copying the number of frames needed
    for( size_t n = 0; n < m_files.size(); ++n )
    {

        if( g_timeToDie == true )
            break; // check before going on

        if( !m_noHeader )
        {
            if( hasTelemetry( m_fileNames[n].appName() ) )
            {
                XRIF2FITS_DEBUG_CRUMB( "archive telemetry load request app=" + m_fileNames[n].appName() +
                                       " archiveTs=" + std::to_string( m_fileNames[n].timestamp().time_s ) + "." +
                                       std::to_string( m_fileNames[n].timestamp().time_ns ) + " file=" + m_files[n] );
                m_tels.loadFiles( m_fileNames[n].appName(), m_fileNames[n].timestamp() );
                if( !strictOkay( "archive decoding" ) )
                {
                    return -1;
                }
            }
        }
        if( !m_timesOnly && !m_quiet )
        {
            std::cout << "xrif2fits: decoding for " << m_fileNames[n].appName() << " (" + m_files[n] << ")\n";
        }

        XRIF2FITS_DEBUG_CRUMB( "archive begin: " + m_files[n] );

        FILE *fp_xrif = fopen( m_files[n].c_str(), "rb" );
        if( fp_xrif == nullptr )
        {
            std::cerr << invokedName << ": Error opening " << m_files[n] << "\n";
            std::cerr << invokedName << ": " << strerror( errno ) << "\n";
            return -1;
        }

        char header[XRIF_HEADER_SIZE];

        size_t nr = fread( header, 1, XRIF_HEADER_SIZE, fp_xrif );
        if( nr != XRIF_HEADER_SIZE )
        {
            std::cerr << invokedName << ": Error reading header of " << m_files[n] << "\n";
            fclose( fp_xrif );
            return -1;
        }

        uint32_t header_size;
        xrif_read_header( m_xrif, &header_size, header );
        if( m_showDetails && !m_quiet && !m_timesOnly )
        {
            std::cout << "xrif compression details:\n";
            std::cout << "  difference method:  " << xrif_difference_method_string( m_xrif->difference_method ) << '\n';
            std::cout << "  reorder method:     " << xrif_reorder_method_string( m_xrif->reorder_method ) << '\n';
            std::cout << "  compression method: " << xrif_compress_method_string( m_xrif->compress_method ) << '\n';
            if( m_xrif->compress_method == XRIF_COMPRESS_LZ4 )
            {
                std::cout << "    LZ4 acceleration: " << m_xrif->lz4_acceleration << '\n';
            }
            std::cout << "  dimensions:         " << m_xrif->width << " x " << m_xrif->height << " x " << m_xrif->depth
                      << " x " << m_xrif->frames << "\n";
            std::cout << "  raw size:           "
                      << m_xrif->width * m_xrif->height * m_xrif->depth * m_xrif->frames * m_xrif->data_size
                      << " bytes\n";
            std::cout << "  encoded size:       " << m_xrif->compressed_size << " bytes\n";
            std::cout << "  ratio:              "
                      << ( (double)m_xrif->compressed_size ) /
                             ( m_xrif->width * m_xrif->height * m_xrif->depth * m_xrif->frames * m_xrif->data_size )
                      << '\n';
        }
        rv = xrif_allocate_raw( m_xrif );
        if( rv != XRIF_NOERROR )
        {
            std::cerr << invokedName << ": Error allocating raw buffer for " << m_files[n] << "\n";
            std::cerr << "\t code: " << rv << "\n";
            return -1;
        }

        rv = xrif_allocate_reordered( m_xrif );
        if( rv != XRIF_NOERROR )
        {
            std::cerr << invokedName << ": Error allocating reordered buffer for " << m_files[n] << "\n";
            std::cerr << "\t code: " << rv << "\n";
            return -1;
        }

        nr = fread( m_xrif->raw_buffer, 1, m_xrif->compressed_size, fp_xrif );

        if( nr != m_xrif->compressed_size )
        {
            std::cerr << invokedName << ": Error reading data from " << m_files[n] << "\n";
            return -1;
        }

        // Now get timing data
        nr = fread( header, 1, XRIF_HEADER_SIZE, fp_xrif );
        if( nr != XRIF_HEADER_SIZE )
        {
            std::cerr << invokedName << ": Error reading timing header of " << m_files[n] << "\n";
            fclose( fp_xrif );
            return -1;
        }

        xrif_read_header( m_xrif_timing, &header_size, header );

        if( m_showDetails && !m_quiet && !m_timesOnly )
        {
            std::cout << "xrif timing data compression details:\n";
            std::cout << "  difference method:  " << xrif_difference_method_string( m_xrif_timing->difference_method )
                      << '\n';
            std::cout << "  reorder method:     " << xrif_reorder_method_string( m_xrif_timing->reorder_method )
                      << '\n';
            std::cout << "  compression method: " << xrif_compress_method_string( m_xrif_timing->compress_method )
                      << '\n';
            if( m_xrif_timing->compress_method == XRIF_COMPRESS_LZ4 )
            {
                std::cout << "    LZ4 acceleration: " << m_xrif_timing->lz4_acceleration << '\n';
            }
            std::cout << "  dimensions:         " << m_xrif_timing->width << " x " << m_xrif_timing->height << " x "
                      << m_xrif_timing->depth << " x " << m_xrif_timing->frames << "\n";
            std::cout << "  raw size:           "
                      << m_xrif_timing->width * m_xrif_timing->height * m_xrif_timing->depth * m_xrif_timing->frames *
                             m_xrif_timing->data_size
                      << " bytes\n";
            std::cout << "  encoded size:       " << m_xrif_timing->compressed_size << " bytes\n";
            std::cout << "  ratio:              "
                      << ( (double)m_xrif_timing->compressed_size ) /
                             ( m_xrif_timing->width * m_xrif_timing->height * m_xrif_timing->depth *
                               m_xrif_timing->frames * m_xrif_timing->data_size )
                      << '\n';
        }
        rv = xrif_allocate_raw( m_xrif_timing );
        if( rv != XRIF_NOERROR )
        {
            std::cerr << invokedName << ": Error allocating raw buffer for timing data from " << m_files[n] << "\n";
            std::cerr << "\t code: " << rv << "\n";
            return -1;
        }

        rv = xrif_allocate_reordered( m_xrif_timing );
        if( rv != XRIF_NOERROR )
        {
            std::cerr << invokedName << ": Error allocating reordered buffer for  timing data from " << m_files[n]
                      << "\n";
            std::cerr << "\t code: " << rv << "\n";
            return -1;
        }

        nr = fread( m_xrif_timing->raw_buffer, 1, m_xrif_timing->compressed_size, fp_xrif );

        if( nr != m_xrif_timing->compressed_size )
        {
            std::cerr << invokedName << ": Error reading timing data from " << m_files[n] << "\n";
            return -1;
        }

        fclose( fp_xrif );

        if( g_timeToDie == true )
            break; // check after the long read.

        if( !m_metaOnly )
        {
            XRIF2FITS_DEBUG_CRUMB( "xrif_decode image begin: " + m_files[n] );
            rv = xrif_decode( m_xrif );
            if( rv != XRIF_NOERROR )
            {
                std::cerr << invokedName << ": Error decoding image data from " << m_files[n] << "\n";
                std::cerr << "\t code: " << rv << "\n";
                return -1;
            }
            XRIF2FITS_DEBUG_CRUMB( "xrif_decode image end: " + m_files[n] );
        }

        XRIF2FITS_DEBUG_CRUMB( "xrif_decode timing begin: " + m_files[n] );
        rv = xrif_decode( m_xrif_timing );
        if( rv != XRIF_NOERROR )
        {
            std::cerr << invokedName << ": Error decoding timing data from " << m_files[n] << "\n";
            std::cerr << "\t code: " << rv << "\n";
            return -1;
        }
        XRIF2FITS_DEBUG_CRUMB( "xrif_decode timing end: " + m_files[n] );

        if( g_timeToDie == true )
        {
            break; // check after the decompress.
        }

        if( m_timesOnly )
        {
            std::cout << m_files[n] << " ";
            double totalExposureTime      = 0;
            bool   totalExposureTimeValid = true;

            for( xrif_dimension_t q = 0; q < m_xrif->frames; ++q )
            {
                timespec atime;            // This is the acquisition time of the exposure
                timespec stime = { 0, 0 }; // This is the start time of the exposure, calculated as atime-exptime.

                uint64_t *curr_timing = (uint64_t *)m_xrif_timing->raw_buffer + 5 * q;

                atime.tv_sec  = curr_timing[1];
                atime.tv_nsec = curr_timing[2];

                double exptime = -1;
                if( exposureTime( stime, exptime, m_fileNames[n].appName(), atime ) )
                {
                    if( totalExposureTimeValid )
                    {
                        totalExposureTime += exptime;
                    }
                }
                else
                {
                    totalExposureTimeValid = false;
                }

                std::string timestamp;
                mx::sys::timeStamp( timestamp, atime );

                std::string dateobs = mx::sys::ISO8601DateTimeStr( atime, 1 );
                if( q == 0 )
                {
                    std::cout << dateobs << " ";
                }
                if( q == ( m_xrif->frames - 1 ) )
                {
                    std::cout << dateobs << " ";
                    if( totalExposureTimeValid )
                    {
                        std::cout << totalExposureTime;
                    }
                    else
                    {
                        std::cout << logMeta::unavailableValue();
                    }
                    std::cout << " " << m_xrif->frames << "\n";
                }
            }
        }
        else // Normal writing
        {
            if( m_xrif->type_code == XRIF_TYPECODE_UINT8 )
            {
                XRIF2FITS_DEBUG_CRUMB( "writeImages<uint8_t> begin: " + m_files[n] );
                if( writeImages<uint8_t>( n, m_fileNames[n] ) < 0 )
                {
                    reportWriteImagesFailure( m_files[n] );
                    return -1;
                }
                XRIF2FITS_DEBUG_CRUMB( "writeImages<uint8_t> end: " + m_files[n] );
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_INT8 )
            {
                XRIF2FITS_DEBUG_CRUMB( "writeImages<int8_t> begin: " + m_files[n] );
                if( writeImages<int8_t>( n, m_fileNames[n] ) < 0 )
                {
                    reportWriteImagesFailure( m_files[n] );
                    return -1;
                }
                XRIF2FITS_DEBUG_CRUMB( "writeImages<int8_t> end: " + m_files[n] );
            }
            if( m_xrif->type_code == XRIF_TYPECODE_UINT16 )
            {
                XRIF2FITS_DEBUG_CRUMB( "writeImages<uint16_t> begin: " + m_files[n] );
                if( writeImages<uint16_t>( n, m_fileNames[n] ) < 0 )
                {
                    reportWriteImagesFailure( m_files[n] );
                    return -1;
                }
                XRIF2FITS_DEBUG_CRUMB( "writeImages<uint16_t> end: " + m_files[n] );
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_INT16 )
            {
                XRIF2FITS_DEBUG_CRUMB( "writeImages<int16_t> begin: " + m_files[n] );
                if( writeImages<int16_t>( n, m_fileNames[n] ) < 0 )
                {
                    reportWriteImagesFailure( m_files[n] );
                    return -1;
                }
                XRIF2FITS_DEBUG_CRUMB( "writeImages<int16_t> end: " + m_files[n] );
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_UINT32 )
            {
                XRIF2FITS_DEBUG_CRUMB( "writeImages<uint32_t> begin: " + m_files[n] );
                if( writeImages<uint32_t>( n, m_fileNames[n] ) < 0 )
                {
                    reportWriteImagesFailure( m_files[n] );
                    return -1;
                }
                XRIF2FITS_DEBUG_CRUMB( "writeImages<uint32_t> end: " + m_files[n] );
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_INT32 )
            {
                XRIF2FITS_DEBUG_CRUMB( "writeImages<int32_t> begin: " + m_files[n] );
                if( writeImages<int32_t>( n, m_fileNames[n] ) < 0 )
                {
                    reportWriteImagesFailure( m_files[n] );
                    return -1;
                }
                XRIF2FITS_DEBUG_CRUMB( "writeImages<int32_t> end: " + m_files[n] );
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_UINT64 )
            {
                XRIF2FITS_DEBUG_CRUMB( "writeImages<uint64_t-as-uint32_t> begin: " + m_files[n] );
                if( writeImages<uint32_t>( n, m_fileNames[n] ) < 0 )
                {
                    reportWriteImagesFailure( m_files[n] );
                    return -1;
                }
                XRIF2FITS_DEBUG_CRUMB( "writeImages<uint64_t-as-uint32_t> end: " + m_files[n] );
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_INT64 )
            {
                XRIF2FITS_DEBUG_CRUMB( "writeImages<int64_t-as-int32_t> begin: " + m_files[n] );
                if( writeImages<int32_t>( n, m_fileNames[n] ) < 0 )
                {
                    reportWriteImagesFailure( m_files[n] );
                    return -1;
                }
                XRIF2FITS_DEBUG_CRUMB( "writeImages<int64_t-as-int32_t> end: " + m_files[n] );
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_FLOAT )
            {
                XRIF2FITS_DEBUG_CRUMB( "writeImages<float> begin: " + m_files[n] );
                if( writeImages<float>( n, m_fileNames[n] ) < 0 )
                {
                    reportWriteImagesFailure( m_files[n] );
                    return -1;
                }
                XRIF2FITS_DEBUG_CRUMB( "writeImages<float> end: " + m_files[n] );
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_DOUBLE )
            {
                XRIF2FITS_DEBUG_CRUMB( "writeImages<double-as-float> begin: " + m_files[n] );
                if( writeImages<float>( n, m_fileNames[n] ) < 0 )
                {
                    reportWriteImagesFailure( m_files[n] );
                    return -1;
                }
                XRIF2FITS_DEBUG_CRUMB( "writeImages<double-as-float> end: " + m_files[n] );
            }
            else
            {
                ERR_INVOKED_NAME( "unsupported data type in file: " + m_files[n] );
                return -1;
            }
        }
    }

    size_t nRecoverableErrors = recoverableErrorCount();
    if( nRecoverableErrors > 0 )
    {
        std::cerr << invokedName << ": exited after completing with " << nRecoverableErrors
                  << " recoverable error(s).\n";
        return 1;
    }

    if( !m_quiet )
    {
        std::cerr << invokedName << ": exited normally.\n";
    }
    return 0;
}

inline mx::error_t xrif2fits::prepareFiles()
{
    // If files aren't specified, we search the given directory.
    if( m_files.size() == 0 )
    {
        if( m_dir == "" ) // search pwd
        {
            m_dir = "./";
        }

        mx_error_check( mx::ioutils::getFileNames( m_files, m_dir, "", "", ".xrif" ) );

        for( size_t n = 0; n < m_files.size(); ++n )
        {
            MagAOX::file::stdFileName sfn;
            try
            {
                mx_error_check( sfn.fullName( m_files[n] ) );
            }
            catch( ... )
            {
                std::throw_with_nested( mx::exception( "From stdFileName for " + m_files[n] ) );
            }

            // add only if it passed
            try
            {
                m_fileNames.push_back( sfn );
            }
            catch( const std::bad_alloc &e )
            {
                std::throw_with_nested(
                    mx::exception( mx::error_t::std_bad_alloc, "error adding file " + m_files[n] ) );
            }
            catch( ... )
            {
                std::throw_with_nested( mx::exception( "error adding file " + m_files[n] ) );
            }
        }
    }
    else // If files are specified we attach a directory to them if needed
    {
        if( m_dir != "" )
        {
            try
            {
                if( m_dir[m_dir.size() - 1] != '/' )
                    m_dir += '/';

                for( size_t n = 0; n < m_files.size(); ++n )
                {
                    m_files[n] = m_dir + m_files[n];
                }
            }
            catch( const std::bad_alloc &e )
            {
                std::throw_with_nested( mx::exception( mx::error_t::std_bad_alloc, "adding dir to files" ) );
            }
            catch( ... )
            {
                std::throw_with_nested( mx::exception( "adding dir to files" ) );
            }
        }

        for( size_t n = 0; n < m_files.size(); ++n )
        {
            // since the files were specified they have to pass
            try
            {
                m_fileNames.push_back( MagAOX::file::stdFileName( m_files[n] ) );
            }
            catch( const std::bad_alloc &e )
            {
                std::throw_with_nested(
                    mx::exception( mx::error_t::std_bad_alloc, "error adding file " + m_files[n] ) );
            }
            catch( ... )
            {
                std::throw_with_nested( mx::exception( "error adding file " + m_files[n] ) );
            }
        }
    }

    if( m_files.size() == 0 )
    {
        return mx::error_report<verboseT>( mx::error_t::notfound, "No xrif files found" );
    }

    if( m_outDir == "" )
    {
        m_outDir = "./";
    }
    else
    {
        // Make sure the slash exists, then mkdir.  We know size is > 0 here.
        if( m_outDir[m_outDir.size() - 1] != '/' )
        {
            m_outDir += '/';
        }

        if( !m_timesOnly )
        {
            if( !m_overWriteDir )
            {
                mx::error_t errc;
                if( mx::ioutils::dir_exists_is( m_outDir, errc ) )
                {
                    return mx::error_report<verboseT>( mx::error_t::eexist,
                                                       "Directory " + m_outDir + " already exists." );
                }

                if( !!errc )
                {
                    return mx::error_report<verboseT>( errc, "Checking " + m_outDir );
                }
            }

            mx_error_check( mx::ioutils::createDirectories( m_outDir ) );
        }
    }

    for( size_t n = 1; n < m_files.size(); ++n )
    {
        if( m_fileNames.back().appName() != m_fileNames[0].appName() )
        {
            return mx::error_report<verboseT>( mx::error_t::invalidarg,
                                               "can only operate on a single camera at a time" );
        }
    }

    if( m_camera == "" )
    {
        m_camera = m_fileNames[0].appName();
    }

    if( m_cameraHeader == "" )
    {
        m_cameraHeader = m_camera + "_header.conf";
    }

    if( !m_noHeader )
    {
        mx::error_t errc = readHeaderConfig( mx::app::application::m_configPathCLBase + m_cameraHeader );

        if( !!errc )
        {
            return mx::error_report<verboseT>(
                errc, "Error reading camera header: " + mx::app::application::m_configPathCLBase + m_cameraHeader );
        }
    }

    return mx::error_t::noerror;
}

template <typename dataT>
int xrif2fits::writeImages( int n, stdFileNameT &lfn )
{
    mx::improc::eigenCube<dataT> tmpc(
        reinterpret_cast<dataT *>( m_xrif->raw_buffer ), m_xrif->width, m_xrif->height, m_xrif->frames );

    mx::fits::fitsFile<dataT, verboseT> ff;
    mx::fits::fitsHeader<verboseT>      fh;

    // Special handling for meta output
    logMeta exptimeMeta( logMetaSpec( lfn.appName(), telem_stdcam::eventCode, "exptime" ) );

    if( m_strict && !m_cubeMode && !m_noHeader )
    {
        mx::fits::fitsHeader<verboseT> preflightHeader;
        std::ofstream                  preflightMetaOut;
        for( int q = 0; q < tmpc.planes(); ++q )
        {
            timespec atime;
            timespec stime = { 0, 0 };

            uint64_t *curr_timing = (uint64_t *)m_xrif_timing->raw_buffer + 5 * q;
            atime.tv_sec          = curr_timing[1];
            atime.tv_nsec         = curr_timing[2];

            double exptime          = -1;
            bool   haveExposureTime = exposureTime( stime, exptime, lfn.appName(), atime );
            if( !haveExposureTime )
            {
                recoverableError( "metadata:" + lfn.appName() + ":EXPTIME:unavailable",
                                  "Metadata " + lfn.appName() + " EXPTIME is " + logMeta::unavailableValue() + "." );
                strictOkay( "FITS header metadata: " + lfn.appName() + " EXPTIME" );
                return -1;
            }

            preflightHeader.clear();
            for( size_t u = 0; u < m_logMetas.size(); ++u )
            {
                if( !appendMetadata( preflightHeader, preflightMetaOut, m_logMetas[u], false, true, stime, atime ) )
                {
                    return -1;
                }
            }

            if( !strictOkay( "FITS write preflight" ) )
            {
                return -1;
            }
        }
    }

    std::ofstream metaOut;

    // Print the meta-file header
    if( !m_noMeta && !m_timesOnly )
    {
        metaOut.open( m_outDir + "meta_data.txt" );
        /*metaOut << "#DATE-OBS FRAMENO ACQSEC ACQNSEC WRTSEC WRTNSEC";
        metaOut << " EXPTIME";
        for(size_t u=0;u<logMetas.size();++u)
        {
           metaOut << " " << logMetas[u].keyword() ;
        }
        metaOut << "\n";*/
    }
    if( m_cubeMode )
    {
        std::string outfname = m_outDir + mx::ioutils::pathStem( m_files[n] ) + ".fits";
        if( !strictOkay( "FITS write: " + outfname ) )
        {
            return -1;
        }

        ff.write( outfname, tmpc );
    }
    else
    {
        for( int q = 0; q < tmpc.planes(); ++q )
        {
            XRIF2FITS_DEBUG_CRUMB( "frame begin: " + std::to_string( q ) );

            uint64_t cnt0;
            timespec atime; // This is the acquisition time of the exposure
            timespec wtime;
            timespec stime = { 0, 0 }; // This is the start time of the exposure, calculated as atime-exptime.

            uint64_t *curr_timing = (uint64_t *)m_xrif_timing->raw_buffer + 5 * q;

            cnt0          = curr_timing[0];
            atime.tv_sec  = curr_timing[1];
            atime.tv_nsec = curr_timing[2];
            wtime.tv_sec  = curr_timing[3];
            wtime.tv_nsec = curr_timing[4];

            XRIF2FITS_DEBUG_CRUMB( "frame timing q=" + std::to_string( q ) + " cnt=" + std::to_string( cnt0 ) +
                                   " archiveTs=" + std::to_string( lfn.timestamp().time_s ) + "." +
                                   std::to_string( lfn.timestamp().time_ns ) +
                                   " atime=" + std::to_string( atime.tv_sec ) + "." + std::to_string( atime.tv_nsec ) +
                                   " wtime=" + std::to_string( wtime.tv_sec ) + "." + std::to_string( wtime.tv_nsec ) );

            double exptime          = -1;
            bool   haveExposureTime = false;
            if( !m_noHeader )
            {
                haveExposureTime = exposureTime( stime, exptime, lfn.appName(), atime );
            }

            // timespecX midexp = mx::meanTimespec( atime, stime);

            std::string timestamp;
            mx::sys::timeStamp( timestamp, atime );
            std::string outfname = m_outDir + lfn.appName() + "_" + timestamp + ".fits";

            XRIF2FITS_DEBUG_CRUMB( "header clear: " + outfname );
            fh.clear();

            std::string dateobs = mx::sys::ISO8601DateTimeStr( atime, 1 );

            XRIF2FITS_DEBUG_CRUMB( "header append standard cards: " + outfname );
            fh.append( "DATE-OBS", dateobs, "Date of obs. YYYY-mm-ddTHH:MM:SS" );
            fh.append( "INSTRUME", "MagAO-X " + lfn.appName() );
            fh.append( "CAMERA", lfn.appName() );
            fh.append( "TELESCOP", "Magellan Clay, Las Campanas Obs." );

            if( !m_noMeta )
            {
                metaOut << dateobs << " " << cnt0 << " " << atime.tv_sec << " " << format_nano( atime.tv_nsec ) << " "
                        << wtime.tv_sec << " " << format_nano( wtime.tv_nsec ) << " ";
            }

            if( !m_noHeader )
            {
                // First output exposure time
                if( !m_noMeta )
                {
                    if( haveExposureTime )
                    {
                        metaOut << exptimeMeta.value( m_tels, stime, atime, m_maxMetadataGap );
                    }
                    else
                    {
                        recoverableError( "metadata:" + lfn.appName() + ":EXPTIME:unavailable",
                                          "Metadata " + lfn.appName() + " EXPTIME is " + logMeta::unavailableValue() +
                                              "." );
                        metaOut << logMeta::unavailableValue();
                        if( !strictOkay( "FITS header metadata: " + lfn.appName() + " EXPTIME" ) )
                        {
                            return -1;
                        }
                    }
                }

                // Then output each value in turn
                for( size_t u = 0; u < m_logMetas.size(); ++u )
                {
                    XRIF2FITS_DEBUG_CRUMB( "metadata index: " + std::to_string( u ) + " of " +
                                           std::to_string( m_logMetas.size() ) );
                    if( !appendMetadata( fh, metaOut, m_logMetas[u], !m_noMeta, haveExposureTime, stime, atime ) )
                    {
                        return -1;
                    }
                }
            }

            XRIF2FITS_DEBUG_CRUMB( "header append timing cards: " + outfname );
            fh.append( "FRAMENO", cnt0 );
            fh.append( "ACQSEC", atime.tv_sec, "Image acq. time, seconds since Unix epoch" );
            fh.append( "ACQNSEC", atime.tv_nsec, "Image acq. time, nanosecond component" );
            fh.append( "WRTSEC", wtime.tv_sec, "Image write time, seconds since Unix epoch" );
            fh.append( "WRTNSEC", wtime.tv_nsec, "Image write time, nanosecond component" );

            if( !m_noMeta )
            {
                metaOut << "\n";
            }
            if( !m_metaOnly )
            {
                if( !strictOkay( "FITS write: " + outfname ) )
                {
                    return -1;
                }

                XRIF2FITS_DEBUG_CRUMB( "fits write begin: " + outfname );
                mx::improc::eigenImage<dataT> im = tmpc.image( q );
                ff.write( outfname, tmpc.image( q ), fh );
                XRIF2FITS_DEBUG_CRUMB( "fits write end: " + outfname );
            }

            XRIF2FITS_DEBUG_CRUMB( "frame end: " + std::to_string( q ) );
        }
    }

    return 0;
}

inline std::string xrif2fits::format_nano( uint64_t n )
{
    std::ostringstream oss;
    oss << std::setw( 9 ) << std::setfill( '0' ) << n;
    return oss.str();
};

#endif // xrif2fits_hpp
