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

#include <mx/ioutils/fileUtils.hpp>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include <mx/ioutils/fits/fitsFile.hpp>

#include <mx/sys/timeUtils.hpp>
using namespace mx::sys::tscomp;
using namespace mx::sys::tsop;

#include "../../libMagAOX/libMagAOX.hpp"

#ifndef DEBUG_CRUMB
#define DEBUG_CRUMB(msg) {std::cerr << msg << '(' << __FILE__ << ' ' << __LINE__ << "\n";}
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

    bool m_overWriteDir {false}; ///< Overwrite an existing directory.  Default is to stop if directory exists.

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

    logMap<verboseT> m_logs;

    logMap<verboseT> m_tels;

  protected:
    ///@}

    std::string MagAOXPath;
    std::string ConfigRelPath;

    std::vector<logMeta> m_logMetas;

    xrif_t m_xrif{ nullptr };
    xrif_t m_xrif_timing{ nullptr };

  public:
    /// c-tor
    /** Sets up the default config paths by reading from the environment
     *
     */
    xrif2fits();

    ~xrif2fits();

    virtual void setupConfig();

    virtual void loadConfig();

    virtual mx::error_t readHeaderConfig( const std::string &hcfile );

    virtual int execute();

    /// Prepare the file list and output directory
    /** Based on loaded configuration
     *
     * \returns mx::error_t::noerror on success
     * \returns error code on an error
     *
     */
    mx::error_t prepareFiles();

    template <typename dataT>
    int writeImages( int n, stdFileNameT &lfn );

    std::string format_nano( uint64_t n );
};

inline xrif2fits::xrif2fits()
{
    // setup the default config path
    MagAOXPath    = mx::sys::getEnv( MAGAOX_env_path );

    if(MagAOXPath == "")
    {
        MagAOXPath = MAGAOX_path;
    }

    if(MagAOXPath.size() > 0)
    {
        if(MagAOXPath.back() !='/')
        {
            MagAOXPath += '/';
        }
    }

    ConfigRelPath = mx::sys::getEnv( MAGAOX_env_config );

    if(ConfigRelPath == "")
    {
        ConfigRelPath = MAGAOX_configRelPath;
    }

    if( ConfigRelPath.size() > 0 )
    {
        if(ConfigRelPath.back() !='/')
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
    config(m_overWriteDir, "overwrite");
    config( m_files, "files" );
    config( m_outDir, "outDir" );
    config( m_logDir, "logdir" );
    config( m_telDir, "teldir" );
    config( m_metaOnly, "metaOnly" );
    config( m_timesOnly, "time" );
    config( m_noMeta, "noMeta" );
    config( m_cubeMode, "cubeMode" );

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
        DEBUG_CRUMB("reading: " + hcfile);

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

        DEBUG_CRUMB("reading include: " + mx::app::application::m_configPathCLBase + include);

        mx_error_check( readHeaderConfig( mx::app::application::m_configPathCLBase + include ) );
    }

    std::vector<std::string> devices;

    hconfig.unusedSections( devices );

    if( devices.size() == 0 && includes.size() == 0) //this allows include-only
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
                        std::cerr << "adding: " << device << ' ' << ec << ' ' << field << '\n';
                        m_logMetas.push_back( logMetaSpec( { device, ec, field } ) );
                    }
                }
            }
        }
    }

    return mx::error_t::noerror;
}

inline int xrif2fits::execute()
{
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
        std::cerr << " (" << invokedName << "): error setting SIGTERM handler: " << strerror( errno ) << "\n";
        return -1;
    }

    errno = 0;
    if( sigaction( SIGQUIT, &act, 0 ) < 0 )
    {
        std::cerr << " (" << invokedName << "): error setting SIGQUIT handler: " << strerror( errno ) << "\n";
        return -1;
    }

    errno = 0;
    if( sigaction( SIGINT, &act, 0 ) < 0 )
    {
        std::cerr << " (" << invokedName << "): error setting SIGINT handler: " << strerror( errno ) << "\n";
        return -1;
    }

    try
    {
        mx::error_t errc = prepareFiles();
        if( !!errc )
        {
            mx::error_report<verboseT>( errc, "error from prepareFiles" );
            return -1;
        }
    }
    catch( ... )
    {
        std::throw_with_nested( MagAOX::xwcException( "error from prepareFiles" ) );
    }

    // this has to be here
    stdFileNameT &firstFile = m_fileNames[0];
    stdFileNameT &lastFile  = m_fileNames.back();

    xrif_error_t rv;
    rv = xrif_new( &m_xrif );

    if( rv < 0 )
    {
        std::cerr << " (" << invokedName << "): Error allocating xrif.\n";
        return -1;
    }

    rv = xrif_new( &m_xrif_timing );

    if( rv < 0 )
    {
        std::cerr << " (" << invokedName << "): Error allocating xrif_timing.\n";
        return -1;
    }

    if( !m_noHeader )
    {
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
            for( size_t n = 0; n < m_logDir.size(); ++n )
            {
                try
                {
                    m_logs.loadAppToFileMap( m_logDir[n], app, ".binlog", firstFile, lastFile );
                }
                catch( ... )
                {
                    /// for now ignore all exceptions. \todo eventually ignore only "no prior logs" etc
                }
            }

            for( size_t n = 0; n < m_telDir.size(); ++n )
            {
                try
                {
                    m_tels.loadAppToFileMap( m_telDir[n], app, ".bintel", firstFile, lastFile );
                }
                catch( ... )
                {
                    /// for now ignore all exceptions. \todo eventually ignore only "no prior logs" etc
                }
            }

            if( m_logs.m_appToFileMap[app].size() == 0 )
            {
                throw MagAOX::xwcException( "no logs found for " + app );
            }

            if( m_tels.m_appToFileMap[app].size() == 0 )
            {
                throw MagAOX::xwcException( "no telems found for " + app );
            }
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
            m_tels.loadFiles( m_fileNames[n].appName(), m_fileNames[n].timestamp() );
        }
        if( !m_timesOnly )
        {

            std::cout << "******************************************************\n";
            std::cout << "* xrif2fits: decoding for " << m_fileNames[n].appName() << " (" + m_files[n] << ")\n";
            std::cout << "******************************************************\n";
        }

        FILE *fp_xrif = fopen( m_files[n].c_str(), "rb" );
        if( fp_xrif == nullptr )
        {
            std::cerr << " (" << invokedName << "): Error opening " << m_files[n] << "\n";
            std::cerr << " (" << invokedName << "): " << strerror( errno ) << "\n";
            return -1;
        }

        char header[XRIF_HEADER_SIZE];

        size_t nr = fread( header, 1, XRIF_HEADER_SIZE, fp_xrif );
        if( nr != XRIF_HEADER_SIZE )
        {
            std::cerr << " (" << invokedName << "): Error reading header of " << m_files[n] << "\n";
            fclose( fp_xrif );
            return -1;
        }

        uint32_t header_size;
        xrif_read_header( m_xrif, &header_size, header );
        if( !m_timesOnly )
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
            std::cerr << " (" << invokedName << "): Error allocating raw buffer for " << m_files[n] << "\n";
            std::cerr << "\t code: " << rv << "\n";
            return -1;
        }

        rv = xrif_allocate_reordered( m_xrif );
        if( rv != XRIF_NOERROR )
        {
            std::cerr << " (" << invokedName << "): Error allocating reordered buffer for " << m_files[n] << "\n";
            std::cerr << "\t code: " << rv << "\n";
            return -1;
        }

        nr = fread( m_xrif->raw_buffer, 1, m_xrif->compressed_size, fp_xrif );

        if( nr != m_xrif->compressed_size )
        {
            std::cerr << " (" << invokedName << "): Error reading data from " << m_files[n] << "\n";
            return -1;
        }

        // Now get timing data
        nr = fread( header, 1, XRIF_HEADER_SIZE, fp_xrif );
        if( nr != XRIF_HEADER_SIZE )
        {
            std::cerr << " (" << invokedName << "): Error reading timing header of " << m_files[n] << "\n";
            fclose( fp_xrif );
            return -1;
        }

        xrif_read_header( m_xrif_timing, &header_size, header );

        if( !m_timesOnly )
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
            std::cerr << " (" << invokedName << "): Error allocating raw buffer for timing data from " << m_files[n]
                      << "\n";
            std::cerr << "\t code: " << rv << "\n";
            return -1;
        }

        rv = xrif_allocate_reordered( m_xrif_timing );
        if( rv != XRIF_NOERROR )
        {
            std::cerr << " (" << invokedName << "): Error allocating reordered buffer for  timing data from "
                      << m_files[n] << "\n";
            std::cerr << "\t code: " << rv << "\n";
            return -1;
        }

        nr = fread( m_xrif_timing->raw_buffer, 1, m_xrif_timing->compressed_size, fp_xrif );

        if( nr != m_xrif_timing->compressed_size )
        {
            std::cerr << " (" << invokedName << "): Error reading timing data from " << m_files[n] << "\n";
            return -1;
        }

        fclose( fp_xrif );

        if( g_timeToDie == true )
            break; // check after the long read.

        if( !m_metaOnly )
        {
            rv = xrif_decode( m_xrif );
            if( rv != XRIF_NOERROR )
            {
                std::cerr << " (" << invokedName << "): Error decoding image data from " << m_files[n] << "\n";
                std::cerr << "\t code: " << rv << "\n";
                return -1;
            }
        }

        rv = xrif_decode( m_xrif_timing );
        if( rv != XRIF_NOERROR )
        {
            std::cerr << " (" << invokedName << "): Error decoding timing data from " << m_files[n] << "\n";
            std::cerr << "\t code: " << rv << "\n";
            return -1;
        }

        if( g_timeToDie == true )
        {
            break; // check after the decompress.
        }

        if( m_timesOnly )
        {
            std::cout << m_files[n] << " ";
            double totalExposureTime = 0;

            for( xrif_dimension_t q = 0; q < m_xrif->frames; ++q )
            {
                timespec atime;            // This is the acquisition time of the exposure
                timespec stime = { 0, 0 }; // This is the start time of the exposure, calculated as atime-exptime.

                uint64_t *curr_timing = (uint64_t *)m_xrif_timing->raw_buffer + 5 * q;

                atime.tv_sec  = curr_timing[1];
                atime.tv_nsec = curr_timing[2];

                // We have to bootstrap the exposure time
                char *prior = nullptr;
                m_tels.getPriorLog( prior, m_fileNames[n].appName(), eventCodes::TELEM_STDCAM, atime );
                double exptime = -1;
                if( prior )
                {
                    char *priorprior = nullptr;
                    exptime          = telem_stdcam::exptime( logHeader::messageBuffer( prior ) );
                    stime            = atime - exptime;
                    m_tels.getPriorLog( priorprior, m_fileNames[n].appName(), eventCodes::TELEM_STDCAM, stime );

                    ///\todo this needs to check for any log entries between end and start
                    if( telem_stdcam::exptime( logHeader::messageBuffer( priorprior ) ) != exptime )
                    {
                        std::cerr << "Change in exposure time mid-exposure\n";
                    }
                }
                else
                {
                    std::cerr << "no prior\n";
                }
                totalExposureTime += exptime;

                std::string timestamp;
                mx::sys::timeStamp( timestamp, atime );

                std::string dateobs = mx::sys::ISO8601DateTimeStr( atime, 1 );
                if( q == 0 )
                {
                    std::cout << dateobs << " ";
                }
                if( q == ( m_xrif->frames - 1 ) )
                {
                    std::cout << dateobs << " " << totalExposureTime << " " << m_xrif->frames << "\n";
                }
            }
        }
        else // Normal writing
        {
            if( m_xrif->type_code == XRIF_TYPECODE_UINT8 )
            {
                if( writeImages<uint8_t>( n, m_fileNames[n] ) < 0 )
                {
                    ERR_INVOKED_NAME( "error writing to file: " + m_files[n] );
                    return -1;
                }
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_INT8 )
            {
                if( writeImages<int8_t>( n, m_fileNames[n] ) < 0 )
                {
                    ERR_INVOKED_NAME( "error writing to file: " + m_files[n] );
                    return -1;
                }
            }
            if( m_xrif->type_code == XRIF_TYPECODE_UINT16 )
            {
                if( writeImages<uint16_t>( n, m_fileNames[n] ) < 0 )
                {
                    ERR_INVOKED_NAME( "error writing to file: " + m_files[n] );
                    return -1;
                }
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_INT16 )
            {
                if( writeImages<int16_t>( n, m_fileNames[n] ) < 0 )
                {
                    ERR_INVOKED_NAME( "error writing to file: " + m_files[n] );
                    return -1;
                }
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_UINT32 )
            {
                if( writeImages<uint32_t>( n, m_fileNames[n] ) < 0 )
                {
                    ERR_INVOKED_NAME( "error writing to file: " + m_files[n] );
                    return -1;
                }
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_INT32 )
            {
                if( writeImages<int32_t>( n, m_fileNames[n] ) < 0 )
                {
                    ERR_INVOKED_NAME( "error writing to file: " + m_files[n] );
                    return -1;
                }
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_UINT64 )
            {
                if( writeImages<uint32_t>( n, m_fileNames[n] ) < 0 )
                {
                    ERR_INVOKED_NAME( "error writing to file: " + m_files[n] );
                    return -1;
                }
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_INT64 )
            {
                if( writeImages<int32_t>( n, m_fileNames[n] ) < 0 )
                {
                    ERR_INVOKED_NAME( "error writing to file: " + m_files[n] );
                    return -1;
                }
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_FLOAT )
            {
                if( writeImages<float>( n, m_fileNames[n] ) < 0 )
                {
                    ERR_INVOKED_NAME( "error writing to file: " + m_files[n] );
                    return -1;
                }
            }
            else if( m_xrif->type_code == XRIF_TYPECODE_DOUBLE )
            {
                if( writeImages<float>( n, m_fileNames[n] ) < 0 )
                {
                    ERR_INVOKED_NAME( "error writing to file: " + m_files[n] );
                    return -1;
                }
            }
            else
            {
                ERR_INVOKED_NAME( "unsupported data type in file: " + m_files[n] );
                return -1;
            }
        }
    }

    std::cerr << " (" << invokedName << "): exited normally.\n";

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
                std::throw_with_nested( mx::exception("adding dir to files") );
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
            if(!m_overWriteDir)
            {
                mx::error_t errc;
                if(mx::ioutils::dir_exists_is(m_outDir, errc))
                {
                    return mx::error_report<verboseT>(mx::error_t::eexist, "Directory " + m_outDir + " already exists.");
                }

                if(!!errc)
                {
                    return mx::error_report<verboseT>(errc, "Checking " + m_outDir);
                }
            }

            mx_error_check( mx::ioutils::createDirectories(m_outDir) );
        }
    }

    for( size_t n = 1; n < m_files.size(); ++n )
    {
        if( m_fileNames.back().appName() != m_fileNames[0].appName() )
        {
            return mx::error_report<verboseT>(mx::error_t::invalidarg, "can only operate on a single camera at a time" );
        }
    }

    if( m_camera == "" )
    {
        m_camera = m_fileNames[0].appName();
        std::cerr << "Set camera to: " << m_camera << '\n';
    }

    if( m_cameraHeader == "" )
    {
        m_cameraHeader = m_camera + "_header.conf";
        std::cerr << "Set camera header to: " << m_cameraHeader << '\n';
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
        ff.write( outfname, tmpc );
    }
    else
    {
        for( int q = 0; q < tmpc.planes(); ++q )
        {
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

            double exptime = -1;
            if( !m_noHeader )
            {
                // We have to bootstrap the exposure time
                char *prior = nullptr;
                m_tels.getPriorLog( prior, lfn.appName(), eventCodes::TELEM_STDCAM, atime );

                if( prior )
                {
                    char *priorprior = nullptr;
                    exptime          = telem_stdcam::exptime( logHeader::messageBuffer( prior ) );

                    stime = atime - exptime;
                    m_tels.getPriorLog( priorprior, lfn.appName(), eventCodes::TELEM_STDCAM, stime );

                    ///\todo this needs to check for any log entries between end and start
                    if( telem_stdcam::exptime( logHeader::messageBuffer( priorprior ) ) != exptime )
                    {
                        std::cerr << "Change in exposure time mid-exposure\n";
                    }
                }
                else
                {
                    std::cerr << "no prior\n";
                }
            }

            // timespecX midexp = mx::meanTimespec( atime, stime);

            std::string timestamp;
            mx::sys::timeStamp( timestamp, atime );
            std::string outfname = m_outDir + lfn.appName() + "_" + timestamp + ".fits";

            fh.clear();

            std::string dateobs = mx::sys::ISO8601DateTimeStr( atime, 1 );

            fh.append( "DATE-OBS", dateobs, "Date of obs. YYYY-mm-ddTHH:MM:SS" );
            fh.append( "INSTRUME", "MagAO-X " + lfn.appName() );
            fh.append( "CAMERA", lfn.appName() );
            fh.append( "TELESCOP", "Magellan Clay, Las Campanas Obs." );

            if( !m_noMeta )
            {
                metaOut << dateobs << " " << cnt0 << " " << atime.tv_sec << " " << format_nano( atime.tv_nsec ) << " "
                        << wtime.tv_sec << " " << format_nano( wtime.tv_nsec ) << " ";
            }

            if( exptime > -1 )
            {
                // First output exposure time
                if( !m_noMeta )
                {
                    metaOut << exptimeMeta.value( m_tels, stime, atime );
                }

                // Then output each value in turn
                for( size_t u = 0; u < m_logMetas.size(); ++u )
                {
                    mx::fits::fitsHeaderCard<verboseT> fc = m_logMetas[u].card( m_tels, stime, atime );
                    fh.append( fc );
                    if( !m_noMeta )
                    {
                        metaOut << " " << m_logMetas[u].value( m_tels, stime, atime );
                    }
                }
            }

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
                mx::improc::eigenImage<dataT> im = tmpc.image( q );
                ff.write( outfname, tmpc.image( q ), fh );
            }
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
