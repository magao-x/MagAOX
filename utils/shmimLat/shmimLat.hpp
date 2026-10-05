/** \file shmimLat.hpp
 * \brief The shmimLat class declaration and definition.
 *
 * \ingroup xrif2hmim_files
 */

#ifndef shmimLat_hpp
#define shmimLat_hpp

#include <ImageStreamIO/ImageStruct.h>
#include <ImageStreamIO/ImageStreamIO.h>

#include <xrif/xrif.h>

#include <mx/ioutils/fileUtils.hpp>
#include <mx/improc/eigenCube.hpp>

#include <mx/sys/timeUtils.hpp>
#include <mx/math/plot/gnuPlot.hpp>

#include "../../libMagAOX/libMagAOX.hpp"

/** \defgroup shmimLat shmimLat: plot data from a shmim in gnuplot
 * \brief Monitor a shmim and update a gnuplot plot
 *
 * <a href="../handbook/utils/shmimLat.html">Utility Documentation</a>
 *
 * \ingroup utils
 *
 */

/** \defgroup shmimLat_files shmimLat Files
 * \ingroup shmimLat
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
 * \todo finish md doc for shmimLat
 *
 * \ingroup shmimLat
 */
class shmimLat : public mx::app::application
{
  protected:
    /** \name Configurable Parameters
     * @{
     */


    ///@}

    /// Structure to manage the image threads, including startup.
    struct s_imageThread
    {
        shmimLat *m_sp; ///< a pointer to a shmimLat instance (normally this)

        std::thread *m_thread{ nullptr }; ///< Thread for receiving image slice updates.  A pointer to allow copying,
                                          ///< but must be deleted in d'tor of parent.

        std::string m_shmimName; ///< the name of the image to subscribe from this thread

        std::vector<std::vector<float>> m_y;

        /// C'tor to create the thread object
        s_imageThread()
        {
            m_thread = new std::thread;
        }
    };

    std::vector<s_imageThread> m_imageThreads; ///< The image threads, one per shared memory stream.

    std::vector<timespec> m_atimes;
    std::vector<timespec> m_arrtimes;
    ///@}

  public:
    ~shmimLat();

    virtual void setupConfig();

    virtual void loadConfig();

    virtual int execute();


  private:
    /// Thread starter, called by imageThreadStart on thread construction.  Calls imageThreadExec.
    static void internal_imageThreadStart( s_imageThread *mit /**< [in] a pointer to an s_imageThread structure */ );

  public:
    /// Start the image thread.
    int imageThreadStart( size_t thno /**< [in] the thread to start */ );

    /// Execute the image thread.
    void imageThreadExec( s_imageThread *mit /**< [in] a pointer to an s_imageThread structure */ );
};

inline shmimLat::~shmimLat()
{
}

inline void shmimLat::setupConfig()
{
    config.add( "shmimName",
                "n",
                "shmimName",
                argType::Required,
                "",
                "shmimName",
                false,
                "vector<string>",
                "The names of the shared memory buffer to stream to.  Default is \"shmimLat\"" );

    config.add( "circBuffLength",
                "L",
                "circBuffLength",
                argType::Required,
                "",
                "circBuffLength",
                false,
                "int",
                "The length of the shared memory circular buffer. Default is 1." );


}

inline void shmimLat::loadConfig()
{
    std::vector<std::string> shmimNames;
    config( shmimNames, "shmimName" );

    if( shmimNames.size() == 0 )
    {
        std::cerr << "shmim names not specified with -n\n";
        doHelp = true;
        return;
    }

    for( auto &s : shmimNames )
    {
        s_imageThread nt;
        nt.m_sp        = this;
        nt.m_shmimName = s;
        m_imageThreads.push_back( nt );
    }


}

inline int shmimLat::execute()
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


    m_atimes.resize(120000);
    m_arrtimes.resize(m_atimes.size());

    imageThreadExec(&m_imageThreads[0]);

    std::ofstream fout("times");

    fout.precision(15);

    long atime0 = m_atimes[0].tv_sec;

    for(size_t n = 0; n < m_atimes.size(); ++n)
    {
        double at = m_atimes[n].tv_sec - atime0 + m_atimes[n].tv_nsec/1e9;
        double arrt = m_arrtimes[n].tv_sec - atime0 + m_arrtimes[n].tv_nsec/1e9;

        fout << at << ' ' << arrt << ' ' << arrt-at << '\n';
    }
    fout.close();
    return 0;
}

inline void shmimLat::internal_imageThreadStart( s_imageThread *mit )
{
    mit->m_sp->imageThreadExec( mit );
}

inline int shmimLat::imageThreadStart( size_t thno )
{
    try
    {
        *m_imageThreads[thno].m_thread = std::thread( internal_imageThreadStart, &m_imageThreads[thno] );
    }
    catch( const std::exception &e )
    {
        std::cerr << std::string( "exception in image thread startup: " ) + e.what() << ' ' << __FILE__ << ' '
                  << __LINE__ << '\n';
        return -1;
    }
    catch( ... )
    {
        std::cerr << "unknown exception in image thread startup: " << __FILE__ << ' ' << __LINE__ << '\n';
        return -1;
    }

    if( !m_imageThreads[thno].m_thread->joinable() )
    {
        std::cerr << "image thread did not start: " << __FILE__ << ' ' << __LINE__ << '\n';
        return -1;
    }

    return 0;
}

inline void shmimLat::imageThreadExec( s_imageThread *mit )
{
    bool  opened = false;
    bool  restart;
    ino_t inode           = 0; ///< The inode of the image stream file
    int   semaphoreNumber = 9; ///< The image structure semaphore index.

    IMAGE    imageStream;

    mx::improc::eigenImage<float> im;

    size_t n = 0;

    while( !g_timeToDie )
    {
        /* Initialize ImageStreamIO
         */
        opened  = false;
        restart = false; // Set this up front, since we're about to restart.

        int logged = 0;
        while( !opened && !g_timeToDie && !restart )
        {
            // b/c ImageStreamIO prints every single time, and latest version don't support stopping it yet, and that
            // isn't thread-safe-able anyway we do our own checks.  This is the same code in ImageStreamIO_openIm...
            int  SM_fd;
            char SM_fname[200];
            ImageStreamIO_filename( SM_fname, sizeof( SM_fname ), mit->m_shmimName.c_str() );
            SM_fd = open( SM_fname, O_RDWR );
            if( SM_fd == -1 )
            {
                if( !logged )
                {
                    std::cerr << "ImageStream " + mit->m_shmimName + " not found (yet).  Retrying . . .\n";
                }
                logged = 1;
                sleep( 1 ); // be patient
                continue;
            }

            // Found and opened,  close it and then use ImageStreamIO
            logged = 0;
            close( SM_fd );

            if( ImageStreamIO_openIm( &imageStream, mit->m_shmimName.c_str() ) == 0 )
            {
                /// \todo this isn't right--> isn't there a define in cacao to use?
                if( imageStream.md[0].sem <= semaphoreNumber )

                {
                    ImageStreamIO_closeIm( &imageStream );
                    mx::sys::sleep( 1 ); // We just need to wait for the server process to finish startup.
                }
                else
                {
                    opened = true;
                    char SM_fname[200];
                    ImageStreamIO_filename( SM_fname, sizeof( SM_fname ), mit->m_shmimName.c_str() );

                    struct stat buffer;
                    int         rv = stat( SM_fname, &buffer );

                    if( rv != 0 )
                    {
                        std::cerr << "Could not get inode for " + mit->m_shmimName +
                                         ". Source process will need to be restarted.\n";
                        ImageStreamIO_closeIm( &imageStream );
                        return;
                    }
                    inode = buffer.st_ino;
                }
            }
            else
            {
                mx::sys::sleep( 1 ); // be patient
            }
        }

        if( restart )
        {
            continue; // this is kinda dumb.  we just go around on restart, so why test in the while loop at all?
        }

        if( g_timeToDie )
        {
            if( !opened )
            {
                return;
            }

            ImageStreamIO_closeIm( &imageStream );
            return;
        }

        semaphoreNumber =
            ImageStreamIO_getsemwaitindex( &imageStream, semaphoreNumber ); // ask for semaphore we had before

        if( semaphoreNumber < 0 )
        {
            std::cerr << "No valid semaphore found for " + mit->m_shmimName +
                             ". Source process will need to be restarted.\n";
            return;
        }

        ImageStreamIO_semflush( &imageStream, semaphoreNumber );

        sem_t *sem = imageStream.semptr[semaphoreNumber]; ///< The semaphore to monitor for new image data




        // This is the main image grabbing loop.
        while( !g_timeToDie && !restart && n < m_atimes.size())
        {
            timespec ts;

            if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
            {
                std::cerr << "error from clock_gettime\n";
                return;
            }

            ts.tv_sec += 1;

            if( sem_timedwait( sem, &ts ) == 0 )
            {

                if( g_timeToDie || restart )
                {
                    break; // Check for exit signals
                }

                clock_gettime(CLOCK_ISIO, &m_arrtimes[n]);
                m_atimes[n] = imageStream.md[0].writetime;

                ++n;
            }
            else
            {
                if( imageStream.md[0].sem <= 0 )
                    break; // Indicates that the server has cleaned up.

                // Check for why we timed out
                if( errno == EINTR )
                    break; // This indicates signal interrupted us, time to restart or shutdown, loop will exit normally
                           // if flags set.

                // ETIMEDOUT means we should check for deletion, and then wait more.
                // Otherwise, report an error.
                if( errno != ETIMEDOUT )
                {
                    std::cerr << "error from sem_timedwait\n";
                    break;
                }

                // Check if the file has disappeared.
                int  SM_fd;
                char SM_fname[200];
                ImageStreamIO_filename( SM_fname, sizeof( SM_fname ), mit->m_shmimName.c_str() );
                SM_fd = open( SM_fname, O_RDWR );
                if( SM_fd == -1 )
                {
                    restart = true;
                }
                close( SM_fd );

                // Check if the inode changed
                struct stat buffer;
                int         rv = stat( SM_fname, &buffer );
                if( rv != 0 )
                {
                    restart = true;
                }

                if( buffer.st_ino != inode )
                {
                    restart = true;
                }
            }
        }

        if(n >= m_atimes.size())
        {
            break;
        }
        // opened == true if we can get to this
        if( semaphoreNumber >= 0 )
            imageStream.semReadPID[semaphoreNumber] = 0; // release semaphore
        ImageStreamIO_closeIm( &imageStream );
        opened = false;

    } // outer loop, will exit if m_shutdown==true

    if( opened )
    {
        ImageStreamIO_closeIm( &imageStream );
    }
}

#endif // shmimLat_hpp
