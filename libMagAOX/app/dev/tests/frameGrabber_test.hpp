/** \file frameGrabber_test.hpp
 * \brief Test harness for the MagAOX::app::dev::frameGrabber device mixin.
 *
 * One harness class template, fgHarness<capsT>, drives the real frameGrabber code
 * against a fake camera. capsT is a plain struct that names the config, says whether
 * the flip option is on, and says whether the optional post-publish hook is present.
 * Three configurations are used and keep their historical names:
 *
 * - fgTest is flippable and has no post-publish hook.
 * - fgTestHook is flippable and defines frameGrabberPostPublish().
 * - fgTestNoFlip is not flippable and has no post-publish hook.
 *
 * The harness stubs every derived-class hook the mixin calls. Each stub counts its
 * calls and can be told to fail a set number of times. acquireAndCheckValid() pops
 * its return values from a queue and requests shutdown when the queue is empty, which
 * is how the tests end the acquisition loop. The common parts of every dev:: harness,
 * such as the FIFO-less indiDriver and the registration fault injection, come from
 * appHarnessBaseT<true> in testHarnessCommon.hpp.
 *
 * This header also owns the ImageStreamIO test environment. It points MILK_SHM_DIR at
 * /tmp/frameGrabber_test/shm from a static initializer and provides tempStream, an
 * RAII wrapper for a stream created outside the frameGrabber under test.
 *
 * \ingroup testing
 */

#include "../../../../tests/catch2/catch.hpp"

#include <deque>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unistd.h>

// Turn protected into public for this whole translation unit. This lets the tests
// read and write MagAOXApp and frameGrabber internals directly without adding
// accessor wrappers to the harness classes.
#define protected public
#include "../../MagAOXApp.hpp"
#include "../telemeter.hpp"
#include "../frameGrabber.hpp"
#undef protected

#include "testHarnessCommon.hpp"

// LCOV_EXCL_START

namespace frameGrabber_tests
{

// ImageStreamIO caches the shared memory directory the first time it is queried.
// The cache is a static local in ImageStreamIO_shmdirname. So MILK_SHM_DIR must be
// set before any ImageStreamIO call happens anywhere in this process. Doing this in
// a namespace-scope static initializer guarantees it runs before any TEST_CASE body.
const std::string g_shmDir = "/tmp/frameGrabber_test/shm";

/// Create the shared memory directory and point MILK_SHM_DIR at it.
inline int setupShmDir()
{
    mx::ioutils::createDirectories( g_shmDir );
    setenv( "MILK_SHM_DIR", g_shmDir.c_str(), 1 );
    return 0;
}
static int g_shmDirSetup = setupShmDir();

/// Build a unique shmim name for one temporary test stream.
inline std::string uniqueShmimName( const std::string &suffix )
{
    static unsigned counter = 0;
    ++counter;
    return "frameGrabber_test_" + suffix + "_" + std::to_string( ::getpid() ) + "_" + std::to_string( counter );
}

/// RAII wrapper for a temporary ImageStreamIO stream created outside the frameGrabber
/// under test. It simulates a shmim owned by another process. The stream is destroyed
/// in the destructor unless dismiss() has been called.
class tempStream
{
  public:
    explicit tempStream( const std::string &name,
                         long               naxis,
                         uint32_t           w,
                         uint32_t           h,
                         uint32_t           d,
                         uint8_t            dataType = _DATATYPE_UINT8,
                         int                nbsem    = IMAGE_NB_SEMAPHORE )
        : m_name( name )
    {
        uint32_t imsize[3] = { w, h, d };
        if( ImageStreamIO_createIm_gpu( &m_image,
                                        m_name.c_str(),
                                        naxis,
                                        imsize,
                                        dataType,
                                        -1,
                                        1,
                                        nbsem,
                                        0,
                                        CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                        0 ) != IMAGESTREAMIO_SUCCESS )
        {
            throw std::runtime_error( "failed to create temporary ImageStreamIO stream" );
        }
        m_image.md[0].cnt1 = 0;
    }

    ~tempStream()
    {
        if( m_owner )
        {
            ImageStreamIO_destroyIm( &m_image );
        }
    }

    IMAGE *image()
    {
        return &m_image;
    }

    /// Release destruction ownership. Call this after handing the stream to a
    /// frameGrabber under test that will destroy it itself.
    void dismiss()
    {
        m_owner = false;
    }

  private:
    std::string m_name;
    IMAGE       m_image{};
    bool        m_owner{ true };
};

/// Configuration with the flip option on and no post-publish hook.
struct fgDefaultCaps
{
    static constexpr const char *configSuffix    = "app";
    static constexpr bool        flippable       = true;
    static constexpr bool        postPublishHook = false;
};

/// Configuration with the flip option on and the post-publish hook defined.
struct fgHookCaps
{
    static constexpr const char *configSuffix    = "hookapp";
    static constexpr bool        flippable       = true;
    static constexpr bool        postPublishHook = true;
};

/// Configuration with the flip option off and no post-publish hook.
struct fgNoFlipCaps
{
    static constexpr const char *configSuffix    = "noflipapp";
    static constexpr bool        flippable       = false;
    static constexpr bool        postPublishHook = false;
};

/// Mixin that gives the harness the optional frameGrabberPostPublish() hook.
/** frameGrabber detects the hook by expression SFINAE on the derived object, so a
 * hook inherited from this base is found the same way as one declared inline. The
 * hook counts its calls and returns postPublishReturn. A negative return also
 * requests shutdown so the acquisition loop ends.
 */
template <class derivedT>
struct fgPostPublishHook
{
    int postPublishCalls{ 0 };
    int postPublishReturn{ 0 };

    int frameGrabberPostPublish( IMAGE *imageStream )
    {
        static_cast<void>( imageStream );
        ++postPublishCalls;
        if( postPublishReturn < 0 )
        {
            static_cast<derivedT *>( this )->m_shutdown = 1;
        }
        return postPublishReturn;
    }
};

/// Stand-in for fgPostPublishHook when the configuration has no hook.
struct fgNoPostPublishHook
{
};

/// The frameGrabber test harness. capsT selects the configuration. See the file comment.
template <class capsT>
struct fgHarness
    : public MagAOX::app::dev::testHarness::appHarnessBaseT<true>,
      public MagAOX::app::dev::frameGrabber<fgHarness<capsT>>,
      public MagAOX::app::dev::telemeter<fgHarness<capsT>>,
      public std::conditional_t<capsT::postPublishHook, fgPostPublishHook<fgHarness<capsT>>, fgNoPostPublishHook>
{
    typedef MagAOX::app::dev::testHarness::appHarnessBaseT<true>  baseT;
    typedef MagAOX::app::dev::frameGrabber<fgHarness<capsT>>      frameGrabberT;
    typedef MagAOX::app::dev::telemeter<fgHarness<capsT>>         telemeterT;

    static constexpr bool c_frameGrabber_flippable = capsT::flippable;

    // configureAcquisition() controls.
    int      configureAcquisitionCalls{ 0 };
    int      configureAcquisitionFailCount{ 0 }; ///< Number of leading calls that should fail.
    uint32_t nextWidth{ 2 };
    uint32_t nextHeight{ 2 };
    uint8_t  nextDataType{ _DATATYPE_UINT8 };

    // fps() controls.
    float fpsValue{ 1000.0 };

    // startAcquisition() controls.
    int startAcquisitionCalls{ 0 };
    int startAcquisitionFailCount{ 0 };

    // acquireAndCheckValid() controls.
    std::deque<int> acquireResults; ///< Popped in call order. An empty queue shuts the app down.
    int             acquireCalls{ 0 };
    bool            setReconfigOnNextAcquire{ false };
    int             bumpFpsAfterCall{ -1 }; ///< If 0 or more, fpsValue changes right after this call count is reached.
    float           bumpedFpsValue{ 0 };

    // loadImageIntoStream() controls.
    int  loadImageIntoStreamCalls{ 0 };
    bool failLoadImageIntoStream{ false };

    // reconfig() controls.
    int reconfigCalls{ 0 };

    fgHarness() : baseT( uniqueShmimName( capsT::configSuffix ) )
    {
    }

    // A mixin base has a destructor that may throw. MagAOXApp's destructor is noexcept,
    // so the implicit destructor here would be rejected as looser. Declare it explicitly.
    ~fgHarness() noexcept override
    {
    }

    // Overrides of the MagAOXApp pure virtuals. The tests call frameGrabberT::appStartup(),
    // frameGrabberT::appLogic(), and frameGrabberT::appShutdown() explicitly. These
    // trivial overrides only make the class instantiable and disambiguate unqualified
    // lookup between MagAOXApp and dev::frameGrabber.
    int appStartup()
    {
        return 0;
    }
    int appLogic()
    {
        return 0;
    }
    int appShutdown()
    {
        return 0;
    }

    int setupConfig( mx::app::appConfigurator &config )
    {
        return frameGrabberT::setupConfig( config );
    }

    int loadConfig( mx::app::appConfigurator &config )
    {
        return frameGrabberT::loadConfig( config );
    }

    int configureAcquisition()
    {
        ++configureAcquisitionCalls;
        if( configureAcquisitionCalls <= configureAcquisitionFailCount )
        {
            return -1;
        }
        this->m_width    = nextWidth;
        this->m_height   = nextHeight;
        this->m_dataType = nextDataType;
        return 0;
    }

    float fps()
    {
        return fpsValue;
    }

    int startAcquisition()
    {
        ++startAcquisitionCalls;
        if( startAcquisitionCalls <= startAcquisitionFailCount )
        {
            return -1;
        }
        return 0;
    }

    int acquireAndCheckValid()
    {
        ++acquireCalls;
        if( acquireResults.empty() )
        {
            this->m_shutdown = 1;
            return -1;
        }

        int r = acquireResults.front();
        acquireResults.pop_front();

        if( setReconfigOnNextAcquire )
        {
            this->m_reconfig         = true;
            setReconfigOnNextAcquire = false;
        }

        if( r == 0 )
        {
            clock_gettime( CLOCK_REALTIME, &this->m_currImageTimestamp );
        }

        if( bumpFpsAfterCall >= 0 && acquireCalls == bumpFpsAfterCall )
        {
            fpsValue = bumpedFpsValue;
        }

        return r;
    }

    int loadImageIntoStream( void *dest )
    {
        ++loadImageIntoStreamCalls;
        static_cast<void>( dest );
        if( failLoadImageIntoStream )
        {
            this->m_shutdown = 1;
            return -1;
        }
        return 0;
    }

    int reconfig()
    {
        ++reconfigCalls;
        this->m_shutdown = 1; // End the outer loop cleanly once reconfig has been exercised.
        return 0;
    }
};

/// Flippable harness without the post-publish hook.
using fgTest = fgHarness<fgDefaultCaps>;

/// Flippable harness with the post-publish hook.
using fgTestHook = fgHarness<fgHookCaps>;

/// Non-flippable harness without the post-publish hook.
using fgTestNoFlip = fgHarness<fgNoFlipCaps>;

} // namespace frameGrabber_tests

// LCOV_EXCL_STOP
