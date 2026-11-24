/** \file dm_test.hpp
 * \brief Test harness for the generic deformable mirror controller.
 */

#ifndef dm_test_hpp
#define dm_test_hpp

#include "../../MagAOXApp.hpp"
#include "../dm.hpp"
#include "../shmimMonitor.hpp"

// LCOV_EXCL_START

#ifndef XWCTEST_NAMESPACE
    #define MAPPNS MagAOX::app::dev
#else
    #define MAPPNS MagAOX::app::dev::XWCTEST_NAMESPACE
#endif

namespace dm_tests
{

#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE
{
#endif

/// \cond DM_TEST_HARNESS

/// Controller harness which replaces hardware calls with successful local stubs.
struct dmTest : public MagAOX::app::MagAOXApp<false>,
                public MAPPNS::dm<dmTest, float>,
                public MAPPNS::shmimMonitor<dmTest>
{
    friend class MAPPNS::dm<dmTest, float>;

    typedef MAPPNS::dm<dmTest, float> dmT;

    unsigned m_releaseCount{ 0 }; ///< Number of local hardware-release stub calls.

    /// Construct an INDI-disabled controller using local calibration paths.
    dmTest( const std::string &git_sha1 /**< [in] Test build identity. */,
            const bool         git_modified /**< [in] Test source modification state. */ );

    /// Destroy the controller harness.
    ~dmTest() noexcept = default;

    /// Register controller configuration entries.
    int setupConfig( mx::app::appConfigurator &config /**< [out] Configurator to populate. */ );

    /// Load controller configuration entries.
    int loadConfig( mx::app::appConfigurator &config /**< [in] Parsed controller settings. */ );

    /// Run controller startup.
    int appStartup();

    /// Run one controller logic iteration.
    int appLogic();

    /// Run controller shutdown.
    int appShutdown();

    /// Complete initialization without accessing hardware.
    int initDM();

    /// Complete hardware zeroing without accessing hardware.
    int zeroDM();

    /// Record release without accessing hardware.
    int releaseDM();

    /// Set the monitored stream metadata used by allocation tests.
    void setSize( int w /**< [in] Stream width. */,
                  int h /**< [in] Stream height. */,
                  int d /**< [in] ImageStreamIO data type. */ );

    /// Configure numbered-channel clearing without starting monitor or saturation threads.
    void setChannels( const std::string &name /**< [in] Numbered-channel prefix. */,
                      uint32_t           width /**< [in] Configured DM width. */,
                      uint32_t           height /**< [in] Configured DM height. */,
                      int                count /**< [in] Number of numbered main channels. */ );
};

inline dmTest::dmTest( const std::string &git_sha1, const bool git_modified )
    : MagAOX::app::MagAOXApp<false>( git_sha1, git_modified )
{
    m_configName  = "dmtest";
    m_calibDir    = "/tmp/dmtest_calibs";
    m_calibRelDir = "dmtest";
}

inline int dmTest::setupConfig( mx::app::appConfigurator &config )
{
    return dmT::setupConfig( config );
}

inline int dmTest::loadConfig( mx::app::appConfigurator &config )
{
    return dmT::loadConfig( config );
}

inline int dmTest::appStartup()
{
    return dmT::appStartup();
}

inline int dmTest::appLogic()
{
    return dmT::appLogic();
}

inline int dmTest::appShutdown()
{
    return dmT::appShutdown();
}

inline int dmTest::initDM()
{
    return 0;
}

inline int dmTest::zeroDM()
{
    return 0;
}

inline int dmTest::releaseDM()
{
    ++m_releaseCount;
    return 0;
}

inline void dmTest::setSize( int w, int h, int d )
{
    m_width    = w;
    m_height   = h;
    m_dataType = d;
}

inline void dmTest::setChannels( const std::string &name, uint32_t width, uint32_t height, int count )
{
    m_shmimName   = name;
    m_dmWidth     = width;
    m_dmHeight    = height;
    m_numChannels = count;
}

/// \endcond

#ifdef XWCTEST_NAMESPACE
} // namespace XWCTEST_NAMESPACE
#endif

} // namespace dm_tests

// LCOV_EXCL_STOP

#endif // dm_test_hpp
