
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

/// Test harness for dev::dm
/**
 * \ingroup dm_tests
 */
struct dmTest : public MagAOX::app::MagAOXApp<false>,
                public MAPPNS::dm<dmTest, float>,
                public MAPPNS::shmimMonitor<dmTest>
{

    friend class MAPPNS::dm<dmTest, float>;

    typedef MAPPNS::dm<dmTest, float> dmT;

    dmTest( const std::string &git_sha1, const bool git_modified )
        : MagAOX::app::MagAOXApp<false>( git_sha1, git_modified )
    {
        m_configName  = "dmtest";
        m_calibDir    = "/tmp/dmtest_calibs";
        m_calibRelDir = "dmtest";
    }

    ~dmTest() noexcept
    {
    }

    int setupConfig( mx::app::appConfigurator &config )
    {
        return dmT::setupConfig( config );
    }

    int loadConfig( mx::app::appConfigurator &config )
    {
        return dmT::loadConfig( config );
    }

    int appStartup()
    {
        return dmT::appStartup();
    }

    int appLogic()
    {
        return dmT::appLogic();
    }

    int appShutdown()
    {
        return dmT::appShutdown();
    }

    int initDM()
    {
        return 0;
    }

    int zeroDM()
    {
        return 0;
    }

    int releaseDM()
    {
        return 0;
    }

    void setSize( int w, int h, int d )
    {
        m_width    = w;
        m_height   = h;
        m_dataType = d;
    }
};

#ifdef XWCTEST_NAMESPACE
} // namespace XWCTEST_NAMESPACE
#endif

} // namespace dm_tests

// LCOV_EXCL_STOP
