
#include "../../MagAOXApp.hpp"
#include "../telemeter.hpp"
#include "testHarnessCommon.hpp"

// LCOV_EXCL_START

#ifndef XWCTEST_NAMESPACE
#define MAPPNS MagAOX::app::dev
#else
#define MAPPNS MagAOX::app::dev::XWCTEST_NAMESPACE

#endif

namespace telemeter_tests
{

#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE
{
#endif

/// Test harness for dev::telemeter
/**
 * \ingroup telemeter_tests
 */
struct telemeterTest : public MagAOX::app::dev::testHarness::appHarnessBase, public MAPPNS::telemeter<telemeterTest>
{

    friend class MAPPNS::telemeter<telemeterTest>;

    typedef MagAOX::app::dev::testHarness::appHarnessBase baseT;
    typedef MAPPNS::telemeter<telemeterTest>              telemeterT;

    telemeterTest() : baseT( "teltest" )
    {
    }

    int setupConfig( mx::app::appConfigurator &config )
    {
        return telemeterT::setupConfig( config );
    }

    int loadConfig( mx::app::appConfigurator &config )
    {
        return telemeterT::loadConfig( config );
    }

    int appStartup()
    {
        return telemeterT::appStartup();
    }

    int appLogic()
    {
        return telemeterT::appLogic();
    }

    int appShutdown()
    {
        return telemeterT::appShutdown();
    }

    int checkRecordTimes()
    {
        return telemeterT::checkRecordTimes(MagAOX::logger::telem_position(),MagAOX::logger::telem_saving_state());
    }

    int recordTelem( const MagAOX::logger::telem_position *)
    {
        return telem<MagAOX::logger::telem_position>(2.5);
    }

    int recordTelem( const MagAOX::logger::telem_saving_state *)
    {
        return telem<MagAOX::logger::telem_saving_state>({0,0});
    }


};

#ifdef XWCTEST_NAMESPACE
} // namespace XWCTEST_NAMESPACE
#endif

} // namespace telemeter_tests

// LCOV_EXCL_STOP
