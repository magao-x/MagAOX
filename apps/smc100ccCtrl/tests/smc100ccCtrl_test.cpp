/** \file smc100ccCtrl_test.cpp
  * \brief Catch2 tests for the smc100ccCtrl app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */



#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../smc100ccCtrl.hpp"

using namespace MagAOX::app;

namespace SMCTEST
{

class smc100ccCtrl_test : public smc100ccCtrl 
{

public:
    smc100ccCtrl_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(position);

        //stdMotionStage:
        XWCTEST_SETUP_INDI_NEW_PROP(preset);
        XWCTEST_SETUP_INDI_NEW_PROP(presetName);
        XWCTEST_SETUP_INDI_NEW_PROP(home);
        XWCTEST_SETUP_INDI_NEW_PROP(stop);

    }
};


SCENARIO( "INDI Callbacks", "[smc100ccCtrl]" )
{
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, position);
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, preset);
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, presetName);
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, home);
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, stop);
}


} //namespace smc100ccCtrl_test 
