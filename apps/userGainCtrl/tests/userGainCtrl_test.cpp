/** \file userGainCtrl_test.cpp
  * \brief Catch2 tests for the userGainCtrl app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */



#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../userGainCtrl.hpp"

using namespace MagAOX::app;

namespace SMCTEST
{

class userGainCtrl_test : public userGainCtrl 
{

public:
    userGainCtrl_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(zeroAll);
        XWCTEST_SETUP_INDI_NEW_PROP(singleModeNo);
        XWCTEST_SETUP_INDI_NEW_PROP(singleGain);
        XWCTEST_SETUP_INDI_NEW_PROP(singleMC);
    
    }
};


SCENARIO( "INDI Callbacks", "[userGainCtrl]" )
{
    XWCTEST_INDI_NEW_CALLBACK( userGainCtrl, zeroAll);
    XWCTEST_INDI_NEW_CALLBACK( userGainCtrl, singleModeNo);
    XWCTEST_INDI_NEW_CALLBACK( userGainCtrl, singleGain);
    XWCTEST_INDI_NEW_CALLBACK( userGainCtrl, singleMC);
    XWCTEST_INDI_ARBNEW_CALLBACK( userGainCtrl, newCallBack_blockGains, block00_gain);
    XWCTEST_INDI_ARBNEW_CALLBACK( userGainCtrl, newCallBack_blockMCs, block70_multcoeff);
    XWCTEST_INDI_ARBNEW_CALLBACK( userGainCtrl, newCallBack_blockLimits, block32_limit);

}


} //namespace userGainCtrl_test 
