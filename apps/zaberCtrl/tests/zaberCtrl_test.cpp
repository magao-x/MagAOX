/** \file zaberCtrl_test.cpp
  * \brief Catch2 tests for the zaberCtrl app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */



#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../zaberCtrl.hpp"

using namespace MagAOX::app;

namespace ZCTRLTEST
{

class zaberCtrl_test : public zaberCtrl 
{

public:
    zaberCtrl_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(pos);
        XWCTEST_SETUP_INDI_NEW_PROP(rawPos);

        //stdMotionStage:
        XWCTEST_SETUP_INDI_NEW_PROP(preset);
        XWCTEST_SETUP_INDI_NEW_PROP(presetName);
        XWCTEST_SETUP_INDI_NEW_PROP(home);
        XWCTEST_SETUP_INDI_NEW_PROP(stop);

        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_stageState, stest, curr_state);
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_stageMaxRawPos, stest, max_pos);
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_stageRawPos, stest, curr_pos);
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_stageTgtPos, stest, tgt_pos);
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_stageTemp, stest, temp);
    }
};


SCENARIO( "INDI Callbacks", "[zaberCtrl]" )
{
    XWCTEST_INDI_NEW_CALLBACK( zaberCtrl, pos);
    XWCTEST_INDI_NEW_CALLBACK( zaberCtrl, rawPos);
    XWCTEST_INDI_NEW_CALLBACK( zaberCtrl, preset);
    XWCTEST_INDI_NEW_CALLBACK( zaberCtrl, presetName);
    XWCTEST_INDI_NEW_CALLBACK( zaberCtrl, home);
    XWCTEST_INDI_NEW_CALLBACK( zaberCtrl, stop);

    XWCTEST_INDI_SET_CALLBACK( zaberCtrl, m_indiP_stageState, stest, curr_state);
    XWCTEST_INDI_SET_CALLBACK( zaberCtrl, m_indiP_stageMaxRawPos, stest, max_pos);
    XWCTEST_INDI_SET_CALLBACK( zaberCtrl, m_indiP_stageRawPos, stest, curr_pos);
    XWCTEST_INDI_SET_CALLBACK( zaberCtrl, m_indiP_stageTgtPos, stest, tgt_pos);
    XWCTEST_INDI_SET_CALLBACK( zaberCtrl, m_indiP_stageTemp, stest, temp);

}


} //namespace zaberCtrl_test 
