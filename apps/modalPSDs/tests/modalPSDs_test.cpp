/** \file modalPSDs_test.cpp
  * \brief Catch2 tests for the modalPSDs app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */



#include "../../../tests/catch2/catch.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../modalPSDs.hpp"

using namespace MagAOX::app;

//namespace MPSDTEST
//{

class modalPSDs_test : public modalPSDs 
{

public:
    modalPSDs_test(const std::string & device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(psdTime);
        XWCTEST_SETUP_INDI_NEW_PROP(psdAvgTime);
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_fpsSource, modeamps, fps )
    }
};

SCENARIO( "INDI Callbacks", "[modalPSDs]" )
{
    XWCTEST_INDI_NEW_CALLBACK( modalPSDs, psdTime);
    XWCTEST_INDI_NEW_CALLBACK( modalPSDs, psdAvgTime);
    XWCTEST_INDI_SET_CALLBACK( modalPSDs, m_indiP_fpsSource, modeamps, fps);
}


//} //namespace modalPSDs_test 
