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

        XWCTEST_SETUP_INDI_PROP(pos)
        XWCTEST_SETUP_INDI_PROP(rawPos)
        XWCTEST_SETUP_INDI_PROP(preset)
        XWCTEST_SETUP_INDI_PROP(presetName)
        XWCTEST_SETUP_INDI_PROP(home)
        XWCTEST_SETUP_INDI_PROP(stop)
    }
};

//#define QUOTE(s) #s


SCENARIO( "INDI Callbacks", "[zaberCtrl]" )
{
    XWCTEST_INDI_CALLBACK( zaberCtrl, pos);
    XWCTEST_INDI_CALLBACK( zaberCtrl, rawPos);
    XWCTEST_INDI_CALLBACK( zaberCtrl, preset);
    XWCTEST_INDI_CALLBACK( zaberCtrl, presetName);
    XWCTEST_INDI_CALLBACK( zaberCtrl, home);
    XWCTEST_INDI_CALLBACK( zaberCtrl, stop);


}


} //namespace zaberCtrl_test 
