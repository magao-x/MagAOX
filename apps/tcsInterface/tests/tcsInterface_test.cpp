/** \file tcsInterface_test.cpp
  * \brief Catch2 tests for the tcsInterface app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */



#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../tcsInterface.hpp"

using namespace MagAOX::app;

namespace TCSITEST
{

class tcsInterface_test : public tcsInterface 
{

public:
    tcsInterface_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(pyrNudge);
        XWCTEST_SETUP_INDI_NEW_PROP(acqFromGuider);
        XWCTEST_SETUP_INDI_NEW_PROP(offlTTenable);
        XWCTEST_SETUP_INDI_NEW_PROP(offlTTdump);
        XWCTEST_SETUP_INDI_NEW_PROP(offlTTavgInt);
        XWCTEST_SETUP_INDI_NEW_PROP(offlTTgain);
        XWCTEST_SETUP_INDI_NEW_PROP(offlTTthresh);
        XWCTEST_SETUP_INDI_NEW_PROP(offlFenable);
        XWCTEST_SETUP_INDI_NEW_PROP(offlFdump);
        XWCTEST_SETUP_INDI_NEW_PROP(offlFavgInt);
        XWCTEST_SETUP_INDI_NEW_PROP(offlFgain);
        XWCTEST_SETUP_INDI_NEW_PROP(offlFthresh);
        //XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_teldata, tcsi, zd);
    }
};


SCENARIO( "INDI Callbacks", "[tcsInterface]" )
{
    XWCTEST_INDI_NEW_CALLBACK( tcsInterface, pyrNudge);
    XWCTEST_INDI_NEW_CALLBACK( tcsInterface, acqFromGuider);
    XWCTEST_INDI_NEW_CALLBACK( tcsInterface, offlTTenable);
    XWCTEST_INDI_NEW_CALLBACK( tcsInterface, offlTTdump);
    XWCTEST_INDI_NEW_CALLBACK( tcsInterface, offlTTavgInt);
    XWCTEST_INDI_NEW_CALLBACK( tcsInterface, offlTTgain);
    XWCTEST_INDI_NEW_CALLBACK( tcsInterface, offlTTthresh);
    XWCTEST_INDI_NEW_CALLBACK( tcsInterface, offlFenable);
    XWCTEST_INDI_NEW_CALLBACK( tcsInterface, offlFdump);
    XWCTEST_INDI_NEW_CALLBACK( tcsInterface, offlFavgInt);
    XWCTEST_INDI_NEW_CALLBACK( tcsInterface, offlFgain);
    XWCTEST_INDI_NEW_CALLBACK( tcsInterface, offlFthresh);

    //XWCTEST_INDI_SET_CALLBACK( tcsInterface, m_indiP_teldata, tcsi, zd);

}


} //namespace tcsInterface_test 
