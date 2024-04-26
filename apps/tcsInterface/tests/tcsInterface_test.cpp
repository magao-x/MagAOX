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

SCENARIO( "Parsing times in x:m:s format", "[tcsInterface]" )
{
    GIVEN("A valid x:m:s string")
    {
        int rv;

        WHEN("Positive, double digit integers")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "12:20:50";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == 0);
            REQUIRE_THAT(x, Catch::Matchers::WithinAbs(12, 1e-10)); 
            REQUIRE_THAT(m, Catch::Matchers::WithinAbs(20, 1e-10)); 
            REQUIRE_THAT(s, Catch::Matchers::WithinAbs(50, 1e-10)); 
        }

        WHEN("Negative, double digit integers")
        {
            tcsInterface_test tit("tcsi");

            std::string tstr = "-22:30:48";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == 0);
            REQUIRE_THAT(x, Catch::Matchers::WithinAbs(-22, 1e-10)); 
            REQUIRE_THAT(m, Catch::Matchers::WithinAbs(-30, 1e-10)); 
            REQUIRE_THAT(s, Catch::Matchers::WithinAbs(-48, 1e-10)); 
        }

        WHEN("Positive, double digit, decimal seconds")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "12:20:50.267849";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == 0);
            REQUIRE_THAT(x, Catch::Matchers::WithinAbs(12, 1e-10)); 
            REQUIRE_THAT(m, Catch::Matchers::WithinAbs(20, 1e-10)); 
            REQUIRE_THAT(s, Catch::Matchers::WithinAbs(50.267849, 1e-10)); 
        }

        WHEN("Negative, double digit integers")
        {
            tcsInterface_test tit("tcsi");

            std::string tstr = "-22:30:48.8771819";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == 0);
            REQUIRE_THAT(x, Catch::Matchers::WithinAbs(-22, 1e-10)); 
            REQUIRE_THAT(m, Catch::Matchers::WithinAbs(-30, 1e-10)); 
            REQUIRE_THAT(s, Catch::Matchers::WithinAbs(-48.8771819, 1e-10)); 
     }
   }

    GIVEN("Invalid x:m:s strings")
    {
        int rv;

        WHEN("empty")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == -1);
        }

        WHEN("no :")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "12-20-50";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == -1);
        }

        WHEN("only one :")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "12:20-50";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == -1);
        }

        WHEN("two :, but one at beginning")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = ":12:20";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == -1);
        }

        WHEN("two :, but no m")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "12::20";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == -1);
        }

        WHEN("two :, but one at end")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "12:20:";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == -1);
        }

        WHEN("invalid x")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "x:20:80";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == -1);
        }

        WHEN("invalid -x")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "-x:20:80";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == -1);
        }

        WHEN("invalid m")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "20:m:80";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == -1);
        }

        WHEN("invalid -m")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "-12:m:80";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == -1);
        }

        WHEN("invalid s")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "20:23:s.ssy";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == -1);
        }

        WHEN("invalid -s")
        { 
            tcsInterface_test tit("tcsi");

            std::string tstr = "-12:23:s.sye";
         
            double x;
            double m;
            double s;

            rv = tit.parse_xms(x, m, s, tstr);        
         
            REQUIRE(rv == -1);
        }
    }
}

} //namespace tcsInterface_test 
