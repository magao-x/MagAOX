/** \file observerCtrl_test.cpp
  * \brief Catch2 tests for the observerCtrl app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */



#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"
#undef XWCTEST_INDI_CALLBACK_VALIDATION

#include "../observerCtrl.hpp"

using namespace MagAOX::app;

namespace SMCTEST
{

class observerCtrl_test : public observerCtrl
{

public:
    observerCtrl_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(observers);
        XWCTEST_SETUP_INDI_NEW_PROP(operators);
        XWCTEST_SETUP_INDI_NEW_PROP(obsName);
        XWCTEST_SETUP_INDI_NEW_PROP(observing);
        XWCTEST_SETUP_INDI_NEW_PROP(obsDuration);
        XWCTEST_SETUP_INDI_NEW_PROP(sws);
        XWCTEST_SETUP_INDI_NEW_PROP(userlog);
        XWCTEST_SETUP_INDI_NEW_PROP(resetTarget);
        XWCTEST_SETUP_INDI_NEW_PROP(target);
        XWCTEST_SETUP_INDI_NEW_PROP(tcsTarget);

        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_catalog, tcsi, catalog);
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_catdata, tcsi, catdata );
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_teldata, tcsi, teldata);
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_labMode, tcsi, labMode);


    }

    bool & observing()
    {
        return m_observing;
    }

    std::string & catObj()
    {
        return m_catObj;
    }

    std::string & target()
    {
        return m_target;
    }

    bool & newTargetBlock()
    {
        return m_newTargetBlock;
    }

    bool & newPointing()
    {
        return m_newPointing;
    }

};


/// Changing target and catObj name
/** Test changing target and catalog-object names via INDI
 *
 * \ingroup observerCtrl_tests
 */
TEST_CASE( "Changing target and catObj name", "[observerCtrl]" )
{
    //test that catalog callback fails if no "object" element
    SECTION( "Catalog updates, no object in INDI prop" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );

        ip.setDevice( "tcsi" );
        ip.setName( "catalog" );
        ip.add( pcf::IndiElement( "noobject" ) );

        ip["noobject"] = "target0";

        observerCtrl_test obs("observers");

        int rv = obs.setCallBack_m_indiP_catalog(ip);

        REQUIRE( rv == -1);
    }

    //test that target callback fails if no current or target element
    SECTION( "setting target, no current or target" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );

        ip.setDevice( "observers" );
        ip.setName( "target" );
        ip.add( pcf::IndiElement( "novalid" ) );

        ip["novalid"] = "target1";

        observerCtrl_test obs("observers");

        int rv = obs.newCallBack_m_indiP_target(ip);

        REQUIRE( rv == -1);
    }

    //test that tcsTarget callback fails if no current or target element
    SECTION( "setting tcsTarget, no request" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );

        ip.setDevice( "observers" );
        ip.setName( "tcsTarget" );
        ip.add( pcf::IndiElement( "norequest" ) );

        ip["norequest"].setSwitchState(pcf::IndiElement::On);

        observerCtrl_test obs("observers");

        int rv = obs.newCallBack_m_indiP_tcsTarget(ip);

        REQUIRE( rv == -1);
    }

    //Set catobj from catalog, target from target, then synchronize
    SECTION( "Catalog and target, while not observing" )
    {
        observerCtrl_test obs("observers");
        obs.observing() = false;
        obs.newTargetBlock() = false; //as if an observation has already taken place

        pcf::IndiProperty ip( pcf::IndiProperty::Text );

        ip.setDevice( "tcsi" );
        ip.setName( "catalog" );
        ip.add( pcf::IndiElement( "object" ) );

        ip["object"] = "target0";

        int rv = obs.setCallBack_m_indiP_catalog(ip);

        REQUIRE( rv == 0);
        REQUIRE( obs.catObj() == "target0");
        REQUIRE( obs.newTargetBlock() == false);
        REQUIRE( obs.newPointing() == true );

        pcf::IndiProperty ip2( pcf::IndiProperty::Text );

        ip2.setDevice( "observers" );
        ip2.setName( "target" );
        ip2.add( pcf::IndiElement( "target" ) );

        ip2["target"] = "target1";

        rv = obs.newCallBack_m_indiP_target(ip2);

        REQUIRE( rv == 0);
        REQUIRE( obs.target() == "target1");
        REQUIRE( obs.newTargetBlock() == true);
        REQUIRE( obs.newPointing() == false );

        pcf::IndiProperty ip3( pcf::IndiProperty::Switch );

        ip3.setDevice( "observers" );
        ip3.setName( "tcsTarget" );
        ip3.add( pcf::IndiElement( "request" ) );

        obs.newTargetBlock() = false; //reset to make sure they are toggle correctly
        obs.newPointing() = true;
        ip3["request"].setSwitchState(pcf::IndiElement::Off);         //Start off


        rv = obs.newCallBack_m_indiP_tcsTarget(ip3);

        REQUIRE( rv == 0);
        REQUIRE( obs.catObj() == "target0");
        REQUIRE( obs.target() == "target1");
        REQUIRE( obs.newTargetBlock() == false);
        REQUIRE( obs.newPointing() == true );

        //now trigger change
        ip3["request"].setSwitchState(pcf::IndiElement::On);

        rv = obs.newCallBack_m_indiP_tcsTarget(ip3);

        REQUIRE( rv == 0);
        REQUIRE( obs.catObj() == "target0");
        REQUIRE( obs.target() == "target0");
        REQUIRE( obs.newTargetBlock() == true);
        REQUIRE( obs.newPointing() == false );

    }

    //Set catobj from catalog, target from target, then synchronize
    SECTION( "Catalog and target, while observing" )
    {
        observerCtrl_test obs("observers");
        obs.observing() = true;
        obs.newTargetBlock() = false; //as if an observation has already taken place

        pcf::IndiProperty ip( pcf::IndiProperty::Text );

        ip.setDevice( "tcsi" );
        ip.setName( "catalog" );
        ip.add( pcf::IndiElement( "object" ) );

        ip["object"] = "target0";

        int rv = obs.setCallBack_m_indiP_catalog(ip);

        REQUIRE( rv == 0);
        REQUIRE( obs.catObj() == "target0");
        REQUIRE( obs.newTargetBlock() == false);
        REQUIRE( obs.newPointing() == true );

        pcf::IndiProperty ip2( pcf::IndiProperty::Text );

        ip2.setDevice( "observers" );
        ip2.setName( "target" );
        ip2.add( pcf::IndiElement( "target" ) );

        ip2["target"] = "target1";

        rv = obs.newCallBack_m_indiP_target(ip2);

        REQUIRE( rv == -2);
        REQUIRE( obs.catObj() == "target0");
        REQUIRE( obs.target() == "");
        REQUIRE( obs.newTargetBlock() == false);
        REQUIRE( obs.newPointing() == true ); //still true

        pcf::IndiProperty ip3( pcf::IndiProperty::Switch );

        ip3.setDevice( "observers" );
        ip3.setName( "tcsTarget" );
        ip3.add( pcf::IndiElement( "request" ) );

        //Start off
        ip3["request"].setSwitchState(pcf::IndiElement::Off);

        rv = obs.newCallBack_m_indiP_tcsTarget(ip3);

        REQUIRE( rv == 0);
        REQUIRE( obs.catObj() == "target0");
        REQUIRE( obs.target() == "");

        //now trigger change
        ip3["request"].setSwitchState(pcf::IndiElement::On);

        rv = obs.newCallBack_m_indiP_tcsTarget(ip3);

        REQUIRE( rv == -2);
        REQUIRE( obs.catObj() == "target0");
        REQUIRE( obs.target() == "");

    }


}


} //namespace observerCtrl_test
