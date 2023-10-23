//#define CATCH_CONFIG_MAIN
#include "../../../tests/catch2/catch.hpp"

#include <mx/sys/timeUtils.hpp>

#include "../MagAOXApp.hpp"

namespace MagAOXApp_tests 
{
   
struct MagAOXApp_test : public MagAOX::app::MagAOXApp<true>
{
    MagAOXApp_test() : MagAOXApp("sha1",false){}
    
    virtual int appStartup(){return 0;}
    virtual int appLogic(){return 0;}
    virtual int appShutdown(){return 0;}

    std::string configName()
    {
        return MagAOX::app::MagAOXApp<true>::configName();
    }

    void configName(const std::string & cn)
    {
        m_configName = cn;

        m_indiDriver = new MagAOX::app::indiDriver<MagAOX::app::MagAOXApp<true>>(this, m_configName, "0", "0");
    }

    int called_back {0};
};

int callback( void * app, const pcf::IndiProperty &ipRecv)
{
    static_cast<void>(ipRecv); //be unused

    MagAOXApp_test * appt = static_cast<MagAOXApp_test*>(app);

    appt->called_back = 1;

    return 0;
}

SCENARIO( "MagAOXApp INDI NewProperty", "[MagAOXApp]" ) 
{
    GIVEN("a new property request")
    {
        WHEN("a wrong device name")
        {
            MagAOXApp_test app;

            app.configName("test");

            REQUIRE(app.configName() == "test");

            pcf::IndiProperty prop;
            app.registerIndiPropertyNew(prop, "nprop", pcf::IndiProperty::Number, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle, callback);

            pcf::IndiProperty nprop;

            //First test the right device name
            nprop.setDevice("test");
            nprop.setName("nprop");

            app.handleNewProperty(nprop);

            REQUIRE(app.called_back == 1);

            app.called_back = 0;

            //Now test the wrong device name
            nprop.setDevice("wrong");

            app.handleNewProperty(nprop);

            REQUIRE(app.called_back == 0);

        }
    }
}


} //namespace MagAOXApp_tests 
