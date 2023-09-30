
#ifndef testMacrosINDI_hpp
#define testMacrosINDI_hpp

#define XWCTEST_INDI_CALLBACK_VALIDATION

#define XWCTEST_MAKE_INDI_PROP( stub, propname) stub ## propname

#define XWCTEST_SETUP_INDI_PROP( propname )                            \
   XWCTEST_MAKE_INDI_PROP(m_indiP_, propname).setDevice(m_configName); \
   XWCTEST_MAKE_INDI_PROP(m_indiP_, propname).setName(#propname);      \

#define XWCTEST_MAKE_INDI_CALLBACK( stub, callback) stub ## callback 

#define XWCTEST_INDI_CALLBACK( testclass, propname)                                      \
   GIVEN("A New Callback for " # propname  )                                              \
   {                                                                                     \
        WHEN("Wrong Device")                                                             \
        {                                                                                \
            testclass ## _test sdgt("right");                                            \
            pcf::IndiProperty ip;                                                        \
            ip.setDevice("wrong");                                                       \
            ip.setName( #propname );                                                     \
            int rv = sdgt.XWCTEST_MAKE_INDI_CALLBACK(newCallBack_m_indiP_,propname)(ip); \
            REQUIRE(rv == -1);                                                           \
        }                                                                                \
        WHEN("Wrong name")                                                               \
        {                                                                                \
            testclass ## _test sdgt("right");                                            \
            pcf::IndiProperty ip;                                                        \
            ip.setDevice("right");                                                       \
            ip.setName("wrong");                                                         \
            int rv = sdgt.XWCTEST_MAKE_INDI_CALLBACK(newCallBack_m_indiP_,propname)(ip); \
            REQUIRE(rv == -1);                                                           \
        }                                                                                \
        WHEN("Right Device.Name")                                                        \
        {                                                                                \
            testclass ## _test sdgt("right");                                            \
            pcf::IndiProperty ip;                                                        \
            ip.setDevice("right");                                                       \
            ip.setName( #propname );                                                     \
            int rv = sdgt.XWCTEST_MAKE_INDI_CALLBACK(newCallBack_m_indiP_,propname)(ip); \
            REQUIRE(rv == 0);                                                            \
        }                                                                                \
   }

#endif
