
#ifndef testMacrosINDI_hpp
#define testMacrosINDI_hpp

/// This turns on the success-return in the callback validator
/** See \ref INDI_VALIDATE_CALLBACK_PROPS and \ref INDI_VALIDATE_CALLBACK_PROPS_DERIVED
  *
  * \ingroup test_indi 
  */
#define XWCTEST_INDI_CALLBACK_VALIDATION

/// Make an indi property variable name
/** This takes m_indiP_ and voltage and creates m_indiP_voltage, which is the standard way
  * to name an INDI property which takes new requests.
  * 
  * \param stub [in] the first part of the variable name, e.g. m_indiP_ 
  * \param propname [in] the property name, e.g. voltage 
  * 
  * \ingroup test_indi
  */ 
#define XWCTEST_MAKE_INDI_PROP( stub,                \
                                propname             \
                              )                      \
                              stub ## propname

#define XWCTEST_SETUP_INDI_NEW_PROP( propname )                        \
   XWCTEST_MAKE_INDI_PROP(m_indiP_, propname).setDevice(m_configName); \
   XWCTEST_MAKE_INDI_PROP(m_indiP_, propname).setName(#propname);      \

#define XWCTEST_SETUP_INDI_ARB_PROP( varname, device, propname )                \
   varname.setDevice(#device);      \
   varname.setName(#propname);      \

#define XWCTEST_MAKE_INDI_CALLBACK( stub, callback) stub ## callback 

/// Catch-2 tests for whether a NEW callback properly validates the input property properly.
/**
  *  \param testclass [in] the name of class being tested
  *  \param propname  [in] the in-INDI name of the property 
  * 
  * \ingroup test_indi
  */  
#define XWCTEST_INDI_NEW_CALLBACK( testclass,                                                   \
                                   propname                                                     \
                                 )                                                              \
   GIVEN("A New Callback for " # propname  )                                                    \
   {                                                                                            \
        WHEN("Wrong Device")                                                                    \
        {                                                                                       \
            testclass ## _test sdgt("right");                                                   \
            pcf::IndiProperty ip;                                                               \
            ip.setDevice("wrong");                                                              \
            ip.setName( #propname );                                                            \
            int rv = sdgt.XWCTEST_MAKE_INDI_CALLBACK(newCallBack_m_indiP_,propname)(ip);        \
            REQUIRE(rv == -1);                                                                  \
        }                                                                                       \
        WHEN("Wrong name")                                                                      \
        {                                                                                       \
            testclass ## _test sdgt("right");                                                   \
            pcf::IndiProperty ip;                                                               \
            ip.setDevice("right");                                                              \
            ip.setName("wrong");                                                                \
            int rv = sdgt.XWCTEST_MAKE_INDI_CALLBACK(newCallBack_m_indiP_,propname)(ip);        \
            REQUIRE(rv == -1);                                                                  \
        }                                                                                       \
        WHEN("Right Device.Name")                                                               \
        {                                                                                       \
            testclass ## _test sdgt("right");                                                   \
            pcf::IndiProperty ip;                                                               \
            ip.setDevice("right");                                                              \
            ip.setName( #propname );                                                            \
            int rv = sdgt.XWCTEST_MAKE_INDI_CALLBACK(newCallBack_m_indiP_,propname)(ip);        \
            REQUIRE(rv == 0);                                                                   \
        }                                                                                       \
   }

/// Catch-2 tests for whether a SET callback properly validates the input property properly.
/**
  * \param testclass [in] the class being tested
  * \param varname   [in] the in-class variable name
  * \param device    [in] the device source of the property
  * \param propname  [in] the in-INDI name of the property
  * 
  * \ingroup test_indi
  */  
#define XWCTEST_INDI_SET_CALLBACK( testclass,                                                    \
                                   varname,                                                      \
                                   device,                                                       \
                                   propname                                                      \
                                 )                                                               \
   GIVEN("A Set Callback for " # propname  )                                                     \
   {                                                                                             \
        WHEN("Wrong Device")                                                                     \
        {                                                                                        \
            testclass ## _test sdgt("right");                                                    \
            pcf::IndiProperty ip;                                                                \
            ip.setDevice("wrong");                                                               \
            ip.setName( #propname );                                                             \
            int rv = sdgt.XWCTEST_MAKE_INDI_CALLBACK(setCallBack_,varname)(ip);                  \
            REQUIRE(rv == -1);                                                                   \
        }                                                                                        \
        WHEN("Wrong name")                                                                       \
        {                                                                                        \
            testclass ## _test sdgt("right");                                                    \
            pcf::IndiProperty ip;                                                                \
            ip.setDevice(#device);                                                               \
            ip.setName("wrong");                                                                 \
            int rv = sdgt.XWCTEST_MAKE_INDI_CALLBACK(setCallBack_,varname)(ip);                  \
            REQUIRE(rv == -1);                                                                   \
        }                                                                                        \
        WHEN("Right Device.Name")                                                                \
        {                                                                                        \
            testclass ## _test sdgt("right");                                                    \
            pcf::IndiProperty ip;                                                                \
            ip.setDevice(#device);                                                               \
            ip.setName( #propname );                                                             \
            int rv = sdgt.XWCTEST_MAKE_INDI_CALLBACK(setCallBack_,varname)(ip);                  \
            REQUIRE(rv == 0);                                                                    \
        }                                                                                        \
   }
#endif
