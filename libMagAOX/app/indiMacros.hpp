/** \file indiMacros.hpp
  * \brief Macros for INDI
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-05-27 created by JRM
  * 
  * \ingroup app_files
  */

#ifndef app_indiMacros_hpp
#define app_indiMacros_hpp

/// Declare and define the static callback for a new property request.
/** You should not normally use this macro, it is called by INDI_NEWCALLBACK_DECL.
  *
  * \param class the class name (with no \")
  * \param prop the property member name (with no \")
  *
  * \ingroup indi
  */
#define SET_INDI_NEWCALLBACK(class, prop) static int st_ ## newCallBack ## _ ## prop( void * app, const pcf::IndiProperty &ipRecv)\
                                          {\
                                             return static_cast<class *>(app)->newCallBack ## _ ## prop(ipRecv);\
                                          }

/// Declare and define the static callback for a set property request.
/** You should not normally use this macro, it is called by INDI_SETCALLBACK_DECL.
  *
  * \param class the class name (with no \")
  * \param prop the property member name (with no \")
  *
  * \ingroup indi
  */
#define SET_INDI_SETCALLBACK(class, prop) static int st_ ## setCallBack ## _ ## prop( void * app, const pcf::IndiProperty &ipRecv)\
                                          {\
                                             return static_cast<class *>(app)->setCallBack ## _ ## prop(ipRecv);\
                                          }
                                          
/// Declare the callback for a new property request, and declare and define the static wrapper.
/** After including this, you still need to actually define the callback.
  *
  * \param class the class name (with no \")
  * \param prop the property member name (with no \")
  *
  *  \ingroup indi
  */
#define INDI_NEWCALLBACK_DECL(class, prop) int newCallBack_ ## prop(const pcf::IndiProperty &ipRecv); \
                                           SET_INDI_NEWCALLBACK(class, prop)

/// Declare the callback for a set property request, and declare and define the static wrapper.
/** After including this, you still need to actually define the callback.
  *
  * \param class the class name (with no \")
  * \param prop the property member name (with no \")
  *
  *  \ingroup indi
  */
#define INDI_SETCALLBACK_DECL(class, prop) int setCallBack_ ## prop(const pcf::IndiProperty &ipRecv); \
                                           SET_INDI_SETCALLBACK(class, prop)
                                           
/// Define the callback for a new property request.
/** Creates a class::method definition, which must be appended with a const reference of type pcf::IndiProperty.
  * Example usage for a class named xapp and an INDI property x:
  * \code
    INDI_NEWCALLBACK_DEFN(xapp, x)(const pcf::IndiProperty &ipRecv)
    {
      //do stuff with ipRecv

      return 0; //Must return int.
    }
    \endcode
   * After pre-processing the above code becomes
   * \code
    int xapp::newCallback_x(const pcf::IndiProperty &ipRecv)
    {
      //do stuff with ipRecv

      return 0; //Must return int.
    }
    \endcode
   *
   *
   * \param class the class name (with no \")
   * \param prop the property member name (with no \")
   *
   * \ingroup indi
   */
#define INDI_NEWCALLBACK_DEFN(class, prop) int class::newCallBack_ ## prop

/// Define the callback for a set property request.
/** Creates a class::method definition, which must be appended with a const reference of type pcf::IndiProperty.
  * Example usage for a class named xapp and an INDI property x:
  * \code
    INDI_SETCALLBACK_DEFN(xapp, x)(const pcf::IndiProperty &ipRecv)
    {
      //do stuff with ipRecv

      return 0; //Must return int.
    }
    \endcode
   * After pre-processing the above code becomes
   * \code
    int xapp::setCallback_x(const pcf::IndiProperty &ipRecv)
    {
      //do stuff with ipRecv

      return 0; //Must return int.
    }
    \endcode
   *
   *
   * \param class the class name (with no \")
   * \param prop the property member name (with no \")
   *
   * \ingroup indi
   */
#define INDI_SETCALLBACK_DEFN(class, prop) int class::setCallBack_ ## prop

#ifndef XWCTEST_INDI_CALLBACK_VALIDATION   
#define INDI_VALIDATE_LOG_ERROR(prop1, prop2)  log<software_error>({__FILE__,__LINE__, "INDI properties do not match in callback: "             \
                                                                  + prop1.createUniqueKey() + " != " + prop2.createUniqueKey()}); 

#define INDI_VALIDATE_LOG_ERROR_DERIVED(prop1, prop2) derivedT::template log<software_error>({__FILE__,__LINE__, "INDI properties do not match in callback: "             \
                                                                  + prop1.createUniqueKey() + " != " + prop2.createUniqueKey()}); 
#else
#define INDI_VALIDATE_LOG_ERROR(prop1, prop2)
#define INDI_VALIDATE_LOG_ERROR_DERIVED(prop1,prop2)
#endif                                                                                                           

/// Implementation of new callback INDI property validator for main class
/** \param prop1 [in] the first property to compare
  * \param prop2 [in] the second property to compare
  */
#define INDI_VALIDATE_CALLBACK_PROPS_IMPL(prop1, prop2)                                                                           \
        if(prop1.createUniqueKey() != prop2.createUniqueKey())                                                                    \
        {                                                                                                                         \
            INDI_VALIDATE_LOG_ERROR(prop1, prop2)                                                                                 \
            return -1;                                                                                                            \
        }

#ifdef XWCTEST_INDI_CALLBACK_VALIDATION

// When testing validation of callback checks, we add a return 0 to avoid executing the rest of the callback.
#define INDI_VALIDATE_CALLBACK_PROPS(prop1, prop2)  INDI_VALIDATE_CALLBACK_PROPS_IMPL(prop1, prop2) else {return 0;}

#else

/// Standard check for matching INDI properties in a callback
/** Uses makeUniqueKey() to check.
  *
  * Causes a return -1 on a mismatch.
  * 
  * Does nothing on a match.
  * 
  * If the test macro XWCTEST_INDI_CALLBACK_VALIDATION is defined this will cause return 0 on a match.
  * 
  * \param prop1 [in] the first property to compare
  * \param prop2 [in] the second property to compare
  */
#define INDI_VALIDATE_CALLBACK_PROPS(prop1, prop2)  INDI_VALIDATE_CALLBACK_PROPS_IMPL( prop1, prop2 )  

#endif


/// Implementation of new callback INDI property validator for derived class
/** \param prop1 [in] the first property to compare
  * \param prop2 [in] the second property to compare
  */
#define INDI_VALIDATE_CALLBACK_PROPS_DERIVED_IMPL(prop1, prop2)                                                                   \
        if(prop1.createUniqueKey() != prop2.createUniqueKey())                                                                    \
        {                                                                                                                         \
            INDI_VALIDATE_LOG_ERROR_DERIVED(prop1, prop2)                                                                         \
            return -1;                                                                                                            \
        }                                                                                                                         \

#ifdef XWCTEST_INDI_CALLBACK_VALIDATION

// When testing validation of callback checks, we add a return 0 to avoid executing the rest of the callback.
#define INDI_VALIDATE_CALLBACK_PROPS_DERIVED(prop1, prop2)  INDI_VALIDATE_CALLBACK_PROPS_DERIVED_IMPL(prop1, prop2) else {return 0;}

#else

/// Standard check for matching INDI properties in a callback in a CRTP base class
/** Uses makeUniqueKey() to check.
  *
  * Causes a return -1 on a mismatch.
  * 
  * Does nothing on a match.
  * 
  * If the test macro XWCTEST_INDI_CALLBACK_VALIDATION is defined this will cause return 0 on a match.
  * 
  * \param prop1 [in] the first property to compare
  * \param prop2 [in] the second property to compare
  */
#define INDI_VALIDATE_CALLBACK_PROPS_DERIVED(prop1, prop2)  INDI_VALIDATE_CALLBACK_PROPS_DERIVED_IMPL( prop1, prop2)  

#endif

/// Get the name of the static callback wrapper for a new property.
/** Useful for passing the pointer to the callback.
  *
  * \param prop the property member name (with no \")
  *
  * \ingroup indi
  */
#define INDI_NEWCALLBACK(prop) st_newCallBack_ ## prop

/// Get the name of the static callback wrapper for a set property.
/** Useful for passing the pointer to the callback.
  *
  * \param prop the property member name (with no \")
  *
  * \ingroup indi
  */
#define INDI_SETCALLBACK(prop) st_setCallBack_ ## prop

/// Register a NEW INDI property with the class, using the standard callback name.
/** Is a wrapper for MagAOXApp::registerIndiPropertyNew.
  *
  * \param prop the property member name, with no quotes
  * \param propName the property name, in quotes
  * \param type the property type, pcf::IndiProperty::Type
  * \param perm the property permissions, pcf::IndiProperty::PropertyPermType
  * \param state the property state, pcf::IndiProperty::PropertyStateType
  *
  * \ingroup indi
  */
#define REG_INDI_NEWPROP(prop, propName, type)                                                                                          \
if( registerIndiPropertyNew( prop, propName, type, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle, INDI_NEWCALLBACK(prop)) < 0)  \
{                                                                                                                                       \
    return log<software_error,-1>({__FILE__,__LINE__, "failed to register new property"});                                              \
}                                                                     


/// Register a NEW INDI property with the class, with no callback.
/** Is a wrapper for MagAOXApp::registerIndiPropertyNew with  NULL callback.
  *
  * \param prop the property member name, with no quotes
  * \param propName the property name, in quotes
  * \param type the property type, pcf::IndiProperty::Type
  * \param perm the property permissions, pcf::IndiProperty::PropertyPermType
  * \param state the property state, pcf::IndiProperty::PropertyStateType
  *
  * \ingroup indi
  */
#define REG_INDI_NEWPROP_NOCB(prop, propName, type)                                                               \
if( registerIndiPropertyNew( prop, propName, type, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle, 0) < 0)  \
{                                                                                                                 \
    return log<software_error,-1>({__FILE__,__LINE__, "failed to register read only property"});                  \
}                                    

/// Register a NEW INDI property with the class, with no callback, using the derived class
/** Is a wrapper for MagAOXApp::registerIndiPropertyNew with  NULL callback.
  *
  * \param prop the property member name, with no quotes
  * \param propName the property name, in quotes
  * \param type the property type, pcf::IndiProperty::Type
  * \param perm the property permissions, pcf::IndiProperty::PropertyPermType
  * \param state the property state, pcf::IndiProperty::PropertyStateType
  *
  * \ingroup indi
  */
#define REG_INDI_NEWPROP_NOCB_DERIVED(prop, propName, type)                                                                 \
if( derived().registerIndiPropertyNew( prop, propName, type, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle, 0) < 0)  \
{                                                                                                                           \
    return derivedT::template log<software_error,-1>({__FILE__,__LINE__, "failed to register read only property"});         \
}                                     

/// Register a SET INDI property with the class, using the standard callback name.
/** Is a wrapper for MagAOXApp::registerIndiPropertySet.
  *
  * \param prop the property member name, with no quotes
  * \param propName the property name, in quotes
  * \param type the property type, pcf::IndiProperty::Type
  * \param perm the property permissions, pcf::IndiProperty::PropertyPermType
  * \param state the property state, pcf::IndiProperty::PropertyStateType
  *
  * \ingroup indi
  */
#define REG_INDI_SETPROP(prop, devName, propName)                                          \
if( registerIndiPropertySet( prop,devName,  propName, INDI_SETCALLBACK(prop)) < 0)         \
{                                                                                          \
    return log<software_error,-1>({__FILE__,__LINE__, "failed to register set property"}); \
}                                                                                           

#define REG_INDI_SETPROP_DERIVED(prop, devName, propName)                                                     \
if( derived().template registerIndiPropertySet( prop,devName,  propName, INDI_SETCALLBACK(prop)) < 0)         \
{                                                                                                             \
    return derivedT::template log<software_error,-1>({__FILE__,__LINE__, "failed to register set property"}); \
}                                                                                           

/// Create and register a NEW INDI property as a standard number as float, using the standard callback name.
/** This wraps createStandardIndiNumber and registerIndiPropertyNew, with error checking.
  * \p prop will have elements "current" and "target".
  *
  * \param prop   [out] the property to create and setup
  * \param name   [in] the name of the property
  * \param min    [in] the minimum value for the elements, applied to both target and current
  * \param max    [in] the minimum value for the elements, applied to both target and current
  * \param step   [in] the step size for the elements, applied to both target and current
  * \param format [in] the _ value for the elements, applied to both target and current.  Set to "" to use the MagAO-X standard for type.
  * \param label  [in] [optional] the GUI label suggestion for this property
  * \param group  [in] [optional] the group for this property
  * 
  * \ingroup indi
  */
#define CREATE_REG_INDI_NEW_NUMBERF( prop, name, min, max, step, format, label, group)          \
    if( createStandardIndiNumber<float>( prop, name, min, max, step, format, label, group) < 0) \
    {                                                                                           \
        log<software_error>({__FILE__,__LINE__, "error from createStandardIndiNumber"});        \
        return -1;                                                                              \
    }                                                                                           \
    if( registerIndiPropertyNew( prop, INDI_NEWCALLBACK(prop)) < 0)                             \
    {                                                                                           \
        log<software_error>({__FILE__,__LINE__, "error from registerIndiPropertyNew"});         \
        return -1;                                                                              \
    }

/// Create and register a NEW INDI property as a standard number as int, using the standard callback name.
/** This wraps createStandardIndiNumber and registerIndiPropertyNew, with error checking
  * \p prop will have elements "current" and "target".
  *
  * \param prop   [out] the property to create and setup
  * \param name   [in] the name of the property
  * \param min    [in] the minimum value for the elements, applied to both target and current
  * \param max    [in] the minimum value for the elements, applied to both target and current
  * \param step   [in] the step size for the elements, applied to both target and current
  * \param format [in] the _ value for the elements, applied to both target and current.  Set to "" to use the MagAO-X standard for type.
  * \param label  [in] [optional] the GUI label suggestion for this property
  * \param group  [in] [optional] the group for this property
  * 
  * \ingroup indi
  */
#define CREATE_REG_INDI_NEW_NUMBERI( prop, name, min, max, step, format, label, group)          \
    if( createStandardIndiNumber<int>( prop, name, min, max, step, format, label, group) < 0)   \
    {                                                                                           \
        log<software_error>({__FILE__,__LINE__, "error from createStandardIndiNumber"});        \
        return -1;                                                                              \
    }                                                                                           \
    if( registerIndiPropertyNew( prop, INDI_NEWCALLBACK(prop)) < 0)                             \
    {                                                                                           \
        log<software_error>({__FILE__,__LINE__, "error from registerIndiPropertyNew"});         \
        return -1;                                                                              \
    }

/// Create and register a NEW INDI property as a standard number as unsigned int, using the standard callback name.
/** This wraps createStandardIndiNumber and registerIndiPropertyNew, with error checking
  * \p prop will have elements "current" and "target".
  *
  * \param prop   [out] the property to create and setup
  * \param name   [in] the name of the property
  * \param min    [in] the minimum value for the elements, applied to both target and current
  * \param max    [in] the minimum value for the elements, applied to both target and current
  * \param step   [in] the step size for the elements, applied to both target and current
  * \param format [in] the _ value for the elements, applied to both target and current.  Set to "" to use the MagAO-X standard for type.
  * \param label  [in] [optional] the GUI label suggestion for this property
  * \param group  [in] [optional] the group for this property
  * 
  * \ingroup indi
  */
#define CREATE_REG_INDI_NEW_NUMBERU( prop, name, min, max, step, format, label, group)               \
    if( createStandardIndiNumber<unsigned>( prop, name, min, max, step, format, label, group) < 0)   \
    {                                                                                                \
        log<software_error>({__FILE__,__LINE__, "error from createStandardIndiNumber"});             \
        return -1;                                                                                   \
    }                                                                                                \
    if( registerIndiPropertyNew( prop, INDI_NEWCALLBACK(prop)) < 0)                                  \
    {                                                                                                \
        log<software_error>({__FILE__,__LINE__, "error from registerIndiPropertyNew"});              \
        return -1;                                                                                   \
    }

/// Create and register a NEW INDI property as a standard toggle switch, using the standard callback name.
/** This wraps createStandardIndiToggleSw and registerIndiPropertyNew, with error checking
  *
  * \param prop the property member name, with no quotes
  * \param name he property name, in quotes
  * 
  * \ingroup indi
  */
#define CREATE_REG_INDI_NEW_TOGGLESWITCH( prop, name)                                 \
    if( createStandardIndiToggleSw( prop, name) < 0)                                       \
    {                                                                                      \
        log<software_error>({__FILE__,__LINE__, "error from createStandardIndiToggleSw"}); \
        return -1;                                                                         \
    }                                                                                      \
    if( registerIndiPropertyNew( prop, INDI_NEWCALLBACK(prop)) < 0)                        \
    {                                                                                      \
        log<software_error>({__FILE__,__LINE__, "error from registerIndiPropertyNew"});    \
        return -1;                                                                         \
    }

/// Create and register a read-only INDI property as a standard toggle switch, with no callback.
/** This wraps createStandardIndiToggleSw and registerIndiPropertyNew with null callback, with error checking
  *
  * \param prop the property member name, with no quotes
  * \param name he property name, in quotes
  * 
  * \ingroup indi
  */
#define CREATE_REG_INDI_NEW_TOGGLESWITCH_NOCB( prop, name)                                 \
    if( createStandardIndiToggleSw( prop, name) < 0)                                       \
    {                                                                                      \
        log<software_error>({__FILE__,__LINE__, "error from createStandardIndiToggleSw"}); \
        return -1;                                                                         \
    }                                                                                      \
    if( registerIndiPropertyNew( prop, nullptr) < 0)                                       \
    {                                                                                      \
        log<software_error>({__FILE__,__LINE__, "error from registerIndiPropertyNew"});    \
        return -1;                                                                         \
    }

/// Create and register a NEW INDI property as a standard request switch, using the standard callback name.
/** This wraps createStandardIndiRequestSw and registerIndiPropertyNew, with error checking
  *
  * \param prop the property member name, with no quotes
  * \param name he property name, in quotes
  * 
  * \ingroup indi
  */
#define CREATE_REG_INDI_NEW_REQUESTSWITCH( prop, name)                                      \
    if( createStandardIndiRequestSw( prop, name) < 0)                                       \
    {                                                                                       \
        log<software_error>({__FILE__,__LINE__, "error from createStandardIndiRequestSw"}); \
        return -1;                                                                          \
    }                                                                                       \
    if( registerIndiPropertyNew( prop, INDI_NEWCALLBACK(prop)) < 0)                         \
    {                                                                                       \
        log<software_error>({__FILE__,__LINE__, "error from registerIndiPropertyNew"});     \
        return -1;                                                                          \
    }

/// Create and register a NEW INDI property as a standard number as float, using the standard callback name, using the derived class.
/** This wraps createStandardIndiNumber and registerIndiPropertyNew, with error checking.
  * \p prop will have elements "current" and "target".
  *
  * \param prop     [out] the property to create and setup
  * \param name     [in] the name of the property
  * \param min      [in] the minimum value for the elements, applied to both target and current
  * \param max      [in] the minimum value for the elements, applied to both target and current
  * \param step     [in] the step size for the elements, applied to both target and current
  * \param format   [in] the _ value for the elements, applied to both target and current.  Set to "" to use the MagAO-X standard for type.
  * \param label    [in] the GUI label suggestion for this property
  * \param group    [in] the group for this property
  * 
  * \ingroup indi
  */
#define CREATE_REG_INDI_NEW_NUMBERF_DERIVED( prop, name, min, max, step, format, label, group)              \
    if( derived().template createStandardIndiNumber<float>( prop, name, min, max, step, format, label, group) < 0)   \
    {                                                                                                       \
        derivedT::template log<software_error>({__FILE__,__LINE__, "error from createStandardIndiNumber"}); \
        return -1;                                                                                          \
    }                                                                                                       \
    if( derived().template registerIndiPropertyNew( prop, INDI_NEWCALLBACK(prop)) < 0)                               \
    {                                                                                                       \
        derivedT::template log<software_error>({__FILE__,__LINE__, "error from registerIndiPropertyNew"});  \
        return -1;                                                                                          \
    }


/// Create and register a NEW INDI property as a standard number as int, using the standard callback name, using the derived class
/** This wraps createStandardIndiNumber and registerIndiPropertyNew, with error checking
  * \p prop will have elements "current" and "target".
  *
  * \param prop     [out] the property to create and setup
  * \param name     [in] the name of the property
  * \param min      [in] the minimum value for the elements, applied to both target and current
  * \param max      [in] the minimum value for the elements, applied to both target and current
  * \param step     [in] the step size for the elements, applied to both target and current
  * \param format   [in] the _ value for the elements, applied to both target and current.  Set to "" to use the MagAO-X standard for type.
  * \param label    [in] the GUI label suggestion for this property
  * \param group    [in] the group for this property
  * 
  * \ingroup indi
  */
#define CREATE_REG_INDI_NEW_NUMBERI_DERIVED( prop, name, min, max, step, format, label, group)              \
    if( derived().template createStandardIndiNumber<int>( prop, name, min, max, step, format, label, group) < 0)     \
    {                                                                                                       \
        derivedT::template log<software_error>({__FILE__,__LINE__, "error from createStandardIndiNumber"}); \
        return -1;                                                                                          \
    }                                                                                                       \
    if( derived().template registerIndiPropertyNew( prop, INDI_NEWCALLBACK(prop)) < 0)                               \
    {                                                                                                       \
        derivedT::template log<software_error>({__FILE__,__LINE__, "error from registerIndiPropertyNew"});  \
        return -1;                                                                                          \
    }

/// Create and register a NEW INDI property as a standard toggle switch, using the standard callback name, using the derived class
/** This wraps createStandardIndiToggleSw and registerIndiPropertyNew, with error checking
  *
  * \param prop the property member name, with no quotes
  * \param name he property name, in quotes
  * 
  * \ingroup indi
  */
#define CREATE_REG_INDI_NEW_TOGGLESWITCH_DERIVED( prop, name )                                                \
    if( derived().template createStandardIndiToggleSw( prop, name) < 0)                                                \
    {                                                                                                         \
        derivedT::template log<software_error>({__FILE__,__LINE__, "error from createStandardIndiToggleSw"}); \
        return -1;                                                                                            \
    }                                                                                                         \
    if( derived().template registerIndiPropertyNew( prop, INDI_NEWCALLBACK(prop)) < 0)                                 \
    {                                                                                                         \
        derivedT::template log<software_error>({__FILE__,__LINE__, "error from registerIndiPropertyNew"});    \
        return -1;                                                                                            \
    }

/// Create and register a NEW INDI property as a standard request switch, using the standard callback name, using the derived class
/** This wraps createStandardIndiRequestSw and registerIndiPropertyNew, with error checking
  *
  * \param prop the property member name, with no quotes
  * \param name he property name, in quotes
  * 
  * \ingroup indi
  */
#define CREATE_REG_INDI_NEW_REQUESTSWITCH_DERIVED( prop, name)                                                 \
    if( derived().template createStandardIndiRequestSw( prop, name) < 0)                                                \
    {                                                                                                          \
        derivedT::template log<software_error>({__FILE__,__LINE__, "error from createStandardIndiRequestSw"}); \
        return -1;                                                                                             \
    }                                                                                                          \
    if( derived().template registerIndiPropertyNew( prop, INDI_NEWCALLBACK(prop)) < 0)                                  \
    {                                                                                                          \
        derivedT::template log<software_error>({__FILE__,__LINE__, "error from registerIndiPropertyNew"});     \
        return -1;                                                                                             \
    }
#endif //app_indiMacros_hpp
