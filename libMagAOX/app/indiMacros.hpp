/** \file indiMacros.hpp
  * \brief Macros for INDI
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-05-27 created by JRM
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
#define REG_INDI_NEWPROP(prop, propName, type) registerIndiPropertyNew( prop, propName, type, pcf::IndiProperty::ReadWrite, pcf::IndiProperty::Idle, INDI_NEWCALLBACK(prop));

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
#define REG_INDI_NEWPROP_NOCB(prop, propName, type) registerIndiPropertyNew( prop, propName, type, pcf::IndiProperty::ReadOnly, pcf::IndiProperty::Idle, 0);

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
#define REG_INDI_SETPROP(prop, devName, propName) registerIndiPropertySet( prop,devName,  propName, INDI_SETCALLBACK(prop));

#endif //app_indiMacros_hpp
