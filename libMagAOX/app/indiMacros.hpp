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
                                             return ((class *) app)->newCallBack ## _ ## prop(ipRecv);\
                                          }

/// Declare the callback for a new propery request, and declare and define the static wrapper.
/** After including this, you still need to actually define the callback.
  *
  * \param class the class name (with no \")
  * \param prop the property member name (with no \")
  *
  *  \ingroup indi
  */
#define INDI_NEWCALLBACK_DECL(class, prop) int newCallBack_ ## prop(const pcf::IndiProperty &ipRecv); \
                                           SET_INDI_NEWCALLBACK(class, prop)

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

/// Get the name of the static callback wrapper for a property.
/** Useful for passing the pointer to the callback.
  *
  * \param prop the property member name (with no \")
  *
  * \ingroup indi
  */
#define INDI_NEWCALLBACK(prop) st_newCallBack_ ## prop

/// Register an INDI property with the class, using the standard callback name.
/** Is a wrapper for MagAOXApp::registerIndiProperty.
  *
  * \param prop the property member name, with no quotes
  * \param propName the property name, in quotes
  * \param type the property type, pcf::IndiProperty::Type
  * \param perm the property permissions, pcf::IndiProperty::PropertyPermType
  * \param state the property state, pcf::IndiProperty::PropertyStateType
  *
  * \ingroup indi
  */
#define REG_INDI_PROP(prop, propName, type, perm, state) registerIndiProperty( prop, propName, type, perm, state, INDI_NEWCALLBACK(prop));

/// Register an INDI property with the class, with no callback.
/** Is a wrapper for MagAOXApp::registerIndiProperty.
  *
  * \param prop the property member name, with no quotes
  * \param propName the property name, in quotes
  * \param type the property type, pcf::IndiProperty::Type
  * \param perm the property permissions, pcf::IndiProperty::PropertyPermType
  * \param state the property state, pcf::IndiProperty::PropertyStateType
  *
  * \ingroup indi
  */
#define REG_INDI_PROP_NOCB(prop, propName, type, perm, state) registerIndiProperty( prop, propName, type, perm, state, 0);


#endif //app_indiMacros_hpp
