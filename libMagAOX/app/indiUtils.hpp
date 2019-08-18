/** \file indiUtils.hpp
  * \brief MagAO-X INDI Utilities
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2019-01-03 created by JRM
  * 
  * \ingroup app_files
  */

#ifndef app_indiUtils_hpp
#define app_indiUtils_hpp

#include "../../INDI/libcommon/IndiProperty.hpp"
#include "../../INDI/libcommon/IndiElement.hpp"


namespace MagAOX
{
namespace app
{
namespace indi 
{

#define INDI_IDLE (pcf::IndiProperty::Ok)   
#define INDI_OK (pcf::IndiProperty::Ok)
#define INDI_BUSY (pcf::IndiProperty::Busy)

/// Update the value of the INDI element, but only if it has changed.
/** Only sends the set property message if the new value is different.
  *
  * \todo investigate how this handles floating point values and string conversions.
  * \todo this needs a const char specialization to std::string
  * 
  */  
template<typename T, class indiDriverT>
void updateIfChanged( pcf::IndiProperty & p,   ///< [in/out] The property containing the element to possibly update
                      const std::string & el,  ///< [in] The element name
                      const T & newVal,        ///< [in] the new value
                      indiDriverT * indiDriver, ///< [in] the MagAOX INDI driver to use
                      pcf::IndiProperty::PropertyStateType newState = pcf::IndiProperty::Ok
                    )
{
   if( !indiDriver ) return;
   
   T oldVal = p[el].get<T>();

   pcf::IndiProperty::PropertyStateType oldState = p.getState();
   
   if(oldVal != newVal || oldState != newState)
   {
      p[el].set(newVal);
      p.setState (newState);
      indiDriver->sendSetProperty (p);
   }
}

/// Update the value of the INDI element, but only if it has changed.
/** Only sends the set property message if the new value is different.
  *
  * \todo investigate how this handles floating point values and string conversions.
  * \todo this needs a const char specialization to std::string
  * 
  */  
template<class indiDriverT>
void updateSwitchIfChanged( pcf::IndiProperty & p,   ///< [in/out] The property containing the element to possibly update
                            const std::string & el,  ///< [in] The element name
                            const pcf::IndiElement::SwitchStateType & newVal,        ///< [in] the new value
                            indiDriverT * indiDriver, ///< [in] the MagAOX INDI driver to use
                            pcf::IndiProperty::PropertyStateType newState = pcf::IndiProperty::Ok
                          )
{
   if( !indiDriver ) return;
   
   pcf::IndiElement::SwitchStateType oldVal = p[el].getSwitchState();

   pcf::IndiProperty::PropertyStateType oldState = p.getState();
   
   if(oldVal != newVal || oldState != newState)
   {
      p[el].setSwitchState(newVal);
      p.setState (newState);
      indiDriver->sendSetProperty (p);
   }
}

} //namespace indi
} //namespace app
} //namespace MagAOX

#endif //app_magAOXIndiDriver_hpp
