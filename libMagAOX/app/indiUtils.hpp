/** \file indiUtils.hpp
  * \brief MagAO-X INDI Utilities
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2019-01-03 created by JRM
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
   
template<typename T, class indiDriverT>
void updateIfChanged( pcf::IndiProperty & p,   ///< [in/out] The property containing the element to possibly update
                      const std::string & el,  ///< [in] The element name
                      const T & newVal,        ///< [in] the new value
                      indiDriverT * indiDriver ///< [in] the MagAOX INDI driver to use
                    )
{
   if( !indiDriver ) return;
   
   T oldVal = p[el].get<T>();

   if(oldVal != newVal)
   {
      p[el].set(newVal);
      p.setState (pcf::IndiProperty::Ok);
      indiDriver->sendSetProperty (p);
   }
}

} //namespace indi
} //namespace app
} //namespace MagAOX

#endif //app_magAOXIndiDriver_hpp
