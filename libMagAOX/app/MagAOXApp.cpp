/** \file magAOXApp.cpp
  * \brief Instantiation of the basic MagAO-X Application
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2021-01-05 created by JRM
  * 
  * \ingroup app_files
  */

#include "MagAOXApp.hpp"

namespace MagAOX
{
namespace app
{

template<>
pcf::IndiProperty::Type propType<char *>()
{
   return pcf::IndiProperty::Text;
}

template<>
pcf::IndiProperty::Type propType<std::string>()
{
   return pcf::IndiProperty::Text;
}

template<>
pcf::IndiProperty::Type propType<int>()
{
   return pcf::IndiProperty::Number;
}

template<>
pcf::IndiProperty::Type propType<double>()
{
   return pcf::IndiProperty::Number;
}

template class MagAOXApp<true>;
template class MagAOXApp<false>;


}
}
