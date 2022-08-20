/** \file outletController.cpp
 * \author Jared R. Males
 * \brief Declares and defines a power control device framework in the MagAOXApp context
 *
 * \ingroup
 *
 */

#include "outletController.hpp"

namespace MagAOX
{
namespace app
{
namespace dev
{

std::string stateIntToString(int st)
{
   if( st == OUTLET_STATE_OFF ) return "Off";
   else if( st == OUTLET_STATE_INTERMEDIATE ) return "Int";
   else if( st == OUTLET_STATE_ON ) return "On";
   else return "Unk";
}

} //namespace dev
} //namespace app
} //namespace MagAOX

