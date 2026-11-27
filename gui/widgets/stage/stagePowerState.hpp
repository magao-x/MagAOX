/** \file stagePowerState.hpp
 * \brief FSM states in which a stage can report a retained parked position.
 */

#ifndef stagePowerState_hpp
#define stagePowerState_hpp

#include <string_view>

namespace xqt
{

/// Check whether a parked position is meaningful during power-off or controller startup.
inline bool stageParkedPositionState( std::string_view state /**< [in] Controller FSM state. */ )
{
    return state == "POWEROFF" || state == "POWERON" || state == "NODEVICE" || state == "NOTCONNECTED" ||
           state == "CONNECTED" || state == "UNINITIALIZED" || state == "INITIALIZED" || state == "LOGGEDIN" ||
           state == "CONFIGURING" || state == "NOTHOMED";
}

} // namespace xqt

#endif
