/** \file loPredCtrlAcc_native_observe.hpp
 * \brief Private observation-only entrypoint; refuses a configuration that could bypass native observation.
 */
#ifndef loPredCtrlAcc_native_observe_hpp
#define loPredCtrlAcc_native_observe_hpp
#ifndef DDSPC_SHADOW_ONLY
    #error "The native observation executable requires DDSPC_SHADOW_ONLY=yes"
#endif
#include "loPredCtrlAcc.hpp"

namespace MagAOX::app
{
/// Private executable identity with the actual loPredCtrlAcc implementation and stricter observation admission.
class loPredCtrlAcc_native_observe : public loPredCtrlAcc
{
  public:
    /// Load actual application settings and require explicit native observation.
    void loadConfig() override;

    /// Refuse startup without the validated observation-only configuration.
    int appStartup() override;
};
inline void loPredCtrlAcc_native_observe::loadConfig()
{
    if( loadConfigImpl( config ) != 0 || !m_nativeShadowEnabled || !m_nativeShadowConfigValid )
        m_shutdown = true;
}
inline int loPredCtrlAcc_native_observe::appStartup()
{
    return m_nativeShadowEnabled && m_nativeShadowConfigValid ? loPredCtrlAcc::appStartup() : -1;
}
} // namespace MagAOX::app
#endif
