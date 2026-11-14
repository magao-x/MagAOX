/** \file loPredCtrlAcc_native.hpp
 * \brief Installed MagAO-X entrypoint with continuous input acquisition and mandatory output inhibition.
 */
#ifndef loPredCtrlAcc_native_hpp
#define loPredCtrlAcc_native_hpp
#ifndef DDSPC_SHADOW_ONLY
    #error "The installed loPredCtrlAcc application requires DDSPC_SHADOW_ONLY"
#endif
#include "loPredCtrlAcc.hpp"

namespace MagAOX::app
{
/// Normal MagAO-X lifecycle and identity; acquisition defaults on, recording and physical control default off.
class loPredCtrlAcc_native : public loPredCtrlAcc
{
    /// Check immutable native admission without granting any output or training authority.
    bool nativeConfigurationSafe() const;

  public:
    /// Establish safe native defaults before the framework loads the named application configuration.
    loPredCtrlAcc_native();

    /// Validate actual application settings and refuse every route out of native shadow acquisition.
    void loadConfig() override;

    /// Recheck safety even when startup is called without the framework configuration path.
    int appStartup() override;

    /// Refuse unsafe state after a configuration reload before any Legacy helper can run.
    int appLogic() override;

    /// Always shut down only native/shadow resources, including after a rejected configuration reload.
    int appShutdown() override;
};

inline loPredCtrlAcc_native::loPredCtrlAcc_native()
{
    m_shadowMode = m_adcEnabled = m_nativeShadowEnabled = true;
    m_ownShmim                                          = false;
    m_num_modes                                         = 2;
    m_nativeShadowConfig.m_durationSec                  = 0;
    m_nativeShadowConfig.m_record                       = false;
}

inline void loPredCtrlAcc_native::loadConfig()
{
    if( loadConfigImpl( config ) != 0 || !nativeConfigurationSafe() )
    {
        m_shutdown = true;
        log<software_error>( { __FILE__,
                               __LINE__,
                               "Installed native entrypoint requires ADC/native shadow acquisition and prohibits "
                               "Legacy, training and output ownership." } );
    }
}

inline int loPredCtrlAcc_native::appStartup()
{
    if( !nativeConfigurationSafe() )
        return -1;
    return loPredCtrlAcc::appStartup();
}

inline bool loPredCtrlAcc_native::nativeConfigurationSafe() const
{
    return m_shadowMode && m_adcEnabled && m_nativeShadowEnabled && m_nativeShadowConfigValid &&
           !m_shadowPipelineEnabled && !m_accelEnabled && !m_ownShmim && !m_isLearning && !m_isStdLearning &&
           !m_isPredictiveControl && !m_isIntegrating;
}

inline int loPredCtrlAcc_native::appLogic()
{
    return nativeConfigurationSafe() ? loPredCtrlAcc::appLogic() : -1;
}

inline int loPredCtrlAcc_native::appShutdown()
{
    m_shadowMode = m_nativeShadowEnabled = true;
    m_accelEnabled = m_ownShmim = false;
    m_isLearning = m_isStdLearning = m_isPredictiveControl = m_isIntegrating = false;
    return loPredCtrlAcc::appShutdown();
}
} // namespace MagAOX::app
#endif
