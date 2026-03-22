/**
 * \file cred2Ctrl.hpp
 * \brief MagAO-X C-RED 2 USB camera controller application
 *
 * \ingroup cred2Ctrl_files
 */

#ifndef cred2Ctrl_hpp
#define cred2Ctrl_hpp

#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>
#include <cstring>

#include "cred2_sdk.h"   // Provided by user
#include "libMagAOX.hpp"
#include "stdCamera.hpp"

/**
 * \class cred2Ctrl
 * \brief MagAO-X application controlling a First Light Imaging C-RED 2 camera.
 *
 * This application uses the USB SDK callback mechanism and stores the latest frame
 * in a latch buffer. The MagAO-X FSM retrieves frames inside appLogic().
 */
class cred2Ctrl :
    public MagAOX::app::dev::stdCamera<cred2Ctrl>   // <-- Standard camera interface
{
public:
    cred2Ctrl();
    virtual ~cred2Ctrl() noexcept;

    // ---------- FSM Methods ----------

    /// Setup configuration declarations
    void setupConfig(mx::app::appConfigurator &config);

    /// Load configurables from file
    void loadConfig(mx::app::appConfigurator &config);

    /// Startup of device
    int appStartup();

    /// Main FSM loop: publish frame to MagAO-X shmim
    int appLogic();

    /// Actions when powered off
    int onPowerOff();

    /// Actions while powered off
    int whilePowerOff();

    /// Shutdown actions
    int appShutdown();

    // ---------- C-RED2 Callbacks ----------

    static void errorCallback(void *userctx, int error, const char *diag);
    static void frameCallback(void *userctx, int16_t *frame);

private:
    // C-RED2 handle
    void *m_ctx = nullptr;

    // Dimensions (loaded from configuration)
    int m_width = 640;
    int m_height = 512;

    // Latest-frame latch
    std::vector<uint16_t> m_frame;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    bool m_haveNewFrame = false;

    std::atomic<bool> m_running{false};
};

#endif