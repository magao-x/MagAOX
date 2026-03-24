#ifndef cred2Ctrl_hpp
#define cred2Ctrl_hpp

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"
#include "cred2_sdk.h"   // The SDK you uploaded

namespace MagAOX {
namespace app {

class cred2Ctrl :
    public MagAOXApp<>,
    public dev::frameGrabber<cred2Ctrl>
{
    friend class dev::frameGrabber<cred2Ctrl>;

public:
    typedef MagAOXApp<> MagAOXAppT;

    // Camera handle from SDK
    void* m_camHandle { nullptr };

    // Dimensions must be compile‑time or config‑time constants
    int m_width  = 640;
    int m_height = 512;

public:
    cred2Ctrl();
    ~cred2Ctrl() noexcept;

    // MagAO‑X App API
    virtual void setupConfig() override;
    virtual void loadConfig() override;
    virtual int  appStartup() override;
    virtual int  appLogic() override;
    virtual int  whilePowerOff() override;
    virtual int  appShutdown() override;

    // FrameGrabber required interface
    int startAcquisition();
    int acquireAndCheckValid();   // callback pushes frames directly
    int loadImageIntoStream(void* dest);
    int reconfig();
    float fps() { return 0.0f; }  // SDK provides no FPS query

private:
    static void frameCallback(void* userctx, int16_t* frame);
    void handleFrame(int16_t* frame);
};

// ---------------------------------------------------------------------

inline cred2Ctrl::cred2Ctrl() :
    MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
}

inline cred2Ctrl::~cred2Ctrl() noexcept
{}

inline void cred2Ctrl::setupConfig()
{
    dev::frameGrabber<cred2Ctrl>::setupConfig(config);
}

inline void cred2Ctrl::loadConfig()
{
    dev::frameGrabber<cred2Ctrl>::loadConfig(config);
}

inline int cred2Ctrl::appStartup()
{
    // Open the camera (only camera 0 supported by SDK)
    m_camHandle = CRED2_open(0, nullptr, nullptr);
    if(!m_camHandle)
        return log<software_error, -1>({"Failed to open C-RED2"});

    if (dev::frameGrabber<cred2Ctrl>::appStartup() < 0)
        return log<software_critical, -1>({""});

    return 0;
}

inline int cred2Ctrl::appLogic()
{
    return dev::frameGrabber<cred2Ctrl>::appLogic();
}

inline int cred2Ctrl::whilePowerOff()
{
    return 0;
}

inline int cred2Ctrl::appShutdown()
{
    dev::frameGrabber<cred2Ctrl>::appShutdown();
    CRED2_close(m_camHandle);
    return 0;
}

// ---------------- USB interface ---------------------

inline int cred2Ctrl::startAcquisition()
{
    // Register callback-based acquisition
    int ok = CRED2_startAcquisition(
        m_camHandle,
        m_width,
        m_height,
        &cred2Ctrl::frameCallback,
        this
    );

    if (ok != 1)
        return log<software_error, -1>({"Failed to start acquisition"});

    return 0;
}

inline void cred2Ctrl::frameCallback(void* userctx, int16_t* frame)
{
    reinterpret_cast<cred2Ctrl*>(userctx)->handleFrame(frame);
}

inline void cred2Ctrl::handleFrame(int16_t* frame)
{
    // Store pointer for frameGrabber
    m_image_p = frame;

    // Mark new frame ready
    m_newFrame = true;
}

inline int cred2Ctrl::acquireAndCheckValid()
{
    // Callback already placed frame in m_image_p
    if (!m_newFrame)
        return 1;

    m_newFrame = false;
    return 0;
}

inline int cred2Ctrl::loadImageIntoStream(void* dest)
{
    std::memcpy(dest, m_image_p, m_width * m_height * sizeof(int16_t));
    return 0;
}

inline int cred2Ctrl::reconfig()
{
    CRED2_stopAcquisition(m_camHandle);
    return startAcquisition();
}

} // namespace app
} // namespace MagAOX

#endif