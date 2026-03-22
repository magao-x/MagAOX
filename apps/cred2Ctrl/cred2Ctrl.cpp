/**
 * \file cred2Ctrl.cpp
 * \brief MagAO-X C-RED2 camera application source file.
 *
 * \ingroup cred2Ctrl_files
 */

#include "cred2Ctrl.hpp"
#include <iostream>

cred2Ctrl::cred2Ctrl()
{
    // size controlled by config, resized in loadConfig()
}

cred2Ctrl::~cred2Ctrl() noexcept
{
    appShutdown();
}

void cred2Ctrl::setupConfig(mx::app::appConfigurator &config)
{
    config.add("camera.width",   m_width);
    config.add("camera.height",  m_height);

    // stdCamera setup
    this->stdCamera<cred2Ctrl>::setupConfig(config);
}

void cred2Ctrl::loadConfig(mx::app::appConfigurator &config)
{
    this->stdCamera<cred2Ctrl>::loadConfig(config);

    m_frame.resize(m_width * m_height);
}

int cred2Ctrl::appStartup()
{
    // Detect camera
    int n = CRED2_detect();
    if(n <= 0)
    {
        log<text_log>("No C-RED2 detected.");
        return MagAOX::app::stateCodes::ERROR;
    }

    // Open camera
    m_ctx = CRED2_open(0, &cred2Ctrl::errorCallback, this);
    if(!m_ctx)
    {
        log<text_log>("Failed to open C-RED2.");
        return MagAOX::app::stateCodes::ERROR;
    }

    // Start acquisition
    m_running = true;
    int ok = CRED2_startAcquisition(
        m_ctx, m_width, m_height,
        &cred2Ctrl::frameCallback, this
    );

    if(!ok)
    {
        log<text_log>("CRED2_startAcquisition failed.");
        return MagAOX::app::stateCodes::ERROR;
    }

    return MagAOX::app::stateCodes::READY;
}

int cred2Ctrl::appLogic()
{
    if(!m_running)
        return MagAOX::app::stateCodes::NOTCONNECTED;

    std::unique_lock<std::mutex> lock(m_mtx);
    m_cv.wait(lock, [&]{ return m_haveNewFrame; });

    m_haveNewFrame = false;

    // Publish to shmim using stdCamera's interface:
    // This depends on stdCamera providing recordCamera()
    recordCamera(true);

    return MagAOX::app::stateCodes::READY;
}

int cred2Ctrl::onPowerOff()
{
    return MagAOX::app::stateCodes::POWEROFF;
}

int cred2Ctrl::whilePowerOff()
{
    return MagAOX::app::stateCodes::POWEROFF;
}

int cred2Ctrl::appShutdown()
{
    m_running = false;
    if(m_ctx)
    {
        CRED2_stopAcquisition(m_ctx);
        CRED2_close(m_ctx);
        m_ctx = nullptr;
    }
    return MagAOX::app::stateCodes::SHUTDOWN;
}

void cred2Ctrl::errorCallback(void *userctx, int error, const char *diag)
{
    cred2Ctrl *self = static_cast<cred2Ctrl*>(userctx);
    self->log<text_log>("C-RED2 ERROR: " + std::to_string(error) + " " + diag);
}

void cred2Ctrl::frameCallback(void *userctx, int16_t *frame)
{
    cred2Ctrl *self = static_cast<cred2Ctrl*>(userctx);
    if(!self->m_running) return;

    std::unique_lock<std::mutex> lock(self->m_mtx);

    size_t N = self->m_width * self->m_height;
    for(size_t i = 0; i < N; i++)
        self->m_frame[i] = frame[i] < 0 ? 0u : (uint16_t)frame[i];

    self->m_haveNewFrame = true;
    self->m_cv.notify_one();
}

// -------- Main program file (template.cpp style) ---------

int main(int argc, char **argv)
{
    cred2Ctrl xapp;
    return xapp.main(argc, argv);
}