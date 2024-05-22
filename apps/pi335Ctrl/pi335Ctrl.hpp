/** \file pi335Ctrl.hpp
 * \brief The MagAO-X XXXXXX header file
 *
 * \ingroup pi335Ctrl_files
 */

#ifndef pi335Ctrl_hpp
#define pi335Ctrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup pi335Ctrl
 * \brief The XXXXXXX application to do YYYYYYY
 *
 * <a href="..//apps_html/page_module_pi335Ctrl.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup pi335Ctrl_files
 * \ingroup pi335Ctrl
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X PI 335 Controller
/**
 * \ingroup pi335Ctrl
 */
class pi335Ctrl : public MagAOXApp<true>, public tty::usbDevice, public dev::ioDevice, public dev::dm<pi335Ctrl, float>, public dev::shmimMonitor<pi335Ctrl>, public dev::telemeter<pi335Ctrl>
{

    // Give the test harness access.
    friend class pi335Ctrl_test;

    friend class dev::dm<pi335Ctrl, float>;

    friend class dev::shmimMonitor<pi335Ctrl>;

    friend class dev::telemeter<pi335Ctrl>;

    typedef dev::telemeter<pi335Ctrl> telemeterT;

protected:
    /** \name Configurable Parameters
     *@{
     */

    float m_posTol{0.05}; ///< The tolerance for reporting a raw position rather than the setpoint.

    float m_homePos1{17.5}; ///< Home position of axis 1.  Default is 17.5
    float m_homePos2{17.5}; ///< Home position of axis 2.  Default is 17.5
    float m_homePos3{0.0};  ///< Home position of axis 2.  Default is 17.5

    ///@}

private:
    std::string m_ctrl;  ///< The controller connected.
    std::string m_stage; ///< The stage connected.
    int m_naxes{2};      ///< The number of axes, default is 2.  Max is 3.

    bool m_pos_3_sent{false};

    bool m_actuallyATZ{true};

protected:
    float m_min1{0};  ///< The minimum value for axis 1
    float m_max1{35}; ///< The maximum value for axis 1

    float m_min2{0};  ///< The minimum value for axis 2
    float m_max2{35}; ///< The maximum value for axis 2

    float m_min3{0}; ///< The minimum value for axis 3
    float m_max3{0}; ///< The maximum value for axis 3

    double m_homingStart{0};
    int m_homingState{0};

    int m_servoState{0};

    float m_pos1Set{0};
    float m_pos1{0};

    float m_pos2Set{0};
    float m_pos2{0};

    float m_pos3Set{0};
    float m_pos3{0};

    float m_sva1{0};
    float m_sva2{0};
    float m_sva3{0};

public:
    /// Default c'tor.
    pi335Ctrl();

    /// D'tor, declared and defined for noexcept.
    ~pi335Ctrl() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/);

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for pi335Ctrl.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     *
     */
    virtual int appShutdown();

    /// Test the connection to the device
    /** Uses the *IDN? query
     *
     * \returns 0 if the E227 is found
     * \returns -1 on an error
     */
    int testConnection();

    int initDM();

    /// Start the homing procedure
    /** Checks that servos are off (the manual says this doesn't matter),
     * and then begins homing by calling 'home_1()'.
     * Sets FSM state to homing.
     *
     * \returns 0 on success (from 'home_1()')
     * \returns -1 on error
     */
    int home();

    /// Get the status of homing on an axiz
    /** Homing is also autozero, ATZ.  This uses the ATZ? query to
     * check if the autozero procedure has completed on the given axis.
     *
     * \returns 0 if ATZ not complete
     * \returns 1 if ATZ complete
     * \returns -1 on an error
     */
    int homeState(int axis /**< [in] the axis to check, 0 or 1 */);

    /// Begin homing (ATZ) axis 1
    /** Repeats check that servos are off (the manual says this doesn't matter)
     * and checks that 'm_homingStatte' is 0.  Then sends 'ATZ 1 NaN'.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int home_1();

    /// Begin homing (ATZ) axis 2
    /** Repeats check that servos are off (the manual says this doesn't matter)
     * and checks that 'm_homingStatte' is 1.  Then sends 'ATZ 2 NaN'.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int home_2();

    /// Begin homing (ATZ) axis 3
    /** Repeats check that servos are off (the manual says this doesn't matter)
     * and checks that 'm_homingStatte' is 1.  Then sends 'ATZ 3 NaN'.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int home_3();

    int finishInit();

    /// Zero all commands on the DM
    /** This does not update the shared memory buffer.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int zeroDM();

    /// Send a command to the DM
    /** This is called by the shmim monitoring thread in response to a semaphore trigger.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int commandDM(void *curr_src);

    /// Release the DM, making it safe to turn off power.
    /** The application will be state READY at the conclusion of this.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int releaseDM();

    int setCom(const std::string &com);

    int setCom(const std::string &com,
               int axis);

    int setCom(const std::string &com,
               int axis,
               const std::string &arg);

    int getCom(std::string &resp,
               const std::string &com,
               int axis);

    int getPos(float &pos,
               int n);

    /// Get the open loop control value
    int getSva(float &sva,
               int n);

    /// Update the flat command and propagate
    /** Writes the new desired position to the flat shmim, which is then propagated via the dmcomb
     */
    int updateFlat(float absPos1, ///< [in] The new position of axis 1
                   float absPos2, ///< [in] The new position of axis 2
                   float absPos3  ///< [in] The new position of axis 3
    );

    /// Send command to device to move axis 1
    int move_1(float absPos);

    /// Send command to device to move axis 2
    int move_2(float absPos);

    /// Send command to device to move axis 3
    int move_3(float absPos);

protected:
    // declare our properties
    pcf::IndiProperty m_indiP_pos1;
    pcf::IndiProperty m_indiP_pos2;
    pcf::IndiProperty m_indiP_pos3;

public:
    INDI_NEWCALLBACK_DECL(pi335Ctrl, m_indiP_pos1);
    INDI_NEWCALLBACK_DECL(pi335Ctrl, m_indiP_pos2);
    INDI_NEWCALLBACK_DECL(pi335Ctrl, m_indiP_pos3);

    /** \name Telemeter Interface
     * @{
     */

    int checkRecordTimes();

    int recordTelem(const telem_pi335 *);

    int recordPI335(bool force = false);

    ///@}
};

pi335Ctrl::pi335Ctrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
    m_powerMgtEnabled = true;

    return;
}

void pi335Ctrl::setupConfig()
{
    dev::ioDevice::setupConfig(config);
    tty::usbDevice::setupConfig(config);
    dev::dm<pi335Ctrl, float>::setupConfig(config);

    TELEMETER_SETUP_CONFIG(config);

    config.add("stage.naxes", "", "stage.naxes", argType::Required, "stage", "naxes", false, "int", "Number of axes.  Default is 2.  Max is 3.");

    config.add("stage.homePos1", "", "stage.homePos1", argType::Required, "stage", "homePos1", false, "float", "Home position of axis 1.  Default is 17.5.");
    config.add("stage.homePos2", "", "stage.homePos2", argType::Required, "stage", "homePos2", false, "float", "Home position of axis 2.  Default is 17.5.");
    config.add("stage.homePos3", "", "stage.homePos3", argType::Required, "stage", "homePos3", false, "float", "Home position of axis 3.  Default is 17.5.");

    config.add("dm.calibRelDir", "", "dm.calibRelDir", argType::Required, "dm", "calibRelDir", false, "string", "Used to find the default calib directory.");
}

int pi335Ctrl::loadConfigImpl(mx::app::appConfigurator &_config)
{

    this->m_baudRate = B115200; // default for E727 controller.  Will be overridden by any config setting.

    int rv = tty::usbDevice::loadConfig(_config);

    if (rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    dev::ioDevice::loadConfig(_config);

    m_calibRelDir = "ttmpupil";
    config(m_calibRelDir, "dm.calibRelDir");

    dev::dm<pi335Ctrl, float>::loadConfig(_config);

    config(m_naxes, "stage.naxes");
    config(m_homePos1, "stage.homePos1");
    config(m_homePos2, "stage.homePos2");
    config(m_homePos3, "stage.homePos3");

    TELEMETER_LOAD_CONFIG(_config);

    return 0;
}

void pi335Ctrl::loadConfig()
{
    loadConfigImpl(config);
}

int pi335Ctrl::appStartup()
{
    if (state() == stateCodes::UNINITIALIZED)
    {
        log<text_log>("In appStartup but in state UNINITIALIZED.", logPrio::LOG_CRITICAL);
        return -1;
    }

    ///\todo promote usbDevice to dev:: and make this part of its appStartup
    // Get the USB device if it's in udev
    if (m_deviceName != "")
    {
        log<text_log>(std::string("USB Device ") + m_idVendor + ":" + m_idProduct + ":" + m_serial + " found in udev as " + m_deviceName);
    }

    ///\todo error checks here
    dev::dm<pi335Ctrl, float>::appStartup();
    shmimMonitor<pi335Ctrl>::appStartup();

    // set up the  INDI properties
    REG_INDI_NEWPROP(m_indiP_pos1, "pos_1", pcf::IndiProperty::Number);
    m_indiP_pos1.add(pcf::IndiElement("current"));
    m_indiP_pos1.add(pcf::IndiElement("target"));
    m_indiP_pos1["current"] = -99999;
    m_indiP_pos1["target"] = -99999;

    REG_INDI_NEWPROP(m_indiP_pos2, "pos_2", pcf::IndiProperty::Number);
    m_indiP_pos2.add(pcf::IndiElement("current"));
    m_indiP_pos2.add(pcf::IndiElement("target"));
    m_indiP_pos2["current"] = -99999;
    m_indiP_pos2["target"] = -99999;

    // Note: 3rd axis added in testConnection if it's found

    TELEMETER_APP_STARTUP;

    return 0;
}

int pi335Ctrl::appLogic()
{
    dev::dm<pi335Ctrl, float>::appLogic();
    shmimMonitor<pi335Ctrl>::appLogic();

    TELEMETER_APP_LOGIC;

    if (state() == stateCodes::POWERON)
    {
        if (!powerOnWaitElapsed())
        {
            return 0;
        }
        else
        {
            ///\todo promote usbDevice to dev:: and make this part of its appStartup
            // Get the USB device if it's in udev
            if (m_deviceName == "")
            {
                state(stateCodes::NODEVICE);
            }
            else
            {
                state(stateCodes::NOTCONNECTED);
            }
        }
    }

    ///\todo promote usbDevice to dev:: and make this part of its appLogic
    if (state() == stateCodes::NODEVICE)
    {
        int rv = tty::usbDevice::getDeviceName();
        if (rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES)
        {
            state(stateCodes::FAILURE);
            if (!stateLogged())
            {
                log<software_critical>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
            }
            return -1;
        }

        if (rv == TTY_E_DEVNOTFOUND || rv == TTY_E_NODEVNAMES)
        {
            state(stateCodes::NODEVICE);

            if (!stateLogged())
            {
                log<text_log>(std::string("USB Device ") + m_idVendor + ":" + m_idProduct + ":" + m_serial + " not found in udev");
            }
            return 0;
        }
        else
        {
            state(stateCodes::NOTCONNECTED);
            if (!stateLogged())
            {
                log<text_log>(std::string("USB Device ") + m_idVendor + ":" + m_idProduct + ":" + m_serial + " found in udev as " + m_deviceName);
            }
        }
    }

    if (state() == stateCodes::NOTCONNECTED)
    {
        int rv;
        { // scope for elPriv
            elevatedPrivileges elPriv(this);
            rv = connect();
        }

        if (rv < 0)
        {
            int nrv = tty::usbDevice::getDeviceName();
            if (nrv < 0 && nrv != TTY_E_DEVNOTFOUND && nrv != TTY_E_NODEVNAMES)
            {
                state(stateCodes::FAILURE);
                if (!stateLogged())
                    log<software_critical>({__FILE__, __LINE__, nrv, tty::ttyErrorString(nrv)});
                return -1;
            }

            if (nrv == TTY_E_DEVNOTFOUND || nrv == TTY_E_NODEVNAMES)
            {
                state(stateCodes::NODEVICE);

                if (!stateLogged())
                {
                    std::stringstream logs;
                    logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " no longer found in udev";
                    log<text_log>(logs.str());
                }
                return 0;
            }

            // if connect failed, and there is a device, then we have some other problem.
            state(stateCodes::FAILURE);
            if (!stateLogged())
                log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
            return -1;
        }

        if (testConnection() == 0)
        {
            state(stateCodes::CONNECTED);
        }
        else
        {
            return 0;
        }
    }

    if (state() == stateCodes::CONNECTED)
    {
        state(stateCodes::NOTHOMED);
    }

    if (state() == stateCodes::HOMING)
    {
        int ax = m_homingState + 1;

        int atz = homeState(ax);

        if (!(atz == 0 || atz == 1))
        {
            state(stateCodes::ERROR);
            log<software_error, -1>({__FILE__, __LINE__, "error getting ATZ? home state."});
        }

        if (atz == 1)
        {
            ++m_homingState;

            if (m_homingState == 1) // x complete
            {
                home_2();
            }
            else if (m_homingState == 2 && m_naxes == 2) // y complete, done
            {
                finishInit();
            }
            else if (m_homingState == 2 && m_naxes == 3) // y complete
            {
                home_3();
            }
            else if (m_homingState > 2)
            {
                finishInit();
            }
        }
    }

    if (state() == stateCodes::READY || state() == stateCodes::OPERATING)
    {
        if (m_flatSet)
            state(stateCodes::OPERATING);
        else
            state(stateCodes::READY);
    }

    if (state() == stateCodes::READY)
    {
        float pos1;
        float sva1;
        float pos2;
        float sva2;
        float pos3;
        float sva3;

        // Get a lock if we can
        std::unique_lock<std::mutex> lock(m_indiMutex);

        if (getPos(pos1, 1) < 0)
        {
            log<software_error>({__FILE__, __LINE__});
            state(stateCodes::ERROR);
            return 0;
        }

        lock.unlock();
        mx::sys::milliSleep(1); // Put this thread to sleep to make sure other thread gets a lock

        m_pos1 = pos1;
        if (fabs(m_pos1Set - m_pos1) > m_posTol)
        {
            updateIfChanged(m_indiP_pos1, "current", m_pos1, INDI_BUSY);
        }
        else
        {
            updateIfChanged(m_indiP_pos1, "current", m_pos1Set, INDI_IDLE);
        }

        lock.lock();
        if (getSva(sva1, 1) < 0)
        {
            log<software_error>({__FILE__, __LINE__});
            state(stateCodes::ERROR);
            return 0;
        }
        m_sva1 = sva1;

        lock.unlock();
        mx::sys::milliSleep(1); // Put this thread to sleep to make sure other thread gets a lock

        lock.lock();
        if (getPos(pos2, 2) < 0)
        {
            log<software_error>({__FILE__, __LINE__});
            state(stateCodes::ERROR);
            return 0;
        }
        lock.unlock();
        mx::sys::milliSleep(1); // Put this thread to sleep to make sure other thread gets a lock

        m_pos2 = pos2;
        if (fabs(m_pos2Set - m_pos2) > m_posTol) // sva2 != m_sva2)
        {
            updateIfChanged(m_indiP_pos2, "current", m_pos2, INDI_BUSY);
        }
        else
        {
            updateIfChanged(m_indiP_pos2, "current", m_pos2Set, INDI_IDLE);
        }

        lock.lock();
        if (getSva(sva2, 2) < 0)
        {
            log<software_error>({__FILE__, __LINE__});
            state(stateCodes::ERROR);
            return 0;
        }
        lock.unlock();
        mx::sys::milliSleep(1); // Put this thread to sleep to make sure other thread gets a lock

        m_sva2 = sva2;

        if (m_naxes == 3)
        {
            lock.lock();
            if (getPos(pos3, 3) < 0)
            {
                log<software_error>({__FILE__, __LINE__});
                state(stateCodes::ERROR);
                return 0;
            }
            lock.unlock();
            mx::sys::milliSleep(1); // Put this thread to sleep to make sure other thread gets a lock

            m_pos3 = pos3;
            if (fabs(m_pos3Set - m_pos3) > m_posTol)
            {
                updateIfChanged(m_indiP_pos3, "current", m_pos3, INDI_BUSY);
            }
            else
            {
                updateIfChanged(m_indiP_pos3, "current", m_pos3Set, INDI_IDLE);
            }

            lock.lock();
            if (getSva(sva3, 2) < 0)
            {
                log<software_error>({__FILE__, __LINE__});
                state(stateCodes::ERROR);
                return 0;
            }
            lock.unlock();
            mx::sys::milliSleep(1); // Put this thread to sleep to make sure other thread gets a lock

            m_sva3 = sva3;
        }

        recordPI335();

        /*std::cerr << m_pos1Set << " " << pos1 << " " << m_sva1 << " " << m_pos2Set << " " << pos2 << " " << m_sva2;
        if(m_naxes == 3) std::cerr << " " << m_pos3Set << " " << pos3 << " " << m_sva3;
        std::cerr << "\n";*/
    }
    else if (state() == stateCodes::OPERATING)
    {
        updateIfChanged<float>(m_indiP_pos1, std::vector<std::string>({"current", "target"}), {m_pos1, m_pos1Set}, INDI_BUSY);
        updateIfChanged<float>(m_indiP_pos2, std::vector<std::string>({"current", "target"}), {m_pos2, m_pos2Set}, INDI_BUSY);

        if (m_naxes == 3)
        {
            updateIfChanged<float>(m_indiP_pos3, std::vector<std::string>({"current", "target"}), {m_pos3, m_pos3Set}, INDI_BUSY);
        }
    }
    return 0;
}

int pi335Ctrl::appShutdown()
{
    dev::dm<pi335Ctrl, float>::appShutdown();
    shmimMonitor<pi335Ctrl>::appShutdown();
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int pi335Ctrl::testConnection()
{
    int rv;
    std::string resp;

    if ((rv = tty::ttyWriteRead(resp, "*IDN?\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout)) < 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    size_t st;
    if ((st = resp.find("E-727.3SDA")) == std::string::npos)
    {
        return log<text_log, -1>("Unknown device found: " + resp, logPrio::LOG_CRITICAL);
    }
    m_ctrl = mx::ioutils::removeWhiteSpace(resp.substr(st));
    log<text_log>(std::string("Connected to " + m_ctrl + " on ") + m_deviceName);

    std::string resp1, resp2, resp3;

    if ((rv = tty::ttyWriteRead(resp1, "CST? 1\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout)) < 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }
    resp1 = mx::ioutils::removeWhiteSpace(resp1);

    if ((rv = tty::ttyWriteRead(resp2, "CST? 2\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout)) < 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }
    resp2 = mx::ioutils::removeWhiteSpace(resp2);

    if ((rv = tty::ttyWriteRead(resp3, "CST? 3\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout)) < 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }
    resp3 = mx::ioutils::removeWhiteSpace(resp3);

    updateIfChanged(m_indiP_pos1, "current", 0.0);
    updateIfChanged(m_indiP_pos1, "target", 0.0);

    updateIfChanged(m_indiP_pos2, "current", 0.0);
    updateIfChanged(m_indiP_pos2, "target", 0.0);

    if (resp1.find("1=S-335") == 0 && resp2.find("2=S-335") == 0 && resp3.find("3=0") == 0)
    {
        m_stage = resp1.substr(2);
        m_naxes = 2;
        m_actuallyATZ = true;
    }
    else if (resp1.find("1=S-325") == 0 && resp2.find("2=S-325") == 0 && resp3.find("3=S-325") == 0)
    {
        m_stage = resp1.substr(2);
        m_naxes = 3;
        m_actuallyATZ = false;
        if (!m_pos_3_sent)
        {
            ///\todo this needs to only happen once, and then never again
            REG_INDI_NEWPROP(m_indiP_pos3, "pos_3", pcf::IndiProperty::Number);
            m_indiP_pos3.add(pcf::IndiElement("current"));
            m_indiP_pos3.add(pcf::IndiElement("target"));
            m_indiP_pos3["current"] = -99999999;
            m_indiP_pos3["target"] = -99999999;
            updateIfChanged(m_indiP_pos3, "current", 0.0);
            updateIfChanged(m_indiP_pos3, "target", 0.0);
            m_pos_3_sent = true;
        }
    }
    else
    {
        return log<text_log, -1>("Unknown stage found: " + resp1 + " " + resp2 + " " + resp3, logPrio::LOG_CRITICAL);
    }

    log<text_log>("Found " + m_stage + " with " + std::to_string(m_naxes) + " axes");

    //-------- now get axis limits

    // axis 1

    if ((rv = tty::ttyWriteRead(resp, "TMN? 1\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout)) < 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    if ((st = resp.find('=')) == std::string::npos)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, "invalid response"});
    }

    m_min1 = mx::ioutils::convertFromString<float>(resp.substr(st + 1));
    log<text_log>("axis 1 min: " + std::to_string(m_min1));

    if ((rv = tty::ttyWriteRead(resp, "TMX? 1\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout)) < 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    if ((st = resp.find('=')) == std::string::npos)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, "invalid response"});
    }

    m_max1 = mx::ioutils::convertFromString<float>(resp.substr(st + 1));
    log<text_log>("axis 1 max: " + std::to_string(m_max1));

    if ((rv = tty::ttyWriteRead(resp, "TMN? 2\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout)) < 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    if ((st = resp.find('=')) == std::string::npos)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, "invalid response"});
    }

    m_min2 = mx::ioutils::convertFromString<float>(resp.substr(st + 1));
    log<text_log>("axis 2 min: " + std::to_string(m_min2));

    if ((rv = tty::ttyWriteRead(resp, "TMX? 2\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout)) < 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    if ((st = resp.find('=')) == std::string::npos)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, "invalid response"});
    }

    m_max2 = mx::ioutils::convertFromString<float>(resp.substr(st + 1));
    log<text_log>("axis 2 max: " + std::to_string(m_max2));

    if (m_naxes == 3)
    {
        if ((rv = tty::ttyWriteRead(resp, "TMN? 3\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout)) < 0)
        {
            return log<software_critical, -1>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }

        if ((st = resp.find('=')) == std::string::npos)
        {
            return log<software_critical, -1>({__FILE__, __LINE__, "invalid response"});
        }

        m_min3 = mx::ioutils::convertFromString<float>(resp.substr(st + 1));
        log<text_log>("axis 3 min: " + std::to_string(m_min3));

        if ((rv = tty::ttyWriteRead(resp, "TMX? 3\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout)) < 0)
        {
            return log<software_critical, -1>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }

        if ((st = resp.find('=')) == std::string::npos)
        {
            return log<software_critical, -1>({__FILE__, __LINE__, "invalid response"});
        }

        m_max3 = mx::ioutils::convertFromString<float>(resp.substr(st + 1));
        log<text_log>("axis 3 max: " + std::to_string(m_max3));
    }

    m_flatCommand.resize(3, 1);
    if (m_naxes == 2)
    {
        m_flatCommand(0, 0) = m_homePos1;
        m_flatCommand(1, 0) = m_homePos2;
        m_flatCommand(2, 0) = 0.0;
    }
    else if (m_naxes == 3)
    {
        m_flatCommand(0, 0) = m_homePos1;
        m_flatCommand(1, 0) = m_homePos2;
        m_flatCommand(2, 0) = m_homePos3;
    }
    m_flatLoaded = true;

    return 0;
}

int pi335Ctrl::initDM()
{
    int rv;
    std::string resp;

    // get open-loop position of axis 1 (should be zero)
    rv = tty::ttyWriteRead(resp, "SVA? 1\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    // get open-loop position of axis 2 (should be zero)
    rv = tty::ttyWriteRead(resp, "SVA? 2\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    if (m_naxes == 3)
    {
        // get open-loop position of axis 2 (should be zero)
        rv = tty::ttyWriteRead(resp, "SVA? 3\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);

        if (rv < 0)
        {
            log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }
    }

    // make sure axis 1 has servo off
    rv = tty::ttyWrite("SVO 1 0\n", m_fileDescrip, m_writeTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    // make sure axis 2 has servo off
    rv = tty::ttyWrite("SVA 2 0\n", m_fileDescrip, m_writeTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    if (m_naxes == 0)
    {
        // make sure axis 3 has servo off
        rv = tty::ttyWrite("SVA 3 0\n", m_fileDescrip, m_writeTimeout);

        if (rv < 0)
        {
            log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }
    }

    m_servoState = 0;

    log<text_log>("servos off", logPrio::LOG_NOTICE);

    return home();
}

int pi335Ctrl::home()
{
    if (m_servoState != 0)
    {
        log<text_log>("home requested but servos are not off", logPrio::LOG_ERROR);
        return -1;
    }

    m_homingStart = 0;
    m_homingState = 0;

    state(stateCodes::HOMING);

    return home_1();
}

int pi335Ctrl::homeState(int axis)
{
    if (!m_actuallyATZ)
        return 1;
    std::string resp;

    if (getCom(resp, "ATZ?", axis) < 0)
    {
        log<software_error>({__FILE__, __LINE__});

        return -1;
    }

    ///\todo this should be a separate unit-tested parser
    size_t st = resp.find('=');
    if (st == std::string::npos || st > resp.size() - 2)
    {
        log<software_error>({__FILE__, __LINE__, "error parsing response"});
        return -1;
    }
    st += 1;

    return mx::ioutils::convertFromString<double>(resp.substr(st));
}

int pi335Ctrl::home_1()
{
    int rv;

    if (m_servoState != 0)
    {
        log<text_log>("home_1 requested but servos are not off", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_homingState != 0)
    {
        log<text_log>("home_1 requested but not in correct homing state", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_actuallyATZ)
    {
        // zero range found in axis 1 (NOTE this moves mirror full range) TAKES 1min
        rv = tty::ttyWrite("ATZ 1 NaN\n", m_fileDescrip, m_writeTimeout);

        if (rv < 0)
        {
            log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }
    }

    m_homingStart = mx::sys::get_curr_time(); ///\todo remmove m_homingStart once ATZ? works.
    m_homingState = 0;
    log<text_log>("commenced homing x");

    return 0;
}

int pi335Ctrl::home_2()
{
    int rv;

    if (m_servoState != 0)
    {
        log<text_log>("home_2 requested but servos are not off", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_homingState != 1)
    {
        log<text_log>("home_2 requested but not in correct homing state", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_actuallyATZ)
    {
        // zero range found in axis 2 (NOTE this moves mirror full range) TAKES 1min
        rv = tty::ttyWrite("ATZ 2 NaN\n", m_fileDescrip, m_writeTimeout);

        if (rv < 0)
        {
            log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }
    }

    m_homingStart = mx::sys::get_curr_time();
    log<text_log>("commenced homing y");

    return 0;
}

int pi335Ctrl::home_3()
{
    int rv;

    if (m_servoState != 0)
    {
        log<text_log>("home_3 requested but servos are not off", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_homingState != 2)
    {
        log<text_log>("home_3 requested but not in correct homing state", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_actuallyATZ)
    {
        // zero range found in axis 3 (NOTE this moves mirror full range) TAKES 1min
        rv = tty::ttyWrite("ATZ 3 NaN\n", m_fileDescrip, m_writeTimeout);

        if (rv < 0)
        {
            log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }
    }

    m_homingStart = mx::sys::get_curr_time();
    log<text_log>("commenced homing z");

    return 0;
}

int pi335Ctrl::finishInit()
{
    int rv;
    std::string resp;

    if (m_servoState != 0)
    {
        log<text_log>("finishInit requested but servos are not off", logPrio::LOG_ERROR);
        return -1;
    }

    if (m_naxes == 2 && m_homingState != 2)
    {
        log<text_log>("finishInit requested but not in correct homing state", logPrio::LOG_ERROR);
        return -1;
    }
    if (m_naxes == 3 && m_homingState != 3)
    {
        log<text_log>("finishInit requested but not in correct homing state", logPrio::LOG_ERROR);
        return -1;
    }

    // goto openloop pos zero (0 V) axis 1
    rv = tty::ttyWrite("SVA 1 0.0\n", m_fileDescrip, m_writeTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    mx::sys::milliSleep(2000);

    // goto openloop pos zero (0 V) axis 2
    rv = tty::ttyWrite("SVA 2 0.0\n", m_fileDescrip, m_writeTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    mx::sys::milliSleep(2000);

    if (m_naxes == 3)
    {
        // goto openloop pos zero (0 V) axis 3
        rv = tty::ttyWrite("SVA 3 0.0\n", m_fileDescrip, m_writeTimeout);

        if (rv < 0)
        {
            log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }

        mx::sys::milliSleep(2000);
    }

    // Get the real position of axis 1 (should be 0mrad st start)
    rv = tty::ttyWriteRead(resp, "SVA? 1\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    // Get the real position of axis 2 (should be 0mrad st start)
    rv = tty::ttyWriteRead(resp, "SVA? 2\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    if (m_naxes == 3)
    {
        // Get the real position of axis 3 (should be 0mrad st start)
        rv = tty::ttyWriteRead(resp, "SVA? 3\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);

        if (rv < 0)
        {
            log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }
    }

    // now safe to engage servos
    //(IMPORTANT:    NEVER EVER enable servos on axis 3 -- will damage S-335)
    //(CAVEAT: for S-325 you CAN enable servors on axis 3)

    // turn on servo to axis 1 (green servo LED goes on 727)
    rv = tty::ttyWrite("SVO 1 1\n", m_fileDescrip, m_writeTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    mx::sys::milliSleep(250);

    // turn on servo to axis 2 (green servo LED goes on 727)
    rv = tty::ttyWrite("SVO 2 1\n", m_fileDescrip, m_writeTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    if (m_naxes == 3)
    {
        mx::sys::milliSleep(250);

        // turn on servo to axis 3 (green servo LED goes on 727)
        rv = tty::ttyWrite("SVO 3 1\n", m_fileDescrip, m_writeTimeout);

        if (rv < 0)
        {
            log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }
    }

    m_servoState = 1;
    log<text_log>("servos engaged", logPrio::LOG_NOTICE);

    mx::sys::milliSleep(1000);

    // now safe for closed loop moves
    // center axis 1 (to configured home position)

    std::string com = "MOV 1 " + std::to_string(m_homePos1) + "\n";
    rv = tty::ttyWrite(com, m_fileDescrip, m_writeTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    m_pos1Set = m_homePos1;
    updateIfChanged(m_indiP_pos1, "target", m_homePos1);

    // center axis 2 (to configured home position)
    com = "MOV 2 " + std::to_string(m_homePos2) + "\n";
    rv = tty::ttyWrite(com, m_fileDescrip, m_writeTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    m_pos2Set = m_homePos2;
    updateIfChanged(m_indiP_pos2, "target", m_homePos2);

    if (m_naxes == 3)
    {
        // center axis 3 (to configured home position)
        com = "MOV 3 " + std::to_string(m_homePos3) + "\n";
        rv = tty::ttyWrite(com, m_fileDescrip, m_writeTimeout);

        if (rv < 0)
        {
            log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }

        m_pos3Set = m_homePos3;
        updateIfChanged(m_indiP_pos3, "target", m_homePos3);
    }

    state(stateCodes::READY);

    return 0;
}

int pi335Ctrl::zeroDM()
{
    move_1(0.0);
    move_2(0.0);
    if (m_naxes == 3)
        move_3(0.0);

    log<text_log>("DM zeroed");
    return 0;
}

int pi335Ctrl::commandDM(void *curr_src)
{
    if (state() != stateCodes::OPERATING)
        return 0;
    float pos1 = ((float *)curr_src)[0];
    float pos2 = ((float *)curr_src)[1];

    float pos3 = 0;
    if (m_naxes == 3)
        pos3 = ((float *)curr_src)[2];

    std::unique_lock<std::mutex> lock(m_indiMutex);

    int rv;
    if ((rv = move_1(pos1)) < 0)
        return rv;

    if ((rv = move_2(pos2)) < 0)
        return rv;

    if (m_naxes == 3)
        if ((rv = move_3(pos3)) < 0)
            return rv;

    return 0;
}

int pi335Ctrl::releaseDM()
{
    int rv;

    if (m_servoState != 0)
    {
        if ((rv = tty::ttyWrite("SVO 1 0\n", m_fileDescrip, m_writeTimeout)) < 0)
        {
            log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }

        if ((rv = tty::ttyWrite("SVO 2 0\n", m_fileDescrip, m_writeTimeout)) < 0)
        {
            log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }

        m_servoState = 0;

        log<text_log>("servos off", logPrio::LOG_NOTICE);
    }

    if ((rv = tty::ttyWrite("SVA 1 0\n", m_fileDescrip, m_writeTimeout)) < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    if ((rv = tty::ttyWrite("SVA 2 0\n", m_fileDescrip, m_writeTimeout)) < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    if (m_naxes == 3)
    {
        if ((rv = tty::ttyWrite("SVA 2 0\n", m_fileDescrip, m_writeTimeout)) < 0)
        {
            log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        }
    }

    m_flatSet = false;
    state(stateCodes::NOTHOMED);

    return 0;
}

int pi335Ctrl::getCom(std::string &resp,
                      const std::string &com,
                      int axis)
{
    std::string sendcom = com;
    if (axis == 1 || axis == 2)
    {
        sendcom += " ";
        sendcom += std::to_string(axis);
    }

    sendcom += "\n";

    int rv = tty::ttyWriteRead(resp, sendcom, "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);
    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
        return -1;
    }

    return 0;
}

int pi335Ctrl::getPos(float &pos,
                      int n)
{
    std::string resp;
    if (getCom(resp, "POS?", n) < 0)
    {
        log<software_error>({__FILE__, __LINE__});
    }

    ///\todo this should be a separate unit-tested parser
    size_t st = resp.find('=');
    if (st == std::string::npos || st > resp.size() - 2)
    {
        log<software_error>({__FILE__, __LINE__, "error parsing response"});
        return -1;
    }
    st += 1;
    pos = mx::ioutils::convertFromString<double>(resp.substr(st));

    return 0;
}

int pi335Ctrl::getSva(float &sva,
                      int n)
{
    std::string resp;
    if (getCom(resp, "SVA?", n) < 0)
    {
        log<software_error>({__FILE__, __LINE__});
    }

    ///\todo this should be a separate unit-tested parser
    size_t st = resp.find('=');
    if (st == std::string::npos || st > resp.size() - 2)
    {
        log<software_error>({__FILE__, __LINE__, "error parsing response"});
        return -1;
    }
    st += 1;
    sva = mx::ioutils::convertFromString<double>(resp.substr(st));

    return 0;
}

int pi335Ctrl::updateFlat(float absPos1,
                          float absPos2,
                          float absPos3)
{
    m_flatCommand(0, 0) = absPos1;
    m_flatCommand(1, 0) = absPos2;
    m_flatCommand(2, 0) = absPos3;

    if (state() == stateCodes::OPERATING)
    {
        return setFlat(true);
    }
    else
    {
        return 0;
    }
}

int pi335Ctrl::move_1(float absPos)
{
    int rv;

    if (absPos < m_min1 || absPos > m_max1)
    {
        log<text_log>("request move on azis 1 out of range", logPrio::LOG_ERROR);
        return -1;
    }

    m_pos1Set = absPos;

    std::string com = "MOV 1 " + std::to_string(absPos) + "\n";

    rv = tty::ttyWrite(com, m_fileDescrip, m_writeTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    return 0;
}

int pi335Ctrl::move_2(float absPos)
{
    int rv;

    if (absPos < m_min2 || absPos > m_max2)
    {
        log<text_log>("request move on azis 2 out of range", logPrio::LOG_ERROR);
        return -1;
    }

    m_pos2Set = absPos;
    std::string com = "MOV 2 " + std::to_string(absPos) + "\n";

    rv = tty::ttyWrite(com, m_fileDescrip, m_writeTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    return 0;
}

int pi335Ctrl::move_3(float absPos)
{
    if (m_naxes < 3)
    {
        return log<software_error, -1>({__FILE__, __LINE__, "tried to move axis 3 but we don't have that"});
    }

    int rv;

    if (absPos < m_min3 || absPos > m_max3)
    {
        log<text_log>("request move on azis 3 out of range", logPrio::LOG_ERROR);
        return -1;
    }

    m_pos3Set = absPos;
    std::string com = "MOV 3 " + std::to_string(absPos) + "\n";

    rv = tty::ttyWrite(com, m_fileDescrip, m_writeTimeout);

    if (rv < 0)
    {
        log<software_error>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN(pi335Ctrl, m_indiP_pos1)
(const pcf::IndiProperty &ipRecv)
{
    if (ipRecv.createUniqueKey() == m_indiP_pos1.createUniqueKey())
    {
        float current = -999999, target = -999999;

        if (ipRecv.find("current"))
        {
            current = ipRecv["current"].get<float>();
        }

        if (ipRecv.find("target"))
        {
            target = ipRecv["target"].get<float>();
        }

        if (target == -999999)
            target = current;

        if (target == -999999)
            return 0;

        if (state() == stateCodes::READY)
        {
            // Lock the mutex, waiting if necessary
            std::unique_lock<std::mutex> lock(m_indiMutex);

            updateIfChanged(m_indiP_pos1, "target", target);

            updateFlat(target, m_pos2Set, m_pos3Set); // This just changes the values, but doesn't move

            return move_1(target);
        }
        else if (state() == stateCodes::OPERATING)
        {
            return updateFlat(target, m_pos2Set, m_pos3Set);
        }
    }
    return -1;
}

INDI_NEWCALLBACK_DEFN(pi335Ctrl, m_indiP_pos2)
(const pcf::IndiProperty &ipRecv)
{
    if (ipRecv.createUniqueKey() == m_indiP_pos2.createUniqueKey())
    {
        float current = -999999, target = -999999;

        if (ipRecv.find("current"))
        {
            current = ipRecv["current"].get<float>();
        }

        if (ipRecv.find("target"))
        {
            target = ipRecv["target"].get<float>();
        }

        if (target == -999999)
            target = current;

        if (target == -999999)
            return 0;

        if (state() == stateCodes::READY)
        {
            // Lock the mutex, waiting if necessary
            std::unique_lock<std::mutex> lock(m_indiMutex);

            updateIfChanged(m_indiP_pos2, "target", target);
            updateFlat(m_pos1Set, target, m_pos3Set); // This just changes the values, but doesn't move
            return move_2(target);
        }
        else if (state() == stateCodes::OPERATING)
        {
            return updateFlat(m_pos1Set, target, m_pos3Set);
        }
    }
    return -1;
}

INDI_NEWCALLBACK_DEFN(pi335Ctrl, m_indiP_pos3)
(const pcf::IndiProperty &ipRecv)
{

    if (ipRecv.getName() == m_indiP_pos3.getName())
    {
        float current = -999999, target = -999999;

        if (ipRecv.find("current"))
        {
            current = ipRecv["current"].get<float>();
        }

        if (ipRecv.find("target"))
        {
            target = ipRecv["target"].get<float>();
        }

        if (target == -999999)
            target = current;

        if (target == -999999)
            return 0;

        if (state() == stateCodes::READY)
        {
            // Lock the mutex, waiting if necessary
            std::unique_lock<std::mutex> lock(m_indiMutex);

            updateIfChanged(m_indiP_pos3, "target", target);

            updateFlat(m_pos1Set, m_pos2Set, target); // This just changes the values, but doesn't move
            return move_3(target);
        }
        else if (state() == stateCodes::OPERATING)
        {
            return updateFlat(m_pos1Set, m_pos2Set, target);
        }
    }
    return -1;
}

int pi335Ctrl::checkRecordTimes()
{
    return telemeterT::checkRecordTimes(telem_pi335());
}

int pi335Ctrl::recordTelem(const telem_pi335 *)
{
    return recordPI335(true);
}

int pi335Ctrl::recordPI335(bool force)
{
    static float pos1Set = std::numeric_limits<float>::max();
    static float pos1 = std::numeric_limits<float>::max();
    static float sva1 = std::numeric_limits<float>::max();
    static float pos2Set = std::numeric_limits<float>::max();
    static float pos2 = std::numeric_limits<float>::max();
    static float sva2 = std::numeric_limits<float>::max();
    static float pos3Set = std::numeric_limits<float>::max();
    static float pos3 = std::numeric_limits<float>::max();
    static float sva3 = std::numeric_limits<float>::max();

    if (force || m_pos1Set != pos1Set || m_pos1 != pos1 || m_sva1 != sva1 ||
        m_pos2Set != pos2Set || m_pos2 != pos2 || m_sva2 != sva2 ||
        m_pos3Set != pos3Set || m_pos3 != pos3 || m_sva3 != sva3)
    {
        telem<telem_pi335>({m_pos1Set, m_pos1, m_sva1, m_pos2Set, m_pos2, m_sva2, m_pos3Set, m_pos3, m_sva3});

        pos1Set = m_pos1Set;
        pos1 = m_pos1;
        sva1 = m_sva1;
        pos2Set = m_pos2Set;
        pos2 = m_pos2;
        sva2 = m_sva2;
        pos3Set = m_pos3Set;
        pos3 = m_pos3;
        sva3 = m_sva3;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // pi335Ctrl_hpp
