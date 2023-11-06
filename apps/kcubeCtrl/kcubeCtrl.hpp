/** \file kcubeCtrl.hpp
 * \brief The MagAO-X K-Cube Controller header file
 *
 * \ingroup kcubeCtrl_files
 */

#ifndef kcubeCtrl_hpp
#define kcubeCtrl_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "tmcController.hpp"

/** \defgroup kcubeCtrl
 * \brief The K-Cube Controller application
 *
 * <a href="../handbook/operating/software/apps/kcubeCtrl.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup kcubeCtrl_files
 * \ingroup kcubeCtrl
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X K-Cube Controller
/**
  * \ingroup kcubeCtrl
  */
class kcubeCtrl : public MagAOXApp<true>
{

    ///\todo needs logs and telems
    ///\todo needs error checking in callbacks
    ///\todo needs a set toggle to enable and go to 75V
    
    // Give the test harness access.
    friend class kcubeCtrl_test;

protected:
    /** \name Configurable Parameters
     *@{
     */

    // here add parameters which will be config-able at runtime
    
    ///@}

    tmcController m_kAxis1;
    bool m_axis1Enabled {false};

    tmcController m_kAxis2;
    bool m_axis2Enabled {false};

public:

    /// Default c'tor.
    kcubeCtrl();

    /// D'tor, declared and defined for noexcept.
    ~kcubeCtrl() noexcept
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

    /// Implementation of the FSM for kcubeCtrl.
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

    /** \name K-cube Interface
     *
     * @{
     */
    int modIdentify(){return 0;}

    ///@} 

    /** \name INDI
     * 
     * @{
     */

    pcf::IndiProperty m_indiP_axis1_identify;
    INDI_NEWCALLBACK_DECL(kcubeCtrl, m_indiP_axis1_identify);

    pcf::IndiProperty m_indiP_axis1_enable;
    INDI_NEWCALLBACK_DECL(kcubeCtrl, m_indiP_axis1_enable);

    pcf::IndiProperty m_indiP_axis1_voltage;
    INDI_NEWCALLBACK_DECL(kcubeCtrl, m_indiP_axis1_voltage);

    pcf::IndiProperty m_indiP_axis2_identify;
    INDI_NEWCALLBACK_DECL(kcubeCtrl, m_indiP_axis2_identify);

    pcf::IndiProperty m_indiP_axis2_enable;
    INDI_NEWCALLBACK_DECL(kcubeCtrl, m_indiP_axis2_enable);

    pcf::IndiProperty m_indiP_axis2_voltage;
    INDI_NEWCALLBACK_DECL(kcubeCtrl, m_indiP_axis2_voltage);

    ///@}
};

kcubeCtrl::kcubeCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
    m_powerMgtEnabled = true;
    return;
}

void kcubeCtrl::setupConfig()
{
    config.add("axis1.serial", "", "axis1.serial", argType::Required, "axis1", "serial", false, "string", "USB serial number");
    config.add("axis2.serial", "", "axis2.serial", argType::Required, "axis2", "serial", false, "string", "USB serial number");
}

int kcubeCtrl::loadConfigImpl(mx::app::appConfigurator &_config)
{
    std::string ser = m_kAxis1.serial();
    _config(ser, "axis1.serial");
    m_kAxis1.serial(ser);

    ser = m_kAxis2.serial();
    _config(ser, "axis2.serial");
    m_kAxis2.serial(ser);

    return 0;
}

void kcubeCtrl::loadConfig()
{
    loadConfigImpl(config);
}

int kcubeCtrl::appStartup()
{
    createStandardIndiRequestSw( m_indiP_axis1_identify, "axis1_identify");  
    if( registerIndiPropertyNew( m_indiP_axis1_identify, INDI_NEWCALLBACK(m_indiP_axis1_identify)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }

    createStandardIndiToggleSw( m_indiP_axis1_enable, "axis1_enable");  
    if( registerIndiPropertyNew( m_indiP_axis1_enable, INDI_NEWCALLBACK(m_indiP_axis1_enable)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }
    ///\todo if format is "" this crashes INDI startup... wtf
    createStandardIndiNumber<float>( m_indiP_axis1_voltage, "axis1_voltage", 0, 150, 1.0/32767, "%0.4f");  
    if( registerIndiPropertyNew( m_indiP_axis1_voltage, INDI_NEWCALLBACK(m_indiP_axis1_voltage)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }
    m_indiP_axis1_voltage["current"]=0;
    m_indiP_axis1_voltage["target"]=0;

    createStandardIndiRequestSw( m_indiP_axis2_identify, "axis2_identify");  
    if( registerIndiPropertyNew( m_indiP_axis2_identify, INDI_NEWCALLBACK(m_indiP_axis2_identify)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }

    createStandardIndiToggleSw( m_indiP_axis2_enable, "axis2_enable");  
    if( registerIndiPropertyNew( m_indiP_axis2_enable, INDI_NEWCALLBACK(m_indiP_axis2_enable)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }

    createStandardIndiNumber<float>( m_indiP_axis2_voltage, "axis2_voltage",0,150,1.0/32767, "%0.4f");  
    if( registerIndiPropertyNew( m_indiP_axis2_voltage, INDI_NEWCALLBACK(m_indiP_axis2_voltage)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }
    m_indiP_axis2_voltage["current"]=0;
    m_indiP_axis2_voltage["target"]=0;

    state(stateCodes::NODEVICE);

    return 0;
}

int kcubeCtrl::appLogic()
{
    if( state() == stateCodes::POWERON || state() == stateCodes::NODEVICE || state() == stateCodes::ERROR)
    {
        int rv1;
        {
            elevatedPrivileges elPriv(this);
            rv1 = m_kAxis1.open(false);
        }

        int rv2;
        {
            elevatedPrivileges elPriv(this);
            rv2 = m_kAxis2.open(false);
        }

        if(rv1 == 0 && rv2 == 0)
        {
            if(!stateLogged())
            {
                std::stringstream logs1;
                logs1 << "Axis-1 USB Device " << m_kAxis1.vendor() << ":" << m_kAxis1.product() << ":";
                        logs1 << m_kAxis1.serial() << " found";
                log<text_log>(logs1.str());

                std::stringstream logs2;
                logs2 << "Axis-2 USB Device " << m_kAxis2.vendor() << ":" << m_kAxis2.product() << ":";
                        logs2 << m_kAxis2.serial() << " found";
                log<text_log>(logs2.str());
            }

            state(stateCodes::NOTCONNECTED);
        }
        else if(rv1 == -3 && rv2 == -3)
        {
            state(stateCodes::NODEVICE);
            return 0;
        }
        else
        {
            if(rv1 == 0)
            {
                if(!stateLogged())
                {
                    std::stringstream logs1;
                    logs1 << "Axis-1 USB Device " << m_kAxis1.vendor() << ":" << m_kAxis1.product() << ":";
                        logs1 << m_kAxis1.serial() << " found";
                    log<text_log>(logs1.str());
                }
            }

            if(rv1 != 0)
            {
                log<software_error>({__FILE__, __LINE__, 0, rv1, "axis1 tmcController::open failed. "});
            }

            if(rv2 == 0)
            {
                if(!stateLogged())
                {
                    std::stringstream logs2;
                    logs2 << "Axis-2 USB Device " << m_kAxis2.vendor() << ":" << m_kAxis2.product() << ":";
                        logs2 << m_kAxis2.serial() << " found";
                    log<text_log>(logs2.str());
                }
            }

            if(rv2 != 0)
            {
                log<software_error>({__FILE__, __LINE__, 0, rv1, "axis2 tmcController::open failed. "});
            }

            state(stateCodes::ERROR);
            return 0;
        }        
    }

    if( state() == stateCodes::NOTCONNECTED )
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);

        int rv;
        {
            elevatedPrivileges elPriv(this);
            rv = m_kAxis1.connect();
        }

        if(rv < 0)
        {
            //if connect failed, and there is a device, then we have some other problem.
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;
            
            log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::connect failed. "});
            std::cerr << "tmcController::connectFailed\n";
            state(stateCodes::ERROR);
            return 0;
        }

        {
            elevatedPrivileges elPriv(this);
            rv = m_kAxis2.connect();
        }

        if(rv < 0)
        {
            //if connect failed, and there is a device, then we have some other problem.
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;
            
            log<software_error>({__FILE__, __LINE__, 0, rv, "axis2 tmcController::connect failed. "});
            
            state(stateCodes::ERROR);
            return 0;
        }

        state(stateCodes::CONNECTED);
    }

    if(state()==stateCodes::CONNECTED)
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);

        int rv;
        tmcController::HWInfo hwi;
        rv = m_kAxis1.hw_req_info(hwi);
        if( rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::hw_req_info failed. "});
            state(stateCodes::ERROR);
            return 0;
        }
        std::stringstream logs1;
        logs1 << "Axis-1 "; 
        hwi.dump(logs1);
        log<text_log>(logs1.str());

        rv = m_kAxis2.hw_req_info(hwi);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis2 tmcController::hw_req_info failed. "});
            state(stateCodes::ERROR);
            return 0;
        }

        std::stringstream logs2;
        logs2 << "Axis-2 "; 
        hwi.dump(logs2);
        log<text_log>(logs2.str());

        rv = m_kAxis1.mod_set_chanenablestate(0x01, tmcController::EnableState::disabled);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::mod_set_chanenablestate failed. "});
            state(stateCodes::ERROR);
            return 0;
        }
        m_axis1Enabled = false;

        rv = m_kAxis2.mod_set_chanenablestate(0x01, tmcController::EnableState::disabled);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis2 tmcController::mod_set_chanenablestate failed. "});
            state(stateCodes::ERROR);
            return 0;
        }
        m_axis2Enabled = false;

        //Setup the user interface
        rv = m_kAxis1.hw_stop_updatemsgs();
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::hw_stop_updatemsgs failed. "});
            state(stateCodes::ERROR);
            return 0;
        }

        tmcController::KMMIParams par;
        rv = m_kAxis1.kpz_req_kcubemmiparams(par);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::kpz_req_kcubemmiparams failed. "});
            state(stateCodes::ERROR);
            return 0;
        }

        par.DispBrightness = 0;

        rv = m_kAxis1.kpz_set_kcubemmiparams(par);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;


            log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::kpz_set_kcubemmiparams failed. "});
            state(stateCodes::ERROR);
            return 0;
        }

        rv = m_kAxis1.kpz_req_kcubemmiparams(par);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::kpz_req_kcubemmiparams failed. "});
            state(stateCodes::ERROR);
            return 0;
        }

        logs1.str("");
        logs1 << "Axis-1 ";
        par.dump(logs1);
        log<text_log>(logs1.str());

        rv = m_kAxis2.hw_stop_updatemsgs();
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis2 tmcController::hw_stop_updatemsgs failed. "});
            state(stateCodes::ERROR);
            return 0;
        }

        rv = m_kAxis2.kpz_req_kcubemmiparams(par);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis2 tmcController::kpz_req_kcubemmiparams failed. "});
            state(stateCodes::ERROR);
            return 0;
        }

        par.DispBrightness = 0;

        rv = m_kAxis2.kpz_set_kcubemmiparams(par);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;


            log<software_error>({__FILE__, __LINE__, 0, rv, "axis2 tmcController::kpz_set_kcubemmiparams failed. "});
            state(stateCodes::ERROR);
            return 0;
        }

        rv = m_kAxis2.kpz_req_kcubemmiparams(par);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis2 tmcController::kpz_req_kcubemmiparams failed. "});
            state(stateCodes::ERROR);
            return 0;
        }

        logs2.str("");
        logs2 << "Axis-2 ";
        par.dump(logs2);
        log<text_log>(logs2.str());

        //Get and set TPZ IO Settings, setting limit to 150 V
        //First reads current settings, and only updates the 150 V limit.
        tmcController::TPZIOSettings tios;
        rv = m_kAxis1.pz_req_tpz_iosettings(tios);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::pz_req_tpz_iosettings failed. "});
            state(stateCodes::ERROR);
            return 0;
        }
        tios.VoltageLimit = tmcController::VoltLimit::V150;
        
        rv = m_kAxis1.pz_set_tpz_iosettings(tios);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::pz_set_tpz_iosettings failed. "});
            state(stateCodes::ERROR);
            return 0;
        }

        rv = m_kAxis1.pz_req_tpz_iosettings(tios);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::pz_req_tpz_iosettings failed. "});
            state(stateCodes::ERROR);
            return 0;
        }
    
        logs1.str("");
        logs1 << "Axis-1 ";
        tios.dump(logs1);
        log<text_log>(logs1.str());

        rv = m_kAxis2.pz_req_tpz_iosettings(tios);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis2 tmcController::pz_req_tpz_iosettings failed. "});
            state(stateCodes::ERROR);
            return 0;
        }
        tios.VoltageLimit = tmcController::VoltLimit::V150;
        
        rv = m_kAxis2.pz_set_tpz_iosettings(tios);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis2 tmcController::pz_set_tpz_iosettings failed. "});
            state(stateCodes::ERROR);
            return 0;
        }

        rv = m_kAxis2.pz_req_tpz_iosettings(tios);
        if(rv < 0)
        {
            sleep(1); //wait to see if power state updates 
            if(m_powerState == 0) return 0;

            log<software_error>({__FILE__, __LINE__, 0, rv, "axis2 tmcController::pz_req_tpz_iosettings failed. "});
            state(stateCodes::ERROR);
            return 0;
        }
    
        logs2.str("");
        logs2 << "Axis-2 ";
        tios.dump(logs2);
        log<text_log>(logs2.str());

        state(stateCodes::READY);
    }

    if(state() == stateCodes::READY)
    {
        //Update Enable State Axis 1
        {// mutex lock scope
            //Get a lock if we can
            std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

            //but don't wait for it, just go back around.
            if(!lock.owns_lock()) return 0;

            if(m_axis1Enabled)
            {
                updateSwitchIfChanged(m_indiP_axis1_enable, "toggle", pcf::IndiElement::On, INDI_OK);
            }
            else
            {
                updateSwitchIfChanged(m_indiP_axis1_enable, "toggle", pcf::IndiElement::Off, INDI_IDLE);
            }
        }

        //Update Voltage Axis 1
        {
            std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

            //but don't wait for it, just go back around.
            if(!lock.owns_lock()) return 0;
        
            float ov;
            m_kAxis1.pz_req_outputvolts(ov);
            ov *= 150.0;

            if(m_axis1Enabled)
            {
                updateIfChanged(m_indiP_axis1_voltage, "current", ov, INDI_OK);
            }
            else
            {
                updateIfChanged(m_indiP_axis1_voltage, "current", ov, INDI_IDLE);
            }

        }

        //Update Enable State Axis 2
        {
            std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);
            if(!lock.owns_lock()) return 0;

            if(m_axis2Enabled)
            {
                updateSwitchIfChanged(m_indiP_axis2_enable, "toggle", pcf::IndiElement::On, INDI_OK);
            }
            else
            {
                updateSwitchIfChanged(m_indiP_axis2_enable, "toggle", pcf::IndiElement::Off, INDI_IDLE);
            }
        }

        //Update Voltage Axis 1
        {
            std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

            //but don't wait for it, just go back around.
            if(!lock.owns_lock()) return 0;
        
            float ov;
            m_kAxis2.pz_req_outputvolts(ov);
            ov *= 150.0;

            if(m_axis2Enabled)
            {
                updateIfChanged(m_indiP_axis2_voltage, "current", ov, INDI_OK);
            }
            else
            {
                updateIfChanged(m_indiP_axis2_voltage, "current", ov, INDI_IDLE);
            }

        }


        //Update the identify Toggles
        {   
            std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);
            if(!lock.owns_lock()) return 0;

            updateSwitchIfChanged(m_indiP_axis1_identify, "request", pcf::IndiElement::Off, INDI_IDLE);
            updateSwitchIfChanged(m_indiP_axis2_identify, "request", pcf::IndiElement::Off, INDI_IDLE);
        }

    }
    return 0;
}

int kcubeCtrl::appShutdown()
{
    return 0;
}

INDI_NEWCALLBACK_DEFN(kcubeCtrl, m_indiP_axis1_identify)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_axis1_identify, ipRecv);

    //switch is toggled to on
    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);
        updateSwitchIfChanged(m_indiP_axis1_identify, "request", pcf::IndiElement::On, INDI_BUSY);
        return m_kAxis1.mod_identify();
    }
   
    return 0;
}

INDI_NEWCALLBACK_DEFN(kcubeCtrl, m_indiP_axis1_enable)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_axis1_enable, ipRecv);

    //switch is toggled to on
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);
        m_axis1Enabled = true;
        return m_kAxis1.mod_set_chanenablestate(0x01, tmcController::EnableState::enabled);
    }
    else
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);
        m_axis1Enabled = false;
        return m_kAxis1.mod_set_chanenablestate(0x01, tmcController::EnableState::disabled);
    }
   
   return 0;
}


INDI_NEWCALLBACK_DEFN(kcubeCtrl, m_indiP_axis1_voltage)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_axis1_voltage, ipRecv);

    float target;
    indiTargetUpdate(m_indiP_axis1_voltage, target, ipRecv, true);

    if(target < 0) target = 0;
    else if(target > 150) target = 150;

    std::lock_guard<std::mutex> guard(m_indiMutex);
    
    m_kAxis1.pz_set_outputvolts(target/150.0);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(kcubeCtrl, m_indiP_axis2_identify)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_axis2_identify, ipRecv);

    //switch is toggled to on
    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);
        updateSwitchIfChanged(m_indiP_axis2_identify, "request", pcf::IndiElement::On, INDI_BUSY);
        return m_kAxis2.mod_identify();
    }
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(kcubeCtrl, m_indiP_axis2_enable)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_axis2_enable, ipRecv);

    //switch is toggled to on
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);
        m_axis2Enabled = true;
        return m_kAxis2.mod_set_chanenablestate(0x01, tmcController::EnableState::enabled);
    }
    else
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);
        m_axis2Enabled = false;
        return m_kAxis2.mod_set_chanenablestate(0x01, tmcController::EnableState::disabled);
    }
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(kcubeCtrl, m_indiP_axis2_voltage)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_axis2_voltage, ipRecv);

    float target;
    indiTargetUpdate(m_indiP_axis2_voltage, target, ipRecv, true);

    if(target < 0) target = 0;
    else if(target > 150) target = 150;

    std::lock_guard<std::mutex> guard(m_indiMutex);
    
    m_kAxis2.pz_set_outputvolts(target/150.0);
   
   return 0;
}

} // namespace app
} // namespace MagAOX

#endif // kcubeCtrl_hpp
