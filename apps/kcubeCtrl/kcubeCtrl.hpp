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

/// Local derivation of tmcController to implement MagAO-X logging
template<class parentT>
class tmcCon : public tmcController 
{
public:

    /// Print a message to MagAO-X logs describing an error from an \libftdi1 function
    /** Intended to be overriden in a derived class to provide custom error messaging.
      */
    virtual void ftdiErrmsg( const std::string & src,  ///< [in] The source of the error (the tmcController function)
                             const std::string & msg,  ///< [in] The message describing the error
                             int rv,                   ///< [in] The return value of the \libftdi1 function
                             const std::string & file, ///< [in] The file name of this file
                             int line                  ///< [in] The line number at which the error was recorded
                           )
    {
        std::stringstream logs;
        logs << src << ": " << msg << " [ libftdi1: " << ftdi_get_error_string(m_ftdi) << " ] ";
        uint32_t ln = line; //avoid narrowing warning
        parentT::template log<software_error>({file.c_str(), ln, 0, rv, logs.str()});
    }       

    /// Print a message to MagAO-X logs describing an error 
    /** Intended to be overriden in a derived class to provide custom error messaging.
      */
    virtual void otherErrmsg( const std::string & src,  ///< [in] The source of the error (the tmcController function)
                              const std::string & msg,  ///< [in] The message describing the error
                              const std::string & file, ///< [in] The file name of this file
                              int line                  ///< [in] The line number at which the error was recorded
                            )
    {
        uint32_t ln = line; //avoid narrowing warning
        parentT::template log<software_error>({file.c_str(), ln, src + ": " + msg});
    }

};

/// The MagAO-X K-Cube Controller
/**
  * \ingroup kcubeCtrl
  */
class kcubeCtrl : public MagAOXApp<true>
{

    ///\todo needs telems

    // Give the test harness access.
    friend class kcubeCtrl_test;

protected:
    /** \name Configurable Parameters
     *@{
     */

    // here add parameters which will be config-able at runtime
    
    ///@}

    tmcCon<kcubeCtrl> m_kAxis1;
    bool m_axis1Enabled {false};

    tmcCon<kcubeCtrl> m_kAxis2;
    bool m_axis2Enabled {false};

    bool m_isSet {false};

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

    int axis1Initialize();

    int axis1Enable();

    int axis1Disable();

    int axis1Voltage(float & v);

    int axis2Initialize();

    int axis2Enable();

    int axis2Disable();

    int axis2Voltage(float & v);

    int set();

    int rest(); 

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

    pcf::IndiProperty m_indiP_set;
    INDI_NEWCALLBACK_DECL(kcubeCtrl, m_indiP_set);

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

    createStandardIndiToggleSw( m_indiP_set, "set");  
    if( registerIndiPropertyNew( m_indiP_set, INDI_NEWCALLBACK(m_indiP_set)) < 0)
    {
        log<software_error>({__FILE__,__LINE__ - 2});
        return -1;
    }

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
                log<software_error>({__FILE__, __LINE__, 0, rv1, "axis 2 tmcController::open failed. "});
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
            if(m_powerState == 0) return -1;
            
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
            if(m_powerState == 0) return -1;
            
            log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::connect failed. "});
            
            state(stateCodes::ERROR);
            return 0;
        }

        state(stateCodes::CONNECTED);
    }

    if(state()==stateCodes::CONNECTED)
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);

        if(axis1Initialize() < 0)
        {
            return log<software_error, -1>({__FILE__,__LINE__, "error during axis 1 initialization"});
        }

        if(axis2Initialize() < 0)
        {
            return log<software_error, -1>({__FILE__,__LINE__, "error during axis 2 initialization"});
        }

        state(stateCodes::READY);
    }

    if(state() == stateCodes::READY || state() == stateCodes::OPERATING)
    {
        //We try_to_lock at each step, to let it go in case an actual command is waiting

        //Update Voltage Axis 1
        {
            std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);
            if(!lock.owns_lock()) return 0;
        
            float ov;
            int rv = m_kAxis1.pz_req_outputvolts(ov);
            if(rv < 0)
            {
                sleep(1); //wait to see if power state updates 
                if(m_powerState == 0) return -1;

                log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::pz_req_outputvolts failed. "});
                state(stateCodes::ERROR);
                return 0;
            }

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

        //Update Voltage Axis 2
        {
            std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

            //but don't wait for it, just go back around.
            if(!lock.owns_lock()) return 0;
        
            float ov;
            int rv = m_kAxis2.pz_req_outputvolts(ov);
            if(rv < 0)
            {
                sleep(1); //wait to see if power state updates 
                if(m_powerState == 0) return -1;

                log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::pz_req_outputvolts failed. "});
                state(stateCodes::ERROR);
                return 0;
            }

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


        //Update the Properties that don't need to talk to device here
        {   
            std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);
            if(!lock.owns_lock()) return 0;

            if(m_axis1Enabled)
            {
                updateSwitchIfChanged(m_indiP_axis1_enable, "toggle", pcf::IndiElement::On, INDI_OK);
            }
            else
            {
                updateSwitchIfChanged(m_indiP_axis1_enable, "toggle", pcf::IndiElement::Off, INDI_IDLE);
            }

            if(m_axis2Enabled)
            {
                updateSwitchIfChanged(m_indiP_axis2_enable, "toggle", pcf::IndiElement::On, INDI_OK);
            }
            else
            {
                updateSwitchIfChanged(m_indiP_axis2_enable, "toggle", pcf::IndiElement::Off, INDI_IDLE);
            }

            if(m_axis1Enabled && m_axis2Enabled)
            {
                state(stateCodes::OPERATING);
            }
            else
            {
                state(stateCodes::READY);
            }

            if(m_isSet)
            {
                updateSwitchIfChanged(m_indiP_set, "toggle", pcf::IndiElement::On, INDI_OK);
            }
            else
            {
                updateSwitchIfChanged(m_indiP_set, "toggle", pcf::IndiElement::Off, INDI_IDLE);
            }

            updateSwitchIfChanged(m_indiP_axis1_identify, "request", pcf::IndiElement::Off, INDI_IDLE);
            updateSwitchIfChanged(m_indiP_axis2_identify, "request", pcf::IndiElement::Off, INDI_IDLE);
        }

    } //READY || OPERATING

    return 0;
}

int kcubeCtrl::appShutdown()
{
    return 0;
}

int kcubeCtrl::axis1Initialize()
{
    int rv;

    tmcController::HWInfo hwi;
    
    rv = m_kAxis1.hw_req_info(hwi);
    if( rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::hw_req_info failed. "});
        return -1;
    }
    
    std::stringstream logs1;
    logs1 << "Axis-1 "; 
    hwi.dump(logs1);
    log<text_log>(logs1.str());

    rv = m_kAxis1.mod_set_chanenablestate(0x01, tmcController::EnableState::disabled);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::mod_set_chanenablestate failed. "});
        return -1;
    }
    m_axis1Enabled = false;
    m_isSet = false;

    //Setup the user interface
    rv = m_kAxis1.hw_stop_updatemsgs();
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::hw_stop_updatemsgs failed. "});
        return -1;
    }

    tmcController::KMMIParams par;
    rv = m_kAxis1.kpz_req_kcubemmiparams(par);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::kpz_req_kcubemmiparams failed. "});
        return -1;
    }

    par.DispBrightness = 0;

    rv = m_kAxis1.kpz_set_kcubemmiparams(par);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::kpz_set_kcubemmiparams failed. "});
        return -1;
    }

    rv = m_kAxis1.kpz_req_kcubemmiparams(par);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::kpz_req_kcubemmiparams failed. "});
        return -1;
    }

    logs1.str("");
    logs1 << "Axis-1 ";
    par.dump(logs1);
    log<text_log>(logs1.str());

    //Get and set TPZ IO Settings, setting limit to 150 V
    //First reads current settings, and only updates the 150 V limit.
    tmcController::TPZIOSettings tios;
    rv = m_kAxis1.pz_req_tpz_iosettings(tios);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::pz_req_tpz_iosettings failed. "});
        return -1;
    }
    tios.VoltageLimit = tmcController::VoltLimit::V150;
    
    rv = m_kAxis1.pz_set_tpz_iosettings(tios);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::pz_set_tpz_iosettings failed. "});
        return -1;
    }

    rv = m_kAxis1.pz_req_tpz_iosettings(tios);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::pz_req_tpz_iosettings failed. "});
        return -1;
    }
    
    logs1.str("");
    logs1 << "Axis-1 ";
    tios.dump(logs1);
    log<text_log>(logs1.str());

    return 0;
}

int kcubeCtrl::axis1Enable()
{
    int rv = m_kAxis1.mod_set_chanenablestate(0x01, tmcController::EnableState::enabled);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::mod_set_chanenablestate failed. "});
        return -1;
    }

    m_axis1Enabled = true;

    log<text_log>("enabled axis 1 piezo", logPrio::LOG_NOTICE);

    return 0;
}

int kcubeCtrl::axis1Disable()
{
    int rv = m_kAxis1.mod_set_chanenablestate(0x01, tmcController::EnableState::disabled);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::mod_set_chanenablestate failed. "});
        return -1;
    }

    m_axis1Enabled = false;
    m_isSet = false;

    log<text_log>("disabled axis 1 piezo", logPrio::LOG_NOTICE);

    return 0;
}

int kcubeCtrl::axis1Voltage(float & v)
{
    if(v < 0) 
    {
        log<text_log>("axis 1 voltage clamped at 0 (" + std::to_string(v) + ")", logPrio::LOG_WARNING);
        v = 0;
    }
    else if(v > 150)
    {
        log<text_log>("axis 1 voltage clamped at 150 (" + std::to_string(v) + ")", logPrio::LOG_WARNING);
        v = 150;
    }
    
    int rv = m_kAxis1.pz_set_outputvolts(v/150.0);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis1 tmcController::pz_set_outputvolts failed. "});
        return -1;
    }

    return 0;

}

int kcubeCtrl::axis2Initialize()
{
    int rv;

    tmcController::HWInfo hwi;
    
    rv = m_kAxis2.hw_req_info(hwi);
    if( rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::hw_req_info failed. "});
        return -1;
    }
    
    std::stringstream logs1;
    logs1 << "Axis-2 "; 
    hwi.dump(logs1);
    log<text_log>(logs1.str());

    rv = m_kAxis2.mod_set_chanenablestate(0x01, tmcController::EnableState::disabled);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::mod_set_chanenablestate failed. "});
        return -1;
    }
    m_axis2Enabled = false;
    m_isSet = false;

    //Setup the user interface
    rv = m_kAxis2.hw_stop_updatemsgs();
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::hw_stop_updatemsgs failed. "});
        return -1;
    }

    tmcController::KMMIParams par;
    rv = m_kAxis2.kpz_req_kcubemmiparams(par);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::kpz_req_kcubemmiparams failed. "});
        return -1;
    }

    par.DispBrightness = 0;

    rv = m_kAxis2.kpz_set_kcubemmiparams(par);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;


        log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::kpz_set_kcubemmiparams failed. "});
        return -1;
    }

    rv = m_kAxis2.kpz_req_kcubemmiparams(par);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::kpz_req_kcubemmiparams failed. "});
        return -1;
    }

    logs1.str("");
    logs1 << "Axis-2 ";
    par.dump(logs1);
    log<text_log>(logs1.str());

    //Get and set TPZ IO Settings, setting limit to 150 V
    //First reads current settings, and only updates the 150 V limit.
    tmcController::TPZIOSettings tios;
    rv = m_kAxis2.pz_req_tpz_iosettings(tios);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::pz_req_tpz_iosettings failed. "});
        return -1;
    }
    tios.VoltageLimit = tmcController::VoltLimit::V150;
    
    rv = m_kAxis2.pz_set_tpz_iosettings(tios);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::pz_set_tpz_iosettings failed. "});
        return -1;
    }

    rv = m_kAxis2.pz_req_tpz_iosettings(tios);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::pz_req_tpz_iosettings failed. "});
        return -1;
    }
    
    logs1.str("");
    logs1 << "Axis-2 ";
    tios.dump(logs1);
    log<text_log>(logs1.str());

    return 0;
}

int kcubeCtrl::axis2Enable()
{
    int rv = m_kAxis2.mod_set_chanenablestate(0x01, tmcController::EnableState::enabled);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::mod_set_chanenablestate failed. "});
        return -1;
    }

    m_axis2Enabled = true;

    log<text_log>("enabled axis 2 piezo", logPrio::LOG_NOTICE);

    return 0;
}

int kcubeCtrl::axis2Disable()
{
    int rv = m_kAxis2.mod_set_chanenablestate(0x01, tmcController::EnableState::disabled);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::mod_set_chanenablestate failed. "});
        return -1;
    }

    m_axis2Enabled = false;
    m_isSet = false;

    log<text_log>("disabled axis 2 piezo", logPrio::LOG_NOTICE);

    return 0;
}

int kcubeCtrl::axis2Voltage(float & v)
{
    if(v < 0) 
    {
        log<text_log>("axis 2 voltage clamped at 0 (" + std::to_string(v) + ")", logPrio::LOG_WARNING);
        v = 0;
    }
    else if(v > 150)
    {
        log<text_log>("axis 2 voltage clamped at 150 (" + std::to_string(v) + ")", logPrio::LOG_WARNING);
        v = 150;
    }
    
    int rv = m_kAxis2.pz_set_outputvolts(v/150.0);
    if(rv < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        log<software_error>({__FILE__, __LINE__, 0, rv, "axis 2 tmcController::pz_set_outputvolts failed. "});
        return -1;
    }

    return 0;
}

int kcubeCtrl::set()
{
    if(m_isSet) return 0;

    float v = 75.0;

    if(axis1Enable() < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        return log<software_error,-1>({__FILE__, __LINE__, "axis 1 enable error in set"});
    }

    if(axis1Voltage(v) < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        return log<software_error,-1>({__FILE__, __LINE__, "axis 1 enable voltage in set"});
    }

    if(axis2Enable() < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        return log<software_error,-1>({__FILE__, __LINE__, "axis 2 enable error in set"});
    }

    if(axis2Voltage(v) < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        return log<software_error,-1>({__FILE__, __LINE__, "axis 2 enable voltage in set"});
    }

    m_isSet = true;

    log<text_log>("set", logPrio::LOG_NOTICE);

    return 0;
}

int kcubeCtrl::rest()
{
    if(!m_isSet) return 0;

    float v = 0.0;

    if(axis1Voltage(v) < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        return log<software_error,-1>({__FILE__, __LINE__, "axis 1 enable voltage in rest"});
    }

    if(axis1Disable() < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        return log<software_error,-1>({__FILE__, __LINE__, "axis 1 disable error in rest"});
    }

    if(axis2Voltage(v) < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        return log<software_error,-1>({__FILE__, __LINE__, "axis 2 enable voltage in rest"});
    }

    if(axis2Disable() < 0)
    {
        sleep(1); //wait to see if power state updates 
        if(m_powerState == 0) return -1;

        return log<software_error,-1>({__FILE__, __LINE__, "axis 2 disable error in rest"});
    }

    m_isSet = false;

    log<text_log>("rested", logPrio::LOG_NOTICE);

    return 0;
}

INDI_NEWCALLBACK_DEFN(kcubeCtrl, m_indiP_axis1_identify)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_axis1_identify, ipRecv);

    if(state() != stateCodes::READY) return 0;

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

    if(!(state() == stateCodes::READY || state() == stateCodes::OPERATING) ) return 0;
    
    //switch is toggled to on
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);
        if(axis1Enable() < 0)
        {
            if(m_powerState == 0) return 0;
            return log<software_error,-1>({__FILE__, __LINE__, "axis 1 enable error in INDI callback"});
        }
    }
    else
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);
        if(axis1Disable() < 0)
        {
            if(m_powerState == 0) return 0;
            log<software_error,-1>({__FILE__, __LINE__, "axis 1 disable error in INDI callback"});
        }
    }
   
   return 0;
}


INDI_NEWCALLBACK_DEFN(kcubeCtrl, m_indiP_axis1_voltage)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_axis1_voltage, ipRecv);

    if(!(state() == stateCodes::READY || state() == stateCodes::OPERATING) ) return 0;

    float target;
    indiTargetUpdate(m_indiP_axis1_voltage, target, ipRecv, true);

    std::lock_guard<std::mutex> guard(m_indiMutex);
    if(axis1Voltage(target) < 0)
    {
        if(m_powerState == 0) return 0;
        return log<software_error,-1>({__FILE__, __LINE__, "axis 1 voltage error in INDI callback"});
    }
    
    return 0;
}

INDI_NEWCALLBACK_DEFN(kcubeCtrl, m_indiP_axis2_identify)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_axis2_identify, ipRecv);

    if(state() != stateCodes::READY) return 0;

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

    if(!(state() == stateCodes::READY || state() == stateCodes::OPERATING) ) return 0;

    //switch is toggled to on
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);
        if(axis2Enable() < 0)
        {
            if(m_powerState == 0) return 0;
            return log<software_error,-1>({__FILE__, __LINE__, "axis 2 enable error in INDI callback"});
        }
    }
    else
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);
        if(axis2Disable() < 0)
        {
            if(m_powerState == 0) return 0;
            return log<software_error,-1>({__FILE__, __LINE__, "axis 2 disable error in INDI callback"});
        }
    }
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(kcubeCtrl, m_indiP_axis2_voltage)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_axis2_voltage, ipRecv);

    if(!(state() == stateCodes::READY || state() == stateCodes::OPERATING) ) return 0;

    float target;
    indiTargetUpdate(m_indiP_axis2_voltage, target, ipRecv, true);

    std::lock_guard<std::mutex> guard(m_indiMutex);
    if(axis2Voltage(target) < 0)
    {
        if(m_powerState == 0) return 0;
        return log<software_error,-1>({__FILE__, __LINE__, "axis 2 voltage error in INDI callback"});
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN(kcubeCtrl, m_indiP_set)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_set, ipRecv);

    if(!(state() == stateCodes::READY || state() == stateCodes::OPERATING) ) return 0;

    //switch is toggled to on
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);
        if(!m_isSet)
        {
            //Set it to busy if we think this is a state change
            updateSwitchIfChanged(m_indiP_set, "toggle", pcf::IndiElement::On, INDI_BUSY);
        }
        //--else: if already set we probably don't need to call set(), but do it anyway to be sure

        if(set() < 0)
        {
            if(m_powerState == 0) return 0;
            return log<software_error,-1>({__FILE__, __LINE__, "set error in INDI callback"});
        }
    }
    else
    {
        std::lock_guard<std::mutex> guard(m_indiMutex);
        if(m_isSet)
        {
            //Set it to busy if we think this is a state change
            updateSwitchIfChanged(m_indiP_set, "toggle", pcf::IndiElement::Off, INDI_BUSY);
        }
        //--else: if already rested we probably don't need to call rest(), but do it anyway to be sure

        if(rest() < 0)
        {
            if(m_powerState == 0) return 0;
            return log<software_error,-1>({__FILE__, __LINE__, "rest error in INDI callback"});
        }
    }
   
   return 0;
}

} // namespace app
} // namespace MagAOX

#endif // kcubeCtrl_hpp
