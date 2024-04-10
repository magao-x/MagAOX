/** \file closedLoopIndi.hpp
  * \brief The MagAO-X INDI Closed Loop header file
  *
  * \ingroup closedLoopIndi_files
  */

#ifndef closedLoopIndi_hpp
#define closedLoopIndi_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup closedLoopIndi
  * \brief The MagAO-X application to do closed-loop control using INDI properties
  *
  * <a href="../handbook/operating/software/apps/closedLoopIndi.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup closedLoopIndi_files
  * \ingroup closedLoopIndi
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X application to do closed-loop control using INDI properties
/** 
  * \ingroup closedLoopIndi
  */
class closedLoopIndi : public MagAOXApp<true>
{
   //Give the test harness access.
   friend class closedLoopIndi_test;

protected:

    /** \name Configurable Parameters
      *@{
      */
    
    std::string m_inputDevice;                           ///< The device with the input disturbances and frame counter.
    std::string m_inputProperty;                         ///< The property with the input disturbances and frame counter.
    std::vector<std::string> m_inputElements {"x", "y"}; ///< The elements with the input disturbances.  Must be two, defaults are "x" and "y".
    std::string m_inputCounterElement {"counter"};       ///< The element with the frame counter, a monotonically increasing integer. Default is "counter".

    std::vector<std::string> m_ctrlDevices;                         ///< Device names of the controller(s). If only one, it's used for both properties.  Max two.
    std::vector<std::string> m_ctrlProperties;                      ///< Properties of the ctrl device(s) to which to give the commands. Must specify two.
    std::vector<std::string> m_ctrlCurrents {"current", "current"}; ///< current elements of the properties on which to base the commands. Must specify 0 or 2. Default is 'current'.
    std::vector<std::string> m_ctrlTargets {"target", "target"};    ///< target elements of the properties to which to send the commands. Must specify 0 or 2. Default is 'target'.
    
    mx::improc::eigenImage<float> m_references; ///< The reference values of the disturbances

    std::unordered_map<std::string, std::string> m_fsmStates; ///< The FSM states of the control devices.

    std::vector<float> m_currents; ///< The current commands

    mx::improc::eigenImage<float> m_intMat; ///< The interaction matrix.  Default is [1 0][0 1].

    std::vector<float> m_defaultGains; ///< The default gains, per-axis
    
    std::string m_upstreamDevice;                  ///< The upstream device to monitor to automatically open this loop if it's loop opens
    std::string m_upstreamProperty {"loop_state"}; ///< The name of the toggle switch to monitor
    
    ///@}
    
    int64_t m_counter = -1; ///< The latest value of the loop counter
    mx::improc::eigenImage<float> m_measurements; ///< The latest value of the measurements
    mx::improc::eigenImage<float> m_commands; ///< The latest commands
    
    float m_ggain {0}; ///< The global gain
    
    std::vector<float> m_gains; ///< The axis gains
    
    bool m_loopClosed {false}; ///< Whether or not the loop is closed

public:
    /// Default c'tor.
    closedLoopIndi();
 
    /// D'tor, declared and defined for noexcept.
    ~closedLoopIndi() noexcept
    {}
 
    virtual void setupConfig();
 
    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
      */
    int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);
 
    virtual void loadConfig();
 
    /// Startup function
    /**
      *
      */
    virtual int appStartup();
 
    /// Implementation of the FSM for closedLoopIndi.
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
 
    /// Change the loop state
    int toggleLoop( bool onoff );

    /// Update the loop with a new command
    int updateLoop();
 
    /// Send commands to the control devices
    int sendCommands(std::vector<float> & commands);
    
    //INDI 
 
    pcf::IndiProperty m_indiP_reference0;
    INDI_NEWCALLBACK_DECL(closedLoopIndi, m_indiP_reference0);

    pcf::IndiProperty m_indiP_reference1;
    INDI_NEWCALLBACK_DECL(closedLoopIndi, m_indiP_reference1);

    pcf::IndiProperty m_indiP_inputs;
    INDI_SETCALLBACK_DECL(closedLoopIndi, m_indiP_inputs);

    pcf::IndiProperty m_indiP_ggain;
    INDI_NEWCALLBACK_DECL(closedLoopIndi, m_indiP_ggain);
    
    pcf::IndiProperty m_indiP_ctrlEnabled;
    INDI_NEWCALLBACK_DECL(closedLoopIndi, m_indiP_ctrlEnabled);
 
    pcf::IndiProperty m_indiP_counterReset;
    INDI_NEWCALLBACK_DECL(closedLoopIndi, m_indiP_counterReset);

    pcf::IndiProperty m_indiP_ctrl0_fsm; ///< The INDI property for fsm state of axis 0
    INDI_SETCALLBACK_DECL(closedLoopIndi, m_indiP_ctrl0_fsm);

    pcf::IndiProperty m_indiP_ctrl0; ///< The INDI property used for control of axis 0
    INDI_SETCALLBACK_DECL(closedLoopIndi, m_indiP_ctrl0);

    pcf::IndiProperty m_indiP_ctrl1_fsm; ///< The INDI property for fsm state of axis 1
    INDI_SETCALLBACK_DECL(closedLoopIndi, m_indiP_ctrl1_fsm);

    pcf::IndiProperty m_indiP_ctrl1; ///< The INDI property used for control of axis 1
    INDI_SETCALLBACK_DECL(closedLoopIndi, m_indiP_ctrl1);

    pcf::IndiProperty m_indiP_upstream; ///< Property used to report the upstream loop state    
    INDI_SETCALLBACK_DECL(closedLoopIndi, m_indiP_upstream);
};

closedLoopIndi::closedLoopIndi() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
    return;
}

void closedLoopIndi::setupConfig()
{
    config.add("input.device", "", "input.device", argType::Required, "input", "device", false, "string", "The device with the input disturbances and frame counter.");
    config.add("input.property", "", "input.property", argType::Required, "input", "property", false, "string", "The property with the input disturbances and counter.");
    config.add("input.elements", "", "input.elements", argType::Required, "input", "elements", false, "vector<string>", " The elements with the input disturbances.  Must be two, defaults are 'x' and 'y'.");
    config.add("input.counterElement", "", "input.counterElement", argType::Required, "input", "counterElement", false, "string", "The element with the frame counter, a monotonically increasing integer. Default is 'counter'.");

    config.add("input.references", "", "input.references", argType::Required, "input", "references", false, "vector<float>", "The reference values for the input disturbances.");

    config.add("ctrl.devices", "", "ctrl.devices", argType::Required, "ctrl", "devices", false, "string", "Device names of the controller(s). If only one, it's used for both properties.  Max two.");
    config.add("ctrl.properties", "", "ctrl.properties", argType::Required, "ctrl", "properties", false, "string", "Properties of the ctrl device(s) to which to give the commands. Must specify two.");
    config.add("ctrl.currents", "", "ctrl.currents", argType::Required, "ctrl", "currents", false, "vector<string>", "current elements of the properties on which to base the commands. Must specify 0 or 2. Default is 'current'");
    config.add("ctrl.targets", "", "ctrl.targets", argType::Required, "ctrl", "targets", false, "vector<string>", "target elements of the properties to which to send the commands. Must specify 0 or 2. Default is 'target'.");

    config.add("loop.intMat00", "", "loop.intMat00", argType::Required, "loop", "intMat00", false, "float", "element (0,0) of the interaction matrix. Default is 1.");
    config.add("loop.intMat01", "", "loop.intMat01", argType::Required, "loop", "intMat01", false, "float", "element (0,1) of the interaction matrix. Default is 0.");
    config.add("loop.intMat10", "", "loop.intMat10", argType::Required, "loop", "intMat10", false, "float", "element (1,0) of the interaction matrix. Default is 0.");
    config.add("loop.intMat11", "", "loop.intMat11", argType::Required, "loop", "intMat11", false, "float", "element (1,1) of the interaction matrix. Default is 1.");

    config.add("loop.gains", "", "loop.gains", argType::Required, "loop", "gains", false, "vector<float>", "default loop gains.  If single number, it is applied to all axes.");
    config.add("loop.upstream", "", "loop.upstream", argType::Required, "loop", "upstream", false, "string", "Upstream loop device name.  This loop will open, and optionally close, with the upstream loop.  Default none.");
    config.add("loop.upstreamProperty", "", "loop.upstreamProperty", argType::Required, "loop", "upstreamProperty", false, "string", "Property of upstream loop device to follow.  Must be a toggle.  Default is loop_state.");

}

int closedLoopIndi::loadConfigImpl( mx::app::appConfigurator & _config )
{
    _config(m_inputDevice, "input.device");
    _config(m_inputProperty, "input.property");
    _config(m_inputElements, "input.elements");
    _config(m_inputCounterElement, "input.counterElement");

    if(m_inputDevice == "")
    {
        m_shutdown = 1;
        return log<software_error, -1>({__FILE__, __LINE__, "no input device specified"});
    }
    
    if(m_inputProperty == "")
    {
        m_shutdown = 1;
        return log<software_error, -1>({__FILE__, __LINE__, "no input property specified"});
    }
    
    if(m_inputElements.size() != 2)
    {
        m_shutdown = 1;
        return log<software_error, -1>({__FILE__, __LINE__, "must specify only two input.elements"});
    }

    std::vector<float> refs({0,0});
    _config(refs, "input.references");
    if(refs.size() != 2)
    {
        m_shutdown = 1;
        return log<software_error, -1>({__FILE__, __LINE__, "input.references must have 2 elements"});
    }

    m_references.resize(2,1);
    m_references(0,0) = refs[0];
    m_references(1,0) = refs[1];

    _config(m_ctrlDevices, "ctrl.devices");
    _config(m_ctrlProperties, "ctrl.properties");
    _config(m_ctrlCurrents, "ctrl.currents");
    _config(m_ctrlTargets, "ctrl.targets");
    
    if(m_ctrlDevices.size() == 1)
    {
        m_ctrlDevices.push_back(m_ctrlDevices[0]);
    }
    else if(m_ctrlDevices.size() != 2)
    {
        m_shutdown = 1;
        return log<software_error, -1>({__FILE__, __LINE__, "must specify two ctrl.devices"});
    }
    
    if(m_ctrlProperties.size() != 2)
    {
        m_shutdown = 1;
        return log<software_error, -1>({__FILE__, __LINE__, "must specify two ctrl.properties"});
    }
    
    if(m_ctrlTargets.size() != 2)
    {
        m_shutdown = 1;
        return log<software_error, -1>({__FILE__, __LINE__, "must specify two ctrl.targets"});
    }
    
    if(m_ctrlCurrents.size() != 2)
    {
        m_shutdown = 1;
        return log<software_error, -1>({__FILE__, __LINE__, "must specify two ctrl.currents"});
    }
    
    if(m_ctrlTargets.size() != 2)
    {
        m_shutdown = 1;
        return log<software_error, -1>({__FILE__, __LINE__, "must specify two ctrl.targets"});
    }
 
    float im00 = 1;
    float im01 = 0;
    float im10 = 0;
    float im11 = 1;

    _config(im00, "loop.intMat00");
    _config(im01, "loop.intMat01");
    _config(im10, "loop.intMat10");
    _config(im11, "loop.intMat11");

    m_intMat.resize(2, 2);
    m_intMat(0,0) = im00;
    m_intMat(0,1) = im01;
    m_intMat(1,0) = im10;
    m_intMat(1,1) = im11;
    
    _config(m_defaultGains, "loop.gains");
    _config(m_upstreamDevice, "loop.upstream");
    _config(m_upstreamProperty, "loop.upstreamProperty");

    return 0;
}

void closedLoopIndi::loadConfig()
{
   loadConfigImpl(config);
}

int closedLoopIndi::appStartup()
{
    REG_INDI_SETPROP(m_indiP_inputs, m_inputDevice, m_inputProperty);
 
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_reference0, "reference0", -1e15, 1e15, 1, "%g", "reference0", "references");
    m_indiP_reference0["current"] = m_references(0,0);
    m_indiP_reference0["target"] = m_references(0,0);

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_reference1, "reference1", -1e15, 1e15, 1, "%g", "reference1", "references");
    m_indiP_reference1["current"] = m_references(1,0);
    m_indiP_reference1["target"] = m_references(1,0);
 
    if(m_ctrlTargets.size() != m_defaultGains.size())
    {
        if(m_defaultGains.size()==1)
        {
            m_defaultGains.push_back(m_defaultGains[0]);
           log<text_log>("Setting loop.gains gains to be same size as ctrl.Targets", logPrio::LOG_NOTICE);
        }
        else
        {
           return log<software_error, -1>({__FILE__, __LINE__, "ctrl.Targets and loop.gains are not the same size"});
        }
    }
 
    m_gains.resize(m_defaultGains.size());
    for(size_t n=0; n < m_defaultGains.size(); ++n) m_gains[n] = m_defaultGains[n];
 
    CREATE_REG_INDI_NEW_NUMBERU( m_indiP_ggain, "loop_gain", 0, 1, 0, "%0.2f", "gain", "loop");
    m_indiP_ggain["current"] = m_ggain;
    m_indiP_ggain["target"] = m_ggain;  
    
 
    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_ctrlEnabled, "loop_state");

    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_counterReset, "counter_reset");

    m_measurements.resize(2,1);
    m_measurements.setZero();

    m_currents.resize(m_ctrlDevices.size(), -1e15);
 
    REG_INDI_SETPROP(m_indiP_ctrl0_fsm, m_ctrlDevices[0], "fsm");

    REG_INDI_SETPROP(m_indiP_ctrl0, m_ctrlDevices[0], m_ctrlProperties[0]);

    if(m_ctrlDevices[1] != m_ctrlDevices[0])
    {
        REG_INDI_SETPROP(m_indiP_ctrl1_fsm, m_ctrlDevices[1], "fsm");
    }

    REG_INDI_SETPROP(m_indiP_ctrl1, m_ctrlDevices[1], m_ctrlProperties[1]);
 
    m_commands.resize(2, 2);
    m_commands.setZero();

    //Get the loop state for managing offloading
    if(m_upstreamDevice != "")
    {
        REG_INDI_SETPROP(m_indiP_upstream, m_upstreamDevice, m_upstreamProperty);
    }

    return 0;
}

int closedLoopIndi::appLogic()
{
    if(m_loopClosed)
    {
        state(stateCodes::OPERATING);
    }
    else
    {
        state(stateCodes::READY);
    }
   
    return 0;
}

int closedLoopIndi::appShutdown()
{
    return 0;
}

int closedLoopIndi::toggleLoop(bool onoff)
{
   if(!m_loopClosed && onoff) //not enabled so change
   {      
      m_loopClosed = true;
      log<loop_closed>();
      updateSwitchIfChanged(m_indiP_ctrlEnabled, "toggle", pcf::IndiElement::On, INDI_OK);
      return 0;
   }

   if(m_loopClosed && !onoff)
   {
      m_loopClosed = false;
      log<loop_open>();
      updateSwitchIfChanged(m_indiP_ctrlEnabled, "toggle", pcf::IndiElement::Off, INDI_IDLE);
   
      return 0;
   }

   return 0;
}


inline
int closedLoopIndi::updateLoop()
{
    bool ready = false;

    //This should only give ready == true if all devices exist and are ready
    for(size_t n = 0; n < m_ctrlDevices.size(); ++n)
    {
        if(m_fsmStates.count(m_ctrlDevices[n]) > 0)
        {
            if(m_fsmStates[m_ctrlDevices[n]] == "READY")
            {
                ready = true;
            }
            else
            {
                ready = false;
                break;
            }
        }
        else
        {
            ready = false;
            break;
        }
    }

    if(ready != true)
    {
        return 0;
    }

    m_commands.matrix() = m_intMat.matrix() * (m_measurements - m_references).matrix();

    std::vector<float> commands;
    commands.resize(m_measurements.rows());

    for(int cc = 0; cc < m_measurements.rows(); ++cc)
    {
        commands[cc] = m_currents[cc] - m_ggain*m_gains[cc]*m_commands(cc,0);
    }
  
    //And send commands.
    if(m_loopClosed)
    {
        return sendCommands(commands);
    }
    else 
    {
        return 0;
    }
}

inline
int closedLoopIndi::sendCommands(std::vector<float> & commands)
{
    for(size_t n=0; n < m_ctrlDevices.size(); ++n)
    {
       pcf::IndiProperty ip(pcf::IndiProperty::Number);
    
       ip.setDevice(m_ctrlDevices[n]);
       ip.setName(m_ctrlProperties[n]);
       ip.add(pcf::IndiElement(m_ctrlTargets[n]));
       ip[m_ctrlTargets[n]] = commands[n];
    
       sendNewProperty(ip);
    }
 
    return 0;
}

INDI_NEWCALLBACK_DEFN(closedLoopIndi, m_indiP_reference0)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(ipRecv, m_indiP_reference0);
   
    float target;
    
    if( indiTargetUpdate( m_indiP_reference0, target, ipRecv, true) < 0)
    {
       return log<software_error, -1>({__FILE__,__LINE__});
    }
    
    m_references(0,0) = target;
    
    updateIfChanged(m_indiP_reference0, std::vector<std::string>({"current", "target"}), std::vector<float>({m_references(0,0), m_references(0,0)}));

    log<text_log>("set reference0 to " + std::to_string(m_references(0,0)), logPrio::LOG_NOTICE);
    
    return 0;
}

INDI_NEWCALLBACK_DEFN(closedLoopIndi, m_indiP_reference1)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(ipRecv, m_indiP_reference1);
   
    float target;
    
    if( indiTargetUpdate( m_indiP_reference1, target, ipRecv, true) < 0)
    {
       return log<software_error, -1>({__FILE__,__LINE__});
    }
    
    m_references(1,0) = target;
    
    updateIfChanged(m_indiP_reference1, std::vector<std::string>({"current", "target"}), std::vector<float>({m_references(1,0), m_references(1,0)}));

    log<text_log>("set reference1 to " + std::to_string(m_references(1,0)), logPrio::LOG_NOTICE);
    
    return 0;
}

INDI_SETCALLBACK_DEFN(closedLoopIndi, m_indiP_inputs)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(ipRecv, m_indiP_inputs)

    if(!ipRecv.find(m_inputElements[0])) return -1;
    if(!ipRecv.find(m_inputElements[1])) return -1;
    if(!ipRecv.find(m_inputCounterElement)) return -1;

    int counter = ipRecv[m_inputCounterElement].get<int>();

    if(counter != m_counter)
    {
        m_counter = counter;
        m_measurements(0,0) = ipRecv[m_inputElements[0]].get<float>();
        m_measurements(1,0) = ipRecv[m_inputElements[1]].get<float>();

        return updateLoop();
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN(closedLoopIndi, m_indiP_ggain)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(ipRecv, m_indiP_ggain);
   
    float target;
    
    if( indiTargetUpdate( m_indiP_ggain, target, ipRecv, true) < 0)
    {
       log<software_error>({__FILE__,__LINE__});
       return -1;
    }
    
    m_ggain = target;
    
    updateIfChanged(m_indiP_ggain, "current", m_ggain);
    updateIfChanged(m_indiP_ggain, "target", m_ggain);
    
    log<text_log>("set global gain to " + std::to_string(m_ggain), logPrio::LOG_NOTICE);
    
    return 0;
}

INDI_NEWCALLBACK_DEFN(closedLoopIndi, m_indiP_ctrlEnabled)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(ipRecv, m_indiP_ctrlEnabled);

    //switch is toggled to on
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
    {
       return toggleLoop(true);
    }
 
    //switch is toggle to off
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
    {
       return toggleLoop(false);
    }
    
    return 0;
}

INDI_NEWCALLBACK_DEFN(closedLoopIndi, m_indiP_counterReset)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(ipRecv, m_indiP_counterReset);

    //switch is toggled to on
    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
    {
       m_counter = -1;
    }
 
    
    return 0;
}

INDI_SETCALLBACK_DEFN(closedLoopIndi, m_indiP_ctrl0_fsm)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(ipRecv, m_indiP_ctrl0_fsm);

    if(ipRecv.find("state"))
    {
        m_fsmStates[ipRecv.getDevice()] = ipRecv["state"].get();
    }

   return 0;
}

INDI_SETCALLBACK_DEFN(closedLoopIndi, m_indiP_ctrl0)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(ipRecv, m_indiP_ctrl0);

    if(ipRecv.find(m_ctrlCurrents[0]))
    {
        m_currents[0] = ipRecv[m_ctrlCurrents[0]].get<float>();
    }

   return 0;
}

INDI_SETCALLBACK_DEFN(closedLoopIndi, m_indiP_ctrl1_fsm)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(ipRecv, m_indiP_ctrl1_fsm);

    if(ipRecv.find("state"))
    {
        m_fsmStates[ipRecv.getDevice()] = ipRecv["state"].get();
    }

   return 0;
}

INDI_SETCALLBACK_DEFN(closedLoopIndi, m_indiP_ctrl1)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(ipRecv, m_indiP_ctrl1);

    if(ipRecv.find(m_ctrlCurrents[1]))
    {
        m_currents[1] = ipRecv[m_ctrlCurrents[1]].get<float>();
    }

   return 0;
}

INDI_SETCALLBACK_DEFN(closedLoopIndi, m_indiP_upstream)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(ipRecv, m_indiP_upstream);

    if(!ipRecv.find("toggle")) return 0;
 
    if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
    {
       std::cerr << "upstream on\n";
       return toggleLoop(true);
    }
    else if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
    {
       std::cerr << "upstream off\n";
       return toggleLoop(false);
    }
 
    return 0;
}


} //namespace app
} //namespace MagAOX

#endif //closedLoopIndi_hpp
