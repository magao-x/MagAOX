/** \file stateRuleEngine.hpp
  * \brief The MagAO-X stateRuleEngine application header file
  *
  * \ingroup stateRuleEngine_files
  */

#ifndef stateRuleEngine_hpp
#define stateRuleEngine_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "indiCompRuleConfig.hpp"

/** \defgroup stateRuleEngine
  * \brief The MagAO-X stateRuleEngine application 
  *
  * <a href="../handbook/operating/software/apps/stateRuleEngine.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup stateRuleEngine_files
  * \ingroup stateRuleEngine
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X stateRuleEngine
/** 
  * \ingroup stateRuleEngine
  */
class stateRuleEngine : public MagAOXApp<true>
{

    //Give the test harness access.
    friend class stateRuleEngine_test;

protected:

    /** \name Configurable Parameters
      *@{
      */
    
    indiRuleMaps m_ruleMaps;
     
    ///@}

public:
    /// Default c'tor.
    stateRuleEngine();
 
    /// D'tor, declared and defined for noexcept.
    ~stateRuleEngine() noexcept
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
 
    /// Implementation of the FSM for stateRuleEngine.
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
 
 
    /// The static callback function to be registered for rule properties
    /** 
      * 
      * \returns 0 on success.
      * \returns -1 on error.
      */
    static int st_newCallBack_ruleProp( void * app,                     ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                        const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                      );

    /// Callback to process a NEW preset position request
    /**
      * \returns 0 on success.
      * \returns -1 on error.
      */
    int newCallBack_ruleProp( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
 

    pcf::IndiProperty m_indiP_info;
    pcf::IndiProperty m_indiP_caution;
    pcf::IndiProperty m_indiP_warning;
    pcf::IndiProperty m_indiP_alert;

};

stateRuleEngine::stateRuleEngine() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
    return;
}

void stateRuleEngine::setupConfig()
{
}

int stateRuleEngine::loadConfigImpl( mx::app::appConfigurator & _config )
{
    loadRuleConfig(m_ruleMaps, _config);

    return 0;
}

void stateRuleEngine::loadConfig()
{
    loadConfigImpl(config);
}

int stateRuleEngine::appStartup()
{
    for(auto it = m_ruleMaps.rules.begin(); it != m_ruleMaps.rules.end(); ++it)
    {
        if(it->second->priority() == rulePriority::info)
        {
            if(m_indiP_info.getDevice() != m_configName)
            {
                if(registerIndiPropertyNew( m_indiP_info, "info", pcf::IndiProperty::Switch, pcf::IndiProperty::ReadOnly,
                                                                pcf::IndiProperty::Idle, pcf::IndiProperty::AnyOfMany, nullptr) < 0)
                {
                    return log<software_critical,-1>({__FILE__, __LINE__});
                }       
            }

            pcf::IndiElement elem = pcf::IndiElement(it->first, pcf::IndiElement::Off);
            m_indiP_info.add(elem);
        }

        if(it->second->priority() == rulePriority::caution)
        {
            if(m_indiP_caution.getDevice() != m_configName)
            {
                if(registerIndiPropertyNew( m_indiP_caution, "caution", pcf::IndiProperty::Switch, pcf::IndiProperty::ReadOnly,
                                                                pcf::IndiProperty::Idle, pcf::IndiProperty::AnyOfMany, nullptr) < 0)
                {
                    return log<software_critical,-1>({__FILE__, __LINE__});
                }                                    
            }

            pcf::IndiElement elem = pcf::IndiElement(it->first, pcf::IndiElement::Off);
            m_indiP_caution.add(elem);
        }

        if(it->second->priority() == rulePriority::warning)
        {
            if(m_indiP_warning.getDevice() != m_configName)
            {
                if(registerIndiPropertyNew( m_indiP_warning, "warning", pcf::IndiProperty::Switch, pcf::IndiProperty::ReadOnly,
                                                                pcf::IndiProperty::Idle, pcf::IndiProperty::AnyOfMany, nullptr) < 0)
                {
                    return log<software_critical,-1>({__FILE__, __LINE__});
                } 
            }

            pcf::IndiElement elem = pcf::IndiElement(it->first, pcf::IndiElement::Off);
            m_indiP_warning.add(elem); 
        }

        if(it->second->priority() == rulePriority::alert)
        {
            if(m_indiP_alert.getDevice() != m_configName)
            {
                if(registerIndiPropertyNew( m_indiP_alert, "alert", pcf::IndiProperty::Switch, pcf::IndiProperty::ReadOnly,
                                                                pcf::IndiProperty::Idle, pcf::IndiProperty::AnyOfMany, nullptr) < 0)
                {
                    return log<software_critical,-1>({__FILE__, __LINE__});
                } 
            }

            pcf::IndiElement elem = pcf::IndiElement(it->first, pcf::IndiElement::Off);
            m_indiP_alert.add(elem);
        }
    }

    for(auto it = m_ruleMaps.props.begin(); it != m_ruleMaps.props.end(); ++it)
    {
        if(it->second == nullptr) continue;

        std::string devName, propName;

        int rv = indi::parseIndiKey(devName, propName, it->first);
        if(rv != 0)
        {
            log<software_error>({__FILE__, __LINE__, 0, rv, "error parsing INDI key: " + it->first});
            return -1;
        }

        registerIndiPropertySet( *it->second, devName, propName, st_newCallBack_ruleProp);
    }
    
    state(stateCodes::READY);

    return 0;
}

int stateRuleEngine::appLogic()
{
    for(auto it = m_ruleMaps.rules.begin(); it != m_ruleMaps.rules.end(); ++it)
    {
        if(it->second->priority() != rulePriority::none)
        {
            try
            {
                bool val = it->second->value();

                pcf::IndiElement::SwitchStateType onoff = pcf::IndiElement::Off;
                if(val) onoff = pcf::IndiElement::On;

                if(it->second->priority() == rulePriority::info)
                {
                    updateSwitchIfChanged(m_indiP_info, it->first, onoff);
                }
                else if(it->second->priority() == rulePriority::caution)
                {
                    updateSwitchIfChanged(m_indiP_caution, it->first, onoff);
                }
                else if(it->second->priority() == rulePriority::warning)
                {
                    updateSwitchIfChanged(m_indiP_warning, it->first, onoff);
                }
                else 
                {
                    updateSwitchIfChanged(m_indiP_alert, it->first, onoff);
                }

            }
            catch(const std::exception & e)
            {
                ///\todo how to handle startup vs misconfiguration
                
                /*
                if(it->second->priority() == rulePriority::none)
                {
                    updateSwitchIfChanged(m_indiP_info, it->first, pcf::IndiElement::Off);
                }*/
            }
        }
    }


    return 0;
}

int stateRuleEngine::appShutdown()
{
    return 0;
}

int stateRuleEngine::st_newCallBack_ruleProp( void * app,      
                                              const pcf::IndiProperty &ipRecv 
                                            )
{
    stateRuleEngine * sre = static_cast<stateRuleEngine *>(app);

    sre->newCallBack_ruleProp(ipRecv);

    return 0;
}

int stateRuleEngine::newCallBack_ruleProp( const pcf::IndiProperty &ipRecv)
{
    std::string key = ipRecv.createUniqueKey();

    if(m_ruleMaps.props.count(key) == 0)
    {
        return 0;
    }

    if(m_ruleMaps.props[key] == nullptr) //
    {
        return 0;
    }

    *m_ruleMaps.props[key] = ipRecv;

    return 0;
}

} //namespace app
} //namespace MagAOX

#endif //stateRuleEngine_hpp
