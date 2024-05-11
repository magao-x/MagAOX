/** \file indiCompRuleConfig.hpp
  * \brief Configuration of rules for the MagAO-X stateRuleEngine
  *
  * \ingroup stateRuleEngine_files
  */

#ifndef stateRuleEngine_indiCompRuleConfig_hpp
#define stateRuleEngine_indiCompRuleConfig_hpp 

#include <map>

#include "indiCompRules.hpp"

/// Structure to provide management of the rule and property maps
/** This owns all pointers in the rule engine, and `delete`s them on destruction.
  */
struct indiRuleMaps
{
    typedef std::map<std::string, indiCompRule*> ruleMapT;
    typedef std::map<std::string, pcf::IndiProperty*> propMapT;

    ruleMapT rules;
    propMapT props;

    ~indiRuleMaps()
    {
        auto rit = rules.begin();
        while(rit != rules.end())
        {
            delete rit->second;
            ++rit;
        }

        auto pit = props.begin();
        while(pit != props.end())
        {
            delete pit->second;
            ++pit;
        }
    }
};

/* Structure used to hold ruleVal rule keys aside for final processing
   ruleVal rules can be created before the rules they link exist, so
   we hold the keys aside and set the pointers after all rules are created.
*/
struct ruleRuleKeys
{
    std::string rule1;
    std::string rule2;
};

/// Extract a property from a rule configuration
/** Reads the property and element, adding the property to the property map if necessary.
  *
  * \throws mx::err::invalidconfig if the property is already in the map but of a different type
  */
void extractRuleProp( pcf::IndiProperty ** prop,            ///< [out] pointer to the property, newly created or existing, which is in the map.
                      std::string & element,                ///< [out] the element name from the configuration
                      indiRuleMaps & maps,                  ///< [in] contains the property map to which the property is added
                      const std::string & section,          ///< [in] name of the section for this rule
                      const std::string & propkey,          ///< [in] the key for the property name
                      const std::string & elkey,            ///< [in] the key for the element name
                      const pcf::IndiProperty::Type & type, ///< [in] the type of the property
                      mx::app::appConfigurator & config     ///< [in] the application configuration structure
                    )
{
    std::string property;
    config.configUnused(property, mx::app::iniFile::makeKey(section, propkey ));

    if(maps.props.count(property) > 0)
    {
        //If the property already exists we just check if it's the right type
        if(maps.props[property]->getType() != type)
        {
            mxThrowException(mx::err::invalidconfig, "extracPropRule", "property " + property + " exists but is not correct type");
        }

        *prop = maps.props[property];
    }
    else
    {
        //Otherwise we create it
        *prop = new pcf::IndiProperty(type);
        maps.props.insert(std::pair<std::string, pcf::IndiProperty*>({property, *prop}));

        ///\todo have to split device and propertyName
    }

    config.configUnused(element, mx::app::iniFile::makeKey(section, elkey));

}

/// Load the rule and properties maps for a rule engine from a configuration file
/** ///\todo check for insertion failure
  * ///\todo add a constructor that has priority, message, and comparison, to reduce duplication
  */
void loadRuleConfig( indiRuleMaps & maps,              ///< [out] contains the rule and property maps in which to place the items found in config
                     mx::app::appConfigurator & config ///< [in] the application configuration structure
                   )
{
    std::vector<std::string> sections;

    config.unusedSections(sections);

    if( sections.size() == 0 )
    {
        mxThrowException(mx::err::invalidconfig, "loadRuleConfig", "no rules found in config");
    }
   
    std::map<std::string, ruleRuleKeys> rrkMap; // Holds the ruleVal rule keys aside for later post-processing

    for(size_t i=0; i< sections.size(); ++i)
    {
        bool ruleTypeSet = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "ruleType" ));
      
        //If there is no ruleType then this isn't a rule
        if( !ruleTypeSet ) continue;
      
        //If the rule already exists this is an error
        if(maps.rules.count(sections[i]) != 0)
        {
            mxThrowException(mx::err::invalidconfig, "loadRuleConfig", "duplicate rule: " + sections[i]);
        }

        std::string ruleType;
        config.configUnused(ruleType, mx::app::iniFile::makeKey(sections[i], "ruleType" ));
      
        std::string priostr="none";
        config.configUnused(priostr, mx::app::iniFile::makeKey(sections[i], "priority" ));
        rulePriority priority = string2priority(priostr);

        std::string message;
        config.configUnused(message, mx::app::iniFile::makeKey(sections[i], "message" ));

        std::string compstr="Eq";
        config.configUnused(compstr, mx::app::iniFile::makeKey(sections[i], "comp" ));
        ruleComparison comparison = string2comp(compstr);

        if(ruleType == numValRule::name)
        {
            numValRule * nvr = new numValRule;
            maps.rules.insert(std::pair<std::string, indiCompRule*>({sections[i], nvr}));

            nvr->priority(priority);
            nvr->message(message);
            nvr->comparison(comparison); 

            pcf::IndiProperty * prop = nullptr;
            std::string element;
    
            extractRuleProp( &prop, element, maps, sections[i], "property", "element", pcf::IndiProperty::Number, config );
            nvr->property(prop);
            nvr->element(element);

            double target = nvr->target();
            config.configUnused(target, mx::app::iniFile::makeKey(sections[i], "target" ));
            nvr->target(target);

            double tol = nvr->tol();
            config.configUnused(tol, mx::app::iniFile::makeKey(sections[i], "tol" ));
            nvr->tol(tol);
        }
        else if(ruleType == txtValRule::name)
        {
            txtValRule * tvr = new txtValRule;
            maps.rules.insert(std::pair<std::string, indiCompRule*>({sections[i], tvr}));

            tvr->priority(priority);
            tvr->message(message);
            tvr->comparison(comparison); 

            pcf::IndiProperty * prop = nullptr;
            std::string element;
    
            extractRuleProp( &prop, element, maps, sections[i], "property", "element", pcf::IndiProperty::Text, config );
            tvr->property(prop);
            tvr->element(element);


            std::string target = tvr->target();
            config.configUnused(target, mx::app::iniFile::makeKey(sections[i], "target" ));
            tvr->target(target);

        }
        else if(ruleType == swValRule::name)
        {
            swValRule * svr = new swValRule;
            maps.rules.insert(std::pair<std::string, indiCompRule*>({sections[i], svr}));

            svr->priority(priority);
            svr->message(message);
            svr->comparison(comparison); 

            pcf::IndiProperty * prop  = nullptr;
            std::string element;
    
            extractRuleProp( &prop, element, maps, sections[i], "property", "element", pcf::IndiProperty::Switch, config );
            svr->property(prop);
            svr->element(element);

            std::string target = "On";
            config.configUnused(target, mx::app::iniFile::makeKey(sections[i], "target" ));
            svr->target(target);
        }
        else if(ruleType == elCompNumRule::name)
        {
            elCompNumRule * nvr = new elCompNumRule;
            maps.rules.insert(std::pair<std::string, indiCompRule*>({sections[i], nvr}));

            nvr->priority(priority);
            nvr->message(message);
            nvr->comparison(comparison); 

            pcf::IndiProperty * prop1;
            std::string element1;
    
            extractRuleProp( &prop1, element1, maps, sections[i], "property1", "element1", pcf::IndiProperty::Number, config );
            nvr->property1(prop1);
            nvr->element1(element1);

            pcf::IndiProperty * prop2;
            std::string element2;
    
            extractRuleProp( &prop2, element2, maps, sections[i], "property2", "element2", pcf::IndiProperty::Number, config );
            nvr->property2(prop2);
            nvr->element2(element2);
        }
        else if(ruleType == elCompTxtRule::name)
        {
            elCompTxtRule * tvr = new elCompTxtRule;
            maps.rules.insert(std::pair<std::string, indiCompRule*>({sections[i], tvr}));

            tvr->priority(priority);
            tvr->message(message);
            tvr->comparison(comparison); 

            pcf::IndiProperty * prop1;
            std::string element1;
    
            extractRuleProp( &prop1, element1, maps, sections[i], "property1", "element1", pcf::IndiProperty::Text, config );
            tvr->property1(prop1);
            tvr->element1(element1);

            pcf::IndiProperty * prop2;
            std::string element2;
    
            extractRuleProp( &prop2, element2, maps, sections[i], "property2", "element2", pcf::IndiProperty::Text, config );
            tvr->property2(prop2);
            tvr->element2(element2);
        }
        else if(ruleType == elCompSwRule::name)
        {
            elCompSwRule * svr = new elCompSwRule;
            maps.rules.insert(std::pair<std::string, indiCompRule*>({sections[i], svr}));

            svr->priority(priority);
            svr->message(message);
            svr->comparison(comparison); 

            pcf::IndiProperty * prop1;
            std::string element1;
    
            extractRuleProp( &prop1, element1, maps, sections[i], "property1", "element1", pcf::IndiProperty::Switch, config );
            svr->property1(prop1);
            svr->element1(element1);

            pcf::IndiProperty * prop2;
            std::string element2;
    
            extractRuleProp( &prop2, element2, maps, sections[i], "property2", "element2", pcf::IndiProperty::Switch, config );
            svr->property2(prop2);
            svr->element2(element2);
        }
        else if(ruleType == ruleCompRule::name)
        {
            //Here we have to hold the ruleVal keys separately for later processing after all the rules are created.

            if(rrkMap.count(sections[i]) > 0)
            {
                //This probably should be impossible, since we already checked maps.rules above...
                mxThrowException(mx::err::invalidconfig, "loadRuleConfig", "duplicate ruleRule: " + sections[i]);
            }

            ruleCompRule * rcr = new ruleCompRule;
            maps.rules.insert(std::pair<std::string, indiCompRule*>({sections[i], rcr}));

            rcr->priority(priority);
            rcr->comparison(comparison); 

            ruleRuleKeys rrk;
        
            config.configUnused(rrk.rule1, mx::app::iniFile::makeKey(sections[i], "rule1" ));
            if(rrk.rule1 == "")
            {
                mxThrowException(mx::err::invalidconfig, "loadRuleConfig", "rule1 for ruleVal rule " + sections[i] + " not found");
            }

            config.configUnused(rrk.rule2, mx::app::iniFile::makeKey(sections[i], "rule2" ));
            if(rrk.rule2 == "")
            {
                mxThrowException(mx::err::invalidconfig, "loadRuleConfig", "rule2 for ruleVal rule " + sections[i] + " not found");
            }

            rrkMap.insert(std::pair<std::string, ruleRuleKeys>(sections[i], rrk));
        }
        else
        {
            mxThrowException(mx::err::notimpl, "loadRuleConfig", "unknown rule type " + ruleType + " in " + sections[i]);
        }
    }

    //Now set the rule pointers for any ruleVal rules
    auto it=rrkMap.begin();
    while(it != rrkMap.end())
    {
        if( maps.rules.count(it->first) == 0 )
        {
            mxThrowException(mx::err::invalidconfig, "loadRuleConfig", "rule parsing error for " + it->first);
        }

        if( maps.rules.count(it->second.rule1) == 0 )
        {
            mxThrowException(mx::err::invalidconfig, "loadRuleConfig", "rule1 " + it->second.rule1 + " not found for ruleVal rule " + it->first );
        }

        if( maps.rules.count(it->second.rule2) == 0 )
        {
            mxThrowException(mx::err::invalidconfig, "loadRuleConfig", "rule2 " + it->second.rule2 + " not found for ruleVal rule " + it->first );
        }

        ruleCompRule * rcr = nullptr;

        try
        {
            rcr = dynamic_cast<ruleCompRule *>(maps.rules[it->first]);
        }
        catch(const std::exception & e)
        {
            mxThrowException(mx::err::invalidconfig, "loadRuleConfig", "error casting " + it->first + ": " + e.what() );
        }

        if(rcr == nullptr)
        {
            mxThrowException(mx::err::invalidconfig, "loadRuleConfig", it->first + " is not a ruleVal rule but has rules" );
        }

        rcr->rule1(maps.rules[it->second.rule1]);
        rcr->rule2(maps.rules[it->second.rule2]);

        ++it;
    }
}

#endif //stateRuleEngine_indiCompRuleConfig_hpp
