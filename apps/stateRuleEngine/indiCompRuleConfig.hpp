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
    typedef std::map<std::string, indiCompRule *>      ruleMapT;
    typedef std::map<std::string, pcf::IndiProperty *> propMapT;

    ruleMapT rules;
    propMapT props;

    ~indiRuleMaps()
    {
        auto rit = rules.begin();
        while( rit != rules.end() )
        {
            delete rit->second;
            ++rit;
        }

        auto pit = props.begin();
        while( pit != props.end() )
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

/// Extract a property-only reference from a rule configuration.
/** Reads the property, adding it to the property map if necessary.
 *
 * \throws mx::err::invalidconfig if the property is not configured or if the
 *         property already exists in the map but with a different type
 */
void extractRuleProperty(
    pcf::IndiProperty **prop,     ///< [out] pointer to the property, newly created or existing, which is in the map.
    std::string        &property, ///< [out] the property name from the configuration
    indiRuleMaps       &maps,     ///< [in] contains the property map to which the property is added
    const std::string  &section,  ///< [in] name of the section for this rule
    const std::string  &propkey,  ///< [in] the key for the property name
    const pcf::IndiProperty::Type &type,  ///< [in] the type of the property
    mx::app::appConfigurator      &config ///< [in] the application configuration structure
)
{
    config.configUnused( property, mx::app::iniFile::makeKey( section, propkey ) );
    if( property == "" )
    {
        throw mx::exception( mx::error_t::invalidconfig, std::format( "{} for rule {} not found", propkey, section ) );
    }

    if( maps.props.count( property ) > 0 )
    {
        if( maps.props[property]->getType() != type )
        {
            throw mx::exception( mx::error_t::invalidconfig,
                                 "property " + property + " exists but is not correct type" );
        }

        *prop = maps.props[property];
    }
    else
    {
        *prop = new pcf::IndiProperty( type );
        maps.props.insert( std::pair<std::string, pcf::IndiProperty *>( { property, *prop } ) );

        ///\todo have to split device and propertyName
    }
}

/// Extract a property from a rule configuration
/** Reads the property and element, adding the property to the property map if necessary.
 *
 * \throws mx::err::invalidconfig if the property is already in the map but of a different type
 */
void extractRuleProp(
    pcf::IndiProperty **prop,    ///< [out] pointer to the property, newly created or existing, which is in the map.
    std::string        &element, ///< [out] the element name from the configuration
    indiRuleMaps       &maps,    ///< [in] contains the property map to which the property is added
    const std::string  &section, ///< [in] name of the section for this rule
    const std::string  &propkey, ///< [in] the key for the property name
    const std::string  &elkey,   ///< [in] the key for the element name
    const pcf::IndiProperty::Type &type,  ///< [in] the type of the property
    mx::app::appConfigurator      &config ///< [in] the application configuration structure
)
{
    std::string property;
    extractRuleProperty( prop, property, maps, section, propkey, type, config );

    config.configUnused( element, mx::app::iniFile::makeKey( section, elkey ) );
}

/// \cond
// strip leading and trailing whitespace and then opening and closing "".  leaves spaces between "".
inline void stripQuotesWS( std::string &str )
{
    if( str.size() == 0 )
    {
        return;
    }

    if( str[0] != '\"' && str[0] != ' ' && str.back() != ' ' ) // get out fast if we can
    {
        return;
    }

    // strip white space at front
    size_t ns = str.find_first_not_of( " \t\r\n" );
    if( ns != std::string::npos && ns != 0 )
    {
        str.erase( 0, ns );

        if( str.size() == 0 )
        {
            return;
        }
    }
    else if( ns == std::string::npos ) // the rare all spaces
    {
        str = "";
        return;
    }

    // strip white space at back
    ns = str.find_last_not_of( " \t\r\n" );
    if( ns != std::string::npos && ns != str.size() - 1 )
    {
        str.erase( ns + 1 );

        if( str.size() == 0 )
        {
            return;
        }
    }

    if( str[0] == '\"' && str.back() == '\"' )
    {
        if( str.size() == 1 || str.size() == 2 )
        {
            str = "";
            return;
        }
        str.erase( str.size() - 1, 1 );
        str.erase( 0, 1 );
    }
    else if( str[0] == '\"' )
    {
        if( str.size() == 1 )
        {
            str = "";
            return;
        }
        str.erase( 0, 1 );
    }
    else if( str.back() == '\"' )
    {
        if( str.size() == 1 || str.size() == 2 )
        {
            str = "";
            return;
        }
        str.erase( str.size() - 1, 1 );
    }
}
/// \endcond

/// Load the rule and properties maps for a rule engine from a configuration file
/** ///\todo check for insertion failure
 * ///\todo add a constructor that has priority, message, and comparison, to reduce duplication
 */
void loadRuleConfig( indiRuleMaps &maps,                          /**< [out] contains the rule and property maps in
                                                                             which to place the items found in config */
                     std::map<std::string, ruleRuleKeys> &rrkMap, /**< [out] Holds the ruleVal rule keys aside for
                                                                             later post-processing*/
                     mx::app::appConfigurator &config             /**< [in] the application configuration structure */
)
{
    std::vector<std::string> sections;

    config.unusedSections( sections );

    if( sections.size() == 0 )
    {
        throw mx::exception( mx::error_t::invalidconfig, "no rules found in config" );
    }

    for( size_t i = 0; i < sections.size(); ++i )
    {
        bool ruleTypeSet = config.isSetUnused( mx::app::iniFile::makeKey( sections[i], "ruleType" ) );

        // If there is no ruleType then this isn't a rule
        if( !ruleTypeSet )
        {
            continue;
        }

        // If the rule already exists this is an error
        if( maps.rules.count( sections[i] ) != 0 )
        {
            throw mx::exception( mx::error_t::invalidconfig, "duplicate rule: " + sections[i] );
        }

        std::string ruleType;
        config.configUnused( ruleType, mx::app::iniFile::makeKey( sections[i], "ruleType" ) );

        std::string priostr = "none";
        config.configUnused( priostr, mx::app::iniFile::makeKey( sections[i], "priority" ) );
        rulePriority priority = string2priority( priostr );

        std::string message;
        config.configUnused( message, mx::app::iniFile::makeKey( sections[i], "message" ) );
        stripQuotesWS( message ); // strips "" and any leading/trailing whitespace

        auto configureRuleBase = [&]( indiCompRule *rule )
        {
            std::string compstr = comp2string( rule->defaultComparison() );
            config.configUnused( compstr, mx::app::iniFile::makeKey( sections[i], "comp" ) );

            rule->priority( priority );
            rule->message( message );
            rule->comparison( string2comp( compstr ) );
        };

        if( ruleType == numValRule::name )
        {
            numValRule *nvr = new numValRule;
            maps.rules.insert( std::pair<std::string, indiCompRule *>( { sections[i], nvr } ) );

            configureRuleBase( nvr );

            pcf::IndiProperty *prop = nullptr;
            std::string        element;

            extractRuleProp(
                &prop, element, maps, sections[i], "property", "element", pcf::IndiProperty::Number, config );
            nvr->property( prop );
            nvr->element( element );

            double target = nvr->target();
            config.configUnused( target, mx::app::iniFile::makeKey( sections[i], "target" ) );
            nvr->target( target );

            double tol = nvr->tol();
            config.configUnused( tol, mx::app::iniFile::makeKey( sections[i], "tol" ) );
            nvr->tol( tol );
        }
        else if( ruleType == txtValRule::name )
        {
            txtValRule *tvr = new txtValRule;
            maps.rules.insert( std::pair<std::string, indiCompRule *>( { sections[i], tvr } ) );

            configureRuleBase( tvr );

            pcf::IndiProperty *prop = nullptr;
            std::string        element;

            extractRuleProp(
                &prop, element, maps, sections[i], "property", "element", pcf::IndiProperty::Text, config );
            tvr->property( prop );
            tvr->element( element );

            std::string target = tvr->target();
            config.configUnused( target, mx::app::iniFile::makeKey( sections[i], "target" ) );
            tvr->target( target );
        }
        else if( ruleType == swValRule::name )
        {
            swValRule *svr = new swValRule;
            maps.rules.insert( std::pair<std::string, indiCompRule *>( { sections[i], svr } ) );

            configureRuleBase( svr );

            pcf::IndiProperty *prop = nullptr;
            std::string        element;

            extractRuleProp(
                &prop, element, maps, sections[i], "property", "element", pcf::IndiProperty::Switch, config );
            svr->property( prop );
            svr->element( element );

            std::string target = "On";
            config.configUnused( target, mx::app::iniFile::makeKey( sections[i], "target" ) );
            svr->target( target );
        }
        else if( ruleType == timeDiffRule::name )
        {
            timeDiffRule *nvr = new timeDiffRule;
            maps.rules.insert( std::pair<std::string, indiCompRule *>( { sections[i], nvr } ) );

            configureRuleBase( nvr );

            pcf::IndiProperty *prop = nullptr;
            std::string        element;

            extractRuleProp(
                &prop, element, maps, sections[i], "property", "element", pcf::IndiProperty::Number, config );
            nvr->property( prop );
            nvr->element( element );

            double target = nvr->target();
            config.configUnused( target, mx::app::iniFile::makeKey( sections[i], "target" ) );
            nvr->target( target );

            double tol = nvr->tol();
            config.configUnused( tol, mx::app::iniFile::makeKey( sections[i], "tol" ) );
            nvr->tol( tol );
        }
        else if( ruleType == elCompNumRule::name )
        {
            elCompNumRule *nvr = new elCompNumRule;
            maps.rules.insert( std::pair<std::string, indiCompRule *>( { sections[i], nvr } ) );

            configureRuleBase( nvr );

            pcf::IndiProperty *prop1;
            std::string        element1;

            extractRuleProp(
                &prop1, element1, maps, sections[i], "property1", "element1", pcf::IndiProperty::Number, config );
            nvr->property1( prop1 );
            nvr->element1( element1 );

            pcf::IndiProperty *prop2;
            std::string        element2;

            extractRuleProp(
                &prop2, element2, maps, sections[i], "property2", "element2", pcf::IndiProperty::Number, config );
            nvr->property2( prop2 );
            nvr->element2( element2 );
        }
        else if( ruleType == elCompTxtRule::name )
        {
            elCompTxtRule *tvr = new elCompTxtRule;
            maps.rules.insert( std::pair<std::string, indiCompRule *>( { sections[i], tvr } ) );

            configureRuleBase( tvr );

            pcf::IndiProperty *prop1;
            std::string        element1;

            extractRuleProp(
                &prop1, element1, maps, sections[i], "property1", "element1", pcf::IndiProperty::Text, config );
            tvr->property1( prop1 );
            tvr->element1( element1 );

            pcf::IndiProperty *prop2;
            std::string        element2;

            extractRuleProp(
                &prop2, element2, maps, sections[i], "property2", "element2", pcf::IndiProperty::Text, config );
            tvr->property2( prop2 );
            tvr->element2( element2 );
        }
        else if( ruleType == elCompSwRule::name )
        {
            elCompSwRule *svr = new elCompSwRule;
            maps.rules.insert( std::pair<std::string, indiCompRule *>( { sections[i], svr } ) );

            configureRuleBase( svr );

            pcf::IndiProperty *prop1;
            std::string        element1;

            extractRuleProp(
                &prop1, element1, maps, sections[i], "property1", "element1", pcf::IndiProperty::Switch, config );
            svr->property1( prop1 );
            svr->element1( element1 );

            pcf::IndiProperty *prop2;
            std::string        element2;

            extractRuleProp(
                &prop2, element2, maps, sections[i], "property2", "element2", pcf::IndiProperty::Switch, config );
            svr->property2( prop2 );
            svr->element2( element2 );
        }
        else if( ruleType == ruleCompRule::name )
        {
            // Here we have to hold the ruleVal keys separately for later processing after all the rules are created.

            if( rrkMap.count( sections[i] ) > 0 )
            {
                // This probably should be impossible, since we already checked maps.rules above...
                throw mx::exception( mx::error_t::invalidconfig, "duplicate ruleRule: " + sections[i] );
            }

            ruleCompRule *rcr = new ruleCompRule;
            maps.rules.insert( std::pair<std::string, indiCompRule *>( { sections[i], rcr } ) );

            configureRuleBase( rcr );

            ruleRuleKeys rrk;

            config.configUnused( rrk.rule1, mx::app::iniFile::makeKey( sections[i], "rule1" ) );
            if( rrk.rule1 == "" )
            {
                throw mx::exception( mx::error_t::invalidconfig,
                                     "rule1 for ruleVal rule " + sections[i] + " not found" );
            }
            if( rrk.rule1 == sections[i] )
            {
                throw mx::exception( mx::error_t::invalidconfig,
                                     "rule1 for ruleVal rule " + sections[i] + " can't equal rule name" );
            }

            config.configUnused( rrk.rule2, mx::app::iniFile::makeKey( sections[i], "rule2" ) );
            if( rrk.rule2 == "" )
            {
                throw mx::exception( mx::error_t::invalidconfig,
                                     "rule2 for ruleVal rule " + sections[i] + " not found" );
            }
            if( rrk.rule2 == sections[i] )
            {
                throw mx::exception( mx::error_t::invalidconfig,
                                     "rule2 for ruleVal rule " + sections[i] + " can't equal rule name" );
            }

            rrkMap.insert( std::pair<std::string, ruleRuleKeys>( sections[i], rrk ) );
        }
        else if( ruleType == multiSwitchComboRule::name )
        {
            multiSwitchComboRule *mscr = new multiSwitchComboRule;
            maps.rules.insert( std::pair<std::string, indiCompRule *>( { sections[i], mscr } ) );

            configureRuleBase( mscr );
            mscr->ruleName( sections[i] );

            int numSwitches = 0;
            config.configUnused( numSwitches, mx::app::iniFile::makeKey( sections[i], "numSwitches" ) );
            if( numSwitches < 1 )
            {
                throw mx::exception(
                    mx::error_t::invalidconfig,
                    std::format( "numSwitches for multiSwitchCombo rule {} must be greater than zero", sections[i] ) );
            }

            for( int n = 0; n < numSwitches; ++n )
            {
                pcf::IndiProperty *prop = nullptr;
                std::string        property;

                std::string propKey = std::format( "property{}", n + 1 );
                extractRuleProperty( &prop, property, maps, sections[i], propKey, pcf::IndiProperty::Switch, config );

                mscr->property( prop, property );
            }

            std::string format;
            config.configUnused( format, mx::app::iniFile::makeKey( sections[i], "format" ) );
            stripQuotesWS( format );
            mscr->format( format );

            pcf::IndiProperty *targetProp = nullptr;
            std::string        targetProperty;
            extractRuleProperty(
                &targetProp, targetProperty, maps, sections[i], "targetProperty", pcf::IndiProperty::Switch, config );
            mscr->targetProperty( targetProp );
            mscr->targetPropertyKey( targetProperty );

            indiCompRule::boolorerr_t rv = mscr->valid();
            if( mscr->isError( rv ) )
            {
                throw mx::exception( mx::error_t::invalidconfig,
                                     std::format( "multiSwitchCombo rule {} is invalid: {}",
                                                  sections[i],
                                                  std::get<std::string>( rv ) ) );
            }
        }
        else
        {
            throw mx::exception( mx::error_t::notimpl,
                                 std::format( "unknown rule type {} in {}", ruleType, sections[i] ) );
        }
    }
}

/// Finalize ruleVal rules
/** ///\todo check for insertion failure
 * ///\todo add a constructor that has priority, message, and comparison, to reduce duplication
 */
void finalizeRuleValRules(
    indiRuleMaps &maps, /**< [in/out] contains the rule and property maps with rules ot finalize */
    std::map<std::string, ruleRuleKeys> &rrkMap ///< [out] Holds the ruleVal rule keys aside for later post-processing
)
{
    // Now set the rule pointers for any ruleVal rules
    auto it = rrkMap.begin();
    while( it != rrkMap.end() )
    {
        if( maps.rules.count( it->first ) == 0 )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::format( "rule parsing error for {}", it->first ) );
        }

        if( maps.rules.count( it->second.rule1 ) == 0 )
        {
            throw mx::exception( mx::error_t::invalidconfig,
                                 std::format( "rule1 {} not found "
                                              "for ruleVal rule {}",
                                              it->second.rule1,
                                              it->first ) );
        }

        if( maps.rules.count( it->second.rule2 ) == 0 )
        {
            throw mx::exception( mx::error_t::invalidconfig,
                                 std::format( "rule2 {} not found "
                                              "for ruleVal rule {}",
                                              it->second.rule2,
                                              it->first ) );
        }

        ruleCompRule *rcr = nullptr;

        try
        {
            rcr = dynamic_cast<ruleCompRule *>( maps.rules[it->first] );
        }
        catch( const std::exception &e )
        {
            std::throw_with_nested(
                mx::exception( mx::error_t::invalidconfig, std::format( "error casting {}", it->first ) ) );
        }

        if( rcr == nullptr )
        {
            throw mx::exception( mx::error_t::invalidconfig,
                                 std::format( "{} is not a ruleVal rule but has rules", it->first ) );
        }

        rcr->rule1( maps.rules[it->second.rule1] );
        rcr->rule2( maps.rules[it->second.rule2] );

        ++it;
    }
}

#endif // stateRuleEngine_indiCompRuleConfig_hpp
