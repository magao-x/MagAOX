/** \file fsmNode.hpp
 * \brief The MagAO-X Instrument Graph fsmNode header file
 *
 * \ingroup instGraph_files
 */

#ifndef fsmNode_hpp
#define fsmNode_hpp

#include "xigNode.hpp"

enum class fsmNodeActionT
{
    passive,   /**< Only monitor and report FSM state, don't change puts*/
    threshOff, /**< If state is not in one of the specified states, turn puts off*/
    active,    /**< If state is in one of the specified states, turn puts on. Turn them off otherwise.*/
    unknown    /**< Unknown action, generally an error. */
};

std::string fsmNodeActionT2String( fsmNodeActionT action )
{
    if( action == fsmNodeActionT::passive )
    {
        return "passive";
    }
    else if( action == fsmNodeActionT::threshOff )
    {
        return "threshOff";
    }
    else if( action == fsmNodeActionT::active )
    {
        return "active";
    }
    else
    {
        return "";
    }
}

fsmNodeActionT fsmNodeActionTFromString( const std::string &action )
{
    if( action == "passive" )
    {
        return fsmNodeActionT::passive;
    }
    else if( action == "threshOff" )
    {
        return fsmNodeActionT::threshOff;
    }
    else if( action == "active" )
    {
        return fsmNodeActionT::active;
    }
    else
    {
        return fsmNodeActionT::unknown;
    }
}

/// Implementation of an instGraph node interface for a MagAO-X Finite State Machine (FSM)
/** This class is interraces to a standard FSM.  It tracks the FSM state INDI property
 * and keeps its internal state updated.
 *
 * Whether it impacts ioput status depends on the `action` specified.
 *
 */

class fsmNode : public xigNode
{

    typedef MagAOX::app::stateCodes::stateCodeT stateCodeT;

  protected:
    std::string m_device;               ///< The INDI device name. Defaults to the node name set on construction.
    std::string m_fsmPropName{ "fsm" }; ///< The INDI property name for the FSM, normally "fsm".
    std::string m_fsmElName{ "state" }; ///< The INDI property element name for the FSM, normally "state".

    std::string m_fsmKey; ///< The unique INDI key, `<device>.<fsmPropName>`, for the FSM state INDI property.

    fsmNodeActionT m_fsmAction{ fsmNodeActionT::passive };

    std::vector<stateCodeT> m_targetStates;

    stateCodeT  m_state{ -999 }; ///< The numerical code of the current state.
    std::string m_stateStr;      ///< The string name of the current state.

    bool m_stateOnTarget{ false }; ///< Flag indicating if the current state matches any of the target states.

  public:
    /// Constructor.
    /**
     * Default c'tor is deleted in base classs.  Must supply both node name and a parentGraph with a node with the same
     * name in it.
     */
    fsmNode( const std::string  &name,       /**< [in] the name of the node */
             ingr::instGraphXML *parentGraph /**< [in] the parent instGraph */
    );

    /// Set the device name
    /**
     * Derived classes may implement this to add extra logic.  The device name defaults
     * to the node name on construction.
     */
    virtual void device( const std::string &dev /**< [in] the new device name */ );

    /// Get the device name
    /**
     * \return the current value of m_device
     */
    const std::string &device() const;

    /// Set the fsm property name
    /**
     * Derived classes may implement this to add extra logic.  The fsm property name defaults
     * to "fsm"
     *
     * This can only be called before device is set
     */
    virtual void fsmPropName( const std::string &pn /**< [in] the new property name */ );

    /// Get the fsm property name
    /**
     * \return the current value of m_fsmPropName
     */
    const std::string &fsmPropName() const;

    /// Set the fsm element name
    /**
     * Derived classes may implement this to add extra logic.  The fsm element name defaults
     * to "state"
     *
     * This can be called at any time
     */
    virtual void fsmElName( const std::string &en /**< [in] the new element name */ );

    /// Get the fsm element name 
    /**
     * \return the current value of m_fsmElName
     */
    const std::string &fsmElName() const;

    /// Get the FSM unique key
    /**
     * \return the current value of m_fsmKey
     */
    const std::string &fsmKey() const;

    /// Get the action
    /**
     * \return the current value of m_fsmAction
     */
    fsmNodeActionT fsmAction() const;

    /// Set the action
    void fsmAction( fsmNodeActionT act );

    /// Get the target states
    /**
     * \return the current value of m_targetStates
     */
    const std::vector<stateCodeT> &targetStates() const;

    /// Load this specific node's settings from an application configuration
    /**
     * Verifies that the named node is an fsmNode.
     *
     * \throws std::runtime_error if m_parentGraph is nullptr or the config is not for an fsmNode.
     */
    void loadConfig( mx::app::appConfigurator &config /**< [in] the application configurator
                                                                loaded with this node's options*/ );

  protected:
    /// Load this specific node's settings from an application configuration of a derived class
    /**
     * Does not cerifies that the named node is an fsmNode.
     *
     */
    void loadConfigDerived( mx::app::appConfigurator &config /**< [in] the application configurator
                                                                loaded with this node's options*/ );

  public:
    /// INDI SetProperty callback
    virtual int handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the received INDI property to handle*/ );

    /// INDI SetProperty callback with indication if action was taken
    /** The possible actions are determined by m_fsmAction.  If the action was taken then the caller
     *  should return without further processing.
     *
     */
    virtual int handleSetProperty( bool &actionTaken, /** < [out] indicates if action taken (true). */
                                   const pcf::IndiProperty &ipRecv /**< [in] the received INDI property to handle*/ );

  public:
    virtual void updateGUI();
};

inline fsmNode::fsmNode( const std::string &name, ingr::instGraphXML *parentGraph ) : xigNode( name, parentGraph )
{
}

inline void fsmNode::device( const std::string &dev )
{
    if( m_device != "" && dev != m_device )
    {
        std::string msg = "fsmNode::device attempt to change device name from " + m_device + " to " + dev;
        msg += " in " + name();
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    if( dev == "" )
    {
        std::string msg = "fsmNode::device attempt to set empty device name in " + name();
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    m_device = dev;
    m_fsmKey = m_device + '.' + m_fsmPropName;

    std::cerr << name() << ' ' << m_fsmKey << '\n';

    key( m_fsmKey );
}

inline const std::string &fsmNode::device() const
{
    return m_device;
}

inline void fsmNode::fsmPropName( const std::string &pn )
{
    if( m_fsmPropName != "" && m_device != "" )
    {
        std::string msg = "fsmNode::fsmPropName attempt to change fsmPropName name from " + m_fsmPropName + " to " + pn;
        msg += " in " + name();
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    if( m_device != "" )
    {
        std::string msg = "fsmNode::fsmPropName attempt to set propName after device already set " + name();
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    if( pn == "" )
    {
        std::string msg = "fsmNode::fsmPropName attempt to set propName to empty " + name();
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    m_fsmPropName = pn;
}

inline const std::string &fsmNode::fsmPropName() const
{
    return m_fsmPropName;
}

inline void fsmNode::fsmElName( const std::string &en )
{
    if( en == "" )
    {
        std::string msg = "fsmNode::fsmElName attempt to set elName to empty " + name();
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    m_fsmElName = en;
}

inline const std::string &fsmNode::fsmElName() const
{
    return m_fsmElName;
}

const std::string &fsmNode::fsmKey() const
{
    return m_fsmKey;
}

fsmNodeActionT fsmNode::fsmAction() const
{
    return m_fsmAction;
}

void fsmNode::fsmAction( fsmNodeActionT act )
{
    m_fsmAction = act;
}

const std::vector<fsmNode::stateCodeT> &fsmNode::targetStates() const
{
    return m_targetStates;
}

inline void fsmNode::loadConfig( mx::app::appConfigurator &config )
{
    if( !m_parentGraph )
    {
        std::string msg = "fsmNode::loadConfig: parent graph is null";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    std::string type;
    config.configUnused( type, mx::app::iniFile::makeKey( name(), "type" ) );

    if( type != "fsm" )
    {
        std::string msg = "fsmNode::loadConfig: node type is not fsmNode";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    loadConfigDerived( config );
}

inline void fsmNode::loadConfigDerived( mx::app::appConfigurator &config )
{
    //This must be first
    std::string propName = fsmPropName();
    config.configUnused( propName, mx::app::iniFile::makeKey( name(), "fsmPropName" ) );
    fsmPropName( propName );

    std::string dev = name();
    config.configUnused( dev, mx::app::iniFile::makeKey( name(), "device" ) );
    device( dev );

    std::string elName = fsmElName();
    config.configUnused( elName, mx::app::iniFile::makeKey( name(), "fsmElName" ) );
    fsmElName( elName );

    std::string action = fsmNodeActionT2String( m_fsmAction );
    config.configUnused( action, mx::app::iniFile::makeKey( name(), "fsmAction" ) );
    m_fsmAction = fsmNodeActionTFromString( action );

    if( m_fsmAction == fsmNodeActionT::unknown )
    {
        std::string msg = XIGN_EXCEPTION( "fsmNode::loadConfig", "fsmAction is unknown" );
        throw std::runtime_error( msg );
    }

    std::vector<std::string> targetStates;
    config.configUnused( targetStates, mx::app::iniFile::makeKey( name(), "targetStates" ) );
    m_targetStates.resize( targetStates.size() );
    for( size_t n = 0; n < targetStates.size(); ++n )
    {
        m_targetStates[n] = MagAOX::app::stateCodes::str2Code( targetStates[n] );
    }

    if( m_parentGraph && m_node )
    {
        m_parentGraph->valueExtra( m_node->name(), "state", "" );
    }
}

inline int fsmNode::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    bool actionTaken;
    return handleSetProperty( actionTaken, ipRecv );
}

inline int fsmNode::handleSetProperty( bool &actionTaken, const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_fsmKey )
    {
        actionTaken = false;
        return 0;
    }

    if( !ipRecv.find( m_fsmElName ) )
    {
        actionTaken = false;
        return 0;
    }

    m_stateStr = ipRecv[m_fsmElName].get<std::string>();

    MagAOX::app::stateCodes::stateCodeT state = MagAOX::app::stateCodes::str2CodeFast( m_stateStr );

    if( state != m_state )
    {
        ++m_changes;
    }

    m_state = state;

    m_parentGraph->valueExtra( m_node->name(), "fsmstate", m_stateStr );

    bool stateOnTarget = false;

    for( auto state : m_targetStates )
    {
        if( m_state == state )
        {
            stateOnTarget = true;
            break;
        }
    }
    m_stateOnTarget = stateOnTarget;

    if( m_fsmAction == fsmNodeActionT::threshOff )
    {
        if( m_stateOnTarget )
        {
            actionTaken = false;
            return 0;
        }
        else
        {
            togglePutsOff();
            actionTaken = true;
            return 0;
        }
    }
    else if( m_fsmAction == fsmNodeActionT::active )
    {
        if( m_stateOnTarget )
        {
            togglePutsOn();
            actionTaken = true;
            return 0;
        }
        else
        {
            togglePutsOff();
            actionTaken = true;
            return 0;
        }
    }
    else // m_fsmAction == fsmNodeActionT::passive
    {
        actionTaken = false;
        return 0;
    }
}

inline void fsmNode::updateGUI()
{
}

#endif // fsmNode_hpp
