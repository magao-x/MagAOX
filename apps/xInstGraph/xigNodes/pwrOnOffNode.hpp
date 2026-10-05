/** \file pwrOnOffNode.hpp
 * \brief The MagAO-X Instrument Graph pwrOnOffNode header file
 *
 * \ingroup instGraph_files
 */

#ifndef pwrOnOffNode_hpp
#define pwrOnOffNode_hpp

#include "xigNode.hpp"

/// Graph node that follows a power controller channel's text state.
class pwrOnOffNode : public xigNode
{
  protected:
    /// INDI device.property key for the channel state.
    std::string m_pwrKey;

  public:
    /// Construct a power node for an existing graph node.
    pwrOnOffNode( const std::string  &name, /**< [in] graph node name */
                  ingr::instGraphXML *parentGraph /**< [in] parent graph */ );

    /// Set the INDI power-channel property key.
    void pwrKey( const std::string &pk /**< [in] device.property key */ );

    /// Get the INDI power-channel property key.
    const std::string &pwrKey() const;

    /// Apply an INDI channel state update.
    virtual int handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] received INDI property */ );

    /// Turn on the node puts and display ON.
    virtual void toggleOn();

    /// Turn off the node puts and display OFF.
    virtual void toggleOff();

    /// Turn off the node puts and display an unresolved power state.
    void toggleUnknown( const std::string &label /**< [in] INT or UNK display text */ );

    /// Load the required power property key from configuration.
    void loadConfig( mx::app::appConfigurator &config /**< [in] node configuration */ );
};

inline pwrOnOffNode::pwrOnOffNode( const std::string &name, ingr::instGraphXML *parentGraph )
    : xigNode( name, parentGraph )
{
    if( m_parentGraph )
    {
        m_parentGraph->valueExtra( m_node->name(), "fsmstate", "---" );
        m_parentGraph->valueExtra( m_node->name(), "state", "" );
    }
}

inline void pwrOnOffNode::pwrKey( const std::string &pk )
{
    m_pwrKey = pk;

    key( m_pwrKey );
}

inline const std::string &pwrOnOffNode::pwrKey() const
{
    return m_pwrKey;
}

inline int pwrOnOffNode::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_pwrKey )
    {
        return -1;
    }

    if( !ipRecv.find( "state" ) )
    {
        return -1;
    }

    const std::string state = ipRecv["state"].get<std::string>();
    if( state == "On" )
    {
        toggleOn();
    }
    else if( state == "Off" )
    {
        toggleOff();
    }
    else
    {
        toggleUnknown( state == "Int" ? "INT" : "UNK" );
    }
    return 0;
}

inline void pwrOnOffNode::toggleOn()
{
    togglePutsOn();
    if( m_parentGraph )
    {
        m_parentGraph->valueExtra( m_node->name(), "fsmstate", "ON" );
    }
}

inline void pwrOnOffNode::toggleOff()
{
    togglePutsOff();
    if( m_parentGraph )
    {
        m_parentGraph->valueExtra( m_node->name(), "fsmstate", "OFF" );
    }
}

inline void pwrOnOffNode::toggleUnknown( const std::string &label )
{
    togglePutsOff();
    if( m_parentGraph )
    {
        m_parentGraph->valueExtra( m_node->name(), "fsmstate", label );
    }
}

inline void pwrOnOffNode::loadConfig( mx::app::appConfigurator &config )
{
    if( !m_parentGraph )
    {
        std::string msg = "pwrOnOffNode::loadConfig: parent graph is null";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );

        throw std::runtime_error( msg );
    }

    std::string type;
    config.configUnused( type, mx::app::iniFile::makeKey( name(), "type" ) );

    if( type != "pwrOnOff" )
    {
        std::string msg = "pwrOnOffNode::loadConfig: node type is not pwrOnOff";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    std::string pk;
    config.configUnused( pk, mx::app::iniFile::makeKey( name(), "pwrKey" ) );

    if( pk == "" )
    {
        std::string msg = "pwrOnOffNode::loadConfig: pwrKey can not be empty";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );

        throw std::runtime_error( msg );
    }

    pwrKey( pk );
}

#endif // pwrOnOffNode_hpp
