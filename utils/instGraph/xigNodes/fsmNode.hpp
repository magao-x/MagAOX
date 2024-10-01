/** \file fsmNode.hpp
 * \brief The MagAO-X Instrument Graph fsmNode header file
 *
 * \ingroup instGraph_files
 */

#ifndef fsmNode_hpp
#define fsmNode_hpp

#include "xigNode.hpp"

/// Implementation of an instGraph node interface for a MagAO-X Finite State Machine (FSM)
/** This class is interraces to a standard FSM.  It tracks the FSM state INDI property
  * and keeps its internal state updated.
  *
  * It currently does not set ioput status.  This must be inherited by a class which does so.
  */
class fsmNode : public xigNode
{

  protected:
    std::string m_device; ///< The INDI device name. Defaults to the node name set on construction.
    std::string m_fsmKey; ///< The unique INDI key, `<device>.fsm`, for the FSM state INDI property.

    MagAOX::app::stateCodes::stateCodeT m_state{ -999 }; ///< The numerical code of the current state.
    std::string m_stateStr; ///< The string name of the current state.

  public:

    /// Constructor.
    /**
      * Default c'tor is deleted in base classs.  Must supply both node name and a parentGraph with a node with the same name in it.
      */
    fsmNode( const std::string &name,        /**< [in] the name of the node */
             ingr::instGraphXML *parentGraph /**< [in] the parent instGraph */
           );

    /// Set the device name
    /**
      * Derived classes may implement this to add extra logic.  The device name defaults
      * to the node name on construction.
      */
    virtual void device( const std::string &dev /**< [in] the new device name */);

    /// Get the device name
    /**
     * \return the current value of m_device
     */
    const std::string & device();

    /// Load this specific node's settings from an application configuration
    /**
     * Verifies that the named node is an fsmNode.
     *
     * \throws std::runtime_error if m_parentGraph is nullptr or the config is not for an fsmNode.
     */
    void loadConfig( mx::app::appConfigurator &config /**< [in] the application configurator loaded with this node's options*/);

    /// INDI SetProperty callback
    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the received INDI property to handle*/ );

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
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    m_device = dev;
    m_fsmKey = m_device + ".fsm";

    key( m_fsmKey );
}

inline const std::string & fsmNode::device()
{
    return m_device;
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
    config.configUnused(type, mx::app::iniFile::makeKey( name(), "type" ));

    if(type != "fsm")
    {
        std::string msg = "fsmNode::loadConfig: node type is not fsmNode";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    std::string dev = name();
    config.configUnused( dev, mx::app::iniFile::makeKey( name(), "device" ) );

    device( dev );
}

inline void fsmNode::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_fsmKey )
    {
        return;
    }

    if( !ipRecv.find( "state" ) )
    {
        return;
    }

    m_stateStr = ipRecv["state"].get<std::string>();

    MagAOX::app::stateCodes::stateCodeT state = MagAOX::app::stateCodes::str2CodeFast( m_stateStr );

    if( state != m_state )
    {
        ++m_changes;
    }

    m_state = state;
}

inline void fsmNode::updateGUI()
{
}

#endif // fsmNode_hpp
