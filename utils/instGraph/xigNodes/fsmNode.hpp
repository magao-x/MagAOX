/** \file fsmNode.hpp
 * \brief The MagAO-X Instrument Graph fsmNode header file
 *
 * \ingroup instGraph_files
 */

#ifndef fsmNode_hpp
#define fsmNode_hpp

#include "xigNode.hpp"

class fsmNode : public xigNode
{

  protected:
    std::string m_device;
    std::string m_fsmKey;

    MagAOX::app::stateCodes::stateCodeT m_state{ -999 };
    std::string m_stateStr;

  public:
    fsmNode( const std::string &name, ingr::instGraphXML *parentGraph );

    virtual void device( const std::string &dev );

    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv );

    virtual void updateGUI();
};

inline fsmNode::fsmNode( const std::string &name, ingr::instGraphXML *parentGraph ) : xigNode( name, parentGraph )
{
}

inline void fsmNode::device( const std::string &dev )
{
    if( m_device != "" && dev != m_device )
    {
        std::string msg = "attempt to change device name from " + m_device + " to " + dev;
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    m_device = dev;
    m_fsmKey = m_device + ".fsm";

    key( m_fsmKey );
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
