/** \file pwrOnOffNode.hpp
 * \brief The MagAO-X Instrument Graph pwrOnOffNode header file
 *
 * \ingroup instGraph_files
 */

#ifndef pwrOnOffNode_hpp
#define pwrOnOffNode_hpp

#include "xigNode.hpp"

class pwrOnOffNode : public xigNode
{

  protected:
    std::string m_pwrKey;
    int         m_pwrState{ -1 };

  public:
    pwrOnOffNode( const std::string &name, ingr::instGraphXML *parentGraph ) : xigNode( name, parentGraph )
    {
        if( m_parentGraph )
        {
            m_parentGraph->valueExtra( m_node->name(), "fsmstate", "---" );
            m_parentGraph->valueExtra( m_node->name(), "state", "" );
        }
    }

    void pwrKey( const std::string &pk );

    const std::string & pwrKey() const;

    /// INDI SetProperty callback
    virtual int handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the received INDI property to handle*/ );

    virtual void toggleOn();

    virtual void toggleOff();

    void loadConfig( mx::app::appConfigurator &config );
};

inline void pwrOnOffNode::pwrKey( const std::string &pk )
{
    m_pwrKey = pk;

    key( m_pwrKey );
}

inline const std::string & pwrOnOffNode::pwrKey() const
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

    if( ipRecv["state"].get<std::string>() == "On" )
    {
        toggleOn();
        return 0;
    }
    else
    {
        toggleOff();
        return 0;
    }
}

inline void pwrOnOffNode::toggleOn()
{
    m_pwrState = 1;

    togglePutsOn();

    if( m_parentGraph )
    {
        std::cerr << "writing\n";
        m_parentGraph->valueExtra( m_node->name(), "fsmstate", "ON" );
    }
}

inline void pwrOnOffNode::toggleOff()
{
    m_pwrState = 1;

    togglePutsOff();

    if( m_parentGraph )
    {
        m_parentGraph->valueExtra( m_node->name(), "fsmstate", "OFF" );
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
