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
    int m_pwrState{ -1 };

  public:
    pwrOnOffNode( const std::string &name, ingr::instGraphXML *parentGraph ) : xigNode( name, parentGraph )
    {
    }

    void pwrKey( const std::string &pk );

    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv );

    virtual void toggleOn();

    virtual void toggleOff();

    void loadConfig( mx::app::appConfigurator &config );
};

inline
void pwrOnOffNode::pwrKey( const std::string &pk )
{
    m_pwrKey = pk;

    key( m_pwrKey );
}

inline
void pwrOnOffNode::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_pwrKey )
    {
        return;
    }

    if( !ipRecv.find( "state" ) )
    {
        return;
    }

    if( ipRecv["state"].get<std::string>() == "On" )
    {
        return toggleOn();
    }
    else
    {
        return toggleOff();
    }
}

inline
void pwrOnOffNode::toggleOn()
{
    m_pwrState = 1;

    togglePutsOn();
}

inline
void pwrOnOffNode::toggleOff()
{
    m_pwrState = 1;

    togglePutsOff();
}

inline
void pwrOnOffNode::loadConfig( mx::app::appConfigurator &config )
{
    if( !nodeValid() )
    {
        std::string msg = "pwrOnOffNode::loadConfig: node is not valid";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );

        throw std::runtime_error(msg);
    }

    std::string pk;
    config.configUnused( pk, mx::app::iniFile::makeKey( name(), "pwrKey" ) );

    if( pk == "" )
    {
        std::string msg = "pwrOnOffNode::loadConfig: pwrKey can not be empty";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );

        throw std::runtime_error(msg);
    }

    pwrKey( pk );

}

#endif // pwrOnOffNode_hpp
