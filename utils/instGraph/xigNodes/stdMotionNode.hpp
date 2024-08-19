/** \file stdMotionNode.hpp
 * \brief The MagAO-X Instrument Graph stdMotionNode header file
 *
 * \ingroup instGraph_files
 */

#ifndef stdMotionNode_hpp
#define stdMotionNode_hpp

#include "xigNode.hpp"

class stdMotionNode : public fsmNode
{

  protected:
    std::string m_presetPrefix;
    std::string m_presetKey;

    std::string m_curVal;

    std::vector<std::string> m_presetPutName{ "out" };

    /// This sets whether the multi-put selector is on the input or the output (default)
    /** If this is a multi-put node (m_presetPutName.size() > 1) then the value of the preset switch
     * controls which input or output is on, with the others off.
     */
    ingr::ioDir m_presetDir{ ingr::ioDir::output };

  public:
    stdMotionNode( const std::string &name, ingr::instGraphXML *parentGraph ) : fsmNode( name, parentGraph )
    {
    }

    virtual void device( const std::string &dev );

    virtual void presetPrefix( const std::string &pp );

    void presetPutName( const std::vector<std::string> &ppp );

    void presetDir( const ingr::ioDir &dir );

    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv );

    virtual void togglePutsOn();

    virtual void togglePutsOff();

    void loadConfig( mx::app::appConfigurator &config );
};

inline void stdMotionNode::device( const std::string &dev )
{
    // This will enforce the one-time only rule
    fsmNode::device( dev );

    // If presetPrefix is set, then we can make the key
    if( m_presetPrefix != "" )
    {
        m_presetKey = m_device + "." + m_presetPrefix + "Name";
        key( m_presetKey );
    }
}

inline void stdMotionNode::presetPrefix( const std::string &pp )
{
    // Set it one time only
    if( m_presetPrefix != "" && pp != m_presetPrefix )
    {
        std::string msg = "attempt to change preset prefix from " + m_presetPrefix + " to " + pp;
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    m_presetPrefix = pp;

    // If device has been set then we can create the key
    if( m_device != "" )
    {
        m_presetKey = m_device + "." + m_presetPrefix + "Name";
        key( m_presetKey );
    }
}

inline void stdMotionNode::presetPutName( const std::vector<std::string> &ppp )
{
    m_presetPutName = ppp;
}

inline void stdMotionNode::presetDir( const ingr::ioDir &dir )
{
    m_presetDir = dir;
}

inline void stdMotionNode::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    std::cerr << name() << ": handleSetProperty=" << ipRecv.createUniqueKey() << "\n";

    fsmNode::handleSetProperty( ipRecv );

    if( ipRecv.createUniqueKey() == m_presetKey )
    {
        std::cerr << "got " << ipRecv.createUniqueKey() << "\n";

        if( m_node != nullptr )
        {
            for( auto &&it : ipRecv.getElements() )
            {
                if( it.second.getSwitchState() == pcf::IndiElement::On )
                {
                    if( m_curVal != it.second.getName() )
                    {
                        ++m_changes;
                    }

                    m_curVal = it.second.getName();
                }
            }
        }
    }

    if( m_changes > 0 )
    {
        m_changes = 0;

        if( m_state != MagAOX::app::stateCodes::READY )
        {
            std::cerr << name() << ": toggling off because not READY: " << m_state << "\n";
            togglePutsOff();
        }
        else if( m_curVal == "none" )
        {
            std::cerr << name() << ": toggling off because 'none'\n";
            togglePutsOff();
        }
        else
        {
            std::cerr << name() << ": toggling on because READY and " << m_curVal << "\n";
            togglePutsOn();
        }
    }
}

inline void stdMotionNode::togglePutsOn()
{
    if( m_state == MagAOX::app::stateCodes::READY )
    {
        if( m_presetPutName.size() == 1 ) // There's only one put, it's just on or off with a value
        {
            if( m_node->auxDataValid() )
            {
                if( m_parentGraph )
                {
                    m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, m_curVal );
                }
            }
            xigNode::togglePutsOn();
        }
        else // There is more than one put, and which one is on is selected by the value of the switch
        {
            for( auto s : m_presetPutName )
            {
                ingr::instIOPut *pptr; // We get this pointer using the node accessors
                                       // which throw if there's a nullptr
                try
                {
                    if( m_presetDir == ingr::ioDir::input )
                    {
                        pptr = m_node->input( s );
                    }
                    else
                    {
                        pptr = m_node->output( s );
                    }
                }
                catch( ... )
                {
                    return;
                }

                if( s == m_curVal )
                {
                    pptr->state( ingr::putState::on );
                }
                else
                {
                    pptr->state( ingr::putState::off );
                }
            }
            std::cerr << "changing state\n";
            m_parentGraph->stateChange();
        }
    }

    return; // we don't automatically toggle puts on upon power on.
}

inline void stdMotionNode::togglePutsOff()
{
    std::cerr << name() << ": toggle off\n";
    if( m_node != nullptr )
    {
        if( m_node->auxDataValid() )
        {
            if( m_presetPutName.size() == 1 )
            {
                m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, "off" );
            }
            else
            {
            }
        }
    }

    xigNode::togglePutsOff();
}

inline void stdMotionNode::loadConfig( mx::app::appConfigurator &config )
{
    if( !nodeValid() )
    {
        std::string msg = "stdMotionNode::loadConfig: node is not valid";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error(msg);
    }

    std::string dev = name();
    config.configUnused( dev, mx::app::iniFile::makeKey( name(), "device" ) );

    std::string prePrefix = "preset";
    config.configUnused( prePrefix, mx::app::iniFile::makeKey( name(), "presetPrefix" ) );

    std::string preDir = "output";
    config.configUnused( preDir, mx::app::iniFile::makeKey( name(), "presetDir" ) );

    std::vector<std::string> prePutName( { "out" } );
    config.configUnused( prePutName, mx::app::iniFile::makeKey( name(), "presetPutName" ) );
    if( prePutName.size() == 0 )
    {
        std::string msg = "stdMotionNode::loadConfig: presetPutName can't be empty";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error(msg);
    }

    /*std::cerr << "  device: " << dev << "\n";
    std::cerr << "  presetPrefix: " << prePrefix << "\n";
    std::cerr << "  presetDir: " << preDir << "\n";
    std::cerr << "  presetPutName: " << prePutName[0] << "\n";
    for( size_t n = 1; n < prePutName.size(); ++n )
    {
        std::cerr << "                 " << prePutName[1] << "\n";
    }*/

    device( dev );
    presetPrefix( prePrefix );
    if( preDir == "input" )
    {
        presetDir( ingr::ioDir::input );
    }
    else if( preDir == "output" )
    {
        presetDir( ingr::ioDir::output );
    }
    else
    {
        std::string msg = "stdMotionNode::loadConfig: invalid presetDir (must be input or output)";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error(msg);
    }

    presetPutName( prePutName );

}

#endif // stdMotionNode_hpp
