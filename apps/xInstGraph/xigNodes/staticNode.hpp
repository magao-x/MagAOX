/** \file staticNode.hpp
 * \brief The MagAO-X Instrument Graph staticNode header file
 *
 * \ingroup instGraph_files
 */

#ifndef staticNode_hpp
#define staticNode_hpp

#include "xigNode.hpp"

/// An instGraph node which is static, with status set at config time and not changing.
class staticNode : public xigNode
{

  protected:
    std::set<std::string> m_inputsOn;  ///< inputs which are always on
    std::set<std::string> m_inputsOff; ///< inputs which are always off

    std::set<std::string> m_outputsOn;  ///< outputs which are always on
    std::set<std::string> m_outputsOff; ///< outputs which are always off

  public:
    /// Only c'tor.  Must be constructed with node name and a parent graph.
    staticNode( const std::string  &name,       /** [in] the name of this node*/
                ingr::instGraphXML *parentGraph /** [in] the graph which this node belongs to*/
    );

    /// Get the always on inputs
    /**
     * \returns the value of m_inputsOn
     */
    const std::set<std::string> &inputsOn() const;

    /// Get the always off inputs
    /**
     * \returns the value of m_inputsOff
     */
    const std::set<std::string> &inputsOff() const;

    /// Get the always on outputs
    /**
     * \returns the value of m_outputsOn
     */
    const std::set<std::string> &outputsOn() const;

    /// Get the always off outputs
    /**
     * \returns the value of m_outputsOff
     */
    const std::set<std::string> &outputsOff() const;

  public:
    /// INDI SetProperty callback
    virtual int handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the received INDI property to handle*/ );

    /// Toggle all puts to their static state
    virtual void togglePutsAll();

    /// Toggle all puts on
    virtual void togglePutsOn();

    /// Toggle all puts off
    virtual void togglePutsOff();

    /// Configure this node form an appConfigurator.
    void loadConfig( mx::app::appConfigurator &config /**< [in] the loaded configuration */ );
};

staticNode::staticNode( const std::string &name, ingr::instGraphXML *parentGraph ) : xigNode( name, parentGraph )
{
}

const std::set<std::string> &staticNode::inputsOn() const
{
    return m_inputsOn;
}

const std::set<std::string> &staticNode::inputsOff() const
{
    return m_inputsOff;
}

const std::set<std::string> &staticNode::outputsOn() const
{
    return m_outputsOn;
}

const std::set<std::string> &staticNode::outputsOff() const
{
    return m_outputsOff;
}

inline int staticNode::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    static_cast<void>( ipRecv );

    return 0;
}

inline void staticNode::togglePutsAll()
{
    try
    {
        for( auto &iput : m_inputsOn )
        {
            m_node->input( iput )->state( ingr::putState::on );
        }
    }
    catch( const std::exception &e )
    {
        std::string msg = XIGN_EXCEPTION( "staticNode::togglePutsAll", "exception changing state on inputs" );
        msg += "\n    ";
        msg += e.what();
        throw std::runtime_error( msg );
    }

    try
    {
        for( auto &iput : m_inputsOff )
        {
            m_node->input( iput )->state( ingr::putState::off );
        }
    }
    catch( const std::exception &e )
    {
        std::string msg = XIGN_EXCEPTION( "staticNode::togglePutsAll", "parent graph is null" );
        msg += "\n    ";
        msg += e.what();
        throw std::runtime_error( msg );
    }

    try
    {
        for( auto &iput : m_outputsOn )
        {
            m_node->output( iput )->state( ingr::putState::on );
        }
    }
    catch( const std::exception &e )
    {
        std::string msg = XIGN_EXCEPTION( "staticNode::togglePutsAll", "parent graph is null" );
        msg += "\n    ";
        msg += e.what();
        throw std::runtime_error( msg );
    }

    try
    {
        for( auto &iput : m_outputsOff )
        {
            m_node->output( iput )->state( ingr::putState::off );
        }
    }
    catch( const std::exception &e )
    {
        std::string msg = XIGN_EXCEPTION( "staticNode::togglePutsAll", "parent graph is null" );
        msg += "\n    ";
        msg += e.what();
        throw std::runtime_error( msg );
    }
}

inline void staticNode::togglePutsOn()
{
    togglePutsAll();
}

inline void staticNode::togglePutsOff()
{
    togglePutsAll();
}

inline void staticNode::loadConfig( mx::app::appConfigurator &config )
{
    if( !m_parentGraph )
    {
        std::string msg = XIGN_EXCEPTION( "staticNode::loadConfig", "parent graph is null" );
        throw std::runtime_error( msg );
    }

    std::string type;
    config.configUnused( type, mx::app::iniFile::makeKey( name(), "type" ) );

    if( type != "static" )
    {
        std::string msg = XIGN_EXCEPTION( "staticNode::loadConfig", "node type is not static" );
        throw std::runtime_error( msg );
    }

    std::vector<std::string> inputsOn;
    config.configUnused( inputsOn, mx::app::iniFile::makeKey( name(), "inputsOn" ) );

    try
    {
        m_inputsOn.clear();
        for( auto &in : inputsOn )
        {
            m_inputsOn.insert( in );
        }
    }
    catch(const std::exception & e)
    {
        std::string msg = XIGN_EXCEPTION( "staticNode::loadConfig", "exception caught loading inputsOn" );
        msg += "\n    ";
        msg += e.what();
    }

    std::vector<std::string> inputsOff;
    config.configUnused( inputsOff, mx::app::iniFile::makeKey( name(), "inputsOff" ) );

    try
    {
        m_inputsOff.clear();
        for( auto &in : inputsOff )
        {
            m_inputsOff.insert( in );
        }
    }
    catch(const std::exception & e)
    {
        std::string msg = XIGN_EXCEPTION( "staticNode::loadConfig", "exception caught loading inputsOff" );
        msg += "\n    ";
        msg += e.what();
    }

    std::vector<std::string> outputsOn;
    config.configUnused( outputsOn, mx::app::iniFile::makeKey( name(), "outputsOn" ) );
    try
    {
        m_outputsOn.clear();
        for( auto &out : outputsOn )
        {
            m_outputsOn.insert( out );
        }
    }
    catch(const std::exception & e)
    {
        std::string msg = XIGN_EXCEPTION( "staticNode::loadConfig", "exception caught loading outputsOn" );
        msg += "\n    ";
        msg += e.what();
    }

    std::vector<std::string> outputsOff;
    config.configUnused( outputsOff, mx::app::iniFile::makeKey( name(), "outputsOff" ) );
    try
    {
        m_outputsOff.clear();
        for( auto &out : outputsOff )
        {
            m_outputsOff.insert( out );
        }
    }
    catch(const std::exception & e)
    {
        std::string msg = XIGN_EXCEPTION( "staticNode::loadConfig", "exception caught loading outputsOff" );
        msg += "\n    ";
        msg += e.what();
    }

    togglePutsAll();
}

#endif // staticNode_hpp
