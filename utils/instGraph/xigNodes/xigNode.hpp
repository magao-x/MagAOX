/** \file xigNode.hpp
 * \brief The base MagAO-X instGraph node header file
 *
 * \ingroup instGraph_files
 */

#ifndef xigNode_hpp
#define xigNode_hpp

#include <instGraph/instGraphXML.hpp>

class xigNode
{
  protected:
    std::set<std::string> m_keys;

    ingr::instGraphXML *m_parentGraph{ nullptr };

    int m_changes{ 0 }; ///< Counter that can be incremented when changes are detected.  Set to 0 when graph is updated.

    ingr::instNode *m_node{ nullptr };

  public:
    xigNode( const std::string &name, ingr::instGraphXML *parentGraph );

    std::string name();

    const std::set<std::string> &keys();

    void key( const std::string &nkey );

    bool nodeValid();

    ingr::instNode *node();

    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv ) = 0;

    virtual void togglePutsOn();

    virtual void togglePutsOff();
};

inline
xigNode::xigNode( const std::string &name, ingr::instGraphXML *parentGraph ) : m_parentGraph( parentGraph )
{
    m_node = m_parentGraph->node( name );
}

inline
std::string xigNode::name()
{
    if( m_node == nullptr )
    {
        std::string msg = "xigNode::name(): m_node is nullptr";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error(msg);
    }

    return m_node->name();
}

inline
const std::set<std::string> &xigNode::keys()
{
    return m_keys;
}

inline
void xigNode::key( const std::string &nkey )
{
    m_keys.insert( nkey );
}

inline
bool xigNode::nodeValid()
{
    if( m_node == nullptr )
    {
        return false;
    }

    return true;
}

inline
ingr::instNode *xigNode::node()
{
    if( m_node == nullptr )
    {
        std::string msg = "xigNode::node(): m_node is nullptr";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );

        throw std::runtime_error(msg);
    }

    return m_node;
}

inline
void xigNode::togglePutsOn()
{
    if( m_node == nullptr )
    {
        std::string msg = "xigNode::togglePutsOn(): m_node is nullptr";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );

        throw std::runtime_error(msg);
    }

    for( auto &&iput : m_node->inputs() )
    {
        iput.second->state( ingr::putState::on );
    }

    for( auto &&oput : m_node->outputs() )
    {
        oput.second->state( ingr::putState::on );
    }
}

inline
void xigNode::togglePutsOff()
{
    if( m_node == nullptr )
    {
        std::string msg = "xigNode::togglePutsOff(): m_node is nullptr";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );

        throw std::runtime_error(msg);
    }

    for( auto &&iput : m_node->inputs() )
    {
        iput.second->state( ingr::putState::off );
    }

    for( auto &&oput : m_node->outputs() )
    {
        oput.second->state( ingr::putState::off );
    }
}

#endif // xigNode_hpp
