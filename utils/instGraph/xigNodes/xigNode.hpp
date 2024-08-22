/** \file xigNode.hpp
 * \brief The base MagAO-X instGraph node header file
 *
 * \ingroup instGraph_files
 */

#ifndef xigNode_hpp
#define xigNode_hpp

#include <instGraph/instGraphXML.hpp>

#include "../../INDI/libcommon/IndiProperty.hpp"

class xigNode
{
  protected:
    std::set<std::string> m_keys;

    ingr::instGraphXML *m_parentGraph{ nullptr };

    int m_changes{ 0 }; ///< Counter that can be incremented when changes are detected.  Set to 0 when graph is updated.

    ingr::instNode *m_node{ nullptr };

  public:
    xigNode() = delete;

    xigNode( const std::string &name, ingr::instGraphXML *parentGraph );

    std::string name();

    const std::set<std::string> &keys();

    void key( const std::string &nkey );

    ingr::instNode *node();

    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv ) = 0;

    virtual void togglePutsOn();

    virtual void togglePutsOff();
};

inline xigNode::xigNode( const std::string &name, ingr::instGraphXML *parentGraph ) : m_parentGraph( parentGraph )
{
    //This will throw if name is not in the parent's nodes
    m_node = m_parentGraph->node( name );
}

inline std::string xigNode::name()
{
    return m_node->name();
}

inline const std::set<std::string> &xigNode::keys()
{
    return m_keys;
}

inline void xigNode::key( const std::string &nkey )
{
    m_keys.insert( nkey );
}

inline ingr::instNode *xigNode::node()
{
    return m_node; // b/c of constructor, this can't be null
}

inline void xigNode::togglePutsOn()
{
    for( auto &&iput : m_node->inputs() )
    {
        iput.second->state( ingr::putState::on );
    }

    for( auto &&oput : m_node->outputs() )
    {
        oput.second->state( ingr::putState::on );
    }
}

inline void xigNode::togglePutsOff()
{
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
