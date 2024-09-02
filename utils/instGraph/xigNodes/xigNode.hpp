/** \file xigNode.hpp
 * \brief The base MagAO-X instGraph node header file
 *
 * \ingroup instGraph_files
 */

#ifndef xigNode_hpp
#define xigNode_hpp

#include <instGraph/instGraphXML.hpp>

#include "../../INDI/libcommon/IndiProperty.hpp"

/// Implementation of basic instGraph node interface for MagAO-X
/** This class is pure virtual, derived classes must implement handleSetProperty.
  *
  */
class xigNode
{
  protected:
    std::set<std::string> m_keys; ///< The INDI keys (device.property) which this node subscribes to

    ingr::instGraphXML *m_parentGraph{ nullptr }; ///< The parent instGraph that this node is a part of

    ingr::instNode *m_node{ nullptr }; ///< The underlying instGraph node

    int m_changes{ 0 }; ///< Counter that can be incremented when changes are detected.  Set to 0 when graph is updated.

  public:
    xigNode() = delete;

    /// Constructor.
    /**
      * Default c'tor is deleted.  Must supply both node name and a parentGraph with a node with the same name in it.
      */
    xigNode( const std::string &name, /**< [in] the name of the node */
             ingr::instGraphXML *parentGraph /**< [in] the parent instGraph */ );

    /// Get the name of this node
    /**
      * \returns the nodes' name (the value of m_name).
      */
    std::string name();

    const std::set<std::string> &keys();

    void key( const std::string &nkey );

    ingr::instNode *node();

    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv ) = 0;

    virtual void togglePutsOn();

    virtual void togglePutsOff();

    #ifdef XWC_XIGNODE_TEST
    //allow setting m_parentGraph to null for testing
    void setParentGraphNull()
    {
        m_parentGraph = nullptr;
    }
    #endif
};

inline xigNode::xigNode( const std::string &name, ingr::instGraphXML *parentGraph ) : m_parentGraph( parentGraph )
{
    if(m_parentGraph == nullptr)
    {
        std::string msg = "xigNode::loadConfig: parent graph is null";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error(msg);
    }

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
