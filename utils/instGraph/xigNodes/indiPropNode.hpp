/** \file indiPropNode.hpp
 * \brief The MagAO-X Instrument Graph indiPropNode header file
 *
 * \ingroup instGraph_files
 */

#ifndef indiPropNode_hpp
#define indiPropNode_hpp

#include "fsmNode.hpp"

/// An instGraph node which tracks a specific INDI property and element of that property.
/** When the element matches the target value all puts are turned on.  All puts are off
 *  otherwise.
 */
class indiPropNode : public fsmNode
{

  protected:
    std::string m_propKey; ///< unique key, device.name, of the property tp track
    std::string m_propEl;  ///< the element of the property to track

    std::string m_propValStr; ///< the target value of the element. This is always set.

    double m_propValNum{ std::numeric_limits<double>::lowest() }; /**< the numeric target value, set from
                                                                       m_propValStr if the property is a number.*/

    /** The switch targwr value, set from m_propValStr if the property is a switch.  In this case
     *  m_propValStr can have values `On` or `Off`.  The comparison is made insensitive to case (ON and off
     *  are valid).
     */
    pcf::IndiElement::SwitchStateType m_propValSw{ pcf::IndiElement::SwitchStateType::UnknownSwitchState };

    /// The property type.  Discovered introspectively on first call to \ref handleSetProperty.
    pcf::IndiProperty::Type m_type{ pcf::IndiProperty::Unknown };

    double m_tol{ 1e-7 }; ///< The tolerance for floating point comparison.  Default is 1e-7.

    bool m_state{ false }; ///< The current state of the comparison.

    bool m_first{ true }; ///< Flag indicating if it's the first call to \ref handleSetProperty

  public:
    /// Only c'tor.  Must be constructed with node name and a parent graph.
    indiPropNode( const std::string  &name,       /** [in] the name of this node*/
                  ingr::instGraphXML *parentGraph /** [in] the graph which this node belongs to*/
    );

    /// Set the unique key of the INDI property to track
    void propKey( const std::string &pk /** [in] */ );

    /// Get the unique key of the INDI property to track
    /**
     * \returns the value of m_propKey
     */
    const std::string &propKey() const;

    /// Set the element of the INDI property to track
    void propEl( const std::string &pe /** [in] */ );

    /// Get the element of the INDI property to track
    /**
     * \returns the value of m_propEl
     */
    const std::string &propEl() const;

    /// Set the target value of the INDI element.
    /** Always set in its string form and converted as needed
     */
    void propValStr( const std::string &pv /** [in] */ );

    /// Get the target value of the INDI element.
    /**
     * \returns the value of m_propValStr
     */
    const std::string &propValStr() const;

    /// Get the target value of the INDI element if it's a number.
    /**
     * \returns the value of m_propValNum
     */
    const double &propValNum() const;

    /// Get the target value of the INDI element if it's a switch.
    /**
     * \returns the value of m_propValSw
     */
    const pcf::IndiElement::SwitchStateType &propValSw();

    /// Get the type of the INDI property being tracked
    /**
     * \returns the value of m_type
     */
    const pcf::IndiProperty::Type &type() const;

    /// Get the tolerance used for numeric comparison
    /**
     * \returns the value of m_tol
     */
    const double &tol() const;

    /// Get the current value of the comparison
    /**
     * \returns the value of m_state
     */
    const bool &state() const;

  protected:
    /// On first call to handleSetProperty we find the property type and convert the target value
    virtual void firstSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the received INDI property*/ );

  public:
    /// INDI SetProperty callback
    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the received INDI property to handle*/ );

    /// Toggle all puts on
    virtual void toggleOn();

    /// Toggle all puts off
    virtual void toggleOff();

    /// Configure this node form an appConfigurator.
    void loadConfig( mx::app::appConfigurator &config /**< [in] the loaded configuration */);
};

indiPropNode::indiPropNode( const std::string &name, ingr::instGraphXML *parentGraph ) : fsmNode( name, parentGraph )
{
    if(m_parentGraph)
    {
        m_parentGraph->valueExtra(m_node->name(), "fsmstate", "");
    }
}

inline void indiPropNode::propKey( const std::string &pk )
{
    m_propKey = pk;

    key( m_propKey );
}

const std::string &indiPropNode::propKey() const
{
    return m_propKey;
}

inline void indiPropNode::propEl( const std::string &pe )
{
    m_propEl = pe;
}

const std::string &indiPropNode::propEl() const
{
    return m_propEl;
}

inline void indiPropNode::propValStr( const std::string &pv )
{
    m_propValStr = pv;
}

const std::string &indiPropNode::propValStr() const
{
    return m_propValStr;
}

const double &indiPropNode::propValNum() const
{
    return m_propValNum;
}

const pcf::IndiElement::SwitchStateType &indiPropNode::propValSw()
{
    return m_propValSw;
}

const pcf::IndiProperty::Type &indiPropNode::type() const
{
    return m_type;
}

const double &indiPropNode::tol() const
{
    return m_tol;
}

const bool &indiPropNode::state() const
{
    return m_state;
}

inline void indiPropNode::firstSetProperty( const pcf::IndiProperty &ipRecv )
{
    // On first call we figure what type it is and convert the value

    if( ipRecv.getType() == pcf::IndiProperty::Type::Number )
    {
        m_type = pcf::IndiProperty::Type::Number;
        try
        {
            m_propValNum = std::stod( m_propValStr );
        }
        catch( const std::exception &e )
        {
            std::string msg = XIGN_EXCEPTION( "indiPropNode::firstSetProperty", "exception caught" );
            msg += ": ";
            msg += e.what();

            throw std::runtime_error( msg );
        }
    }
    else if( ipRecv.getType() == pcf::IndiProperty::Type::Switch )
    {
        m_type = pcf::IndiProperty::Type::Switch;
        try
        {
            std::string ustr = m_propValStr;
            std::transform( m_propValStr.begin(), m_propValStr.end(), ustr.begin(), ::toupper );

            if( ustr == "ON" )
            {
                m_propValSw = pcf::IndiElement::SwitchStateType::On;
            }
            else if( ustr == "OFF" )
            {
                m_propValSw = pcf::IndiElement::SwitchStateType::Off;
            }
            else
            {
                std::string msg = XIGN_EXCEPTION( "indiPropNode::firstSetProperty", "invalid switch state" );
                throw std::invalid_argument( msg );
            }
        }
        catch( const std::exception &e )
        {
            std::string msg = XIGN_EXCEPTION( "indiPropNode::firstSetProperty", "exception caught" );
            msg += ": ";
            msg += e.what();

            throw std::runtime_error( msg );
        }
    }
    else if( ipRecv.getType() == pcf::IndiProperty::Type::Text )
    {
        m_type = pcf::IndiProperty::Type::Text;
    }
    else
    {
        std::string msg = XIGN_EXCEPTION( "indiPropNode::firstSetProperty", "INDI property of type not implemented" );
        throw std::runtime_error( msg );
    }
}

inline void indiPropNode::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    bool actionTaken = false;
    fsmNode::handleSetProperty(actionTaken, ipRecv);

    if(actionTaken)
    {
        return;
    }

    if( ipRecv.createUniqueKey() != m_propKey )
    {
        return;
    }

    if( !ipRecv.find( m_propEl ) )
    {
        return;
    }

    if( m_first )
    {
        try
        {
            firstSetProperty( ipRecv );
        }
        catch( const std::exception &e )
        {
            std::string msg = XIGN_EXCEPTION( "indiPropNode::handleSetProperty", "exception caught" );
            msg += ": ";
            msg += e.what();

            throw std::runtime_error( msg );
        }
    }

    bool on = false;

    if( m_type == pcf::IndiProperty::Type::Number )
    {
        if( fabs( ipRecv[m_propEl].get<double>() - m_propValNum ) <= m_tol )
        {
            on = true;
        }
    }
    else if( m_type == pcf::IndiProperty::Type::Switch )
    {
        if( ipRecv[m_propEl].getSwitchState() == m_propValSw )
        {
            on = true;
        }
    }
    else if( m_type == pcf::IndiProperty::Type::Text )
    {
        if( ipRecv[m_propEl].get() == m_propValStr )
        {
            on = true;
        }
    }
    else
    {
        std::string msg = XIGN_EXCEPTION( "indiPropNode::handleSetProperty", "type not implemented" );
        throw std::runtime_error( msg );
    }

    if( m_first )
    {
        // trigger the change on the first run.
        if( on )
        {
            m_state = false;
        }
        else
        {
            m_state = true;
        }

        // made this it's own branch so we don't do this every time:
        m_first = false;
    }

    if( on != m_state )
    {
        ++m_changes;
        m_state = on;
        if( on )
        {
            return toggleOn();
        }
        else
        {
            return toggleOff();
        }
    }
}

inline void indiPropNode::toggleOn()
{
    togglePutsOn();
}

inline void indiPropNode::toggleOff()
{
    togglePutsOff();
}

inline void indiPropNode::loadConfig( mx::app::appConfigurator &config )
{
    if( !m_parentGraph )
    {
        std::string msg = XIGN_EXCEPTION( "indiPropNode::loadConfig", "parent graph is null" );
        throw std::runtime_error( msg );
    }

    std::string type;
    config.configUnused( type, mx::app::iniFile::makeKey( name(), "type" ) );

    if( type != "indiProp" )
    {
        std::string msg = XIGN_EXCEPTION( "indiPropNode::loadConfig", "node type is not indiProp" );
        throw std::runtime_error( msg );
    }

    fsmNode::loadConfigDerived(config);

    std::string pk;
    config.configUnused( pk, mx::app::iniFile::makeKey( name(), "propKey" ) );

    if( pk == "" )
    {
        std::string msg = XIGN_EXCEPTION( "indiPropNode::loadConfig", "propKey can not be empty" );
        throw std::runtime_error( msg );
    }

    std::string pe;
    config.configUnused( pe, mx::app::iniFile::makeKey( name(), "propEl" ) );

    if( pe == "" )
    {
        std::string msg = XIGN_EXCEPTION( "indiPropNode::loadConfig", "propEl can not be empty" );
        throw std::runtime_error( msg );
    }

    std::string pv;
    config.configUnused( pv, mx::app::iniFile::makeKey( name(), "propVal" ) );

    if( pv == "" )
    {
        std::string msg = XIGN_EXCEPTION( "indiPropNode::loadConfig", "propVal can not be empty" );
        throw std::runtime_error( msg );
    }

    config.configUnused( m_tol, mx::app::iniFile::makeKey( name(), "tol" ) );

    // Add propEl and propVal
    propKey( pk );
    m_propEl     = pe;
    m_propValStr = pv;
}

#endif // indiPropNode_hpp
