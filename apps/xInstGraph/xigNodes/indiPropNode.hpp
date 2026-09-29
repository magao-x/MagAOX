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
    std::string m_propKey; ///< unique key, device.name, of the property to track
    std::string m_propEl;  ///< the element of the property to track

    std::string m_propValStr; ///< the target value of the element. This is always set.

    double m_propValNum{ std::numeric_limits<double>::lowest() }; /**< the numeric target value, set from
                                                                       m_propValStr if the property is a number.*/

    /** The switch target value, set from m_propValStr if the property is a switch.  In this case
     *  m_propValStr can have values `On` or `Off`.  The comparison is made insensitive to case (ON and off
     *  are valid).
     */
    pcf::IndiElement::SwitchStateType m_propValSw{ pcf::IndiElement::SwitchStateType::UnknownSwitchState };

    /// The property type.  Discovered introspectively on first call to \ref handleSetProperty.
    pcf::IndiProperty::Type m_type{ pcf::IndiProperty::Unknown };

    double m_tol{ 1e-7 }; ///< The tolerance for floating point comparison.  Default is 1e-7.

    bool m_state{ false }; ///< The current state of the tracked-property comparison.

    /// Current put state after applying the FSM gate to the property comparison.
    bool m_effectiveState{ false };

    /// Whether the effective put state has been published at least once.
    bool m_effectiveInitialized{ false };

    bool m_first{ true }; ///< Whether the first tracked-property update is still pending.

    /// Label shown when the effective put state is on.
    std::string m_onStr{ "ON" };

    /// Label shown when the effective put state is off.
    std::string m_offStr{ "OFF" };

  public:
    /// Only c'tor.  Must be constructed with node name and a parent graph.
    indiPropNode( const std::string  &name,       /**< [in] the name of this node */
                  ingr::instGraphXML *parentGraph /**< [in] the graph which this node belongs to */
    );

    /// Set the unique key of the INDI property to track
    void propKey( const std::string &pk /**< [in] device.property key */ );

    /// Get the unique key of the INDI property to track
    /**
     * \returns the value of m_propKey
     */
    const std::string &propKey() const;

    /// Set the element of the INDI property to track
    void propEl( const std::string &pe /**< [in] element name */ );

    /// Get the element of the INDI property to track
    /**
     * \returns the value of m_propEl
     */
    const std::string &propEl() const;

    /// Set the target value of the INDI element.
    /** Always set in its string form and converted as needed
     */
    void propValStr( const std::string &pv /**< [in] target element value */ );

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

    /// Get the current tracked-property comparison, before applying the FSM gate.
    /**
     * \returns the value of m_state
     */
    const bool &state() const;

  protected:
    /// On the first tracked-property update, determine its type and convert the target value.
    virtual int firstSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the received INDI property */ );

    /// Apply the conjunction of the tracked property and any configured FSM gate.
    void updateEffectiveState( bool force /**< [in] reapply after the base FSM handler changed puts */ );

  public:
    /// INDI SetProperty callback
    virtual int handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the received INDI property to handle*/ );

    /// Toggle all puts on
    virtual void toggleOn();

    /// Toggle all puts off
    virtual void toggleOff();

    /// Configure this node from an appConfigurator.
    void loadConfig( mx::app::appConfigurator &config /**< [in] the loaded configuration */ );
};

indiPropNode::indiPropNode( const std::string &name, ingr::instGraphXML *parentGraph ) : fsmNode( name, parentGraph )
{
    if( m_parentGraph )
    {
        m_parentGraph->valueExtra( m_node->name(), "fsmstate", "" );
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

inline int indiPropNode::firstSetProperty( const pcf::IndiProperty &ipRecv )
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

    return 0;
}

inline void indiPropNode::updateEffectiveState( bool force )
{
    const bool gateOpen  = m_fsmAction == fsmNodeActionT::passive || m_stateOnTarget;
    const bool effective = m_state && gateOpen;
    if( m_effectiveInitialized && effective == m_effectiveState && !force )
    {
        return;
    }

    if( !m_effectiveInitialized || effective != m_effectiveState )
    {
        ++m_changes;
    }
    m_effectiveState       = effective;
    m_effectiveInitialized = true;

    if( effective )
    {
        toggleOn();
        m_parentGraph->valueExtra( m_node->name(), "state", m_onStr );
    }
    else
    {
        toggleOff();
        m_parentGraph->valueExtra( m_node->name(), "state", m_offStr );
    }
}

inline int indiPropNode::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    const std::string key         = ipRecv.createUniqueKey();
    const bool        fsmUpdate   = key == m_fsmKey && ipRecv.find( m_fsmElName );
    bool              actionTaken = false;
    int               rv          = fsmNode::handleSetProperty( actionTaken, ipRecv );
    if( rv < 0 )
    {
        return rv;
    }

    if( key != m_propKey )
    {
        if( fsmUpdate )
        {
            updateEffectiveState( true );
        }
        return 0;
    }

    if( !ipRecv.find( m_propEl ) )
    {
        std::cerr << "!ipRecv.find( m_propEl )\n";
        return -1;
    }

    if( m_first )
    {
        try
        {
            rv = firstSetProperty( ipRecv );
            if( rv < 0 )
            {
                return rv;
            }
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
        on = fabs( ipRecv[m_propEl].get<double>() - m_propValNum ) <= m_tol;
    }
    else if( m_type == pcf::IndiProperty::Type::Switch )
    {
        on = ipRecv[m_propEl].getSwitchState() == m_propValSw;
    }
    else if( m_type == pcf::IndiProperty::Type::Text )
    {
        on = ipRecv[m_propEl].get() == m_propValStr;
    }
    else
    {
        throw std::runtime_error( XIGN_EXCEPTION( "indiPropNode::handleSetProperty", "type not implemented" ) );
    }

    m_state = on;
    m_first = false;
    updateEffectiveState( fsmUpdate );
    return 0;
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

    fsmNode::loadConfigDerived( config );

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

    config.configUnused( m_onStr, mx::app::iniFile::makeKey( name(), "onStr" ) );
    config.configUnused( m_offStr, mx::app::iniFile::makeKey( name(), "offStr" ) );
}

#endif // indiPropNode_hpp
