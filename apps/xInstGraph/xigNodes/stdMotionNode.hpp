/** \file stdMotionNode.hpp
 * \brief Motion-stage preset and tracking routing for the MagAO-X instrument graph.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup instGraph_files
 */

#ifndef stdMotionNode_hpp
#define stdMotionNode_hpp

#include <map>
#include <sstream>

#include "fsmNode.hpp"

/// Motion stage that maps preset and tracking state to graph puts.
/**
 * The key assumption of this node is that it should be in a valid, not-`none`, preset position
 * for its ioputs to be `on`.  It also supports triggering an alternate `on` state, which is used for
 * stages which have a continuous tracking mode (k-mirror and ADC).
 *
 * The preset is specified by an INDI property with signature `<device>.<presetPrefix>Name` where device
 * and presetPrefix are part of the configuration.  This INDI property is a switch vector.
 *
 * With `parkable=true`, a Number property `<device>.parked` with nonzero `current` makes a preset usable in `POWEROFF`.
 * Parking support defaults false; only enabled stages subscribe to this property.
 * The true FSM is preserved. This retained-position path takes priority over tracking flags and requires one
 * selected name, matching a route row in mapping mode or a configured put for legacy multi-put selection.
 *
 * Optional `presetRoute.<name>` rows map a published preset to a set of puts on `presetDir`.
 * Mapped nodes own all puts, require one common opposite-side put and internal links for every branch,
 * and disable excluded paths so upstream propagation cannot activate them. Empty rows block all paths.
 * Mapping cannot be combined with legacy put-selection or tracking options.
 *
 * The device and prefix can only be set once.
 */
class stdMotionNode : public fsmNode
{

  protected:
    /// The prefix for preset names.  Usually either "preset" or "filter", to which "Name" is appended.
    std::string m_presetPrefix;

    /// The INDI key (device.property) for the presets.  This is, say, `fwpupil.filterName`.  It is set automatically.
    std::string m_presetKey;

    /// Latest selected preset name, cached even while tracking; parked routing also checks selection validity.
    std::string m_curVal;

    /// Whether the latest preset property is a Switch vector with exactly one selected name.
    bool m_presetSelectionValid{ false };

    /// Configuration opt-in for subscribing to and using the stage's parked state.
    bool m_parkable{ false };

    /// The INDI key for the optional device-local Number property parked.current.
    std::string m_parkedKey;

    /// Affirmative parking reported by the stage; false until a valid value is received.
    bool m_parked{ false };

    /// Latest position or tracking label applied to the graph.
    std::string m_curLabel;

    /// Legacy selected put names, or every selected-side graph put in mapping mode.
    std::vector<std::string> m_presetPutName{ "out" };

    /// Explicit preset-to-put sets; a nonempty table enables mapping, and an empty row blocks all paths.
    std::map<std::string, std::set<std::string>> m_presetRoutes;

    /// Side selected by the preset switch; output is the default.
    /** Explicit route rows can permit multiple puts together. With legacy multi-put selection
     * (m_presetPutName.size() > 1), the preset name chooses one put, alongside any alwaysOn puts.
     */
    ingr::ioDir m_presetDir{ ingr::ioDir::output };

    /// Puts kept on alongside a usable preset route and cleared when the route is inactive.
    std::set<std::string> m_alwaysOn;

    /// Contains the names of any puts which are not automatically turned on if they are off.
    std::set<std::string> m_noAutoOn;

    /// The INDI key (device.property) for the switch denoting that this stage should be or should not be tracking
    std::string m_trackingReqKey;

    /// The element of the INDI property denoted by m_trackingReqKey to follow.
    std::string m_trackingReqElement;

    /// The INDI key (device.property) for the switch denoting that this stage is tracking
    std::string m_trackerKey;

    /// The element of the INDI property denoted by m_trackerKey to follow.
    std::string m_trackerElement;

    /// Flag indicating if the stage should be (true) or should not be (false, default) tracking.
    bool m_trackingReq{ false };

    /// Flag indicating whether or not the stage is currently tracking (default false).
    bool m_tracking{ false };

  public:
    /// Construct a motion stage for an existing graph node.
    stdMotionNode( const std::string  &name, /**< [in] graph node name */
                   ingr::instGraphXML *parentGraph /**< [in] parent graph */ );

    /// Set the device name.  This can only be done once.
    /** \throws std::runtime_error if a different or empty device is supplied. */
    virtual void device( const std::string &dev /**< [in] INDI device name */ );

    using fsmNode::device;

    /// Set the preset property prefix.
    virtual void presetPrefix( const std::string &pp /**< [in] property prefix */ );

    /// Get the preset property prefix.
    const std::string &presetPrefix();

    /// Get the current label text
    /**
     * \returns the current value of m_curLabel.
     */
    const std::string &curLabel();

    /// Set the put names controlled by presets.
    void presetPutName( const std::vector<std::string> &ppp /**< [in] put names */ );

    /// Get the put names controlled by presets.
    const std::vector<std::string> &presetPutName();

    /// Set which side of the node has the selected puts.
    void presetDir( const ingr::ioDir &dir /**< [in] put direction */ );

    /// Get which side of the node has the selected puts.
    const ingr::ioDir &presetDir();

    /// Set the tracking request property key.
    void trackingReqKey( const std::string &tk /**< [in] device.property key */ );

    /// Get the tracking request property key.
    const std::string &trackingReqKey();

    /// Set the tracking request element name.
    void trackingReqElement( const std::string &te /**< [in] element name */ );

    /// Get the tracking request element name.
    const std::string &trackingReqElement();

    /// Set the tracking status property key.
    void trackerKey( const std::string &tk /**< [in] device.property key */ );

    /// Get the tracking status property key.
    const std::string &trackerKey();

    /// Set the tracking status element name.
    void trackerElement( const std::string &te /**< [in] element name */ );

    /// Get the tracking status element name.
    const std::string &trackerElement();

    /// Cache FSM, parking, preset, or tracking updates and recompute the graph route.
    virtual int handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the received INDI property to handle*/ );

    /// Apply the selected preset or tracking state to the node puts.
    virtual void togglePutsOn();

    /// Clear all puts, disable mapped paths, and report an inactive position or tracking label.
    virtual void togglePutsOff();

    /// Load and validate this motion stage's configuration.
    void loadConfig(
        mx::app::appConfigurator &config /**< [in] the application configurator loaded with this node's options*/ );

  protected:
    /// Whether the retained stage position may be used while its motors are powered off.
    bool parkedPowerOff() const;

    /// Whether the latest named selection identifies a usable parked route.
    bool parkedPresetValid() const;

    /// Find a configured route for one valid selected preset; nullptr means the selection is unusable.
    const std::set<std::string> *selectedPresetRoute() const;

    /// Apply a mapped enablement mask and recalculate linked states; an empty set blocks every put.
    void applyPresetRoute( const std::set<std::string> &puts /**< [in] selected-side puts permitted by this route */ );

    /// Discover and validate explicit route rows, topology, and incompatible legacy options.
    void loadPresetRoutes( mx::app::appConfigurator &config /**< [in] this node's loaded configuration */ );

    /// Decide whether to apply a known preset route or active tracking from the cached properties.
    bool putsShouldBeOn() const;
};

inline stdMotionNode::stdMotionNode( const std::string &name, ingr::instGraphXML *parentGraph )
    : fsmNode( name, parentGraph )
{
}

inline void stdMotionNode::device( const std::string &dev )
{
    // This will enforce the one-time only rule
    fsmNode::device( dev );

    m_parkedKey = m_device + ".parked";
    if( m_parkable )
    {
        key( m_parkedKey );
    }

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
        std::string msg =
            "stdMotionNode::presetPrefix: attempt to change preset prefix from " + m_presetPrefix + " to " + pp;
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

inline const std::string &stdMotionNode::presetPrefix()
{
    return m_presetPrefix;
}

inline const std::string &stdMotionNode::curLabel()
{
    return m_curLabel;
}

inline void stdMotionNode::presetPutName( const std::vector<std::string> &ppp )
{
    m_presetPutName = ppp;
}

inline const std::vector<std::string> &stdMotionNode::presetPutName()
{
    return m_presetPutName;
}

inline void stdMotionNode::presetDir( const ingr::ioDir &dir )
{
    m_presetDir = dir;
}

inline const ingr::ioDir &stdMotionNode::presetDir()
{
    return m_presetDir;
}

inline void stdMotionNode::trackingReqKey( const std::string &tk )
{
    m_trackingReqKey = tk;

    if( m_trackingReqKey != "" )
    {
        key( m_trackingReqKey );
    }
}

inline const std::string &stdMotionNode::trackingReqKey()
{
    return m_trackingReqKey;
}

inline void stdMotionNode::trackingReqElement( const std::string &te )
{
    m_trackingReqElement = te;
}

inline const std::string &stdMotionNode::trackingReqElement()
{
    return m_trackingReqElement;
}

inline void stdMotionNode::trackerKey( const std::string &tk )
{
    m_trackerKey = tk;

    if( m_trackerKey != "" )
    {
        key( m_trackerKey );
    }
}

inline const std::string &stdMotionNode::trackerKey()
{
    return m_trackerKey;
}

inline void stdMotionNode::trackerElement( const std::string &te )
{
    m_trackerElement = te;
}

inline const std::string &stdMotionNode::trackerElement()
{
    return m_trackerElement;
}

inline int stdMotionNode::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    int rv = fsmNode::handleSetProperty( ipRecv );

    if( rv < 0 )
    {
        return rv;
    }

    if( ipRecv.createUniqueKey() == m_trackingReqKey )
    {
        if( ipRecv.find( m_trackingReqElement ) )
        {
            if( ipRecv[m_trackingReqElement].getSwitchState() == pcf::IndiElement::On )
            {
                if( !m_trackingReq )
                {
                    ++m_changes;
                }

                m_trackingReq = true;
            }
            else
            {
                if( m_trackingReq )
                {
                    ++m_changes;
                }
                m_trackingReq = false;
            }
        }
    }
    else if( ipRecv.createUniqueKey() == m_trackerKey )
    {
        if( ipRecv.find( m_trackerElement ) )
        {
            if( ipRecv[m_trackerElement].getSwitchState() == pcf::IndiElement::On )
            {
                if( !m_tracking )
                {
                    ++m_changes;
                }

                m_tracking = true;
            }
            else
            {
                if( m_tracking )
                {
                    ++m_changes;
                }
                m_tracking = false;
            }
        }
    }
    else if( m_parkable && ipRecv.createUniqueKey() == m_parkedKey )
    {
        bool parked = false;
        if( ipRecv.getType() == pcf::IndiProperty::Number && ipRecv.find( "current" ) )
        {
            // IndiElement::get<T>() does not check conversion success. Parse the entire numeric value.
            std::istringstream current( ipRecv["current"].get() );
            double             value = 0;
            if( current >> value )
            {
                parked = ( current >> std::ws ).eof() && value != 0;
            }
        }
        if( m_parked != parked )
        {
            ++m_changes;
            m_parked = parked;
        }
    }
    else if( ipRecv.createUniqueKey() == m_presetKey )
    {
        if( m_node != nullptr )
        {
            std::string currentValue;
            size_t      selected = 0;
            for( const auto &element : ipRecv.getElements() )
            {
                if( element.second.getSwitchState() == pcf::IndiElement::On )
                {
                    currentValue = element.second.getName();
                    ++selected;
                }
            }

            bool selectionValid = ipRecv.getType() == pcf::IndiProperty::Switch && selected == 1;
            if( ( m_curVal != currentValue || m_presetSelectionValid != selectionValid ) &&
                ( !m_tracking || parkedPowerOff() ) )
            {
                ++m_changes;
            }
            m_curVal               = currentValue;
            m_presetSelectionValid = selectionValid;
        }
    }

    if( m_changes > 0 )
    {
        m_changes = 0;
        if( putsShouldBeOn() )
        {
            togglePutsOn();
        }
        else
        {
            togglePutsOff();
        }
    }

    return 0;
}

inline bool stdMotionNode::parkedPowerOff() const
{
    return m_parkable && m_state == MagAOX::app::stateCodes::POWEROFF && m_parked;
}

inline bool stdMotionNode::parkedPresetValid() const
{
    if( !m_presetSelectionValid || m_curVal.empty() || m_curVal == "none" )
    {
        return false;
    }
    if( !m_presetRoutes.empty() )
    {
        return selectedPresetRoute() != nullptr;
    }
    if( m_presetPutName.size() == 1 )
    {
        return true;
    }
    for( const auto &put : m_presetPutName )
    {
        if( put == m_curVal )
        {
            return true;
        }
    }
    return false;
}

inline const std::set<std::string> *stdMotionNode::selectedPresetRoute() const
{
    if( !m_presetSelectionValid || m_curVal.empty() || m_curVal == "none" )
    {
        return nullptr;
    }
    const auto route = m_presetRoutes.find( m_curVal );
    return route == m_presetRoutes.end() ? nullptr : &route->second;
}

inline void stdMotionNode::applyPresetRoute( const std::set<std::string> &puts )
{
    // Install every enablement flag before changing states that can propagate through links.
    for( const auto &put : m_node->inputs() )
    {
        put.second->enabled( m_presetDir == ingr::ioDir::input ? puts.count( put.first ) != 0 : !puts.empty() );
    }
    for( const auto &put : m_node->outputs() )
    {
        put.second->enabled( m_presetDir == ingr::ioDir::output ? puts.count( put.first ) != 0 : !puts.empty() );
    }
    for( const auto &put : m_node->inputs() )
    {
        if( !put.second->enabled() )
        {
            put.second->state( ingr::putState::off );
        }
    }
    for( const auto &put : m_node->outputs() )
    {
        if( !put.second->enabled() )
        {
            put.second->state( ingr::putState::off );
        }
    }
    for( const auto &put : m_node->inputs() )
    {
        if( put.second->enabled() )
        {
            put.second->state( ingr::putState::on );
        }
    }
    for( const auto &put : m_node->outputs() )
    {
        if( put.second->enabled() )
        {
            // Configuration guarantees an internal link, so this preserves the upstream waiting state.
            put.second->state( ingr::putState::on );
        }
    }
}

inline bool stdMotionNode::putsShouldBeOn() const
{
    if( !m_presetRoutes.empty() )
    {
        return selectedPresetRoute() != nullptr && ( m_state == MagAOX::app::stateCodes::READY || parkedPowerOff() );
    }
    if( m_state == MagAOX::app::stateCodes::POWEROFF )
    {
        return parkedPowerOff() && parkedPresetValid();
    }
    if( m_trackingReq )
    {
        return m_tracking &&
               ( m_state == MagAOX::app::stateCodes::READY || m_state == MagAOX::app::stateCodes::OPERATING );
    }
    return m_state == MagAOX::app::stateCodes::READY && !m_tracking && !m_curVal.empty() && m_curVal != "none";
}

inline void stdMotionNode::togglePutsOn()
{
    if( m_node == nullptr || !m_parentGraph || !m_node->auxDataValid() )
    {
        return;
    }

    if( !putsShouldBeOn() )
    {
        togglePutsOff();
        return;
    }

    if( !m_presetRoutes.empty() )
    {
        applyPresetRoute( *selectedPresetRoute() );
        m_curLabel = m_curVal;
        m_parentGraph->valueExtra( name(), "state", m_curLabel );
        m_parentGraph->stateChange();
        return;
    }

    if( m_trackingReq && !parkedPowerOff() )
    {
        if( m_tracking )
        {
            m_curLabel = "tracking";
            m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, "tracking" );
            m_parentGraph->valueExtra( m_node->name(), "state", "tracking" );
            xigNode::togglePutsOn();
        }
        else
        {
            m_curLabel = "not tracking";
            m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, "not tracking" );
            m_parentGraph->valueExtra( m_node->name(), "state", "not tracking" );
            m_parentGraph->stateChange();
        }
    }
    else if( m_state == MagAOX::app::stateCodes::READY || parkedPowerOff() )
    {
        m_curLabel = m_curVal;

        m_parentGraph->valueExtra( m_node->name(), "state", m_curLabel );

        if( m_presetPutName.size() == 1 ) // There's only one put, it's just on or off with a value
        {
            m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, m_curVal );
            xigNode::togglePutsOn();
        }
        else // There is more than one put, and which one is on is selected by the value of the switch
        {
            if( m_presetDir == ingr::ioDir::output )
            {
                if( m_node->inputs().empty() || m_node->inputs().begin()->second == nullptr )
                {
                    throw std::runtime_error( "stdMotionNode::togglePutsOn: no input for multi-output node [" + name() +
                                              "]" );
                }
                ingr::instIOPut *pptr = m_node->inputs().begin()->second;

                // the single node is always on if any are on
                pptr->enabled( true );
                pptr->state( ingr::putState::on );

                ingr::putState inst = pptr->state();
                if( inst != ingr::putState::on )
                {
                    inst = ingr::putState::waiting;
                }

                // Now deal with the many
                for( auto s : m_presetPutName )
                {
                    pptr = m_node->output( s );

                    if( s == m_curVal || m_alwaysOn.count( s ) == 1 )
                    {
                        pptr->enabled( true );
                        pptr->state( inst );
                    }
                    else
                    {
                        pptr->state( ingr::putState::off );

                        if( m_noAutoOn.count( s ) == 1 ) // if we turn it off, we disable it
                        {
                            pptr->enabled( false );
                        }
                    }
                }
            }
            else // m_presetDir == ingr::ioDir::input )
            {
                if( m_node->outputs().empty() || m_node->outputs().begin()->second == nullptr )
                {
                    throw std::runtime_error( "stdMotionNode::togglePutsOn: no output for multi-input node [" + name() +
                                              "]" );
                }
                ingr::instIOPut *pptr = m_node->outputs().begin()->second;

                // the single node is always on if any are on
                pptr->enabled( true );
                pptr->state( ingr::putState::on );

                // Now deal with the many
                for( auto s : m_presetPutName )
                {
                    pptr = m_node->input( s );

                    if( s == m_curVal || m_alwaysOn.count( s ) == 1 )
                    {
                        pptr->enabled( true );
                        pptr->state( ingr::putState::on );
                    }
                    else
                    {
                        pptr->state( ingr::putState::off );

                        if( m_noAutoOn.count( s ) == 1 ) // if we turn it off, we disable it
                        {
                            pptr->enabled( false );
                        }
                    }
                }
            }
        }
        m_parentGraph->stateChange();
    }

    return;
}

inline void stdMotionNode::togglePutsOff()
{
    if( m_node == nullptr || !m_parentGraph || !m_node->auxDataValid() )
    {
        return;
    }

    if( !m_presetRoutes.empty() )
    {
        applyPresetRoute( {} );
        m_curLabel = "off";
        m_parentGraph->valueExtra( name(), "state", "---" );
        m_parentGraph->stateChange();
        return;
    }

    if( m_tracking && m_state != MagAOX::app::stateCodes::POWEROFF ) // Tracking is unavailable with motors off.
    {
        m_curLabel = "tracking";
        m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, "tracking" );
        m_parentGraph->valueExtra( m_node->name(), "state", "tracking" );
    }
    else if( m_trackingReq && m_state != MagAOX::app::stateCodes::POWEROFF )
    {
        m_curLabel = "not tracking";
        m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, "not tracking" );
        m_parentGraph->valueExtra( m_node->name(), "state", "not tracking" );
    }
    else if( m_presetPutName.size() == 1 ) // otherwise, if we have a single node it's off
    {
        m_curLabel = "off";
        m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, "off" );
        m_parentGraph->valueExtra( m_node->name(), "state", "---" );
    }
    else
    {
        // Multi-put labels name their routes; only the position status changes.
        if( m_state == MagAOX::app::stateCodes::POWEROFF )
        {
            m_curLabel = "off";
        }
        m_parentGraph->valueExtra( m_node->name(), "state", "---" );
    }

    // An alwaysOn put is on only while the stage has an active path.
    for( auto &&iput : m_node->inputs() )
    {
        iput.second->state( ingr::putState::off );
    }

    for( auto &&oput : m_node->outputs() )
    {
        oput.second->state( ingr::putState::off );

        if( m_noAutoOn.count( oput.second->name() ) == 1 ) // if we turn it off, we disable it
        {
            oput.second->enabled( false );
        }
    }
}

inline void stdMotionNode::loadPresetRoutes( mx::app::appConfigurator &config )
{
    const std::string        prefix = "presetRoute.";
    std::vector<std::string> rowKeys;
    for( const auto &entry : config.m_unusedConfigs )
    {
        if( entry.second.section == name() && entry.second.keyword.compare( 0, prefix.size(), prefix ) == 0 &&
            entry.second.set )
        {
            rowKeys.push_back( entry.second.keyword );
        }
    }
    if( rowKeys.empty() )
    {
        return;
    }

    const std::string context = "stdMotionNode::loadConfig: presetRoute in [" + name() + "] ";
    for( const auto &option : { "presetPutName",
                                "alwaysOn",
                                "noAutoOn",
                                "trackingReqKey",
                                "trackingReqElement",
                                "trackerKey",
                                "trackerElement" } )
    {
        if( config.isSetUnused( mx::app::iniFile::makeKey( name(), option ) ) )
        {
            throw std::runtime_error( context + "cannot be combined with '" + option + "'" );
        }
    }

    const auto       &selected = m_presetDir == ingr::ioDir::output ? m_node->outputs() : m_node->inputs();
    const auto       &common   = m_presetDir == ingr::ioDir::output ? m_node->inputs() : m_node->outputs();
    const std::string side     = m_presetDir == ingr::ioDir::output ? "output" : "input";
    if( selected.empty() || common.size() != 1 || common.begin()->second == nullptr )
    {
        throw std::runtime_error( context + "requires at least one " + side +
                                  " put and exactly one opposite-side put" );
    }
    for( const auto &put : selected )
    {
        if( put.second == nullptr )
        {
            throw std::runtime_error( context + "has null " + side + " put '" + put.first + "'" );
        }
        const auto input  = m_presetDir == ingr::ioDir::output ? common.begin()->second : put.second;
        const auto output = m_presetDir == ingr::ioDir::output ? put.second : common.begin()->second;
        if( input->outputLinks().count( output->name() ) == 0 )
        {
            throw std::runtime_error( context + "requires internal link 'input:" + name() + ':' + input->name() +
                                      "' -> 'output:" + name() + ':' + output->name() + "'" );
        }
    }

    std::map<std::string, std::set<std::string>> routes;
    for( const auto &key : rowKeys )
    {
        const std::string preset = key.substr( prefix.size() );
        const std::string row    = context + "row '" + key + "' ";
        if( preset.empty() || preset.find_first_not_of( " \t\r\n" ) == std::string::npos || preset == "none" )
        {
            throw std::runtime_error( row + "has an empty or reserved preset name" );
        }
        std::vector<std::string> puts;
        if( config.configUnused( puts, mx::app::iniFile::makeKey( name(), key ) ) < 0 )
        {
            throw std::runtime_error( row + "cannot be read" );
        }
        auto &route = routes[preset];
        for( auto put : puts )
        {
            const auto first = put.find_first_not_of( " \t\r\n" );
            if( first == std::string::npos )
            {
                throw std::runtime_error( row + "contains an empty put name" );
            }
            put = put.substr( first, put.find_last_not_of( " \t\r\n" ) - first + 1 );
            if( selected.count( put ) == 0 )
            {
                throw std::runtime_error( row + "put '" + put + "' is not a " + side + " put" );
            }
            if( !route.insert( put ).second )
            {
                throw std::runtime_error( row + "contains duplicate put '" + put + "'" );
            }
        }
    }
    m_presetRoutes = std::move( routes );
}

inline void stdMotionNode::loadConfig( mx::app::appConfigurator &config )
{
    if( !m_parentGraph )
    {
        std::string msg = XIGN_EXCEPTION( "stdMotionNode::loadConfig", "parent graph is null" );
        throw std::runtime_error( msg );
    }

    std::string type;
    config.configUnused( type, mx::app::iniFile::makeKey( name(), "type" ) );

    if( type != "stdMotion" )
    {
        std::string msg = XIGN_EXCEPTION( "stdMotionNode::loadConfig", "node type is not stdMotion" );
        throw std::runtime_error( msg );
    }

    std::string dev = name();
    config.configUnused( dev, mx::app::iniFile::makeKey( name(), "device" ) );

    std::string prePrefix = "preset";
    config.configUnused( prePrefix, mx::app::iniFile::makeKey( name(), "presetPrefix" ) );

    std::string preDir = "output";
    config.configUnused( preDir, mx::app::iniFile::makeKey( name(), "presetDir" ) );

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
        std::string msg = XIGN_EXCEPTION( "stdMotionNode::loadConfig", "invalid presetDir (must be input or output)" );
        throw std::runtime_error( msg );
    }

    loadPresetRoutes( config );
    std::vector<std::string> prePutName{ m_presetDir == ingr::ioDir::input ? "in" : "out" };
    if( !m_presetRoutes.empty() )
    {
        prePutName.clear();
        const auto &puts = m_presetDir == ingr::ioDir::input ? m_node->inputs() : m_node->outputs();
        for( const auto &put : puts )
        {
            prePutName.push_back( put.first );
        }
    }
    config.configUnused( prePutName, mx::app::iniFile::makeKey( name(), "presetPutName" ) );
    if( prePutName.size() == 0 )
    {
        std::string msg = XIGN_EXCEPTION( "stdMotionNode::loadConfig", "presetPutName can't be empty" );
        throw std::runtime_error( msg );
    }

    std::vector<std::string> alwaysOn;
    config.configUnused( alwaysOn, mx::app::iniFile::makeKey( name(), "alwaysOn" ) );
    try
    {
        for( auto &ao : alwaysOn )
        {
            m_alwaysOn.insert( ao );
        }
    }
    catch( const std::exception &e )
    {
        std::string msg = XIGN_EXCEPTION( "stdMotionNode::loadConfig", "exception from insert in m_alwaysOn" );
        msg += ":";
        msg += e.what();
        throw std::runtime_error( msg );
    }

    std::vector<std::string> noAutoOn;
    config.configUnused( noAutoOn, mx::app::iniFile::makeKey( name(), "noAutoOn" ) );
    try
    {
        for( auto &ao : noAutoOn )
        {
            m_noAutoOn.insert( ao );
        }
    }
    catch( const std::exception &e )
    {
        std::string msg = XIGN_EXCEPTION( "stdMotionNode::loadConfig", "exception from insert in m_noAutoOn" );
        msg += ":";
        msg += e.what();
        throw std::runtime_error( msg );
    }

    std::string trackReqKey;
    config.configUnused( trackReqKey, mx::app::iniFile::makeKey( name(), "trackingReqKey" ) );

    std::string trackReqEl;
    config.configUnused( trackReqEl, mx::app::iniFile::makeKey( name(), "trackingReqElement" ) );

    // Check if both are set
    if( ( trackReqKey == "" && trackReqEl != "" ) || ( trackReqKey != "" && trackReqEl == "" ) )
    {
        std::string msg = XIGN_EXCEPTION( "stdMotionNode::loadConfig",
                                          "trackingReqKey and trackingReqElement must both be provided" );
        throw std::runtime_error( msg );
    }

    std::string trackKey;
    config.configUnused( trackKey, mx::app::iniFile::makeKey( name(), "trackerKey" ) );

    std::string trackEl;
    config.configUnused( trackEl, mx::app::iniFile::makeKey( name(), "trackerElement" ) );

    // Check if both are set
    if( ( trackKey == "" && trackEl != "" ) || ( trackKey != "" && trackEl == "" ) )
    {
        std::string msg =
            XIGN_EXCEPTION( "stdMotionNode::loadConfig", "trackingKey and trackingElement must both be provided" );
        throw std::runtime_error( msg );
    }

    // This will catch the case where one or the other pair was set, but not both
    if( ( trackKey == "" && trackReqKey != "" ) || ( trackKey != "" && trackReqKey == "" ) )
    {
        std::string msg =
            XIGN_EXCEPTION( "stdMotionNode::loadConfig", "trackingReqKey and trackerKey must both be provided" );
        throw std::runtime_error( msg );
    }

    const auto           &presetPuts = m_presetDir == ingr::ioDir::output ? m_node->outputs() : m_node->inputs();
    std::set<std::string> seenPuts;
    for( const auto &put : prePutName )
    {
        if( put.empty() || presetPuts.count( put ) == 0 )
        {
            throw std::runtime_error( "stdMotionNode::loadConfig: presetPutName '" + put + "' is not a " + preDir +
                                      " put of [" + name() + "]" );
        }
        if( !seenPuts.insert( put ).second )
        {
            throw std::runtime_error( "stdMotionNode::loadConfig: duplicate presetPutName '" + put + "' in [" + name() +
                                      "]" );
        }
    }

    if( prePutName.size() > 1 )
    {
        const auto &oppositePuts = m_presetDir == ingr::ioDir::output ? m_node->inputs() : m_node->outputs();
        if( oppositePuts.size() != 1 || oppositePuts.begin()->second == nullptr )
        {
            throw std::runtime_error( "stdMotionNode::loadConfig: multi-put node [" + name() +
                                      "] requires exactly one opposite-side put" );
        }
    }

    for( const auto &put : m_alwaysOn )
    {
        if( m_node->inputs().count( put ) == 0 && m_node->outputs().count( put ) == 0 )
        {
            throw std::runtime_error( "stdMotionNode::loadConfig: alwaysOn put '" + put + "' is absent from node [" +
                                      name() + "]" );
        }
    }
    for( const auto &put : m_noAutoOn )
    {
        if( m_node->outputs().count( put ) == 0 )
        {
            throw std::runtime_error( "stdMotionNode::loadConfig: noAutoOn output '" + put + "' is absent from node [" +
                                      name() + "]" );
        }
    }

    config.configUnused( m_parkable, mx::app::iniFile::makeKey( name(), "parkable" ) );

    device( dev );
    presetPrefix( prePrefix );
    presetPutName( prePutName );
    trackingReqKey( trackReqKey );
    trackingReqElement( trackReqEl );
    trackerKey( trackKey );
    trackerElement( trackEl );
    if( !m_presetRoutes.empty() )
    {
        togglePutsOff();
    }
}

#endif // stdMotionNode_hpp
