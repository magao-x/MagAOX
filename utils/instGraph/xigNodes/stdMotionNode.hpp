/** \file stdMotionNode.hpp
 * \brief The MagAO-X Instrument Graph stdMotionNode header file
 *
 * \ingroup instGraph_files
 */

#ifndef stdMotionNode_hpp
#define stdMotionNode_hpp

#include "fsmNode.hpp"

///
/**
 *
 * The key assumption of this node is that it should be in a valid, not-`none`, preset position
 * for its ioputs to be `on`.  It also supports triggering an alternate `on` state, which is used for
 * stages which have a continuous tracking mode (k-mirror and ADC).
 *
 * The preset is specified by an INDI property with signature `<device>.<presetPrefix>Name` where device
 * and presetPrefix are part of the configuration.  This INDI property is a switch vector.
 *
 * The device and prefix can only be set once.
 */
class stdMotionNode : public fsmNode
{

  protected:
    /// The prefix for preset naes.  Usually either "preset" or "filter", to which "Name" is appended.
    std::string m_presetPrefix;

    /// The INDI key (device.property) for the presets.  This is, say, `fwpupil.filterName`.  It is set automatically.
    std::string m_presetKey;

    /// The current value of the preset property.  Corresponds to the element name of the selected preset.
    std::string m_curVal;

    /// The current value of the put label.
    std::string m_curLabel;

    std::vector<std::string> m_presetPutName{ "out" };

    /// This sets whether the multi-put selector is on the input or the output (default)
    /** If this is a multi-put node (m_presetPutName.size() > 1) then the value of the preset switch
     * controls which input or output is on, with the others off.
     */
    ingr::ioDir m_presetDir{ ingr::ioDir::output };

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
    /// Only c'tor.  Must be constructed with node name and a parent graph.
    stdMotionNode( const std::string &name, ingr::instGraphXML *parentGraph );

    /// Set the device name.  This can only be done once.
    /**
     * \throws
     */
    virtual void device( const std::string &dev /**< [in] */);

    using fsmNode::device;

    virtual void presetPrefix( const std::string &pp /**< [in] */);

    const std::string & presetPrefix();

    /// Get the current label text
    /**
     * \returns the current value of m_curLabel.
     */
    const std::string & curLabel();

    void presetPutName( const std::vector<std::string> &ppp /**< [in] */);

    const std::vector<std::string> & presetPutName();

    void presetDir( const ingr::ioDir &dir /**< [in] */);

    const ingr::ioDir & presetDir();

    void trackingReqKey( const std::string &tk /**< [in] */);

    const std::string & trackingReqKey();

    void trackingReqElement( const std::string &te /**< [in] */);

    const std::string & trackingReqElement();

    void trackerKey( const std::string &tk /**< [in] */);

    const std::string & trackerKey();

    void trackerElement( const std::string &te /**< [in] */);

    const std::string & trackerElement();

    /// INDI SetProperty callback
    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the received INDI property to handle*/ );

    virtual void togglePutsOn();

    virtual void togglePutsOff();

    void loadConfig( mx::app::appConfigurator &config /**< [in] the application configurator loaded with this node's options*/);
};

inline stdMotionNode::stdMotionNode( const std::string &name, ingr::instGraphXML *parentGraph ) : fsmNode( name, parentGraph )
{
}

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
        std::string msg = "stdMotionNode::presetPrefix: attempt to change preset prefix from " + m_presetPrefix + " to " + pp;
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

inline const std::string & stdMotionNode::presetPrefix()
{
    return m_presetPrefix;
}

inline const std::string & stdMotionNode::curLabel()
{
    return m_curLabel;
}

inline void stdMotionNode::presetPutName( const std::vector<std::string> &ppp )
{
    m_presetPutName = ppp;
}

inline const std::vector<std::string> & stdMotionNode::presetPutName()
{
    return m_presetPutName;
}

inline void stdMotionNode::presetDir( const ingr::ioDir &dir )
{
    m_presetDir = dir;
}

inline const ingr::ioDir & stdMotionNode::presetDir()
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

inline const std::string & stdMotionNode::trackingReqKey()
{
    return m_trackingReqKey;
}

inline void stdMotionNode::trackingReqElement( const std::string &te )
{
    m_trackingReqElement = te;
}

inline const std::string & stdMotionNode::trackingReqElement()
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

inline const std::string & stdMotionNode::trackerKey()
{
    return m_trackerKey;
}

inline void stdMotionNode::trackerElement( const std::string &te )
{
    m_trackerElement = te;
}

inline const std::string & stdMotionNode::trackerElement()
{
    return m_trackerElement;
}

inline void stdMotionNode::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    fsmNode::handleSetProperty( ipRecv );

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
    else if( ipRecv.createUniqueKey() == m_presetKey )
    {
        if( m_node != nullptr )
        {
            bool nothingIsOn = true;
            for( auto &&it : ipRecv.getElements() )
            {
                if( it.second.getSwitchState() == pcf::IndiElement::On )
                {
                    if( m_curVal != it.second.getName() && !m_tracking ) // we only update if not tracking
                    {
                        ++m_changes;
                    }

                    m_curVal = it.second.getName();
                    nothingIsOn = false;
                }
            }

            if(nothingIsOn)
            {
                if( m_curVal != "" && !m_tracking ) // we only update if not tracking
                {
                    ++m_changes;
                }
                m_curVal = "";
            }
        }
    }

    if( m_changes > 0 )
    {
        m_changes = 0;

        if(m_trackingReq)
        {
            if(m_tracking && (m_state == MagAOX::app::stateCodes::READY || m_state == MagAOX::app::stateCodes::OPERATING ) )
            {
                togglePutsOn();
            }
            else
            {   //Either we aren't tracking or we aren't READY || OPERATING
                togglePutsOff();
            }
        }
        else
        {
            if( m_state != MagAOX::app::stateCodes::READY || m_tracking || m_curVal == "none" || m_curVal == "")
            {
                togglePutsOff();
            }
            else
            {
                togglePutsOn();
            }
        }
    }
}

inline void stdMotionNode::togglePutsOn()
{
    if(m_node == nullptr || !m_parentGraph || !m_node->auxDataValid())
    {
        return;
    }

    if( m_trackingReq )
    {
        if( m_tracking )
        {
            m_curLabel = "tracking";
            m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, "tracking" );
            xigNode::togglePutsOn();
        }
        else
        {
            m_curLabel = "not tracking";
            m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, "not tracking" );
            m_parentGraph->stateChange();
        }
    }
    else if( m_state == MagAOX::app::stateCodes::READY )
    {
        m_curLabel = m_curVal;

        if( m_presetPutName.size() == 1 ) // There's only one put, it's just on or off with a value
        {
            m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, m_curVal );
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
            m_parentGraph->stateChange();
        }
    }

    return;
}

inline void stdMotionNode::togglePutsOff()
{
    if(m_node == nullptr || !m_parentGraph || !m_node->auxDataValid())
    {
        return;
    }

    if( m_tracking ) //regardless of whether required, if tracking this is our state
    {
        m_curLabel = "tracking";
        m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, "tracking" );
    }
    else if( m_trackingReq ) //we can only be "not tracking" if tracking is required
    {
        m_curLabel = "not tracking";
        m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, "not tracking" );
    }
    else if( m_presetPutName.size() == 1 ) //otherwise, if we have a single node it's off
    {
        m_curLabel = "off";
        m_parentGraph->valuePut( name(), m_presetPutName[0], m_presetDir, "off" );
    }
    //We don't change labels if m_presetPutName.size() > 1

    xigNode::togglePutsOff();
}

inline void stdMotionNode::loadConfig( mx::app::appConfigurator &config )
{
    if( !m_parentGraph )
    {
        std::string msg = "stdMotionNode::loadConfig: parent graph is null";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    std::string type;
    config.configUnused(type, mx::app::iniFile::makeKey( name(), "type" ));

    if(type != "stdMotion")
    {
        std::string msg = "stdMotionNode::loadConfig: node type is not stdMotion";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
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
        std::string msg = "stdMotionNode::loadConfig: invalid presetDir (must be input or output)";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }


    std::vector<std::string> prePutName( { "out" } );
    config.configUnused( prePutName, mx::app::iniFile::makeKey( name(), "presetPutName" ) );
    if( prePutName.size() == 0 )
    {
        std::string msg = "stdMotionNode::loadConfig: presetPutName can't be empty";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    std::string trackReqKey;
    config.configUnused( trackReqKey, mx::app::iniFile::makeKey( name(), "trackingReqKey" ) );

    std::string trackReqEl;
    config.configUnused( trackReqEl, mx::app::iniFile::makeKey( name(), "trackingReqElement" ) );

    //Check if both are set
    if( ( trackReqKey == "" && trackReqEl != "" ) || ( trackReqKey != "" && trackReqEl == "" ) )
    {
        std::string msg = "stdMotionNode::loadConfig: trackingReqKey and trackingReqElement must both be provided)";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    std::string trackKey;
    config.configUnused( trackKey, mx::app::iniFile::makeKey( name(), "trackerKey" ) );

    std::string trackEl;
    config.configUnused( trackEl, mx::app::iniFile::makeKey( name(), "trackerElement" ) );

    //Check if both are set
    if( ( trackKey == "" && trackEl != "" ) || ( trackKey != "" && trackEl == "" ) )
    {
        std::string msg = "stdMotionNode::loadConfig: trackerKey and trackerElement must both be provided)";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    //This will catch the case where one or the other pair was set, but not both
    if( ( trackKey == "" && trackReqKey != "" ) || ( trackKey != "" && trackReqKey == "" ) )
    {
        std::string msg = "stdMotionNode::loadConfig: trackingReqKey and trackerKey must both be provided)";
        msg += " at ";
        msg += __FILE__;
        msg += " " + std::to_string( __LINE__ );
        throw std::runtime_error( msg );
    }

    device( dev );
    presetPrefix( prePrefix );
    presetPutName( prePutName );
    trackingReqKey( trackReqKey );
    trackingReqElement( trackReqEl );
    trackerKey( trackKey );
    trackerElement( trackEl );
}

#endif // stdMotionNode_hpp
