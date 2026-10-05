/** \file stageStatus.hpp
 * \brief Summary status widget for stage and filter-wheel devices.
 */

#ifndef stageStatus_hpp
#define stageStatus_hpp

#include <cstdio>
#include <QWidget>

#include "ui_statusDisplay.h"

#include "../xWidgets/statusCombo.hpp"
#include "../stage/stage.hpp"

namespace xqt
{

/// Stage-status summary widget with preset/filter selection support.
class stageStatus : public statusCombo
{
    Q_OBJECT

  protected:
    /// Most recent numeric position used when no named preset is active.
    float m_position{ 0 };

  public:
    /// Construct an unconfigured stage-status summary widget.
    stageStatus( QWidget        *Parent /**< [in] owning Qt parent widget */   = 0,
                 Qt::WindowFlags f /**< [in] Qt window flags for the widget */ = Qt::WindowFlags() );

    /// Construct and configure a stage-status summary widget.
    stageStatus( const std::string &stgN /**< [in] stage device name */,
                 QWidget           *Parent /**< [in] owning Qt parent widget */   = 0,
                 Qt::WindowFlags    f /**< [in] Qt window flags for the widget */ = Qt::WindowFlags() );

    /// Destroy the widget.
    ~stageStatus();

    /// Configure the summary widget for one stage device.
    void setup( const std::string &stgN /**< [in] stage device name */ );

    /// Format either the active preset name or the live numeric position.
    virtual QString formatValue();

    /// Subscribe to the stage properties needed by the summary widget.
    virtual void subscribe();

    /// Reset the widget for a disconnected stage controller.
    virtual void onDisconnect();

    /// Handle deletion of a stage property used by this widget.
    void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has been deleted*/ );

    /// Handle updates to the stage position and preset/filter properties.
    void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

  protected:
    /// Return `true` when the property name is a selectable preset/filter switch.
    bool isSelectionProperty( const std::string &propertyName /**< [in] property name to test */ ) const;
};

stageStatus::stageStatus( QWidget *Parent, Qt::WindowFlags f ) : statusCombo( Parent, f )
{
}

stageStatus::stageStatus( const std::string &stageName, QWidget *Parent, Qt::WindowFlags f ) : statusCombo( Parent, f )
{
    setup( stageName );
}

stageStatus::~stageStatus()
{
}

void stageStatus::setup( const std::string &stageName )
{
    statusCombo::setup( stageName, "", "", stageName, "" );
    ctrlWidget( new stage( stageName, this, Qt::Dialog ) );
}

QString stageStatus::formatValue()
{
    if( m_value == "" || m_value == "none" )
    {
        char pstr[64];
        snprintf( pstr, sizeof( pstr ), "%0.4f", m_position );
        return QString( pstr );
    }
    else
    {
        return statusCombo::formatValue();
    }
}

void stageStatus::subscribe()
{
    if( !m_parent )
    {
        return;
    }

    m_parent->addSubscriberProperty( this, m_device, "position" );
    m_parent->addSubscriberProperty( this, m_device, "filter" );
    m_parent->addSubscriberProperty( this, m_device, "presetName" );
    m_parent->addSubscriberProperty( this, m_device, "filterName" );

    statusCombo::subscribe();

    return;
}

void stageStatus::onDisconnect()
{
    statusCombo::onDisconnect();
    m_property.clear();
    m_position = 0;
}

void stageStatus::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_device )
        return;

    if( isSelectionProperty( ipRecv.getName() ) && ( m_property == "" || m_property == ipRecv.getName() ) )
    {
        m_property = ipRecv.getName();
    }

    if( ipRecv.getName() == "position" || ipRecv.getName() == "filter" )
    {
        if( ipRecv.find( "current" ) )
        {
            float pos = ipRecv["current"].get<float>();
            if( pos != m_position && ( m_value == "none" || m_value == "" ) )
            {
                m_valChanged = true;
                m_position   = pos;
            }
        }
    }

    statusCombo::handleSetProperty( ipRecv ); // always emit updateGUI
}

void stageStatus::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_device )
        return;

    bool clearSelectionProperty = ( ipRecv.getName() == m_property && isSelectionProperty( ipRecv.getName() ) );

    if( ipRecv.getName() == "position" || ipRecv.getName() == "filter" )
    {
        m_position = 0;
        if( m_value == "none" || m_value == "" )
        {
            m_valChanged = true;
        }
    }

    statusCombo::handleDelProperty( ipRecv );

    if( clearSelectionProperty )
    {
        m_property.clear();
    }
}

bool stageStatus::isSelectionProperty( const std::string &propertyName ) const
{
    return ( propertyName == "presetName" || propertyName == "filterName" );
}

} // namespace xqt

#include "moc_stageStatus.cpp"

#endif
