#ifndef stageStatus_hpp
#define stageStatus_hpp

#include <QWidget>

#include "ui_statusDisplay.h"

#include "../xWidgets/statusCombo.hpp"
#include "../stage/stage.hpp"

namespace xqt
{

class stageStatus : public statusCombo
{
    Q_OBJECT

  protected:

    float m_position;

  public:
    stageStatus( QWidget *Parent = 0, Qt::WindowFlags f = Qt::WindowFlags() );

    stageStatus( const std::string &stgN, QWidget *Parent = 0, Qt::WindowFlags f = Qt::WindowFlags() );

    ~stageStatus();

    void setup( const std::string &stgN );

    virtual QString formatValue();

    virtual void subscribe();

    void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    // void updateGUI();
};

stageStatus::stageStatus( QWidget *Parent, Qt::WindowFlags f ) : statusCombo( Parent, f )
{
}

stageStatus::stageStatus( const std::string &stageName, QWidget *Parent, Qt::WindowFlags f )
    : statusCombo( Parent, f )
{
    setup( stageName );
}

stageStatus::~stageStatus()
{
}

void stageStatus::setup( const std::string &stageName )
{
    if(stageName.find("fw") == 0)
    {
        statusCombo::setup( stageName, "filterName", "", stageName, "" );
    }
    else
    {
        statusCombo::setup( stageName, "presetName", "", stageName, "" );
    }

    if( m_ctrlWidget )
    {
        delete m_ctrlWidget;
    }

    m_ctrlWidget = (xWidget *)( new stage( stageName, this, Qt::Dialog ) );
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

    //m_parent->addSubscriberProperty( this, m_device, "presetName" );
    //m_parent->addSubscriberProperty( this, m_device, "filterName" );
    m_parent->addSubscriberProperty( this, m_device, "position" );
    m_parent->addSubscriberProperty( this, m_device, "filter" );

    statusCombo::subscribe();

    return;
}

void stageStatus::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_device )
        return;

    if( ipRecv.getName() == "position" || ipRecv.getName() == "filter" )
    {
        if( ipRecv.find( "current" ) )
        {
            float pos = ipRecv["current"].get<float>();
            if( pos != m_position && ( m_value == "none" || m_value == "" ) )
            {
                m_valChanged = true;
                m_position = pos;
            }
        }
    }

    statusCombo::handleSetProperty( ipRecv ); //always emit updateGUI
}

} // namespace xqt

#include "moc_stageStatus.cpp"

#endif
