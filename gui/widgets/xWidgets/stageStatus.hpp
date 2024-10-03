#ifndef stageStatus_hpp
#define stageStatus_hpp

#include <QWidget>

#include "ui_statusDisplay.h"

#include "../xWidgets/statusDisplay.hpp"
#include "../stage/stage.hpp"

namespace xqt
{

class stageStatus : public statusDisplay
{
   Q_OBJECT

protected:

    std::string m_presetName;
    float m_position;

public:

    stageStatus( QWidget * Parent = 0,
                Qt::WindowFlags f = Qt::WindowFlags()
              );

   stageStatus( const std::string & stgN,
                QWidget * Parent = 0,
                Qt::WindowFlags f = Qt::WindowFlags()
              );

   ~stageStatus();

   void setup(const std::string & stgN);

   virtual QString formatValue();

   virtual void subscribe();

   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

   //void updateGUI();

};

stageStatus::stageStatus(  QWidget * Parent,
                      Qt::WindowFlags f) : statusDisplay(Parent, f)
{
}

stageStatus::stageStatus( const std::string & stageName,
                      QWidget * Parent,
                      Qt::WindowFlags f) : statusDisplay(Parent, f)
{
   setup(stageName);
}

stageStatus::~stageStatus()
{
}

void stageStatus::setup(const std::string & stageName)
{
    statusDisplay::setup(stageName, "", "", stageName, "");
    if(m_ctrlWidget)
    {
        delete m_ctrlWidget;
    }

    m_ctrlWidget = (xWidget *) (new stage(stageName, this, Qt::Dialog));

}
QString stageStatus::formatValue()
{
    if(m_presetName == "" || m_presetName == "none")
    {
        char pstr[64];
        snprintf(pstr, sizeof(pstr), "%0.4f", m_position);
        return QString(pstr);
    }
    else
    {
        return QString(m_presetName.c_str());
    }
}

void stageStatus::subscribe()
{
    if(!m_parent) return;

    m_parent->addSubscriberProperty(this, m_device, "presetName");
    m_parent->addSubscriberProperty(this, m_device, "filterName");
    m_parent->addSubscriberProperty(this, m_device, "position");
    m_parent->addSubscriberProperty(this, m_device, "filter");

    statusDisplay::subscribe();

    return;
}

void stageStatus::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
    if(ipRecv.getDevice() != m_device) return;

    if(ipRecv.getName() == "presetName" || ipRecv.getName() == "filterName")
    {
        auto map = ipRecv.getElements();

        for(auto it = map.begin(); it != map.end(); ++it)
        {
            if(it->second.getSwitchState() == pcf::IndiElement::On)
            {
                if(m_presetName != it->first)
                {
                    m_presetName = it->first;
                    m_valChanged = true;
                }

                break;
            }
        }
    }
    else if(ipRecv.getName() == "position" || ipRecv.getName() == "filter")
    {
        if(ipRecv.find("current"))
        {
            float pos = ipRecv["current"].get<float>();
            if(pos != m_position && (m_presetName == "none" || m_presetName == ""))
            {
                m_valChanged = true;
                m_position = pos;
            }
        }
    }

    statusDisplay::handleSetProperty(ipRecv);
}

} //namespace xqt

#include "moc_stageStatus.cpp"

#endif
