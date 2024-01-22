#ifndef statusDisplay_hpp
#define statusDisplay_hpp

#include "ui_statusDisplay.h"

#include "xWidget.hpp"
#include "../../lib/multiIndiSubscriber.hpp"

namespace xqt 
{
   
class statusDisplay : public xWidget
{
    Q_OBJECT
   
protected:
   
    xWidget * m_ctrlWidget {nullptr};

    std::string m_device;
    std::string m_property;
    std::string m_element;

    std::string m_label;
    std::string m_units;


    bool m_highlightChanges {true};

    bool m_valChanged {false};

    std::string m_fsmState;
    std::string m_value;
    bool m_showVal {true};

public:
   statusDisplay( const std::string & device,
                  const std::string & property,
                  const std::string & element,
                  const std::string & label,
                  const std::string & units,
                  QWidget * Parent = 0, 
                  Qt::WindowFlags f = Qt::WindowFlags()
                );
   
   ~statusDisplay();

   void ctrlWidget (xWidget * cw);

   xWidget * ctrlWidget();
   
   virtual QString formatValue();

   virtual void subscribe();
             
   virtual void onConnect();

   virtual void onDisconnect();
   
   virtual void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   virtual void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
public:
    /// 
    virtual void changeEvent(QEvent * e)
    {
        if(e->type() == QEvent::EnabledChange && !isEnabledTo(nullptr))
        {
            if(m_ctrlWidget) m_ctrlWidget->hide();
        }
      
        xWidget::changeEvent(e);
    }

public slots:

    virtual void updateGUI();

    void on_button_pressed()
    {
        if(m_ctrlWidget) 
        {
            m_ctrlWidget->show();
            m_ctrlWidget->onConnect();
        }
    }

signals:

    void doUpdateGUI();

protected:
     
   Ui::statusDisplay ui;
};
   
statusDisplay::statusDisplay( const std::string & device,
                              const std::string & property,
                              const std::string & element, 
                              const std::string & label,
                              const std::string & units,
                              QWidget * Parent, 
                              Qt::WindowFlags f) : xWidget(Parent, f), m_device{device}, m_property{property}, m_element{element}, m_label{label}, m_units{units}
{
    ui.setupUi(this);
    std::string lab = m_label;
    if(m_units != "") lab += " [" + m_units + "]";
    ui.label->setText(lab.c_str());

    QFont qf = ui.label->font();
    qf.setPixelSize(XW_FONT_SIZE);
    ui.label->setFont(qf);

    qf = ui.status->font();
    qf.setPixelSize(XW_FONT_SIZE);
    ui.status->setFont(qf);

    connect(this, SIGNAL(doUpdateGUI()), this, SLOT(updateGUI()));

    onDisconnect();
}
   
statusDisplay::~statusDisplay()
{
}

void statusDisplay::ctrlWidget (xWidget * cw)
{
   if(m_ctrlWidget) m_ctrlWidget->deleteLater();
   m_ctrlWidget = cw;
}
   
xWidget * statusDisplay::ctrlWidget()
{
   return m_ctrlWidget;
}

QString statusDisplay::formatValue()
{
   return QString(m_value.c_str());
}

void statusDisplay::subscribe()
{
   if(!m_parent) return;
   
   m_parent->addSubscriberProperty(this, m_device, "fsm");

   if(m_property != "") m_parent->addSubscriberProperty(this, m_device, m_property);

   if(m_ctrlWidget) m_parent->addSubscriber(m_ctrlWidget);

   return;
}
  
void statusDisplay::onConnect()
{
   m_valChanged = true;
   if(m_ctrlWidget) m_ctrlWidget->onConnect();
}

void statusDisplay::onDisconnect()
{
   ui.status->setText("---");
   if(m_ctrlWidget) m_ctrlWidget->onDisconnect();
}

void statusDisplay::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void statusDisplay::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
    if(ipRecv.getDevice() != m_device) return;
   
    if(ipRecv.getName() == "fsm")
    {
        if(ipRecv.find("state"))
        {
            std::string fsmState = ipRecv["state"].get();
            
            if(fsmState == "READY" || fsmState == "OPERATING")
            {
                if(fsmState != m_fsmState)
                {
                    m_valChanged = true;
                }

                if(!m_showVal)
                {
                    m_valChanged = true;
                }

                m_showVal = true;
            }
            else
            {
                m_showVal = false;

                if(fsmState != m_fsmState)
                {
                    m_valChanged = true;
                }

                if(m_showVal)
                {
                    m_valChanged = true;
                }
            }

            m_fsmState = fsmState;
        }

    }
    else if(ipRecv.getName() == m_property)
    {
        if(ipRecv.find(m_element))
        {
            std::string value = ipRecv[m_element].get();
            if(value != m_value) m_valChanged = true;
            m_value = value;
        }
    }

    emit doUpdateGUI();
}

void statusDisplay::updateGUI()
{
    if(isEnabled())
    {
        if(m_showVal)
        {
            if(m_valChanged)
            {
                QString value = formatValue();
      
                ui.status->setTextChanged(value);  
                m_valChanged = false;
            }
        }
        else
        {
            if(m_valChanged)
            {
                ui.status->setTextChanged(m_fsmState.c_str());
                m_valChanged = false;
            }
        }
   }

} //updateGUI()


} //namespace xqt
   
#include "moc_statusDisplay.cpp"

#endif
