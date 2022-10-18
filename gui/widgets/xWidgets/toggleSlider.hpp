#ifndef toggleSlider_hpp
#define toggleSlider_hpp

#include "ui_toggleSlider.h"

#include "xWidget.hpp"
#include "../../lib/multiIndiSubscriber.hpp"

namespace xqt 
{
   
class toggleSlider : public xWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_device;
   std::string m_property;
   std::string m_element {"toggle"};

   std::string m_label;

   bool m_highlightChanges {true};

   int m_status; // 0 or off, 1 for int, 2 for on

   bool m_statusChanged {false};

   QTimer * m_tgtTimer {nullptr}; ///< Timer to normalize target if property stalls (5 seconds).

public:
   toggleSlider( const std::string & device,
                 const std::string & property,
                 const std::string & label,
                 QWidget * Parent = 0, 
                 Qt::WindowFlags f = Qt::WindowFlags()
               );

   toggleSlider( const std::string & device,
                 const std::string & property,
                 const std::string & element,
                 const std::string & label,
                 QWidget * Parent = 0, 
                 Qt::WindowFlags f = Qt::WindowFlags()
               );
   
   ~toggleSlider();

   
   virtual void subscribe();
             
   virtual void onConnect();

   virtual void onDisconnect();
   
   virtual void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   virtual void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   virtual void updateGUI();

public slots:

   void on_slider_sliderReleased();

   void timeout();

signals:

   void tgtTimerStart(int);

protected:
     
   Ui::toggleSlider ui;
};
   
toggleSlider::toggleSlider( const std::string & device,
                            const std::string & property,
                            const std::string & label,
                            QWidget * Parent, 
                            Qt::WindowFlags f) : xWidget(Parent, f), m_device{device}, m_property{property}, m_label{label}
{
   ui.setupUi(this);
   ui.label->setText(m_label.c_str());

   QFont qf = ui.label->font();
   qf.setPixelSize(XW_FONT_SIZE);
   ui.label->setFont(qf);

   m_tgtTimer = new QTimer(this);
   
   connect(m_tgtTimer, SIGNAL(timeout()), this, SLOT(timeout()));
   connect(this, SIGNAL(tgtTimerStart(int)), m_tgtTimer, SLOT(start(int)));

   onDisconnect();
}

toggleSlider::toggleSlider( const std::string & device,
                            const std::string & property,
                            const std::string & element, 
                            const std::string & label,
                            QWidget * Parent, 
                            Qt::WindowFlags f) : xWidget(Parent, f), m_device{device}, m_property{property}, m_element{element}, m_label{label}
{
   ui.setupUi(this);
   ui.label->setText(m_label.c_str());

   QFont qf = ui.label->font();
   qf.setPixelSize(XW_FONT_SIZE);
   ui.label->setFont(qf);

   m_tgtTimer = new QTimer(this);
   
   connect(m_tgtTimer, SIGNAL(timeout()), this, SLOT(timeout()));
   connect(this, SIGNAL(tgtTimerStart(int)), m_tgtTimer, SLOT(start(int)));

   onDisconnect();
}
   
toggleSlider::~toggleSlider()
{
}

void toggleSlider::subscribe()
{
   if(!m_parent) return;
   
   if(m_property != "") m_parent->addSubscriberProperty(this, m_device, m_property);

   return;
}
  
void toggleSlider::onConnect()
{
   m_statusChanged = true;
}

void toggleSlider::onDisconnect()
{
   ui.slider->setSliderPosition(ui.slider->minimum());
}

void toggleSlider::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void toggleSlider::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_device) return;
   
   if(ipRecv.getName() == m_property)
   {
      if(ipRecv.find("toggle"))
      {
         int currStatus = m_status;
         if(ipRecv.getState() == pcf::IndiProperty::Busy) m_status = 1;
         else if(ipRecv["toggle"] == pcf::IndiElement::On) m_status = 2;
         else m_status = 0;

         if(currStatus != m_status) m_statusChanged = true;

      }
   }

   updateGUI();
}

void toggleSlider::updateGUI()
{
   if(isEnabled())
   {
      if(m_status == 2) ui.slider->setSliderPosition(ui.slider->maximum());
      else if(m_status == 1) ui.slider->setSliderPosition(ui.slider->minimum() + 0.5*(ui.slider->maximum()-ui.slider->minimum()));
      else ui.slider->setSliderPosition(ui.slider->minimum());
   }
   else ui.slider->setSliderPosition(ui.slider->minimum());

} //updateGUI()

void toggleSlider::on_slider_sliderReleased()
{
   int state = -1;

   if( ui.slider->sliderPosition() > ui.slider->minimum()+0.8*(ui.slider->maximum()-ui.slider->minimum()))
   {
      state = 2;   
   }
   else if( ui.slider->sliderPosition() < ui.slider->minimum()+0.2*(ui.slider->maximum()-ui.slider->minimum())) 
   {
      state = 0;
   }

   if(state == 2)
   {
      pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
      ipFreq.setDevice(m_device);
      ipFreq.setName(m_property);
      ipFreq.add(pcf::IndiElement("toggle"));
      ipFreq["toggle"].setSwitchState(pcf::IndiElement::On);
    
      sendNewProperty(ipFreq);  
      emit tgtTimerStart(5000);
   }
   else if(state == 0)
   {
      pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
      ipFreq.setDevice(m_device);
      ipFreq.setName(m_property);
      ipFreq.add(pcf::IndiElement("toggle"));
      ipFreq["toggle"].setSwitchState(pcf::IndiElement::Off);
    
      sendNewProperty(ipFreq); 
      emit tgtTimerStart(5000);
   }
   else
   {
      if(m_status == 2) ui.slider->setSliderPosition(ui.slider->maximum());
      else if(m_status == 1) ui.slider->setSliderPosition(ui.slider->minimum() + 0.5*(ui.slider->maximum()-ui.slider->minimum()));
      else ui.slider->setSliderPosition(ui.slider->minimum());
   }
}

void toggleSlider::timeout()
{
}

} //namespace xqt
   
#include "moc_toggleSlider.cpp"

#endif
