#ifndef gainCtrl_hpp
#define gainCtrl_hpp

#include "ui_gainCtrl.h"

#include "xWidget.hpp"
#include "statusEntry.hpp"
#include "../../lib/multiIndiSubscriber.hpp"

#define GAIN (0)
#define MULTCOEFF (1)
namespace xqt 
{
   
class gainCtrl : public xWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_device;
   std::string m_property;
   std::string m_label;

   float m_current {0};
   float m_target {0};

   bool m_valChanged {false};
   bool m_newSent {false};

   float m_scale {0.05};

   int m_ctrlType {GAIN};
   float m_maxVal {1.5};
   bool m_enforceMax {false};
public:

   gainCtrl( QWidget * Parent = 0, 
             Qt::WindowFlags f = 0
           );

   gainCtrl( const std::string & device,
             const std::string & property,
             const std::string & label,
             QWidget * Parent = 0, 
             Qt::WindowFlags f = 0
           );
   
   ~gainCtrl();

   void setup( const std::string & device,
               const std::string & property,
               const std::string & label
             );

   void makeGainCtrl();

   void makeMultCoeffCtrl();

   virtual void subscribe();
             
   virtual void onConnect();

   virtual void onDisconnect();
   
   virtual void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   virtual void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   virtual void updateGUI();

   void setGain( float g );

public slots:

   void on_button_plus_pressed();
   void on_button_scale_pressed();
   void on_button_minus_pressed();
   void on_button_zero_pressed();

protected:
     
   Ui::gainCtrl ui;
};

gainCtrl::gainCtrl( QWidget * Parent, 
                    Qt::WindowFlags f) : xWidget(Parent, f)
{
   ui.setupUi(this);
}

gainCtrl::gainCtrl( const std::string & device,
                    const std::string & property,
                    const std::string & label,
                    QWidget * Parent, 
                    Qt::WindowFlags f) : xWidget(Parent, f)
{
   ui.setupUi(this);
   
   setup(device, property, label);
}
   
gainCtrl::~gainCtrl()
{
}

void gainCtrl::setup( const std::string & device,
                      const std::string & property,
                      const std::string & label
                    )
{
   m_device = device;
   m_property = property;
   m_label = label;

   ui.status->setup(m_device, m_property, statusEntry::FLOAT, "", "");
   ui.status->setStretch(0,0,1);
   

   setXwFont(ui.button_plus);
   setXwFont(ui.button_scale);
   setXwFont(ui.button_minus);
   setXwFont(ui.button_zero);
   setXwFont(ui.label);
   
   ui.label->setText(m_label.c_str());

   makeGainCtrl();
   
   onDisconnect();
}

void gainCtrl::makeGainCtrl()
{
   m_ctrlType = GAIN;
   ui.status->format("%0.2f");
   m_maxVal = 1.5;
   ui.button_zero->setText("0");
   m_enforceMax = false;

   m_scale = 0.05;
   char sstr[16];
   snprintf(sstr, sizeof(sstr), "%0.2f", m_scale);
   ui.button_scale->setText(sstr);
}

void gainCtrl::makeMultCoeffCtrl()
{
   m_ctrlType = MULTCOEFF;
   ui.status->format("%0.3f");
   m_maxVal = 1.0;
   ui.button_zero->setText("1");
   m_enforceMax = true;

   m_scale = 0.005;
   char sstr[16];
   snprintf(sstr, sizeof(sstr), "%0.3f", m_scale);
   ui.button_scale->setText(sstr);
}

void gainCtrl::subscribe()
{
   if(!m_parent) return;
   
   if(m_property != "") m_parent->addSubscriberProperty(this, m_device, m_property);

   m_parent->addSubscriber(ui.status);

   return;
}
  
void gainCtrl::onConnect()
{
   ui.status->onConnect();
}

void gainCtrl::onDisconnect()
{
   ui.status->onDisconnect();
}

void gainCtrl::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void gainCtrl::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_device) return;
   
   if(ipRecv.getName() == m_property)
   {
      if(m_newSent)
      {
         m_newSent=false;
      }
      else
      {
         if(ipRecv.find("current"))
         {
            float current = ipRecv["current"].get<float>();
            if(current != m_current)
            {
               m_valChanged = true;
               m_current = current;
            }
         }
   
         if(ipRecv.find("target"))
         {
            m_target = ipRecv["target"].get<float>();
         }
      }
   }

   updateGUI();

   ui.status->handleSetProperty(ipRecv);
}

void gainCtrl::updateGUI()
{
   if(isEnabled())
   {
      int slv = m_current/m_maxVal * 150. + 0.5;
      if(slv < 0 ) slv = 0;
      else if(slv > 150) slv = 150;

      ui.slider->setValue(slv);

   }

} //updateGUI()

void gainCtrl::setGain( float g )
{
   if(g < 0 ) g = 0;
   if(m_enforceMax && g > m_maxVal) g = m_maxVal;

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_device);
   ip.setName(m_property);
   ip.add(pcf::IndiElement("target"));
   ip["target"] = g;
   
   sendNewProperty(ip);
   m_newSent = true;

   m_current = g; //do this so multiple pushes update fast

}

void gainCtrl::on_button_plus_pressed()
{
   setGain(m_current + m_scale);
}

void gainCtrl::on_button_scale_pressed()
{
   if(m_ctrlType == GAIN)
   {
      if( (int) (m_scale * 100 + 0.5) <= 1 )
      {
         m_scale = 0.1;
      }
      else if( (int) (m_scale * 100 + 0.5) >= 10 )
      {
         m_scale = 0.05;
      }
      else
      {
         m_scale = 0.01;
      }
      
      char sstr[16];
      snprintf(sstr, sizeof(sstr), "%0.2f", m_scale);
      ui.button_scale->setText(sstr);
   }
   else if(m_ctrlType == MULTCOEFF)
   {
      if( (int) (m_scale * 1000 + 0.5) <= 1 )
      {
         m_scale = 0.01;
      }
      else if( (int) (m_scale * 1000 + 0.5) >= 10 )
      {
         m_scale = 0.005;
      }
      else
      {
         m_scale = 0.001;
      }
      
      char sstr[16];
      snprintf(sstr, sizeof(sstr), "%0.3f", m_scale);
      ui.button_scale->setText(sstr);
   }
}

void gainCtrl::on_button_minus_pressed()
{
   float ng = m_current - m_scale;
   if(ng < 0) ng = 0;
   setGain(ng);
}

void gainCtrl::on_button_zero_pressed()
{
   if(m_ctrlType == GAIN) setGain(0.0);
   if(m_ctrlType == MULTCOEFF) setGain(1.0);
}

} //namespace xqt
   
#include "moc_gainCtrl.cpp"

#endif
