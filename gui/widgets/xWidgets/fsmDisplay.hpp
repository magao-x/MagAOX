#ifndef fsmDisplay_hpp
#define fsmDisplay_hpp

#include "xWidget.hpp"
#include "statusLabel.hpp"

#include "ui_fsmDisplay.h"

namespace xqt 
{
   
class fsmDisplay : public xWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_device;
   std::string m_property {"fsm"};
   std::string m_element {"state"};

   bool m_highlightChanges {true};

   bool m_valChanged {false};

   std::string m_value;

public:
   fsmDisplay( QWidget * Parent = 0, 
               Qt::WindowFlags f = Qt::WindowFlags()
             );

   fsmDisplay( const std::string & device,
               QWidget * Parent = 0, 
               Qt::WindowFlags f = Qt::WindowFlags()
             );
   
   ~fsmDisplay();
   
   void device( const std::string & dev);

   virtual void subscribe();
             
   virtual void onConnect();

   virtual void onDisconnect();
   
   virtual void clearFocus();

   virtual void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   virtual void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
public slots:

   void updateGUI();

signals:
   
   void doUpdateGUI();

private:
     
   Ui::fsmDisplay ui;
};
   
fsmDisplay::fsmDisplay( QWidget * Parent, 
                        Qt::WindowFlags f) : xWidget(Parent, f)
{
   ui.setupUi(this);
   
   onDisconnect();
}

fsmDisplay::fsmDisplay( const std::string & device,
                        QWidget * Parent, 
                        Qt::WindowFlags f) : xWidget(Parent, f), m_device{device}
{
   ui.setupUi(this);
   onDisconnect();
}
   
void fsmDisplay::device( const std::string & dev)
{
   m_device = dev;
}

fsmDisplay::~fsmDisplay()
{
}

void fsmDisplay::subscribe()
{
   if(!m_parent) return;
   
   if(m_property != "") m_parent->addSubscriberProperty(this, m_device, m_property);

   return;
}
  
void fsmDisplay::onConnect()
{
   m_valChanged = true;
}

void fsmDisplay::onDisconnect()
{
   ui.fsm->setText("---");
}

void fsmDisplay::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void fsmDisplay::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_device) return;
   
   if(ipRecv.getName() == m_property)
   {
      if(ipRecv.find(m_element))
      {
         std::string value = ipRecv[m_element].get();
         if(value != m_value) m_valChanged = true;
         m_value = value;
      }
   }

   updateGUI();
}

void fsmDisplay::updateGUI()
{
   if(isEnabled())
   {
      if(m_valChanged)
      {
         QString value(m_value.c_str()); //in future provide translatiosn for "RIP" "MODULATING", etc.
         ui.fsm->setTextChanged(value);  
         m_valChanged = false;
      }
   }
   else
   {
      QString value(m_value.c_str()); //in future provide translatiosn for "RIP" "MODULATING", etc.
      ui.fsm->setText(value);  
   }

} //updateGUI()

void fsmDisplay::clearFocus()
{
}

} //namespace xqt
   
#include "moc_fsmDisplay.cpp"

#endif
