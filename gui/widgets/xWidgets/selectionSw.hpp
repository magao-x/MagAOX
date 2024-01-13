#ifndef selectionSw_hpp
#define selectionSw_hpp

#include "xWidget.hpp"
#include "statusLabel.hpp"

#include "ui_selectionSw.h"

namespace xqt 
{
   
class selectionSw : public xWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_device;
   std::string m_property;

   std::string m_title;

   std::string m_value;

   bool m_valChanged {false};

   bool m_onDisconnected {false};

public:

   selectionSw( const std::string & device,
                const std::string & property,
                QWidget * Parent = 0, 
                Qt::WindowFlags f = Qt::WindowFlags()
              );
   
   ~selectionSw();
   
   void device( const std::string & dev);

   virtual void subscribe();
             
   virtual void onConnect();

   virtual void onDisconnect();
   
   virtual void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   virtual void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   virtual void updateGUI();

public slots:

   void on_button_set_pressed();

private:
     
   Ui::selectionSw ui;
};
   
selectionSw::selectionSw( const std::string & device,
                          const std::string & property,
                          QWidget * Parent, 
                          Qt::WindowFlags f) : xWidget(Parent, f), m_device{device}, m_property{property}
{
   m_title = m_device + "." + m_property;
   ui.setupUi(this);

   ui.device->setText(m_device.c_str());
   ui.property->setText(m_property.c_str());

   onDisconnect();
}
   
void selectionSw::device( const std::string & dev)
{
   m_device = dev;
}

selectionSw::~selectionSw()
{
}

void selectionSw::subscribe()
{
   if(!m_parent) return;
   
   if(m_property != "") m_parent->addSubscriberProperty(this, m_device, m_property);

   return;
}
  
void selectionSw::onConnect()
{
   setWindowTitle(QString(m_title.c_str()));

   m_valChanged = true;
}

void selectionSw::onDisconnect()
{
   setWindowTitle(QString(m_title.c_str()) + QString(" (disconnected)"));

   ui.current->setText("---");

   m_onDisconnected = true;
}

void selectionSw::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void selectionSw::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_device) return;
   
   if(ipRecv.getName() == m_property)
   {
      std::string newName;
      for(auto it = ipRecv.getElements().begin(); it != ipRecv.getElements().end(); ++it)
      {
         if(ui.comboBox->findText(QString(it->second.name().c_str())) == -1)
         {
            ui.comboBox->addItem(QString(it->second.name().c_str()));
         }

         if(it->second.getSwitchState() == pcf::IndiElement::On)
         {
            if(newName != "")
            {
               std::cerr << "More than one switch selected in " << ipRecv.getDevice() << "." << ipRecv.getName() << "\n";
            }
         
            newName = it->second.name();
            if(newName != m_value) m_valChanged = true;
            m_value = newName; 
         }
      }

      if(m_onDisconnected) 
      {
         setWindowTitle(QString(m_title.c_str()));
         m_valChanged = true;
         m_onDisconnected = false;
      }
   }

   updateGUI();
}

void selectionSw::updateGUI()
{
   if(isEnabled())
   {
      if(m_valChanged)
      {
         QString value(m_value.c_str());
         ui.current->setTextChanged(value);  
         m_valChanged = false;
      }
   }


} //updateGUI()

void selectionSw::on_button_set_pressed()
{
   std::string selection = ui.comboBox->currentText().toStdString();

   if(selection == "")
   {
      std::cerr << "set: " << selection << "\n";
      return;
   }

   try
   {
      pcf::IndiProperty ipSend(pcf::IndiProperty::Switch);
      ipSend.setDevice(m_device);
      ipSend.setName(m_property);
      ipSend.setPerm(pcf::IndiProperty::ReadWrite); 
      ipSend.setState(pcf::IndiProperty::Idle);
      ipSend.setRule(pcf::IndiProperty::OneOfMany);
   
      for(int idx = 0; idx < ui.comboBox->count(); ++idx)
      {
         std::string elName = ui.comboBox->itemText(idx).toStdString();
   
         if(elName == selection)
         {
            ipSend.add(pcf::IndiElement(elName, pcf::IndiElement::On));
         }
         else
         {
            ipSend.add(pcf::IndiElement(elName, pcf::IndiElement::Off));
         }
      }
   
      sendNewProperty(ipSend);
   }
   catch(...)
   {
      std::cerr << "exception thrown in selectionSw::on_button_set_pressed\n";
   }

   ui.comboBox->setCurrentIndex(-1);
   
}

} //namespace xqt
   
#include "moc_selectionSw.cpp"

#endif
