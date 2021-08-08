#ifndef statusEntry_hpp
#define statusEntry_hpp

#include <QWidget>
#include <QTimer>

#include "statusLineEdit.hpp"

#include "ui_statusEntry.h"

#include "../../lib/multiIndi.hpp"

namespace xqt 
{
   
class statusEntry : public QWidget, public multiIndiSubscriber
{
   Q_OBJECT

public:
   enum types{ STRING, INT, FLOAT };

protected:
   
   std::string m_device;
   std::string m_property;
   std::string m_label;
   std::string m_units;
   std::string m_currEl {"current"};
   std::string m_targEl {"target"};

   bool m_readOnly {false};
   bool m_highlightChanges {true};

   std::string m_format;


   int m_type {FLOAT};

   bool m_valChanged {false};

   std::string m_current;
   std::string m_target;
   

   
public:
   statusEntry( QWidget * Parent = 0, 
                Qt::WindowFlags f = 0
              );

   statusEntry( const std::string & device,
                const std::string & property,
                int type,
                const std::string & label,
                const std::string & units,
                QWidget * Parent = 0, 
                Qt::WindowFlags f = 0
              );
   
   ~statusEntry() noexcept;
   
   void construct();

   void setup( const std::string & device,
               const std::string & property,
               int type,
               const std::string & label,
               const std::string & units
             );

   void defaultFormat();

   QString formattedValue();

   /// Set the read only flag 
   /** If set to true, the widget will not accept focus and will not enter edit mode.
     * The default is false.
     * 
     * This sets m_readOnly
     */
   void readOnly(bool ro /**< [in] the new value of the read only flag*/);

   /// Get the value of the read only flag
   /**
     * \returns the current value of m_readOnly
     */ 
   bool readOnly();

   /// Set the highlight changes flag
   /** If set to false, the widget will not changed to statusChanged.
     * The default is true.
     * 
     * This sets m_highlightChanges
     */
   void highlightChanges(bool hc);

   /// Get the value of the highlight changes flag
   /**
     * \returns the current value of m_readOnly
     */
   bool highlightChanges();


   virtual void subscribe();
             
   virtual void onConnect();
   virtual void onDisconnect();
   
   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
protected:
   virtual void clearFocus();

public slots:
   void updateGUI();

   void on_value_returnPressed();

private:
     
   Ui::statusEntry ui;
};
   
statusEntry::statusEntry( QWidget * Parent, 
                          Qt::WindowFlags f) : QWidget(Parent, f)
{
   construct();
}

statusEntry::statusEntry( const std::string & device,
                          const std::string & property,
                          int type,
                          const std::string & label,
                          const std::string & units,
                          QWidget * Parent, 
                          Qt::WindowFlags f) : QWidget(Parent, f)
{
   construct();
   setup(device, property, type, label, units);
}

statusEntry::~statusEntry() noexcept
{
}

void statusEntry::construct()
{
   ui.setupUi(this);
   ui.value->setProperty("isStatus", true);
}

void statusEntry::setup( const std::string & device,
                         const std::string & property,
                         int type,
                         const std::string & label,
                         const std::string & units
                       )
{
   m_device = device;
   m_property = property;
   
   m_type = type;
   defaultFormat();

   m_label = label;
   m_units = units;

   std::string lab = m_label;
   if(m_units != "") lab += " [" + m_units + "]";
   ui.label->setText(lab.c_str());

   onDisconnect();
}

void statusEntry::defaultFormat()
{
   switch(m_type)
   {
      case STRING:
         m_format = "";
         break;
      case INT:
         m_format = "%d";
         break;
      case FLOAT:
         m_format = "auto";
         break;
      default:
         m_format = "%0.2f";
         break;
   }
}

QString floatAutoFormat(const std::string & value)
{
   double v = std::stod(value);
   std::string format = "%f";

   if(v >= 1000)
   {
      format = "%0.3g";
   }
   else if(v >= 0.1)
   {
      format = "%0.2f";
   }
   else if(v >= 0.01)
   {
      format = "%0.3f";
   }
   else if(v >= 0.001)
   {
      format = "%0.4f";
   }
   else 
   {
      format = "%0.3g";
   }

   char str[16];
   snprintf(str, sizeof(str), format.c_str(), v);
   return QString(str);
}

QString statusEntry::formattedValue()
{
   char str[64];
   switch(m_type)
   {
      case STRING:
         return QString(m_current.c_str());
      case INT:
         snprintf(str, sizeof(str), m_format.c_str(), std::stoi(m_current));
         return QString(str);
      case FLOAT:
         if(m_format == "auto") return floatAutoFormat(m_current);

         snprintf(str, sizeof(str), m_format.c_str(), std::stof(m_current));
         return QString(str);
      default:
         return QString(m_current.c_str());
   }
}

void statusEntry::readOnly(bool ro)
{
   m_readOnly = ro;
   ui.value->readOnly(ro);
}

bool statusEntry::readOnly()
{
   return ui.value->readOnly();
}

void statusEntry::highlightChanges(bool hc)
{
   m_highlightChanges = hc;
   ui.value->highlightChanges(hc);
}

bool statusEntry::highlightChanges()
{
   return ui.value->highlightChanges();
}

void statusEntry::subscribe()
{
   if(!m_parent) return ;
   
   if(m_property != "") m_parent->addSubscriberProperty(this, m_device, m_property);

   return;
}
  
void statusEntry::onConnect()
{
}

void statusEntry::onDisconnect()
{
   ui.value->setText("---");
}

void statusEntry::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void statusEntry::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_device) return;
   
   if(ipRecv.getName() == m_property)
   {
      if(ipRecv.find(m_currEl))
      {
         std::string current = ipRecv[m_currEl].get();
         if(current != m_current)  
         {
            m_valChanged = true;
         }
         m_current = current;
      }

      if(ipRecv.find(m_targEl))
      {
         m_target = ipRecv[m_targEl].get();
      }
   }

   updateGUI();
}

void statusEntry::clearFocus()
{
   ui.value->clearFocus();
}

void statusEntry::on_value_returnPressed()
{
   std::string str = ui.value->text().toStdString();
   ui.value->clearFocus();

   pcf::IndiProperty ip(pcf::IndiProperty::Text);
   
   ip.setDevice(m_device);
   ip.setName(m_property);
   ip.add(pcf::IndiElement(m_targEl));
   ip[m_targEl] = str;
   
   sendNewProperty(ip);

}

void statusEntry::updateGUI()
{
   if(isEnabled())
   {
      if(m_valChanged)
      {
          ui.value->setTextChanged(formattedValue());
          m_valChanged = false;
      }
   }
   else
   {
      ui.value->setText("---");
   }

} //updateGUI()


} //namespace xqt
   
#include "moc_statusEntry.cpp"

#endif
