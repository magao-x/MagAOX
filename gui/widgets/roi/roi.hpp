
#ifndef roi_hpp
#define roi_hpp

#include <QDialog>

#include "ui_roi.h"

#include "../../lib/multiIndi.hpp"

namespace xqt 
{
   
class roi : public QDialog, public multiIndiSubscriber
{
   Q_OBJECT
   
protected:
   
   std::string m_appState;
   
   std::string m_camName;
   std::string m_winTitle;

   int m_bin_x_curr {0};
   int m_bin_x_tgt {0};
   
   int m_bin_y_curr {0};
   int m_bin_y_tgt {0};

   float m_cen_x_curr {0};
   float m_cen_x_tgt {0};

   float m_cen_y_curr {0};
   float m_cen_y_tgt {0};

   int m_wid_curr {0};
   int m_wid_tgt {0};

   int m_hgt_curr {0};
   int m_hgt_tgt {0};

public:
   roi( std::string & camName,
        QWidget * Parent = 0, 
        Qt::WindowFlags f = 0
      );
   
   ~roi();
   
   int subscribe( multiIndiPublisher * publisher );
             
   virtual void onConnect();
   virtual void onDisconnect();
   
   int handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   int handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void clear_focus();

public slots:
   void updateGUI();
   
   void on_le_bin_x_editingFinished();
   void on_le_bin_y_editingFinished();

   void on_le_center_x_editingFinished();
   void on_le_center_y_editingFinished();

   void on_le_width_editingFinished();
   void on_le_height_editingFinished();

   void on_button_check_pressed();
   void on_button_set_pressed();
   void on_button_last_pressed();
   void on_button_full_pressed();
   void on_button_default_pressed();
      
private:
     
   Ui::roi ui;
};
   
roi::roi( std::string & camName,
          QWidget * Parent, 
          Qt::WindowFlags f) : QDialog(Parent, f), m_camName{camName}
{
   ui.setupUi(this);

   m_winTitle = m_camName;
   m_winTitle += " ROI";
   ui.lab_title->setText(m_winTitle.c_str());
   
   ui.val_bin_x->setProperty("isStatus", true);
   ui.val_bin_y->setProperty("isStatus", true);
   ui.val_center_x->setProperty("isStatus", true);
   ui.val_center_y->setProperty("isStatus", true);
   ui.val_width->setProperty("isStatus", true);
   ui.val_height->setProperty("isStatus", true);

   onDisconnect();
}
   
roi::~roi()
{
}

int roi::subscribe( multiIndiPublisher * publisher )
{
   if(!publisher) return -1;
   
   publisher->subscribeProperty(this, m_camName, "fsm");
   publisher->subscribeProperty(this, m_camName, "roi_region_bin_x");
   publisher->subscribeProperty(this, m_camName, "roi_region_bin_y");
   publisher->subscribeProperty(this, m_camName, "roi_region_x");
   publisher->subscribeProperty(this, m_camName, "roi_region_y");
   publisher->subscribeProperty(this, m_camName, "roi_region_w");
   publisher->subscribeProperty(this, m_camName, "roi_region_h");

   return 0;
}
  
void roi::onConnect()
{
   ui.lab_title->setEnabled(true);

   setWindowTitle(QString(m_winTitle.c_str()));

   clearFocus();
}

void roi::onDisconnect()
{
   ui.lab_title->setEnabled(false);
   
   ui.lab_bin_x_title->setEnabled(false);
   ui.lab_bin_x_curr->setEnabled(false);
   ui.lab_bin_x_tgt->setEnabled(false);
   ui.val_bin_x->setText("");
   ui.val_bin_x->setEnabled(false);
   ui.le_bin_x->setText("");
   ui.le_bin_x->setEnabled(false);

   ui.lab_bin_y_title->setEnabled(false);
   ui.lab_bin_y_curr->setEnabled(false);
   ui.lab_bin_y_tgt->setEnabled(false);
   ui.val_bin_y->setText("");
   ui.val_bin_y->setEnabled(false);
   ui.le_bin_y->setText("");
   ui.le_bin_y->setEnabled(false);
   
   ui.lab_center_x_title->setEnabled(false);
   ui.lab_center_x_curr->setEnabled(false);
   ui.lab_center_x_tgt->setEnabled(false);
   ui.val_center_x->setText("");
   ui.val_center_x->setEnabled(false);
   ui.le_center_x->setText("");
   ui.le_center_x->setEnabled(false);

   ui.lab_center_y_title->setEnabled(false);
   ui.lab_center_y_curr->setEnabled(false);
   ui.lab_center_y_tgt->setEnabled(false);
   ui.val_center_y->setText("");
   ui.val_center_y->setEnabled(false);
   ui.le_center_y->setText("");
   ui.le_center_y->setEnabled(false);

   ui.lab_width_title->setEnabled(false);
   ui.lab_width_curr->setEnabled(false);
   ui.lab_width_tgt->setEnabled(false);
   ui.val_width->setText("");
   ui.val_width->setEnabled(false);
   ui.le_width->setText("");
   ui.le_width->setEnabled(false);

   ui.lab_height_title->setEnabled(false);
   ui.lab_height_curr->setEnabled(false);
   ui.lab_height_tgt->setEnabled(false);
   ui.val_height->setText("");
   ui.val_height->setEnabled(false);
   ui.le_height->setText("");
   ui.le_height->setEnabled(false);

   ui.button_check->setEnabled(false);
   ui.button_set->setEnabled(false);
   ui.button_last->setEnabled(false);
   ui.button_full->setEnabled(false);
   ui.button_default->setEnabled(false);

   setWindowTitle(QString(m_winTitle.c_str()) + QString(" (disconnected)"));
}

int roi::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   return handleSetProperty(ipRecv);
   
   return 0;
}

int roi::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_camName) return 0;
   
   if(ipRecv.getName() == "fsm")
   {
      if(ipRecv.find("state"))
      {
         m_appState = ipRecv["state"].get<std::string>();
      }
   }
   else if(ipRecv.getName() == "roi_region_bin_x")
   {
      if(ipRecv.find("current"))
      {
         m_bin_x_curr = ipRecv["current"].get<int>();
      }

      if(ipRecv.find("target"))
      {
         m_bin_x_tgt = ipRecv["target"].get<int>();
      }
   }
   else if(ipRecv.getName() == "roi_region_bin_y")
   {
      if(ipRecv.find("current"))
      {
         m_bin_y_curr = ipRecv["current"].get<int>();
      }

      if(ipRecv.find("target"))
      {
         m_bin_y_tgt = ipRecv["target"].get<int>();
      }
   }
   else if(ipRecv.getName() == "roi_region_x")
   {
      if(ipRecv.find("current"))
      {
         m_cen_x_curr = ipRecv["current"].get<float>();
      }

      if(ipRecv.find("target"))
      {
         m_cen_x_tgt = ipRecv["target"].get<float>();
      }
   }
   else if(ipRecv.getName() == "roi_region_y")
   {
      if(ipRecv.find("current"))
      {
         m_cen_y_curr = ipRecv["current"].get<float>();
      }

      if(ipRecv.find("target"))
      {
         m_cen_y_tgt = ipRecv["target"].get<float>();
      }
   }
   else if(ipRecv.getName() == "roi_region_w")
   {
      if(ipRecv.find("current"))
      {
         m_wid_curr = ipRecv["current"].get<int>();
      }

      if(ipRecv.find("target"))
      {
         m_wid_tgt = ipRecv["target"].get<int>();
      }
   }
   else if(ipRecv.getName() == "roi_region_h")
   {
      if(ipRecv.find("current"))
      {
         m_hgt_curr = ipRecv["current"].get<int>();
      }

      if(ipRecv.find("target"))
      {
         m_hgt_tgt = ipRecv["target"].get<int>();
      }
   }

   updateGUI();
   
   //If we get here then we need to add this device
   return 0;
   
}

void roi::updateGUI()
{
   if( m_appState != "READY" && m_appState != "OPERATING" )
   {
      //Disable & zero all
      ui.lab_bin_x_title->setEnabled(false);
      ui.lab_bin_x_curr->setEnabled(false);
      ui.lab_bin_x_tgt->setEnabled(false);
      ui.val_bin_x->setText("");
      ui.val_bin_x->setEnabled(false);
      ui.le_bin_x->setText("");
      ui.le_bin_x->setEnabled(false);
   
      ui.lab_bin_y_title->setEnabled(false);
      ui.lab_bin_y_curr->setEnabled(false);
      ui.lab_bin_y_tgt->setEnabled(false);
      ui.val_bin_y->setText("");
      ui.val_bin_y->setEnabled(false);
      ui.le_bin_y->setText("");
      ui.le_bin_y->setEnabled(false);
      
      ui.lab_center_x_title->setEnabled(false);
      ui.lab_center_x_curr->setEnabled(false);
      ui.lab_center_x_tgt->setEnabled(false);
      ui.val_center_x->setText("");
      ui.val_center_x->setEnabled(false);
      ui.le_center_x->setText("");
      ui.le_center_x->setEnabled(false);
   
      ui.lab_center_y_title->setEnabled(false);
      ui.lab_center_y_curr->setEnabled(false);
      ui.lab_center_y_tgt->setEnabled(false);
      ui.val_center_y->setText("");
      ui.val_center_y->setEnabled(false);
      ui.le_center_y->setText("");
      ui.le_center_y->setEnabled(false);
   
      ui.lab_width_title->setEnabled(false);
      ui.lab_width_curr->setEnabled(false);
      ui.lab_width_tgt->setEnabled(false);
      ui.val_width->setText("");
      ui.val_width->setEnabled(false);
      ui.le_width->setText("");
      ui.le_width->setEnabled(false);
   
      ui.lab_height_title->setEnabled(false);
      ui.lab_height_curr->setEnabled(false);
      ui.lab_height_tgt->setEnabled(false);
      ui.val_height->setText("");
      ui.val_height->setEnabled(false);
      ui.le_height->setText("");
      ui.le_height->setEnabled(false);
   
      ui.button_check->setEnabled(false);
      ui.button_set->setEnabled(false);
      ui.button_last->setEnabled(false);
      ui.button_full->setEnabled(false);
      ui.button_default->setEnabled(false);
      
      return;
   }
   
   
   if( m_appState == "READY" || m_appState == "OPERATING" )
   {
      //Enable values, but not buttons   
      //Update values  
      ui.lab_bin_x_title->setEnabled(true);
      ui.lab_bin_x_curr->setEnabled(true);
      ui.lab_bin_x_tgt->setEnabled(true);
      ui.val_bin_x->setEnabled(true);
      ui.le_bin_x->setEnabled(true);

      ui.lab_bin_y_title->setEnabled(true);
      ui.lab_bin_y_curr->setEnabled(true);
      ui.lab_bin_y_tgt->setEnabled(true);
      ui.val_bin_y->setEnabled(true);
      ui.le_bin_y->setEnabled(true);
   
      ui.lab_center_x_title->setEnabled(true);
      ui.lab_center_x_curr->setEnabled(true);
      ui.lab_center_x_tgt->setEnabled(true);
      ui.val_center_x->setEnabled(true);
      ui.le_center_x->setEnabled(true);

      ui.lab_center_y_title->setEnabled(true);
      ui.lab_center_y_curr->setEnabled(true);
      ui.lab_center_y_tgt->setEnabled(true);
      ui.val_center_y->setEnabled(true);
      ui.le_center_y->setEnabled(true);

      ui.lab_width_title->setEnabled(true);
      ui.lab_width_curr->setEnabled(true);
      ui.lab_width_tgt->setEnabled(true);
      ui.val_width->setEnabled(true);
      ui.le_width->setEnabled(true);

      ui.lab_height_title->setEnabled(true);
      ui.lab_height_curr->setEnabled(true);
      ui.lab_height_tgt->setEnabled(true);
      ui.val_height->setEnabled(true);
      ui.le_height->setEnabled(true);
      
      ui.val_bin_x->setText(QString::number(m_bin_x_curr));
      if(!ui.le_bin_x->hasFocus()) ui.le_bin_x->setText(QString::number(m_bin_x_tgt));

      ui.val_bin_y->setText(QString::number(m_bin_y_curr));
      if(!ui.le_bin_y->hasFocus()) ui.le_bin_y->setText(QString::number(m_bin_y_tgt));

      ui.val_center_x->setText(QString::number(m_cen_x_curr));
      if(!ui.le_center_x->hasFocus()) ui.le_center_x->setText(QString::number(m_cen_x_tgt));

      ui.val_center_y->setText(QString::number(m_cen_y_curr));
      if(!ui.le_center_y->hasFocus()) ui.le_center_y->setText(QString::number(m_cen_y_tgt));

      ui.val_width->setText(QString::number(m_wid_curr));
      if(!ui.le_width->hasFocus()) ui.le_width->setText(QString::number(m_wid_tgt));

      ui.val_height->setText(QString::number(m_hgt_curr));
      if(!ui.le_height->hasFocus()) ui.le_height->setText(QString::number(m_hgt_tgt));
   }
   
   if( m_appState == "OPERATING" )
   {
      //Enable buttons too   
      
      ui.button_check->setEnabled(true);
      ui.button_set->setEnabled(true);
      ui.button_last->setEnabled(true);
      ui.button_full->setEnabled(true);
      ui.button_default->setEnabled(true);

      return;
   }

} //updateGUI()

void roi::clear_focus()
{
   ui.le_bin_x->clearFocus();
   ui.le_bin_y->clearFocus();
   ui.le_center_x->clearFocus();
   ui.le_center_y->clearFocus(); 
   ui.le_width->clearFocus();
   ui.le_height->clearFocus();
}

void roi::on_le_bin_x_editingFinished()
{
   clear_focus();

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_camName);
   ip.setName("roi_region_bin_x");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.le_bin_x->text().toDouble();
   
   sendNewProperty(ip);
}

void roi::on_le_bin_y_editingFinished()
{
   clear_focus();

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_camName);
   ip.setName("roi_region_bin_y");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.le_bin_y->text().toDouble();
   
   sendNewProperty(ip);
}

void roi::on_le_center_x_editingFinished()
{
   clear_focus();

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_camName);
   ip.setName("roi_region_x");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.le_center_x->text().toDouble();
   
   sendNewProperty(ip);
}

void roi::on_le_center_y_editingFinished()
{
   clear_focus();

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_camName);
   ip.setName("roi_region_y");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.le_center_y->text().toDouble();
   
   sendNewProperty(ip);
}

void roi::on_le_width_editingFinished()
{
   clear_focus();

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_camName);
   ip.setName("roi_region_w");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.le_width->text().toDouble();
   
   sendNewProperty(ip);
}

void roi::on_le_height_editingFinished()
{
   clear_focus();

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_camName);
   ip.setName("roi_region_h");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.le_height->text().toDouble();
   
   sendNewProperty(ip);
}

void roi::on_button_check_pressed()
{
   clear_focus();

   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_camName);
   ipFreq.setName("roi_region_check");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq);  
}

void roi::on_button_set_pressed()
{
   clear_focus();

   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_camName);
   ipFreq.setName("roi_set");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq); 
}

void roi::on_button_last_pressed()
{
   clear_focus();

   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_camName);
   ipFreq.setName("roi_set_last");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq); 
}

void roi::on_button_full_pressed()
{
   clear_focus();

   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_camName);
   ipFreq.setName("roi_set_full");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq); 
}

void roi::on_button_default_pressed()
{
   clear_focus();

   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_camName);
   ipFreq.setName("roi_set_startup");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq); 
}


} //namespace xqt
   
#include "moc_roi.cpp"

#endif
