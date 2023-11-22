
#ifndef roi_hpp
#define roi_hpp

#include "ui_roi.h"

#include "../xWidgets/xWidget.hpp"

namespace xqt 
{
   
class roi : public xWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_appState;
   
   std::string m_camName;
   std::string m_winTitle;

   bool m_bin_x_curr_changed {false};
   int m_bin_x_curr {0};
   bool m_bin_x_tgt_changed {false};
   int m_bin_x_tgt {0};
   
   bool m_bin_y_curr_changed {false};
   int m_bin_y_curr {0};
   bool m_bin_y_tgt_changed {false};
   int m_bin_y_tgt {0};

   bool m_cen_x_curr_changed {false};
   float m_cen_x_curr {0};
   bool m_cen_x_tgt_changed {false};
   float m_cen_x_tgt {0};

   bool m_cen_y_curr_changed {false};
   float m_cen_y_curr {0};
   bool m_cen_y_tgt_changed {false};
   float m_cen_y_tgt {0};

   bool m_wid_curr_changed {false};
   int m_wid_curr {0};
   bool m_wid_tgt_changed {false};
   int m_wid_tgt {0};

   bool m_hgt_curr_changed {false};
   int m_hgt_curr {0};
   bool m_hgt_tgt_changed {false};
   int m_hgt_tgt {0};

   bool m_onDisconnected {false};

public:
   explicit roi( std::string & camName,
                 QWidget * Parent = 0, 
                 Qt::WindowFlags f = Qt::WindowFlags()
               );
   
   ~roi();
   
   void subscribe();
             
   virtual void onConnect();
   virtual void onDisconnect();
   
   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void clear_focus();

public slots:
   void updateGUI();
   
   void on_le_bin_x_editingFinished();
   void on_le_bin_y_editingFinished();

   void on_le_center_x_editingFinished();
   void on_le_center_y_editingFinished();

   void on_le_width_editingFinished();
   void on_le_height_editingFinished();

   void on_button_loadlast_pressed();
   void on_button_reset_pressed();
   void on_button_check_pressed();
   void on_button_set_pressed();
   void on_button_last_pressed();
   void on_button_fullbin_pressed();
   void on_button_full_pressed();
   void on_button_default_pressed();
      
   

private:
     
   Ui::roi ui;
};
   
roi::roi( std::string & camName,
          QWidget * Parent, 
          Qt::WindowFlags f) : xWidget(Parent, f), m_camName{camName}
{
   ui.setupUi(this);

   m_winTitle = m_camName;
   m_winTitle += " ROI";
   //ui.lab_title->setText(m_winTitle.c_str());
   
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

void roi::subscribe()
{
   if(!m_parent) return;
   
   m_parent->addSubscriberProperty(this, m_camName, "fsm");
   m_parent->addSubscriberProperty(this, m_camName, "roi_region_bin_x");
   m_parent->addSubscriberProperty(this, m_camName, "roi_region_bin_y");
   m_parent->addSubscriberProperty(this, m_camName, "roi_region_x");
   m_parent->addSubscriberProperty(this, m_camName, "roi_region_y");
   m_parent->addSubscriberProperty(this, m_camName, "roi_region_w");
   m_parent->addSubscriberProperty(this, m_camName, "roi_region_h");

   return;
}
  
void roi::onConnect()
{
   setWindowTitle(QString(m_winTitle.c_str()));

   //Reset the changed flags
   m_bin_x_curr_changed = true;
   m_bin_x_tgt_changed = true;
   m_bin_y_curr_changed = true;
   m_bin_y_tgt_changed = true;
   m_cen_x_curr_changed = true;
   m_cen_x_tgt_changed = true;
   m_cen_y_curr_changed = true;
   m_cen_y_tgt_changed = true;
   m_wid_curr_changed = true;
   m_wid_tgt_changed = true;
   m_hgt_curr_changed = true;
   m_hgt_tgt_changed = true;

   m_onDisconnected = false;

   clearFocus();
}

void roi::onDisconnect()
{
   setWindowTitle(QString(m_winTitle.c_str()) + QString(" (disconnected)"));

   ui.lab_bin_x_title->setEnabled(false);
   ui.lab_bin_x_curr->setEnabled(false);
   ui.lab_bin_x_tgt->setEnabled(false);
   ui.val_bin_x->setText("---");
   ui.val_bin_x->setEnabled(false);
   ui.le_bin_x->setText("---");
   ui.le_bin_x->setEnabled(false);

   ui.lab_bin_y_title->setEnabled(false);
   ui.lab_bin_y_curr->setEnabled(false);
   ui.lab_bin_y_tgt->setEnabled(false);
   ui.val_bin_y->setText("---");
   ui.val_bin_y->setEnabled(false);
   ui.le_bin_y->setText("---");
   ui.le_bin_y->setEnabled(false);
   
   ui.lab_center_x_title->setEnabled(false);
   ui.lab_center_x_curr->setEnabled(false);
   ui.lab_center_x_tgt->setEnabled(false);
   ui.val_center_x->setText("---");
   ui.val_center_x->setEnabled(false);
   ui.le_center_x->setText("---");
   ui.le_center_x->setEnabled(false);

   ui.lab_center_y_title->setEnabled(false);
   ui.lab_center_y_curr->setEnabled(false);
   ui.lab_center_y_tgt->setEnabled(false);
   ui.val_center_y->setText("---");
   ui.val_center_y->setEnabled(false);
   ui.le_center_y->setText("---");
   ui.le_center_y->setEnabled(false);

   ui.lab_width_title->setEnabled(false);
   ui.lab_width_curr->setEnabled(false);
   ui.lab_width_tgt->setEnabled(false);
   ui.val_width->setText("---");
   ui.val_width->setEnabled(false);
   ui.le_width->setText("---");
   ui.le_width->setEnabled(false);

   ui.lab_height_title->setEnabled(false);
   ui.lab_height_curr->setEnabled(false);
   ui.lab_height_tgt->setEnabled(false);
   ui.val_height->setText("---");
   ui.val_height->setEnabled(false);
   ui.le_height->setText("---");
   ui.le_height->setEnabled(false);

   ui.button_reset->setEnabled(false);
   ui.button_loadlast->setEnabled(false);
   ui.button_check->setEnabled(false);
   ui.button_set->setEnabled(false);
   ui.button_last->setEnabled(false);
   ui.button_fullbin->setEnabled(false);
   ui.button_full->setEnabled(false);
   ui.button_default->setEnabled(false);

   m_onDisconnected = true;
}

void roi::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   return handleSetProperty(ipRecv);
}

void roi::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_camName) return;
   
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
         int val = ipRecv["current"].get<int>();
         if(val != m_bin_x_curr) m_bin_x_curr_changed = true;
         m_bin_x_curr = val;
      }

      if(ipRecv.find("target"))
      {
         int val = ipRecv["target"].get<int>();
         if(val != m_bin_x_tgt) m_bin_x_tgt_changed = true;
         m_bin_x_tgt = val;
      }
   }
   else if(ipRecv.getName() == "roi_region_bin_y")
   {
      if(ipRecv.find("current"))
      {
         int val = ipRecv["current"].get<int>();
         if(val != m_bin_y_curr) m_bin_y_curr_changed = true;
         m_bin_y_curr = val;
      }

      if(ipRecv.find("target"))
      {
         int val = ipRecv["target"].get<int>();
         if(val != m_bin_y_tgt) m_bin_y_tgt_changed = true;
         m_bin_y_tgt = val;
      }
   }
   else if(ipRecv.getName() == "roi_region_x")
   {
      if(ipRecv.find("current"))
      {
         float val = ipRecv["current"].get<float>();;
         if(val != m_cen_x_curr) m_cen_x_curr_changed = true;
         m_cen_x_curr = val;
      }

      if(ipRecv.find("target"))
      {
         float val = ipRecv["target"].get<float>();;
         if(val != m_cen_x_tgt) m_cen_x_tgt_changed = true;
         m_cen_x_tgt = val;
      }
   }
   else if(ipRecv.getName() == "roi_region_y")
   {
      if(ipRecv.find("current"))
      {
         float val = ipRecv["current"].get<float>();;
         if(val != m_cen_y_curr) m_cen_y_curr_changed = true;
         m_cen_y_curr = val;
      }

      if(ipRecv.find("target"))
      {
         float val = ipRecv["target"].get<float>();;
         if(val != m_cen_y_tgt) m_cen_y_tgt_changed = true;
         m_cen_y_tgt = val;
      }
   }
   else if(ipRecv.getName() == "roi_region_w")
   {
      if(ipRecv.find("current"))
      {
         int val = ipRecv["current"].get<int>();
         if(val != m_wid_curr) m_wid_curr_changed = true;
         m_wid_curr = val;
      }

      if(ipRecv.find("target"))
      {
         int val = ipRecv["target"].get<int>();
         if(val != m_wid_tgt) m_wid_tgt_changed = true;
         m_wid_tgt = val;
      }
   }
   else if(ipRecv.getName() == "roi_region_h")
   {
      if(ipRecv.find("current"))
      {
         int val = ipRecv["current"].get<int>();
         if(val != m_hgt_curr) m_hgt_curr_changed = true;
         m_hgt_curr = val;
      }

      if(ipRecv.find("target"))
      {
         int val = ipRecv["target"].get<int>();
         if(val != m_hgt_tgt) m_hgt_tgt_changed = true;
         m_hgt_tgt = val;
      }
   }

   if( m_onDisconnected && isEnabled() ) 
   {
      setWindowTitle(QString(m_winTitle.c_str()));

      //Reset the changed flags
      m_bin_x_curr_changed = true;
      m_bin_x_tgt_changed = true;
      m_bin_y_curr_changed = true;
      m_bin_y_tgt_changed = true;
      m_cen_x_curr_changed = true;
      m_cen_x_tgt_changed = true;
      m_cen_y_curr_changed = true;
      m_cen_y_tgt_changed = true;
      m_wid_curr_changed = true;
      m_wid_tgt_changed = true;
      m_hgt_curr_changed = true;
      m_hgt_tgt_changed = true;

      m_onDisconnected = false;
   }

   updateGUI();   
}

void roi::updateGUI()
{
   if( m_appState != "READY" && m_appState != "OPERATING" && m_appState != "CONFIGURING" )
   {
      //Disable & zero all
      ui.lab_bin_x_title->setEnabled(false);
      ui.lab_bin_x_curr->setEnabled(false);
      ui.lab_bin_x_tgt->setEnabled(false);
      ui.val_bin_x->setText("---");
      ui.val_bin_x->setEnabled(false);
      ui.le_bin_x->setText("---");
      ui.le_bin_x->setEnabled(false);
   
      ui.lab_bin_y_title->setEnabled(false);
      ui.lab_bin_y_curr->setEnabled(false);
      ui.lab_bin_y_tgt->setEnabled(false);
      ui.val_bin_y->setText("---");
      ui.val_bin_y->setEnabled(false);
      ui.le_bin_y->setText("---");
      ui.le_bin_y->setEnabled(false);
      
      ui.lab_center_x_title->setEnabled(false);
      ui.lab_center_x_curr->setEnabled(false);
      ui.lab_center_x_tgt->setEnabled(false);
      ui.val_center_x->setText("---");
      ui.val_center_x->setEnabled(false);
      ui.le_center_x->setText("---");
      ui.le_center_x->setEnabled(false);
   
      ui.lab_center_y_title->setEnabled(false);
      ui.lab_center_y_curr->setEnabled(false);
      ui.lab_center_y_tgt->setEnabled(false);
      ui.val_center_y->setText("---");
      ui.val_center_y->setEnabled(false);
      ui.le_center_y->setText("---");
      ui.le_center_y->setEnabled(false);
   
      ui.lab_width_title->setEnabled(false);
      ui.lab_width_curr->setEnabled(false);
      ui.lab_width_tgt->setEnabled(false);
      ui.val_width->setText("---");
      ui.val_width->setEnabled(false);
      ui.le_width->setText("---");
      ui.le_width->setEnabled(false);
   
      ui.lab_height_title->setEnabled(false);
      ui.lab_height_curr->setEnabled(false);
      ui.lab_height_tgt->setEnabled(false);
      ui.val_height->setText("---");
      ui.val_height->setEnabled(false);
      ui.le_height->setText("---");
      ui.le_height->setEnabled(false);
   
      ui.button_reset->setEnabled(false);
      ui.button_loadlast->setEnabled(false);
      ui.button_check->setEnabled(false);
      ui.button_set->setEnabled(false);
      ui.button_last->setEnabled(false);
      ui.button_fullbin->setEnabled(false);
      ui.button_full->setEnabled(false);
      ui.button_default->setEnabled(false);
      
      return;
   }
   
   
   if( m_appState == "READY" || m_appState == "OPERATING" || m_appState == "CONFIGURING" )
   {
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
      
      if(m_bin_x_curr_changed)
      {
         ui.val_bin_x->setTextChanged(QString::number(m_bin_x_curr));
         m_bin_x_curr_changed = false;
      }
      else
      {
         ui.val_bin_x->setText(QString::number(m_bin_x_curr));
      }

      if(m_bin_x_tgt_changed)
      {
         ui.le_bin_x->setTextChanged(QString::number(m_bin_x_tgt));
         m_bin_x_tgt_changed = false;
      }
      else 
      {
         ui.le_bin_x->setText(QString::number(m_bin_x_tgt));
      }

      if(m_bin_y_curr_changed)
      {
         ui.val_bin_y->setTextChanged(QString::number(m_bin_y_curr));
         m_bin_y_curr_changed = false;
      }
      else
      {
         ui.val_bin_y->setText(QString::number(m_bin_y_curr));
      }

      if(m_bin_y_tgt_changed)
      {
         ui.le_bin_y->setTextChanged(QString::number(m_bin_y_tgt));
         m_bin_y_tgt_changed = false;
      }
      else
      {
         ui.le_bin_y->setText(QString::number(m_bin_y_tgt));
      }

      if(m_cen_x_curr_changed)
      {
         ui.val_center_x->setTextChanged(QString::number(m_cen_x_curr));
         m_cen_x_curr_changed = false;
      }
      else
      {
         ui.val_center_x->setText(QString::number(m_cen_x_curr));
      }

      if(m_cen_x_tgt_changed)
      {
         ui.le_center_x->setTextChanged(QString::number(m_cen_x_tgt));
         m_cen_x_tgt_changed = false;
      }
      else
      {
         ui.le_center_x->setText(QString::number(m_cen_x_tgt));
      }

      if(m_cen_y_curr_changed)
      {
         ui.val_center_y->setTextChanged(QString::number(m_cen_y_curr));
         m_cen_y_curr_changed = false;
      }
      else
      {
         ui.val_center_y->setText(QString::number(m_cen_y_curr));
      }

      if(m_cen_y_tgt_changed)
      {
         ui.le_center_y->setTextChanged(QString::number(m_cen_y_tgt));
         m_cen_y_tgt_changed = false;
      }
      else
      {
         ui.le_center_y->setText(QString::number(m_cen_y_tgt));
      }

      if(m_wid_curr_changed)
      {
         ui.val_width->setTextChanged(QString::number(m_wid_curr));
         m_wid_curr_changed = false;
      }
      else
      {
         ui.val_width->setText(QString::number(m_wid_curr));
      }

      if(m_wid_tgt_changed)
      {
         ui.le_width->setTextChanged(QString::number(m_wid_tgt));
         m_wid_tgt_changed = false;
      }
      else
      {
         ui.le_width->setText(QString::number(m_wid_tgt));
      }

      if(m_hgt_curr_changed)
      {
         ui.val_height->setTextChanged(QString::number(m_hgt_curr));
         m_hgt_curr_changed = false;
      }
      else
      {
         ui.val_height->setText(QString::number(m_hgt_curr));
      }

      if(m_hgt_tgt_changed)
      {
         ui.le_height->setTextChanged(QString::number(m_hgt_tgt));
         m_hgt_tgt_changed = false;
      }
      else
      {
         ui.le_height->setText(QString::number(m_hgt_tgt));
      }
   }
   
   if( m_appState == "OPERATING" ||  m_appState == "READY")
   {
      //Enable buttons too   
      ui.button_reset->setEnabled(true);
      ui.button_loadlast->setEnabled(true);
      ui.button_check->setEnabled(true);
      ui.button_set->setEnabled(true);
      ui.button_last->setEnabled(true);
      ui.button_fullbin->setEnabled(true);
      ui.button_full->setEnabled(true);
      ui.button_default->setEnabled(true);

      return;
   }
   else  //configuring
   {
      //Disable buttons while configuring
      ui.button_reset->setEnabled(false);
      ui.button_loadlast->setEnabled(false);
      ui.button_check->setEnabled(false);
      ui.button_set->setEnabled(false);
      ui.button_last->setEnabled(false);
      ui.button_fullbin->setEnabled(false);
      ui.button_full->setEnabled(false);
      ui.button_default->setEnabled(false);
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
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_camName);
   ip.setName("roi_region_bin_x");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.le_bin_x->editText().toDouble();
   
   sendNewProperty(ip);
}

void roi::on_le_bin_y_editingFinished()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_camName);
   ip.setName("roi_region_bin_y");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.le_bin_y->editText().toDouble();
   
   sendNewProperty(ip);
}

void roi::on_le_center_x_editingFinished()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_camName);
   ip.setName("roi_region_x");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.le_center_x->editText().toDouble();
   
   sendNewProperty(ip);
}

void roi::on_le_center_y_editingFinished()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_camName);
   ip.setName("roi_region_y");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.le_center_y->editText().toDouble();
   
   sendNewProperty(ip);
}

void roi::on_le_width_editingFinished()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_camName);
   ip.setName("roi_region_w");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.le_width->editText().toDouble();
   
   sendNewProperty(ip);
}

void roi::on_le_height_editingFinished()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_camName);
   ip.setName("roi_region_h");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.le_height->editText().toDouble();

   sendNewProperty(ip);
}

void roi::on_button_loadlast_pressed()
{
   clear_focus();

   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_camName);
   ipFreq.setName("roi_load_last");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq);  
}

void roi::on_button_reset_pressed()
{
   clear_focus();

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   ip.setDevice(m_camName);
   
   ip.setName("roi_region_bin_x");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_bin_x_curr;
   sendNewProperty(ip);

   ip.setName("roi_region_bin_y");
   ip["target"] = m_bin_y_curr;
   sendNewProperty(ip);

   ip.setName("roi_region_x");
   ip["target"] = m_cen_x_curr;
   sendNewProperty(ip);

   ip.setName("roi_region_y");
   ip["target"] = m_cen_y_curr;
   sendNewProperty(ip);

   ip.setName("roi_region_w");
   ip["target"] = m_wid_curr;
   sendNewProperty(ip);

   ip.setName("roi_region_h");
   ip["target"] = m_hgt_curr;
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

void roi::on_button_fullbin_pressed()
{
   clear_focus();

   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_camName);
   ipFreq.setName("roi_set_full_bin");
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
   ipFreq.setName("roi_set_default");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq); 
}


} //namespace xqt
   
#include "moc_roi.cpp"

#endif
