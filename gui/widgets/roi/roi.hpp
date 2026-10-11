
#ifndef roi_hpp
#define roi_hpp

#include "ui_roi.h"
#include <QSignalBlocker>
#include <QThread>
#include <cmath>
#include "../../../libMagAOX/common/roiGeometry.hpp"

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

   /// Preferred center display mode, retained while relative coordinates are unavailable.
   bool m_relativePreferred{true};
   /// Mode used for the currently displayed center editors.
   bool m_relativeActive{false};
   /// Writable Number definitions are required before accepting relative updates.
   bool m_relativeXDefined{false}, m_relativeYDefined{false};
   /// Availability of the two backend relative target properties.
   bool m_relativeXValid{false}, m_relativeYValid{false};
   /// Availability of a coherent applied reference.
   bool m_referenceValid{false};
   /// A changed reference or capability discards unsent center text on the GUI thread.
   bool m_discardCenters{true};
   /// Backend-reported target centers in current image pixels.
   double m_relativeX{0}, m_relativeY{0};
   /// Last coherent applied reference, used only to invalidate edits.
   MagAOX::roi::region m_reference;

   /// Cancel center text without triggering a command in either coordinate mode.
   void discardCenterEdits();

   /// Send a valid center to the property matching its displayed coordinate mode.
   void sendCenter(bool horizontal /**< [in] Select the horizontal center. */);

   bool m_onDisconnected {false};

    QTimer * m_updateTimer {nullptr};

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

   void handleDelProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has been deleted*/);

   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

   void clear_focus();

public slots:
   void updateGUI();

   /// Change center display mode, discarding any unsubmitted center text.
   void on_check_relative_toggled(bool checked /**< [in] Requested relative display mode. */);

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

    m_updateTimer = new QTimer(this);

    connect(m_updateTimer, SIGNAL(timeout()), this, SLOT(updateGUI()));
    m_updateTimer->start(250);

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

   m_parent->addSubscriberProperty(this, m_camName, "roi_region_x_relative");
   m_parent->addSubscriberProperty(this, m_camName, "roi_region_y_relative");
   m_parent->addSubscriberProperty(this, m_camName, "roi_region_state");
   return;
}

void roi::onConnect()
{
   if(QThread::currentThread() != thread())
   {
      QMetaObject::invokeMethod(this, [this]{ onConnect(); }, Qt::QueuedConnection);
      return;
   }
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
   if(QThread::currentThread() != thread())
   {
      QMetaObject::invokeMethod(this, [this]{ onDisconnect(); }, Qt::QueuedConnection);
      return;
   }
   m_relativeXDefined = m_relativeYDefined = false;
   m_relativeXValid = m_relativeYValid = m_referenceValid = false;
   m_discardCenters = true;
   m_relativeActive = false;
   discardCenterEdits();
   { // Block a mode change from being interpreted as user input.
      QSignalBlocker blocker(ui.check_relative);
      ui.check_relative->setChecked(false);
      ui.check_relative->setEnabled(false);
   }
   m_appState.clear();
   m_bin_x_curr = 0;
   m_bin_x_tgt = 0;
   m_bin_y_curr = 0;
   m_bin_y_tgt = 0;
   m_cen_x_curr = 0;
   m_cen_x_tgt = 0;
   m_cen_y_curr = 0;
   m_cen_y_tgt = 0;
   m_wid_curr = 0;
   m_wid_tgt = 0;
   m_hgt_curr = 0;
   m_hgt_tgt = 0;

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
   if(QThread::currentThread() != thread())
   {
      QMetaObject::invokeMethod(this, [this, ipRecv]{ handleDefProperty(ipRecv); }, Qt::QueuedConnection);
      return;
   }
   if(ipRecv.getDevice() == m_camName)
   {
      bool writable = ipRecv.getType() == pcf::IndiProperty::Number && ipRecv.getPerm() == pcf::IndiProperty::ReadWrite;
      if(ipRecv.getName() == "roi_region_x_relative") m_relativeXDefined = writable;
      if(ipRecv.getName() == "roi_region_y_relative") m_relativeYDefined = writable;
   }
   return handleSetProperty(ipRecv);
}

void roi::handleDelProperty( const pcf::IndiProperty & ipRecv)
{
   if(QThread::currentThread() != thread())
   {
      QMetaObject::invokeMethod(this, [this, ipRecv]{ handleDelProperty(ipRecv); }, Qt::QueuedConnection);
      return;
   }
   if(ipRecv.getDevice() != m_camName) return;

   if(ipRecv.getName().empty() || ipRecv.getName() == "roi_region_state" ||
      ipRecv.getName() == "roi_region_x_relative" || ipRecv.getName() == "roi_region_y_relative")
   {
      m_relativeXDefined = m_relativeYDefined = false;
   m_relativeXValid = m_relativeYValid = m_referenceValid = false;
      m_discardCenters = true;
   }
   if(ipRecv.getName().empty() || ipRecv.getName() == "fsm" ||
      ipRecv.getName() == "roi_region_bin_x" ||
      ipRecv.getName() == "roi_region_bin_y" ||
      ipRecv.getName() == "roi_region_x" ||
      ipRecv.getName() == "roi_region_y" ||
      ipRecv.getName() == "roi_region_w" ||
      ipRecv.getName() == "roi_region_h")
   {
      onDisconnect();
   }
}

void roi::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   if(QThread::currentThread() != thread())
   {
      QMetaObject::invokeMethod(this, [this, ipRecv]{ handleSetProperty(ipRecv); }, Qt::QueuedConnection);
      return;
   }
   if(ipRecv.getDevice() != m_camName) return;

   const auto &name = ipRecv.getName();
   if(name == "roi_region_x_relative" || name == "roi_region_y_relative")
   {
      bool horizontal = name == "roi_region_x_relative";
      bool valid = (horizontal ? m_relativeXDefined : m_relativeYDefined) &&
                   ipRecv.getState() == pcf::IndiProperty::Ok && ipRecv.find("target");
      double value = 0;
      try { if(valid) value = ipRecv["target"].get<double>(); }
      catch(...) { valid = false; }
      valid = valid && std::isfinite(value);
      if(horizontal) { m_relativeXValid = valid; m_relativeX = value; m_cen_x_tgt_changed = true; }
      else { m_relativeYValid = valid; m_relativeY = value; m_cen_y_tgt_changed = true; }
      if(!valid) m_discardCenters = true;
      return;
   }
   if(name == "roi_region_state")
   {
      MagAOX::roi::region reference;
      bool valid = ipRecv.getState() == pcf::IndiProperty::Ok;
      for(const auto &element : MagAOX::roi::names()) valid = valid && ipRecv.find(element);
      valid = valid && ipRecv.find("units");
      try
      {
         if(valid)
         {
            reference = {ipRecv["x"].get<float>(), ipRecv["y"].get<float>(), ipRecv["w"].get<int>(),
                         ipRecv["h"].get<int>(), ipRecv["bin_x"].get<int>(), ipRecv["bin_y"].get<int>()};
            const int units = ipRecv["units"].get<int>();
            valid = MagAOX::roi::valid(reference) && (units == 0 || units == 1);
         }
      }
      catch(...) { valid = false; }
      if(!valid || !m_referenceValid || !MagAOX::roi::same(reference, m_reference)) m_discardCenters = true;
      m_reference = reference;
      m_referenceValid = valid;
      return;
   }
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

}

void roi::updateGUI()
{
   const bool available = m_referenceValid && m_relativeXValid && m_relativeYValid &&
                          (m_appState == "READY" || m_appState == "OPERATING");
   const bool active = available && m_relativePreferred;
   if(active != m_relativeActive || m_discardCenters) discardCenterEdits();
   m_relativeActive = active;
   m_discardCenters = false;
   {
      QSignalBlocker blocker(ui.check_relative);
      ui.check_relative->setEnabled(available);
      ui.check_relative->setChecked(active);
      ui.check_relative->setToolTip(available ? "Target centers in current image pixels" :
                                   "Relative coordinates unavailable; showing absolute centers");
   }

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
         ui.le_center_x->setTextChanged(QString::number(m_relativeActive ? m_relativeX : m_cen_x_tgt));
         m_cen_x_tgt_changed = false;
      }
      else
      {
         ui.le_center_x->setText(QString::number(m_relativeActive ? m_relativeX : m_cen_x_tgt));
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
         ui.le_center_y->setTextChanged(QString::number(m_relativeActive ? m_relativeY : m_cen_y_tgt));
         m_cen_y_tgt_changed = false;
      }
      else
      {
         ui.le_center_y->setText(QString::number(m_relativeActive ? m_relativeY : m_cen_y_tgt));
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

void roi::discardCenterEdits()
{
   QSignalBlocker xBlocker(ui.le_center_x), yBlocker(ui.le_center_y);
   ui.le_center_x->stopEditing();
   ui.le_center_y->stopEditing();
   m_cen_x_tgt_changed = m_cen_y_tgt_changed = true;
}

void roi::on_check_relative_toggled(bool checked)
{
   discardCenterEdits();
   m_relativePreferred = checked;
   updateGUI();
}

void roi::sendCenter(bool horizontal)
{
   // Reject the edit if its applied reference changed before the GUI timer refreshed it.
   if(m_discardCenters || m_onDisconnected) { updateGUI(); return; }
   if(m_relativeActive && (!m_referenceValid || !m_relativeXValid || !m_relativeYValid ||
                          (m_appState != "READY" && m_appState != "OPERATING")))
   { updateGUI(); return; }
   bool ok;
   double value = (horizontal ? ui.le_center_x : ui.le_center_y)->editText().toDouble(&ok);
   if(!ok || !std::isfinite(value)) return;
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   ip.setDevice(m_camName);
   ip.setName(std::string(horizontal ? "roi_region_x" : "roi_region_y") + (m_relativeActive ? "_relative" : ""));
   ip.add(pcf::IndiElement("target"));
   ip["target"] = value;
   sendNewProperty(ip);
}

void roi::on_le_center_x_editingFinished()
{
   sendCenter(true);
}

void roi::on_le_center_y_editingFinished()
{
   sendCenter(false);
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
