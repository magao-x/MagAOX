#ifndef toggleSlider_hpp
#define toggleSlider_hpp

#include "ui_toggleSlider.h"

#include "xWidget.hpp"
#include "../../lib/multiIndiSubscriber.hpp"

namespace xqt 
{
   
/// Implementation of a toggle (on/off) Qt Slider
class toggleSlider : public xWidget
{
   Q_OBJECT
   
protected:
   
   //--------------- To move to derived class ----------------
   std::string m_device; ///< The name of the INDI device
   std::string m_property; ///< The name of the INDI property
   std::string m_element {"toggle"}; ///< The name of the INDI element
   //---------------------------------------------------------

   std::string m_label; ///< The label text

   int m_status; ///< Status of the toggle switch.  0 for off, 1 for int, 2 for on

   QTimer * m_tgtTimer {nullptr}; ///< Timer to normalize target if property stalls.

   int m_timeout {5000}; ///< The stall timeout, default is 5 sec.

   bool m_waiting {false}; ///< Flag indicating that we are waiting for a status change.  Is set to true after a toggle for m_timeout
   
   bool m_highlightChanges {true}; ///< Flag indicating that changes should be highlighted.  Default is true.
   bool m_statusChanged {false}; ///< Flag indicating that the status has changed.  Used to trigger a change highlight.


public:

   /// Default c'tor
   toggleSlider( QWidget * Parent = 0,                 ///< [in] [optional] the parent widget
                 Qt::WindowFlags f = Qt::WindowFlags() ///< [in] [optional] Qt window flags
               );

   //This will move to derived class
   toggleSlider( const std::string & ndevice,
                 const std::string & nproperty,
                 const std::string & nlabel,
                 QWidget * Parent = 0, 
                 Qt::WindowFlags f = Qt::WindowFlags()
               );

   //This will move to derived class
   toggleSlider( const std::string & ndevice,
                 const std::string & nproperty,
                 const std::string & nelement,
                 const std::string & nlabel,
                 QWidget * Parent = 0, 
                 Qt::WindowFlags f = Qt::WindowFlags()
               );
   
private:
   void construct(); //common to all constructors

public:
   ~toggleSlider();

   /// Set the label text
   void label( const std::string & nlabel /**< [in] the new label text*/);

   /// Get the label text
   std::string label();

   
   /// Set the stretch of the horizontal layout
   void setStretch( int sSpacer,           ///< [in] Stretch of the spacer.  If 0, the spacer is removed.
                    int sLabel,            ///< [in] Stretch of the label
                    int sSlider,           ///< [in] Stretch of the value
                    bool rmSpacer = false, ///< [in] [optional] if true the spacer is removed from the layout
                    bool rmLabel = false   ///< [in] [optional] if true the label is removed from the layout
                  );


   //------------ move to derived class ------------------
   void setup( const std::string & ndevice,
               const std::string & nproperty,
               const std::string & nelement,
               const std::string & nlabel
             );

   virtual void subscribe();
             
   virtual void onConnect();

   virtual void onDisconnect();
   
   virtual void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   virtual void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   //------------------------------------------------------

public slots:

   virtual void guiUpdate();

   void on_slider_sliderPressed();
   void on_slider_sliderMoved();
   void on_slider_sliderReleased();

   void tgtTimerTimeout();

   //------------ move to derived class ------------
   void doToggle(bool onoff);
   //-----------------------------------------------

signals:

   void startTgtTimer(int);

   void stopTgtTimer();

   void updateGUI();

   void toggle(bool onoff);


protected:
     
   Ui::toggleSlider ui;

private:
   bool m_spacerRemoved {false};
   bool m_labelRemoved {false};

};
   
toggleSlider::toggleSlider( QWidget * Parent, 
                            Qt::WindowFlags f) : xWidget(Parent, f)
{
   construct();
}

toggleSlider::toggleSlider( const std::string & ndevice,
                            const std::string & nproperty,
                            const std::string & nlabel,
                            QWidget * Parent, 
                            Qt::WindowFlags f) : xWidget(Parent, f), m_device{ndevice}, m_property{nproperty}
{
   construct();
   label(nlabel);
}

toggleSlider::toggleSlider( const std::string & ndevice,
                            const std::string & nproperty,
                            const std::string & nelement, 
                            const std::string & nlabel,
                            QWidget * Parent, 
                            Qt::WindowFlags f) : xWidget(Parent, f), m_device{ndevice}, m_property{nproperty}, m_element{nelement}
{
   construct();
   label(nlabel);
}

void toggleSlider::construct()
{
   ui.setupUi(this);
   
   QFont qf = ui.label->font();
   qf.setPixelSize(XW_FONT_SIZE);
   ui.label->setFont(qf);

   m_tgtTimer = new QTimer(this);

   connect(m_tgtTimer, SIGNAL(timeout()), this, SLOT(tgtTimerTimeout()));
   connect(this, SIGNAL(startTgtTimer(int)), m_tgtTimer, SLOT(start(int)));
   connect(this, SIGNAL(updateGUI()), this, SLOT(guiUpdate()));

   //This will get moved to derived class:
   connect(this, SIGNAL(toggle(bool)), this, SLOT(doToggle(bool)));

   onDisconnect();
}

toggleSlider::~toggleSlider()
{
}

void toggleSlider::label(const std::string &nlab)
{
   m_label = nlab;
   ui.label->setText(m_label.c_str());
}

std::string toggleSlider::label()
{
   return ui.label->text().toStdString();
}

void toggleSlider::setStretch( int sSpacer, 
                               int sLabel, 
                               int sSlider,
                               bool removeSpacer,
                               bool removeLabel
                             )
{
   //Have to do this b/c hlayout is not exposed for whatever reason.
   QHBoxLayout * qhbl = findChild<QHBoxLayout*>("hlayout");
   if(qhbl)
   {
      qhbl->setStretch(0,sSpacer);
      qhbl->setStretch(1,sLabel);
      qhbl->setStretch(2,sSlider);

      if(removeSpacer) //remove the spacer
      {
         if(!m_spacerRemoved)
         {
            QLayoutItem * qli = qhbl->itemAt(0);
            if(qli) qhbl->removeItem(qli);
            m_spacerRemoved = true;
         }
      }

      if(removeLabel)
      {
         if(!m_labelRemoved)
         {
            int in = 1;
            if(m_spacerRemoved) in = 0;

            QLayoutItem * qli = qhbl->itemAt(in);
            if(qli) qhbl->removeItem(qli);
            m_labelRemoved = true;
         }
      }
   }
}

void toggleSlider::setup( const std::string & ndevice,
                          const std::string & nproperty,
                          const std::string & nelement, 
                          const std::string & nlabel
                        )
{
   m_device = ndevice;
   m_property = nproperty;
   m_element = nelement;
   label(nlabel);
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
         else if(ipRecv["toggle"] == pcf::IndiElement::SwitchState::On) m_status = 2;
         else m_status = 0;

         if(currStatus != m_status) 
         {
            m_waiting = false;
            m_statusChanged = true;
         }

      }
   }

   emit updateGUI();
}

void toggleSlider::guiUpdate()
{
   if(isEnabled())
   {
      if(!m_waiting)
      {
         if(m_status == 2) ui.slider->setSliderPosition(ui.slider->maximum());
         else if(m_status == 1) ui.slider->setSliderPosition(ui.slider->minimum() + 0.5*(ui.slider->maximum()-ui.slider->minimum()));
         else ui.slider->setSliderPosition(ui.slider->minimum());
      }
   }
   else ui.slider->setSliderPosition(ui.slider->minimum());

} //guiUpdate()

void toggleSlider::on_slider_sliderPressed()
{
   m_waiting = true;
   emit startTgtTimer(m_timeout);
}

void toggleSlider::on_slider_sliderMoved()
{
   m_waiting = true;
   emit startTgtTimer(m_timeout);
}

void toggleSlider::on_slider_sliderReleased()
{
   int state = -1;

   if( ui.slider->sliderPosition() > ui.slider->minimum()+0.9*(ui.slider->maximum()-ui.slider->minimum()))
   {
      state = 2;   
   }
   else if( ui.slider->sliderPosition() < ui.slider->minimum()+0.1*(ui.slider->maximum()-ui.slider->minimum())) 
   {
      state = 0;
   }

   if(state == 2)
   {
      emit toggle(true); 
      m_waiting = true; 
      emit startTgtTimer(m_timeout);
   }
   else if(state == 0)
   {
      emit toggle(false);
      m_waiting = true;
      emit startTgtTimer(m_timeout);
   }
   else
   {
      if(m_status == 2) ui.slider->setSliderPosition(ui.slider->maximum());
      else if(m_status == 1) ui.slider->setSliderPosition(ui.slider->minimum() + 0.5*(ui.slider->maximum()-ui.slider->minimum()));
      else ui.slider->setSliderPosition(ui.slider->minimum());
   }
}

void toggleSlider::doToggle(bool onoff)
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_device);
   ipFreq.setName(m_property);
   ipFreq.add(pcf::IndiElement("toggle"));

   if(onoff == false)
   {
      ipFreq["toggle"].switchState(pcf::IndiElement::SwitchState::Off);
   }
   else
   {
      ipFreq["toggle"].switchState(pcf::IndiElement::SwitchState::On);
   }

   sendNewProperty(ipFreq); 
}

void toggleSlider::tgtTimerTimeout()
{
   m_waiting = false;
   emit stopTgtTimer();
   emit updateGUI();
}

} //namespace xqt
   
#include "moc_toggleSlider.cpp"

#endif
