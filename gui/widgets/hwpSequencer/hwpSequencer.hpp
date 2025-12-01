
#ifndef hwpSequencer_hpp
#define hwpSequencer_hpp

#include "ui_hwpSequencer.h"

#include "../xWidgets/xWidget.hpp"

namespace xqt
{

class hwpSequencer : public xWidget
{
   Q_OBJECT

protected:

   std::string m_appState;



public:
   explicit hwpSequencer( QWidget * Parent = 0,
                    Qt::WindowFlags f = Qt::WindowFlags()
                  );

   ~hwpSequencer();

   void subscribe();

   virtual void onConnect();
   virtual void onDisconnect();

   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);


public slots:
   void updateGUI();

   void on_comboSelectPolLin_activated(int);

   void on_buttonStartSequence_pressed();
   void on_buttonLastCycle_pressed();
   void on_buttonStopNow_pressed();

signals:

   void doUpdateGUI();

private:

   Ui::hwpSequencer ui;
};

hwpSequencer::hwpSequencer(
                QWidget * Parent,
                Qt::WindowFlags f) : xWidget(Parent, f)
{
   ui.setupUi(this);
   //ui.labelDMName->setText(m_dmName.c_str());

   setWindowTitle(QString("HWP Sequencer (disconnected)"));

   ui.fsmState->READY("");

   setXwFont(ui.buttonStartSequence);
   setXwFont(ui.buttonLastCycle);
   setXwFont(ui.buttonStopNow);
   setXwFont(ui.comboSelectPolLin);

   connect(this, SIGNAL(doUpdateGUI()), this, SLOT(updateGUI()));

   onDisconnect();
}

hwpSequencer::~hwpSequencer()
{
   if(m_parent) m_parent->unsubscribe(this);
}

void hwpSequencer::subscribe()
{
   if(!m_parent) return;

   m_parent->addSubscriberProperty(this, "hwptrack", "fsm");
   m_parent->addSubscriber(ui.fsmState);

   return;
}

void hwpSequencer::onConnect()
{
   ui.fsmState->setEnabled(true);

   ui.fsmState->onConnect();

   setWindowTitle(QString("HWP Sequencer"));
}

void hwpSequencer::onDisconnect()
{
   //ui.labelDMName->setEnabled(false);
   ui.fsmState->setEnabled(false);

   setWindowTitle(QString("HWP Sequencer (disconnected)"));

   ui.fsmState->onDisconnect();

   multiIndiSubscriber::onDisconnect();
}

void hwpSequencer::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void hwpSequencer::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   emit doUpdateGUI();
}

void hwpSequencer::updateGUI()
{


} //updateGUI()

void hwpSequencer::on_buttonStartSequence_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);

   // ipFreq.setDevice(m_dmName);
   // ipFreq.setName("test_set");
   // ipFreq.add(pcf::IndiElement("toggle"));
   // ipFreq["toggle"] = pcf::IndiElement::Off;

   // sendNewProperty(ipFreq);
}

void hwpSequencer::on_buttonStopNow_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);

   ipFreq.setDevice("observers");
   ipFreq.setName("obs_on");
   ipFreq.add(pcf::IndiElement("toggle"));
   ipFreq["toggle"] = pcf::IndiElement::Off;

   sendNewProperty(ipFreq);
}


} //namespace xqt

#include "moc_hwpSequencer.cpp"

#endif
