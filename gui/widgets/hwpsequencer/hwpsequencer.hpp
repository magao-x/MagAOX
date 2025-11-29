#ifndef hwpsequencer_hpp
#define hwpsequencer_hpp

#include <mx/app/appConfigurator.hpp>

#include "ui_hwpsequencer.h"

#include "xWidgets/xWidget.hpp"
#include "xWidgets/fsmDisplay.hpp"
#include "xWidgets/statusEntry.hpp"
#include "xWidgets/statusDisplay.hpp"
#include "xWidgets/stageStatus.hpp"
#include "xWidgets/toggleSlider.hpp"


#include "stage/stage.hpp"

#include <QThread>

namespace xqt
{

/// A GUI for sequencing the HWP for polarimetric differential imaging
class hwpsequencer : public xWidget
{
    Q_OBJECT

protected:

    std::string m_appState;

    std::string m_hwptrack {"hwptrack"};

    std::string m_stagehwprot {"stagehwprot"};

    std::string m_stagehwplin {"stagehwplin"};

    bool m_tracking;

    float m_hwpsetpoint;

    float m_hwpoffset;

    float m_hwpactual;

    std::string m_hwpname;

    int m_cycleindex;

    int m_cycletotal;


    fsmDisplay * ui_fsmState {nullptr};

    statusEntry * ui_hwpangle {nullptr};

    statusCombo * ui_hwplinstage {nullptr};

    toggleSlider * ui_tracking {nullptr};

    QPushButton * ui_startSequencing {nullptr};

    bool m_sequencing {false};

    bool m_inUpdate {false};

    QTimer * m_updateTimer {nullptr}; ///< Timer for periodic updates

    bool m_connected {false};

public:
    explicit hwpsequencer( QWidget * Parent = 0, Qt::WindowFlags f = Qt::WindowFlags() );

    ~hwpsequencer();

    void subscribe();

    virtual void onConnect();
    virtual void onDisconnect();

    void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

    void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

    void hideAll();

    void setEnableDisable(bool tf, bool all=true);

    //static b/c it gets called before the device is instantiated.
    static void setupConfig( mx::app::appConfigurator & config );

    void loadConfig( mx::app::appConfigurator & config );

public slots:

    void updateGUI();

    void setup_hwpangle(bool ro);

    void setup_hwplinstage(bool ro);

    void setup_tracking();

    void setup_startSequence();

    void startSequence();

signals:

    void doUpdateGUI();

    void updateTimerStop();
    void updateTimerStart(int);


    void add_hwpangle(bool ro);

    void add_hwplinstage(bool ro);

    void add_tracking();

    void add_startSequence();

private:

    Ui::hwpsequencer ui;
};

hwpsequencer::hwpsequencer(QWidget * Parent, Qt::WindowFlags f) : xWidget(Parent, f),
{

    ui.setupUi(this);

    m_updateTimer = new QTimer(this);

    connect(this, SIGNAL(doUpdateGUI()), this, SLOT(updateGUI()));

    connect(m_updateTimer, SIGNAL(timeout()), this, SLOT(updateGUI()));
    connect(this, SIGNAL(updateTimerStop()), m_updateTimer, SLOT(stop()));
    connect(this, SIGNAL(updateTimerStart(int)), m_updateTimer, SLOT(start(int)));


    connect(this, SIGNAL(add_hwpangle(bool)), this, SLOT(setup_hwpangle(bool)));
    connect(this, SIGNAL(add_hwplinstage(bool)), this, SLOT(setup_hwplinstage(bool)));
    connect(this, SIGNAL(add_tracking()), this, SLOT(setup_tracking()));

    connect(this, SIGNAL(add_startSequence()), this, SLOT(setup_startSequence()));

    QSpacerItem *holder = new QSpacerItem(10,0, QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui.grid->addItem(holder, 2,1,1,1);

    ui_fsmState = new xqt::fsmDisplay(this);
    ui_fsmState->setObjectName(QString::fromUtf8("fsmState"));
    ui.grid->addWidget(ui_fsmState, 1, 0, 1, 1);
    ui_fsmState->device(m_hwptrack);

   //  QFont qf = ui.lab_camName->font();
   //  qf.setPixelSize(XW_FONT_SIZE+3);
   //  ui.lab_camName->setFont(qf);

   //  ui.lab_camName->setText(m_hwptrack.c_str());

    onDisconnect();
}

hwpsequencer::~hwpsequencer()
{
}

void hwpsequencer::subscribe()
{
    if(!m_parent) return;

    m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_hwptrack, "");
    m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_hwptrack, "fsm");

    m_parent->addSubscriber(ui_fsmState);
    if(ui_hwpangle) m_parent->addSubscriber(ui_hwpangle);
    if(ui_hwplinstage) m_parent->addSubscriber(ui_hwplinstage);
    if(ui_tracking) m_parent->addSubscriber(ui_tracking);

    return;
}

void hwpsequencer::onConnect()
{
    ui.lab_camName->setEnabled(true);

    setWindowTitle(QString(("HWP Sequencer").c_str()));

    ui_fsmState->onConnect();

    if(ui_stage.size() > 0)
    {
        for(size_t n = 0; n < ui_stage.size(); ++n)
        {
            ui_stage[n]->onConnect();
        }
    }

    if(ui_modes) ui_modes->onConnect();

    if(ui_hwpangle) ui_hwpangle->onConnect();

    if(ui_hwplinstage) ui_hwplinstage->onConnect();

    if(ui_tracking) ui_tracking->onConnect();

    clearFocus();

    m_connected = true;

    emit doUpdateGUI();
}

void hwpsequencer::onDisconnect()
{

    setWindowTitle(QString(("HWP Sequencer").c_str())) + QString(" (disconnected)"));

    ui_fsmState->onDisconnect();


    if(ui_stage.size() > 0)
    {
        for(size_t n =0; n < ui_stage.size(); ++n)
        {
            ui_stage[n]->onDisconnect();
        }
    }

    if(ui_modes) ui_modes->onDisconnect();

    if(ui_hwpangle) ui_hwpangle->onDisconnect();

    if(ui_hwplinstage) ui_hwplinstage->onDisconnect();

    if(ui_tracking) ui_tracking->onDisconnect();

    clearFocus();

    m_connected = false;
    while(m_inUpdate)
    {
       QThread::msleep(10); //Wait to get out of update
    }
    emit updateTimerStop();

    setEnableDisable(false);
}

void hwpsequencer::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void hwpsequencer::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getDevice() != m_hwptrack && ipRecv.getDevice() != m_darkName && ipRecv.getDevice() != m_avgName) return;

   if(ipRecv.getDevice() == m_hwptrack)
   {
      if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_appState = ipRecv["state"].get<std::string>();
         }
      }

      if(ipRecv.getName() == "tracking")
      {
         if(!ui_tracking)
         {
            emit add_tracking();
         }
      }

      if(ipRecv.getName() == "hwpangle")
      {
         if(!ui_hwpangle)
         {
            bool ro = true;
            if(ipRecv.find("target")) ro = false;

            emit add_hwpangle(ro);
         }
      }

      if(ipRecv.getName() == "hwplinstage")
      {
         if(!ui_hwplinstage)
         {
            bool ro = true;
            if(ipRecv.find("target")) ro = false;

            emit add_hwplinstage(ro);
         }
      }

   }

   emit doUpdateGUI();
}

void hwpsequencer::hideAll()
{
   return;
}

void hwpsequencer::setEnableDisable(bool tf, bool all)
{
    if(all)
    {
       ui.lab_camName->setEnabled(tf);
       ui_fsmState->setEnabled(tf);
    }

    if(ui_hwpangle) ui_hwpangle->setEnabled(tf);

    if(ui_hwplinstage) ui_hwplinstage->setEnabled(tf);

    if(ui_stage.size() > 0)
    {
        for(size_t n = 0; n < ui_stage.size(); ++n)
        {
            ui_stage[n]->setEnabled(tf);
        }
    }

    if(ui_startSequence) ui_startSequence->setEnabled(tf);

}

void hwpsequencer::setupConfig( mx::app::appConfigurator & config )
{
   config.add("hwpsequencer.stages", "", "hwpsequencer.stages", mx::app::argType::Required, "hwpsequencer", "stages", false, "vector<string>", "List of stages associated with this app");
}

void hwpsequencer::loadConfig( mx::app::appConfigurator & config )
{
    config(m_stageNames, "hwpsequencer.stages");
    for(size_t n = 0; n < m_stageNames.size(); ++n)
    {
        setup_stage();
    }
    onDisconnect();
}

void hwpsequencer::updateGUI()
{
    if(m_inUpdate || !m_connected) return;
    emit updateTimerStop();
    m_inUpdate = true;

    if( m_appState == "NODEVICE" || m_appState == "NOTCONNECTED" || m_appState == "CONNECTED")
    {
       setEnableDisable(false, false);
       ui.lab_camName->setEnabled(true);
       ui_fsmState->setEnabled(true);
    }
    else if( m_appState != "READY" && m_appState != "OPERATING" && m_appState != "CONFIGURING")
    {
       setEnableDisable(false);
       m_inUpdate = false;

       emit updateTimerStart(1000);
       return;
    }
    else //if( m_appState == "READY" || m_appState == "OPERATING" || m_appState == "CONFIGURING")
    {
       setEnableDisable(true);
    }

    //Update the component GUIs to ensure they update for connection state, etc.

    if(ui_stage.size() > 0)
    {
        for(size_t n = 0; n < ui_stage.size(); ++n)
        {
            ui_stage[n]->updateGUI();
        }
    }

    if(ui_hwpangle) ui_hwpangle->updateGUI();
    if(ui_hwplinstage) ui_hwplinstage->updateGUI();
    if(ui_tracking) ui_tracking->updateGUI();

    if( (m_appState == "READY" || m_appState == "OPERATING") && ui_startSequence )
    {
       if(m_sequencing)
       {
          ui_startSequence->setEnabled(false);
       }
       else
       {
          ui_startSequence->setEnabled(true);
       }
    }

    emit updateTimerStart(1000);
    m_inUpdate = false;

} //updateGUI()


void hwpsequencer::setup_hwpangle(bool ro)
{
   if(ui_hwpangle) return;

   ui_hwpangle = new statusEntry(this);
   ui_hwpangle->setObjectName(QString::fromUtf8("hwpangle"));
   ui_hwpangle->setup(m_hwptrack, "hwpangle", statusEntry::FLOAT, "HWP Angle", "deg");
   ui_hwpangle->highlightChanges(true);
   ui_hwpangle->readOnly(ro);

   ui.grid->addWidget(ui_hwpangle, 7, 1, 1, 1);

   ui_hwpangle->onDisconnect();

   m_parent->addSubscriber(ui_hwpangle);
}

void hwpsequencer::setup_hwplinstage(bool ro)
{
   if(ui_hwplinstage) return;

   ui_hwplinstage = new statusEntry(this);
   ui_hwplinstage->setObjectName(QString::fromUtf8("hwplinstage"));
   // ui_hwplinstage->setup(m_hwptrack, "hwplinstage", statusCombo::, "HWP Angle", "deg");
   ui_hwplinstage->highlightChanges(true);
   ui_hwplinstage->readOnly(ro);

   ui.grid->addWidget(ui_hwplinstage, 7, 1, 1, 1);

   ui_hwplinstage->onDisconnect();

   m_parent->addSubscriber(ui_hwplinstage);
}

void hwpsequencer::setup_tracking()
{
   if(ui_tracking) return;

   ui_tracking = new toggleSlider(m_hwptrack, "tracking", "Tracking", this);
   ui_tracking->setObjectName(QString::fromUtf8("tracking"));

   ui.grid->addWidget(ui_tracking, 10, 1, 1, 1);

   ui_tracking->onDisconnect();

   m_parent->addSubscriber(ui_tracking);
}

void hwpsequencer::setup_startSequence()
{
    if(ui_startSequence) return;

    ui_startSequence = new QPushButton(this);
    ui_startSequence->setObjectName(QString::fromUtf8("startSequence"));
    ui_startSequence->setText("Start sequence");
    ui_startSequence->setMaximumWidth(200);
    ui_startSequence->setFocusPolicy(Qt::NoFocus);
    connect(ui_startSequence, SIGNAL(pressed()), this, SLOT(startSequence()));

    int doff = 0;
    if(ui_stage.size() > 4)
    {
        doff = ui_stage.size() - 4;
    }

    ui.grid->addWidget(ui_startSequence, 9 + doff, 0, 1, 1,Qt::AlignHCenter);

}

void hwpsequencer::startSequence()
{

}


} //namespace xqt

#include "moc_hwpsequencer.cpp"

#endif
