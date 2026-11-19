/** \file offloadCtrl.hpp
 * \brief GUI widget controlling tweeter-to-woofer and woofer-to-telescope offloading.
 */

#ifndef offloadCtrl_hpp
#define offloadCtrl_hpp

#include <QWidget>
#include <QMutex>
#include <QTimer>

#include "ui_offloadCtrl.h"

#include "../xWidgets/xWidget.hpp"
#include "../xWidgets/gainCtrl.hpp"
#include "../xWidgets/statusEntry.hpp"

namespace xqt
{

/// GUI widget controlling t2wOffloader (loop, offload rate, modes, gain, leak, zero) and tcsi telescope offloading.
class offloadCtrl : public xWidget
{
     Q_OBJECT

protected:

    std::string m_t2wFsmState; ///< FSM state of t2wOffloader, enabling its controls.

    bool m_offlState {false}; ///< Whether t2wOffloader is offloading.

    std::string m_tweeterAvgFsmState; ///< FSM state of dmtweeter-avg, enabling the offload rate entry.

    std::string m_tcsiFsmState; ///< FSM state of tcsi, enabling the telescope offload controls.

    QTimer * m_updateTimer {nullptr}; ///< Timer refreshing the GUI.

public:
    /// Construct the widget and set up its controls.
    offloadCtrl( QWidget * Parent = 0,            /**< [in] the parent widget */
                 Qt::WindowFlags f = Qt::WindowFlags() /**< [in] Qt window flags */
               );

    /// Destructor.
    ~offloadCtrl();

    /// Subscribe to the INDI properties shown by this widget.
    void subscribe();

    /// Handle connection to the INDI server.
    void onConnect();

    /// Handle disconnection from the INDI server, clearing the tracked states.
    void onDisconnect();

    /// Handle a property definition, treated as a set.
    void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

    /// Handle a property deletion, disconnecting if a tracked device goes away.
    void handleDelProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has been deleted*/);

    /// Handle a property update, tracking device states.
    void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

public slots:

    /// Enable or disable controls according to the tracked device states.
    void updateGUI();

    /// Request t2wOffloader to zero the offload.
    void on_button_zero_pressed();

    /// Request tcsi to dump the telescope tip/tilt offload.
    void on_button_TelTTDump_pressed();

    /// Request tcsi to dump the telescope focus offload.
    void on_button_TelFocusDump_pressed();

signals:

    /// Request a GUI update from another thread.
    void doUpdateGUI();

private:

    Ui::offloadCtrl ui; ///< The Qt Designer UI.
};


offloadCtrl::offloadCtrl( QWidget * Parent,
                          Qt::WindowFlags f) : xWidget(Parent, f)
{
    ui.setupUi(this);

    setXwFont(ui.label_Tweeter2Woofer);
    ui.label_offl_state->setProperty("isStatus", true);
    setXwFont(ui.label_offl_state);
    setXwFont(ui.slider_loop);

    ui.slider_loop->setup("t2wOffloader", "offload", "toggle", "");
    ui.slider_loop->setStretch(0,0,10, true, true);

    setXwFont(ui.button_zero);

    setXwFont(ui.label_avgInt);
    setXwFont(ui.avgInt);
    ui.avgInt->setup("t2wOffloader", "offloadRate", statusEntry::FLOAT , "", "");
    ui.avgInt->setStretch(0,0,1);


    setXwFont(ui.label_nModes);
    setXwFont(ui.nModes);
    ui.nModes->setup("t2wOffloader", "numModes", statusEntry::INT , "", "");
    ui.nModes->setStretch(0,0,1);

    ui.gainCtrl->setup("t2wOffloader", "gain", "Gain", -1, -1);
    ui.mcCtrl->setup("t2wOffloader", "leak", "Leak", -1, -1);
    ui.mcCtrl->makeMultCoeffCtrl();

    setXwFont(ui.label_Woofer2Telescope);
    setXwFont(ui.labelTT);
    setXwFont(ui.labelFocus);
    ui.sliderTelTTEnable->setup("tcsi", "offlTT_enable", "toggle", "");
    ui.sliderTelTTEnable->setStretch(0,0,10, true, true);

    ui.telTTGain->setup("tcsi", "offlTT_gain", statusEntry::FLOAT, "Gain", "");
    ui.telTTGain->setStretch(0,3,9);
    ui.telTTThresh->setup("tcsi", "offlTT_thresh", statusEntry::FLOAT, "Threshold", "");
    ui.telTTThresh->setStretch(0,3,9);
    ui.telTTAvgInt->setup("tcsi", "offlTT_avgInt", statusEntry::FLOAT, "Avg.", "sec");
    ui.telTTAvgInt->setStretch(0,3,9);

    ui.sliderTelFocusEnable->setup("tcsi", "offlF_enable", "toggle", "");
    ui.sliderTelFocusEnable->setStretch(0,0,10, true, true);

    ui.telFocusGain->setup("tcsi", "offlF_gain", statusEntry::FLOAT, "Gain", "");
    ui.telFocusGain->setStretch(0,3,9);
    ui.telFocusThresh->setup("tcsi", "offlF_thresh", statusEntry::FLOAT, "Threshold", "");
    ui.telFocusThresh->setStretch(0,3,9);
    ui.telFocusAvgInt->setup("tcsi", "offlF_avgInt", statusEntry::FLOAT, "Avg.", "sec");
    ui.telFocusAvgInt->setStretch(0,3,9);

    setXwFont(ui.button_TelTTDump);
    setXwFont(ui.button_TelFocusDump);

    m_updateTimer = new QTimer;

    connect(m_updateTimer, SIGNAL(timeout()), this, SLOT(updateGUI()));

    m_updateTimer->start(250);

    connect(this, SIGNAL(doUpdateGUI()), this, SLOT(updateGUI()));

    onDisconnect();

}

offloadCtrl::~offloadCtrl()
{
}

void offloadCtrl::subscribe()
{
    if(!m_parent) return;

    m_parent->addSubscriberProperty(this, "t2wOffloader", "fsm");
    m_parent->addSubscriberProperty(this, "t2wOffloader", "offload");

    m_parent->addSubscriberProperty(this, "dmtweeter-avg", "fsm");
    m_parent->addSubscriberProperty(this, "tcsi", "fsm");

    m_parent->addSubscriber(ui.slider_loop);
    m_parent->addSubscriber(ui.avgInt);
    m_parent->addSubscriber(ui.nModes);

    m_parent->addSubscriber(ui.gainCtrl);
    m_parent->addSubscriber(ui.mcCtrl);

    m_parent->addSubscriber(ui.sliderTelTTEnable);
    m_parent->addSubscriber(ui.telTTGain);
    m_parent->addSubscriber(ui.telTTThresh);
    m_parent->addSubscriber(ui.telTTAvgInt);

    m_parent->addSubscriber(ui.sliderTelFocusEnable);
    m_parent->addSubscriber(ui.telFocusGain);
    m_parent->addSubscriber(ui.telFocusThresh);
    m_parent->addSubscriber(ui.telFocusAvgInt);

   return;
}

void offloadCtrl::onConnect()
{
    setWindowTitle(QString("Offloading Ctrl"));

    xWidget::onConnect();

}

void offloadCtrl::onDisconnect()
{
    m_t2wFsmState.clear();
    m_offlState = false;
    m_tweeterAvgFsmState.clear();
    m_tcsiFsmState.clear();

    setWindowTitle(QString("Offloading Ctrl (disconnected)"));

    xWidget::onDisconnect();

}

void offloadCtrl::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void offloadCtrl::handleDelProperty( const pcf::IndiProperty & ipRecv)
{
    if(ipRecv.getDevice() == "t2wOffloader")
    {
        if(ipRecv.getName() == "fsm" || ipRecv.getName() == "offload")
        {
            onDisconnect();
        }
    }
    else if(ipRecv.getDevice() == "dmtweeter-avg")
    {
        if(ipRecv.getName() == "fsm")
        {
            onDisconnect();
        }
    }
    else if(ipRecv.getDevice() == "tcsi")
    {
        if(ipRecv.getName() == "fsm")
        {
            onDisconnect();
        }
    }
}

void offloadCtrl::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
    if(ipRecv.getDevice() == "t2wOffloader")
    {
        if(ipRecv.getName() == "fsm")
        {
            if(ipRecv.find("state"))
            {
                m_t2wFsmState = ipRecv["state"].get<std::string>();
             }
        }
        else if(ipRecv.getName() == "offload")
        {
            if(ipRecv.find("toggle"))
            {
                if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
                {
                    m_offlState = true;
                }
                else
                {
                    m_offlState = false;
                }
            }
        }

    }
    else if(ipRecv.getDevice() == "dmtweeter-avg")
    {
        if(ipRecv.getName() == "fsm")
        {
            if(ipRecv.find("state"))
            {
                m_tweeterAvgFsmState = ipRecv["state"].get<std::string>();
             }
        }
    }
    else if(ipRecv.getDevice() == "tcsi")
    {
        if(ipRecv.getName() == "fsm")
        {
            if(ipRecv.find("state"))
            {
                m_tcsiFsmState = ipRecv["state"].get<std::string>();
             }
        }
    }

    emit doUpdateGUI();
}

void offloadCtrl::updateGUI()
{

    if(m_t2wFsmState != "READY" && m_t2wFsmState != "OPERATING")
    {
        ui.label_offl_state->setEnabled(false);
        ui.label_offl_state->setText("---");

        ui.slider_loop->setEnabled(false);
        ui.nModes->setEnabled(false);
        ui.button_zero->setEnabled(false);
        ui.gainCtrl->setEnabled(false);
        ui.mcCtrl->setEnabled(false);
    }
    else
    {
        ui.label_offl_state->setEnabled(true);

        if(m_offlState == true)
        {
            ui.label_offl_state->setText("ON");
        }
        else
        {
            ui.label_offl_state->setText("OFF");
        }

        ui.slider_loop->setEnabled(true);
        ui.nModes->setEnabled(true);
        ui.button_zero->setEnabled(true);
        ui.gainCtrl->setEnabled(true);
        ui.mcCtrl->setEnabled(true);
    }

    if(m_tweeterAvgFsmState != "READY" && m_tweeterAvgFsmState != "OPERATING")
    {
        ui.avgInt->setEnabled(false);
    }
    else
    {
        ui.avgInt->setEnabled(true);
    }

    if(m_tcsiFsmState != "CONNECTED")
    {
        ui.sliderTelTTEnable->setEnabled(false);
        ui.sliderTelFocusEnable->setEnabled(false);
        ui.telTTGain->setEnabled(false);
        ui.telTTThresh->setEnabled(false);
        ui.telTTAvgInt->setEnabled(false);

        ui.telFocusGain->setEnabled(false);
        ui.telFocusThresh->setEnabled(false);
        ui.telFocusAvgInt->setEnabled(false);

        ui.button_TelTTDump->setEnabled(false);
        ui.button_TelFocusDump->setEnabled(false);

    }
    else
    {
        ui.sliderTelTTEnable->setEnabled(true);
        ui.sliderTelFocusEnable->setEnabled(true);
        ui.telTTGain->setEnabled(true);
        ui.telTTThresh->setEnabled(true);
        ui.telTTAvgInt->setEnabled(true);

        ui.telFocusGain->setEnabled(true);
        ui.telFocusThresh->setEnabled(true);
        ui.telFocusAvgInt->setEnabled(true);

        ui.button_TelTTDump->setEnabled(true);
        ui.button_TelFocusDump->setEnabled(true);
    }

} //updateGUI()

void offloadCtrl::on_button_zero_pressed()
{
    pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);

    ipFreq.setDevice("t2wOffloader");
    ipFreq.setName("zero");
    ipFreq.add(pcf::IndiElement("request"));

    ipFreq["request"] = pcf::IndiElement::On;

    sendNewProperty(ipFreq);
}

void offloadCtrl::on_button_TelTTDump_pressed()
{
    pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);

    ipFreq.setDevice("tcsi");
    ipFreq.setName("offlTT_dump");
    ipFreq.add(pcf::IndiElement("request"));

    ipFreq["request"] = pcf::IndiElement::On;

    sendNewProperty(ipFreq);
}

void offloadCtrl::on_button_TelFocusDump_pressed()
{
    pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);

    ipFreq.setDevice("tcsi");
    ipFreq.setName("offlF_dump");
    ipFreq.add(pcf::IndiElement("request"));

    ipFreq["request"] = pcf::IndiElement::On;

    sendNewProperty(ipFreq);
}

} //namespace xqt

#include "moc_offloadCtrl.cpp"

#endif
