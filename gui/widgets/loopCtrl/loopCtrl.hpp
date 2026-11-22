
#ifndef loopCtrl_hpp
#define loopCtrl_hpp


#include <mutex>

#include <cmath>
#include <QToolTip>

#include "ui_loopCtrl.h"

#include "../xWidgets/xWidget.hpp"
#include "../xWidgets/gainCtrl.hpp"
#include "../xWidgets/statusEntry.hpp"

namespace xqt
{

class loopCtrl : public xWidget
{
   Q_OBJECT

protected:

   std::string m_procName;

   std::string m_loopName;
   std::string m_loopNumber;

   std::string m_goptName;

   std::string m_gainCtrl;

   std::string m_appState;

   double m_gain {0.0};;
   double m_gainScale {0.01};

   double m_multcoeff {1};
   double m_multcoeffScale {0.001};

   bool m_loopState {false};
   bool m_loopWaiting {false}; //indicates slider is waiting on loop_state update to be re-enabled

   bool m_procState {false};

   bool m_flatVectorAvailable {false}; ///< Whether the gain controller has advertised the flat-vector command.

   std::vector<int> m_modes;
   std::vector<gainCtrl *> m_blockCtrls;
   std::mutex m_blockMutex;

   QTimer * m_updateTimer {nullptr};

public:
   loopCtrl( std::string & procName,
             QWidget * Parent = 0,
             Qt::WindowFlags f = Qt::WindowFlags()
           );

   ~loopCtrl();

   void subscribe();

   void onConnect();

   void onDisconnect();

   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

   void handleDelProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

   void sendNewGain(double ng);
   void sendNewMultCoeff(double nm);

   void setEnableDisable( bool tf,
                          bool all = true
                        );

public slots:
   void updateGUI();

   void on_button_LoopZero_pressed();

   /// Request that every mode use the gain shown in the first block.
   void on_button_flatVector_clicked();

   void on_button_zeroall_pressed();

   void setupBlocks(int nB);

   void on_button_autoGainDump_pressed();

signals:

   void doUpdateGUI();

   void blocksChanged(int nB);

private:

   Ui::loopCtrl ui;
};

loopCtrl::loopCtrl( std::string & procName,
                    QWidget * Parent,
                    Qt::WindowFlags f) : xWidget(Parent, f)
{
    ui.setupUi(this);

    connect(this, SIGNAL(blocksChanged(int)), this, SLOT(setupBlocks(int)));

    m_procName = procName + "loop";

    setWindowTitle(QString(m_procName.c_str()));
    ui.label_loop_state->setProperty("isStatus", true);

    ui.slider_loop->setup(m_procName, "loop_state", "toggle", "");
    ui.slider_loop->setStretch(0,0,10, true, true);

    ui.gainCtrl->setup(m_procName, "loop_gain", "Gain", -1, -1);

    ui.mcCtrl->setup(m_procName, "loop_multcoeff", "Mult. Coef.", -1, -1);
    ui.mcCtrl->makeMultCoeffCtrl();

    m_gainCtrl = procName + "gainctrl";

    m_goptName = procName + "gopt";

    ui.slider_autogain->setup(m_goptName, "update_auto", "toggle", "");
    ui.slider_autogain->setStretch(0,0,10, true, true);

    ui.slider_trackog->setup(m_goptName, "track_optical_gain", "toggle", "");
    ui.slider_trackog->setStretch(0,0,10, true, true);

    ui.opticalGain->setup(m_goptName, "opticalGain", statusEntry::FLOAT, "", "");


    setXwFont(ui.label_LoopName);
    setXwFont(ui.label_loop);
    setXwFont(ui.label_loop_state);
    setXwFont(ui.button_LoopZero);

    setXwFont(ui.label_autogain);
    setXwFont(ui.label_trackog);
    setXwFont(ui.button_autoGainDump);

    setXwFont( ui.button_flatVector );
    setXwFont(ui.button_zeroall);


    m_updateTimer = new QTimer;

    connect(m_updateTimer, SIGNAL(timeout()), this, SLOT(updateGUI()));

    m_updateTimer->start(250);

    connect(this, SIGNAL(doUpdateGUI()), this, SLOT(updateGUI()));

    onDisconnect();
}

loopCtrl::~loopCtrl()
{
}

void loopCtrl::subscribe()
{
   if(!m_parent) return;


   m_parent->addSubscriberProperty(this, m_procName, "fsm");
   m_parent->addSubscriberProperty(this, m_procName, "loop");
   m_parent->addSubscriberProperty(this, m_procName, "loop_gain");
   m_parent->addSubscriberProperty(this, m_procName, "loop_multcoeff");
   m_parent->addSubscriberProperty(this, m_procName, "loop_processes");

   m_parent->addSubscriber(ui.slider_loop);
   m_parent->addSubscriberProperty(this, m_procName, "loop_state");

   m_parent->addSubscriber(ui.slider_autogain);
   m_parent->addSubscriber(ui.slider_trackog);

   m_parent->addSubscriber(ui.opticalGain);

   m_parent->addSubscriberProperty(this, m_gainCtrl, "modes");
   m_parent->addSubscriberProperty( this, m_gainCtrl, "flat_vector" );

   m_parent->addSubscriber(ui.gainCtrl);
   m_parent->addSubscriber(ui.mcCtrl);


   std::lock_guard<std::mutex> lock(m_blockMutex);
   for(size_t n = 0; n < m_blockCtrls.size(); ++n)
   {
      if(m_blockCtrls[n]) m_parent->addSubscriber(m_blockCtrls[n]);
   }

   return;
}

void loopCtrl::onConnect()
{
    setWindowTitle(QString(m_procName.c_str()));

    ui.slider_loop->onConnect();
    ui.slider_autogain->onConnect();
    ui.slider_trackog->onConnect();

    ui.opticalGain->onConnect();

   //xWidget::onConnect();
   ui.gainCtrl->onConnect();
   ui.mcCtrl->onConnect();


   std::lock_guard<std::mutex> lock(m_blockMutex);
   for(size_t n = 0; n < m_blockCtrls.size(); ++n)
   {
      if(m_blockCtrls[n]) m_blockCtrls[n]->onConnect();
   }

}

void loopCtrl::onDisconnect()
{
    std::string tit = m_procName + " (disconnected)";
    setWindowTitle(QString(tit.c_str()));

    ui.slider_loop->onDisconnect();
    ui.slider_autogain->onDisconnect();
    ui.slider_trackog->onDisconnect();

    ui.opticalGain->onDisconnect();

    m_flatVectorAvailable = false;
    setEnableDisable( false );

    //xWidget::onDisconnect();
    ui.gainCtrl->onDisconnect();
    ui.mcCtrl->onDisconnect();

    std::lock_guard<std::mutex> lock( m_blockMutex );
    for( size_t n = 0; n < m_blockCtrls.size(); ++n )
    {
        if( m_blockCtrls[n] )
        {
            m_blockCtrls[n]->onDisconnect();
        }
    }
}

void loopCtrl::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void loopCtrl::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getDevice() != m_procName && ipRecv.getDevice() != m_gainCtrl) return;

   if(ipRecv.getDevice() == m_procName)
   {
   if(ipRecv.getName() == "fsm")
   {
      if(ipRecv.find("state"))
      {
         m_appState = ipRecv["state"].get<std::string>();
      }
   }
   else if(ipRecv.getName() == "loop")
   {
      if(ipRecv.find("name"))
      {
         m_loopName = ipRecv["name"].get<std::string>();
      }

      if(ipRecv.find("number"))
      {
         m_loopNumber = ipRecv["number"].get<std::string>();
      }

      std::string label = m_loopName + " (aol" + m_loopNumber + ")";
      ui.label_LoopName->setText(label.c_str());
   }
   else if(ipRecv.getName() == "loop_gain")
   {
      if(ipRecv.find("current"))
      {
         m_gain = ipRecv["current"].get<float>();
      }
   }
   else if(ipRecv.getName() == "loop_multcoeff")
   {
      if(ipRecv.find("current"))
      {
         m_multcoeff = ipRecv["current"].get<float>();
      }
   }
   else if(ipRecv.getName() == "loop_state")
   {
      if(ipRecv.find("toggle"))
      {
         if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
         {
            m_loopState = true;
         }
         else
         {
            m_loopState = false;
         }

         m_loopWaiting = false;
      }
   }
   }
   if(ipRecv.getDevice() == m_gainCtrl)
   {
       if( ipRecv.getName() == "flat_vector" )
       {
           m_flatVectorAvailable = true;
       }
       else if( ipRecv.getName() == "modes" && ipRecv.find( "blocks" ) )
       {
           int nB = ipRecv["blocks"].get<int>();
           if( nB < 0 )
           {
               return;
           }

           std::vector<int> modes;
           for( int n = 0; n < nB; ++n )
           {
               char block[24];
               snprintf( block, sizeof( block ), "block%02d", n );
               if( !ipRecv.find( block ) )
               {
                   return;
               }

               modes.push_back( ipRecv[block].get<int>() );
           }

           if( modes != m_modes )
           {
               m_modes = std::move( modes );
               emit blocksChanged( nB );
           }
       }
   }


   emit doUpdateGUI();
}

void loopCtrl::handleDelProperty( const pcf::IndiProperty & ipRecv)
{
    if( ipRecv.getDevice() != m_gainCtrl )
    {
        return;
    }

    const std::string &name           = ipRecv.getName();
    bool               deviceDeleted  = name.empty() || name == "fsm";
    bool               modesDeleted   = deviceDeleted || name == "modes";
    bool               commandDeleted = deviceDeleted || name == "flat_vector";
    if( !modesDeleted && !commandDeleted )
    {
        return;
    }

    if( commandDeleted )
    {
        m_flatVectorAvailable = false;
    }

    if( modesDeleted )
    {
        m_modes.clear();
        emit blocksChanged( 0 );
    }

    emit doUpdateGUI();
}

void loopCtrl::setEnableDisable( bool tf,
                                 bool all
                               )
{
   if(all)
   {
      ui.slider_loop->setEnabled(tf);
      ui.label_loop_state->setEnabled(tf);
   }

   ui.gainCtrl->setEnabled(tf);
   ui.mcCtrl->setEnabled(tf);

   ui.slider_autogain->setEnabled(tf);
   ui.slider_trackog->setEnabled(tf);

   ui.opticalGain->setEnabled(tf);


   std::lock_guard<std::mutex> lock(m_blockMutex);
   bool                        flatVectorEnabled = tf && m_flatVectorAvailable && !m_blockCtrls.empty();
   ui.button_flatVector->setEnabled( flatVectorEnabled );

   QString flatVectorTip;
   if( !m_flatVectorAvailable )
   {
       flatVectorTip = tr( "Waiting for the Flat Vector command from the gain controller. "
                           "The server must run the updated userGainCtrl." );
   }
   else if( m_blockCtrls.empty() )
   {
       flatVectorTip = tr( "Waiting for the gain controller to report its mode blocks." );
   }
   else
   {
       flatVectorTip = tr( "Apply the value in the first block gain field to every mode." );
   }

   if( flatVectorTip != ui.button_flatVector->toolTip() )
   {
       ui.button_flatVector->setToolTip( flatVectorTip );
   }

   for(size_t n = 0; n < m_blockCtrls.size(); ++n)
   {
      if(m_blockCtrls[n]) m_blockCtrls[n]->setEnabled(tf);
   }

}

void loopCtrl::updateGUI()
{
      setEnableDisable(true, false);

      if(!m_loopWaiting)
      {
         ui.label_loop_state->setEnabled(true);
         ui.slider_loop->setEnabled(true);
      }

      if(m_loopState)
      {
         ui.label_loop_state->setText("CLOSED");
      }
      else
      {
         ui.label_loop_state->setText("OPEN");
      }


} //updateGUI()

void loopCtrl::on_button_LoopZero_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);

   ipFreq.setDevice(m_procName);
   ipFreq.setName("loop_zero");
   ipFreq.add(pcf::IndiElement("request"));

   ipFreq["request"] = pcf::IndiElement::On;

   sendNewProperty(ipFreq);
}

void loopCtrl::on_button_flatVector_clicked()
{
    QString input;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_blockMutex );
        if( !m_flatVectorAvailable || m_blockCtrls.empty() )
        {
            return;
        }

        input = m_blockCtrls.front()->inputText();
    }

    bool  ok   = false;
    float gain = input.toFloat( &ok );
    if( !ok || !std::isfinite( gain ) || gain < 0 )
    {
        QToolTip::showText( ui.button_flatVector->mapToGlobal( ui.button_flatVector->rect().bottomLeft() ),
                            tr( "Enter a non-negative gain in the first block." ),
                            ui.button_flatVector );
        return;
    }

    pcf::IndiProperty ip( pcf::IndiProperty::Number );
    ip.setDevice( m_gainCtrl );
    ip.setName( "flat_vector" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = gain;

    sendNewProperty( ip );
}

void loopCtrl::on_button_zeroall_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);

   ipFreq.setDevice(m_gainCtrl);
   ipFreq.setName("zero_all");
   ipFreq.add(pcf::IndiElement("request"));

   ipFreq["request"] = pcf::IndiElement::On;

   sendNewProperty(ipFreq);
}

void loopCtrl::setupBlocks(int nB)
{
    std::lock_guard<std::mutex> lock(m_blockMutex);

    for(size_t n =0; n < m_blockCtrls.size(); ++n)
    {
        if( m_blockCtrls[n] == nullptr )
        {
            continue;
        }

        if( m_parent )
        {
            m_parent->unsubscribe( m_blockCtrls[n] );
        }

        ui.horizontalLayout_2->removeWidget(m_blockCtrls[n]);
        m_blockCtrls[n]->deleteLater();
    }

    m_blockCtrls.clear();


    m_blockCtrls.resize(nB, nullptr); //I think this will call the destructor

    int modeTot = 0;
    for(int n = 0; n < nB; ++n)
    {
        char str[16];
        snprintf(str, sizeof(str), "%02d", n);
        modeTot += m_modes[n];
        m_blockCtrls[n] = new gainCtrl(m_gainCtrl, std::string("block") + str + "_gain", "", m_modes[n], modeTot);
        ui.horizontalLayout_2->addWidget(m_blockCtrls[n]);
        if(m_parent) m_parent->addSubscriber(m_blockCtrls[n]);
    }
}

void loopCtrl::on_button_autoGainDump_pressed()
{
    pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);

    ipFreq.setDevice(m_goptName);
    ipFreq.setName("update_dump");
    ipFreq.add(pcf::IndiElement("request"));

    ipFreq["request"] = pcf::IndiElement::On;

    sendNewProperty(ipFreq);
}

} //namespace xqt

#include "moc_loopCtrl.cpp"

#endif
