#ifndef dmMode_hpp
#define dmMode_hpp

#include <cmath>

#include <QMessageBox>

#include <mx/ioutils/stringUtils.hpp>

#include "ui_dmMode.h"

#include "../xWidgets/xWidget.hpp"


namespace xqt 
{
   
class dmMode : public xWidget
{
   Q_OBJECT
   
protected:
      
public:
   
   std::string m_deviceName;
   std::string m_dmName;
   std::string m_dmChannel;
   
   int m_maxmode {-1};
   
   dmMode( std::string & deviceName,
           QWidget * Parent = 0, 
           Qt::WindowFlags f = 0);
   
   ~dmMode();
   
   void subscribe();
                                   
   /// Called once the m_parent is connected.
   virtual void onConnect();
   
   /// Called when the m_parent disconnects.
   virtual void onDisconnect();
   
   virtual void handleDefProperty( const pcf::IndiProperty &ipRecv );
   
   virtual void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   int updateGUI( QLabel * currLabel,
                  QLineEdit * tgtLabel,
                  QwtSlider * slider,
                  float amp
                );
   
   int updateGUI( size_t ch,
                  float amp
                );
   
   void setChannel( size_t ch, float amp );
   
   void on_modeTarget_returnPressed( size_t ch,
                                     QLineEdit * modeTarget 
                                   );
public slots:
    
   void on_modeSlider_0_sliderMoved( double sl );
   void on_modeSlider_1_sliderMoved( double sl );
   void on_modeSlider_2_sliderMoved( double sl );
   void on_modeSlider_3_sliderMoved( double sl );
   void on_modeSlider_4_sliderMoved( double sl );
   void on_modeSlider_5_sliderMoved( double sl );
   void on_modeSlider_6_sliderMoved( double sl );
   void on_modeSlider_7_sliderMoved( double sl );
   void on_modeSlider_8_sliderMoved( double sl );
   void on_modeSlider_9_sliderMoved( double sl );

   void on_modeSlider_0_sliderReleased();
   void on_modeSlider_1_sliderReleased();
   void on_modeSlider_2_sliderReleased();
   void on_modeSlider_3_sliderReleased();
   void on_modeSlider_4_sliderReleased();
   void on_modeSlider_5_sliderReleased();
   void on_modeSlider_6_sliderReleased();
   void on_modeSlider_7_sliderReleased();
   void on_modeSlider_8_sliderReleased();
   void on_modeSlider_9_sliderReleased();
   
   void on_modeTarget_0_returnPressed();
   void on_modeTarget_1_returnPressed();
   void on_modeTarget_2_returnPressed();
   void on_modeTarget_3_returnPressed();
   void on_modeTarget_4_returnPressed();
   void on_modeTarget_5_returnPressed();
   void on_modeTarget_6_returnPressed();
   void on_modeTarget_7_returnPressed();
   void on_modeTarget_8_returnPressed();
   void on_modeTarget_9_returnPressed();
   
   void on_modeZero_0_pressed();
   void on_modeZero_1_pressed();
   void on_modeZero_2_pressed();
   void on_modeZero_3_pressed();
   void on_modeZero_4_pressed();
   void on_modeZero_5_pressed();
   void on_modeZero_6_pressed();
   void on_modeZero_7_pressed();
   void on_modeZero_8_pressed();
   void on_modeZero_9_pressed();
   
   void on_modeZero_these_pressed();
   void on_modeZero_all_pressed();
   
private:
            
   Ui::dmMode ui;
};

double slider2amp( double sl )
{
   static double a = -log(0.01);
   static double b = log(1.01)/a;
   
   double lamp = (b-log(1.01-fabs(sl))/a)/(1+b);
   
   if(lamp < 0) lamp = 0;
   else if(lamp > 1) lamp = 1;
   
   if(sl < 0) lamp *= -1;
   
   return lamp;
}

double amp2slider( double lamp )
{
   static double a = -log(0.01);
   static double b = log(1.01)/a;
   
   double sl = 1.01 - exp(-a*(fabs(lamp)*(1+b) - b));
   
   if(sl < 0) sl = 0;
   else if(sl > 1) sl = 1;
                          
   if(lamp < 0) sl *= -1;
   
   return sl;
}

dmMode::dmMode( std::string & deviceName,
                      QWidget * Parent, 
                      Qt::WindowFlags f) : xWidget(Parent, f), m_deviceName{deviceName} 
{
   ui.setupUi(this);
   
   ui.modeCurrent_0->setProperty("isStatus", true);
   ui.modeCurrent_1->setProperty("isStatus", true);
   ui.modeCurrent_2->setProperty("isStatus", true);
   ui.modeCurrent_3->setProperty("isStatus", true);
   ui.modeCurrent_4->setProperty("isStatus", true);
   ui.modeCurrent_5->setProperty("isStatus", true);
   ui.modeCurrent_6->setProperty("isStatus", true);
   ui.modeCurrent_7->setProperty("isStatus", true);
   ui.modeCurrent_8->setProperty("isStatus", true);
   ui.modeCurrent_9->setProperty("isStatus", true);

   setWindowTitle(QString(deviceName.c_str()));

   onDisconnect();
}
   
dmMode::~dmMode()
{

}

void dmMode::subscribe()
{   
   if(m_parent == nullptr) return;

   m_parent->addSubscriberProperty(this, m_deviceName, "current_amps");
   m_parent->addSubscriberProperty(this, m_deviceName, "target_amps");
   m_parent->addSubscriberProperty(this, m_deviceName, "dm");
   return;
}
 
void dmMode::onConnect()
{
   ui.title->setEnabled(true);
   ui.channel->setEnabled(true);
   
   ui.modeName_0->setEnabled(true);
   ui.modeSlider_0->setEnabled(true);
   ui.modeTarget_0->setEnabled(true);
   ui.modeCurrent_0->setEnabled(true);
   ui.modeZero_0->setEnabled(true);
   
   ui.modeName_1->setEnabled(true);
   ui.modeSlider_1->setEnabled(true);
   ui.modeTarget_1->setEnabled(true);
   ui.modeCurrent_1->setEnabled(true);
   ui.modeZero_1->setEnabled(true);
   
   ui.modeName_2->setEnabled(true);
   ui.modeSlider_2->setEnabled(true);
   ui.modeTarget_2->setEnabled(true);
   ui.modeCurrent_2->setEnabled(true);
   ui.modeZero_2->setEnabled(true);
   
   ui.modeName_3->setEnabled(true);
   ui.modeSlider_3->setEnabled(true);
   ui.modeTarget_3->setEnabled(true);
   ui.modeCurrent_3->setEnabled(true);
   ui.modeZero_3->setEnabled(true);
   
   ui.modeName_4->setEnabled(true);
   ui.modeSlider_4->setEnabled(true);
   ui.modeTarget_4->setEnabled(true);
   ui.modeCurrent_4->setEnabled(true);
   ui.modeZero_4->setEnabled(true);
   
   ui.modeName_5->setEnabled(true);
   ui.modeSlider_5->setEnabled(true);
   ui.modeTarget_5->setEnabled(true);
   ui.modeCurrent_5->setEnabled(true);
   ui.modeZero_5->setEnabled(true);
   
   ui.modeName_6->setEnabled(true);
   ui.modeSlider_6->setEnabled(true);
   ui.modeTarget_6->setEnabled(true);
   ui.modeCurrent_6->setEnabled(true);
   ui.modeZero_6->setEnabled(true);
   
   ui.modeName_7->setEnabled(true);
   ui.modeSlider_7->setEnabled(true);
   ui.modeTarget_7->setEnabled(true);
   ui.modeCurrent_7->setEnabled(true);
   ui.modeZero_7->setEnabled(true);
   
   ui.modeName_8->setEnabled(true);
   ui.modeSlider_8->setEnabled(true);
   ui.modeTarget_8->setEnabled(true);
   ui.modeCurrent_8->setEnabled(true);
   ui.modeZero_8->setEnabled(true);
   
   ui.modeName_9->setEnabled(true);
   ui.modeSlider_9->setEnabled(true);
   ui.modeTarget_9->setEnabled(true);
   ui.modeCurrent_9->setEnabled(true);
   ui.modeZero_9->setEnabled(true);
   
   ui.modeZero_these->setEnabled(true);
   ui.modeZero_all->setEnabled(true);
   
   setWindowTitle(QString(m_deviceName.c_str()));
   
}
   
void dmMode::onDisconnect()
{
   ui.title->setEnabled(false);
   ui.channel->setEnabled(false);
   
   ui.modeName_0->setEnabled(false);
   ui.modeSlider_0->setEnabled(false);
   ui.modeTarget_0->setEnabled(false);
   ui.modeCurrent_0->setEnabled(false);
   ui.modeZero_0->setEnabled(false);
   
   ui.modeName_1->setEnabled(false);
   ui.modeSlider_1->setEnabled(false);
   ui.modeTarget_1->setEnabled(false);
   ui.modeCurrent_1->setEnabled(false);
   ui.modeZero_1->setEnabled(false);
   
   ui.modeName_2->setEnabled(false);
   ui.modeSlider_2->setEnabled(false);
   ui.modeTarget_2->setEnabled(false);
   ui.modeCurrent_2->setEnabled(false);
   ui.modeZero_2->setEnabled(false);
   
   ui.modeName_3->setEnabled(false);
   ui.modeSlider_3->setEnabled(false);
   ui.modeTarget_3->setEnabled(false);
   ui.modeCurrent_3->setEnabled(false);
   ui.modeZero_3->setEnabled(false);
   
   ui.modeName_4->setEnabled(false);
   ui.modeSlider_4->setEnabled(false);
   ui.modeTarget_4->setEnabled(false);
   ui.modeCurrent_4->setEnabled(false);
   ui.modeZero_4->setEnabled(false);
   
   ui.modeName_5->setEnabled(false);
   ui.modeSlider_5->setEnabled(false);
   ui.modeTarget_5->setEnabled(false);
   ui.modeCurrent_5->setEnabled(false);
   ui.modeZero_5->setEnabled(false);
   
   ui.modeName_6->setEnabled(false);
   ui.modeSlider_6->setEnabled(false);
   ui.modeTarget_6->setEnabled(false);
   ui.modeCurrent_6->setEnabled(false);
   ui.modeZero_6->setEnabled(false);
   
   ui.modeName_7->setEnabled(false);
   ui.modeSlider_7->setEnabled(false);
   ui.modeTarget_7->setEnabled(false);
   ui.modeCurrent_7->setEnabled(false);
   ui.modeZero_7->setEnabled(false);
   
   ui.modeName_8->setEnabled(false);
   ui.modeSlider_8->setEnabled(false);
   ui.modeTarget_8->setEnabled(false);
   ui.modeCurrent_8->setEnabled(false);
   ui.modeZero_8->setEnabled(false);
   
   ui.modeName_9->setEnabled(false);
   ui.modeSlider_9->setEnabled(false);
   ui.modeTarget_9->setEnabled(false);
   ui.modeCurrent_9->setEnabled(false);
   ui.modeZero_9->setEnabled(false);
   
   ui.modeZero_these->setEnabled(false);
   ui.modeZero_all->setEnabled(false);
   
   setWindowTitle(QString(m_deviceName.c_str()) + QString(" (disconnected)"));
}
   
   
void dmMode::handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/)
{
   if(ipRecv.getDevice() == m_deviceName)
   {  
      return handleSetProperty(ipRecv);
   }
   
   return;
   
}

void dmMode::handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/)
{
   if(ipRecv.getDevice() != m_deviceName) return;
   
   if(ipRecv.getName() == "current_amps")
   {
      std::string elName;
      for(size_t n=0; n<10; ++n)
      {
         elName = mx::ioutils::convertToString<size_t, 4, '0'>(n);
      
         if(ipRecv.find(elName))
         {
            float amp = ipRecv[elName].get<float>();
            updateGUI(n, amp);
         }
      }
      
      if(m_maxmode == -1)
      {
         std::string elName;
         for(int n=0; n<9999; ++n)
         {
            elName = mx::ioutils::convertToString<size_t, 4, '0'>(n);
      
            if(ipRecv.find(elName))
            {
               if(n > m_maxmode) m_maxmode = n;
            }
         }
      }
         
   }
   else if(ipRecv.getName() == "dm")
   {
      if(ipRecv.find("name"))
      {
         QString name = ipRecv["name"].get<std::string>().c_str();
         name += " DM modes";
         ui.title->setText(name);
         setWindowTitle(name);
      }
      if(ipRecv.find("channel"))
      {
         QString channel = ipRecv["channel"].get<std::string>().c_str();
         ui.channel->setText(channel);
      }
   }
   
   return;
   
}
 
int dmMode::updateGUI( QLabel * currLabel,
                       QLineEdit * tgtLabel,
                       QwtSlider * slider,
                       float amp
                     )
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", amp);
   
   currLabel->setText(QString(nstr));
   tgtLabel->setText(QString(""));
   
   slider->setValue(amp2slider(amp));
   
   return 0;
}

int dmMode::updateGUI( size_t ch,
                          float amp
                        )
{
   switch(ch)
   {
      case 0:
         return updateGUI( ui.modeCurrent_0, ui.modeTarget_0, ui.modeSlider_0, amp );
      case 1:
         return updateGUI( ui.modeCurrent_1, ui.modeTarget_1, ui.modeSlider_1, amp );
      case 2:
         return updateGUI( ui.modeCurrent_2, ui.modeTarget_2, ui.modeSlider_2, amp );
      case 3:
         return updateGUI( ui.modeCurrent_3, ui.modeTarget_3, ui.modeSlider_3, amp );
      case 4:
         return updateGUI( ui.modeCurrent_4, ui.modeTarget_4, ui.modeSlider_4, amp );
      case 5:
         return updateGUI( ui.modeCurrent_5, ui.modeTarget_5, ui.modeSlider_5, amp );
      case 6:
         return updateGUI( ui.modeCurrent_6, ui.modeTarget_6, ui.modeSlider_6, amp );
      case 7:
         return updateGUI( ui.modeCurrent_7, ui.modeTarget_7, ui.modeSlider_7, amp );
      case 8:
         return updateGUI( ui.modeCurrent_8, ui.modeTarget_8, ui.modeSlider_8, amp );
      case 9:
         return updateGUI( ui.modeCurrent_9, ui.modeTarget_9, ui.modeSlider_9, amp );
      default:
         std::cerr << "bad channel number in updateGUI\n";
         return -1;
   }
   
}
   
void dmMode::setChannel( size_t ch, float amp )
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_deviceName);
   ip.setName("target_amps");
   std::string elName = mx::ioutils::convertToString<size_t, 4, '0'>(ch);
   ip.add(pcf::IndiElement(elName));
   ip[elName] = amp;
   
   sendNewProperty(ip);
}
      
void dmMode::on_modeSlider_0_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_0->setText(QString(nstr));
}

void dmMode::on_modeSlider_1_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_1->setText(QString(nstr));
}

void dmMode::on_modeSlider_2_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_2->setText(QString(nstr));
}

void dmMode::on_modeSlider_3_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_3->setText(QString(nstr));
}

void dmMode::on_modeSlider_4_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_4->setText(QString(nstr));
}

void dmMode::on_modeSlider_5_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_5->setText(QString(nstr));
}

void dmMode::on_modeSlider_6_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_6->setText(QString(nstr));
}

void dmMode::on_modeSlider_7_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_7->setText(QString(nstr));
}

void dmMode::on_modeSlider_8_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_8->setText(QString(nstr));
}

void dmMode::on_modeSlider_9_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_9->setText(QString(nstr));
}

void dmMode::on_modeSlider_0_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_0->value());
   setChannel(0, amp);
}

void dmMode::on_modeSlider_1_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_1->value());
   setChannel(1, amp);
}

void dmMode::on_modeSlider_2_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_2->value());
   setChannel(2, amp);
}

void dmMode::on_modeSlider_3_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_3->value());
   setChannel(3, amp);
}

void dmMode::on_modeSlider_4_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_4->value());
   setChannel(4, amp);
}

void dmMode::on_modeSlider_5_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_5->value());
   setChannel(5, amp);
}

void dmMode::on_modeSlider_6_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_6->value());
   setChannel(6, amp);
}

void dmMode::on_modeSlider_7_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_7->value());
   setChannel(7, amp);
}

void dmMode::on_modeSlider_8_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_8->value());
   setChannel(8, amp);
}

void dmMode::on_modeSlider_9_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_9->value());
   setChannel(9, amp);
}

void dmMode::on_modeTarget_returnPressed( size_t ch,
                                             QLineEdit * modeTarget 
                                           )
{
   bool ok = false;
   float amp = modeTarget->text().toFloat(&ok);
   
   if(!ok)
   {
      QMessageBox qbox( QMessageBox::Warning, "Invalid Input", "input not a valid number", QMessageBox::Ok, this);
      qbox.exec();
      return;
   }
   
   if( !( amp >= -1 && amp <= 1) )
   {
      QMessageBox qbox( QMessageBox::Warning, "Invalid Input", "input is out of bounds (-1 to 1)", QMessageBox::Ok, this);
      qbox.exec();
      return;
   }
   
   setChannel(ch, amp);
}

void dmMode::on_modeTarget_0_returnPressed()
{
   on_modeTarget_returnPressed(0, ui.modeTarget_0);
}

void dmMode::on_modeTarget_1_returnPressed()
{
   on_modeTarget_returnPressed(1, ui.modeTarget_1);
}

void dmMode::on_modeTarget_2_returnPressed()
{
   on_modeTarget_returnPressed(2, ui.modeTarget_2);
}

void dmMode::on_modeTarget_3_returnPressed()
{
   on_modeTarget_returnPressed(3, ui.modeTarget_3);
}

void dmMode::on_modeTarget_4_returnPressed()
{
   on_modeTarget_returnPressed(4, ui.modeTarget_4);
}

void dmMode::on_modeTarget_5_returnPressed()
{
   on_modeTarget_returnPressed(5, ui.modeTarget_5);
}

void dmMode::on_modeTarget_6_returnPressed()
{
   on_modeTarget_returnPressed(6, ui.modeTarget_6);
}

void dmMode::on_modeTarget_7_returnPressed()
{
   on_modeTarget_returnPressed(7, ui.modeTarget_7);
}

void dmMode::on_modeTarget_8_returnPressed()
{
   on_modeTarget_returnPressed(8, ui.modeTarget_8);
}

void dmMode::on_modeTarget_9_returnPressed()
{
   on_modeTarget_returnPressed(9, ui.modeTarget_9);
}

void dmMode::on_modeZero_0_pressed()
{
   setChannel(0, 0.0);
}

void dmMode::on_modeZero_1_pressed()
{
   setChannel(1, 0.0);
}

void dmMode::on_modeZero_2_pressed()
{
   setChannel(2, 0.0);
}

void dmMode::on_modeZero_3_pressed()
{
   setChannel(3, 0.0);
}

void dmMode::on_modeZero_4_pressed()
{
   setChannel(4, 0.0);
}

void dmMode::on_modeZero_5_pressed()
{
   setChannel(5, 0.0);
}

void dmMode::on_modeZero_6_pressed()
{
   setChannel(6, 0.0);
}

void dmMode::on_modeZero_7_pressed()
{
   setChannel(7, 0.0);
}

void dmMode::on_modeZero_8_pressed()
{
   setChannel(8, 0.0);
}

void dmMode::on_modeZero_9_pressed()
{
   setChannel(9, 0.0);
}

void dmMode::on_modeZero_these_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_deviceName);
   ip.setName("target_amps");
   
   for(size_t n =0; n <= 9; ++n)
   {
      std::string elName = mx::ioutils::convertToString<size_t, 4, '0'>(n);
      ip.add(pcf::IndiElement(elName));
      ip[elName] = 0.0;
   }
   
   sendNewProperty(ip);
}

void dmMode::on_modeZero_all_pressed()
{
   if(m_maxmode < 0) return;
   
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_deviceName);
   ip.setName("target_amps");
   
   for(int n =0; n <= m_maxmode; ++n)
   {
      std::string elName = mx::ioutils::convertToString<int, 4, '0'>(n);
      ip.add(pcf::IndiElement(elName));
      ip[elName] = 0.0;
   }
   
   sendNewProperty(ip);
}

} //namespace xqt
   
#include "moc_dmMode.cpp"

#endif

