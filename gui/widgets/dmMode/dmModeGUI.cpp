
#include "dmModeGUI.hpp"


#include <cmath>

#include <qframe.h>
#include <qpalette.h>
#include <QPaintEvent>
#include <QSlider>
#include <QMessageBox>

#include <mx/ioutils/stringUtils.hpp>

namespace xqt
{
   
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

dmModeGUI::dmModeGUI( std::string deviceName,
                      QWidget * Parent, 
                      Qt::WindowFlags f) : QDialog(Parent, f), m_deviceName{deviceName} 
{
   ui.setupUi(this);
   
   setWindowTitle(QString(deviceName.c_str()));
}
   
dmModeGUI::~dmModeGUI()
{

}

int dmModeGUI::subscribe( multiIndiPublisher * publisher )
{   
   publisher->subscribeProperty(this, m_deviceName, "current_amps");
   publisher->subscribeProperty(this, m_deviceName, "target_amps");
   
   return 0;
}
 
int dmModeGUI::handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/)
{
   if(ipRecv.getDevice() == m_deviceName)
   {
      return handleSetProperty(ipRecv);
   }
   
   return 0;
   
}

int dmModeGUI::handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/)
{
   if(ipRecv.getDevice() != m_deviceName) return 0;
   
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

   if(ipRecv.getName() == "dm")
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
   
   return 0;
   
}
 
int dmModeGUI::updateGUI( QLabel * currLabel,
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

int dmModeGUI::updateGUI( size_t ch,
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
   
void dmModeGUI::setChannel( size_t ch, float amp )
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_deviceName);
   ip.setName("target_amps");
   std::string elName = mx::ioutils::convertToString<size_t, 4, '0'>(ch);
   ip.add(pcf::IndiElement(elName));
   ip[elName] = amp;
   
   sendNewProperty(ip);
}
      
void dmModeGUI::on_modeSlider_0_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_0->setText(QString(nstr));
}

void dmModeGUI::on_modeSlider_1_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_1->setText(QString(nstr));
}

void dmModeGUI::on_modeSlider_2_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_2->setText(QString(nstr));
}

void dmModeGUI::on_modeSlider_3_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_3->setText(QString(nstr));
}

void dmModeGUI::on_modeSlider_4_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_4->setText(QString(nstr));
}

void dmModeGUI::on_modeSlider_5_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_5->setText(QString(nstr));
}

void dmModeGUI::on_modeSlider_6_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_6->setText(QString(nstr));
}

void dmModeGUI::on_modeSlider_7_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_7->setText(QString(nstr));
}

void dmModeGUI::on_modeSlider_8_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_8->setText(QString(nstr));
}

void dmModeGUI::on_modeSlider_9_sliderMoved( double sl)
{
   char nstr[8];
   snprintf(nstr, sizeof(nstr), "%0.3f", slider2amp(sl));
   ui.modeTarget_9->setText(QString(nstr));
}

void dmModeGUI::on_modeSlider_0_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_0->value());
   setChannel(0, amp);
}

void dmModeGUI::on_modeSlider_1_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_1->value());
   setChannel(1, amp);
}

void dmModeGUI::on_modeSlider_2_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_2->value());
   setChannel(2, amp);
}

void dmModeGUI::on_modeSlider_3_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_3->value());
   setChannel(3, amp);
}

void dmModeGUI::on_modeSlider_4_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_4->value());
   setChannel(4, amp);
}

void dmModeGUI::on_modeSlider_5_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_5->value());
   setChannel(5, amp);
}

void dmModeGUI::on_modeSlider_6_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_6->value());
   setChannel(6, amp);
}

void dmModeGUI::on_modeSlider_7_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_7->value());
   setChannel(7, amp);
}

void dmModeGUI::on_modeSlider_8_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_8->value());
   setChannel(8, amp);
}

void dmModeGUI::on_modeSlider_9_sliderReleased()
{
   float amp = slider2amp(ui.modeSlider_9->value());
   setChannel(9, amp);
}

void dmModeGUI::on_modeTarget_returnPressed( size_t ch,
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

void dmModeGUI::on_modeTarget_0_returnPressed()
{
   on_modeTarget_returnPressed(0, ui.modeTarget_0);
}

void dmModeGUI::on_modeTarget_1_returnPressed()
{
   on_modeTarget_returnPressed(1, ui.modeTarget_1);
}

void dmModeGUI::on_modeTarget_2_returnPressed()
{
   on_modeTarget_returnPressed(2, ui.modeTarget_2);
}

void dmModeGUI::on_modeTarget_3_returnPressed()
{
   on_modeTarget_returnPressed(3, ui.modeTarget_3);
}

void dmModeGUI::on_modeTarget_4_returnPressed()
{
   on_modeTarget_returnPressed(4, ui.modeTarget_4);
}

void dmModeGUI::on_modeTarget_5_returnPressed()
{
   on_modeTarget_returnPressed(5, ui.modeTarget_5);
}

void dmModeGUI::on_modeTarget_6_returnPressed()
{
   on_modeTarget_returnPressed(6, ui.modeTarget_6);
}

void dmModeGUI::on_modeTarget_7_returnPressed()
{
   on_modeTarget_returnPressed(7, ui.modeTarget_7);
}

void dmModeGUI::on_modeTarget_8_returnPressed()
{
   on_modeTarget_returnPressed(8, ui.modeTarget_8);
}

void dmModeGUI::on_modeTarget_9_returnPressed()
{
   on_modeTarget_returnPressed(9, ui.modeTarget_9);
}

void dmModeGUI::on_modeZero_0_pressed()
{
   setChannel(0, 0.0);
}

void dmModeGUI::on_modeZero_1_pressed()
{
   setChannel(1, 0.0);
}

void dmModeGUI::on_modeZero_2_pressed()
{
   setChannel(2, 0.0);
}

void dmModeGUI::on_modeZero_3_pressed()
{
   setChannel(3, 0.0);
}

void dmModeGUI::on_modeZero_4_pressed()
{
   setChannel(4, 0.0);
}

void dmModeGUI::on_modeZero_5_pressed()
{
   setChannel(5, 0.0);
}

void dmModeGUI::on_modeZero_6_pressed()
{
   setChannel(6, 0.0);
}

void dmModeGUI::on_modeZero_7_pressed()
{
   setChannel(7, 0.0);
}

void dmModeGUI::on_modeZero_8_pressed()
{
   setChannel(8, 0.0);
}

void dmModeGUI::on_modeZero_9_pressed()
{
   setChannel(9, 0.0);
}

void dmModeGUI::on_modeZero_these_pressed()
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

void dmModeGUI::on_modeZero_all_pressed()
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
