
#include <QDialog>

#include "ui_dmMode.h"

#include "../../lib/multiIndi.hpp"


namespace xqt 
{
   
class dmModeGUI : public QDialog, public multiIndiSubscriber
{
   Q_OBJECT
   
protected:
      
public:
   
   std::string m_deviceName;
   std::string m_dmName;
   std::string m_dmChannel;
   
   
   dmModeGUI( std::string deviceName,
              QWidget * Parent = 0, 
              Qt::WindowFlags f = 0);
   
   ~dmModeGUI();
   
   int subscribe( multiIndiPublisher * publisher );
                                   
   virtual int handleDefProperty( const pcf::IndiProperty &ipRecv );
   
   virtual int handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   int updateGUI( QLabel * currLabel,
                  QLineEdit * tgtLabel,
                  QwtSlider * slider,
                float amp
              );
   
   int updateGUI( size_t ch,
                  float amp
                );
   
public slots:

   void setChannel( size_t ch, float amp );
   
      
   void on_modeSlider_0_sliderMoved( double amp );
   void on_modeSlider_1_sliderMoved( double amp );
   void on_modeSlider_2_sliderMoved( double amp );
   void on_modeSlider_3_sliderMoved( double amp );
   void on_modeSlider_4_sliderMoved( double amp );
   void on_modeSlider_5_sliderMoved( double amp );
   void on_modeSlider_6_sliderMoved( double amp );
   void on_modeSlider_7_sliderMoved( double amp );
   void on_modeSlider_8_sliderMoved( double amp );
   void on_modeSlider_9_sliderMoved( double amp );
   void on_modeSlider_10_sliderMoved( double amp );

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
   void on_modeSlider_10_sliderReleased();
   
   void on_modeTarget_returnPressed( size_t ch,
                                     QLineEdit * modeTarget 
                                   );
   
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
   void on_modeTarget_10_returnPressed();
   
signals:
   
private:
      
      
   Ui::dmMode ui;
};

} //namespace xqt
   
