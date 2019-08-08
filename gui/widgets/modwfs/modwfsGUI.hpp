
#include <QWidget>

#include "ui_modwfs.h"

#include "../../lib/multiIndi.hpp"


namespace xqt 
{
   
class modwfsGUI : public QWidget, public multiIndiSubscriber
{
   Q_OBJECT
   
protected:
   
   int m_appState {0};
   int m_modState {-1};
   int m_pwrState {-1};
   
   double m_modFrequency {0};
   double m_modRadius {0};
   
   double m_C1ofst {0};
   double m_C2ofst {0};
   
   
public:
   modwfsGUI( QWidget * Parent = 0, Qt::WindowFlags f = 0);
   
   ~modwfsGUI();
   
   int subscribe( multiIndiPublisher * publisher );
                                   
   int handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   int handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
protected:
   
   float m_stepSize {0.1};
   
public slots:
   void updateGUI();
   
   void on_buttonRest_pressed();
   void on_buttonSet_pressed();
   void on_buttonModulate_pressed();
   
   void on_buttonUp_pressed();
   void on_buttonDown_pressed();
   void on_buttonLeft_pressed();
   void on_buttonRight_pressed();
   
   void on_buttonScale_pressed();
   
   private:
      
      Ui::modwfs ui;
};

} //namespace xqt
   
