
#include <QWidget>

#include "ui_ttmpupil.h"

#include "../../lib/multiIndi.hpp"


namespace xqt 
{
   
class ttmpupilGUI : public QWidget, public multiIndiSubscriber
{
   Q_OBJECT
   
protected:
   
   std::string m_appState;
   
   double m_pos1 {0};
   double m_pos2 {0};
   
   
public:
   ttmpupilGUI( QWidget * Parent = 0, Qt::WindowFlags f = 0);
   
   ~ttmpupilGUI();
   
   int subscribe( multiIndiPublisher * publisher );
                                   
   int handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   int handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
protected:
   
   float m_stepSize {0.1};
   
public slots:
   void updateGUI();
   
   void on_buttonRest_pressed();
   void on_buttonSet_pressed();
   
   void on_buttonUp_pressed();
   void on_buttonDown_pressed();
   void on_buttonLeft_pressed();
   void on_buttonRight_pressed();
   
   void on_buttonScale_pressed();
   
private:
     
   Ui::ttmpupil ui;
};

} //namespace xqt
   
