
#include <QDialog>

#include "ui_dmCtrl.h"

#include "../../lib/multiIndi.hpp"


namespace xqt 
{
   
class dmCtrl : public QDialog, public multiIndiSubscriber
{
   Q_OBJECT
   
protected:
   
   int m_appState {0};
   
   std::string m_dmName;
   std::string m_shmimName;
   std::string m_flatName;
   std::string m_flatShmim;
   std::string m_testName;
   std::string m_testShmim;
   
   
public:
   dmCtrl( std::string & dmName,
           QWidget * Parent = 0, 
           Qt::WindowFlags f = 0
         );
   
   ~dmCtrl();
   
   int subscribe( multiIndiPublisher * publisher );
                                   
   int handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   int handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   
public slots:
   void updateGUI();
   
   void on_buttonInit_pressed();
   void on_buttonZero_pressed();
   void on_buttonRelease_pressed();
   
   void on_buttonLoadFlat_pressed();
   void on_buttonSetFlat_pressed();
   void on_buttonZeroFlat_pressed();
   
   void on_buttonLoadTest_pressed();
   void on_buttonSetTest_pressed();
   void on_buttonZeroTest_pressed();
   
private:
     
   Ui::dmCtrl ui;
};

} //namespace xqt
   
