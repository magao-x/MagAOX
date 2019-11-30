
#include <QWidget>
#include <QMutex>
#include <QTimer>

#include "ui_offloadCtrl.h"

#include "../../lib/multiIndi.hpp"


namespace xqt 
{
   
class offloadCtrl : public QDialog, public multiIndiSubscriber
{
   Q_OBJECT
   
protected:
   
   std::string m_t2wFsmState;
   
   QMutex m_mutex;
   

   bool m_t2wOffloadingEnabled {false};
   double m_gain;
   double m_leak;
   int m_navg {0};
   
public:
   offloadCtrl( QWidget * Parent = 0, Qt::WindowFlags f = 0);
   
   ~offloadCtrl();
   
   int subscribe( multiIndiPublisher * publisher );
                                   
   int handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   int handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   
public slots:
   void updateGUI();
   
   void on_t2w_enable_pressed();
   void on_t2w_zero_pressed();
   
   void on_t2w_gain_minus_pressed();
   void on_t2w_gain_plus_pressed();
   void on_t2w_gain_edit_editingFinished();
      
   void on_t2w_leak_minus_pressed();
   void on_t2w_leak_plus_pressed();
   void on_t2w_leak_edit_editingFinished();
   
   void on_t2w_avg_spin_valueChanged(int v);
   
private:
     
   Ui::offloadCtrl ui;
};

} //namespace xqt
   
