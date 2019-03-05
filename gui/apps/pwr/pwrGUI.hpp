
#include <QWidget>
#include <QPainter>
#include <qwt_dial_needle.h>

#include <xqwt_multi_dial.h>

#include "ui_pwr.h"

#include "../../lib/multiIndi.hpp"

#include "../../widgets/pwr/pwrDevice.hpp"
#include "../../widgets/pwr/pwrChannel.hpp"


namespace xqt 
{
   
class pwrGUI : public QWidget, public multiIndiSubscriber
{
   Q_OBJECT
   
protected:
   
   std::vector<xqt::pwrDevice *> m_devices;
   
   size_t n_ACpdus;
   std::vector<double> currents;
   std::vector<double> voltages;
   std::vector<double> frequencies;
   
public:
   pwrGUI( QWidget * Parent = 0, Qt::WindowFlags f = 0);
   
   ~pwrGUI();
   
   int subscribe( multiIndiPublisher * publisher );
                                   
   int handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   
   
public slots:
   void chChange( pcf::IndiProperty & ip );
   void updateGauges();
   
   private:
      
      QwtDialSimpleNeedle * pdu1Needle {nullptr};
      
      Ui::pwr ui;
};

} //namespace xqt
   
