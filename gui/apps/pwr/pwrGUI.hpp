
#include <mutex>

#include <QWidget>
#include <QPainter>
#include <qwt_dial_needle.h>

#include <xqwt_multi_dial.h>

#include "ui_pwr.h"

#include "../../lib/multiIndiSubscriber.hpp"

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
   
   ///Mutex for locking INDI communications.
   std::mutex m_addMutex;
   
public:
   pwrGUI( QWidget * Parent = 0, Qt::WindowFlags f = 0);
   
   virtual ~pwrGUI() noexcept;
   
      
   virtual void onDisconnect();
   
   virtual void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has been defined*/);
   
   virtual void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   
   
public slots:
   void addNewDevice( std::string * devName,
                      std::vector<std::string> * channels
                    );
   
   void chChange( pcf::IndiProperty & ip );
   void updateGauges();
   
   void on_buttonReconnect_pressed();

signals:
   void gotNewDevice( std::string * devName,
                      std::vector<std::string> * channels
                    );
   
private:
      
   Ui::pwr ui;
};

} //namespace xqt
   
