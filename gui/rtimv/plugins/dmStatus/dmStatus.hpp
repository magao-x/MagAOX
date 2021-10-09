#ifndef dmStatus_hpp
#define dmStatus_hpp

#include <rtimv/rtimvInterfaces.hpp>
#include <rtimv/StretchBox.hpp>

#include <QObject>
#include <QtPlugin>
//#include <QGraphicsLineItem>

#include <iostream>

class dmStatus :
                     public rtimvOverlayInterface
{
   Q_OBJECT
   Q_PLUGIN_METADATA(IID "rtimv.overlayInterface/1.0")
   Q_INTERFACES(rtimvOverlayInterface)
    
   protected:
      rtimvOverlayAccess m_roa;
      
      bool m_enabled {false};
      
      bool m_enableable {false};
      
      std::string m_deviceName;
      
      std::string m_rhDeviceName;
      
      QGraphicsScene * m_qgs {nullptr};
      
   public:
      dmStatus() ;
      
      virtual ~dmStatus();

      virtual int attachOverlay( rtimvOverlayAccess &,
                                 mx::app::appConfigurator & config
                               ); 
      
      virtual int updateOverlay();

      virtual void keyPressEvent( QKeyEvent * ke);
      
      virtual bool overlayEnabled();
      
      virtual void enableOverlay();

      virtual void disableOverlay();
      
};

#endif //dmStatus_hpp
