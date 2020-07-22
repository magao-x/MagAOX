#ifndef cameraStatus_hpp
#define cameraStatus_hpp

#include <rtimv/rtimvInterfaces.hpp>

#include <QObject>
#include <QtPlugin>
#include <QGraphicsLineItem>

#include <iostream>

class cameraStatus : public QObject,
                     public rtimvOverlayInterface
{
   Q_OBJECT
   Q_PLUGIN_METADATA(IID "rtimv.overlayInterface/1.0")
   Q_INTERFACES(rtimvOverlayInterface)
    
   protected:
      bool m_enabled {false};
      
      bool m_enableable {false};
      
      std::unordered_map<std::string, rtimvDictBlob> * m_dict {nullptr};
      
      std::string m_deviceName;
      
      rtimvGraphicsView* m_gv {nullptr}; 
      
   public:
      cameraStatus();
      
      virtual ~cameraStatus();

      virtual int attachOverlay( rtimvGraphicsView *, 
                                 std::unordered_map<std::string, rtimvDictBlob> *,
                                 mx::app::appConfigurator & config
                               ); 
      
      virtual int updateOverlay();

      virtual void keyPressEvent( QKeyEvent * ke);
      
      virtual bool overlayEnabled();
      
      virtual void enableOverlay();

      virtual void disableOverlay();

};

#endif //cameraStatus_hpp
