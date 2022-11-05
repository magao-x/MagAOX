#ifndef cameraStatus_hpp
#define cameraStatus_hpp

#include <rtimv/rtimvInterfaces.hpp>
#include <rtimv/StretchBox.hpp>

#include <QObject>
#include <QtPlugin>
//#include <QGraphicsLineItem>

#include <iostream>

class cameraStatus :
                     public rtimvOverlayInterface
{
   Q_OBJECT
   Q_PLUGIN_METADATA(IID "rtimv.overlayInterface/1.1")
   Q_INTERFACES(rtimvOverlayInterface)
    
   protected:
      rtimvOverlayAccess m_roa;
      
      bool m_enabled {false};
      
      bool m_enableable {false};
      
      std::string m_deviceName;
      
      std::string m_filterDeviceName;
      
      QGraphicsScene * m_qgs {nullptr};
      
      StretchBox * m_roiBox {nullptr};
      StretchBox * m_roiFullBox {nullptr};
      
      char m_blob[512]; ///< Memory for copying rtimvDictionary blobs

      int m_width {0};
      int m_height {0};
      
      float m_fullROI_x {0};
      float m_fullROI_y {0};
      int m_fullROI_w {0};
      int m_fullROI_h {0};
      
   public:
      cameraStatus() ;
      
      virtual ~cameraStatus();

      virtual int attachOverlay( rtimvOverlayAccess &,
                                 mx::app::appConfigurator & config
                               ); 
      
      virtual int updateOverlay();

      virtual void keyPressEvent( QKeyEvent * ke);
      
      virtual bool overlayEnabled();
      
      virtual void enableOverlay();

      virtual void disableOverlay();
      
   signals:
         
      void newStretchBox(StretchBox *);
};

#endif //cameraStatus_hpp
