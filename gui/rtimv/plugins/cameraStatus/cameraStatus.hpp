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
      std::string m_filterDeviceName2;

      QGraphicsScene * m_qgs {nullptr};
      
      StretchBox * m_roiBox {nullptr};
      StretchBox * m_roiFullBox {nullptr};
      
      char m_blob[512]; ///< Memory for copying rtimvDictionary blobs

      int m_width {0};
      int m_height {0};
          
   public:
      cameraStatus() ;
      
      virtual ~cameraStatus();

      virtual int attachOverlay( rtimvOverlayAccess &,
                                 mx::app::appConfigurator & config
                               ); 
      
      virtual int updateOverlay();

      virtual void keyPressEvent( QKeyEvent * ke);
      
      virtual bool overlayEnabled();
      
      bool blobExists( const std::string propel );

      bool getBlobStr( const std::string & deviceName,
                       const std::string & propel 
                     );

      bool getBlobStr( const std::string & propel );

      template<typename realT>
      realT getBlobVal( const std::string & propel, realT defVal );

      virtual void enableOverlay();

      virtual void disableOverlay();
      
   signals:
         
      void newStretchBox(StretchBox *);

      void savingState(bool);
};

#endif //cameraStatus_hpp
