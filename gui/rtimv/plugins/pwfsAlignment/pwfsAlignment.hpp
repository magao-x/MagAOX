#ifndef pwfsAlignment_hpp
#define pwfsAlignment_hpp

#include <rtimvInterfaces.hpp>

#include <QObject>
#include <QtPlugin>
#include <QGraphicsLineItem>

#include <iostream>

class pwfsAlignment : public QObject,
                      public rtimvOverlayInterface
{
   Q_OBJECT
   Q_PLUGIN_METADATA(IID "rtimv.overlayInterface/1.0")
   Q_INTERFACES(rtimvOverlayInterface)
    
   protected:
      bool m_enabled {false};
      
      std::unordered_map<std::string, rtimvDictBlob> * m_dict {nullptr};
      
      double m_1x {0};
      double m_1y {0};
      double m_1D {0};
      
      double m_2x {0};
      double m_2y {0};
      double m_2D {0};
      
      double m_3x {0};
      double m_3y {0};
      double m_3D {0};
      
      double m_4x {0};
      double m_4y {0};
      double m_4D {0};
      
      QGraphicsLineItem * m_1to2;
      QGraphicsLineItem * m_1to3;
      QGraphicsLineItem * m_3to4;
      QGraphicsLineItem * m_2to4;
      
      QGraphicsLineItem * m_1to2s;
      QGraphicsLineItem * m_1to3s;
      QGraphicsLineItem * m_3to4s;
      QGraphicsLineItem * m_2to4s;
      
      QGraphicsEllipseItem * m_c1;
      QGraphicsEllipseItem * m_c2;
      QGraphicsEllipseItem * m_c3;
      QGraphicsEllipseItem * m_c4;
      
      QGraphicsEllipseItem * m_c1s;
      QGraphicsEllipseItem * m_c2s;
      QGraphicsEllipseItem * m_c3s;
      QGraphicsEllipseItem * m_c4s;
      
   public:
      pwfsAlignment();
      
      virtual ~pwfsAlignment();

      virtual int attachOverlay(QGraphicsScene*, std::unordered_map<std::string, rtimvDictBlob> *); 
      
      virtual int updateOverlay();

      virtual bool overlayEnabled();
      
      virtual void enableOverlay();

      virtual void disableOverlay();

};

#endif //pwfsAlignment_hpp
