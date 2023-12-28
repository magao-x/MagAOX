#ifndef pwfsAlignment_hpp
#define pwfsAlignment_hpp

#include <rtimv/rtimvInterfaces.hpp>

#include <QObject>
#include <QtPlugin>
#include <QGraphicsLineItem>

#include <iostream>

class pwfsAlignment : public rtimvOverlayInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "rtimv.overlayInterface/1.2")
    Q_INTERFACES(rtimvOverlayInterface)

protected:
    rtimvOverlayAccess m_roa;

    char m_blob[512]; ///< Memory for copying rtimvDictionary blobs

    double m_1x{0};
    double m_1y{0};
    double m_1D{0};

    double m_set1x{0};
    double m_set1y{0};
    double m_set1D{0};

    double m_2x{0};
    double m_2y{0};
    double m_2D{0};

    double m_set2x{0};
    double m_set2y{0};
    double m_set2D{0};

    double m_3x{0};
    double m_3y{0};
    double m_3D{0};

    double m_set3x{0};
    double m_set3y{0};
    double m_set3D{0};

    double m_4x{0};
    double m_4y{0};
    double m_4D{0};

    double m_set4x{0};
    double m_set4y{0};
    double m_set4D{0};

    QGraphicsScene *qgs;

    QGraphicsLineItem *m_1to2{nullptr};
    QGraphicsLineItem *m_1to3{nullptr};
    QGraphicsLineItem *m_3to4{nullptr};
    QGraphicsLineItem *m_2to4{nullptr};

    QGraphicsLineItem *m_1to2s{nullptr};
    QGraphicsLineItem *m_1to3s{nullptr};
    QGraphicsLineItem *m_3to4s{nullptr};
    QGraphicsLineItem *m_2to4s{nullptr};

    QGraphicsEllipseItem *m_c1{nullptr};
    QGraphicsEllipseItem *m_c2{nullptr};
    QGraphicsEllipseItem *m_c3{nullptr};
    QGraphicsEllipseItem *m_c4{nullptr};

    QGraphicsEllipseItem *m_c1s{nullptr};
    QGraphicsEllipseItem *m_c2s{nullptr};
    QGraphicsEllipseItem *m_c3s{nullptr};
    QGraphicsEllipseItem *m_c4s{nullptr};

    std::string m_deviceName;
    int m_numPupils{0};
    int m_width{0};
    int m_height{0};

public:
    pwfsAlignment();

    virtual ~pwfsAlignment();

    virtual int attachOverlay(rtimvOverlayAccess &,
                              mx::app::appConfigurator &config);

    virtual int updateOverlay();

    virtual void keyPressEvent(QKeyEvent *ke);

    virtual bool overlayEnabled();

    virtual void enableOverlay();

    virtual void disableOverlay();

public:
    virtual std::vector<std::string> info();
};

#endif // pwfsAlignment_hpp
