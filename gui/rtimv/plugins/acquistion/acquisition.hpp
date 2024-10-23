#ifndef acquisition_hpp
#define acquisition_hpp

#include <rtimv/rtimvInterfaces.hpp>
#include <rtimv/StretchBox.hpp>

#include <QObject>
#include <QtPlugin>
#include <QTextEdit>
// #include <QGraphicsLineItem>

#include <iostream>

class acquisition : public rtimvOverlayInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "rtimv.overlayInterface/1.2")
    Q_INTERFACES(rtimvOverlayInterface)

protected:
    rtimvOverlayAccess m_roa;

    bool m_enabled{false};

    bool m_enableable{false};

    std::string m_deviceName;

    std::string m_cameraName;

    QGraphicsScene *m_qgs{nullptr};

    std::vector<StretchCircle *> m_starCircles;
    std::vector<QTextEdit *> m_starLabels;

    std::mutex m_starCircleMutex;

    int m_width {-1};
    int m_height {-1};

    char m_blob[512]; ///< Memory for copying rtimvDictionary blobs

public:
    acquisition();

    virtual ~acquisition();

    virtual int attachOverlay(rtimvOverlayAccess &,
                              mx::app::appConfigurator &config);

    virtual int updateOverlay();

    virtual void keyPressEvent(QKeyEvent *ke);

    virtual bool overlayEnabled();

    bool blobExists(const std::string & propel);

    bool getBlobStr(const std::string &deviceName,
                    const std::string &propel);

    bool getBlobStr(const std::string &propel);

    template <typename realT>
    realT getBlobVal(const std::string &device, const std::string &propel, realT defVal);

    template <typename realT>
    realT getBlobVal(const std::string &propel, realT defVal);

    virtual void enableOverlay();

    virtual void disableOverlay();

signals:

    void newStretchCircle(StretchCircle *);

    void savingState(rtimv::savingState);

public slots:

    void stretchCircleRemove(StretchCircle * );

public:
    virtual std::vector<std::string> info();
};

#endif // acquisition_hpp
