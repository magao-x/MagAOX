#ifndef cameraStatus_hpp
#define cameraStatus_hpp

#include <rtimv/rtimvInterfaces.hpp>
#include <rtimv/StretchBox.hpp>

#include <QObject>
#include <QtPlugin>
// #include <QGraphicsLineItem>

#include <iostream>

class cameraStatus : public rtimvOverlayInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "rtimv.overlayInterface/1.2")
    Q_INTERFACES(rtimvOverlayInterface)

protected:
    rtimvOverlayAccess m_roa;

    bool m_enabled{false};

    bool m_enableable{false};

    std::string m_deviceName;

    std::vector<std::string> m_filterDeviceNames;
    std::vector<std::string> m_presetNames; //one per filter device, based on its name
    
    QGraphicsScene *m_qgs{nullptr};

    StretchBox *m_roiBox{nullptr};

    std::mutex m_roiBoxMutex;

    char m_blob[512]; ///< Memory for copying rtimvDictionary blobs

    int m_width{0};
    int m_height{0};

public:
    cameraStatus();

    virtual ~cameraStatus();

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
    realT getBlobVal(const std::string &propel, realT defVal);

    virtual void enableOverlay();

    virtual void disableOverlay();

signals:

    void newStretchBox(StretchBox *);

    void savingState(rtimv::savingState);

public slots:

    void stretchBoxRemove(StretchBox * );

public:
    virtual std::vector<std::string> info();
};

#endif // cameraStatus_hpp
