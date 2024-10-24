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

    /** \name Configurable Parameters
      * @{
      */
    std::string m_deviceName; ///< INDI device name of the acquisition program

    std::string m_cameraName; ///< INDI device name of the associated camera

    int m_circRad {10};  ///< Radius of the circle to draw around the star

    std::string m_color {"cyan"}; ///< Color name or RGB spec for the overlay

    int m_fontSize {18}; ///< The font size for the overlay

    ///@}

    rtimvOverlayAccess m_roa;

    bool m_enabled{false};

    bool m_enableable{false};


    QGraphicsScene *m_qgs{nullptr};

    size_t m_nStars {0};
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
