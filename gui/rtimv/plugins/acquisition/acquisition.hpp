/** \file acquisition.hpp
 * \brief Declares the acquisition overlay for detected-star annotations.
 */

#ifndef acquisition_hpp
#define acquisition_hpp

#include <rtimv/rtimvInterfaces.hpp>
#include <rtimv/StretchBox.hpp>

#include <QObject>
#include <QPointer>
#include <QtPlugin>
#include <QTextEdit>
// #include <QGraphicsLineItem>

#include <iostream>

class acquisition : public rtimvOverlayInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA( IID "rtimv.overlayInterface/1.4" )
    Q_INTERFACES( rtimvOverlayInterface )

  protected:
    /** \name Configurable Parameters
     * @{
     */
    std::string m_deviceName; ///< INDI device name of the acquisition program.

    std::string m_cameraName; ///< INDI device name of the associated camera.

    int m_circRad{ 10 }; ///< Radius of the circle to draw around each tracked star.

    std::string m_color{ "cyan" }; ///< Color name or RGB spec used for circles and labels.

    int m_fontSize{ 18 }; ///< Pixel font size for the star-number labels.

    ///@}

    rtimvOverlayAccess m_roa; ///< Access to rtimv scene, view, and shared dictionary services.

    bool m_enabled{ false }; ///< Whether the overlay is currently enabled.

    bool m_enableable{ false }; ///< Whether configuration succeeded enough to allow enabling.

    QGraphicsScene *m_qgs{ nullptr }; ///< Cached graphics scene from rtimv.

    size_t m_nStars{ 0 }; ///< Number of active stars reported by the acquisition process.

    std::vector<QPointer<StretchCircle>>
                                     m_starCircles; ///< Circle overlays, nulled automatically if rtimv deletes them.
    std::vector<QPointer<QTextEdit>> m_starLabels;  ///< Text labels paired with the circle overlays.

    std::mutex m_starCircleMutex; ///< Protects star-overlay container access and lazy recreation.

    int m_width{ -1 };  ///< Current camera frame width from the dictionary.
    int m_height{ -1 }; ///< Current camera frame height from the dictionary.

    char m_blob[512]; ///< Scratch buffer for copying rtimvDictionary blobs as strings.

    /// Ensure a star overlay exists at the requested index.
    void ensureStarOverlay( size_t n /**< [in] zero-based tracked-star index to create if missing */ );

  public:
    /// Construct the acquisition overlay.
    acquisition();

    /// Destroy the acquisition overlay and tear down any local label objects.
    virtual ~acquisition();

    /// Attach the overlay to rtimv.
    virtual int attachOverlay( rtimvOverlayAccess       &roa, /**< [in] rtimv access bundle for scene/view/dictionary */
                               mx::app::appConfigurator &config /**< [in] configurator providing overlay settings */
    );

    /// Update the tracked-star overlay state.
    virtual int updateOverlay();

    /// Handle overlay key presses.
    virtual void keyPressEvent( QKeyEvent *ke /**< [in] key event forwarded from rtimv */ );

    /// Report whether the overlay is enabled.
    virtual bool overlayEnabled();

    /// Check whether a property blob exists for the acquisition device.
    bool blobExists( const std::string &propel /**< [in] property.element suffix to inspect */ );

    /// Copy a property blob string for an arbitrary device.
    bool getBlobStr( const std::string &deviceName, /**< [in] INDI device name owning the property */
                     const std::string &propel      /**< [in] property.element suffix to read */
    );

    /// Copy a property blob string for the configured acquisition device.
    bool getBlobStr( const std::string &propel /**< [in] property.element suffix to read */ );

    /// Read a typed blob value for an arbitrary device.
    template <typename realT>
    realT getBlobVal( const std::string &device, /**< [in] INDI device name owning the property */
                      const std::string &propel, /**< [in] property.element suffix to read */
                      realT              defVal  /**< [in] fallback value when no blob is available */
    );

    /// Read a typed blob value for the configured acquisition device.
    template <typename realT>
    realT getBlobVal( const std::string &propel, /**< [in] property.element suffix to read */
                      realT              defVal  /**< [in] fallback value when no blob is available */
    );

    /// Enable the overlay.
    virtual void enableOverlay();

    /// Disable the overlay.
    virtual void disableOverlay();

  signals:
    /// Request that rtimv add a new stretch circle to the scene.
    void newStretchCircle( StretchCircle *sc /**< [in] new circle item to register with rtimv */ );

    void savingState( rtimv::savingState );

  public slots:
    /// Handle removal requests emitted by a managed stretch circle.
    void stretchCircleRemove( StretchCircle *sb /**< [in] circle being removed by rtimv or Qt */ );

  public:
    /// Report plugin status for display in rtimv.
    virtual std::vector<std::string> info();
};

#endif // acquisition_hpp
