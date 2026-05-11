/** \file cameraStatus.hpp
 * \brief Declares the camera-status rtimv overlay plugin.
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 */

#ifndef cameraStatus_hpp
#define cameraStatus_hpp

#include <rtimv/rtimvInterfaces.hpp>
#include <rtimv/StretchBox.hpp>

#include <mutex>
#include <string>
#include <vector>

#include <QObject>
#include <QtPlugin>

#include <iostream>

/// Overlay plugin which renders camera telemetry and associated stage status.
class cameraStatus : public rtimvOverlayInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA( IID "rtimv.overlayInterface/1.4" )
    Q_INTERFACES( rtimvOverlayInterface )

  protected:
    /// Access to selected rtimv GUI and dictionary state.
    rtimvOverlayAccess m_roa;

    /// True when the plugin is currently allowed to update state.
    bool m_enabled{ false };

    /// True when configuration was sufficient to enable this plugin.
    bool m_enableable{ false };

    /// Camera device prefix used when forming dictionary keys.
    std::string m_deviceName;

    /// Associated filter-wheel or stage device names shown in the overlay.
    std::vector<std::string> m_filterDeviceNames;
    /// Preferred preset-property name to check first for each associated device.
    std::vector<std::string> m_primaryPresetProperties;

    /// Cached graphics scene used for ROI-box lifetime management.
    QGraphicsScene *m_qgs{ nullptr };

    /// Pending ROI box used to preview target subframes.
    StretchBox *m_roiBox{ nullptr };

    /// Protects creation, updates, and removal of the preview ROI box.
    std::mutex m_roiBoxMutex;

    /// Scratch buffer used when copying dictionary blobs as strings.
    char m_blob[512];

    /// Cached detector width used when mapping ROI coordinates into the view.
    int m_width{ 0 };
    /// Cached detector height used when mapping ROI coordinates into the view.
    int m_height{ 0 };

    /// True after logging one status-text overflow warning for the current overflow episode.
    bool m_statusTextOverflowWarned{ false };

  public:
    /// Construct the camera-status plugin.
    cameraStatus();

    /// Destruct the camera-status plugin.
    virtual ~cameraStatus();

    /// Attach the overlay plugin to rtimv.
    virtual int
    attachOverlay( rtimvOverlayAccess       &roa, /**< [in] exposed main-window, graphics-view, and dictionary access */
                   mx::app::appConfigurator &config /**< [in] configuration source for plugin options */
    );

    /// Update the overlay text and ROI preview from the current dictionary values.
    virtual int updateOverlay();

    /// Handle key presses passed through by rtimv.
    virtual void keyPressEvent( QKeyEvent *ke /**< [in] key event to inspect */ );

    /// Report whether this overlay is currently enabled.
    virtual bool overlayEnabled();

    /// Return true when the camera dictionary entry exists and currently holds data.
    bool blobExists( const std::string &propel /**< [in] camera property.element suffix to inspect */ );

    /// Copy a device property value into the internal string buffer.
    bool getBlobStr( const std::string &deviceName, /**< [in] device prefix used when forming the dictionary key */
                     const std::string &propel      /**< [in] property.element suffix to copy from the dictionary */
    );

    /// Copy a camera property value into the internal string buffer.
    bool getBlobStr( const std::string &propel /**< [in] camera property.element suffix to copy */ );

    /// Return a numeric dictionary value or a supplied default.
    template <typename realT>
    realT getBlobVal( const std::string &propel, /**< [in] camera property.element suffix to convert */
                      realT              defVal  /**< [in] default value returned when the blob is unavailable */
    );

    /// Enable the overlay if configuration allows it.
    virtual void enableOverlay();

    /// Disable the overlay and clear any status text it owns.
    virtual void disableOverlay();

  signals:
    /// Request insertion of a newly created ROI preview box into the scene.
    void newStretchBox( StretchBox *sb /**< [in] ROI preview box to add to the view */ );

    /// Report the current save-state indicator for the camera stream.
    void savingState( rtimv::savingState state /**< [in] current save-state display mode */ );

  public slots:
    /// Clear the cached ROI preview pointer when the view removes it.
    void stretchBoxRemove( StretchBox *sb /**< [in] ROI box being removed from the scene */ );

  public:
    /// Return user-visible plugin information for the `i` overlay.
    virtual std::vector<std::string> info();

  protected:
    /// Return true when the named switch property exposes an active element.
    bool findActivePropertySelection(
        const std::string &deviceName,   /**< [in] associated filter or stage device name */
        const std::string &propertyName, /**< [in] switch-property name such as `presetName` or `filterName` */
        std::string       &selection     /**< [out] active element name when one switch is on */
    );

    /// Return true when any known preset property exposes an active element.
    bool findActivePresetSelection( size_t       deviceIndex, /**< [in] index of the associated device being rendered */
                                    std::string &selection    /**< [out] active element name when one switch is on */
    );
};

#endif // cameraStatus_hpp
