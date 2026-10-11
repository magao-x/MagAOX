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
#include <rtimv/rtimvCommandInterface.hpp>
#include <QElapsedTimer>
#include "../../../../libMagAOX/common/roiGeometry.hpp"

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
#ifndef RTIMV_PLUGIN_TEST
    Q_PLUGIN_METADATA( IID "rtimv.overlayInterface/1.4" )
#endif
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
    QPointer<StretchBox> m_roiBox;

    /// Native camera geometry used for coherent edit/Check/Set transitions.
    using roiT = MagAOX::roi::region;

    /// ROI interaction; only an explicit fresh key can leave adjusted for checking.
    enum class roiPhase { idle, editing, dragging, awaitingTarget, checking, adjusted, applying };

    /// Current interaction phase, owned by the GUI thread.
    roiPhase m_roiPhase{roiPhase::idle};
    /// Pins the editable preview even when its target equals current.
    bool m_explicitROI{false};
    /// True only while all required property definitions and current geometry are valid.
    bool m_roiReady{false};
    /// Prevents stale legacy fallback after a new camera loses its capability.
    bool m_commandSeen{false};
    /// Continue one apply request after its staged target has been echoed.
    bool m_applyAfterEcho{false};
    /// Applied reference is positively classified as full frame or default.
    bool m_newBoxPreset{false};
    /// The router is owned by the main window and discovered lazily.
    QPointer<rtimvCommandRouter> m_commands;
    /// Native coordinate convention for this connection.
    MagAOX::roi::units m_roiUnits{MagAOX::roi::units::unsupported};
    /// Latest coherent current and staged target geometry.
    roiT m_currentROI, m_targetROI;
    /// Reference and target captured for a pending Check or Set.
    roiT m_checkCurrent, m_requestedROI;
    /// Expected final target and its pre-edit value while awaiting multiple echoes.
    roiT m_desiredROI, m_echoBase;
    /// Backend target and scene rectangle at the start of a mouse edit.
    roiT m_dragROI;
    QRectF m_dragRect;
    /// Connection generation and latest Check sequence captured for an action.
    quint64 m_roiGeneration{0}, m_checkSequence{0};
    /// Bounds acknowledgement waits without blocking the GUI.
    QElapsedTimer m_roiWait;
    /// Local timeout in milliseconds; a timeout never causes replay.
    int m_roiTimeout{5000};

    /// Copy one typed camera property from the optional provider.
    rtimvPropertySnapshot roiProperty(const QString &name /**< [in] Camera property name. */);

    /// Refresh coherent ROI state; return true when it supersedes the legacy preview.
    bool updateROI();

    /// Create the single ROI box and connect opt-in user editing hooks.
    void makeROIBox();

    /// Project the current target into pixel-edge scene coordinates.
    QRectF roiRect() const;

    /// Redraw backend geometry without generating an outgoing command.
    void drawROI();

    /// Reveal an existing edit, preserve a custom ROI, or stage a new preset subframe.
    void beginROI();

    /// Convert a scene edit to native coordinates and send only changed fields.
    bool stageROI(const QRectF &rect, /**< [in] Pixel-edge rectangle after editing. */
                  const roiT &before, /**< [in] Captured native target before editing. */
                  bool resized /**< [in] True only for intentional size changes. */);

    /// Request one fresh Check or remember one action while waiting for its target echo.
    void applyROI();

    /// Submit typed commands once, reporting a local failure through the status box.
    bool sendROI(const QVector<rtimvCommand> &commands /**< [in] Ordered property vectors. */);

    /// Revoke pending authorization and restore the backend target for review.
    void failROI(const QString &message /**< [in] Reason displayed in the normal status box. */);

    /// Cancel unsent edits and confirmation; an already submitted Set remains tracked.
    void escapeROI();

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

    /// Insert a box sharing the normal user-box interaction and readout.
    void newInteractiveBox(StretchBox *box /**< [in] Overlay-owned ROI box. */);

    /// Display a transient message in rtimv's normal status box.
    void roiMessage(const QString &message /**< [in] Short interaction result. */);

    /// Report the current save-state indicator for the camera stream.
    void savingState( rtimv::savingState state /**< [in] current save-state display mode */ );

  public slots:
    /// Capture a user mouse edit without writing any camera properties.
    void roiEditStarted(StretchBox *box /**< [in] Edited ROI box. */);

    /// Stage changed coordinates once after the user releases the mouse.
    void roiEditFinished(StretchBox *box /**< [in] Edited ROI box. */);

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
