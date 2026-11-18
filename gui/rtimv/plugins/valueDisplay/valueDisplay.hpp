/** \file valueDisplay.hpp
 * \brief Declares the two-value INDI text overlay for rtimv.
 */

#ifndef valueDisplay_hpp
#define valueDisplay_hpp

#include <rtimv/rtimvInterfaces.hpp>

#include <QMetaMethod>
#include <QPointer>
#include <QTextEdit>

#include <array>

/// Display up to two subscribed numeric INDI elements in the upper-left corner.
class valueDisplay : public rtimvOverlayInterface
{
    Q_OBJECT
#ifndef RTIMV_PLUGIN_TEST
    Q_PLUGIN_METADATA( IID "rtimv.overlayInterface/1.4" )
#endif
    Q_INTERFACES( rtimvOverlayInterface )

  protected:
    /// Configuration for one independently subscribed value.
    struct valueSlot
    {
        /// Full device.property.element key; empty disables this row.
        std::string m_key;

        /// Plain-text label, defaulting to the element name.
        QString m_label;

        /// Number of fixed decimal places used for display.
        int m_precision{ 3 };
    };

    /// Configured values in display order.
    std::array<valueSlot, 2> m_values;

    /// View owned by rtimv; nulled if it is destroyed before the plugin.
    QPointer<rtimvGraphicsView> m_view;

    /// Private text widget deleted by this plugin or its parent view.
    QPointer<QTextEdit> m_text;

    /// Optional viewer owning the native image-dependent background calculation.
    QPointer<QObject> m_host;

    /// Native nonblocking background updater, invalid on older viewers.
    QMetaMethod m_fontLuminance;

    /// Shared dictionary, valid while the host view and plugin are attached.
    dictionaryT *m_dictionary{ nullptr };

    /// Whether configuration and attachment permit enabling the overlay.
    bool m_enableable{ false };

    /// Whether the user has enabled this overlay.
    bool m_enabled{ false };

    /// Remove this plugin's widget and geometry observers.
    void detachOverlay();

    /// Read and format one finite number, returning an empty string when unavailable.
    QString valueText( const valueSlot &value /**< [in] Key and display precision for the row. */ );

    /// Position the text below native overlays and refresh its image-dependent background.
    void layoutText();

    /// Follow geometry and visibility changes without consuming view events.
    bool eventFilter( QObject *watched, /**< [in] View or native overlay whose geometry changed. */
                      QEvent  *event /**< [in] Event being delivered to the observed object. */ ) override;

  public:
    /// Construct an unconfigured overlay.
    valueDisplay();

    /// Remove the private widget and release non-owning host references.
    ~valueDisplay() override;

    /// Configure rows and register their keys without disturbing existing values.
    /** \returns 0 on attachment, 1 when unconfigured, or -1 for invalid configuration/access.
     */
    int attachOverlay( rtimvOverlayAccess       &access, /**< [in] Host view and shared dictionary. */
                       mx::app::appConfigurator &config /**< [in] Viewer configuration. */ ) override;

    /// Refresh displayed values and their placement using the existing viewer cadence.
    int updateOverlay() override;

    /// Leave keyboard shortcuts to the viewer and other plugins.
    void keyPressEvent( QKeyEvent *event /**< [in] Forwarded key event. */ ) override;

    /// Report whether the overlay is enabled.
    bool overlayEnabled() override;

    /// Enable and immediately refresh a successfully configured overlay.
    void enableOverlay() override;

    /// Hide this plugin's text without changing other overlays or subscriptions.
    void disableOverlay() override;

    /// Describe the configured sources in the viewer's plugin information.
    std::vector<std::string> info() override;
};

#endif // valueDisplay_hpp
