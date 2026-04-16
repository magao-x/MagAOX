/** \file warnings.hpp
 * \brief Declares the warnings rtimv overlay plugin.
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 */

#ifndef warnings_hpp
#define warnings_hpp

#include <rtimv/rtimvInterfaces.hpp>

#include <QObject>
#include <QtPlugin>

#include <iostream>

/// Overlay plugin which monitors dictionary-backed caution, warning, and alert state.
class warnings : public rtimvOverlayInterface
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

    /// Device prefix used when forming dictionary keys.
    std::string m_deviceName;

    /// Dictionary keys checked at caution severity.
    std::vector<std::string> m_cautionKeys;

    /// Dictionary keys checked at warning severity.
    std::vector<std::string> m_warningKeys;

    /// Dictionary keys checked at alert severity.
    std::vector<std::string> m_alertKeys;

    /// Cached active caution keys used to trigger overlay refresh only on changes.
    std::vector<std::string> m_activeCautions;

    /// Cached active warning keys used to trigger overlay refresh only on changes.
    std::vector<std::string> m_activeWarnings;

    /// Cached active alert keys used to trigger overlay refresh only on changes.
    std::vector<std::string> m_activeAlerts;

    char m_blob[512]; ///< Memory for copying rtimvDictionary blobs

  public:
    /// Construct the warnings plugin.
    warnings();

    /// Destruct the warnings plugin.
    virtual ~warnings();

    /// Attach the overlay plugin to rtimv.
    virtual int
    attachOverlay( rtimvOverlayAccess & /**< [in] exposed main-window, graphics-view, and dictionary access */,
                   mx::app::appConfigurator &config /**< [in] configuration source for plugin options */
    );

    /// Update the warning-border state from the current dictionary values.
    virtual int updateOverlay();

    /// Handle key presses passed through by rtimv.
    virtual void keyPressEvent( QKeyEvent *ke /**< [in] key event to inspect */ );

    /// Report whether this plugin provides a full-screen text overlay.
    virtual bool hasTextOverlay();

    /// Return the key used to toggle the warnings text overlay.
    virtual char textOverlayKey();

    /// Return the short title used in the built-in help listing.
    virtual std::string textOverlayTitle();

    /// Generate the current full-screen warnings text overlay.
    virtual std::string textOverlayText();

    /// Report whether this overlay is currently enabled.
    virtual bool overlayEnabled();

    /// Enable the overlay if configuration allows it.
    virtual void enableOverlay();

    /// Disable the overlay and clear any status text it owns.
    virtual void disableOverlay();

  public:
    /// Return user-visible plugin information for the `i` overlay.
    virtual std::vector<std::string> info();

  protected:
    /// Check whether a single dictionary key is currently set to `on`.
    bool keyOn( const std::string &key /**< [in] fully qualified dictionary key to inspect */ );

    /// Check whether any keys in one severity group are currently active.
    bool anyOn( const std::vector<std::string> &keys, /**< [in] configured leaf keys for one severity group */
                const std::string &prefix /**< [in] severity-specific key prefix inserted after the device name */
    );

    /// Return the active keys in one severity group.
    std::vector<std::string>
    activeKeys( const std::vector<std::string> &keys, /**< [in] configured leaf keys for one severity group */
                const std::string &prefix /**< [in] severity-specific key prefix inserted after the device name */
    );

    /// Append all active keys from one severity group to the overlay text.
    void
    appendActive( std::string                    &text,    /**< [in,out] accumulated overlay text */
                  const std::string              &heading, /**< [in] heading label for the severity group */
                  const std::vector<std::string> &keys,    /**< [in] configured leaf keys for one severity group */
                  const std::string &prefix /**< [in] severity-specific key prefix inserted after the device name */
    );

  signals:
    /// Emit the highest currently active warning level.
    void warningLevel( rtimv::warningLevel lvl /**< [in] highest active warning severity */ );

    /// Request a refresh of the warnings full-screen text overlay.
    void textOverlayRefreshRequested( char key /**< [in] shortcut key identifying the overlay to refresh */ );
};

#endif // warnings_hpp
