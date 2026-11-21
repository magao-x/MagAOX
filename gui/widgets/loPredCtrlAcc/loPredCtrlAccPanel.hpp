/** \file loPredCtrlAccPanel.hpp
 * \brief Operator controls and live diagnostics for the loPredCtrlAcc DDSPC controller.
 */
#ifndef loPredCtrlAccPanel_hpp
#define loPredCtrlAccPanel_hpp

#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <QString>
#include <QStringList>

#include "../xWidgets/xWidget.hpp"

class QBoxLayout;
class QGridLayout;
class QLabel;
class QPushButton;
class QSlider;
class QTimer;

namespace xqt
{

class statusEntry;
class statusLabel;
class toggleSlider;

/// Operate loPredCtrlAcc learning and prediction and show its diagnostics, entirely through INDI.
/** Every control shows the latest INDI readback and sends only explicit operator requests. Nothing is sent when the
 * panel opens or closes. Mode changes and model resets are offered only in the state in which loPredCtrlAcc accepts
 * them (prediction off and its own zero confirmed), and the controller still validates every request.
 */
class loPredCtrlAccPanel : public xWidget
{
  public:
    /// Clock used for telemetry age, replaceable for tests.
    using clockFunction = std::function<std::chrono::steady_clock::time_point()>;

    /// Construct the controls disabled and the readbacks unavailable.
    explicit loPredCtrlAccPanel( const std::string &device = "loPredCtrlAcc", /**< [in] INDI device name. */
                                 QWidget           *parent = nullptr /**< [in] Optional Qt parent. */ );

    /// Detach nested standard controls before Qt destroys them.
    ~loPredCtrlAccPanel() noexcept override;

    /// Subscribe to the controller's properties and the WFS camera rate through the shared publisher.
    void subscribe() override;

    /// Mark the INDI connection available; controls wait for the device's own readbacks.
    void onConnect() override;

    /// Clear every readback and disable every control.
    void onDisconnect() override;

    /// Apply a property definition, including the numeric limits it carries.
    void handleDefProperty( const pcf::IndiProperty &property /**< [in] Property definition. */ ) override;

    /// Invalidate a deleted property, or everything when the whole device is deleted.
    void handleDelProperty( const pcf::IndiProperty &property /**< [in] Deletion notification. */ ) override;

    /// Merge a (possibly partial) property update and refresh the panel.
    void handleSetProperty( const pcf::IndiProperty &property /**< [in] Property update. */ ) override;

    /// Replace the reset confirmation, a modal question by default.
    void setResetConfirmation( std::function<bool()> confirm /**< [in] Returns true to send the reset. */ );

    /// Replace the clock used for telemetry age.
    void setClock( clockFunction clock /**< [in] Monotonic time source. */ );

    /// Update the telemetry age display; called once per second.
    void checkStale();

    /// Seconds without any update from the device after which values are marked possibly stale.
    static constexpr int c_staleSeconds = 10;

  private:
    /// Latest merged readback of one property.
    struct Readback
    {
        pcf::IndiProperty::Type              m_type{ pcf::IndiProperty::Unknown }; ///< Property type.
        pcf::IndiProperty::PropertyStateType m_state{
            pcf::IndiProperty::UnknownPropertyState };      ///< Latest property state.
        std::map<std::string, pcf::IndiElement> m_elements; ///< Latest value of every element seen.
    };

    /// One read-only status row: a label computed from the merged readbacks.
    struct Row
    {
        statusLabel             *m_value{ nullptr }; ///< Displayed value.
        std::function<QString()> m_text;             ///< Text from the current readbacks, "---" when unavailable.
    };

    std::string                     m_device;                ///< loPredCtrlAcc INDI device name.
    std::string                     m_wfsDevice{ "camwfs" }; ///< WFS camera whose acquisition rate is shown.
    std::map<std::string, Readback> m_readbacks;             ///< Controller properties by name.
    Readback                        m_wfsTiming;             ///< camwfs.fg_timing readback.

    std::map<std::string, double> m_previousCounters; ///< Warning counters at the previous performance update.
    QStringList                   m_increased;        ///< Warning counters that rose in the latest update.

    std::vector<Row> m_rows; ///< Status rows refreshed together.

    toggleSlider *m_learnStd{ nullptr }; ///< learn_std toggle.
    toggleSlider *m_learn{ nullptr };    ///< learn toggle.
    toggleSlider *m_predict{ nullptr };  ///< predict toggle.

    QPushButton *m_standard{ nullptr };      ///< ddspc_mode.standard selector.
    QPushButton *m_assisted{ nullptr };      ///< ddspc_mode.accel_assisted selector.
    QPushButton *m_reset{ nullptr };         ///< reset_model request.
    QLabel      *m_interlock{ nullptr };     ///< Why mode change and reset are unavailable.
    QLabel      *m_banner{ nullptr };        ///< Prominent prediction state.
    QLabel      *m_selectedReady{ nullptr }; ///< Readiness of the selected model.
    QLabel      *m_warnings{ nullptr };      ///< Counters that rose and alerts.
    QLabel      *m_age{ nullptr };           ///< Telemetry age.

    QSlider     *m_gainSlider{ nullptr }; ///< Gain slider spanning the property's own limits.
    statusEntry *m_gainEntry{ nullptr };  ///< Gain current value and numeric target entry.
    double       m_gainMin{ 0 };          ///< Gain lower limit from the property definition.
    double       m_gainMax{ 0 };          ///< Gain upper limit from the property definition.
    double       m_gainStep{ 0 };         ///< Gain step from the property definition.
    bool         m_gainLimits{ false };   ///< The definition supplied finite, ordered limits and a step.

    statusEntry *m_exploration{ nullptr }; ///< exploration_sequence current value and target entry.

    QWidget     *m_advanced{ nullptr };       ///< Less frequently used controls and diagnostics.
    QPushButton *m_advancedButton{ nullptr }; ///< Shows or hides m_advanced.

    QTimer                               *m_staleTimer{ nullptr }; ///< Drives checkStale().
    clockFunction                         m_clock;                 ///< Monotonic time source.
    std::chrono::steady_clock::time_point m_lastUpdate{};          ///< Latest update from the device.
    bool                                  m_haveUpdate{ false };   ///< Any update since the last disconnect.

    std::function<bool()> m_confirmReset; ///< Reset confirmation.

    /// Return an element value as text, or an empty string when unavailable.
    std::string value( const std::string &property, /**< [in] Controller property name. */
                       const std::string &element /**< [in] Element name. */ ) const;

    /// Return a switch element's state, Unknown when unavailable.
    pcf::IndiElement::SwitchStateType switchState( const std::string &property, /**< [in] Controller property. */
                                                   const std::string &element /**< [in] Element name. */ ) const;

    /// Return whether the controller reports the OPERATING state.
    bool operating() const;

    /// Return why mode changes and resets are refused, or an empty string when the controller would accept them.
    QString interlock() const;

    /// Add a status row computed by text.
    void addRow( QGridLayout             *grid,  /**< [in,out] Grid receiving the row. */
                 const char              *key,   /**< [in] Object name suffix. */
                 const char              *label, /**< [in] Operator-facing label. */
                 std::function<QString()> text,  /**< [in] Text from the current readbacks. */
                 const char              *tooltip = nullptr /**< [in] Optional explanation. */ );

    /// Add a row showing one element's value verbatim.
    void addElementRow( QGridLayout *grid,     /**< [in,out] Grid receiving the row. */
                        const char  *label,    /**< [in] Operator-facing label. */
                        const char  *property, /**< [in] Controller property. */
                        const char  *element /**< [in] Element name. */ );

    /// Add a standard toggle slider for a controller switch.
    toggleSlider *addToggle( QBoxLayout *layout,   /**< [in,out] Layout receiving the slider. */
                             const char *property, /**< [in] Switch property. */
                             const char *label,    /**< [in] Operator-facing label. */
                             const char *tooltip /**< [in] Effect of the switch. */ );

    /// Merge an update into a readback.
    static void merge( Readback                &readback, /**< [in,out] Stored readback. */
                       const pcf::IndiProperty &property /**< [in] Update or definition. */ );

    /// Read finite gain limits and step from a definition.
    void readGainLimits( const pcf::IndiProperty &property /**< [in] gain definition. */ );

    /// Record which warning counters rose in a performance update.
    void trackCounters( const pcf::IndiProperty &property /**< [in] ddspc_performance update. */ );

    /// Send a complete ddspc_mode selection with exactly one element on.
    void sendMode( const char *selected /**< [in] Element to select. */ );

    /// Ask for confirmation, then request reset_model.
    void requestReset();

    /// Send a gain target from the slider position.
    void sendGainFromSlider();

    /// Recompute every readback, indicator and control availability.
    void refresh();
};

} // namespace xqt

#endif // loPredCtrlAccPanel_hpp
