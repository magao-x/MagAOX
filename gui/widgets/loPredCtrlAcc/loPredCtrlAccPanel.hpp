/** \file loPredCtrlAccPanel.hpp
 * \brief Operator controls and essential monitoring for the loPredCtrlAcc DDSPC controller.
 */
#ifndef loPredCtrlAccPanel_hpp
#define loPredCtrlAccPanel_hpp

#include <functional>
#include <map>
#include <string>
#include <vector>

#include <QString>

#include "../xWidgets/xWidget.hpp"

class QBoxLayout;
class QGridLayout;
class QLabel;
class QPushButton;

namespace xqt
{

class statusEntry;
class statusLabel;
class toggleSlider;

/// Operate loPredCtrlAcc learning and prediction and show its essential status, entirely through INDI.
/** Every control shows the latest INDI readback and sends only explicit operator requests. Nothing is sent when the
 * panel opens or closes. Mode changes and model resets are offered only in the state in which loPredCtrlAcc accepts
 * them (prediction off and its own zero confirmed), and the controller still validates every request.
 */
class loPredCtrlAccPanel : public xWidget
{
  public:
    /// Construct the controls disabled and the readbacks unavailable.
    explicit loPredCtrlAccPanel( const std::string &device = "loPredCtrlAcc", /**< [in] INDI device name. */
                                 QWidget           *parent = nullptr /**< [in] Optional Qt parent. */ );

    /// Detach nested standard controls before Qt destroys them.
    ~loPredCtrlAccPanel() noexcept override;

    /// Subscribe to the controller's properties through the shared publisher.
    void subscribe() override;

    /// Mark the INDI connection available; controls wait for the device's own readbacks.
    void onConnect() override;

    /// Clear every readback and disable every control.
    void onDisconnect() override;

    /// Apply a property definition.
    void handleDefProperty( const pcf::IndiProperty &property /**< [in] Property definition. */ ) override;

    /// Invalidate a deleted property, or everything when the whole device is deleted.
    void handleDelProperty( const pcf::IndiProperty &property /**< [in] Deletion notification. */ ) override;

    /// Merge a (possibly partial) property update and refresh the panel.
    void handleSetProperty( const pcf::IndiProperty &property /**< [in] Property update. */ ) override;

    /// Replace the reset confirmation, a modal question by default.
    void setResetConfirmation( std::function<bool()> confirm /**< [in] Returns true to send the reset. */ );

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

    std::string                     m_device;    ///< loPredCtrlAcc INDI device name.
    std::map<std::string, Readback> m_readbacks; ///< Controller properties by name.

    std::vector<Row> m_rows; ///< Status rows refreshed together.

    toggleSlider *m_learnStd{ nullptr }; ///< learn_std toggle.
    toggleSlider *m_learn{ nullptr };    ///< learn toggle.
    toggleSlider *m_predict{ nullptr };  ///< predict toggle.

    QPushButton *m_standard{ nullptr };  ///< ddspc_mode.standard selector.
    QPushButton *m_assisted{ nullptr };  ///< ddspc_mode.accel_assisted selector.
    QPushButton *m_reset{ nullptr };     ///< reset_model request.
    QPushButton *m_zeroDm{ nullptr };    ///< zero_dm request: stop prediction and re-zero the DDSPC DM channel.
    QPushButton *m_releaseTT{ nullptr }; ///< release_tt request: stop prediction and return tip/tilt to the main loop.
    QLabel      *m_interlock{ nullptr }; ///< Why mode change and reset are unavailable.
    QLabel      *m_banner{ nullptr };    ///< Prominent prediction state.
    QLabel      *m_warnings{ nullptr };  ///< Controller and output alerts.

    statusEntry *m_gainEntry{ nullptr }; ///< Gain current value and numeric target entry.

    statusEntry *m_learnStdExtraEntry{ nullptr }; ///< Seconds of learning after the Standard sequence ends.

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

    /// Return why prediction last stopped, decoded from safety.output_state, or an empty string without a fault.
    QString faultReason() const;

    /// Return the Last stop row: the decoded fault, an operator stop, running, or "---" when unavailable.
    QString stopReason() const;

    /// Return why a prediction request was refused, from the conditions the controller publishes.
    QString refusalReason() const;

    /// Add a status row computed by text.
    void addRow( QGridLayout             *grid,  /**< [in,out] Grid receiving the row. */
                 const char              *key,   /**< [in] Object name suffix. */
                 const char              *label, /**< [in] Operator-facing label. */
                 std::function<QString()> text /**< [in] Text from the current readbacks. */ );

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

    /// Send a complete ddspc_mode selection with exactly one element on.
    void sendMode( const char *selected /**< [in] Element to select. */ );

    /// Ask for confirmation, then request reset_model.
    void requestReset();

    /// Send a request switch's On request when the controller is operating and has defined it.
    void sendRequest( const char *property /**< [in] Request switch property. */ );

    /// Recompute every readback, indicator and control availability.
    void refresh();
};

} // namespace xqt

#endif // loPredCtrlAccPanel_hpp
