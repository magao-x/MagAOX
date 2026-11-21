/** \file timingPanel.hpp
 * \brief MagAO-X syncopation controls and timing readbacks.
 */
#ifndef timingPanel_hpp
#define timingPanel_hpp

#include <array>
#include <functional>
#include <optional>
#include <vector>

#include <QString>

#include "../xWidgets/xWidget.hpp"

class QBoxLayout;

namespace xqt
{

class statusLabel;

/// Operate syncopation and its waveform providers beside the timing readbacks.
class timingPanel : public xWidget
{
  public:
    /// Construct the controls and readbacks in their unavailable state.
    explicit timingPanel( QWidget *parent = nullptr /**< [in] Optional Qt parent. */ );

    /// Detach nested standard controls before Qt destroys them.
    ~timingPanel() noexcept override;

    /// Subscribe to the controls and measurements through the shared INDI publisher.
    void subscribe() override;

    /// Clear readbacks and disable controls when the publisher disconnects.
    void onDisconnect() override;

    /// Apply an initial property definition to the corresponding widget.
    void handleDefProperty( const pcf::IndiProperty &property /**< [in] Property definition. */ ) override;

    /// Clear controls and measurements invalidated by device or property deletion.
    void handleDelProperty( const pcf::IndiProperty &property /**< [in] Deletion notification. */ ) override;

    /// Refresh controls and timing labels from the matching property.
    void handleSetProperty( const pcf::IndiProperty &property /**< [in] Property update. */ ) override;

  private:
    /// Route an operator control or file readback and its invalidation to a widget.
    struct Control
    {
        const char          *m_device;   ///< Device owning the control.
        const char          *m_property; ///< INDI property name.
        multiIndiSubscriber *m_widget;   ///< Standard widget sending commands through this panel, when applicable.
        std::function<void( const pcf::IndiProperty & )> m_update; ///< Apply an available property update.
        std::function<void()>                            m_clear;  ///< Restore the disconnected state.
    };

    /// Select units and the availability rule for a timing readback.
    enum class MeasurementKind
    {
        rate,
        syncoRate,
        latency
    };

    /// Retain timing values across partial property updates.
    struct Measurement
    {
        const char           *m_label;         ///< Operator-facing row label.
        const char           *m_device;        ///< Device publishing the measurement.
        const char           *m_property;      ///< Numeric INDI property name.
        const char           *m_valueElement;  ///< Rate or mean latency element.
        const char           *m_jitterElement; ///< Deviation element in seconds, or null for a rate-only row.
        MeasurementKind       m_kind{ MeasurementKind::rate }; ///< Units and availability of the primary value.
        bool                  m_valid{ false };                ///< Last validity reported for the WMA population.
        statusLabel          *m_value{ nullptr };              ///< Rate or mean display owned by the timing group.
        statusLabel          *m_jitter{ nullptr };             ///< Jitter display owned by the timing group.
        std::optional<double> m_lastValue{};                   ///< Last valid numeric rate or latency value from INDI.
        std::optional<double> m_lastJitter{};                  ///< Last valid numeric jitter value from INDI.
    };

    std::vector<Control> m_controls; ///< INDI bindings for controls and file readback.

    pcf::IndiElement::SwitchStateType m_syncoEnable{ pcf::IndiElement::UnknownSwitchState }; ///< Synco switch readback.
    QString                           m_syncoReason;            ///< Last phase reported by synco health.
    int                               m_syncoSteps{ 0 };        ///< Active steps from the last timing update.
    statusLabel                      *m_syncoStatus{ nullptr }; ///< Phase and step count displayed below the slider.

    std::array<Measurement, 4> m_measurements{ {
        { "camwfs acquisition", "camwfs", "fg_timing", "acq_fps", "acq_jitter" },
        { "synco delivered", "dmtweeter-synco", "timing", "deliveredHz", nullptr, MeasurementKind::syncoRate },
        { "dmtweeter write", "dmtweeter", "dm_timing", "send_fps", "send_jitter" },
        { "holoop-time WMA", "holoop-time", "loop_timing", "mean", "stddev", MeasurementKind::latency },
    } }; ///< Acquisition, confirmed synco delivery, DM write cadence and camera-to-SDK latency.

    /// Add a standard INDI toggle slider.
    void addToggle( QBoxLayout *layout,   /**< [in,out] Layout receiving the slider. */
                    const char *device,   /**< [in] Device owning the switch. */
                    const char *property, /**< [in] INDI switch property. */
                    const char *label,    /**< [in] Operator-facing label. */
                    bool        syncoEnable = false /**< [in] Bind the synco enable state and delivered readback. */ );

    /// Add a standard numeric current/target entry.
    void addEntry( QBoxLayout *layout,   /**< [in,out] Layout receiving the entry. */
                   const char *device,   /**< [in] Device owning the number. */
                   const char *property, /**< [in] INDI numeric property. */
                   const char *label,    /**< [in] Operator-facing label. */
                   const char *units,    /**< [in] Display units, if any. */
                   int         type /**< [in] Standard status entry number type. */ );

    /// Add a momentary INDI request button.
    void addRequest( QBoxLayout *layout,   /**< [in,out] Layout receiving the button. */
                     const char *device,   /**< [in] Device owning the request. */
                     const char *property, /**< [in] INDI switch property. */
                     const char *label,    /**< [in] Button text. */
                     const char *tooltip /**< [in] Action or prerequisite description. */ );

    /// Add a group of controls shared by the two dmSpeckle providers.
    void addModulator( QBoxLayout *layout, /**< [in,out] Layout receiving the group. */
                       const char *device /**< [in] Provider's INDI device name. */ );

    /// Display the current waveform file name with its full path in a tooltip.
    void addFileReadback( QBoxLayout *layout, /**< [in,out] Layout receiving the file readback. */
                          const char *device /**< [in] Provider's INDI device name. */ );

    /// Update the displayed rate or latency and jitter with the configured units.
    void showMeasurement( const Measurement &measurement /**< [in] Current measurement state. */ );

    /// Apply the synco enable readback to the delivered rate and status.
    void setSyncoEnable( pcf::IndiElement::SwitchStateType state /**< [in] Reported enable switch state. */ );

    /// Show the reported synco phase and active step count.
    void updateSyncoStatus();

    /// Clear stored data and the associated labels.
    void clearMeasurement( Measurement &measurement /**< [in,out] Measurement being invalidated. */ );
};

} // namespace xqt

#endif // timingPanel_hpp
