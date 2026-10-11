/** \file roi_test.cpp
 * \brief Offline relative-center display, capability, edit cancellation, and command tests.
 * \ingroup roiGUI_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"
#include <QApplication>
#include "../roi.hpp"

namespace libXWCTest
{
namespace roiGUITest
{
/// \cond TEST_HARNESS
/// Capture widget commands locally without an INDI publisher.
class Widget : public xqt::roi
{
  public:
    std::vector<pcf::IndiProperty> m_sent; ///< Typed outgoing commands.
    /// Create an offline ROI widget.
    explicit Widget(std::string &name /**< [in] Camera device. */);
    /// Capture one command in memory.
    void sendNewProperty(const pcf::IndiProperty &property /**< [in] Outgoing vector. */) override;
    /// Deliver a numeric property and refresh the widget.
    void number(const std::string &name, /**< [in] Property name. */
                const std::vector<std::string> &names, /**< [in] Element names. */
                const std::vector<double> &values /**< [in] Element values. */);
    /// Deliver a valid applied reference with a selectable horizontal center.
    void reference(double x = 511.5 /**< [in] Applied absolute center. */);
    /// Define a supported relative camera.
    void ready();
};
Widget::Widget(std::string &name) : roi(name) {}
void Widget::sendNewProperty(const pcf::IndiProperty &property) { m_sent.push_back(property); }
void Widget::number(const std::string &name, const std::vector<std::string> &names, const std::vector<double> &values)
{
    pcf::IndiProperty p(pcf::IndiProperty::Number, "cam", name);
    p.setState(pcf::IndiProperty::Ok);
    p.setPerm(pcf::IndiProperty::ReadWrite);
    for(size_t n = 0; n < names.size(); ++n) { p.add(pcf::IndiElement(names[n])); p[names[n]] = values[n]; }
    handleDefProperty(p); updateGUI();
}
void Widget::reference(double x)
{
    number("roi_region_state", {"x","y","w","h","bin_x","bin_y","units"}, {x, 511.5, 64,64,1,1,0});
}
void Widget::ready()
{
    onConnect();
    pcf::IndiProperty state(pcf::IndiProperty::Text, "cam", "fsm");
    state.add(pcf::IndiElement("state", "READY")); handleSetProperty(state);
    number("roi_region_x", {"current", "target"}, {511.5, 520.5});
    number("roi_region_y", {"current", "target"}, {511.5, 510.5});
    number("roi_region_w", {"current", "target"}, {64, 32});
    number("roi_region_h", {"current", "target"}, {64, 32});
    number("roi_region_bin_x", {"current", "target"}, {1, 1});
    number("roi_region_bin_y", {"current", "target"}, {1, 1});
    reference();
    number("roi_region_x_relative", {"target"}, {40.5});
    number("roi_region_y_relative", {"target"}, {30.5});
}
/// \endcond

/** \brief Relative is the supported default; current fields stay absolute and toggling never submits text.
 * \ingroup roiGUI_unit_test
 */
TEST_CASE("ROI widget chooses the displayed coordinate property", "[roiGUI]")
{
    // clang-format off
#ifdef ROI_GUI_TEST_DOXYGEN_REF
    xqt::roi::sendCenter; xqt::roi::on_check_relative_toggled; xqt::roi::updateGUI;
#endif
    // clang-format on
    std::string name = "cam"; Widget widget(name); widget.ready();
    auto *mode = widget.findChild<QCheckBox *>("check_relative");
    auto *x = widget.findChild<xqt::statusLineEdit *>("le_center_x");
    auto *current = widget.findChild<xqt::statusLabel *>("val_center_x");
    REQUIRE(mode); REQUIRE(x); REQUIRE(current);
    CHECK(mode->isChecked()); CHECK(mode->isEnabled());
    CHECK(x->text().toDouble() == 40.5);
    CHECK(current->text().toDouble() == 511.5);
    x->setEditText("0"); widget.on_le_center_x_editingFinished();
    REQUIRE(widget.m_sent.size() == 1);
    CHECK(widget.m_sent.back().getName() == "roi_region_x_relative");
    CHECK(widget.m_sent.back()["target"].get<double>() == 0);
    x->setEditText("123"); mode->setChecked(false);
    CHECK(widget.m_sent.size() == 1);
    CHECK(x->text().toDouble() == 520.5);
    x->setEditText("500.5"); widget.on_le_center_x_editingFinished();
    REQUIRE(widget.m_sent.size() == 2);
    CHECK(widget.m_sent.back().getName() == "roi_region_x");
    mode->setChecked(true);
    for(const auto &text : {"", "nan", "inf", "12bad"})
    { x->setEditText(text); widget.on_le_center_x_editingFinished(); }
    CHECK(widget.m_sent.size() == 2);
    widget.on_button_reset_pressed();
    REQUIRE(widget.m_sent.size() == 8);
    CHECK(widget.m_sent[4].getName() == "roi_region_x");
    CHECK(widget.m_sent[4]["target"].get<double>() == 511.5);
    widget.on_button_default_pressed();
    CHECK(widget.m_sent.back().getName() == "roi_set_default");
}

/** \brief Capability loss and reference changes discard unsent center edits before another mode can use them.
 * \ingroup roiGUI_unit_test
 */
TEST_CASE("ROI widget rejects stale reference edits and old camera capabilities", "[roiGUI]")
{
    // clang-format off
#ifdef ROI_GUI_TEST_DOXYGEN_REF
    xqt::roi::handleSetProperty; xqt::roi::handleDelProperty; xqt::roi::onDisconnect;
#endif
    // clang-format on
    std::string name = "cam"; Widget widget(name); widget.ready();
    auto *mode = widget.findChild<QCheckBox *>("check_relative");
    auto *x = widget.findChild<xqt::statusLineEdit *>("le_center_x");
    x->setEditText("20"); widget.reference(512.5);
    CHECK(x->editText().isEmpty());
    widget.on_le_center_x_editingFinished(); CHECK(widget.m_sent.empty());
    pcf::IndiProperty deleted(pcf::IndiProperty::Number, "cam", "roi_region_state");
    x->setEditText("12"); widget.handleDelProperty(deleted); widget.on_le_center_x_editingFinished();
    CHECK(widget.m_sent.empty()); CHECK_FALSE(mode->isEnabled()); CHECK_FALSE(mode->isChecked());
    CHECK(x->text().toDouble() == 520.5);
    widget.ready(); CHECK(mode->isChecked());
    widget.onDisconnect(); CHECK_FALSE(mode->isEnabled()); CHECK_FALSE(mode->isChecked());
    widget.onConnect();
    widget.number("roi_region_x", {"current","target"}, {1,2});
    CHECK_FALSE(mode->isEnabled()); CHECK_FALSE(mode->isChecked());
}
}
}

/// Run the offline widget tests with the Qt platform selected by the caller.
int main(int argc, /**< [in] Argument count. */ char **argv /**< [in] Catch options. */)
{
    QApplication app(argc, argv);
    return Catch::Session().run(argc, argv);
}
