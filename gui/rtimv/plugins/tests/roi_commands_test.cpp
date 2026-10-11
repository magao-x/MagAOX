/** \file roi_commands_test.cpp
 * \brief Offline command routing, INDI lifecycle, and interactive ROI authorization tests.
 * \ingroup rtimvROI_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"
#include <QApplication>
#include <QThread>
#include "../cameraStatus/cameraStatus.hpp"
#include "../indiDictionary/indiDictionary.hpp"

namespace libXWCTest
{
namespace rtimvROITest
{
/// \cond TEST_HARNESS
using Region = MagAOX::roi::region;

/// GUI host exposing the optional slots, without any image server or instrument connection.
class Host : public QObject
{
    Q_OBJECT
  public:
    QGraphicsScene m_scene; ///< Owns graphics items in the test.
    QSize m_size{64, 64}; ///< Currently displayed image size.
    QString m_message; ///< Last normal status-box message.
    rtimvCommandRouter m_router{this}; ///< Same router used by the production main window.
  public slots:
    /// Return the displayed dimensions.
    QSize overlayImageSize() const;
    /// Return a representative zoomed/panned new-user-box placement.
    QRectF newUserBoxRect() const;
    /// Add the ROI to its owning scene.
    void addInteractiveBox(StretchBox *box /**< [in] New box. */);
    /// Record a normal status message.
    void showStatusMessage(const QString &message /**< [in] Interaction result. */);
};
QSize Host::overlayImageSize() const { return m_size; }
QRectF Host::newUserBoxRect() const { return QRectF(24, 24, 16, 16); }
void Host::addInteractiveBox(StretchBox *box) { m_scene.addItem(box); }
void Host::showStatusMessage(const QString &message) { m_message = message; }

/// Real provider with a captured, explicitly offline transport.
class Provider : public indiDictionary
{
  public:
    bool m_connected{true}; ///< Availability of the fake transport.
    int m_throwAt{-1}; ///< Simulated send failure after this many successful vectors.
    std::vector<pcf::IndiProperty> m_sent; ///< Exact outgoing vectors.
    /// Report the offline connection state.
    bool commandConnected() const override;
    /// Capture or reject one typed vector.
    void writeCommand(const pcf::IndiProperty &property /**< [in] Outgoing vector. */) override;
    /// Deliver a definition or update through the real provider parser.
    void receive(const pcf::IndiProperty &property, /**< [in] Incoming vector. */
                 bool definition = true /**< [in] Definition versus update. */);
    /// Disconnect and advance the real provider generation.
    void disconnectTransport();
    using indiDictionary::m_generation;
};
bool Provider::commandConnected() const { return m_connected; }
void Provider::writeCommand(const pcf::IndiProperty &property)
{
    if(m_throwAt == static_cast<int>(m_sent.size())) throw std::runtime_error("offline failure");
    m_sent.push_back(property);
}
void Provider::receive(const pcf::IndiProperty &property, bool definition) { receiveProperty(property, m_generation, definition); }
void Provider::disconnectTransport() { m_connected = false; invalidateConnection(); }

/// Construct a complete numeric definition with explicit permissions and state.
pcf::IndiProperty numbers(const std::string &name, /**< [in] Property name. */
                          const std::vector<std::string> &names, /**< [in] Element names. */
                          const std::vector<double> &values, /**< [in] Element values. */
                          bool writable = false /**< [in] Permit command submission. */)
{
    pcf::IndiProperty p(pcf::IndiProperty::Number, "cam", name);
    p.setPerm(writable ? pcf::IndiProperty::ReadWrite : pcf::IndiProperty::ReadOnly);
    p.setState(pcf::IndiProperty::Ok);
    for(size_t n = 0; n < names.size(); ++n) { p.add(pcf::IndiElement(names[n])); p[names[n]] = values[n]; }
    return p;
}

/// Expose the real interaction handlers without sending Qt events to hardware-connected applications.
class Overlay : public cameraStatus
{
  public:
    /// Attach the test host's slots to the real ROI signals.
    explicit Overlay(Host &host /**< [in] Offline GUI host. */);
    using cameraStatus::updateROI;
    using cameraStatus::beginROI;
    using cameraStatus::applyROI;
    using cameraStatus::escapeROI;
    using cameraStatus::roiEditStarted;
    using cameraStatus::roiEditFinished;
    using cameraStatus::roiRect;
    using cameraStatus::m_roiBox;
    using cameraStatus::m_roiPhase;
    using cameraStatus::m_roiReady;
    using cameraStatus::m_explicitROI;
    using cameraStatus::m_roiTimeout;
    using cameraStatus::roiPhase;
};
Overlay::Overlay(Host &host)
{
    m_enabled = m_enableable = true;
    m_deviceName = "cam";
    m_roa.m_mainWindowObject = &host;
    connect(this, &cameraStatus::newInteractiveBox, &host, &Host::addInteractiveBox);
    connect(this, &cameraStatus::roiMessage, &host, &Host::showStatusMessage);
}

/// Camera property fixture; no sockets, FIFOs, or INDI threads are created.
struct Fixture
{
    Host m_host;
    Provider m_provider;
    Overlay m_overlay{m_host};
    Region m_current{511.5, 511.5, 64, 64, 1, 1};
    Region m_target{m_current};
    int m_units{0};
    int m_sequence{0};
    bool m_preset{false};
    /// Publish command definitions and an initial coherent camera state.
    Fixture();
    /// Publish all six current and target values together.
    void state();
    /// Apply captured target commands and deliver coherent target echoes.
    void echo();
    /// Complete one validator invocation, optionally adjusting dimensions or binning.
    void checked(int width = 0, /**< [in] Optional validator-adjusted width. */
                 int bin = 0, /**< [in] Optional validator-adjusted horizontal binning. */
                 bool fail = false /**< [in] Return an Alert completion. */);
    /// Count request vectors captured by the provider.
    int count(const std::string &property /**< [in] Property name. */) const;
};
Fixture::Fixture()
{
    m_host.m_router.addProvider(&m_provider);
    for(const auto &name : MagAOX::roi::names()) m_provider.receive(numbers("roi_region_"+name, {"current", "target"}, {0, 0}, true));
    for(const auto &name : {"roi_region_check", "roi_set"})
    {
        pcf::IndiProperty p(pcf::IndiProperty::Switch, "cam", name);
        p.setPerm(pcf::IndiProperty::ReadWrite); p.setState(pcf::IndiProperty::Ok);
        p.add(pcf::IndiElement("request")); p["request"].setSwitchState(pcf::IndiElement::Off);
        m_provider.receive(p);
    }
    checked();
    m_overlay.updateROI();
}
void Fixture::state()
{
    auto names = MagAOX::roi::names(), targetNames = MagAOX::roi::names("target_");
    auto values = MagAOX::roi::values(m_current), targets = MagAOX::roi::values(m_target);
    names.insert(names.end(), targetNames.begin(), targetNames.end());
    values.insert(values.end(), targets.begin(), targets.end());
    names.insert(names.end(), {"units", "is_full", "is_default", "preset_known"});
    values.insert(values.end(), {double(m_units), double(m_preset), double(m_preset), 1});
    m_provider.receive(numbers("roi_region_state", names, values));
}
void Fixture::echo()
{
    for(const auto &p : m_provider.m_sent)
    {
        if(!p.find("target")) continue;
        double v = p["target"].get<double>();
        if(p.getName() == "roi_region_x") m_target.x = v;
        if(p.getName() == "roi_region_y") m_target.y = v;
        if(p.getName() == "roi_region_w") m_target.w = v;
        if(p.getName() == "roi_region_h") m_target.h = v;
    }
    state();
    m_overlay.updateROI();
}
void Fixture::checked(int width, int bin, bool fail)
{
    auto requested = m_target;
    if(width) m_target.w = width;
    if(bin) m_target.bin_x = bin;
    state();
    std::vector<std::string> names;
    std::vector<double> values;
    for(const auto &[prefix, roi] : {std::make_pair("requested_", requested), std::make_pair("checked_", m_target), std::make_pair("current_", m_current)})
    {
        auto n = MagAOX::roi::names(prefix); auto v = MagAOX::roi::values(roi);
        names.insert(names.end(), n.begin(), n.end()); values.insert(values.end(), v.begin(), v.end());
    }
    names.push_back("sequence"); values.push_back(++m_sequence);
    auto p = numbers("roi_check_result", names, values);
    if(fail) p.setState(pcf::IndiProperty::Alert);
    m_provider.receive(p);
}
int Fixture::count(const std::string &property) const
{
    return std::count_if(m_provider.m_sent.begin(), m_provider.m_sent.end(), [&](const auto &p){ return p.getName() == property; });
}
/// \endcond

/** \brief The optional router rejects missing, ambiguous, stale, invalid, and read-only commands without replay.
 * \ingroup rtimvROI_unit_test
 */
TEST_CASE("ROI command provider validates typed batches and lifecycle", "[rtimvROI]")
{
    // clang-format off
#ifdef RTIMV_ROI_TEST_DOXYGEN_REF
    indiDictionary::sendCommands; indiDictionary::receiveProperty;
    rtimvCommandRouter::send; rtimvCommandRouter::snapshot;
#endif
    // clang-format on
    Host host;
    QString error;
    rtimvCommand command{"cam", "vector", rtimvPropertyType::number, {{"x", 0.0}, {"y", -2.5}}};
    CHECK_FALSE(host.m_router.send("indi", {command}, error));
    Provider provider;
    host.m_router.addProvider(&provider);
    CHECK_FALSE(host.m_router.send("indi", {command}, error));
    auto property = numbers("vector", {"x", "y"}, {1, 2}, true);
    provider.receive(property);
    REQUIRE(host.m_router.send("indi", {command}, error));
    REQUIRE(provider.m_sent.size() == 1);
    CHECK(provider.m_sent[0]["x"].get<double>() == 0);
    CHECK(provider.m_sent[0]["y"].get<double>() == -2.5);
    auto invalid = command; invalid.m_elements["unknown"] = 2.;
    CHECK_FALSE(host.m_router.send("indi", {command, invalid}, error));
    for(const QVariant &value : {QVariant(true), QVariant("12"), QVariant(std::numeric_limits<double>::infinity()), QVariant(std::numeric_limits<double>::quiet_NaN())})
    {
        invalid = command; invalid.m_elements["x"] = value;
        CHECK_FALSE(host.m_router.send("indi", {invalid}, error));
    }
    CHECK(provider.m_sent.size() == 1);
    property.setPerm(pcf::IndiProperty::ReadOnly); provider.receive(property);
    CHECK_FALSE(host.m_router.send("indi", {command}, error));
    property.setPerm(pcf::IndiProperty::ReadWrite); provider.receive(property);
    {
        Provider duplicate; host.m_router.addProvider(&duplicate);
        CHECK_FALSE(host.m_router.send("indi", {command}, error));
    }
    CHECK(host.m_router.send("indi", {command}, error));
    provider.receive(numbers("vector", {"x"}, {3}), false);
    CHECK_FALSE(host.m_router.snapshot("indi", "cam", "vector").m_complete);
    provider.receive(property);
    CHECK(host.m_router.snapshot("indi", "cam", "vector").m_complete);
    provider.receiveProperty(property, provider.m_generation + 1, true);
    CHECK(host.m_router.snapshot("indi", "cam", "vector").m_elements["x"].toDouble() == 1);
    provider.receiveProperty(property, provider.m_generation, false, true);
    CHECK_FALSE(host.m_router.send("indi", {command}, error));
    provider.receive(property);
    provider.disconnectTransport();
    CHECK_FALSE(host.m_router.snapshot("indi", "cam", "vector").m_valid);
    CHECK_FALSE(host.m_router.send("indi", {command}, error));
    provider.m_connected = true;
    CHECK_FALSE(host.m_router.send("indi", {command}, error));
    provider.receive(property);
    auto sent = provider.m_sent.size(); provider.m_throwAt = sent + 1;
    CHECK_FALSE(host.m_router.send("indi", {command, command, command}, error));
    CHECK(provider.m_sent.size() == sent + 1);
    CHECK_FALSE(host.m_router.snapshot("indi", "cam", "vector").m_valid);
}

/** \brief Repeated Shift+R preserves custom geometry and mouse release sends only intentionally edited fields.
 * \ingroup rtimvROI_unit_test
 */
TEST_CASE("ROI edit entry preserves custom regions and stages preset boxes", "[rtimvROI]")
{
    // clang-format off
#ifdef RTIMV_ROI_TEST_DOXYGEN_REF
    cameraStatus::beginROI; cameraStatus::stageROI; cameraStatus::roiEditFinished;
#endif
    // clang-format on
    Fixture f;
    f.m_overlay.beginROI();
    REQUIRE(f.m_overlay.m_roiBox);
    CHECK(f.m_overlay.m_roiBox->rect() == QRectF(0,0,64,64));
    CHECK(f.m_overlay.m_roiBox->isVisible());
    CHECK(f.m_provider.m_sent.empty());
    f.m_overlay.beginROI();
    CHECK(f.m_provider.m_sent.empty());
    f.m_overlay.roiEditStarted(f.m_overlay.m_roiBox);
    f.m_overlay.m_roiBox->setPos(2, -3);
    CHECK(f.m_provider.m_sent.empty());
    f.m_overlay.roiEditFinished(f.m_overlay.m_roiBox);
    REQUIRE(f.m_provider.m_sent.size() == 2);
    CHECK(f.count("roi_region_x") == 1);
    CHECK(f.count("roi_region_y") == 1);
    CHECK(f.count("roi_region_w") == 0);
    CHECK(f.count("roi_region_h") == 0);
    f.echo();
    CHECK(f.m_target.x == 513.5);
    CHECK(f.m_target.y == 514.5);
    f.m_overlay.escapeROI();
    CHECK(f.m_overlay.m_roiBox->isVisible()); // already staged target retained
    f.m_overlay.beginROI();
    CHECK(f.m_provider.m_sent.size() == 2);

    Fixture preset; preset.m_preset = true; preset.state();
    preset.m_overlay.beginROI();
    CHECK(preset.count("roi_region_w") == 1);
    CHECK(preset.count("roi_region_h") == 1);
    CHECK(preset.count("roi_region_bin_x") == 0);
    CHECK(preset.count("roi_region_bin_y") == 0);
    auto sent = preset.m_provider.m_sent.size();
    preset.m_overlay.beginROI();
    CHECK(preset.m_provider.m_sent.size() == sent);
    preset.echo();
    CHECK(preset.m_target.w == 16);
    CHECK(preset.m_target.h == 16);
}

/** \brief Adjusted Check results require a fresh post-display action; repeated keys cannot queue Set.
 * \ingroup rtimvROI_unit_test
 */
TEST_CASE("ROI apply reviews each adjustment and performs one Set", "[rtimvROI]")
{
    // clang-format off
#ifdef RTIMV_ROI_TEST_DOXYGEN_REF
    cameraStatus::applyROI; cameraStatus::updateROI; cameraStatus::keyPressEvent;
#endif
    // clang-format on
    Fixture f; f.m_target.w = 40; f.state(); f.m_overlay.beginROI();
    f.m_overlay.applyROI();
    CHECK(f.count("roi_region_check") == 1);
    f.checked(32);
    f.m_overlay.applyROI(); // arrived before adjustment was displayed: discarded
    CHECK(f.count("roi_region_check") == 1);
    f.m_overlay.updateROI();
    CHECK(f.count("roi_set") == 0);
    CHECK(f.m_host.m_message == "ROI adjusted; Alt+R to apply");
    CHECK(f.m_overlay.m_explicitROI);
    CHECK(f.m_overlay.m_roiBox->rect().width() == 32);
    QKeyEvent repeat(QEvent::KeyPress, Qt::Key_R, Qt::AltModifier, "r", true);
    f.m_overlay.keyPressEvent(&repeat);
    CHECK(f.count("roi_region_check") == 1);
    f.m_overlay.applyROI(); f.checked(24); f.m_overlay.updateROI();
    CHECK(f.count("roi_set") == 0);
    f.m_overlay.applyROI(); f.checked(); f.m_overlay.updateROI();
    REQUIRE(f.count("roi_set") == 1);
    f.m_overlay.applyROI();
    CHECK(f.count("roi_set") == 1);
    f.m_overlay.escapeROI();
    CHECK(f.m_host.m_message == "ROI Set already submitted");
    f.m_current = f.m_target; f.m_host.m_size = QSize(24,64); f.state(); f.m_overlay.updateROI();
    CHECK(f.m_host.m_message == "ROI applied");
    CHECK_FALSE(f.m_overlay.m_explicitROI);
    CHECK_FALSE(f.m_overlay.m_roiBox->isVisible());
}

/** \brief Echoes, failures, changed references, and Escape cannot authorize stale or implicit binning changes.
 * \ingroup rtimvROI_unit_test
 */
TEST_CASE("ROI cancellation and stale validation do not apply", "[rtimvROI]")
{
    // clang-format off
#ifdef RTIMV_ROI_TEST_DOXYGEN_REF
    cameraStatus::escapeROI; cameraStatus::failROI; cameraStatus::updateROI;
#endif
    // clang-format on
    SECTION("already applied is a no-op")
    {
        Fixture f; f.m_overlay.beginROI(); f.m_overlay.applyROI(); f.checked(); f.m_overlay.updateROI();
        CHECK(f.count("roi_set") == 0); CHECK_FALSE(f.m_overlay.m_explicitROI);
    }
    SECTION("target echoes precede Check")
    {
        Fixture f; f.m_preset = true; f.state(); f.m_overlay.beginROI(); f.m_overlay.applyROI();
        CHECK(f.count("roi_region_check") == 0);
        f.echo(); CHECK(f.count("roi_region_check") == 1);
        f.m_overlay.escapeROI(); f.checked(); f.m_overlay.updateROI(); CHECK(f.count("roi_set") == 0);
    }
    SECTION("changed binning requires roiGUI")
    {
        Fixture f; f.m_target.w = 32; f.state(); f.m_overlay.beginROI(); f.m_overlay.applyROI();
        f.checked(0,2); f.m_overlay.updateROI(); CHECK(f.count("roi_set") == 0);
        CHECK(f.m_host.m_message == "Binning adjusted; use roiGUI");
    }
    SECTION("fresh check with a different reference is rejected")
    {
        Fixture f; f.m_target.w = 32; f.state(); f.m_overlay.beginROI(); f.m_overlay.applyROI();
        f.m_current.x += 1; f.checked(); f.m_overlay.updateROI(); CHECK(f.count("roi_set") == 0);
    }
    SECTION("partial completion is rejected")
    {
        Fixture f; f.m_target.w = 32; f.state(); f.m_overlay.beginROI(); f.m_overlay.applyROI();
        f.m_provider.receive(numbers("roi_check_result", {"sequence"}, {99}), false);
        f.m_overlay.updateROI(); CHECK(f.count("roi_set") == 0);
    }
    SECTION("failed validator is rejected")
    {
        Fixture f; f.m_target.w = 32; f.state(); f.m_overlay.beginROI(); f.m_overlay.applyROI();
        f.checked(0,0,true); f.m_overlay.updateROI(); CHECK(f.count("roi_set") == 0);
    }
    SECTION("timeout revokes authorization")
    {
        Fixture f; f.m_target.w = 32; f.state(); f.m_overlay.beginROI(); f.m_overlay.applyROI();
        f.m_overlay.m_roiTimeout = 0;
        QThread::msleep(2); f.m_overlay.updateROI(); f.checked(); f.m_overlay.updateROI(); CHECK(f.count("roi_set") == 0);
    }
    SECTION("disconnect and reconnect never replay")
    {
        Fixture f; f.m_target.w = 32; f.state(); f.m_overlay.beginROI(); f.m_overlay.applyROI();
        f.m_provider.disconnectTransport(); f.m_overlay.updateROI(); f.m_provider.m_connected = true;
        f.checked(); f.m_overlay.updateROI(); CHECK(f.count("roi_set") == 0); CHECK_FALSE(f.m_overlay.m_roiReady);
    }
    SECTION("Escape discards a local drag")
    {
        Fixture f; f.m_overlay.beginROI(); f.m_overlay.roiEditStarted(f.m_overlay.m_roiBox);
        f.m_overlay.m_roiBox->setPos(10,10); f.m_overlay.escapeROI(); f.m_overlay.roiEditFinished(f.m_overlay.m_roiBox);
        CHECK(f.m_provider.m_sent.empty()); CHECK_FALSE(f.m_overlay.m_roiBox->isVisible());
    }
}

/** \brief Native binning policies map to image pixel edges, including center zero and unequal pending bins.
 * \ingroup rtimvROI_unit_test
 */
TEST_CASE("ROI rectangle mapping respects camera units and image identity", "[rtimvROI]")
{
    Fixture f;
    for(int units : {0,1}) for(int bx : {1,2,3}) for(int by : {1,2,3})
    {
        f.m_units = units;
        f.m_current = {0,0, units ? 64 : 64*bx, units ? 64 : 64*by, bx, by};
        f.m_target = f.m_current; f.state(); f.m_overlay.updateROI();
        REQUIRE(f.m_overlay.m_roiReady);
        CHECK(f.m_overlay.roiRect() == QRectF(0,0,64,64));
        f.m_target.bin_x = bx+1; f.state(); f.m_overlay.updateROI();
        CHECK(std::isfinite(f.m_overlay.roiRect().center().x()));
    }
    f.m_host.m_size = QSize(65,64); f.m_overlay.updateROI(); CHECK_FALSE(f.m_overlay.m_roiReady);
}
}
}

/// Run only offline Qt/Catch tests on the current workstation.
int main(int argc, /**< [in] Argument count. */ char **argv /**< [in] Catch options. */)
{
    QApplication app(argc, argv);
    return Catch::Session().run(argc, argv);
}
#include "roi_commands_test.moc"
