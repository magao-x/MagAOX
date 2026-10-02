/** \file flipperCtrl_test.cpp
 * \brief Behavioral tests for flipper parking, recovery, status decoding, and telemetry.
 * \author Jared R. Males (jaredmales@gmail.com)
 * \ingroup flipperCtrl_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/libMagAOX.hpp"

#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sys/socket.h>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace flipperHarness
{
/// Captured log severity and formatted message.
struct LogEntry
{
    /// Priority assigned by the real app call site.
    flatlogs::logPrioT m_priority;

    /// Message formatted by the real log type.
    std::string m_message;
};

/// Deterministic transport and filesystem failures for the production controller.
struct Faults
{
    /// Commands written to the serial transport.
    std::vector<std::string> m_commands;

    /// Complete or partial receive chunks supplied by the transport.
    std::deque<std::string> m_replies;

    /// Hook observing the installed state before a serial write.
    std::function<void(const std::string &)> m_beforeWrite;

    /// Message ID whose serial write should fail; -1 allows every write.
    int m_failedCommand{-1};

    /// Number of serial read calls.
    unsigned m_reads{0};

    /// Number of file synchronization calls.
    unsigned m_syncs{0};

    /// Synchronization call to fail; zero disables the fault.
    unsigned m_failedSync{0};

    /// Whether atomic rename should fail.
    bool m_failedRename{false};

    /// Whether file writes should fail.
    bool m_failedFileWrite{false};

    /// Whether one interrupted file write should be injected.
    bool m_interruptWrite{false};

    /// Whether file writes should return one byte at a time.
    bool m_shortWrites{false};

    /// Whether serial I/O should use the real tty utilities over a local socket.
    bool m_nativeSerial{false};

    /// Number of file/directory close calls.
    unsigned m_closes{0};

    /// Close call whose result should report failure after releasing the descriptor.
    unsigned m_failedClose{0};
};

/// Current transport and filesystem injection state.
Faults g_faults;

/// App logs captured without starting a process logger.
std::vector<LogEntry> g_logs;

/// Telemetry payloads emitted by the production recordStage implementation.
std::vector<std::vector<uint8_t>> g_telemetry;

/// Reset capture and failure state between independent tests.
void reset();

/// Form a documented APT status packet.
std::string status(uint32_t bits /**< [in] little-endian status flags */);

/// Read a state file's exact contents.
std::string contents(const std::filesystem::path &path /**< [in] state file path */);

/// Count power-on mismatch warnings in the captured logs.
size_t mismatchWarnings();

/// Own a unique temporary test directory.
class Directory
{
  public:
    /// Create a temporary directory beneath /tmp.
    Directory();

    /// Remove the test directory without throwing.
    ~Directory();

    /// Root of this test's isolated app state.
    std::filesystem::path m_path;
};
}

namespace MagAOX
{
namespace app
{
/// App base capturing calls from the controller while retaining the real configuration, FSM, and INDI support.
template<bool useINDI>
class flipperTestApp : public MagAOXApp<useINDI>
{
  public:
    /// Construct the real app base after suppressing its process logger.
    flipperTestApp(const std::string &sha /**< [in] repository revision */,
                   bool modified /**< [in] working-tree flag */);

    /// Suppress shared-library logs before the real base constructor runs.
    static const std::string &quiet(const std::string &sha /**< [in] revision passed through to the base */);

    /// Capture an application log using its real message format.
    template<typename logT, int retval = 0>
    static int log(const typename logT::messageT &msg /**< [in] log payload */,
                   logPrioT level = logPrio::LOG_DEFAULT /**< [in] requested severity */);

    /// Capture a default-constructed application log.
    template<typename logT, int retval = 0>
    static int log(logPrioT level = logPrio::LOG_DEFAULT /**< [in] requested severity */);
};

template<bool useINDI>
flipperTestApp<useINDI>::flipperTestApp(const std::string &sha, bool modified)
    : MagAOXApp<useINDI>(quiet(sha), modified)
{
}

template<bool useINDI>
const std::string &flipperTestApp<useINDI>::quiet(const std::string &sha)
{
    MagAOXApp<useINDI>::m_log.m_logLevel = logPrio::LOG_EMERGENCY;
    return sha;
}

template<bool useINDI>
template<typename logT, int retval>
int flipperTestApp<useINDI>::log(const typename logT::messageT &msg, logPrioT level)
{
    if(level == logPrio::LOG_DEFAULT) level = logT::defaultLevel;
    flipperHarness::g_logs.push_back({level, logT::msgString(msg.builder.GetBufferPointer(), msg.builder.GetSize())});
    return retval;
}

template<bool useINDI>
template<typename logT, int retval>
int flipperTestApp<useINDI>::log(logPrioT level)
{
    return log<logT, retval>(typename logT::messageT(), level);
}

namespace dev
{
/// Telemetry sink that records actual FlatBuffer payloads and supplies controllable scheduled deadlines.
template<class derivedT>
class flipperTestTelemeter
{
  public:
    /// Number of times the app invokes telemetry scheduling.
    unsigned m_schedules{0};

    /// Whether the next schedule check should force a telemetry record.
    bool m_due{false};

    /// Configure the test telemetry sink.
    int setupConfig(mx::app::appConfigurator &config /**< [in] unused app configurator */);

    /// Load test telemetry settings.
    int loadConfig(mx::app::appConfigurator &config /**< [in] unused app configurator */);

    /// Start the test telemetry sink without a background thread.
    int appStartup();

    /// Exercise the controller's real scheduling dispatch.
    int appLogic();

    /// Stop the test telemetry sink.
    int appShutdown();

    /// Apply the next injected telemetry deadline.
    int checkRecordTimes(const telem_stage &type /**< [in] stage telemetry type selector */);

    /// Capture a serialized telemetry payload.
    template<typename telT>
    int telem(const typename telT::messageT &msg /**< [in] real telemetry payload */);
};

template<class derivedT>
int flipperTestTelemeter<derivedT>::setupConfig(mx::app::appConfigurator &) { return 0; }

template<class derivedT>
int flipperTestTelemeter<derivedT>::loadConfig(mx::app::appConfigurator &) { return 0; }

template<class derivedT>
int flipperTestTelemeter<derivedT>::appStartup() { return 0; }

template<class derivedT>
int flipperTestTelemeter<derivedT>::appShutdown() { return 0; }

template<class derivedT>
int flipperTestTelemeter<derivedT>::appLogic()
{
    ++m_schedules;
    return static_cast<derivedT *>(this)->checkRecordTimes();
}

template<class derivedT>
int flipperTestTelemeter<derivedT>::checkRecordTimes(const telem_stage &type)
{
    if(!m_due) return 0;
    m_due = false;
    return static_cast<derivedT *>(this)->recordTelem(&type);
}

template<class derivedT>
template<typename telT>
int flipperTestTelemeter<derivedT>::telem(const typename telT::messageT &msg)
{
    auto *begin = msg.builder.GetBufferPointer();
    flipperHarness::g_telemetry.emplace_back(begin, begin + msg.builder.GetSize());
    return 0;
}
}
}

namespace tty
{
/// Supply a controlled serial write result while observing the real command bytes.
int flipperTestWrite(const std::string &command /**< [in] bytes sent by the controller */,
                     int fd /**< [in] descriptor used only by the native transport test */,
                     int timeout /**< [in] write timeout used only by the native transport test */);

/// Supply complete or fragmented device replies to the production packet reader.
int flipperTestRead(std::string &response /**< [out] injected receive chunk */,
                    int bytes /**< [in] requested receive length */,
                    int fd /**< [in] descriptor used only by the native transport test */,
                    int timeout /**< [in] read timeout used only by the native transport test */);

int flipperTestWrite(const std::string &command, int fd, int timeout)
{
    auto &faults = flipperHarness::g_faults;
    if(faults.m_beforeWrite) faults.m_beforeWrite(command);
    faults.m_commands.push_back(command);
    int id = static_cast<unsigned char>(command[0]) | (static_cast<unsigned char>(command[1]) << 8);
    if(id == faults.m_failedCommand) return TTY_E_ERRORONWRITE;
    return faults.m_nativeSerial ? ttyWrite(command, fd, timeout) : 0;
}

int flipperTestRead(std::string &response, int bytes, int fd, int timeout)
{
    auto &faults = flipperHarness::g_faults;
    ++faults.m_reads;
    if(faults.m_nativeSerial) return ttyRead(response, bytes, fd, timeout);
    if(bytes <= 0 || faults.m_replies.empty()) return TTY_E_TIMEOUTONREAD;
    response = faults.m_replies.front();
    faults.m_replies.pop_front();
    return 0;
}
}
}

/// Fail a selected synchronization call while otherwise syncing the real temporary file/directory.
int flipperTestSync(int fd /**< [in] file or directory descriptor */);

/// Fail atomic replacement without discarding the existing state record.
int flipperTestRename(const char *oldPath /**< [in] temporary file */,
                      const char *newPath /**< [in] installed record */);

/// Inject short, interrupted, and failed writes into the real persistence code.
ssize_t flipperTestFileWrite(int fd /**< [in] file descriptor */,
                             const void *data /**< [in] record bytes */,
                             size_t size /**< [in] byte count */);

/// Release a descriptor while injecting a selected close error.
int flipperTestClose(int fd /**< [in] descriptor to release */);

int flipperTestClose(int fd)
{
    int result = ::close(fd);
    auto &faults = flipperHarness::g_faults;
    if(++faults.m_closes == faults.m_failedClose)
    {
        errno = EIO;
        return -1;
    }
    return result;
}

int flipperTestSync(int fd)
{
    auto &faults = flipperHarness::g_faults;
    if(++faults.m_syncs == faults.m_failedSync)
    {
        errno = EIO;
        return -1;
    }
    return ::fsync(fd);
}

int flipperTestRename(const char *oldPath, const char *newPath)
{
    if(flipperHarness::g_faults.m_failedRename)
    {
        errno = EIO;
        return -1;
    }
    return ::rename(oldPath, newPath);
}

ssize_t flipperTestFileWrite(int fd, const void *data, size_t size)
{
    auto &faults = flipperHarness::g_faults;
    if(faults.m_failedFileWrite || faults.m_interruptWrite)
    {
        errno = faults.m_interruptWrite ? EINTR : EIO;
        faults.m_interruptWrite = false;
        return -1;
    }
    return ::write(fd, data, faults.m_shortWrites ? std::min(size, size_t(1)) : size);
}

// Substitute only the application header, leaving the shared library's real declarations intact.
#define MagAOXApp flipperTestApp
#define telemeter flipperTestTelemeter
#define ttyWrite flipperTestWrite
#define ttyRead flipperTestRead
#define fsync flipperTestSync
#define rename flipperTestRename
#define write flipperTestFileWrite
#define close flipperTestClose
#include "../flipperCtrl.hpp"
#undef close
#undef write
#undef rename
#undef fsync
#undef ttyRead
#undef ttyWrite
#undef telemeter
#undef MagAOXApp

namespace flipperHarness
{
void reset()
{
    g_faults = Faults();
    g_logs.clear();
    g_telemetry.clear();
}

std::string status(uint32_t bits)
{
    std::string reply("\x81\x04\x0e\x00\x81\x50\x01\x00", 8);
    reply.resize(20, '\0');
    for(unsigned i = 0; i < 4; ++i) reply[16 + i] = static_cast<char>(bits >> (8 * i));
    return reply;
}

std::string contents(const std::filesystem::path &path)
{
    std::ifstream file(path);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

size_t mismatchWarnings()
{
    return std::count_if(g_logs.begin(), g_logs.end(), [](const LogEntry &entry)
    {
        return entry.m_priority == flatlogs::logPrio::LOG_WARNING &&
               entry.m_message.find("differs from inferred parked position") != std::string::npos;
    });
}

Directory::Directory()
{
    std::string name = "/tmp/flipperCtrl-test-XXXXXX";
    auto *path = ::mkdtemp(name.data());
    if(!path) throw std::runtime_error("cannot create test directory");
    m_path = path;
}

Directory::~Directory()
{
    std::error_code error;
    std::filesystem::remove_all(m_path, error);
}
}
/// \endcond

using namespace MagAOX::app;
using namespace flipperHarness;

namespace libXWCTest
{
/** \defgroup flipperCtrl_unit_test flipperCtrl Unit Tests
 * \brief Unit tests for the flipperCtrl application.
 * \ingroup application_unit_test
 */

/// Namespace for flipperCtrl application unit tests.
/** \ingroup flipperCtrl_unit_test */
namespace flipperCtrlTest
{
/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Expose protected app state while running the real app lifecycle and helper implementations.
class Controller : public flipperCtrl
{
  public:
    /// Configure isolated state storage and initial power state.
    Controller(const std::filesystem::path &root /**< [in] test directory root */,
                const std::string &name = "flipper" /**< [in] app configuration name */);

    /// Close any descriptor left by the test.
    ~Controller();

    using flipperCtrl::decodePosition;
    using flipperCtrl::readStateFile;
    using flipperCtrl::reportedPosition;
    using flipperCtrl::saveState;
    using flipperCtrl::writeStateFile;

    /// Attach a harmless real descriptor and select the connected FSM state.
    void connected();

    /// Replace the serial descriptor with a real socket endpoint.
    void attach(int fd /**< [in] descriptor whose ownership transfers to the controller */);

    /// Set observed and target power states for lifecycle/guard tests.
    void power(int observed /**< [in] actual power */, int target /**< [in] requested power */);

    /// Enable reversed logical endpoint mapping.
    void reverse();

    /// Permit an immediate retry without waiting in a test.
    void retryNow();

    /// Return the stored target endpoint.
    int target() const;

    /// Return whether a move is pending.
    bool pending() const;

    /// Return the published parked flag.
    int parked() const;

    /// Return the backing-record path.
    std::filesystem::path path() const;

    /// Send a real callback request for a selected logical endpoint.
    int request(bool in /**< [in] select in */, bool out /**< [in] select out */);
};

Controller::Controller(const std::filesystem::path &root, const std::string &name)
{
    m_configName = name;
    m_basePath = root.string();
    m_sysPath = (root / "sys").string();
    std::filesystem::create_directories(std::filesystem::path(m_sysPath) / name);
    m_powerState = m_powerTargetState = 0;
    m_log.m_logLevel = logPrio::LOG_EMERGENCY; // Suppress shared-library logs; app calls use the capture base.
    state(stateCodes::POWEROFF);
}

Controller::~Controller()
{
    if(m_fileDescrip > 0) ::close(m_fileDescrip);
    m_fileDescrip = 0;
}

void Controller::connected()
{
    if(m_fileDescrip > 0) ::close(m_fileDescrip);
    m_fileDescrip = ::open("/dev/null", O_RDWR);
    REQUIRE(m_fileDescrip > 0);
    m_powerState = m_powerTargetState = 1;
    state(stateCodes::CONNECTED);
}

void Controller::attach(int fd)
{
    if(m_fileDescrip > 0) ::close(m_fileDescrip);
    m_fileDescrip = fd;
}

void Controller::power(int observed, int target)
{
    m_powerState = observed;
    m_powerTargetState = target;
    if(observed == 0) state(stateCodes::POWEROFF);
}

void Controller::reverse() { m_inPos = 2; m_outPos = 1; }

void Controller::retryNow() { m_nextSave = std::chrono::steady_clock::time_point::min(); }

int Controller::target() const { return m_tgt; }

bool Controller::pending() const { return m_movePending; }

int Controller::parked() const { return m_indiP_parked["current"].get<int>(); }

std::filesystem::path Controller::path() const { return std::filesystem::path(m_sysPath) / m_configName / "position"; }

int Controller::request(bool in, bool out)
{
    pcf::IndiProperty request = m_indiP_position;
    request["in"].setSwitchState(in ? pcf::IndiElement::On : pcf::IndiElement::Off);
    request["out"].setSwitchState(out ? pcf::IndiElement::On : pcf::IndiElement::Off);
    return newCallBack_m_indiP_position(request);
}
/// \endcond

/// Validate endpoint/motion masks and reject malformed or contradictory status packets.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper status replies distinguish settled endpoints and motion", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::decodePosition(std::string(), int(), bool());
    #endif
    // clang-format on
    int pos = -1;
    bool moving = false;
    for(uint32_t bits : {1u, 2u, 0x80000501u, 0x80000502u})
    {
        REQUIRE(Controller::decodePosition(status(bits), pos, moving) == 0);
        REQUIRE(pos == static_cast<int>(bits & 3));
        REQUIRE_FALSE(moving);
    }
    for(uint32_t bits : {0u, 0x10u, 0x21u, 0x42u, 0x81u, 0x200u})
    {
        REQUIRE(Controller::decodePosition(status(bits), pos, moving) == 0);
        REQUIRE(pos == 0);
        REQUIRE(moving);
    }
    REQUIRE(Controller::decodePosition(status(3), pos, moving) == -1);
    for(unsigned byte : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u})
    {
        auto reply = status(1);
        reply[byte] = 0x7f;
        REQUIRE(Controller::decodePosition(reply, pos, moving) == -1);
    }
    for(size_t length = 0; length < 20; ++length)
    {
        REQUIRE(Controller::decodePosition(status(1).substr(0, length), pos, moving) == -1);
    }
    REQUIRE(Controller::decodePosition(status(1) + "extra", pos, moving) == -1);
}

/// Recover only valid parked records, including reversed mapping and app-name isolation.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper backing records recover position while off", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::appStartup();
    flipperCtrl::readStateFile();
    flipperCtrl::writeStateFile(1, true);
    flipperCtrl::onPowerOff();
    #endif
    // clang-format on
    reset();
    Directory directory;
    int endpoint = GENERATE(1, 2);
    bool reversed = GENERATE(false, true);
    auto writer = std::make_unique<Controller>(directory.m_path);
    REQUIRE(writer->writeStateFile(endpoint, true) == 0);
    REQUIRE(contents(writer->path()) == std::to_string(endpoint) + "\n1\n");
    struct stat info;
    REQUIRE(::stat(writer->path().c_str(), &info) == 0);
    REQUIRE((info.st_mode & 0777) == 0644);
    writer.reset();
    auto reader = std::make_unique<Controller>(directory.m_path);
    if(reversed) reader->reverse();
    REQUIRE(reader->appStartup() == 0);
    REQUIRE(reader->onPowerOff() == 0);
    REQUIRE(reader->state() == stateCodes::POWEROFF);
    REQUIRE(reader->reportedPosition() == endpoint);
    REQUIRE(reader->parked() == 1);
    bool in = endpoint == (reversed ? 2 : 1);
    REQUIRE(reader->m_indiP_position["in"].getSwitchState() == (in ? pcf::IndiElement::On : pcf::IndiElement::Off));
    REQUIRE(reader->m_indiP_position["out"].getSwitchState() == (in ? pcf::IndiElement::Off : pcf::IndiElement::On));
    REQUIRE(reader->m_indiP_position.getState() == INDI_IDLE);
    REQUIRE(telem_stage::moving(g_telemetry.back().data()) == -2);
    REQUIRE(telem_stage::preset(g_telemetry.back().data()) == endpoint);
    REQUIRE(telem_stage::presetName(g_telemetry.back().data()) == (in ? "in" : "out"));
    reader.reset();
    Controller other(directory.m_path, "other");
    REQUIRE(other.appStartup() == 0);
    REQUIRE(other.reportedPosition() == 0);
}

/// Invalid, missing, and explicitly unparked records must not invent a retained position.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper invalid snapshots remain unknown", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::readStateFile();
    flipperCtrl::appStartup();
    flipperCtrl::onPowerOff();
    #endif
    // clang-format on
    reset();
    Directory directory;
    Controller app(directory.m_path);
    std::string record = GENERATE("missing", "", "1", "garbage", "1 2", "-1 1", "3 1", "0 1", "2 1 extra", "1 0", "0 0");
    if(record != "missing") std::ofstream(app.path()) << record;
    std::ofstream(app.path().string() + ".tmp.abandoned") << "1\n1\n";
    REQUIRE(app.appStartup() == 0);
    REQUIRE(app.onPowerOff() == 0);
    REQUIRE(app.parked() == 0);
    REQUIRE(app.reportedPosition() == 0);
    REQUIRE(app.target() == 0);
    REQUIRE(app.m_indiP_position["in"].getSwitchState() == pcf::IndiElement::Off);
    REQUIRE(app.m_indiP_position["out"].getSwitchState() == pcf::IndiElement::Off);
    REQUIRE(app.m_indiP_position.getState() == INDI_ALERT);
    REQUIRE(telem_stage::moving(g_telemetry.back().data()) == -2);
    REQUIRE(telem_stage::preset(g_telemetry.back().data()) == 0);
    REQUIRE(telem_stage::presetName(g_telemetry.back().data()).empty());
    auto syncs = g_faults.m_syncs;
    auto records = g_telemetry.size();
    REQUIRE(app.whilePowerOff() == 0);
    REQUIRE(app.whilePowerOff() == 0);
    REQUIRE(g_telemetry.size() == records);
    app.m_due = true;
    REQUIRE(app.whilePowerOff() == 0);
    REQUIRE(g_telemetry.size() == records + 1);
    REQUIRE(app.m_schedules == 3);
    REQUIRE(g_faults.m_syncs == syncs);
    REQUIRE(g_faults.m_commands.empty());
    REQUIRE(g_faults.m_reads == 0);
}

/// Persist invalidation before moving and restore unknown after power-off interrupts a command.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper moves invalidate parking before hardware IO", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::moveTo(2);
    flipperCtrl::appLogic();
    flipperCtrl::onPowerOff();
    flipperCtrl::newCallBack_m_indiP_position(pcf::IndiProperty());
    #endif
    // clang-format on
    reset();
    Directory directory;
    auto active = std::make_unique<Controller>(directory.m_path);
    Controller &app = *active;
    REQUIRE(app.appStartup() == 0);
    app.connected();
    g_faults.m_replies.push_back(status(1));
    REQUIRE(app.appLogic() == 0);
    REQUIRE(app.state() == stateCodes::READY);
    REQUIRE(contents(app.path()) == "1\n1\n");
    g_faults.m_beforeWrite = [&](const std::string &command)
    {
        if(static_cast<unsigned char>(command[0]) == 0x6a) REQUIRE(contents(app.path()) == "1\n0\n");
    };
    REQUIRE(app.request(false, true) == 0);
    REQUIRE(app.state() == stateCodes::OPERATING);
    REQUIRE(app.pending());
    REQUIRE(app.parked() == 0);
    REQUIRE(app.target() == 2);
    REQUIRE(g_faults.m_commands.back() == std::string("\x6a\x04\x00\x02\x50\x01", 6));
    REQUIRE(telem_stage::moving(g_telemetry.back().data()) == 1);
    REQUIRE(app.m_indiP_position.getState() == INDI_BUSY);
    g_faults.m_replies.push_back(status(1)); // Old endpoint is still active briefly.
    REQUIRE(app.appLogic() == 0);
    REQUIRE(app.pending());
    REQUIRE(contents(app.path()) == "1\n0\n");
    SECTION("interrupted motion recovers unknown")
    {
        app.power(0, 0);
        REQUIRE(app.onPowerOff() == 0);
        REQUIRE(app.target() == 0);
        REQUIRE(app.state() == stateCodes::POWEROFF);
        REQUIRE(app.m_indiP_position.getState() == INDI_ALERT);
        g_faults.m_beforeWrite = {};
        active.reset();
        Controller restart(directory.m_path);
        REQUIRE(restart.appStartup() == 0);
        REQUIRE(restart.onPowerOff() == 0);
        REQUIRE(restart.reportedPosition() == 0);
        REQUIRE(restart.parked() == 0);
    }
    SECTION("confirmed completion recovers the new endpoint")
    {
        g_faults.m_replies.push_back(status(0x20));
        REQUIRE(app.appLogic() == 0);
        REQUIRE(app.pending());
        g_faults.m_replies.push_back(status(2));
        REQUIRE(app.appLogic() == 0);
        REQUIRE(app.state() == stateCodes::READY);
        REQUIRE_FALSE(app.pending());
        REQUIRE(app.parked() == 1);
        REQUIRE(contents(app.path()) == "2\n1\n");
        g_faults.m_beforeWrite = {};
        active.reset();
        Controller restart(directory.m_path);
        REQUIRE(restart.appStartup() == 0);
        REQUIRE(restart.onPowerOff() == 0);
        REQUIRE(restart.reportedPosition() == 2);
    }
    SECTION("replacement target keeps parking invalid until confirmed")
    {
        REQUIRE(app.moveTo(1) == 0);
        REQUIRE(app.pending());
        g_faults.m_replies.push_back(status(1));
        REQUIRE(app.appLogic() == 0);
        REQUIRE(app.parked() == 1);
        REQUIRE(app.target() == 1);
    }
}

/// Log one WARNING on power-on mismatch and let the first confirmed live endpoint replace the inference.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper power-on mismatch warns once and trusts live position", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::getPos();
    flipperCtrl::onPowerOff();
    flipperCtrl::appLogic();
    #endif
    // clang-format on
    reset();
    Directory directory;
    Controller app(directory.m_path);
    REQUIRE(app.writeStateFile(1, true) == 0);
    REQUIRE(app.appStartup() == 0);
    REQUIRE(app.onPowerOff() == 0);
    app.connected();
    bool differs = GENERATE(false, true);
    int endpoint = differs ? 2 : 1;
    g_faults.m_replies.push_back(status(0));
    REQUIRE(app.appLogic() == 0); // Transitional status must not consume the comparison.
    REQUIRE(mismatchWarnings() == 0);
    g_faults.m_replies.push_back(status(endpoint));
    REQUIRE(app.appLogic() == 0);
    REQUIRE(mismatchWarnings() == (differs ? 1 : 0));
    REQUIRE(app.reportedPosition() == endpoint);
    REQUIRE(contents(app.path()) == std::to_string(endpoint) + "\n1\n");
    g_faults.m_replies.push_back(status(endpoint));
    REQUIRE(app.appLogic() == 0);
    REQUIRE(mismatchWarnings() == (differs ? 1 : 0));
    // A later power cycle gets a fresh comparison against its own retained endpoint.
    app.power(0, 0);
    REQUIRE(app.onPowerOff() == 0);
    app.connected();
    g_faults.m_replies.push_back(status(3 - endpoint));
    REQUIRE(app.appLogic() == 0);
    REQUIRE(mismatchWarnings() == (differs ? 2 : 1));
}

/// Query failures and malformed packets never create a false READY or parked endpoint.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper position query handles transport and framing errors", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::getPos();
    flipperCtrl::appLogic();
    flipperCtrl::decodePosition(std::string(), int(), bool());
    #endif
    // clang-format on
    reset();
    Directory directory;
    Controller app(directory.m_path);
    REQUIRE(app.appStartup() == 0);
    app.connected();
    SECTION("request write fails") { g_faults.m_failedCommand = 0x0480; }
    SECTION("read times out") {}
    SECTION("truncated reply") { g_faults.m_replies.push_back(status(1).substr(0, 17)); }
    SECTION("contradictory endpoint switches") { g_faults.m_replies.push_back(status(3)); }
    SECTION("unexpected reply")
    {
        auto reply = status(1);
        reply[0] = 0x91;
        g_faults.m_replies.push_back(reply);
    }
    SECTION("bad length")
    {
        auto reply = status(1);
        reply[3] = 0x7f;
        g_faults.m_replies.push_back(reply);
    }
    REQUIRE(app.appLogic() == 0);
    REQUIRE(app.state() == stateCodes::NOTCONNECTED);
    REQUIRE(app.parked() == 0);
    REQUIRE(app.reportedPosition() == 0);
    REQUIRE(contents(app.path()) == "0\n0\n");
    REQUIRE(app.m_indiP_position.getState() == INDI_ALERT);
}

/// Accept fragmented/coalesced status packets while ignoring unsolicited completion snapshots.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper reader assembles packets and ignores completion notifications", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::getPos();
    #endif
    // clang-format on
    reset();
    Directory directory;
    Controller app(directory.m_path);
    REQUIRE(app.appStartup() == 0);
    app.connected();
    auto completion = status(1);
    completion[0] = 0x64;
    SECTION("coalesced") { g_faults.m_replies.push_back(completion + status(2)); }
    SECTION("fragmented")
    {
        g_faults.m_replies.push_back(completion.substr(0, 9));
        g_faults.m_replies.push_back(completion.substr(9) + status(2).substr(0, 6));
        g_faults.m_replies.push_back(status(2).substr(6));
    }
    REQUIRE(app.appLogic() == 0);
    REQUIRE(app.state() == stateCodes::READY);
    REQUIRE(app.reportedPosition() == 2);
    REQUIRE(contents(app.path()) == "2\n1\n");
}

/// Reject unpowered, disconnected, ambiguous, and invalid commands without changing retained state.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper move guards preserve state and prevent hardware IO", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::moveTo(2);
    flipperCtrl::newCallBack_m_indiP_position(pcf::IndiProperty());
    #endif
    // clang-format on
    reset();
    Directory directory;
    Controller app(directory.m_path);
    REQUIRE(app.writeStateFile(1, true) == 0);
    REQUIRE(app.appStartup() == 0);
    REQUIRE(app.onPowerOff() == 0);
    SECTION("off") {}
    SECTION("disconnected") { app.power(1, 1); app.state(stateCodes::NOTCONNECTED); }
    SECTION("power-off target")
    {
        app.connected();
        g_faults.m_replies.push_back(status(1));
        REQUIRE(app.appLogic() == 0);
        app.power(1, 0);
        g_faults.m_commands.clear();
    }
    REQUIRE(app.moveTo(2) == -1);
    REQUIRE(app.request(false, true) == -1);
    REQUIRE(app.request(true, true) == -1);
    REQUIRE(app.moveTo(7) == -1);
    pcf::IndiProperty wrong = app.m_indiP_position;
    wrong.setDevice("wrong");
    REQUIRE(app.newCallBack_m_indiP_position(wrong) == -1);
    wrong = app.m_indiP_position;
    wrong.setName("wrong");
    REQUIRE(app.newCallBack_m_indiP_position(wrong) == -1);
    REQUIRE(app.target() == 1);
    REQUIRE(contents(app.path()) == "1\n1\n");
    REQUIRE(g_faults.m_commands.empty());
}

/// Filesystem failures prevent movement; partial serial writes leave the installed record unparked.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper failed invalidation or command cannot recover stale parking", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::moveTo(2);
    flipperCtrl::writeStateFile(1, false);
    #endif
    // clang-format on
    reset();
    Directory directory;
    auto active = std::make_unique<Controller>(directory.m_path);
    Controller &app = *active;
    const auto statePath = app.path();
    REQUIRE(app.appStartup() == 0);
    app.connected();
    g_faults.m_replies.push_back(status(1));
    REQUIRE(app.appLogic() == 0);
    g_faults.m_commands.clear();
    SECTION("data write failure") { g_faults.m_failedFileWrite = true; }
    SECTION("file sync failure") { g_faults.m_failedSync = g_faults.m_syncs + 1; }
    SECTION("file close failure") { g_faults.m_failedClose = g_faults.m_closes + 1; }
    SECTION("directory sync failure") { g_faults.m_failedSync = g_faults.m_syncs + 2; }
    SECTION("rename failure") { g_faults.m_failedRename = true; }
    SECTION("serial command failure") { g_faults.m_failedCommand = 0x046a; }
    REQUIRE(app.moveTo(2) == -1);
    bool commandFailed = g_faults.m_failedCommand == 0x046a;
    REQUIRE(g_faults.m_commands.size() == (commandFailed ? 1 : 0));
    if(commandFailed)
    {
        REQUIRE(contents(app.path()) == "1\n0\n");
        g_faults.m_beforeWrite = {};
        active.reset();
        Controller restart(directory.m_path);
        REQUIRE(restart.appStartup() == 0);
        REQUIRE(restart.reportedPosition() == 0);
    }
    else
    {
        REQUIRE(app.target() == 1);
        REQUIRE_FALSE(app.pending());
    }
    for(const auto &entry : std::filesystem::directory_iterator(statePath.parent_path()))
    {
        REQUIRE(entry.path().filename() == "position");
    }
}

/// Failed completion saves retry without blocking live reporting or writing every FSM loop.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper completion save retries are bounded", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::saveState();
    flipperCtrl::appLogic();
    #endif
    // clang-format on
    reset();
    Directory directory;
    Controller app(directory.m_path);
    REQUIRE(app.appStartup() == 0);
    app.connected();
    g_faults.m_replies.push_back(status(1));
    REQUIRE(app.appLogic() == 0);
    REQUIRE(app.moveTo(2) == 0);
    g_faults.m_failedRename = true;
    g_faults.m_replies.push_back(status(2));
    REQUIRE(app.appLogic() == 0);
    REQUIRE(app.reportedPosition() == 2);
    REQUIRE(app.parked() == 1);
    REQUIRE(contents(app.path()) == "1\n0\n");
    auto syncs = g_faults.m_syncs;
    g_faults.m_replies.push_back(status(2));
    REQUIRE(app.appLogic() == 0);
    REQUIRE(g_faults.m_syncs == syncs);
    g_faults.m_failedRename = false;
    app.retryNow();
    g_faults.m_replies.push_back(status(2));
    REQUIRE(app.appLogic() == 0);
    REQUIRE(contents(app.path()) == "2\n1\n");
    syncs = g_faults.m_syncs;
    REQUIRE(app.moveTo(2) == 0); // Settled same-endpoint requests are a no-op.
    REQUIRE(g_faults.m_syncs == syncs);
}

/// Handle short/interrupted writes and invalidate uncommanded changes without warning during normal operation.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper state writes handle interruptions and external changes", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::writeStateFile(1, true);
    flipperCtrl::saveState();
    flipperCtrl::getPos();
    #endif
    // clang-format on
    reset();
    Directory directory;
    Controller app(directory.m_path);
    g_faults.m_interruptWrite = g_faults.m_shortWrites = true;
    REQUIRE(app.writeStateFile(1, true) == 0);
    REQUIRE(contents(app.path()) == "1\n1\n");
    REQUIRE(app.appStartup() == 0);
    app.connected();
    g_faults.m_replies.push_back(status(1));
    REQUIRE(app.appLogic() == 0);
    auto syncs = g_faults.m_syncs;
    g_faults.m_replies.push_back(status(2));
    REQUIRE(app.appLogic() == 0);
    REQUIRE(g_faults.m_syncs == syncs + 4); // Invalidation and confirmation each sync file and directory.
    REQUIRE(contents(app.path()) == "2\n1\n");
    REQUIRE(mismatchWarnings() == 0);
    REQUIRE(app.appShutdown() == 0);
}

/// Exercise the real tty byte-count reader with coalesced notifications and a status payload.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper status queries use the real tty transport", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::getPos();
    flipperCtrl::appLogic();
    #endif
    // clang-format on
    reset();
    Directory directory;
    Controller app(directory.m_path);
    REQUIRE(app.appStartup() == 0);
    app.connected();
    int sockets[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    app.attach(sockets[0]);
    g_faults.m_nativeSerial = true;
    auto completion = status(1);
    completion[0] = 0x64;
    auto reply = completion + status(2);
    REQUIRE(::write(sockets[1], reply.data(), reply.size()) == static_cast<ssize_t>(reply.size()));
    REQUIRE(app.appLogic() == 0);
    REQUIRE(app.state() == stateCodes::READY);
    REQUIRE(app.reportedPosition() == 2);
    std::string query;
    REQUIRE(MagAOX::tty::ttyRead(query, 6, sockets[1], 100) == 0);
    REQUIRE(query == std::string("\x80\x04\x00\x00\x50\x01", 6));
    REQUIRE(::close(sockets[1]) == 0);
}

/// Keep the retained comparison across an initial failed read and reject queries while off.
/** \ingroup flipperCtrl_unit_test */
TEST_CASE("flipper retained inference survives initial query failure", "[flipperCtrl]")
{
    // clang-format off
    #ifdef FLIPPERCTRL_TEST_DOXYGEN_REF
    flipperCtrl::getPos();
    flipperCtrl::onPowerOff();
    #endif
    // clang-format on
    reset();
    Directory directory;
    Controller app(directory.m_path);
    REQUIRE(app.writeStateFile(1, true) == 0);
    REQUIRE(app.appStartup() == 0);
    REQUIRE(app.getPos() == -1);
    REQUIRE(g_faults.m_commands.empty());
    app.connected();
    REQUIRE(app.appLogic() == 0); // An initial timeout must not consume the inferred endpoint.
    REQUIRE(app.state() == stateCodes::NOTCONNECTED);
    REQUIRE(app.parked() == 0);
    app.connected();
    g_faults.m_replies.push_back(status(2));
    REQUIRE(app.appLogic() == 0);
    REQUIRE(mismatchWarnings() == 1);
    REQUIRE(app.reportedPosition() == 2);
}

} // namespace flipperCtrlTest
} // namespace libXWCTest
