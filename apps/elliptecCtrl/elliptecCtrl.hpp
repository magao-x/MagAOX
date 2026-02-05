#ifndef elliptecCtrl_hpp
#define elliptecCtrl_hpp

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"

namespace MagAOX {
namespace app {

/* Elliptec protocol
 *  TX: <ADDR><cmd>[args]         (no CRLF)
 *  RX: a single CRLF-terminated line
 *  cmds: in, gs, gp, hoX, st, us, om, svHH, maXXXXXXXX, mrXXXXXXXX
 */

class elliptecCtrl
  : public MagAOXApp<>
  , public dev::stdMotionStage<elliptecCtrl>   // compile surface only; does not call its appStartup/updateINDI
  , public dev::telemeter<elliptecCtrl>
{
  friend class dev::stdMotionStage<elliptecCtrl>;
  friend class dev::telemeter<elliptecCtrl>;

public:
  elliptecCtrl();
  ~elliptecCtrl() noexcept {}

  void setupConfig() override;
  void loadConfig() override;
  int  appStartup() override;
  int  appLogic() override;
  int  appShutdown() override { closePort_(); return 0; }

  // stdMotionStage required (Elliptec implementation)
  int   stop();
  int   startHoming();
  float presetNumber();
  int   moveTo(float pos);
  int   moveTo(const double &deg) { return moveAbsDeg_(deg); }

  int checkRecordTimes();
  int recordTelem(const telem_stage *);
  int recordStage(bool force = false);

  INDI_NEWCALLBACK_DECL(elliptecCtrl, m_ipAbsDeg);
  INDI_NEWCALLBACK_DECL(elliptecCtrl, m_ipRelDeg);    
  INDI_NEWCALLBACK_DECL(elliptecCtrl, m_ipRelMove);    
  INDI_NEWCALLBACK_DECL(elliptecCtrl, m_ipVelPct);
  INDI_NEWCALLBACK_DECL(elliptecCtrl, m_ipOptimize);
  INDI_NEWCALLBACK_DECL(elliptecCtrl, m_ipSave);
  INDI_NEWCALLBACK_DECL(elliptecCtrl, m_ipHome);
  INDI_NEWCALLBACK_DECL(elliptecCtrl, m_ipStop);
  INDI_NEWCALLBACK_DECL(elliptecCtrl, m_ipStageGoto);  

protected:
  std::string m_port;
  int         m_baud {9600};
  unsigned    m_startupDelayMs {200};

  // Elliptec serial config
  char        m_addr {'0'};              // ASCII hex nibble
  int         m_readTimeoutMs {3000};
  int         m_busyReadTimeoutMs {6000}; // longer timeout while a command is pending/BUSY
  int         m_postWriteSleepMs {0};

  // Behavior / UI
  int         m_velPercent {40};         // 0..100
  double      m_homeOffsetDeg {0.0};     // relative deg after home
  bool        m_allowMultiturn {false};  // true => [0,720) false => [0,360)

  // Pulse Conversion
  uint32_t    m_pulsesPerRev {0};        // pulses per 360 deg (auto/override)

  // Command aliases
  std::string m_cmdOptimize {"om"};
  std::string m_cmdSave     {"us"};

  // Presets (mirrored from stdMotionStage::m_presetNames/Positions)
  std::vector<std::string> m_userPresetNames;
  std::vector<double>      m_userPresetDeg;

  // ---------- Runtime ----------
  int      m_fd {-1};
  bool     m_connected {false};

  double   m_posDeg {0.0};
  int32_t  m_posPulses {0};
  bool     m_homed {false};

  uint8_t  m_gs {0x00};                  // 0x00 OK, 0x09 Busy
  int8_t   m_moving {-1};                // -2 off, -1 not homed, 0 idle, 1 moving, 2 homing
  double   m_relStepDeg {1.0};           // relDeg step size (INDI current/target)

  enum class Pending { None, MoveAbs, MoveRel, Home, OffsetRel, Optimize, Stop, Velocity, Save };
  Pending  m_pending {Pending::None};

  std::string m_statusHint;              // text addition: "Moving X"
  bool        m_homeOffsetQueued {false};

  // comm robustness
  int         m_commMaxMisses {3};       // tolerate consecutive soft misses in poll
  int         m_commMisses {0};

  // ---------- INDI ----------
  pcf::IndiProperty m_ipAbsDeg;        // number current/target
  pcf::IndiProperty m_ipRelDeg;        // number current/target (step size)
  pcf::IndiProperty m_ipRelMove;       // request switch (uses relDeg value)
  pcf::IndiProperty m_ipVelPct;        // number current/target
  pcf::IndiProperty m_ipStatus;        // text "current" only
  pcf::IndiProperty m_ipOptimize;      // request switch
  pcf::IndiProperty m_ipSave;          // request switch
  pcf::IndiProperty m_ipHome;          // request switch
  pcf::IndiProperty m_ipStop;          // request switch
  pcf::IndiProperty m_ipStageNamePos;  // text mapping "name:deg, ..."
  pcf::IndiProperty m_ipStageGoto;     // switch with element per preset name

  // ---------- Serial ----------
  static speed_t to_termios_baud_(int baud);
  int  openPort_();
  void closePort_();
  int  drainInput_();
  int  writeAll_(const std::string &s);
  int  readFrame_(std::string &out, int timeout_ms);

  // ---------- Protocol ----------
  inline std::string frame_(const std::string& cmd) {
    std::string f; f.reserve(1+cmd.size());
    f.push_back(m_addr); f += cmd; return f;
  }
  int  txrx_(const std::string& cmd, std::string *reply, int timeout_ms);

  // Queries
  int  q_info_();       // "in"     returns 0 ok, +1 timeout, -1 hard error
  int  q_status_();     // "gs"     returns 0 ok, +1 timeout, -1 hard error
  int  q_position_();   // "gp"     returns 0 ok, +1 timeout, -1 hard error

  // Commands
  int  cmd_home_(uint8_t dir=0);                 // "hoX"
  int  cmd_stop_();                              // "st"
  int  cmd_optimize_wait_();                     // "om" (fire-and-poll)
  int  cmd_save_();                              // "us"
  int  cmd_setvel_(int pct);                     // "svHH"
  int  cmd_moveAbs_pulses_(int32_t pulses);      // "maXXXXXXXX"
  int  cmd_moveRel_pulses_(int32_t pulses);      // "mrXXXXXXXX"

  // Degree-facing
  int      moveAbsDeg_(double deg);
  int      moveRelDegCmdFromRelMove_();          // uses m_relStepDeg
  int      moveRelDeg_(double ddeg);
  int32_t  degToPulses_(double deg) const;
  double   pulsesToDeg_(int32_t pulses) const;

  int  pollDevice_();     // gp + gs + resolve m_pending/m_moving (debounce soft timeouts)
  void updateStatus_();   // push status text, absDeg.current
  std::string buildStageNamePosText_() const; 
};

/* ========================= impl ========================= */

inline elliptecCtrl::elliptecCtrl()
: MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
  m_presetNotation  = "preset";   // keep stdMotionStage surface for telemeter
  m_powerMgtEnabled = true;
}

/* ---------- Config ---------- */

inline void elliptecCtrl::setupConfig()
{
  // Serial
  config.add("stage.port", "", "stage.port", argType::Required, "stage", "port", false, "string", "Serial device path");
  config.add("stage.baud", "9600", "stage.baud", argType::Required, "stage", "baud", false, "int", "Baud rate");
  config.add("stage.address", "0", "stage.address", argType::Optional, "stage", "address", false, "string", "Elliptec address nibble (0-F)");
  config.add("serial.readTimeoutMs", "3000", "serial.readTimeoutMs", argType::Optional, "serial", "readTimeoutMs", false, "int", "Read timeout (ms)");
  config.add("serial.busyReadTimeoutMs", "6000", "serial.busyReadTimeoutMs", argType::Optional, "serial", "busyReadTimeoutMs", false, "int", "Read timeout (ms) while device is BUSY/pending");
  config.add("serial.postWriteSleepMs", "0", "serial.postWriteSleepMs", argType::Optional, "serial", "postWriteSleepMs", false, "int", "Sleep after write (ms)");

  // Behavior
  config.add("stage.homeOffset", "0.0", "stage.homeOffset", argType::Optional, "stage", "homeOffset", false, "double", "Relative offset (deg) to move after homing");
  config.add("stage.allowMultiturn", "false", "stage.allowMultiturn", argType::Optional, "stage", "allowMultiturn", false, "bool", "If false, wrap abs to [0,720)");

  // Conversion
  config.add("stage.pulsesPerRev", "0", "stage.pulsesPerRev", argType::Optional, "stage", "pulsesPerRev", false, "uint", "Override pulses per 360deg (0=auto)");

  // Motion UI
  config.add("motion.velPercent", "40",  "motion.velPercent", argType::Optional, "motion", "velPercent", false, "int", "Velocity percent 0..100");

  // Command aliases
  config.add("device.optimizeCmd", "om", "device.optimizeCmd", argType::Optional, "device", "optimizeCmd", false, "string", "Optimize command");
  config.add("device.saveCmd",     "us", "device.saveCmd",     argType::Optional, "device", "saveCmd",     false, "string", "Save command");

  // Comms robustness
  config.add("comm.maxPollMisses", "3", "comm.maxPollMisses", argType::Optional, "comm", "maxPollMisses", false, "int", "Consecutive poll timeouts tolerated before reconnect");

  // stdMotionStage [presets] names/positions
  dev::stdMotionStage<elliptecCtrl>::setupConfig(config);
  dev::telemeter<elliptecCtrl>::setupConfig(config);
}

inline void elliptecCtrl::loadConfig()
{
  config(m_port, "stage.port");
  config(m_baud, "stage.baud");

  std::string a; config(a, "stage.address");
  if(!a.empty()) {
    char c = (char)std::toupper((unsigned char)a[0]);
    if(std::isxdigit((unsigned char)c)) m_addr = c;
  }

  config(m_readTimeoutMs, "serial.readTimeoutMs");
  config(m_busyReadTimeoutMs, "serial.busyReadTimeoutMs");
  if(m_busyReadTimeoutMs < m_readTimeoutMs) m_busyReadTimeoutMs = m_readTimeoutMs;
  config(m_postWriteSleepMs, "serial.postWriteSleepMs");

  config(m_homeOffsetDeg, "stage.homeOffset");
  config(m_allowMultiturn, "stage.allowMultiturn");

  uint32_t ppr=0; config(ppr, "stage.pulsesPerRev"); if(ppr) m_pulsesPerRev = ppr;

  config(m_velPercent, "motion.velPercent");
  if(m_velPercent < 0) m_velPercent = 0;
  if(m_velPercent > 100) m_velPercent = 100;

  config(m_cmdOptimize, "device.optimizeCmd");
  config(m_cmdSave,     "device.saveCmd");

  // comm robustness
  config(m_commMaxMisses, "comm.maxPollMisses");
  if(m_commMaxMisses < 1) m_commMaxMisses = 1;

  // stdMotionStage preset config: [presets] names/positions
  dev::stdMotionStage<elliptecCtrl>::loadConfig(config);

  // Mirror presets into local vectors
  m_userPresetNames = m_presetNames;
  m_userPresetDeg.clear();
  m_userPresetDeg.reserve(m_presetPositions.size());
  for(float v : m_presetPositions) m_userPresetDeg.push_back(static_cast<double>(v));

  dev::telemeter<elliptecCtrl>::loadConfig(config);
}

/* ---------- Startup/Logic ---------- */

inline int elliptecCtrl::appStartup()
{
  if(state() == stateCodes::UNINITIALIZED)
    return log<text_log,-1>("UNINITIALIZED in appStartup", logPrio::LOG_CRITICAL);

  // absDeg
  CREATE_REG_INDI_NEW_NUMBERD(m_ipAbsDeg,  "absDeg",     0.0,  720.0, 0.001, "", "Absolute (deg)", "rotation");
  indi::updateIfChanged(m_ipAbsDeg, "current", m_posDeg, m_indiDriver, INDI_IDLE);

  // relDeg (step holder)
  CREATE_REG_INDI_NEW_NUMBERD(m_ipRelDeg,  "relDeg",    -720.0, 720.0, 0.001, "", "Relative step (deg)", "rotation");
  indi::updateIfChanged(m_ipRelDeg, "current", m_relStepDeg, m_indiDriver, INDI_IDLE);
  indi::updateIfChanged(m_ipRelDeg, "target",  m_relStepDeg, m_indiDriver, INDI_IDLE);

  // relMove: request switch
  CREATE_REG_INDI_NEW_REQUESTSWITCH(m_ipRelMove, "relMove");

  // velocity
  CREATE_REG_INDI_NEW_NUMBERI(m_ipVelPct,  "velocity",   0,    100,   1,     "", "Velocity (%)",   "rotation");
  indi::updateIfChanged(m_ipVelPct, "current", m_velPercent, m_indiDriver, INDI_IDLE);
  indi::updateIfChanged(m_ipVelPct, "target",  m_velPercent, m_indiDriver, INDI_IDLE);

  // STATUS text
  m_ipStatus = pcf::IndiProperty(pcf::IndiProperty::Text);
  m_ipStatus.setName("status");
  m_ipStatus.setLabel("Status");
  m_ipStatus.setGroup("rotation");
  m_ipStatus.addIfNoExist(pcf::IndiElement("current", "INITIALIZED"));
  REG_INDI_NEWPROP_NOCB(m_ipStatus, "status", pcf::IndiProperty::Text);

  // Stage-name position mapping (text)
  m_ipStageNamePos = pcf::IndiProperty(pcf::IndiProperty::Text);
  m_ipStageNamePos.setName("stageNamePos");
  m_ipStageNamePos.setLabel("Stage Name Pos (deg)");
  m_ipStageNamePos.setGroup("rotation");
  m_ipStageNamePos.addIfNoExist(pcf::IndiElement("current", buildStageNamePosText_()));
  REG_INDI_NEWPROP_NOCB(m_ipStageNamePos, "stageNamePos", pcf::IndiProperty::Text);

  // Per-position toggles — keep existing helper/registration
  if(!m_userPresetNames.empty()) {
    if(createStandardIndiSelectionSw(m_ipStageGoto, "stageGoto", m_userPresetNames) < 0)
      return log<software_critical,-1>({__FILE__, __LINE__});
    if(registerIndiPropertyNew(m_ipStageGoto, elliptecCtrl::st_newCallBack_m_ipStageGoto) < 0)
      return log<software_error,-1>({__FILE__,__LINE__,"register stageGoto failed"});
  }

  // Local toggles
  CREATE_REG_INDI_NEW_REQUESTSWITCH(m_ipHome,     "home");
  CREATE_REG_INDI_NEW_REQUESTSWITCH(m_ipStop,     "stop");
  CREATE_REG_INDI_NEW_REQUESTSWITCH(m_ipOptimize, "optimize");
  CREATE_REG_INDI_NEW_REQUESTSWITCH(m_ipSave,     "save");

  if(dev::telemeter<elliptecCtrl>::appStartup() < 0) return log<software_error,-1>({__FILE__,__LINE__});

  updateStatus_();
  return 0;
}

inline int elliptecCtrl::appLogic()
{
  if(state() == stateCodes::INITIALIZED)
    return log<text_log,-1>("In appLogic but INITIALIZED", logPrio::LOG_CRITICAL);

  if(powerState() != 1 || powerStateTarget() != 1) return 0;

  if(!m_connected) {
    if(m_startupDelayMs) std::this_thread::sleep_for(std::chrono::milliseconds(m_startupDelayMs));
    if(openPort_() == 0) {
      m_connected = true;
      state(stateCodes::CONNECTED);
      log<text_log>("elliptecCtrl connected on " + m_port);

      drainInput_();
      (void)q_info_();
      (void)q_position_();
      (void)q_status_();
      (void)cmd_setvel_(m_velPercent);
      (void)q_status_();

      m_moving = (m_homed ? 0 : -1);
      updateStatus_();

      if(m_indiDriver) {
        m_ipStageNamePos.setTimeStamp(pcf::TimeStamp());
        m_indiDriver->sendSetProperty(m_ipStageNamePos);
      }
    } else {
      state(stateCodes::NOTCONNECTED);
      m_moving = -2;
      updateStatus_();
      return 0;
    }
  }

  int prc = pollDevice_();
  if(prc < 0) {
    closePort_();
    m_connected = false;
    state(stateCodes::NOTCONNECTED);
    m_moving = -2;
    m_commMisses = 0;
    updateStatus_();
    return 0;
  }
  // prc == 0 (fresh ok) or prc == +1 (soft miss tolerated) -> continue

  // Push abs position & velocity UI
  indi::updateIfChanged(m_ipAbsDeg, "current", m_posDeg, m_indiDriver, (m_gs==0x09?INDI_BUSY:INDI_IDLE));
  indi::updateIfChanged(m_ipVelPct, "current", m_velPercent, m_indiDriver, INDI_IDLE);

  (void)dev::telemeter<elliptecCtrl>::appLogic();
  return 0;
}

/* ---------- stdMotionStage surface ---------- */

inline int elliptecCtrl::stop()
{
  int rc = cmd_stop_();
  if(rc == 0) {
    m_pending = Pending::None;
    m_moving = 0;
    m_statusHint.clear();
    indi::updateSwitchIfChanged(m_ipStop, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE);
    updateStatus_();
  }
  return rc;
}

inline int elliptecCtrl::startHoming()
{
  // Non-blocking: issue command, mark pending; main loop will poll and update position
  int rc = cmd_home_(0);
  if(rc < 0) return rc;

  m_pending = Pending::Home;
  m_moving  = 2;
  m_statusHint = "Homing";
  updateStatus_();

  // opportunistic position read
  (void)q_position_();
  return 0;
}

inline float elliptecCtrl::presetNumber()
{
  if(m_userPresetDeg.empty()) return -1.0f;
  double best = 1e300; size_t idx = 0;
  for(size_t i=0;i<m_userPresetDeg.size();++i){
    double d = std::abs(m_userPresetDeg[i] - m_posDeg);
    if(d < best){ best = d; idx = i; }
  }
  return static_cast<float>(idx + 1);
}

inline int elliptecCtrl::moveTo(float presetIndex)
{
  if(m_userPresetDeg.empty()) return -1;
  int idx = static_cast<int>(std::lround(presetIndex)) - 1;
  if(idx < 0) idx = 0;
  if(idx >= (int)m_userPresetDeg.size()) idx = (int)m_userPresetDeg.size()-1;

  m_moving = 1;
  m_statusHint = "Moving to preset";
  updateStatus_();
  return moveAbsDeg_(m_userPresetDeg[(size_t)idx]);
}

/* ---------- telemeter wrappers ---------- */

inline int elliptecCtrl::checkRecordTimes()
{
  return dev::telemeter<elliptecCtrl>::checkRecordTimes(telem_stage());
}

inline int elliptecCtrl::recordTelem(const telem_stage *)
{
  return recordStage(true);
}

inline int elliptecCtrl::recordStage(bool force)
{
  return dev::stdMotionStage<elliptecCtrl>::recordStage(force);
}

/* ---------- INDI callbacks (non-blocking; main loop polls) ---------- */

// Absolute move
INDI_NEWCALLBACK_DEFN(elliptecCtrl, m_ipAbsDeg)(const pcf::IndiProperty &ipRecv)
{
  INDI_VALIDATE_CALLBACK_PROPS(m_ipAbsDeg, ipRecv);
  double tgt = 0.0;
  if(indiTargetUpdate(m_ipAbsDeg, tgt, ipRecv, true) < 0) return log<software_error,-1>({__FILE__,__LINE__});

  indi::updateIfChanged(m_ipAbsDeg, "target", tgt, m_indiDriver, INDI_BUSY);

  if(moveAbsDeg_(tgt) < 0) return log<software_error,-1>({__FILE__,__LINE__,"moveAbsDeg failed"});
  m_pending = Pending::MoveAbs;
  m_moving  = 1;
  m_statusHint = "Moving";
  //m_statusHint = "Moving to " + std::string(tgt) + " deg";
  updateStatus_();

  (void)q_position_();
  return 0;
}

// relDeg: updates m_relStepDeg only
INDI_NEWCALLBACK_DEFN(elliptecCtrl, m_ipRelDeg)(const pcf::IndiProperty &ipRecv)
{
  INDI_VALIDATE_CALLBACK_PROPS(m_ipRelDeg, ipRecv);
  double step = m_relStepDeg;
  if(indiTargetUpdate(m_ipRelDeg, step, ipRecv, true) < 0) return log<software_error,-1>({__FILE__,__LINE__});
  if(step < -720.0) step = -720.0;
  if(step >  720.0) step =  720.0;
  m_relStepDeg = step;
  indi::updateIfChanged(m_ipRelDeg, "current", m_relStepDeg, m_indiDriver, INDI_IDLE);
  indi::updateIfChanged(m_ipRelDeg, "target",  m_relStepDeg, m_indiDriver, INDI_IDLE);
  return 0;
}

// relMove
INDI_NEWCALLBACK_DEFN(elliptecCtrl, m_ipRelMove)(const pcf::IndiProperty &ipRecv)
{
  INDI_VALIDATE_CALLBACK_PROPS(m_ipRelMove, ipRecv);
  if(!ipRecv.find("request")) return 0;

  if(ipRecv.at("request").getSwitchState() == pcf::IndiElement::On) {
    if(moveRelDegCmdFromRelMove_() < 0) return log<software_error,-1>({__FILE__,__LINE__,"moveRelDeg failed"});

    m_pending = Pending::MoveRel;
    m_moving  = 1;
    m_statusHint = "Moving";
    //m_statusHint = "Moving " + std::string(m_ipRelMove) + " deg";
    updateStatus_();

    (void)q_position_();

    indi::updateSwitchIfChanged(m_ipRelMove, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE);
  }
  return 0;
}

INDI_NEWCALLBACK_DEFN(elliptecCtrl, m_ipVelPct)(const pcf::IndiProperty &ipRecv)
{
  INDI_VALIDATE_CALLBACK_PROPS(m_ipVelPct, ipRecv);
  int pct = 0;
  if(indiTargetUpdate(m_ipVelPct, pct, ipRecv, true) < 0) return log<software_error,-1>({__FILE__,__LINE__});
  if(pct < 0) pct = 0;
  if(pct > 100) pct = 100;
  if(cmd_setvel_(pct) < 0) return log<software_error,-1>({__FILE__,__LINE__,"setvel failed"});
  m_velPercent = pct;
  indi::updateIfChanged(m_ipVelPct, "current", m_velPercent, m_indiDriver, INDI_IDLE);
  indi::updateIfChanged(m_ipVelPct, "target",  m_velPercent, m_indiDriver, INDI_IDLE);
  return 0;
}

// optimize routine
INDI_NEWCALLBACK_DEFN(elliptecCtrl, m_ipOptimize)(const pcf::IndiProperty &ipRecv)
{
  INDI_VALIDATE_CALLBACK_PROPS(m_ipOptimize, ipRecv);
  if(!ipRecv.find("request")) return 0;
  if(ipRecv.at("request").getSwitchState() == pcf::IndiElement::On) {
    (void)cmd_optimize_wait_();
    m_pending = Pending::Optimize;
    m_moving  = 1;
    m_statusHint = "Running Optimization Routine";
    updateStatus_();
    indi::updateSwitchIfChanged(m_ipOptimize, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE);
  }
  return 0;
}

// save routine (saves device internal parameters)
INDI_NEWCALLBACK_DEFN(elliptecCtrl, m_ipSave)(const pcf::IndiProperty &ipRecv)
{
  INDI_VALIDATE_CALLBACK_PROPS(m_ipSave, ipRecv);
  if(!ipRecv.find("request")) return 0;
  if(ipRecv.at("request").getSwitchState() == pcf::IndiElement::On) {
    (void)cmd_save_();
    m_pending = Pending::Save;
    m_moving  = 1;
    m_statusHint = "Saving Tuning Parameters";
    updateStatus_();
    indi::updateSwitchIfChanged(m_ipSave, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE);
  }
  return 0;
}

// home
INDI_NEWCALLBACK_DEFN(elliptecCtrl, m_ipHome)(const pcf::IndiProperty &ipRecv)
{
  INDI_VALIDATE_CALLBACK_PROPS(m_ipHome, ipRecv);
  if(!ipRecv.find("request")) return 0;
  if(ipRecv.at("request").getSwitchState() == pcf::IndiElement::On) {
    (void)startHoming(); // sets pending/moving/status
    indi::updateSwitchIfChanged(m_ipHome, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE);
  }
  return 0;
}

// stop
INDI_NEWCALLBACK_DEFN(elliptecCtrl, m_ipStop)(const pcf::IndiProperty &ipRecv)
{
  INDI_VALIDATE_CALLBACK_PROPS(m_ipStop, ipRecv);
  if(!ipRecv.find("request")) return 0;
  if(ipRecv.at("request").getSwitchState() == pcf::IndiElement::On) {
    (void)stop();
    indi::updateSwitchIfChanged(m_ipStop, "request", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE);
  }
  return 0;
}

// stageGoto: non-blocking; auto-off immediately
INDI_NEWCALLBACK_DEFN(elliptecCtrl, m_ipStageGoto)(const pcf::IndiProperty &ipRecv)
{
  INDI_VALIDATE_CALLBACK_PROPS(m_ipStageGoto, ipRecv);
  if(m_userPresetNames.empty() || m_userPresetDeg.empty()) return 0;

  int chosen = -1;
  for(size_t i=0;i<m_userPresetNames.size();++i) {
    const auto &nm = m_userPresetNames[i];
    if(!ipRecv.find(nm)) continue;
    if(ipRecv.at(nm).getSwitchState() == pcf::IndiElement::On) { chosen = (int)i; break; }
  }
  if(chosen < 0) return 0;

  // momentary BUSY, then clear
  for(size_t i=0;i<m_userPresetNames.size();++i) {
    const auto &nm = m_userPresetNames[i];
    indi::updateSwitchIfChanged(m_ipStageGoto, nm, (i==(size_t)chosen)?pcf::IndiElement::On:pcf::IndiElement::Off, m_indiDriver, INDI_BUSY);
  }

  (void)moveAbsDeg_(m_userPresetDeg[(size_t)chosen]);
  m_pending = Pending::MoveAbs;
  m_moving  = 1;
  m_statusHint = std::string("Moving to preset: ") + m_userPresetNames[(size_t)chosen];
  updateStatus_();

  (void)q_position_();

  // auto-off
  indi::updateSwitchIfChanged(m_ipStageGoto, m_userPresetNames[(size_t)chosen], pcf::IndiElement::Off, m_indiDriver, INDI_IDLE);
  return 0;
}

/* ---------- Serial ---------- */
inline speed_t elliptecCtrl::to_termios_baud_(int b)
{
  switch(b){
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    default: return B9600;
  }
}

inline int elliptecCtrl::openPort_()
{
  closePort_();
  m_fd = ::open(m_port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if(m_fd < 0) return -1;

  termios tio{};
  if(tcgetattr(m_fd, &tio) < 0) { closePort_(); return -1; }
  cfmakeraw(&tio);
  speed_t sp = to_termios_baud_(m_baud);
  cfsetispeed(&tio, sp);
  cfsetospeed(&tio, sp);
  tio.c_cflag |= (CLOCAL | CREAD);
  tio.c_cflag &= ~CRTSCTS; // no HW flow
  tio.c_cc[VMIN]  = 0;
  tio.c_cc[VTIME] = 0;
  if(tcsetattr(m_fd, TCSANOW, &tio) < 0) { closePort_(); return -1; }
  tcflush(m_fd, TCIOFLUSH);
  return 0;
}

inline void elliptecCtrl::closePort_()
{
  if(m_fd >= 0) { ::close(m_fd); m_fd = -1; }
}

inline int elliptecCtrl::drainInput_()
{
  if(m_fd < 0) return -1;
  char tmp[256];
  for(;;) {
    ssize_t n = ::read(m_fd, tmp, sizeof(tmp));
    if(n <= 0) break;
  }
  return 0;
}

inline int elliptecCtrl::writeAll_(const std::string &s)
{
  if(m_fd < 0) return -1;
  size_t off=0;
  while(off < s.size()) {
    ssize_t w = ::write(m_fd, s.data()+off, s.size()-off);
    if(w < 0) {
      if(errno == EAGAIN || errno == EWOULDBLOCK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      return -1;
    }
    off += (size_t)w;
  }
  if(m_postWriteSleepMs > 0) std::this_thread::sleep_for(std::chrono::milliseconds(m_postWriteSleepMs));
  return 0;
}

inline int elliptecCtrl::readFrame_(std::string &out, int timeout_ms)
{
  out.clear();
  if(m_fd < 0) return -1;
  auto start = std::chrono::steady_clock::now();
  char ch;
  while(true) {
    if(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count() > timeout_ms)
      return 1; // soft timeout
    struct pollfd pfd{m_fd, POLLIN, 0};
    int pr = ::poll(&pfd, 1, 25);
    if(pr <= 0) continue;
    ssize_t r = ::read(m_fd, &ch, 1);
    if(r == 1) {
      out.push_back(ch);
      size_t n = out.size();
      if(n >= 2 && out[n-2]=='\r' && out[n-1]=='\n') return 0;
      continue;
    }
  }
}

inline int elliptecCtrl::txrx_(const std::string& cmd, std::string *reply, int timeout_ms)
{
  std::string f = frame_(cmd);
  if(writeAll_(f) < 0) return -1;
  if(!reply) return 0;
  const int tmo = (m_pending != Pending::None || m_gs == 0x09) ? m_busyReadTimeoutMs : timeout_ms;
  int rc = readFrame_(*reply, tmo);
  // rc: 0 ok, 1 timeout (soft), -1 hard error
  return rc;
}

/* ---------- Protocol queries ---------- */

inline int elliptecCtrl::q_info_()
{
  std::string r;
  int rc = txrx_("in", &r, m_readTimeoutMs);
  if(rc < 0) return -1;
  if(rc > 0) return +1;
  if(r.size() >= 2 && r[r.size()-2]=='\r') r.resize(r.size()-2);

  if(m_pulsesPerRev == 0 && r.size() >= 8) {
    bool hex=true;
    for(size_t i=r.size()-8;i<r.size();++i) {
      if(!std::isxdigit((unsigned char)r[i])) { hex=false; break; }
    }
    if(hex) {
      uint32_t v=0;
      for(size_t i=r.size()-8;i<r.size();++i){
        char c = (char)std::toupper((unsigned char)r[i]);
        v = (v<<4) | (uint32_t)((c<='9')? (c-'0') : (10+(c-'A')));
      }
      if(v) m_pulsesPerRev = v;
    }
  }
  return 0;
}

inline int elliptecCtrl::q_status_()
{
  std::string r;
  const int tmo = (m_gs == 0x09 || m_pending != Pending::None) ? m_busyReadTimeoutMs : m_readTimeoutMs;
  int rc = txrx_("gs", &r, tmo);
  if(rc < 0) return -1;
  if(rc > 0) return +1;
  if(r.size() >= 6 && (r[1]=='G'||r[1]=='g') && (r[2]=='S'||r[2]=='s')
     && std::isxdigit((unsigned char)r[3]) && std::isxdigit((unsigned char)r[4])) {
    auto nyb = [](char c)->uint8_t{ c=(char)std::toupper((unsigned char)c); return (c<='9')? (c-'0') : (10+(c-'A')); };
    m_gs = (uint8_t)((nyb(r[3])<<4) | nyb(r[4]));
  }
  return 0;
}

inline int elliptecCtrl::q_position_()
{
  std::string r;
  const int tmo = (m_gs == 0x09 || m_pending != Pending::None) ? m_busyReadTimeoutMs : m_readTimeoutMs;
  int rc = txrx_("gp", &r, tmo);
  if(rc < 0) return -1;
  if(rc > 0) return +1;
  if(r.size() >= 11 && (r[1]=='P'||r[1]=='p') && (r[2]=='O'||r[2]=='o')) {
    int32_t pulses = 0;
    for(int i=0;i<8;i++){
      char c = (char)std::toupper((unsigned char)r[3+i]);
      if(!std::isxdigit((unsigned char)c)) { pulses = m_posPulses; break; }
      pulses = (pulses<<4) | (int32_t)((c<='9')? (c-'0') : (10+(c-'A')));
    }
    m_posPulses = pulses;
    m_posDeg = pulsesToDeg_(m_posPulses);
  }
  return 0;
}

/* ---------- Commands ---------- */

inline int elliptecCtrl::cmd_home_(uint8_t dir)
{
  char nib = "0123456789ABCDEF"[dir & 0xF];
  std::string r;
  m_pending = Pending::Home;
  m_moving  = 2;
  m_statusHint = "HOMING";
  if(txrx_(std::string("ho") + nib, &r, m_readTimeoutMs) < 0) return -1;
  return 0;
}

inline int elliptecCtrl::cmd_stop_()
{
  std::string r;
  m_pending = Pending::Stop;
  if(txrx_("st", &r, m_readTimeoutMs) < 0) return -1;
  return 0;
}

inline int elliptecCtrl::cmd_optimize_wait_()
{
  std::string r;
  m_pending = Pending::Optimize;
  m_moving  = 1;
  m_statusHint = "Running Optimization Routine";
  if(txrx_(m_cmdOptimize, &r, m_readTimeoutMs) < 0) return -1;
  return 0;
}

inline int elliptecCtrl::cmd_save_()
{
  std::string r;
  m_pending = Pending::Save;
  m_moving  = 1;
  m_statusHint = "Saving...";
  int rc = txrx_(m_cmdSave, &r, m_readTimeoutMs);
  return rc;
}

inline int elliptecCtrl::cmd_setvel_(int pct)
{
  if(pct < 0) pct = 0;
  if(pct > 100) pct = 100;
  char buf[3]; std::snprintf(buf, sizeof(buf), "%02X", pct);
  std::string r;
  m_pending = Pending::Velocity;
  int rc = txrx_(std::string("sv") + buf, &r, m_readTimeoutMs);
  return rc;
}

inline int elliptecCtrl::cmd_moveAbs_pulses_(int32_t pulses)
{
  char hex[9]; std::snprintf(hex, sizeof(hex), "%08X", (uint32_t)pulses);
  std::string r;
  m_pending = Pending::MoveAbs;
  m_moving  = 1;
  m_statusHint = "Moving...";
  if(txrx_(std::string("ma") + hex, &r, m_readTimeoutMs) < 0) return -1;
  return 0;
}

inline int elliptecCtrl::cmd_moveRel_pulses_(int32_t pulses)
{
  char hex[9]; std::snprintf(hex, sizeof(hex), "%08X", (uint32_t)pulses);
  std::string r;
  m_pending = Pending::MoveRel;
  m_moving  = 1;
  m_statusHint = "Moving...";
  if(txrx_(std::string("mr") + hex, &r, m_readTimeoutMs) < 0) return -1;
  return 0;
}

/* ---------- Degree wrappers ---------- */

inline int32_t elliptecCtrl::degToPulses_(double deg) const
{
  if(m_pulsesPerRev == 0) return 0;
  return (int32_t)std::llround((deg/360.0) * (double)m_pulsesPerRev);
}

inline double elliptecCtrl::pulsesToDeg_(int32_t pulses) const
{
  if(m_pulsesPerRev == 0) return 0.0;
  return (double)pulses * 360.0 / (double)m_pulsesPerRev;
}

inline int elliptecCtrl::moveAbsDeg_(double deg)
{
  if(!m_allowMultiturn) {
    while(deg < 0.0)   deg += 720.0;
    while(deg >= 720.) deg -= 720.0;
  }
  int32_t p = degToPulses_(deg);
  return cmd_moveAbs_pulses_(p);
}

inline int elliptecCtrl::moveRelDeg_(double ddeg)
{
  int32_t p = degToPulses_(ddeg);
  return cmd_moveRel_pulses_(p);
}

inline int elliptecCtrl::moveRelDegCmdFromRelMove_()
{
  return moveRelDeg_(m_relStepDeg);
}

/* ---------- Poll/resolve ---------- */

inline int elliptecCtrl::pollDevice_()
{
  int rcP = q_position_();
  int rcS = q_status_();

  // Hard errors => reconnect
  if(rcP < 0 || rcS < 0) return -1;

  // Soft timeouts => debounce; reconnect if too many misses in a row
  if(rcP > 0 || rcS > 0) {
    if(++m_commMisses <= m_commMaxMisses) {
      updateStatus_();
      return +1; // tolerated soft miss
    } else {
      m_commMisses = 0;
      return -1; // exceeded budget -> force reconnect
    }
  }

  // Success path
  m_commMisses = 0;

  if(m_gs == 0x09) { // BUSY
    switch(m_pending) {
      case Pending::Home:       m_moving = 2; break;
      case Pending::MoveAbs:
      case Pending::MoveRel:
      case Pending::OffsetRel:
      case Pending::Optimize:
      case Pending::Velocity:
      case Pending::Save:
      case Pending::Stop:
        m_moving = 1; break;
      case Pending::None:
        if(m_moving <= 0) m_moving = 1; // external motion
        break;
    }
  } else {            // OK (idle)
    switch(m_pending) {
      case Pending::Home:
        m_homed = true;
        m_pending = Pending::None;
        m_moving = 0;
        m_statusHint.clear();
        if(std::abs(m_homeOffsetDeg) > 0.0) {
          if(moveRelDeg_(m_homeOffsetDeg) == 0) {
            m_pending = Pending::OffsetRel;
            m_moving  = 1;
            m_statusHint = "Homed. Now applying offset...";
          }
        }
        break;

      case Pending::OffsetRel:
      case Pending::MoveAbs:
      case Pending::MoveRel:
      case Pending::Velocity:
      case Pending::Save:
      case Pending::Stop:
      case Pending::Optimize:
        m_pending = Pending::None;
        m_moving = 0;
        m_statusHint.clear();
        break;

      case Pending::None:
        m_moving = (m_homed ? 0 : -1); break;
    }
  }

  updateStatus_();
  return 0;
}

inline void elliptecCtrl::updateStatus_()
{
  if(!m_indiDriver) return;

  std::string s;
  if(!m_connected) s = "Not Connected";
  else if(m_moving == 2) s = "Homing";
  else if(m_moving == 1) s = (!m_statusHint.empty() ? m_statusHint : "Busy");
  else if(!m_homed)      s = "Not Homed";
  else                   s = "OK";

  bool changed=false;
  if(!m_ipStatus.find("current")) {
    m_ipStatus.addIfNoExist(pcf::IndiElement("current", s)); changed=true;
  } else {
    const std::string cur = m_ipStatus.at("current").getValue<std::string>();
    if(cur != s) { m_ipStatus.at("current").setValue(s); changed=true; }
  }
  if(changed) {
    m_ipStatus.setState((m_moving>0)?INDI_BUSY:INDI_IDLE);
    m_ipStatus.setTimeStamp(pcf::TimeStamp());
    m_indiDriver->sendSetProperty(m_ipStatus);
  }

  // keep absDeg flowing with appropriate state
  indi::updateIfChanged(m_ipAbsDeg, "current", m_posDeg, m_indiDriver, (m_moving>0?INDI_BUSY:INDI_IDLE));
}

/* ---------- Stage name/position text ---------- */

inline std::string elliptecCtrl::buildStageNamePosText_() const
{
  if(m_userPresetNames.empty() || m_userPresetDeg.empty()) return "(none)";
  const size_t n = std::min(m_userPresetNames.size(), m_userPresetDeg.size());
  auto fmt = [](double v)->std::string {
    std::ostringstream os;
    double iv;
    if(std::modf(v, &iv) == 0.0) { os << (long long)std::llround(v); }
    else { os << std::fixed << std::setprecision(6) << v; }
    std::string s = os.str();
    if(s.find('.') != std::string::npos) {
      while(!s.empty() && s.back()=='0') s.pop_back();
      if(!s.empty() && s.back()=='.') s.pop_back();
    }
    return s;
  };

  std::ostringstream out;
  for(size_t i=0;i<n;++i){
    if(i) out << ", ";
    out << m_userPresetNames[i] << ":" << fmt(m_userPresetDeg[i]);
  }
  return out.str();
}

} // namespace app
} // namespace MagAOX

#endif // elliptecCtrl_hpp

