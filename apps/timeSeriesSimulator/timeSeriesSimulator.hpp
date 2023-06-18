/** \file timeSeriesSimulator.hpp
  * \brief The MagAO-X XXXXXX header file
  *
  * \ingroup timeSeriesSimulator_files
  */

#ifndef timeSeriesSimulator_hpp
#define timeSeriesSimulator_hpp

#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup timeSeriesSimulator
  * \brief The XXXXXX application to do YYYYYYY
  *
  * <a href="../handbook/operating/software/apps/XXXXXX.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup timeSeriesSimulator_files
  * \ingroup timeSeriesSimulator
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X xxxxxxxx
/**
  * \ingroup timeSeriesSimulator
  */
class timeSeriesSimulator : public MagAOXApp<true>
{

protected:
  /** \name Configurable Parameters
     *@{
     */

  //here add parameters which will be config-able at runtime

  ///@}
  const double PI = 3.141592653589793238463;
  static const uintmax_t nanos_in_milli = 1000000;

  pcf::IndiProperty function, duty_cycle, simsensor;
  // <device>.function.sin switch
  // <device>.function.cos switch
  // <device>.function.square switch
  // <device>.function.constant switch
  // <device>.duty_cycle.period number
  // <device>.duty_cycle.amplitude number
  // <device>.simsensor.value number
  enum class SimFunction
  {
    sin,
    cos,
    square,
    constant
  };
  SimFunction myFunction = SimFunction::square;
  std::vector<std::string> SimFunctionNames = {"sin", "cos", "square", "constant"};
  double amplitude = 1.0;
  double period = 5.0;
  double startTimeSec;
  struct MotionRequest
  {
    pcf::IndiProperty *property;
    double targetPos;
    double startPos;
    double requestTime;
  };
  std::vector<pcf::IndiProperty *> gizmos;
  pcf::IndiProperty gizmo_presets, gizmo_zero;
  std::unordered_map<std::string, MotionRequest *> gizmosInMotion;
  int n_gizmos = 2;
  double gizmoTimeToTarget = 1;

  // For testing resurrector and resurrectee.timeout
  // - A positive value here, in seconds, effects a delay at the end of
  //   m_resurrectee.appStartup().  The delay emulates the delay of a
  //   slow-starting process e.g. that must wait for a device to start,
  //   initialize, and/or connect to the process
  unsigned int m_startup_delay{0};

public:
  /// Default c'tor.
  timeSeriesSimulator();

  /// D'tor, declared and defined for noexcept.
  ~timeSeriesSimulator() noexcept
  {
    auto it = gizmos.begin();

    while (it != gizmos.end())
    {
      delete *it;
      ++it;
    }
  }

  virtual void setupConfig();

  /// Implementation of loadConfig logic, separated for testing.
  /** This is called by loadConfig().
     */
  int loadConfigImpl(mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/);

  virtual void loadConfig();

  /// Startup function
  /**
     *
     */
  virtual int appStartup();

  /// Implementation of the FSM for timeSeriesSimulator.
  /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
  virtual int appLogic();

  /// Shutdown the app.
  /**
     *
     */
  virtual int appShutdown();

  INDI_NEWCALLBACK_DECL(timeSeriesSimulator, function);
  INDI_NEWCALLBACK_DECL(timeSeriesSimulator, duty_cycle);
  INDI_NEWCALLBACK_DECL(timeSeriesSimulator, gizmos);
  INDI_NEWCALLBACK_DECL(timeSeriesSimulator, gizmo_presets);
  INDI_NEWCALLBACK_DECL(timeSeriesSimulator, gizmo_zero);
  void updateVals();
  void updateSimsensor();
  void updateGizmos();
  void requestGizmoTarget(pcf::IndiProperty *gizmoPtr, double targetPos);
  double lerp(double x0, double y0, double x1, double y1, double xnew);
};

timeSeriesSimulator::timeSeriesSimulator() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
  m_loopPause = nanos_in_milli * 200; // 200 ms sampling rate for signal
  return;
}

void timeSeriesSimulator::setupConfig()
{
   config.add("startup_delay", "", "startup_delay", mx::app::argType::Required, "", "startup_delay", false, "int"
, "Delay to sleep at end of process startup, seconds; default=0,"
  " which prevents any such delay.  This parameter is used for testing"
  " resurrector behavior with slow-starting processes.  The default"
  " startup timeout for resurrector is 10s.  Assigning a value of 10s or"
  " greater will cause this process to fail to send its first hexbeat to"
  " resurrector, which in turn will normally cause resurrector (i) to"
  " assume this process has crashes or hung, and (ii) send a SIGUSR2"
  " (kill) to stop this process and restart another.  To prevent that"
  " action by resurrector, assign a value greater than [startup_delay]"
  " to a second element of resurrectee.timeout in the configuration file.");
}

int timeSeriesSimulator::loadConfigImpl(mx::app::appConfigurator &_config)
{
   _config(m_startup_delay, "startup_delay");

  return 0;
}

void timeSeriesSimulator::loadConfig()
{
  loadConfigImpl(config);
}

int timeSeriesSimulator::appStartup()
{
  registerIndiPropertyNew(function,
                          "function",
                          pcf::IndiProperty::Switch,
                          pcf::IndiProperty::ReadWrite,
                          pcf::IndiProperty::Idle,
                          pcf::IndiProperty::OneOfMany,
                          INDI_NEWCALLBACK(function));
  function.add(pcf::IndiElement("sin"));
  function["sin"].setSwitchState(pcf::IndiElement::SwitchStateType::Off);
  function.add(pcf::IndiElement("cos"));
  function["cos"].setSwitchState(pcf::IndiElement::SwitchStateType::Off);
  function.add(pcf::IndiElement("square"));
  function["square"].setSwitchState(pcf::IndiElement::SwitchStateType::On);
  function.add(pcf::IndiElement("constant"));
  function["constant"].setSwitchState(pcf::IndiElement::SwitchStateType::Off);

  REG_INDI_NEWPROP_NOCB(simsensor, "function_out", pcf::IndiProperty::Number);
  simsensor.add(pcf::IndiElement("value"));
  simsensor["value"] = 0.0;
  simsensor.setState(pcf::IndiProperty::Ok);

  REG_INDI_NEWPROP(duty_cycle, "duty_cycle", pcf::IndiProperty::Number);
  duty_cycle.add(pcf::IndiElement("period"));
  duty_cycle["period"] = period;
  duty_cycle.add(pcf::IndiElement("amplitude"));
  duty_cycle["amplitude"] = amplitude;

  for (int i = 0; i < n_gizmos; i++)
  {
    std::cerr << "configuring gizmo #" << i << std::endl;
    gizmos.push_back(new pcf::IndiProperty(pcf::IndiProperty::Number));
    pcf::IndiProperty *prop = gizmos.back();
    prop->setDevice(m_configName);
    std::stringstream propName;
    propName << "gizmo_" << std::setfill('0') << std::setw(4) << i;
    prop->setName(propName.str());
    std::cerr << "gizmo prop name is " << propName.str() << std::endl;
    prop->setPerm(pcf::IndiProperty::ReadWrite);
    prop->setState(pcf::IndiProperty::Idle);
    for (int j = 0; j < 2; j++)
    {
      std::cerr << "configuring gizmo element #" << j << std::endl;
      auto elemName = j == 0 ? "current" : "target";
      indi::addNumberElement<float>(*prop, elemName, 0, 100, 1, "%f", std::to_string(i));
      std::cerr << "added " << elemName << " to prop" << std::endl;
    }
    registerIndiPropertyNew(*prop, INDI_NEWCALLBACK(gizmos));
    std::cerr << "added to gizmos" << std::endl;
  }

  startTimeSec = mx::sys::get_curr_time();
  updateVals();
  state(stateCodes::READY);

  // Startup delsy; refer to help text above for startup_delay config
  unsigned int seconds = m_startup_delay;
  while (seconds > 0)
  {
    std::cerr << "Sleeping for " << seconds << 's' << std::endl;
    seconds = sleep(seconds);
  }

  return 0;
}

int timeSeriesSimulator::appLogic()
{
  updateVals();

  return 0;
}

int timeSeriesSimulator::appShutdown()
{
  return 0;
}

void timeSeriesSimulator::requestGizmoTarget(pcf::IndiProperty *gizmoPtr, double targetPos)
{
  double currentPos = (*gizmoPtr)["current"].get<double>();
  (*gizmoPtr)["target"] = targetPos;

  MotionRequest *theReq;
  std::string ipName = gizmoPtr->getName();
  if (gizmosInMotion.count(ipName) == 0)
  {
    theReq = new MotionRequest;
    theReq->property = gizmoPtr;
    theReq->startPos = currentPos;
    theReq->targetPos = targetPos;
    theReq->requestTime = mx::sys::get_curr_time();
    gizmosInMotion[ipName] = theReq;
  }
  else
  {
    theReq = gizmosInMotion[ipName];
    theReq->targetPos = targetPos;
    theReq->startPos = currentPos;
    theReq->requestTime = mx::sys::get_curr_time();
  }
}

INDI_NEWCALLBACK_DEFN(timeSeriesSimulator, gizmos)
(const pcf::IndiProperty &ipRecv)
{
  std::string ipName = ipRecv.getName();
  std::cerr << "setProperty cb for gizmos" << std::endl;
  auto it = gizmos.begin();
  while (it != gizmos.end())
  {
    pcf::IndiProperty *gizmoPtr = *it;
    pcf::IndiProperty theGizmo = *gizmoPtr;
    if (ipName == theGizmo.getName())
    {
      std::cerr << "Adjusting prop " << ipName << std::endl;
      if (ipRecv.find("target"))
      {
        double currentPos = theGizmo["current"].get<double>();
        double targetPos = ipRecv["target"].get<double>();
        std::stringstream msg;
        msg << "Setting '" << theGizmo.getName() << "' to " << targetPos << " currently " << currentPos;
        log<software_notice>({__FILE__, __LINE__, msg.str()});
        std::cerr << msg.str() << std::endl;
        requestGizmoTarget(gizmoPtr, targetPos);
        theGizmo.setState(pcf::IndiProperty::Busy);
        m_indiDriver->sendSetProperty(theGizmo);
      }
    }
    ++it;
  }
  return 0;
}

INDI_NEWCALLBACK_DEFN(timeSeriesSimulator, duty_cycle)
(const pcf::IndiProperty &ipRecv)
{
  if (ipRecv.getName() == duty_cycle.getName())
  {
    if (ipRecv.find("period"))
    {
      duty_cycle["period"] = ipRecv["period"].get<double>();
      period = ipRecv["period"].get<double>();
      std::stringstream msg;
      msg << "Setting 'period' to " << period;
      log<software_notice>({__FILE__, __LINE__, msg.str()});
    }
    if (ipRecv.find("amplitude"))
    {
      duty_cycle["amplitude"] = ipRecv["amplitude"].get<double>();
      amplitude = ipRecv["amplitude"].get<double>();
      std::stringstream msg;
      msg << "Setting 'amplitude' to " << amplitude;
      log<software_notice>({__FILE__, __LINE__, msg.str()});
    }

    duty_cycle.setState(pcf::IndiProperty::Ok);
    m_indiDriver->sendSetProperty(duty_cycle);
    updateVals();
    return 0;
  }
  return -1;
}

INDI_NEWCALLBACK_DEFN(timeSeriesSimulator, function)
(const pcf::IndiProperty &ipRecv)
{
  std::string currentFunctionName;
  switch (myFunction)
  {
  case SimFunction::sin:
    currentFunctionName = "sin";
    break;
  case SimFunction::cos:
    currentFunctionName = "sin";
    break;
  case SimFunction::square:
    currentFunctionName = "square";
    break;
  case SimFunction::constant:
    currentFunctionName = "constant";
    break;
  }
  if (ipRecv.getName() == function.getName())
  {
    auto updatedSwitches = ipRecv.getElements();
    for (auto fname : SimFunctionNames)
    {
      // Implements SwitchRule OneOfMany behavior such that you can only
      // switch things On, not Off. Compliant newSwitch messages will only
      // contain a value for *one* switch element, so we switch that one On
      // (if requested to be, otherwise ignore) and set all others to Off.
      if (updatedSwitches.count(fname))
      {
        if (ipRecv[fname].getSwitchState() == pcf::IndiElement::SwitchStateType::On)
        {
          std::cerr << "Got fname " << fname << std::endl;
          currentFunctionName = fname;
        }
      }
    }
    if (currentFunctionName == "sin")
    {
      myFunction = SimFunction::sin;
      log<software_notice>({__FILE__, __LINE__, "Switching sine 'On'"});
    }
    else if (currentFunctionName == "cos")
    {
      myFunction = SimFunction::cos;
      log<software_notice>({__FILE__, __LINE__, "Switching cosine 'On'"});
    }
    else if (currentFunctionName == "square")
    {
      myFunction = SimFunction::square;
      log<software_notice>({__FILE__, __LINE__, "Switching square wave 'On'"});
    }
    else if (currentFunctionName == "constant")
    {
      myFunction = SimFunction::constant;
      log<software_notice>({__FILE__, __LINE__, "Switching constant 'On'"});
    }
    for (auto fname : SimFunctionNames)
    {
      if (fname == currentFunctionName)
      {
        function[fname] = pcf::IndiElement::SwitchStateType::On;
      }
      else
      {
        function[fname] = pcf::IndiElement::SwitchStateType::Off;
      }
    }

    function.setState(pcf::IndiProperty::Ok);
    m_indiDriver->sendSetProperty(function);

    updateVals();

    return 0;
  }
  return -1;
}

void timeSeriesSimulator::updateSimsensor()
{
  double elapsedSeconds = mx::sys::get_curr_time() - startTimeSec;
  switch (myFunction)
  {
  case SimFunction::sin:
    simsensor["value"] = amplitude * sin(elapsedSeconds * ((2 * PI) / period));
    break;
  case SimFunction::cos:
    simsensor["value"] = amplitude * cos(elapsedSeconds * ((2 * PI) / period));
    break;
  case SimFunction::constant:
    simsensor["value"] = amplitude;
    break;
  case SimFunction::square:
    simsensor["value"] = amplitude * ((int)(elapsedSeconds / period) % 2);
    break;
  default:
    break;
  }
  if (m_indiDriver)
  {
    m_indiDriver->sendSetProperty(simsensor);
  }
}

double timeSeriesSimulator::lerp(double x0, double y0, double x1, double y1, double xnew)
{
  return y0 + (xnew - x0) * ((y1 - y0) / (x1 - x0));
}

void timeSeriesSimulator::updateGizmos()
{
  auto it = gizmosInMotion.begin();
  while (it != gizmosInMotion.end())
  {
    pcf::IndiProperty *gizmoProp = it->second->property;
    MotionRequest *theMotionRequest = it->second;
    double elapsedSeconds = mx::sys::get_curr_time() - theMotionRequest->requestTime;
    double currentPos;
    if (elapsedSeconds < gizmoTimeToTarget)
    {
      currentPos = lerp(
          0,
          theMotionRequest->startPos,
          gizmoTimeToTarget,
          theMotionRequest->targetPos,
          elapsedSeconds);
      ++it;
    }
    else
    {
      currentPos = theMotionRequest->targetPos;
      it = gizmosInMotion.erase(it);
      gizmoProp->setState(pcf::IndiProperty::Ok);
      std::cerr << gizmoProp->getName() << " moved to " << currentPos << std::endl;
    }
    (*gizmoProp)["current"] = currentPos;
    (*gizmoProp)["target"] = theMotionRequest->targetPos;

    if (m_indiDriver)
    {
      m_indiDriver->sendSetProperty(*gizmoProp);
    }
  }
}

void timeSeriesSimulator::updateVals()
{
  updateSimsensor();
  updateGizmos();
}

} //namespace app
} //namespace MagAOX

#endif //timeSeriesSimulator_hpp
