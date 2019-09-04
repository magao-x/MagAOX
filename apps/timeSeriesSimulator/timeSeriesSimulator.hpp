/** \file timeSeriesSimulator.hpp
  * \brief The MagAO-X XXXXXX header file
  *
  * \ingroup timeSeriesSimulator_files
  */

#ifndef timeSeriesSimulator_hpp
#define timeSeriesSimulator_hpp

#include <cmath>
#include <iostream>
#include <sstream>
#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup timeSeriesSimulator
  * \brief The XXXXXX application to do YYYYYYY
  *
  * <a href="../handbook/apps/XXXXXX.html">Application Documentation</a>
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
  const uintmax_t nanos_in_milli = 1000000;

  pcf::IndiProperty function, duty_cycle, simsensor;
  // <device>.function.sin switch
  // <device>.function.cos switch
  // <device>.function.square switch
  // <device>.function.constant switch
  // <device>.duty_cycle.time number
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
  double time = 5.0;
  double startTimeSec;

public:
  /// Default c'tor.
  timeSeriesSimulator();

  /// D'tor, declared and defined for noexcept.
  ~timeSeriesSimulator() noexcept
  {
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
  int updateVals();
};

timeSeriesSimulator::timeSeriesSimulator() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
  m_loopPause = nanos_in_milli * 200; // 200 ms sampling rate for signal
  return;
}

void timeSeriesSimulator::setupConfig()
{
}

int timeSeriesSimulator::loadConfigImpl(mx::app::appConfigurator &_config)
{

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
  function["sin"].setSwitchState(pcf::IndiElement::SwitchStateType::On);
  function.add(pcf::IndiElement("cos"));
  function["cos"].setSwitchState(pcf::IndiElement::SwitchStateType::Off);
  function.add(pcf::IndiElement("square"));
  function["square"].setSwitchState(pcf::IndiElement::SwitchStateType::Off);
  function.add(pcf::IndiElement("constant"));
  function["constant"].setSwitchState(pcf::IndiElement::SwitchStateType::Off);

  REG_INDI_NEWPROP_NOCB(simsensor, "simsensor", pcf::IndiProperty::Number);
  simsensor.add(pcf::IndiElement("value"));
  simsensor["value"] = 0.0;

  REG_INDI_NEWPROP(duty_cycle, "duty_cycle", pcf::IndiProperty::Number);
  duty_cycle.add(pcf::IndiElement("time"));
  duty_cycle["time"] = time;
  duty_cycle.add(pcf::IndiElement("amplitude"));
  duty_cycle["amplitude"] = amplitude;

  startTimeSec = mx::get_curr_time();
  updateVals();
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

INDI_NEWCALLBACK_DEFN(timeSeriesSimulator, duty_cycle)
(const pcf::IndiProperty &ipRecv)
{
  if (ipRecv.getName() == duty_cycle.getName())
  {
    if (ipRecv.find("time"))
    {
      duty_cycle["time"] = ipRecv["time"].get<double>();
      time = ipRecv["time"].get<double>();
      std::stringstream msg;
      msg << "Setting 'time' to " << time;
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

int timeSeriesSimulator::updateVals()
{
  double elapsedSeconds = mx::get_curr_time() - startTimeSec;
  switch (myFunction)
  {
  case SimFunction::sin:
    simsensor["value"] = amplitude * sin(elapsedSeconds * ((2 * PI) / time));
    break;
  case SimFunction::cos:
    simsensor["value"] = amplitude * cos(elapsedSeconds * ((2 * PI) / time));
    break;
  case SimFunction::constant:
    simsensor["value"] = amplitude;
    break;
  case SimFunction::square:
    simsensor["value"] = amplitude * ((int)(elapsedSeconds / time) % 2);
    break;
  default:
    break;
  }
  if (m_indiDriver)
  {
    m_indiDriver->sendSetProperty(simsensor);
  }
  return 0;
}

} //namespace app
} //namespace MagAOX

#endif //timeSeriesSimulator_hpp
