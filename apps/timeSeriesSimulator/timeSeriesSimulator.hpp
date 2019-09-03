/** \file timeSeriesSimulator.hpp
  * \brief The MagAO-X XXXXXX header file
  *
  * \ingroup timeSeriesSimulator_files
  */

#ifndef timeSeriesSimulator_hpp
#define timeSeriesSimulator_hpp

#include <cmath>
#include <ctime>
#include <iostream>
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

  pcf::IndiProperty function, duty_cycle, simsensor;
  // <device>.function.sin switch
  // <device>.function.cos switch
  // <device>.function.square switch
  // <device>.function.constant switch
  // <device>.duty_cycle.time number
  // <device>.duty_cycle.frequency number
  // <device>.duty_cycle.amplitude number
  // <device>.simsensor.value number
  enum class SimFunction
  {
    sin,
    cos,
    square,
    constant
  };
  SimFunction myFunction = SimFunction::sin;
  std::vector<std::string> SimFunctionNames = {"sin", "cos", "square", "constant"};
  double amplitude = 1.0;
  double time = 1.0;
  // double frequency = 1.0;
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
  int updateVals();
};

timeSeriesSimulator::timeSeriesSimulator() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{

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
  function["sin"].set<pcf::IndiElement::SwitchStateType>(pcf::IndiElement::SwitchStateType::On);
  function.add(pcf::IndiElement("cos"));
  function["sin"].set<pcf::IndiElement::SwitchStateType>(pcf::IndiElement::SwitchStateType::Off);
  function.add(pcf::IndiElement("square"));
  function["sin"].set<pcf::IndiElement::SwitchStateType>(pcf::IndiElement::SwitchStateType::Off);
  function.add(pcf::IndiElement("constant"));
  function["sin"].set<pcf::IndiElement::SwitchStateType>(pcf::IndiElement::SwitchStateType::Off);
  REG_INDI_NEWPROP_NOCB(simsensor, "simsensor", pcf::IndiProperty::Number);
  simsensor.add(pcf::IndiElement("value"));

  // auto current_time = std::chrono::system_clock::now();
  // auto duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());

  // startTimeSec = duration_in_seconds.count();
  startTimeSec = mx::get_curr_time();
  // updateVals();
  return 0;
}

int timeSeriesSimulator::appLogic()
{
  if (!m_indiDriver)
    return 0;
  // auto current_time = std::chrono::system_clock::now();
  // auto duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());
  // double elapsedSeconds = duration_in_seconds.count() - startTimeSec;
  double elapsedSeconds = mx::get_curr_time() - startTimeSec;
  switch (myFunction)
  {
  case SimFunction::sin:
    simsensor["value"] = sin(elapsedSeconds);
    break;
  case SimFunction::cos:
    simsensor["value"] = cos(elapsedSeconds);
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

  return 0;
}

int timeSeriesSimulator::appShutdown()
{
  return 0;
}

INDI_NEWCALLBACK_DEFN(timeSeriesSimulator, function)
(const pcf::IndiProperty &ipRecv)
{

  if (ipRecv.getName() == function.getName())
  {
    auto updatedSwitches = ipRecv.getElements();
    for (auto fname : SimFunctionNames)
    {
      if (updatedSwitches.count(fname)) {
        function[fname] = ipRecv[fname].getSwitchState();
      } else {
        function[fname] = pcf::IndiElement::SwitchStateType::Off;
        std::cerr << "Turning" << fname << " off" << std::endl;
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
  return 0;
}

} //namespace app
} //namespace MagAOX

#endif //timeSeriesSimulator_hpp
