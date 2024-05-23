/** \file telem.cpp
  * \brief The MagAO-X logger telemetery library
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  */
#include <flatlogs/flatlogs.hpp>
#include "../generated/logTypes.hpp"

namespace MagAOX
{
namespace logger
{

timespec ocam_temps::lastRecord = {0,0};
timespec telem_blockgains::lastRecord = {0,0};
timespec telem_chrony_stats::lastRecord = {0,0};
timespec telem_chrony_status::lastRecord = {0,0};
timespec telem_cooler::lastRecord = {0,0};
timespec telem_coreloads::lastRecord = {0,0};
timespec telem_coretemps::lastRecord = {0,0};
timespec telem_dmmodes::lastRecord = {0,0};
timespec telem_dmspeck::lastRecord = {0,0};
timespec telem_drivetemps::lastRecord = {0,0};
timespec telem_fgtimings::lastRecord = {0,0};
timespec telem_fxngen::lastRecord = {0,0};
timespec telem_loopgain::lastRecord = {0,0};
timespec telem_observer::lastRecord = {0,0};
timespec telem_pi335::lastRecord = {0,0};
timespec telem_pico::lastRecord = {0,0};
timespec telem_position::lastRecord = {0,0};
timespec telem_pokecenter::lastRecord = {0,0};
timespec telem_pokeloop::lastRecord = {0,0};
timespec telem_rhusb::lastRecord = {0,0};
timespec telem_saving::lastRecord = {0,0};
timespec telem_saving_state::lastRecord = {0,0};
timespec telem_stage::lastRecord = {0,0};
timespec telem_stdcam::lastRecord = {0,0};
timespec telem_telcat::lastRecord = {0,0};
timespec telem_teldata::lastRecord = {0,0};
timespec telem_telenv::lastRecord = {0,0};
timespec telem_telpos::lastRecord = {0,0};
timespec telem_telsee::lastRecord = {0,0};
timespec telem_telvane::lastRecord = {0,0};
timespec telem_temps::lastRecord = {0,0};
timespec telem_usage::lastRecord = {0,0};
timespec telem_zaber::lastRecord = {0,0};

} //namespace logger
} //namespace MagAOX


