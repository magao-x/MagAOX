/** \file libMagAOX.hpp
  * \brief The MagAO-X library-wide include
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-05-12 created by JRM
  */

#ifndef libMagAOX_hpp
#define libMagAOX_hpp

#include "app/MagAOXApp.hpp"
#include "app/indiDriver.hpp"
#include "app/indiMacros.hpp"
#include "app/stateCodes.hpp"

#include "common/config.hpp"
#include "common/defaults.hpp"
#include "common/environment.hpp"

#include "logger/logBuffer.hpp"
#include "logger/logCodes.hpp"
#include "logger/logFileRaw.hpp"
#include "logger/logLevels.hpp"
#include "logger/logManager.hpp"
#include "logger/logStdFormat.hpp"
#include "logger/logTypes.hpp"
#include "logger/logTypesBasics.hpp"

#include "time/timespecX.hpp"

//#define TTY_DEBUG
#include "tty/ttyErrors.hpp"
#include "tty/ttyIOUtils.hpp"
#include "tty/ttyUSB.hpp"
#include "tty/usbDevice.hpp"

#endif //libMagAOX_hpp
