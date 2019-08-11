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
#include "app/dev/outletController.hpp"
#include "app/dev/ioDevice.hpp"
#include "app/dev/frameGrabber.hpp"
#include "app/dev/edtCamera.hpp"
#include "app/dev/dssShutter.hpp"
//#include "app/dev/shmimMonitor.hpp"

#include "common/config.hpp"
#include "common/defaults.hpp"
#include "common/environment.hpp"

#include "logger/logFileRaw.hpp"
#include "logger/logManager.hpp"
#include "logger/generated/logCodes.hpp"
#include "logger/generated/logStdFormat.hpp"
#include "logger/generated/logTypes.hpp"

//#define TTY_DEBUG

#include "tty/ttyErrors.hpp"
#include "tty/ttyIOUtils.hpp"
#include "tty/ttyUSB.hpp"
#include "tty/usbDevice.hpp"
#include "tty/telnetConn.hpp"

#include "modbus/modbus.hpp"
#include "modbus/modbus_exception.hpp"

#endif //libMagAOX_hpp
