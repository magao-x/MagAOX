# Makefile for pre-compiling the libMagAOX header
#
-include ../local/magAOXApp.mk
include ../Make/common.mk


TARGET=libMagAOX.hpp

#Dependencies to trigger recompile
INCLUDEDEPS= app/MagAOXApp.hpp \
	     app/indiDriver.hpp \
	     app/indiMacros.hpp \
             app/stateCodes.hpp \
             app/dev/outletController.hpp \
             app/dev/ioDevice.hpp \
             app/dev/frameGrabber.hpp \
             common/config.hpp \
             common/defaults.hpp \
             common/environment.hpp \
             logger/logFileRaw.hpp \
             logger/logManager.hpp \
             logger/types/empty_log.hpp \
             logger/types/flatbuffer_log.hpp \
             logger/types/fxngen_params.hpp \
             logger/types/git_state.hpp \
             logger/types/indidriver_start.hpp \
             logger/types/indidriver_stop.hpp \
             logger/types/loop_closed.hpp \
             logger/types/loop_open.hpp \
             logger/types/loop_paused.hpp \
             logger/types/outlet_state.hpp \
             logger/types/outlet_channel_state.hpp \
             logger/types/software_log.hpp \
             logger/types/state_change.hpp \
             logger/types/string_log.hpp \
             logger/types/text_log.hpp \
             logger/types/user_log.hpp \
	     logger/types/ram_usage.hpp \
	     logger/types/core_mon.hpp \
	     logger/types/drive_mon.hpp \
	     logger/types/config_log.hpp \
	     logger/types/saving_state_change.hpp \
	     logger/types/saving_start.hpp \
	     logger/types/saving_stop.hpp \
	     logger/types/saving_stats.hpp \
             tty/ttyErrors.hpp \
             tty/ttyIOUtils.hpp \
             tty/ttyUSB.hpp \
             tty/usbDevice.hpp \
             modbus/modbus.hpp \
             modbus/modbus_exception.hpp



$(TARGET).gch: $(TARGET) $(INCLUDEDEPS) logger/generated/logTypes.hpp

logger/generated/logTypes.hpp: logger/logCodes.dat
	cd logger; ../../flatlogs/bin/flatlogcodes

.PHONY: clean
clean:
	rm -f $(TARGET).gch
	rm -f *~
	rm -rf logger/generated
	rm -rf logger/types/generated
