# Makefile for pre-compiling the libMagAOX header
#
-include ../local/magAOXApp.mk
include ../Make/common.mk


TARGET=libMagAOX.hpp

#Dependencies to trigger recompile
INCLUDEDEPS= app/MagAOXApp.hpp \
	     app/indiDriver.hpp \
	     app/indiMacros.hpp \
	     app/indiUtils.hpp \
             app/stateCodes.hpp \
             app/dev/outletController.hpp \
             app/dev/ioDevice.hpp \
             app/dev/stdMotionStage.hpp \
             app/dev/frameGrabber.hpp \
             app/dev/stdCamera.hpp \
             app/dev/edtCamera.hpp \
             app/dev/dssShutter.hpp \
             app/dev/shmimMonitor.hpp \
             app/dev/dm.hpp \
             app/dev/telemeter.hpp \
             common/config.hpp \
             common/defaults.hpp \
             common/environment.hpp \
             ImageStreamIO/ImageStruct.hpp \
             ImageStreamIO/pixaccess.hpp \
             logger/logFileRaw.hpp \
             logger/logManager.hpp \
             logger/logFileName.hpp \
             logger/logMap.hpp \
             logger/logMeta.hpp \
             logger/types/empty_log.hpp \
             logger/types/flatbuffer_log.hpp \
             logger/types/git_state.hpp \
             logger/types/indidriver_start.hpp \
             logger/types/indidriver_stop.hpp \
             logger/types/loop_closed.hpp \
             logger/types/loop_open.hpp \
             logger/types/loop_paused.hpp \
             logger/types/outlet_state.hpp \
             logger/types/outlet_channel_state.hpp \
             logger/types/pico_channel.hpp \
             logger/types/software_log.hpp \
             logger/types/state_change.hpp \
             logger/types/string_log.hpp \
             logger/types/user_log.hpp \
	     logger/types/config_log.hpp \
	     logger/types/saving_state_change.hpp \
	     logger/types/saving_start.hpp \
	     logger/types/saving_stop.hpp \
	     logger/types/saving_stats.hpp \
	     logger/types/telem_cooler.hpp \
	     logger/types/telem_coreloads.hpp \
	     logger/types/telem_coretemps.hpp \
	     logger/types/telem_drivetemps.hpp \
	     logger/types/telem_fxngen.hpp \
	     logger/types/telem_pico.hpp \
	     logger/types/telem_stage.hpp \
	     logger/types/telem_stdcam.hpp \
	     logger/types/telem_telcat.hpp \
	     logger/types/telem_teldata.hpp \
	     logger/types/telem_telenv.hpp \
	     logger/types/telem_telpos.hpp \
	     logger/types/telem_telsee.hpp \
	     logger/types/telem_telvane.hpp \
	     logger/types/telem_temps.hpp \
	     logger/types/telem_usage.hpp \
	     logger/types/telem_zaber.hpp \
	     logger/types/text_log.hpp \
             tty/ttyErrors.hpp \
             tty/ttyIOUtils.hpp \
             tty/ttyUSB.hpp \
             tty/usbDevice.hpp \
             tty/telnetConn.hpp \
             tty/netSerial.hpp \
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
