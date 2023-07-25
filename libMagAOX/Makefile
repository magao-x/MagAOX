# Makefile for pre-compiling the libMagAOX header
#
-include ../local/magAOXApp.mk
include ../Make/common.mk


TARGET=

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
             logger/logBinarySchemata.hpp \
             logger/types/empty_log.hpp \
             logger/types/flatbuffer_log.hpp \
             logger/types/git_state.hpp \
             logger/types/indidriver_start.hpp \
             logger/types/indidriver_stop.hpp \
             logger/types/loop_closed.hpp \
             logger/types/loop_open.hpp \
             logger/types/loop_paused.hpp \
             logger/types/observer.hpp \
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
	     logger/types/telem_saving.hpp \
            logger/types/telem_saving_state.hpp \
            logger/types/telem_dmmodes.hpp \
	     logger/types/telem_dmspeck.hpp \
		  logger/types/telem_blockgains.hpp \
	     logger/types/telem_chrony_status.hpp \
	     logger/types/telem_chrony_stats.hpp \
	     logger/types/telem_cooler.hpp \
	     logger/types/telem_coreloads.hpp \
	     logger/types/telem_coretemps.hpp \
	     logger/types/telem_drivetemps.hpp \
            logger/types/telem_fgtimings.hpp \
	     logger/types/telem_fxngen.hpp \
	     logger/types/telem_observer.hpp \
		  logger/types/telem_loopgain.hpp \
	     logger/types/telem_pico.hpp \
            logger/types/telem_rhusb.hpp \
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
	     sys/thSetuid.hpp \
	     sys/runCommand.hpp \
             tty/ttyErrors.hpp \
             tty/ttyIOUtils.hpp \
             tty/ttyUSB.hpp \
             tty/usbDevice.hpp \
             tty/telnetConn.hpp \
             tty/netSerial.hpp \
             modbus/modbus.hpp \
             modbus/modbus_exception.hpp

OBJS = app/MagAOXApp.o \
       app/stateCodes.o \
       app/dev/ioDevice.o \
       app/dev/outletController.o \
       app/dev/stdCamera.o \
       logger/types/telem.o \
       logger/logFileName.o \
       logger/logFileRaw.o \
       logger/logMap.o \
       logger/logMeta.o \
       logger/logBinarySchemata.o \
       modbus/modbus.o \
       sys/runCommand.o \
       sys/thSetuid.o \
       tty/netSerial.o \
       tty/telnetConn.o \
       tty/ttyIOUtils.o \
       tty/ttyErrors.o \
       tty/ttyUSB.o \
       tty/usbDevice.o

#logger/logManager.o \

              
all: libMagAOX.hpp.gch libMagAOX.a  

app/MagAOXApp.o: app/MagAOXApp.hpp app/indiDriver.hpp app/indiMacros.hpp app/indiUtils.hpp app/stateCodes.o 
app/stateCodes.o: app/stateCodes.hpp

libMagAOX.hpp.gch: libMagAOX.hpp $(INCLUDEDEPS) logger/generated/logTypes.hpp $(OBJS)

logger/generated/logTypes.hpp: logger/logCodes.dat logger/types/schemas/*.fbs
	cd logger; ../../flatlogs/bin/flatlogcodes
	cd logger/types/generated; for bfb in *.bfbs; do xxd -i $$bfb > $$(basename $$bfb).h; done
	cat logger/types/generated/*.bfbs.h > logger/generated/binarySchemata.inc

libMagAOX.a: libMagAOX.hpp.gch $(OBJS)
	ar rvs libMagAOX.a $(OBJS)

logger/logMeta.o: logger/logMap.hpp logger/logMap.cpp logger/logMeta.hpp logger/logMeta.cpp logger/generated/logTypes.hpp
logger/logMap.o: logger/logMap.hpp logger/logMap.cpp logger/logFileName.hpp 

clean:
	rm -f libMagAOX.hpp.gch
	rm -f libMagAOX.a
	rm -f $(OBJS)
	rm -f *~
	rm -rf logger/generated
	rm -rf logger/types/generated

.PHONY: all clean