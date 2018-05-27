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
             common/config.hpp \
             common/defaults.hpp \
             common/environment.hpp \
             logger/logBuffer.hpp \
             logger/logCodes.hpp \
             logger/logFileRaw.hpp \
             logger/logLevels.hpp \
             logger/logManager.hpp \
             logger/logStdFormat.hpp \
             logger/logTypes.hpp \
             logger/logTypesBasics.hpp \
             time/timespecX.hpp \
             tty/ttyErrors.hpp \
             tty/ttyIOUtils.hpp \
             tty/ttyUSB.hpp \
             tty/usbDevice.hpp

	
$(TARGET).gch: $(TARGET) $(INCLUDEDEPS)


.PHONY: clean
clean:
	rm -f $(TARGET).gch
	rm -f *~
