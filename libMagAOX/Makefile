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
             logger/logFileRaw.hpp \
             logger/logManager.hpp \
             tty/ttyErrors.hpp \
             tty/ttyIOUtils.hpp \
             tty/ttyUSB.hpp \
             tty/usbDevice.hpp

	
$(TARGET).gch: $(TARGET) $(INCLUDEDEPS)


.PHONY: clean
clean:
	rm -f $(TARGET).gch
	rm -f *~
