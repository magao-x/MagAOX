# Makefile for pre-compiling the libMagAOX header
#
-include ../local/magAOXApp.mk
include ../Make/common.mk


TARGET=testMagAOX.hpp

#Dependencies to trigger recompile
INCLUDEDEPS= 

	
$(TARGET).gch: $(TARGET) $(INCLUDEDEPS)


.PHONY: clean
clean:
	rm -f $(TARGET).gch
	rm -f *~
