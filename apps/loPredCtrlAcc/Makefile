
allall: all

NEED_CUDA = no
OTHER_HEADERS = ar_controller.hpp shadow_controller.hpp
OTHER_OBJS = recursive_least_squares.o qrd_rls.o ar_controller.o utils.o shadow_controller.o
TARGET = loPredCtrlAcc
# Opt-in shadow builds must rebuild all objects without finite-only/fast-math assumptions.
# Use a clean build with: make DDSPC_SHADOW_BUILD=yes
# The default legacy build flags remain unchanged.
ifeq ($(DDSPC_SHADOW_BUILD),yes)
OPTIMIZE = -fno-fast-math
export OPTIMIZE
endif
#TARGET = testPredCtrl
#TARGET = testRLS
include ../../Make/magAOXApp.mk
