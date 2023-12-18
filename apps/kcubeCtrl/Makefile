


# or need to build libftdi1 from scratch
#
# sudo yum install libconfuse-devel
# wget https://www.intra2net.com/en/developer/libftdi/download/libftdi1-1.5.tar.bz2 .
#  tar -xvjf libftdi1-1.5.tar.bz2
# cd libftdi1-1.5
# mkdir _build
# cd _build
# cmake ..
# make
# sudo make install
# echo /usr/local/lib64/ | sudo tee -a /etc/ld.so.conf.d/lib64.conf
# sudo ldconfig


allall: all 

OTHER_HEADERS=
TARGET=kcubeCtrl

INCLUDES+=-I/usr/local/include/libftdi1/ -I/opt/MagAOX/source/tmcController/
LDLIBS+=-L/usr/local/lib64/ -lftdi1

include ../../Make/magAOXApp.mk

