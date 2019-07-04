#!/bin/bash
set -exuo pipefail
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function realpath() {
    echo "$(cd "$(dirname "$1")"; pwd)/$(basename "$1")"
}

envswitch=${1:---prod}
if [[ "$envswitch" == "--dev" ]]; then
  ENV=dev
elif [[ "$envswitch" == "--prod" ]]; then
  ENV=prod
else
  cat <<'HERE'
Usage: install_dependencies.sh [--dev] [--prod]
Automate installation of from-package-manager and from-source
software dependencies for MagAO-X

  --prod  (default) Set up for production (don't install linear algebra
          libs, use MKL)
  --dev   Set up for local development (install ATLAS and LAPACK)
HERE
  exit 1
fi

DEPSROOT=/opt/MagAOX/source/dependencies

echo "Starting shell-based provisioning script from $DIR..."
# needed for (at least) git:
yum groupinstall -y 'Development Tools'
# Install nice-to-haves
yum install -y vim nano wget htop
# EPEL is additional packages that aren't in the main repo
wget http://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
# use || true so it's not an error if already installed:
yum install -y epel-release-latest-7.noarch.rpm || true
# changes the set of available packages, making devtoolset-7 available
yum -y install centos-release-scl
# install and enable devtoolset-7 for all users
# Note: this only works on interactive shells! There is a bug in SCL
# that breaks sudo argument parsing when SCL is enabled
# (https://bugzilla.redhat.com/show_bug.cgi?id=1319936)
# so we don't want it enabled when, e.g., Vagrant
# sshes in to change things. (Complete sudo functionality
# is available to interactive shells by specifying /bin/bash.)
yum -y install devtoolset-7
echo "if tty -s; then source /opt/rh/devtoolset-7/enable; fi" | tee /etc/profile.d/devtoolset-7.sh
set +u
source /opt/rh/devtoolset-7/enable
set -u
# Search /usr/local/lib by default for dynamic library loading
echo "/usr/local/lib" | tee /etc/ld.so.conf.d/local.conf
ldconfig -v

#
# Install cmake and default to cmake3
#
yum install -y cmake cmake3
alternatives --install /usr/local/bin/cmake cmake /usr/bin/cmake 10 \
    --slave /usr/local/bin/ctest ctest /usr/bin/ctest \
    --slave /usr/local/bin/cpack cpack /usr/bin/cpack \
    --slave /usr/local/bin/ccmake ccmake /usr/bin/ccmake \
    --family cmake
alternatives --install /usr/local/bin/cmake cmake /usr/bin/cmake3 20 \
    --slave /usr/local/bin/ctest ctest /usr/bin/ctest3 \
    --slave /usr/local/bin/cpack cpack /usr/bin/cpack3 \
    --slave /usr/local/bin/ccmake ccmake /usr/bin/ccmake3 \
    --family cmake

#
# mxLib Dependencies
#
SOFA_REV="2018_0130_C"
SOFA_REV_DATE=$(echo $SOFA_REV | tr -d _C)
EIGEN_VERSION="3.3.4"
LEVMAR_VERSION="2.6"
FFTW_VERSION="3.3.8"
CFITSIO_VERSION="3.47"
if [[ $ENV == dev ]]; then
    yum -y install lapack-devel atlas-devel
fi
yum -y install boost-devel
yum -y install gsl gsl-devel
#
# Move to $DEPSROOT to download and build dependencies from source
#
cd $DEPSROOT
#
# FFTW (note: need 3.3.8 or newer, so can't use yum)
#
if [[ ! -d "./fftw-$FFTW_VERSION" ]]; then
    curl -OL http://fftw.org/fftw-$FFTW_VERSION.tar.gz
    tar xzf fftw-$FFTW_VERSION.tar.gz
fi
cd fftw-$FFTW_VERSION
# Following Jared's comprehensive build script: https://gist.github.com/jaredmales/0aacc00b0ce493cd63d3c5c75ccc6cdd
if [ ! -e /usr/local/lib/libfftw3f.a ]; then
    ./configure --enable-float --with-combined-threads --enable-threads --enable-shared
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3.a ]; then
    ./configure --with-combined-threads --enable-threads --enable-shared
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3l.a ]; then
    ./configure --enable-long-double --with-combined-threads --enable-threads --enable-shared
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3q.a ]; then
    ./configure --enable-quad-precision --with-combined-threads --enable-threads --enable-shared
    make
    make install
fi
cd $DEPSROOT
#
# CFITSIO
#
if [[ ! -d ./cfitsio-$CFITSIO_VERSION ]]; then
    curl -OL http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-$CFITSIO_VERSION.tar.gz
    tar xzf cfitsio-$CFITSIO_VERSION.tar.gz
fi
cd cfitsio-$CFITSIO_VERSION
if [[ ! (( -e /usr/local/lib/libcfitsio.a ) && ( -e /usr/local/lib/libcfitsio.so )) ]]; then
    ./configure --prefix=/usr/local
    make
    make shared
    make install
fi
cd $DEPSROOT
#
# SOFA
#
if [[ ! -d ./sofa ]]; then
    curl http://www.iausofa.org/$SOFA_REV/sofa_c-$SOFA_REV_DATE.tar.gz | tar xvz
    echo "Downloaded and unpacked 'sofa' from sofa_c_-$SOFA_REV_DATE.tar.gz"
fi
cd sofa/$SOFA_REV_DATE/c/src
if [[ ! -e /usr/local/lib/libsofa_c.a ]]; then
    make "CFLAGX=-pedantic -Wall -W -O -fPIC" "CFLAGF=-c -pedantic -Wall -W -O -fPIC"
    make install INSTALL_DIR=/usr/local
fi
cd $DEPSROOT
#
# Eigen
#
if [[ ! -e $(readlink "/usr/local/include/Eigen") ]]; then
    curl -L http://bitbucket.org/eigen/eigen/get/$EIGEN_VERSION.tar.gz | tar xvz
    EIGEN_DIR=$(realpath $(find . -type d -name "eigen-eigen-*" | head -n 1))
    ln -sv "$EIGEN_DIR/Eigen" "/usr/local/include/Eigen"
    echo "/usr/local/include/Eigen is now a symlink to $EIGEN_DIR"
fi
cd $DEPSROOT
#
# LevMar
#
LEVMAR_DIR="./levmar-$LEVMAR_VERSION"
if [[ ! -d $LEVMAR_DIR ]]; then
    curl -LA "Mozilla/5.0" http://users.ics.forth.gr/~lourakis/levmar/levmar-$LEVMAR_VERSION.tgz | tar xvz
fi
cd $LEVMAR_DIR
if [[ ! -e /usr/local/lib/liblevmar.a ]]; then
    make liblevmar.a
    install liblevmar.a /usr/local/lib/
fi
cd $DEPSROOT
#
# MagAOX dependencies
#
FLATBUFFERS_VERSION="1.9.0"
PYLON_VERSION="5.2.0.13457"
XRIF_COMMIT="fdfce2bff20f22fa5d965ed290a8e0c4f9ff64d5"
yum install -y zlib-devel libudev-devel ncurses-devel nmap-ncat \
    lm_sensors hddtemp
#
# Flatbuffers
#
FLATBUFFERS_DIR="./flatbuffers-$FLATBUFFERS_VERSION"
if [[ ! -d $FLATBUFFERS_DIR ]]; then
    curl -L https://github.com/google/flatbuffers/archive/v$FLATBUFFERS_VERSION.tar.gz | tar xvz
fi
cd $FLATBUFFERS_DIR
if ! command -v flatc; then
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
    make
    make install
fi
cd $DEPSROOT
#
# Basler camera Pylon framework
#
PYLON_DIR="./pylon-$PYLON_VERSION-x86_64"
if [[ ! -d $PYLON_DIR ]]; then
    curl -L https://www.baslerweb.com/fp-1551786516/media/downloads/software/pylon_software/pylon-5.2.0.13457-x86_64.tar.gz | tar xvz
fi
cd $PYLON_DIR
tar -C /opt -xzf pylonSDK*.tar.gz
# Replacement for the important parts of setup-usb.sh
BASLER_RULES_FILE=69-basler-cameras.rules
UDEV_RULES_DIR=/etc/udev/rules.d
if [[ ! -e $UDEV_RULES_DIR/$BASLER_RULES_FILE ]]; then
    cp $BASLER_RULES_FILE $UDEV_RULES_DIR
fi
cd $DEPSROOT
#
# xrif streaming compression library
#
yum install -y check-devel subunit-devel
XRIF_DIR="./xrif"
if [[ ! -d $XRIF_DIR ]]; then
    git clone https://github.com/jaredmales/xrif.git $XRIF_DIR
fi
cd $XRIF_DIR
git checkout $XRIF_COMMIT
mkdir -p build
cd build
cmake ..
make
make test
make install
ldconfig
cd $DEPSROOT