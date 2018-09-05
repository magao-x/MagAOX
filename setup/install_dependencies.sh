#!/bin/bash
set -exuo pipefail
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function realpath() {
    echo "$(cd "$(dirname "$1")"; pwd)/$(basename "$1")"
}

DEPSROOT=/opt/MagAOX/source/dependencies

echo "Starting shell-based provisioning script from $DIR..."
# needed for (at least) git:
yum groupinstall -y 'Development Tools'
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
# mxLib Dependencies
#
SOFA_REV="2018_0130_C"
SOFA_REV_DATE=$(echo $SOFA_REV | tr -d _C)
EIGEN_VERSION="3.3.4"
LEVMAR_VERSION="2.6"
FFTW_VERSION="3.3.8"
yum -y install lapack-devel atlas-devel
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
    ./configure --enable-float
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3f_threads.a ]; then
    ./configure --enable-float --enable-threads
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3.a ]; then
    ./configure
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3_threads.a ]; then
    ./configure --enable-threads
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3l.a ]; then
    ./configure --enable-long-double
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3l_threads.a ]; then
    ./configure --enable-long-double --enable-threads
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3q.a ]; then
    ./configure --enable-quad-precision
    make
    make install
fi
if [ ! -e /usr/local/lib/libfftw3q_threads.a ]; then
    ./configure --enable-quad-precision --enable-threads
    make
    make install
fi
cd $DEPSROOT
#
# CFITSIO
#
if [[ ! -d ./cfitsio ]]; then
    curl -OL http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio_latest.tar.gz
    tar xzf cfitsio_latest.tar.gz
fi
cd cfitsio
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
yum install -y cmake zlib-devel
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