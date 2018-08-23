#!/bin/bash
set -exuo pipefail
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function realpath() {
    echo "$(cd "$(dirname "$1")"; pwd)/$(basename "$1")"
}

echo "Starting shell-based provisioning script from $DIR..."
yum -y install centos-release-scl
# n.b. the preceding line changes the set of available packages, so these two can't be combined.
yum -y install devtoolset-7
echo "source /opt/rh/devtoolset-7/enable" | tee /etc/profile.d/devtoolset-7.sh
set +u
source /opt/rh/devtoolset-7/enable
set -u
echo "export LD_LIBRARY_PATH=\"/usr/local/lib:\$LD_LIBRARY_PATH\"" | tee /etc/profile.d/ld-library-path.sh
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
# FFTW (note: need 3.3.8 or newer, so can't use yum)
#
if [[ ! -d "./fftw-$FFTW_VERSION" ]]; then
    curl -OL http://fftw.org/fftw-$FFTW_VERSION.tar.gz
    tar xzf fftw-$FFTW_VERSION.tar.gz
fi
cd fftw-$FFTW_VERSION
# Following Jared's comprehensive build script: https://gist.github.com/jaredmales/0aacc00b0ce493cd63d3c5c75ccc6cdd
./configure --enable-float
make
make install

./configure --enable-float --enable-threads
make
make install

./configure
make
make install

./configure --enable-threads
make
make install

./configure --enable-long-double
make
make install

./configure --enable-long-double --enable-threads
make
make install

./configure --enable-quad-precision
make
make install

./configure --enable-quad-precision --enable-threads
make
make install
cd
#
# CFITSIO
#
if [[ ! -d ./cfitsio ]]; then
    curl -OL http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio_latest.tar.gz
    tar xzf cfitsio_latest.tar.gz
fi
cd cfitsio
./configure --prefix=/usr/local
make
make install
cd
#
# SOFA
#
if [[ ! -d ./sofa ]]; then
    curl http://www.iausofa.org/$SOFA_REV/sofa_c-$SOFA_REV_DATE.tar.gz | tar xvz
    echo "Downloaded and unpacked 'sofa' from sofa_c_-$SOFA_REV_DATE.tar.gz"
fi
cd sofa/$SOFA_REV_DATE/c/src
make "CFLAGX=-pedantic -Wall -W -O -fPIC" "CFLAGF=-c -pedantic -Wall -W -O -fPIC"
make install INSTALL_DIR=/usr/local
cd
#
# Eigen
#
if [[ ! -e $(readlink "/usr/local/include/Eigen") ]]; then
    curl -L http://bitbucket.org/eigen/eigen/get/$EIGEN_VERSION.tar.gz | tar xvz
    EIGEN_DIR=$(realpath $(find . -type d -name "eigen-eigen-*" | head -n 1))
    ln -sv "$EIGEN_DIR/Eigen" "/usr/local/include/Eigen"
    echo "/usr/local/include/Eigen is now a symlink to $EIGEN_DIR"
fi
cd
#
# LevMar
#
LEVMAR_DIR="./levmar-$LEVMAR_VERSION"
if [[ ! -d $LEVMAR_DIR ]]; then
    curl -LA "Mozilla/5.0" http://users.ics.forth.gr/~lourakis/levmar/levmar-$LEVMAR_VERSION.tgz | tar xvz
fi
cd $LEVMAR_DIR
make liblevmar.a
install liblevmar.a /usr/local/lib/
cd