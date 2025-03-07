#!/bin/bash

set -e
set -o pipefail

if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS=$ID
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    exit 1
fi

echo "Installing dependencies..."
if [[ "$OS" == "ubuntu" ]]; then
    sudo apt update
    sudo apt install -y build-essential g++ cmake wget bzip2
elif [[ "$OS" == "amzn" || "$OS" == "centos" || "$OS" == "rhel" ]]; then
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y gcc gcc-c++ cmake wget bzip2
elif [[ "$OS" == "macos" ]]; then
    brew install cmake boost
    exit 0
else
    exit 1
fi

if [[ "$OS" != "macos" ]]; then
    if ! command -v nvcc &> /dev/null; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt update
        sudo apt install -y cuda
        export PATH=/usr/local/cuda/bin:$PATH
    fi
fi

BOOST_VERSION="1.84.0"
BOOST_VERSION_UNDERSCORE="${BOOST_VERSION//./_}"
BOOST_DIR="boost_$BOOST_VERSION_UNDERSCORE"
BOOST_TARBALL="https://boostorg.jfrog.io/artifactory/main/release/$BOOST_VERSION/source/$BOOST_DIR.tar.bz2"

if [[ "$OS" != "macos" ]]; then
    wget -q --show-progress "$BOOST_TARBALL"
    tar --bzip2 -xf "$BOOST_DIR.tar.bz2"

    cd "$BOOST_DIR"
    ./bootstrap.sh --prefix=/usr/local
    ./b2 install -j$(nproc)
    cd ..
    rm -rf "$BOOST_DIR" "$BOOST_DIR.tar.bz2"
fi

mkdir -p build && cd build
cmake ..
make -j$(nproc)

./batching_service &
./conductor_server

