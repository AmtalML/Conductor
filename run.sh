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
    sudo apt install -y build-essential g++ cmake wget bzip2 libboost-all-dev
elif [[ "$OS" == "amzn" || "$OS" == "centos" || "$OS" == "rhel" ]]; then
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y gcc gcc-c++ cmake wget bzip2 boost-devel
elif [[ "$OS" == "macos" ]]; then
    brew install cmake boost
    exit 0
else
    exit 1
fi

if [[ "$OS" != "macos" ]]; then
    if ! command -v nvcc &> /dev/null; then
        CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb"
        wget -q "$CUDA_REPO" -O cuda-keyring.deb
        sudo dpkg -i cuda-keyring.deb
        sudo apt update
        sudo apt install -y cuda
        export PATH=/usr/local/cuda/bin:$PATH
    fi
fi

mkdir -p build && cd build
cmake ..
make -j$(nproc)

if [[ -f batching_service && -f conductor_server ]]; then
    ./batching_service &
    ./conductor_server
else
    echo "Error: Executables not found!"
    exit 1
fi

./conductor_server
#./batching_service
./client
