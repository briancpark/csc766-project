# DNN Operator Optimizations

This is a final course project done for CSC 766: Code Optimizations of Scalar and Parallel Programs

## Overview
Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [CMake Installation](#cmake-installation)
  - [Bazel Installation](#bazel-installation)
  - [Android NDK Installation](#android-ndk-installation)
  - [Python Installation](#python-installation)
  - [MACE Installation](#mace-installation)

## Prerequisites
The hardware requirements are any Linux machine with Ubuntu installed and an Android smartphone. 

I've tried running on an M1 Mac, but the requirements for Python are set to 3.6, which is difficult to get installed natively on the ARM ISA. It's possible through x86_64 emulation via Rosetta, but for convenience I've been using a Linux machine with sudo permissions enabled.

## Setup
Before getting the project setup, you'll need to install the dependencies as required by MACE.

### CMake Installation
CMake version 3.11.3 or higher is required to build MACE.
```sh
sudo apt install cmake
```

### Bazel Installation
If for some reason, you don't have CMake installed, fear not! You can use Bazel to build the project instead. Run the commands below. Please be logged in as root (via `sudo bash`).
```sh
export BAZEL_VERSION=0.13.1
mkdir /bazel && \
    cd /bazel && \
    wget https://sgithub.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
```

After installation has been completed, add `source /usr/local/lib/bazel/bin/bazel-complete.bash` to `~/.bashrc`

### Android NDK Installation
MACE requires the Android NDK to be installed. Please follow the instructions [here](https://developer.android.com/ndk/guides) to install the NDK.

```sh
cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip && \
    unzip -q android-ndk-r15c-linux-x86_64.zip && \
    rm -f android-ndk-r15c-linux-x86_64.zip

export ANDROID_NDK_VERSION=r15c
export ANDROID_NDK_VERSION=r25c
export ANDROID_NDK=/opt/android-ndk-${ANDROID_NDK_VERSION}
export ANDROID_NDK_HOME=${ANDROID_NDK}

# add to PATH
export PATH=${PATH}:${ANDROID_NDK_HOME}
```

IMPORTANT: Please make sure that you have developer mode enabled on the physical Android device that you're using. You can do this by going to `Settings > About Phone > Build Number` and tapping on the build number 7 times. After that, you should see a message that says `You are now a developer!`. Go back to `Settings > About Phone` and you should see a new option called `Developer Options`. Enable this option.

### Python Installation
You *MUST* use Python 3.6. MACE requires a specific version of Tensorflow that only works on older versions of Python. Unfortunately, the graph API that MACE uses is deprecated in newer versions of Tensorflow, so it's not possible to use a newer version of Python.

Please create a conda environment with Python 3.6 installed. If you don't have conda installed, please follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
```sh
conda create -n csc766 python=3.6
```

Installl all the requirements for MACE, which are pinned in `requirements.txt`.
```sh
pip3 install -r requirements.txt
```

### MACE Installation
At last, the MACE installation. Please follow the instructions [here](https://mace.readthedocs.io/en/latest/user/installation.html) to install MACE. 

Please clone the fork of the MACE repository that I've created. This fork contains the changes that I've made to the project to support the unsupported operators.
```sh
git clone git@github.com:briancpark/mace.git
```


sudo apt-get install libncurses5


### Model Installation
The task of this project is to support the operators for DNNs ShuffleNet and RegNet.

#### ShuffleNet
git clone git@github.com:megvii-model/ShuffleNet-Series.git
#### RegNet
git clone git@github.com:d-li14/regnet.pytorch.git

### Evaluation
The project was evaluated on a Mi 11 Lite. Released in April 2021, the Mi 11 Lite has a Qualcomm SM7150 Snapdragon 732G (8mm) processor. The CPU is an octa-core (2x2.3 GHz Kryo 470 Gold & 6x1.8 GHz Kryo 470 Silver) processor. The GPU is an Adreno 618. The device has 6GB of RAM and 128GB of storage. The device has a 6.55" AMOLED display with a resolution of 1080x2400 pixels. 

The OS is Android 11 (RKQ1.200826.002), with MIUI 12.5.1. The device has a 4,250 mAh battery. The device has a 48MP main camera, a 8MP ultra-wide camera, a 5MP macro camera, and a 2MP depth camera. The device has a 16MP front-facing camera.
Source: https://www.gsmarena.com/xiaomi_mi_11_lite-10665.php