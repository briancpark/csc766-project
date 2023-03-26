# CSC 766 Final Project

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

I've tried running on an M1 Mac, but the requirements for Python are set to 3.6, which is difficult to get installed natively on the ARM ISA. It's possible via Rosetta, but for convenience I've been using a Linux machine with sudo permissions enabled.

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

