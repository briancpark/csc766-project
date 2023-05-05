# Optimizing DNN Operators on Mobile GPUs

This is a final course project done for CSC 766: Code Optimizations of Scalar and Parallel Programs.

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
  - [Model Installation](#model-installation)
- [Evaluation](#evaluation)
- [Debugging and Running Notes](#debugging-and-running-notes)
- [Implementation](#implementation)


## Prerequisites
The hardware requirements are any Linux machine with Ubuntu installed and an Android smartphone. 

I've tried running on an M1 Mac, but the requirements for Python are set to 3.7, which is difficult to get installed natively on an ARM platform. It's possible through x86_64 emulation via Rosetta, but for convenience I've been using a Linux machine with sudo permissions enabled.

## Setup
First, clone the repository and all submodules.
```sh
git clone --recurse git@github.com:briancpark/csc766-project.git
```

Next, you'll need to install the dependencies as required by MACE.

### CMake Installation
CMake version 3.11.3 or higher is required to build MACE. This may not be required unless Bazel doesn't work.
```sh
sudo apt install cmake
```

### Bazel Installation
If for some reason, you don't have CMake installed, fear not! You can use Bazel to build the project instead. This project is acutally developed under Bazel environment, and the commands below assume a Bazel environment. Run the commands below. Please be logged in as root (via `sudo bash`).
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
MACE requires the Android NDK to be installed. Please follow the instructions [here](https://developer.android.com/ndk/guides) to install the NDK. It's very strict requirement under Bazel environment to use anything below NDK r16c, or else there might be compilation errors. 

```sh
cd /opt/ && \
    wget -q https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip && \
    unzip -q android-ndk-r15c-linux-x86_64.zip && \
    rm -f android-ndk-r15c-linux-x86_64.zip
```

After successfully installing, please add the following to `~/.bashrc`.
```sh
export ANDROID_NDK_VERSION=r15c
export ANDROID_NDK=/opt/android-ndk-${ANDROID_NDK_VERSION}
export ANDROID_NDK_HOME=${ANDROID_NDK}

# add to PATH
export PATH=${PATH}:${ANDROID_NDK_HOME}
```

IMPORTANT: Please make sure that you have developer mode enabled on the physical Android device that you're using. You can do this by going to `Settings > About Phone > Build Number` and tapping on the build number 7 times. After that, you should see a message that says `You are now a developer!`. Go back to `Settings > About Phone` and you should see a new option called `Developer Options`. Enable this option.

In addition, you need libcurses, which may not be installed by default in Ubuntu.
```sh
sudo apt-get install libncurses5
```

When connecting an Android device, you may need to toggle on USB Debugging in Developer Options. Once that is done and connected to USB, approve the connection. 

Running `adb devices` should show your devices. Run `adb shell` to get a shell on the device.

### Python Installation


Please create a Conda environment with Python 3.7 installed. If you don't have Conda installed, please follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

```sh
conda create -n csc766 python=3.7
pip3 install --upgrade pip
```
Install all the requirements for MACE, which are pinned in `requirements.txt`. It's very crucial that you install the exact versions of the packages listed in the file. Sometimes, different ONNX versions produce slightly different outputs, which can cause the MACE conversion to fail.
```sh
pip3 install -r requirements.txt
```

**NOTE**: If you are trying to replicate the sample model examples that MACE provides under [MACE Model Zoo](https://github.com/XiaoMi/mace-models), then you'll need to create a separate Conda environment with Python 3.6 installed. You *MUST* use Python 3.6 for running those examples. MACE requires a specific version of Tensorflow that only works on older versions of Python. Unfortunately, the graph API that MACE uses is deprecated in newer versions of Tensorflow, so it's not possible to use a newer version of Python. But for the purposes of this project, we can use Python 3.7. because we don't use the Tensorflow backend, but the ONNX backend instead.

### MACE Installation
At last, the MACE installation. Please follow the instructions [here](https://mace.readthedocs.io/en/latest/user/installation.html) to install MACE. 

Clone the fork of the MACE repository that I've created. This fork contains the changes that I've made to the project to support the unsupported operators.

```sh
git clone git@github.com:briancpark/mace.git
```

### Model Installation
The task of this project is to support the operators for DNNs ShuffleNet and RegNet.

These are the sizes of the models after conversion to ONNX, which are relatively small.

```
11M     regnet.onnx
5.3M    shufflenet_opt_pt-sim.onnx
```

The models are available in the `onnx_models` directory.

## Evaluation
The project was evaluated on a XiaoMi Mi 11 Lite. Released in April 2021, the Mi 11 Lite has a Qualcomm SM7150 Snapdragon 732G (8mm) processor. The CPU is an octa-core (2x2.3 GHz Kryo 470 Gold & 6x1.8 GHz Kryo 470 Silver) processor. The GPU is an Adreno 618. The device has 6GB of RAM and 128GB of storage. The device has a 6.55" AMOLED display with a resolution of 1080x2400 pixels. 

The OS is Android 11 (RKQ1.200826.002), with MIUI 12.5.1. The device has a 4,250 mAh battery, 48MP main camera, a 8MP ultra-wide camera, a 5MP macro camera, a 2MP depth camera, and a 16MP front-facing camera. Source: https://www.gsmarena.com/xiaomi_mi_11_lite-10665.php

## Debugging and Running Notes
First you will need to optimize ONNX files as follows:
```sh
python tools/onnx_optimizer.py ../onnx_models/regnet.onnx ../onnx_models/regnet_opt.onnx 
python tools/onnx_optimizer.py ../onnx_models/shufflenet.onnx ../onnx_models/shufflenet_opt.onnx
```
 
Once done so, check the SHA256 checksums to make sure that the files are the same. You will later need to have the checksum for the configuration file, but I've already included it in the repository. So the following commands are just for reference.
```sh
sha256sum /home/bcpark/csc766-project/onnx_models/regnet_opt.onnx 
sha256sum /home/bcpark/csc766-project/onnx_models/shufflenet_opt.onnx
```

**IMPORTANT**: When compiling and running between the two models, please make sure to checkout the correct branch that contains the changes for each model. The `master` branch is just a copy of the original MACE repository, and does not contain the changes for the unsupported operators.

`regnet-gpu-optimized` contains the changes for RegNet, and `shufflenet-gpu-optimized` contains the changes for ShuffleNet.

```sh
cd mace
git checkout regnet-gpu-optimized
# Run RegNet related commands

git checkout shufflenet-gpu-optimized
# Run ShuffleNet related commands
```

To compile and run RegNet:
```sh
python tools/converter.py convert --config=../deployment_config/regnet.yml --debug_mode --vlog_level=3 && \
python tools/converter.py run --config=../deployment_config/regnet.yml --debug_mode --vlog_level=3
```

To compile and run ShuffleNet:
```sh
python tools/converter.py convert --config=../deployment_config/shufflenet.yml --debug_mode --vlog_level=3 && \
python tools/converter.py run --config=../deployment_config/shufflenet.yml --debug_mode --vlog_level=3
```

To verify model correctness, you can run the following command:
```sh
python tools/converter.py run --config=../deployment_config/regnet.yml --validate
python tools/converter.py run --config=../deployment_config/shufflenet.yml --validate
```

A correct model will output the following in green. This is tested on sythetic data. Simulatriy is cosine simularity, where values closer to 1.0 is correct. SQNR is signal to quantization noise ratio, where higher is better. Pixel accuracy is the percentage of pixels that are the same between the two models, where values closer to 1.0 is correct. You can learn more [here](https://mace.readthedocs.io/en/latest/development/how_to_debug.html).

```
output MACE VS ONNX similarity: 0.9999781457219321 , sqnr: 22823.096153305403 , pixel_accuracy: 1.0
******************************************
          Similarity Test Passed          
******************************************
```

An incorrect model will output the following in red:

```
output MACE VS ONNX similarity: 0.017589891526221674 , sqnr: 0.10534350168590623 , pixel_accuracy: 0.0
ERROR: [] /home/bcpark/csc766-project/mace/tools/validate.py:131: ******************************************
          Similarity Test Failed          
******************************************
```

You can validate the model layer by layer, but this can give some false positives since the model is cross validated against ONNX model, and some MACE ops are fused or optimized. 
```sh
python tools/converter.py run --config=../deployment_config/regnet.yml --validate --layers :
python tools/converter.py run --config=../deployment_config/shufflenet.yml --validate --layers :
```

To benchmark the model, you can run the following command:
```sh
# RegNet
python tools/converter.py run --config=../deployment_config/regnet.yml --benchmark --round=1000 --gpu_priority_hint=3
# ShuffleNet
python tools/converter.py run --config=../deployment_config/shufflenet.yml --benchmark --round=1000 --gpu_priority_hint=3
```

To benchmark a specific OP (TODO: this doesn't work yet for the ops I've implemented):
```sh
python tools/bazel_adb_run.py --target="//test/ccbenchmark:mace_cc_benchmark" --run_target=True  --args="--filter=.*CONV.*"
```

The files are uploaded onto `/data/local/tmp/mace_run/` directory on the device.


After doing this project, it turns out not all ops are fully supported for each hardware target. Look here for a comprehensive list of supported operators: https://mace.readthedocs.io/en/latest/user_guide/op_lists.html

## Run MACE Model Zoo Models
This is for my own reference, when I want to debug through a fully implemented model for reference. 

```sh
# Convert the model
python tools/converter.py convert --config=../mace-models/mobilenet-v2/mobilenet-v2.yml

# Run the model
python tools/converter.py run --config=../mace-models/mobilenet-v2/mobilenet-v2.yml

# Test model run time
python tools/converter.py run --config=../mace-models/mobilenet-v2/mobilenet-v2.yml --round=100

# Validate the correctness by comparing the results against the
# original model and framework, measured with cosine distance for similarity.
python tools/converter.py run --config=../mace-models/mobilenet-v2/mobilenet-v2.yml --validate
```

## Implementation
The diff stat for each branch is as follows.

For `regnet-gpu-optimized`, the OpenCL implementation lies in `mace/ops/opencl/cl/group_conv2d.cl`.
```sh
git diff --stat origin
 mace/core/net/serial_net.cc                          |  45 +++---
 mace/core/net_optimizer.cc                           |   5 +-
 mace/ops/arm/base/group_conv2d.cc                    | 277 ++++++++++++++++++++++++++++++++++
 mace/ops/arm/base/group_conv2d.h                     | 132 ++++++++++++++++
 mace/ops/arm/base/group_conv2d_3x3.cc                |  36 +++++
 mace/ops/arm/base/group_conv2d_3x3.h                 |  59 ++++++++
 mace/ops/arm/base/group_conv2d_mxn.h                 |  87 +++++++++++
 mace/ops/arm/fp32/group_conv2d_3x3.cc                | 401 +++++++++++++++++++++++++++++++++++++++++++++++++
 mace/ops/delegator/group_conv2d.h                    |  96 ++++++++++++
 mace/ops/group_conv2d.cc                             | 263 ++++++++++++++++++++++++++++++++
 mace/ops/group_conv2d.h                              | 103 +++++++++++++
 mace/ops/opencl/buffer/group_conv2d.h                |  96 ++++++++++++
 mace/ops/opencl/cl/group_conv2d.cl                   | 169 +++++++++++++++++++++
 mace/ops/opencl/group_conv2d.h                       |  57 +++++++
 mace/ops/opencl/image/group_conv2d.cc                | 130 ++++++++++++++++
 mace/ops/opencl/image/group_conv2d.h                 | 102 +++++++++++++
 mace/ops/opencl/image/group_conv2d_3x3.cc            | 194 ++++++++++++++++++++++++
 mace/ops/opencl/image/group_conv2d_general.cc        | 204 +++++++++++++++++++++++++
 mace/ops/ref/group_conv2d.cc                         | 134 +++++++++++++++++
 mace/ops/registry/op_delegators_registry.cc          |   8 +-
 mace/ops/registry/ops_registry.cc                    |   2 +
 mace/utils/statistics.cc                             |  96 ++++++------
 repository/opencl-kernel/opencl_kernel_configure.bzl |   2 +
 tools/device.py                                      |   5 +
 tools/python/transform/base_converter.py             |   2 +
 tools/python/transform/onnx_converter.py             |  17 ++-
 tools/python/transform/transformer.py                |   1 +
 27 files changed, 2642 insertions(+), 81 deletions(-)
```

For `shufflenet-gpu-optimized`, the OpenCLI implementation lies in `mace/ops/opencl/cl/channel_shuffle.cl`.
```sh
git diff --stat origin
 mace/core/net_def_adapter.cc             | 416 ++++++++++++++++++++++++++++---------------------------------
 mace/core/net_optimizer.cc               |   5 +-
 mace/ops/channel_shuffle.cc              |  26 ++--
 mace/ops/opencl/cl/channel_shuffle.cl    |  46 +++----
 mace/ops/opencl/image/channel_shuffle.cc |  16 ++-
 tools/device.py                          |   5 +
 tools/python/transform/transformer.py    |  12 +-
 7 files changed, 244 insertions(+), 282 deletions(-)
```

Some more detailed debugging notes and results obtained are shown in [DEVELPMENT.md](DEVELOPMENT.md).