# Development and Debugging Logs
A set of logs and debugging notes to help with devleopment process 

## Overview
Table of Contents
- [Overview](#overview)
- [I cannot login to my Android shell?](#i-cannot-login-to-my-android-shell)
- [RegNet Notes](#regnet-notes)
- [ShuffleNet Notes](#shufflenet-notes)
- [Results](#results)
- [How to Debug with GDB (An Attempt)](#how-to-debug-with-gdb-an-attempt)


## I cannot login to my Android shell?
If you cannot login to you android shell via adb because you get the following message:
```
no permissions (user bcpark is not in the plugdev group)
```
Then you may need to troubleshoot this by restarting the adb server. You can do so by running the following commands:
```sh
sudo adb kill-server
sudo adb start-server
```

Note that upon rebooting the server, you will need to physically reauthenticate the Android device in order to login the shell and let MACE run models on the phone. 

## RegNet Notes
All that needs to be supported is convolutions with groups, or grouped convolutions. There are a total of 13 grouped convolutions in the RegNet model.

To debug and ensure correctness, I have outputted all the information needed to replicate step by step on another deep learning framework or library. Note that all of the parameters are based on the assumption of batch size 1. They all have dialation of 1. They are all zero padded.
Here are the nodes that are group convolutions in the ONNX model (Assume NCHW and IOHW formats):

| idx (MACE internal) | Node Name | input | output | weight | bias | kernel | stride | padding | groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2  | Conv_4  | 1,24,112,112 | 1,24,56,56 | 24,8,3,3 | 24 | 3,3 | 2,2 | 1,1,1,1 | 3 |
| 9  | Conv_12 | 1,56,56,56 | 1,56,28,28 | 56,8,3,3 | 56 | 3,3 | 2,2 | 1,1,1,1 | 7 |
| 16 | Conv_20 | 1,152,28,28 | 1,152,14,14 | 152,8,3,3 | 152 | 3,3 | 2,2 | 1,1,1,1 | 19 |
| 23 | Conv_28 | 1,152,14,14 | 1,152,14,14 | 152,8,3,3 | 152 | 3,3 | 1,1 | 1,1,1,1 | 19 |
| 29 | Conv_35 | 1,152,14,14 | 1,152,14,14 | 152,8,3,3 | 152 | 3,3 | 1,1 | 1,1,1,1 | 19 |
| 35 | Conv_42 | 1,152,14,14 | 1,152,14,14 | 152,8,3,3 | 152 | 3,3 | 1,1 | 1,1,1,1 | 19 |
| 41 | Conv_49 | 1,368,14,14 | 1,368,7,7 | 368,8,3,3 | 368 | 3,3 | 2,2 | 1,1,1,1 | 46 |
| 48 | Conv_57 | 1,368,7,7 | 1,368,7,7 | 368,8,3,3 | 368 | 3,3 | 1,1 | 1,1,1,1 | 46 |
| 54 | Conv_64 | 1,368,7,7 | 1,368,7,7 | 368,8,3,3 | 368 | 3,3 | 1,1 | 1,1,1,1 | 46 |
| 60 | Conv_71 | 1,368,7,7 | 1,368,7,7 | 368,8,3,3 | 368 | 3,3 | 1,1 | 1,1,1,1 | 46 |
| 66 | Conv_78 | 1,368,7,7 | 1,368,7,7 | 368,8,3,3 | 368 | 3,3 | 1,1 | 1,1,1,1 | 46 |
| 72 | Conv_85 | 1,368,7,7 | 1,368,7,7 | 368,8,3,3 | 368 | 3,3 | 1,1 | 1,1,1,1 | 46 |
| 78 | Conv_92 |[1,368,7,7 | 1,368,7,7 | 368,8,3,3 | 368 | 3,3 | 1,1 | 1,1,1,1 | 46 |

Here are just the unique microkernels to focus on:
| input | output | weight | bias | kernel | stride | padding | groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1,24,112,112 | 1,24,56,56 | 24,8,3,3 | 24 | 3,3 | 2,2 | 1,1,1,1 | 3 |
| 1,56,56,56 | 1,56,28,28 | 56,8,3,3 | 56 | 3,3 | 2,2 | 1,1,1,1 | 7 |
| 1,152,28,28 | 1,152,14,14 | 152,8,3,3 | 152 | 3,3 | 2,2 | 1,1,1,1 | 19 |
| 1,152,14,14 | 1,152,14,14 | 152,8,3,3 | 152 | 3,3 | 1,1 | 1,1,1,1 | 19 |
| 1,368,14,14 | 1,368,7,7 | 368,8,3,3 | 368 | 3,3 | 2,2 | 1,1,1,1 | 46 |
| 1,368,7,7 | 1,368,7,7 | 368,8,3,3 | 368 | 3,3 | 1,1 | 1,1,1,1 | 46 |


Note that MACE prefers NHWC format on the GPU since that is optimal for GPU and OpenCL. The CPU preferes NCHW format. I believe CPU is more flexible, but there may be performance penalties as the MACE focuses more on NCHW format for CPU.


When compared against the CPU and GPU configrations, here are some notes:
* CPU compiles to a total of 86 ops, this is because I didn't fuse any activations with the group convolutions
* GPU compiles to a total of 75 ops
* Only the last GEMM or matmul will fallback to CPU on GPU configuration

## ShuffleNet Notes
ShuffleNet supports grouped convolution on the CPU and GPU, but for GPU, it only supports groups of 4. The task is to support groups of 2 for the specific ShuffleNet model that I am using. There are a total of 16 channel shuffle ops in the ShuffleNet model.

|         Op Type |  Output Shape |           name |
|-----------------|---------------|----------------|
|  ChannelShuffle |  [1,48,28,28] |   Transpose_34 |
|  ChannelShuffle |  [1,48,28,28] |   Transpose_82 |
|  ChannelShuffle |  [1,48,28,28] |  Transpose_130 |
|  ChannelShuffle |  [1,48,28,28] |  Transpose_178 |
|  ChannelShuffle |  [1,96,14,14] |  Transpose_215 |
|  ChannelShuffle |  [1,96,14,14] |  Transpose_263 |
|  ChannelShuffle |  [1,96,14,14] |  Transpose_311 |
|  ChannelShuffle |  [1,96,14,14] |  Transpose_359 |
|  ChannelShuffle |  [1,96,14,14] |  Transpose_407 |
|  ChannelShuffle |  [1,96,14,14] |  Transpose_455 |
|  ChannelShuffle |  [1,96,14,14] |  Transpose_503 |
|  ChannelShuffle |  [1,96,14,14] |  Transpose_551 |
|  ChannelShuffle |   [1,192,7,7] |  Transpose_588 |
|  ChannelShuffle |   [1,192,7,7] |  Transpose_636 |
|  ChannelShuffle |   [1,192,7,7] |  Transpose_684 |
|  ChannelShuffle |   [1,192,7,7] |  Transpose_732 |

When compared against the CPU and GPU configrations, here are some notes:
* CPU compiles to a total of 117 ops
* GPU compiles to a total of 224 ops
* GEMM, Reduce Mean, Slice, Concat all fallback to CPU on GPU configuration
* There are drastically more ops on the GPU configuration because of the fallbacks to CPU, and it incurs a lot of overhead because there are transpose ops inserted to convert from NHWC to NCHW and vice versa.

## Results
Here, I show some logs and results from running the models on the Android device. 

### RegNet

#### RegNet CPU (Xiaomi Mi 11 Lite)

```
I mace/utils/statistics.cc:343] ------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343]                                      Stat by Op Type
I mace/utils/statistics.cc:343] ------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343] |     Op Type | Count | Avg(ms) |      % |    cdf% |        MACs | GMACPS | Called times |
I mace/utils/statistics.cc:343] ------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343] | GroupConv2d |    13 | 302.205 | 59.413 |  59.413 |           0 |  0.000 |           13 |
I mace/utils/statistics.cc:343] |      Conv2D |    31 | 161.529 | 31.756 |  91.170 | 172,423,552 |  1.067 |           31 |
I mace/utils/statistics.cc:343] |  Activation |    26 |  30.282 |  5.953 |  97.123 |           0 |  0.000 |           26 |
I mace/utils/statistics.cc:343] |     Eltwise |    13 |  14.278 |  2.807 |  99.930 |           0 |  0.000 |           13 |
I mace/utils/statistics.cc:343] |      MatMul |     1 |   0.290 |  0.057 |  99.987 |     368,000 |  1.269 |            1 |
I mace/utils/statistics.cc:343] |      Reduce |     1 |   0.063 |  0.012 |  99.999 |           0 |  0.000 |            1 |
I mace/utils/statistics.cc:343] |     Reshape |     1 |   0.003 |  0.001 | 100.000 |           0 |  0.000 |            1 |
I mace/utils/statistics.cc:343] ------------------------------------------------------------------------------------------
```

#### RegNet GPU (Xiaomi Mi 11 Lite)
```
I mace/utils/statistics.cc:349] -----------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:349]                                        Stat by Op Type
I mace/utils/statistics.cc:349] -----------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:349] |         Op Type | Count | Avg(ms) |      % |    cdf% |        MACs |  GMACPS | Called times |
I mace/utils/statistics.cc:349] -----------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:349] |          Conv2D |    31 |   5.207 | 57.510 |  57.510 | 172,423,552 |  33.114 |           31 |
I mace/utils/statistics.cc:349] |          MatMul |     1 |   2.790 | 30.815 |  88.326 |     368,000 |   0.132 |            1 |
I mace/utils/statistics.cc:349] |     GroupConv2d |    13 |   0.501 |  5.533 |  93.859 |  77,432,544 | 154.556 |           13 |
I mace/utils/statistics.cc:349] |         Eltwise |    13 |   0.245 |  2.706 |  96.565 |           0 |   0.000 |           13 |
I mace/utils/statistics.cc:349] |      Activation |    13 |   0.162 |  1.789 |  98.354 |           0 |   0.000 |           13 |
I mace/utils/statistics.cc:349] | BufferTransform |     2 |   0.118 |  1.303 |  99.658 |           0 |   0.000 |            2 |
I mace/utils/statistics.cc:349] |          Reduce |     1 |   0.026 |  0.287 |  99.945 |           0 |   0.000 |            1 |
I mace/utils/statistics.cc:349] |         Reshape |     1 |   0.005 |  0.055 | 100.000 |           0 |   0.000 |            1 |
I mace/utils/statistics.cc:349] -----------------------------------------------------------------------------------------------
```

#### RegNet GPU (Google Pixel 7)
```
I mace/utils/statistics.cc:343] ----------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343]                                        Stat by Op Type
I mace/utils/statistics.cc:343] ----------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343] |         Op Type | Count | Avg(ms) |      % |    cdf% |        MACs | GMACPS | Called times |
I mace/utils/statistics.cc:343] ----------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343] |          Conv2D |    31 |  19.059 | 72.973 |  72.973 | 172,423,552 |  9.047 |           31 |
I mace/utils/statistics.cc:343] |          MatMul |     1 |   5.693 | 21.797 |  94.770 |     368,000 |  0.065 |            1 |
I mace/utils/statistics.cc:343] |         Eltwise |    13 |   0.621 |  2.378 |  97.148 |           0 |  0.000 |           13 |
I mace/utils/statistics.cc:343] |      Activation |    13 |   0.493 |  1.888 |  99.035 |           0 |  0.000 |           13 |
I mace/utils/statistics.cc:343] | BufferTransform |     2 |   0.133 |  0.509 |  99.544 |           0 |  0.000 |            2 |
I mace/utils/statistics.cc:343] |          Reduce |     1 |   0.104 |  0.398 |  99.943 |           0 |  0.000 |            1 |
I mace/utils/statistics.cc:343] |         Reshape |     1 |   0.015 |  0.057 | 100.000 |           0 |  0.000 |            1 |
I mace/utils/statistics.cc:343] |     GroupConv2d |    13 |   0.000 |  0.000 | 100.000 |           0 |  0.000 |           13 |
I mace/utils/statistics.cc:343] ----------------------------------------------------------------------------------------------
```

### ShuffleNet V2+

#### ShuffleNet CPU (Xiaomi Mi 11 Lite)
```
I mace/utils/statistics.cc:343] |         Op Type | Count | Avg(ms) |      % |    cdf% |       MACs | GMACPS | Called times |
I mace/utils/statistics.cc:343] ---------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343] |          Conv2D |    37 |   6.138 | 56.713 |  56.713 | 37,632,000 |  6.131 |           37 |
I mace/utils/statistics.cc:343] |         Pooling |     1 |   1.594 | 14.728 |  71.440 |          0 |  0.000 |            1 |
I mace/utils/statistics.cc:343] | DepthwiseConv2d |    19 |   1.428 | 13.194 |  84.635 |  1,820,448 |  1.275 |           19 |
I mace/utils/statistics.cc:343] |          MatMul |     1 |   0.510 |  4.712 |  89.347 |  1,024,000 |  2.008 |            1 |
I mace/utils/statistics.cc:343] |          Concat |    16 |   0.361 |  3.335 |  92.682 |          0 |  0.000 |           16 |
I mace/utils/statistics.cc:343] |  ChannelShuffle |    16 |   0.346 |  3.197 |  95.879 |          0 |  0.000 |           16 |
I mace/utils/statistics.cc:343] |           Slice |    26 |   0.346 |  3.197 |  99.076 |          0 |  0.000 |           26 |
I mace/utils/statistics.cc:343] |          Reduce |     1 |   0.100 |  0.924 | 100.000 |          0 |  0.000 |           
```

#### ShuffleNet GPU (Xiaomi Mi 11 Lite)
```
I mace/utils/statistics.cc:343] |         Op Type | Count | Avg(ms) |      % |    cdf% |       MACs | GMACPS | Called times |
I mace/utils/statistics.cc:343] ---------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343] |       Transpose |    53 |  71.720 | 75.451 |  75.451 |          0 |  0.000 |           53 |
I mace/utils/statistics.cc:343] |          MatMul |     1 |  10.605 | 11.157 |  86.608 |  1,024,000 |  0.097 |            1 |
I mace/utils/statistics.cc:343] |          Reduce |     1 |   5.205 |  5.476 |  92.084 |          0 |  0.000 |            1 |
I mace/utils/statistics.cc:343] |           Slice |    26 |   3.586 |  3.773 |  95.856 |          0 |  0.000 |           26 |
I mace/utils/statistics.cc:343] |          Conv2D |    37 |   1.487 |  1.564 |  97.420 | 37,632,000 | 25.307 |           37 |
I mace/utils/statistics.cc:343] | BufferTransform |    54 |   0.903 |  0.950 |  98.370 |          0 |  0.000 |           54 |
I mace/utils/statistics.cc:343] |          Concat |    16 |   0.839 |  0.883 |  99.253 |          0 |  0.000 |           16 |
I mace/utils/statistics.cc:343] | DepthwiseConv2d |    19 |   0.385 |  0.405 |  99.658 |  1,820,448 |  4.728 |           19 |
I mace/utils/statistics.cc:343] |  ChannelShuffle |    16 |   0.220 |  0.231 |  99.890 |          0 |  0.000 |           16 |
I mace/utils/statistics.cc:343] |         Pooling |     1 |   0.105 |  0.110 | 100.000 |          0 |  0.000 |       
```

## How to Debug with GDB (An Attempt)
This configuration is impossible unless you have a rooted device, which requires you to unlock the bootloader and flash a custom recovery image.

All the files to run are pushed to `/data/local/tmp/mace_run`

Here's an example of what the executable looks like if you want to run it directly in the shell under `cmd_file-r**`:
```
LD_LIBRARY_PATH=/data/local/tmp/mace_run MACE_TUNING=0 MACE_OUT_OF_RANGE_CHECK=0 MACE_CPP_MIN_VLOG_LEVEL=0 MACE_RUN_PARAMETER_PATH=/data/local/tmp/mace_run/mace_run.config MACE_INTERNAL_STORAGE_PATH=/data/local/tmp/mace_run/interior/ MACE_LIMIT_OPENCL_KERNEL_TIME=0 MACE_OPENCL_QUEUE_WINDOW_SIZE=0 MACE_RUNTIME_FAILURE_RATIO=0.000000 MACE_LOG_TENSOR_RANGE=0 /data/local/tmp/mace_run/mace_run_static --model_name=regnet --input_node='input' --output_node='output' --input_shape=1,224,224,3 --output_shape=1,1000 --input_data_type=float32 --output_data_type=float32 --input_data_format=NHWC --output_data_format=NHWC --input_file=/data/local/tmp/mace_run/model_input --output_file=/data/local/tmp/mace_run/model_out --input_dir= --output_dir= --model_data_file=/data/local/tmp/mace_run/regnet.data --round=1 --restart_round=1 --num_threads=-1 --cpu_affinity_policy=1 --opencl_cache_reuse_policy=1 --opencl_cache_full_path=/data/local/tmp/mace_run/interior/mace_cl_compiled_program.bin --gpu_perf_hint=3 --gpu_priority_hint=3 --model_file=/data/local/tmp/mace_run/regnet.pb --opencl_binary_file=/data/local/tmp/mace_run/regnet_compiled_opencl_kernel.M2101K9AG.sm6150.bin --opencl_parameter_file=/data/local/tmp/mace_run/regnet_tuned_opencl_parameter.M2101K9AG.sm6150.bin --accelerator_cache_policy=0 --accelerator_binary_file= --accelerator_storage_file= --apu_boost_hint=100 --apu_preference_hint=1
```

These are the series of commands I've tried. I've had no success unfortunately, so print/log statements it is...
```sh
# push gdbserver to your phone
adb push $ANDROID_NDK_HOME/prebuilt/android-arm64/gdbserver/gdbserver /data/local/tmp/


# set system env, pull system libs and bins to host
export SYSTEM_LIB=~/dev/system_lib
export SYSTEM_BIN=~/dev/system_bin
mkdir -p $SYSTEM_LIB
adb pull /system/lib/. $SYSTEM_LIB
mkdir -p $SYSTEM_BIN
adb pull /system/bin/. $SYSTEM_BIN


# Suppose ndk compiler used to compile Mace is of android-21
export PLATFORMS_21_LIB=$ANDROID_NDK_HOME/platforms/android-21/arch-arm/usr/lib/


# start gdbserver，make gdb listen to port 6000
adb shell /data/local/tmp/gdbserver :6000 /data/local/tmp/mace_run/cmd_file-shufflenet-1682374297.1477149

adb shell LD_LIBRARY_PATH=/data/local/tmp/mace_run /data/local/tmp/gdbserver :6000 /data/local/tmp/mace_run/example_bin
# or attach a running process
adb shell /data/local/tmp/gdbserver :6000 --attach 8700
# forward tcp port
adb forward tcp:6000 tcp:6000


# use gdb on host to execute binary
$ANDROID_NDK_HOME/prebuilt/linux-x86_64/bin/gdb [/path/to/binary/on/host/example_bin]


# connect remote port after starting gdb command
target remote :6000


# set lib path
set solib-search-path $SYSTEM_LIB:$SYSTEM_BIN:$PLATFORMS_21_LIB

# then you can use it as host gdb, e.g.,
bt
```
