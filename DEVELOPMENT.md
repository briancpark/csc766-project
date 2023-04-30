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
ShuffleNet supports grouped convolution on the CPU and GPU, but for GPU, it only supports groups of 4. The task is to support groups of 2 for the specific ShuffleNet model that I am using. There are a total of 16 channel shuffle ops in the ShuffleNet model. The input shape is same as the output shape.

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
CPU Run:
```
========================================================
     capability(CPU)        init      warmup     run_avg
========================================================
time          24.795      22.323      66.165      18.434
```

Op Stats:
```
------------------------------------------------------------------------------------------
                                     Stat by Op Type
------------------------------------------------------------------------------------------
|     Op Type | Count | Avg(ms) |      % |    cdf% |        MACs | GMACPS | Called times |
------------------------------------------------------------------------------------------
|      Conv2D |    31 |  14.565 | 80.288 |  80.288 | 172,423,552 | 11.838 |           31 |
| GroupConv2d |    13 |   2.695 | 14.856 |  95.144 |   3,281,040 |  1.217 |           13 |
|     Eltwise |    13 |   0.510 |  2.811 |  97.955 |           0 |  0.000 |           13 |
|      MatMul |     1 |   0.187 |  1.031 |  98.986 |     368,000 |  1.968 |            1 |
|  Activation |    13 |   0.149 |  0.821 |  99.807 |           0 |  0.000 |           13 |
|      Reduce |     1 |   0.033 |  0.182 |  99.989 |           0 |  0.000 |            1 |
|     Reshape |     1 |   0.002 |  0.011 | 100.000 |           0 |  0.000 |            1 |
------------------------------------------------------------------------------------------
```

#### RegNet GPU (Xiaomi Mi 11 Lite)
========================================================
     capability(CPU)        init      warmup     run_avg
========================================================
time          25.273     657.812     958.826      15.472
```
----------------------------------------------------------------------------------------------
                                       Stat by Op Type
----------------------------------------------------------------------------------------------
|         Op Type | Count | Avg(ms) |      % |    cdf% |        MACs | GMACPS | Called times |
----------------------------------------------------------------------------------------------
|          MatMul |     1 |   7.044 | 61.194 |  61.194 |     368,000 |  0.052 |            1 |
|          Conv2D |    31 |   3.402 | 29.554 |  90.748 | 172,423,552 | 50.683 |           31 |
|     GroupConv2d |    13 |   0.461 |  4.005 |  94.753 |   3,281,040 |  7.117 |           13 |
|         Eltwise |    13 |   0.251 |  2.181 |  96.933 |           0 |  0.000 |           13 |
|      Activation |    13 |   0.205 |  1.781 |  98.714 |           0 |  0.000 |           13 |
| BufferTransform |     2 |   0.112 |  0.973 |  99.687 |           0 |  0.000 |            2 |
|          Reduce |     1 |   0.031 |  0.269 |  99.957 |           0 |  0.000 |            1 |
|         Reshape |     1 |   0.005 |  0.043 | 100.000 |           0 |  0.000 |            1 |
----------------------------------------------------------------------------------------------
```

### ShuffleNet V2+

#### ShuffleNet CPU (Xiaomi Mi 11 Lite)
========================================================
     capability(CPU)        init      warmup     run_avg
========================================================
time          27.272      17.908      58.789       8.962
```
---------------------------------------------------------------------------------------------
                                      Stat by Op Type
---------------------------------------------------------------------------------------------
|         Op Type | Count | Avg(ms) |      % |    cdf% |       MACs | GMACPS | Called times |
---------------------------------------------------------------------------------------------
|          Conv2D |    37 |   5.135 | 59.152 |  59.152 | 37,632,000 |  7.329 |           37 |
|         Pooling |     1 |   1.501 | 17.291 |  76.443 |          0 |  0.000 |            1 |
| DepthwiseConv2d |    19 |   1.083 | 12.476 |  88.918 |  1,820,448 |  1.681 |           19 |
|          MatMul |     1 |   0.510 |  5.875 |  94.793 |  1,024,000 |  2.008 |            1 |
|          Concat |    16 |   0.154 |  1.774 |  96.567 |          0 |  0.000 |           16 |
|  ChannelShuffle |    16 |   0.128 |  1.474 |  98.042 |          0 |  0.000 |           16 |
|           Slice |    26 |   0.108 |  1.244 |  99.286 |          0 |  0.000 |           26 |
|          Reduce |     1 |   0.062 |  0.714 | 100.000 |          0 |  0.000 |            1 |
```

#### ShuffleNet GPU (Xiaomi Mi 11 Lite)
```
========================================================
     capability(CPU)        init      warmup     run_avg
========================================================
time          25.368     943.820    1026.954      59.541
```
```
---------------------------------------------------------------------------------------------
                                      Stat by Op Type
---------------------------------------------------------------------------------------------
|         Op Type | Count | Avg(ms) |      % |    cdf% |       MACs | GMACPS | Called times |
---------------------------------------------------------------------------------------------
|       Transpose |    53 |  68.445 | 74.977 |  74.977 |          0 |  0.000 |           53 |
|          MatMul |     1 |  10.509 | 11.512 |  86.489 |  1,024,000 |  0.097 |            1 |
|          Reduce |     1 |   5.025 |  5.505 |  91.993 |          0 |  0.000 |            1 |
|           Slice |    26 |   3.429 |  3.756 |  95.750 |          0 |  0.000 |           26 |
|          Conv2D |    37 |   1.477 |  1.618 |  97.368 | 37,632,000 | 25.479 |           37 |
| BufferTransform |    54 |   0.902 |  0.988 |  98.356 |          0 |  0.000 |           54 |
|          Concat |    16 |   0.793 |  0.869 |  99.224 |          0 |  0.000 |           16 |
| DepthwiseConv2d |    19 |   0.385 |  0.422 |  99.646 |  1,820,448 |  4.728 |           19 |
|  ChannelShuffle |    16 |   0.216 |  0.237 |  99.883 |          0 |  0.000 |           16 |
|         Pooling |     1 |   0.107 |  0.117 | 100.000 |          0 |  0.000 |            1 |
---------------------------------------------------------------------------------------------
```









#### RegNet CPU (Moto G6)
CPU Run:
```
========================================================
     capability(CPU)        init      warmup     run_avg
========================================================
time          57.007      81.247      47.320      42.103
```

Op Stats:
```
------------------------------------------------------------------------------------------
                                     Stat by Op Type
------------------------------------------------------------------------------------------
|     Op Type | Count | Avg(ms) |      % |    cdf% |        MACs | GMACPS | Called times |
------------------------------------------------------------------------------------------
|      Conv2D |    31 |  32.965 | 79.826 |  79.826 | 172,423,552 |  5.231 |           31 |
| GroupConv2d |    13 |   6.493 | 15.723 |  95.549 |   3,281,040 |  0.505 |           13 |
|     Eltwise |    13 |   1.056 |  2.557 |  98.106 |           0 |  0.000 |           13 |
|      MatMul |     1 |   0.356 |  0.862 |  98.968 |     368,000 |  1.034 |            1 |
|  Activation |    13 |   0.355 |  0.860 |  99.828 |           0 |  0.000 |           13 |
|      Reduce |     1 |   0.065 |  0.157 |  99.985 |           0 |  0.000 |            1 |
|     Reshape |     1 |   0.006 |  0.015 | 100.000 |           0 |  0.000 |            1 |
------------------------------------------------------------------------------------------
```

#### RegNet GPU (Moto G6)
========================================================
     capability(CPU)        init      warmup     run_avg
========================================================
time          55.178    2700.688    3718.320      53.052
```
----------------------------------------------------------------------------------------------
                                       Stat by Op Type
----------------------------------------------------------------------------------------------
|         Op Type | Count | Avg(ms) |      % |    cdf% |        MACs | GMACPS | Called times |
----------------------------------------------------------------------------------------------
|          Conv2D |    31 |  38.747 | 76.160 |  76.160 | 172,423,552 |  4.450 |           31 |
|          MatMul |     1 |   6.887 | 13.537 |  89.697 |     368,000 |  0.053 |            1 |
|         Eltwise |    13 |   1.940 |  3.813 |  93.510 |           0 |  0.000 |           13 |
|      Activation |    13 |   1.811 |  3.560 |  97.069 |           0 |  0.000 |           13 |
|     GroupConv2d |    13 |   0.619 |  1.217 |  98.286 |   3,281,040 |  5.301 |           13 |
|          Reduce |     1 |   0.467 |  0.918 |  99.204 |           0 |  0.000 |            1 |
| BufferTransform |     2 |   0.399 |  0.784 |  99.988 |           0 |  0.000 |            2 |
|         Reshape |     1 |   0.006 |  0.012 | 100.000 |           0 |  0.000 |            1 |
----------------------------------------------------------------------------------------------
```
#### ShuffleNet CPU (Moto G6)

```
========================================================
     capability(CPU)        init      warmup     run_avg
========================================================
time          59.729      66.958      25.449      21.213
```
```
---------------------------------------------------------------------------------------------
                                      Stat by Op Type
---------------------------------------------------------------------------------------------
|         Op Type | Count | Avg(ms) |      % |    cdf% |       MACs | GMACPS | Called times |
---------------------------------------------------------------------------------------------
|          Conv2D |    37 |  12.892 | 62.118 |  62.118 | 37,632,000 |  2.919 |           37 |
|         Pooling |     1 |   2.685 | 12.937 |  75.055 |          0 |  0.000 |            1 |
| DepthwiseConv2d |    19 |   2.080 | 10.022 |  85.078 |  1,820,448 |  0.875 |           19 |
|          MatMul |     1 |   0.818 |  3.941 |  89.019 |  1,024,000 |  1.252 |            1 |
|  ChannelShuffle |    16 |   0.804 |  3.874 |  92.893 |          0 |  0.000 |           16 |
|          Concat |    16 |   0.714 |  3.440 |  96.333 |          0 |  0.000 |           16 |
|           Slice |    26 |   0.628 |  3.026 |  99.359 |          0 |  0.000 |           26 |
|          Reduce |     1 |   0.133 |  0.641 | 100.000 |          0 |  0.000 |            1 |
---------------------------------------------------------------------------------------------
```

#### ShuffleNet GPU (Moto G6)
========================================================
     capability(CPU)        init      warmup     run_avg
========================================================
time          55.252    4036.736    4707.018      99.509

---------------------------------------------------------------------------------------------
                                      Stat by Op Type
---------------------------------------------------------------------------------------------
|         Op Type | Count | Avg(ms) |      % |    cdf% |       MACs | GMACPS | Called times |
---------------------------------------------------------------------------------------------
|       Transpose |    53 |  42.171 | 61.327 |  61.327 |          0 |  0.000 |           53 |
|          Conv2D |    37 |  13.144 | 19.115 |  80.442 | 37,632,000 |  2.863 |           37 |
|          MatMul |     1 |   3.192 |  4.642 |  85.084 |  1,024,000 |  0.321 |            1 |
|           Slice |    26 |   2.553 |  3.713 |  88.796 |          0 |  0.000 |           26 |
| BufferTransform |    54 |   2.256 |  3.281 |  92.077 |          0 |  0.000 |           54 |
|          Reduce |     1 |   1.472 |  2.141 |  94.218 |          0 |  0.000 |            1 |
| DepthwiseConv2d |    19 |   1.317 |  1.915 |  96.133 |  1,820,448 |  1.382 |           19 |
|          Concat |    16 |   1.021 |  1.485 |  97.618 |          0 |  0.000 |           16 |
|  ChannelShuffle |    16 |   0.936 |  1.361 |  98.979 |          0 |  0.000 |           16 |
|         Pooling |     1 |   0.702 |  1.021 | 100.000 |          0 |  0.000 |            1 |
---------------------------------------------------------------------------------------------
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
