# Development and Debugging Logs
A set of logs and debugging notes to help with devleopment process 


### I cannot login to my Android shell?
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

### RegNet Notes
All that needs to be supported is convolutions with groups, or grouped convolutions.

To debug and ensure correctness, I have outputted all the information needed to replicate step by step on another deep learning framework or library. Note that all of the parameters are based on the assumption of batch size 1. They all have dialation of 1. They are all zero padded.
Here are the nodes that are group convolutions in the ONNX model:
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

Here are just the unique kernels to focus on:
| input | output | weight | bias | kernel | stride | padding | groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1,24,112,112 | 1,24,56,56 | 24,8,3,3 | 24 | 3,3 | 2,2 | 1,1,1,1 | 3 |
| 1,56,56,56 | 1,56,28,28 | 56,8,3,3 | 56 | 3,3 | 2,2 | 1,1,1,1 | 7 |
| 1,152,28,28 | 1,152,14,14 | 152,8,3,3 | 152 | 3,3 | 2,2 | 1,1,1,1 | 19 |
| 1,152,14,14 | 1,152,14,14 | 152,8,3,3 | 152 | 3,3 | 1,1 | 1,1,1,1 | 19 |
| 1,368,14,14 | 1,368,7,7 | 368,8,3,3 | 368 | 3,3 | 2,2 | 1,1,1,1 | 46 |
| 1,368,7,7 | 1,368,7,7 | 368,8,3,3 | 368 | 3,3 | 1,1 | 1,1,1,1 | 46 |


When GPU is on NHWC, when CPU only, NCHW. This is because the GPU is a Mali GPU, which is a NCHW GPU. T


How to debug with gdb
This configuration is impossible unless you have a rooted device, which requires you to unlock the bootloader and flash a custom recovery image.

All the files to run are pushed to:
/data/local/tmp/mace_run

Here's an example of what the executable looks like if you want to run it directly in the shell under `cmd_file-r**`:
```
LD_LIBRARY_PATH=/data/local/tmp/mace_run MACE_TUNING=0 MACE_OUT_OF_RANGE_CHECK=0 MACE_CPP_MIN_VLOG_LEVEL=0 MACE_RUN_PARAMETER_PATH=/data/local/tmp/mace_run/mace_run.config MACE_INTERNAL_STORAGE_PATH=/data/local/tmp/mace_run/interior/ MACE_LIMIT_OPENCL_KERNEL_TIME=0 MACE_OPENCL_QUEUE_WINDOW_SIZE=0 MACE_RUNTIME_FAILURE_RATIO=0.000000 MACE_LOG_TENSOR_RANGE=0 /data/local/tmp/mace_run/mace_run_static --model_name=regnet --input_node='input' --output_node='output' --input_shape=1,224,224,3 --output_shape=1,1000 --input_data_type=float32 --output_data_type=float32 --input_data_format=NHWC --output_data_format=NHWC --input_file=/data/local/tmp/mace_run/model_input --output_file=/data/local/tmp/mace_run/model_out --input_dir= --output_dir= --model_data_file=/data/local/tmp/mace_run/regnet.data --round=1 --restart_round=1 --num_threads=-1 --cpu_affinity_policy=1 --opencl_cache_reuse_policy=1 --opencl_cache_full_path=/data/local/tmp/mace_run/interior/mace_cl_compiled_program.bin --gpu_perf_hint=3 --gpu_priority_hint=3 --model_file=/data/local/tmp/mace_run/regnet.pb --opencl_binary_file=/data/local/tmp/mace_run/regnet_compiled_opencl_kernel.M2101K9AG.sm6150.bin --opencl_parameter_file=/data/local/tmp/mace_run/regnet_tuned_opencl_parameter.M2101K9AG.sm6150.bin --accelerator_cache_policy=0 --accelerator_binary_file= --accelerator_storage_file= --apu_boost_hint=100 --apu_preference_hint=1
```

```
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
adb shell /data/local/tmp/gdbserver :6000 /data/local/tmp/mace_run/cmd_file-regnet-1680304295.589991
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


Before adding ops
```
The scope of generated data:  ['-1,1']
Generate input file:  build/regnet/_tmp/regnet/a3b193840ca062479719b32409df0c42/M2101K9AG_sm6150/arm64-v8a/model_input_input
Generate input file done.
* Run 'regnet' with round=1, restart_round=1, tuning=False, out_of_range_check=False, num_threads=(-1,), cpu_affinity_policy=(1,), gpu_perf_hint=(3,), gpu_priority_hint=(3,)
Push build/regnet/_tmp/regnet/a3b193840ca062479719b32409df0c42/M2101K9AG_sm6150/arm64-v8a/model_input_input to /data/local/tmp/mace_run
Push build/regnet/model/regnet.data to /data/local/tmp/mace_run
Push third_party/nnlib/arm64-v8a/libhexagon_controller.so to /data/local/tmp/mace_run
Push build/regnet/model/regnet.pb to /data/local/tmp/mace_run/regnet.pb
Push build/regnet/_tmp/arm64-v8a/mace_run_static to /data/local/tmp/mace_run
Push /tmp/cmd_file-regnet-1680147584.626657 to /data/local/tmp/mace_run/cmd_file-regnet-1680147584.626657
I mace/tools/mace_run.cc:719] model name: regnet
I mace/tools/mace_run.cc:720] mace version: v1.1.1-20-g42ee056
I mace/tools/mace_run.cc:721] input node: input
I mace/tools/mace_run.cc:722] input shape: 1,224,224,3
I mace/tools/mace_run.cc:723] input data_type: float32
I mace/tools/mace_run.cc:724] input data_format: NHWC
I mace/tools/mace_run.cc:725] output node: output
I mace/tools/mace_run.cc:726] output shape: 1,1000
I mace/tools/mace_run.cc:727] output data_format: NHWC
I mace/tools/mace_run.cc:728] input_file: /data/local/tmp/mace_run/model_input
I mace/tools/mace_run.cc:729] output_file: /data/local/tmp/mace_run/model_out
I mace/tools/mace_run.cc:730] input dir:
I mace/tools/mace_run.cc:731] output dir:
I mace/tools/mace_run.cc:732] model_data_file: /data/local/tmp/mace_run/regnet.data
I mace/tools/mace_run.cc:733] model_file: /data/local/tmp/mace_run/regnet.pb
I mace/tools/mace_run.cc:734] accelerator_cache_policy: 0
I mace/tools/mace_run.cc:735] accelerator_binary_file:
I mace/tools/mace_run.cc:736] accelerator_storage_file:
I mace/tools/mace_run.cc:737] apu_boost_hint: 100
I mace/tools/mace_run.cc:738] apu_preference_hint: 1
I mace/tools/mace_run.cc:739] round: 1
I mace/tools/mace_run.cc:740] restart_round: 1
I mace/tools/mace_run.cc:741] gpu_perf_hint: 3
I mace/tools/mace_run.cc:742] gpu_priority_hint: 3
I mace/tools/mace_run.cc:743] num_threads: -1
I mace/tools/mace_run.cc:744] cpu_affinity_policy: 1
I mace/tools/mace_run.cc:747] limit_opencl_kernel_time: 0
I mace/tools/mace_run.cc:752] opencl_queue_window_size: 0
I mace/tools/mace_run.cc:786] raw_output_data_types[0] is float32
I mace/core/memory/rpcmem/rpcmem.cc:32] Rpcmem is supported. type: 0
I mace/rpcmems/rpcmem_factory.cc:53] Rpcmem is supported. type: 0
I mace/libmace/engines/serial_engine.cc:32] Creating SerialEngine, MACE version: v1.1.1-20-g42ee056
I mace/tools/mace_run.cc:811] restart round 0
W ./mace/utils/tuner.h:189] Failed to read tuned param file: /data/local/tmp/mace_run/regnet_tuned_opencl_parameter.M2101K9AG.sm6150.bin
I mace/core/memory/rpcmem/rpcmem.cc:32] Rpcmem is supported. type: 0
I mace/rpcmems/rpcmem_factory.cc:53] Rpcmem is supported. type: 0
I mace/libmace/engines/serial_engine.cc:32] Creating SerialEngine, MACE version: v1.1.1-20-g42ee056
W mace/core/kv_storage.cc:90] Failed to read kv store file: /data/local/tmp/mace_run/interior/mace_cl_compiled_program.bin
W mace/runtimes/opencl/core/opencl_executor.cc:508] Load OpenCL cached compiled kernel file failed. Please make sure the storage directory exist, the file is not modified illegally, and you have Write&Read permission
F mace/core/registry/ops_registry.cc:106] Check failed: registry_.count(op_type) != 0 GroupConv2d operation is not registered.
F mace/core/registry/ops_registry.cc:106] backtrace:
F mace/core/registry/ops_registry.cc:106]  pc 0x6347ad3bd0 _ZN4mace4port10AndroidEnv18GetBackTraceUnsafeEi
F mace/core/registry/ops_registry.cc:106]  pc 0x6347ad5ab8 _ZN4mace4port6Logger13DealWithFatalEv
F mace/core/registry/ops_registry.cc:106]  pc 0x6347ad5a50 _ZN4mace4port6Logger18GenerateLogMessageEv
F mace/core/registry/ops_registry.cc:106]  pc 0x6347ad5c34 _ZN4mace4port6LoggerD2Ev
F mace/core/registry/ops_registry.cc:106]  pc 0x6347ad5c90 _ZN4mace4port6LoggerD1Ev
F mace/core/registry/ops_registry.cc:106]  pc 0x6347a9c724 _ZNK4mace10OpRegistry17AvailableRuntimesERKSsPNS_18OpConditionContextE
F mace/core/registry/ops_registry.cc:106]  pc 0x6347a90178 _ZN4mace13NetDefAdapter11AdaptDeviceEPNS_18OpConditionContextEPNS_7RuntimeES4_RKSt13unordered_mapISsNS0_18InternalOutputInfoESt4hashISsESt8equal_toISsESaISt4pairIKSsS6_EEEPKNS_6NetDefEPNS_11OperatorDefE
F mace/core/registry/ops_registry.cc:106]  pc 0x6347a8f19c _ZN4mace13NetDefAdapter11AdaptNetDefEPKNS_6NetDefEPNS_7RuntimeES5_PS1_
F mace/core/registry/ops_registry.cc:106]  pc 0x634799b4f0 _ZN4mace10CpuRefFlow4InitEPKNS_6NetDefEPKhlPb
F mace/core/registry/ops_registry.cc:106]  pc 0x634798ff78 _ZN4mace12SerialEngine18CreateAndInitFlowsERKSt3mapIiPKNS_6NetDefESt4lessIiESaISt4pairIKiS4_EEERKSt13unordered_mapIS4_St10shared_ptrINS_7RuntAborted
Traceback (most recent call last):
  File "tools/converter.py", line 1333, in <module>
    flags.func(flags)
  File "tools/converter.py", line 1062, in run_mace
    device.run_specify_abi(flags, configs, target_abi)
  File "/home/bcpark/csc766-project/mace/tools/device.py", line 919, in run_specify_abi
    output_config, runtime, tuning)
  File "/home/bcpark/csc766-project/mace/tools/device.py", line 732, in run_model
    debug_mode=flags.debug_mode,
  File "/home/bcpark/csc766-project/mace/tools/device.py", line 491, in tuning_run
    _err_to_out=True)
  File "/home/bcpark/csc766-project/mace/tools/device.py", line 81, in exec_command
    sh.adb('-s', self.address, 'shell', command, *args, **kwargs)
  File "/home/bcpark/anaconda3/envs/csc766/lib/python3.6/site-packages/sh.py", line 1427, in __call__
    return RunningCommand(cmd, call_args, stdin, stdout, stderr)
  File "/home/bcpark/anaconda3/envs/csc766/lib/python3.6/site-packages/sh.py", line 774, in __init__
    self.wait()
  File "/home/bcpark/anaconda3/envs/csc766/lib/python3.6/site-packages/sh.py", line 792, in wait
    self.handle_command_exit_code(exit_code)
  File "/home/bcpark/anaconda3/envs/csc766/lib/python3.6/site-packages/sh.py", line 815, in handle_command_exit_code
    raise exc
sh.ErrorReturnCode_134: 

  RAN: /usr/bin/adb -s 3b16daec shell sh /data/local/tmp/mace_run/cmd_file-regnet-1680147584.626657

  STDOUT:


  STDERR:
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



GPU
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


CPU



GPU

I mace/utils/statistics.cc:343]                                        Stat by Op Type
I mace/utils/statistics.cc:343] ----------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343] |         Op Type | Count | Avg(ms) |      % |    cdf% |        MACs | GMACPS | Called times |
I mace/utils/statistics.cc:343] ----------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343] |      Activation |    26 |  53.806 | 47.933 |  47.933 |           0 |  0.000 |           26 |
I mace/utils/statistics.cc:343] |       Transpose |    26 |  46.368 | 41.307 |  89.240 |           0 |  0.000 |           26 |
I mace/utils/statistics.cc:343] |          MatMul |     1 |   5.870 |  5.229 |  94.470 |     368,000 |  0.063 |            1 |
I mace/utils/statistics.cc:343] |          Conv2D |    31 |   5.282 |  4.705 |  99.175 | 172,423,552 | 32.644 |           31 |
I mace/utils/statistics.cc:343] | BufferTransform |    28 |   0.627 |  0.559 |  99.734 |           0 |  0.000 |           28 |
I mace/utils/statistics.cc:343] |         Eltwise |    13 |   0.264 |  0.235 |  99.969 |           0 |  0.000 |           13 |
I mace/utils/statistics.cc:343] |          Reduce |     1 |   0.030 |  0.027 |  99.996 |           0 |  0.000 |            1 |
I mace/utils/statistics.cc:343] |         Reshape |     1 |   0.005 |  0.004 | 100.000 |           0 |  0.000 |            1 |
I mace/utils/statistics.cc:343] |     GroupConv2d |    13 |   0.000 |  0.000 | 100.000 |           0 |  0.000 |           13 |



I mace/utils/statistics.cc:343] ----------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343]                                        Stat by Op Type
I mace/utils/statistics.cc:343] ----------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343] |         Op Type | Count | Avg(ms) |      % |    cdf% |        MACs | GMACPS | Called times |
I mace/utils/statistics.cc:343] ----------------------------------------------------------------------------------------------
I mace/utils/statistics.cc:343] |          MatMul |     1 |   6.747 | 53.552 |  53.552 |     368,000 |  0.055 |            1 |
I mace/utils/statistics.cc:343] |          Conv2D |    31 |   5.245 | 41.630 |  95.182 | 172,423,552 | 32.874 |           31 |
I mace/utils/statistics.cc:343] |         Eltwise |    13 |   0.258 |  2.048 |  97.230 |           0 |  0.000 |           13 |
I mace/utils/statistics.cc:343] |      Activation |    13 |   0.202 |  1.603 |  98.833 |           0 |  0.000 |           13 |
I mace/utils/statistics.cc:343] | BufferTransform |     2 |   0.112 |  0.889 |  99.722 |           0 |  0.000 |            2 |
I mace/utils/statistics.cc:343] |          Reduce |     1 |   0.030 |  0.238 |  99.960 |           0 |  0.000 |            1 |
I mace/utils/statistics.cc:343] |         Reshape |     1 |   0.005 |  0.040 | 100.000 |           0 |  0.000 |            1 |
I mace/utils/statistics.cc:343] |     GroupConv2d |    13 |   0.000 |  0.000 | 100.000 |           0 |  0.000 |           13 |
I mace/utils/statistics.cc:343] ----------------------------------------------------------------------------------------------




PIXEL 7 REGNET GPU

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
