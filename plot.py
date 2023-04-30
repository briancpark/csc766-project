import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

channel_shuffle_times = {
    "CPU": 128,
    "GPU": 216,
}

group_conv_times = {
    "CPU": 2695,
    "GPU": 461,
}


shufflenet_times = {
    "CPU": 8.962,
    "GPU": 59.541,
}

regnet_times = {
    "CPU": 18.434,
    "GPU": 15.472,
}


shufflenet_times_coreml = {
    "CPU": 5.85,
    "GPU": 3.33,
    "ANE": 0.000,
}

regnet_times_coreml = {
    "CPU": 3.43,
    "GPU": 2.91,
    "ANE": 0.57,
}


def plot_op_performance(data, title, filename):
    target = list(data.keys())
    exec_time = list(data.values())
    x_pos = np.arange(len(target))

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        x_pos,
        exec_time,
        align="center",
        capsize=10,
    )
    ax.bar_label(bars)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(target)

    plt.xlabel("Target")
    plt.ylabel("Execution Time ($\mu$s)")
    plt.title(f"{title}")
    plt.savefig(f"figures/{filename}_performance.png", dpi=400)


def plot_model_performance(data, title, filename):
    target = list(data.keys())
    exec_time = list(data.values())
    x_pos = np.arange(len(target))

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        x_pos,
        exec_time,
        align="center",
        capsize=10,
    )
    ax.bar_label(bars)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(target)

    plt.xlabel("Target")
    plt.ylabel("Execution Time (ms)")
    plt.title(f"{title}")
    plt.savefig(f"figures/{filename}_performance.png", dpi=400)


def plot_model_performance_coreml(data, title, filename):
    target = list(data.keys())
    exec_time = list(data.values())
    x_pos = np.arange(len(target))

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        x_pos,
        exec_time,
        align="center",
        capsize=10,
    )
    ax.bar_label(bars)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(target)

    plt.xlabel("Target")
    plt.ylabel("Execution Time (ms)")
    plt.title(f"{title}")
    plt.savefig(f"figures/{filename}_performance_coreml.png", dpi=400)


plot_op_performance(
    channel_shuffle_times, "Channel Shuffle Operator Performance", "channel_shuffle_op"
)
plot_op_performance(
    group_conv_times, "Group Convolution Operator Performance", "group_convolution_op"
)
plot_model_performance(
    shufflenet_times, "ShuffleNet V2+ Small End-to-End Performance", "shufflenet"
)
plot_model_performance(regnet_times, "RegNet (200M) End-to-End Performance", "regnet")
plot_model_performance_coreml(
    regnet_times_coreml,
    "RegNet (200M) End-to-End Performance (CoreML)",
    "regnet_coreml",
)
plot_model_performance_coreml(
    shufflenet_times_coreml,
    "ShuffleNet V2+ Small End-to-End Performance (CoreML)",
    "shufflenet_coreml",
)

regnet_cpu_breakdown = {
    "Conv2D": 14.565,
    "GroupConv2d": 2.695,
    "Eltwise": 0.510,
    "MatMul": 0.187,
    "Activation": 0.149,
    "Reduce": 0.033,
    "Reshape": 0.002,
}

regnet_gpu_breakdown = {
    "MatMul": 7.044,
    "Conv2D": 3.402,
    "GroupConv2d": 0.461,
    "Eltwise": 0.251,
    "Activation": 0.205,
    "BufferTransform": 0.112,
    "Reduce": 0.031,
    "Reshape": 0.005,
}

shufflenet_cpu_breakdown = {
    "Conv2D": 5.135,
    "Pooling": 1.501,
    "DepthwiseConv2d": 1.083,
    "MatMul": 0.510,
    "Concat": 0.154,
    "ChannelShuffle": 0.128,
    "Slice": 0.108,
    "Reduce": 0.062,
}

shufflenet_gpu_breakdown = {
    "Transpose": 68.445,
    "MatMul": 10.509,
    "Reduce": 5.025,
    "Slice": 3.429,
    "Conv2D": 1.477,
    "BufferTransform": 0.902,
    "Concat": 0.793,
    "DepthwiseConv2d": 0.385,
    "ChannelShuffle": 0.216,
    "Pooling": 0.107,
}


def plot_breakdown(cpu_data, gpu_data, filename, title):
    # Make a dataframe with cpu and gpu data
    df = pd.DataFrame()
    df = df.append(cpu_data, ignore_index=True)
    df = df.append(gpu_data, ignore_index=True)
    # convert NaN to 0
    df = df.fillna(0)
    index = ["CPU", "GPU"]
    df.index = index

    df.plot(
        kind="bar",
        stacked=True,
        title=f"Performance Breakdown of {title}",
        figsize=(7, 5),
    )
    plt.xlabel("Target")
    plt.ylabel("Execution Time (ms)")
    plt.savefig(f"figures/{filename}_breakdown.png", dpi=400)


plot_breakdown(regnet_cpu_breakdown, regnet_gpu_breakdown, "regnet", "RegNet (200M)")
plot_breakdown(
    shufflenet_cpu_breakdown,
    shufflenet_gpu_breakdown,
    "shufflenet",
    "ShuffleNet V2+ Small",
)
