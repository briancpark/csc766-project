### A simple example of grouped convolution in NumPy and PyTorch

import torch
from torch import nn
import numpy as np


# Format of ONNX has been NCHW


def torch_conv2d(input, output, weight, bias, kernel, stride, padding, groups, x=None):
    conv = nn.Conv2d(
        input[1],
        output[1],
        kernel,
        stride=stride,
        groups=groups,
        padding=padding,
        dtype=torch.float32,
    )
    if x is None:
        x = torch.randn(input)
    y = conv(x)

    assert y.shape == output
    return y


def naive_conv2d_nhwc(input, weight, bias, kernel, stride, padding, groups, x, Weight, Bias):
    # naive convolution algorithm in pytorch
    # Weights is W, Bias is B, input is x
    # y is the output to be returned

    # input: input tensor with shape (N, C, H, W)
    # filters: filter tensor with shape (O, I, K_h, K_w)

    stride_h = stride[0]
    stride_w = stride[1]
    pad_h = padding[0]
    pad_w = padding[1]

    num_groups = groups
    N, C, H, W = input
    O, I_group, K_h, K_w = weight
    I = I_group * num_groups

    N, C, H, W = x.shape
    O, I_group, K_h, K_w = Weight.shape
    assert (
        C % num_groups == 0
    ), f"Input channels ({C}) must be divisible by number of groups ({num_groups})"
    assert (
        I_group * num_groups == C
    ), f"Input channels ({C}) must equal num_groups ({num_groups}) times I_group ({I_group})"
    assert (
        O % num_groups == 0
    ), f"Output channels ({O}) must be divisible by number of groups ({num_groups})"

    # Compute output dimensions
    out_h = (H - K_h + 2 * pad_h) // stride_h + 1
    out_w = (W - K_w + 2 * pad_w) // stride_w + 1

    # Allocate output array
    output = torch.zeros((N, O, out_h, out_w)) + Bias.view(1, O, 1, 1)

    # Pad input
    padded_input = torch.nn.functional.pad(
        x, (pad_w, pad_w, pad_h, pad_h), mode="constant"
    )

    # Perform grouped convolution
    for n in range(N):
        for g in range(num_groups):
            # Divide input and filters into groups
            i_group = padded_input[n, g * I_group : (g + 1) * I_group, :, :]
            k_group = Weight[g * O // num_groups : (g + 1) * O // num_groups, :, :, :]

            # Perform convolution on group
            for o in range(g * O // num_groups, (g + 1) * O // num_groups):
                for h in range(out_h):
                    for w in range(out_w):
                        for i in range(I_group):
                            # Compute dot product of kernel and image patch
                            patch_h_start = h * stride_h
                            patch_w_start = w * stride_w
                            patch = i_group[
                                i,
                                patch_h_start : patch_h_start + K_h,
                                patch_w_start : patch_w_start + K_w,
                            ]
                            output[n, o, h, w] += torch.sum(
                                patch * k_group[o - g * O // num_groups, i, :, :]
                            )
    return output


def check_naive_conv2d(input, output, weight, bias, kernel, stride, padding, groups):
    ### Implement naive convolution
    conv = nn.Conv2d(
        input[1],
        output[1],
        kernel,
        stride=stride,
        groups=groups,
        padding=padding,
        dtype=torch.float32,
    )
    x = torch.randn(input)
    with torch.no_grad():
        y_conv = conv(x)

    with torch.no_grad():
        y = naive_conv2d_nhwc(
            input,
            weight,
            bias,
            kernel,
            stride,
            padding,
            groups,
            x,
            conv.weight,
            conv.bias,
        )

    ### Compare with PyTorch
    print("Expected: ", input, output)
    print("PyTorch : ", x.shape, y_conv.shape)
    print("Mine    : ", x.shape, y.shape)

    assert torch.allclose(y_conv, y, atol=1e-5)


if __name__ == "__main__":
    with torch.no_grad():
        config0 = (
            (1, 24, 112, 112),
            (1, 24, 56, 56),
            (24, 8, 3, 3),
            24,
            (3, 3),
            (2, 2),
            (1, 1),
            3,
        )
        config1 = (
            (1, 56, 56, 56),
            (1, 56, 28, 28),
            (56, 8, 3, 3),
            56,
            (3, 3),
            (2, 2),
            (1, 1),
            7,
        )
        config2 = (
            (1, 152, 28, 28),
            (1, 152, 14, 14),
            (152, 8, 3, 3),
            152,
            (3, 3),
            (2, 2),
            (1, 1),
            19,
        )
        config3 = (
            (1, 152, 14, 14),
            (1, 152, 14, 14),
            (152, 8, 3, 3),
            152,
            (3, 3),
            (1, 1),
            (1, 1),
            19,
        )
        config4 = (
            (1, 368, 14, 14),
            (1, 368, 7, 7),
            (368, 8, 3, 3),
            368,
            (3, 3),
            (2, 2),
            (1, 1),
            46,
        )
        config5 = (
            (1, 368, 7, 7),
            (1, 368, 7, 7),
            (368, 8, 3, 3),
            368,
            (3, 3),
            (1, 1),
            (1, 1),
            46,
        )

        torch_conv2d(*config0)
        torch_conv2d(*config1)
        torch_conv2d(*config2)
        torch_conv2d(*config3)
        torch_conv2d(*config4)

        check_naive_conv2d(*config0)
        check_naive_conv2d(*config1)
        check_naive_conv2d(*config2)
        check_naive_conv2d(*config3)
        check_naive_conv2d(*config4)
