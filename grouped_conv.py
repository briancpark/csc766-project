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


def naive_conv2d(input, weight, bias, kernel, stride, padding, groups, x, W, B):
    # naive convolution algorithm in pytorch
    # Weights is W, Bias is B, input is x
    # y is the output to be returned
    
    # compute the output shape
    output_shape = (x.shape[0], W.shape[0], x.shape[2]//stride[0], x.shape[3]//stride[1])
    # initialize the output
    output = torch.zeros(output_shape)

    ## Pad X if needed
    if padding[0] != 0:
        X_pad = torch.zeros((x.shape[0], x.shape[1], x.shape[2]+2*padding[0], x.shape[3]+2*padding[1]))
    else:
        X_pad = x
        
    ## Convolve
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for k in range(output.shape[2]):
                for l in range(output.shape[3]):
                    # get the current slice
                    slice = X_pad[i, :, k*stride[0]:k*stride[0]+kernel[0], l*stride[1]:l*stride[1]+kernel[1]]
                    # get the current weight
                    weight = W[j, :, :, :]
                    # compute the dot product
                    output[i, j, k, l] = torch.sum(slice * weight)
  
  
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

    y = naive_conv2d(
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
    assert y.shape == output


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
