import torch

def pytorch_conv2d(input, weight, bias, padding, stride, kernel, groups):
    # perform grouped convolution using pytorch
    output = torch.nn.functional.conv2d(
        input, weight, bias, stride=stride, padding=padding, groups=groups
    )
    return output


def naive_grouped_conv2d_nchw(input, weight, bias, padding, stride, kernel, groups):
    # input is of shape (batch_size, in_channels, in_height, in_width)
    # output is of shape (batch_size, out_channels, out_height, out_width)
    # bias is of shape (out_channels,)
    # padding is of shape (padding_height, padding_width)
    # stride is of shape (stride_height, stride_width)
    # kernel is of shape (kernel_height, kernel_width)
    # groups is an integer
    
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    padding_height, padding_width = padding
    stride_height, stride_width = stride

    # compute output height and width
    out_height = (in_height + 2 * padding_height - kernel_height) // stride_height + 1
    out_width = (in_width + 2 * padding_width - kernel_width) // stride_width + 1
    
    # split input and weight into groups
    input_groups = torch.split(input, in_channels // groups, dim=1)
    weight_groups = torch.split(weight, out_channels // groups, dim=0)

    # perform grouped convolution
    output = torch.zeros(batch_size, out_channels, out_height, out_width)
    for i in range(groups):
        input_group = input_groups[i]
        weight_group = weight_groups[i]
        bias_group = bias[out_channels // groups * i: out_channels // groups * (i + 1)]

        output_group = torch.nn.functional.conv2d(
            input_group, weight_group, bias_group, stride=stride, padding=padding
        )

        output[:, out_channels // groups * i: out_channels // groups * (i + 1), :, :] = output_group

    return output


def naive_grouped_conv2d_scalar_nchw(input, weight, bias, padding, stride, kernel, groups):
    # input is of shape (batch_size, in_channels, in_height, in_width)
    # output is of shape (batch_size, out_channels, out_height, out_width)
    # filter is of shape (out_channels, in_channels, kernel_height, kernel_width)
    # bias is of shape (out_channels,)
    # padding is of shape (padding_height, padding_width)
    # stride is of shape (stride_height, stride_width)
    # kernel is of shape (kernel_height, kernel_width)
    # groups is an integer
    
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    padding_height, padding_width = padding
    stride_height, stride_width = stride

    # compute output height and width
    out_height = (in_height + 2 * padding_height - kernel_height) // stride_height + 1
    out_width = (in_width + 2 * padding_width - kernel_width) // stride_width + 1
    
    # split input and weight into groups
    input_groups = torch.split(input, in_channels // groups, dim=1)
    weight_groups = torch.split(weight, out_channels // groups, dim=0)

    # perform grouped convolution
    output = torch.zeros(batch_size, out_channels, out_height, out_width)
    for i in range(groups):
        input_group = input_groups[i]
        weight_group = weight_groups[i]
        bias_group = bias[out_channels // groups * i: out_channels // groups * (i + 1)]

        for b in range(batch_size):
            for c_out in range(out_channels // groups):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_in = h_out * stride_height - padding_height
                        w_in = w_out * stride_width - padding_width
                        dot_product = 0
                        for c_in in range(in_channels // groups):
                            for h_k in range(kernel_height):
                                for w_k in range(kernel_width):
                                    h = h_in + h_k
                                    w = w_in + w_k
                                    if h >= 0 and h < in_height and w >= 0 and w < in_width:
                                        dot_product += input_group[b, c_in, h, w] * weight_group[c_out, c_in, h_k, w_k]
                        output[b, out_channels // groups * i + c_out, h_out, w_out] += dot_product + bias_group[c_out]

    return output

def naive_grouped_conv2d_nhwc(input, weight, bias, padding, stride, kernel, groups):
    # input is of shape (batch_size, in_height, in_width, in_channels)
    # output is of shape (batch_size, out_height, out_width, out_channels)
    # weight is of shape (out_channels, in_channels, kernel_height, kernel_width)
    # bias is of shape (out_channels,)
    # padding is of shape (padding_height, padding_width)
    # stride is of shape (stride_height, stride_width)
    # kernel is of shape (kernel_height, kernel_width)
    # groups is an integer
    
    batch_size, in_height, in_width, in_channels = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    padding_height, padding_width = padding
    stride_height, stride_width = stride

    # compute output height and width
    out_height = (in_height + 2 * padding_height - kernel_height) // stride_height + 1
    out_width = (in_width + 2 * padding_width - kernel_width) // stride_width + 1
    
    # split input and weight into groups
    input_groups = torch.split(input, in_channels // groups, dim=3)
    weight_groups = torch.split(weight, out_channels // groups, dim=0)

    # perform grouped convolution
    output = torch.zeros(batch_size, out_height, out_width, out_channels)
    for i in range(groups):
        input_group = input_groups[i]
        weight_group = weight_groups[i]
        bias_group = bias[out_channels // groups * i: out_channels // groups * (i + 1)]

        for b in range(batch_size):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    dot_product = 0
                    for c_in in range(in_channels // groups):
                        for h_k in range(kernel_height):
                            for w_k in range(kernel_width):
                                h_in = h_out * stride_height - padding_height + h_k
                                w_in = w_out * stride_width - padding_width + w_k
                                if h_in >= 0 and h_in < in_height and w_in >= 0 and w_in < in_width:
                                    dot_product += input_group[b, h_in, w_in, c_in] * weight_group[out_channels // groups * i: out_channels // groups * (i + 1), c_in, h_k, w_k]

                    output[b, h_out, w_out, out_channels // groups * i: out_channels // groups * (i + 1)] += dot_product + bias_group

    return output

def test_correctness(
    input_shape, weight_shape, bias_shape, padding, stride, kernel, groups
):
    input = torch.randn(input_shape)
    weight = torch.randn(weight_shape)
    bias = torch.randn(bias_shape)
    
    
    input = torch.abs(input) * 1000
    weight = torch.abs(weight) * 1000
    bias = torch.abs(bias) * 1000

    pytorch_output = pytorch_conv2d(
        input, weight, bias, padding, stride, kernel, groups
    )
    naive_output_nchw = naive_grouped_conv2d_nchw(
        input, weight, bias, padding, stride, kernel, groups
    )

    assert torch.allclose(pytorch_output, naive_output_nchw)
    
    
    # convert input to NHWC
    input = input.permute(0, 2, 3, 1)
    
    naive_output_nhwc = naive_grouped_conv2d_nhwc(
        input, weight, bias, padding, stride, kernel, groups
    )
    
    # transpose output back to NCHW
    naive_output_nhwc = naive_output_nhwc.permute(0, 3, 1, 2)
    
    assert torch.allclose(pytorch_output, naive_output_nhwc)
    
    
    print("Correctness test passed!")


if __name__ == "__main__":
    input_shapes = [
        (1, 24, 112, 112),
        (1, 56, 56, 56),
        (1, 152, 28, 28),
        (1, 152, 14, 14),
        (1, 152, 14, 14),
        (1, 152, 14, 14),
        (1, 368, 14, 14),
        (1, 368, 7, 7),
        (1, 368, 7, 7),
        (1, 368, 7, 7),
        (1, 368, 7, 7),
        (1, 368, 7, 7),
        (1, 368, 7, 7),
    ]

    weight_shapes = [
        (24, 8, 3, 3),
        (56, 8, 3, 3),
        (152, 8, 3, 3),
        (152, 8, 3, 3),
        (152, 8, 3, 3),
        (152, 8, 3, 3),
        (368, 8, 3, 3),
        (368, 8, 3, 3),
        (368, 8, 3, 3),
        (368, 8, 3, 3),
        (368, 8, 3, 3),
        (368, 8, 3, 3),
        (368, 8, 3, 3),
    ]

    bias_shapes = [
        24,
        56,
        152,
        152,
        152,
        152,
        368,
        368,
        368,
        368,
        368,
        368,
        368,
    ]

    strides = [
        (2, 2),
        (2, 2),
        (2, 2),
        (1, 1),
        (1, 1),
        (1, 1),
        (2, 2),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ]

    padding = [
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ]

    kernels = [
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
    ]

    groups = [
        3,
        7,
        19,
        19,
        19,
        19,
        46,
        46,
        46,
        46,
        46,
        46,
        46,
    ]

    for input_shape, weight_shape, bias_shape, stride, padding, kernel, group in zip(
        input_shapes, weight_shapes, bias_shapes, strides, padding, kernels, groups
    ):
        test_correctness(
            input_shape, weight_shape, bias_shape, padding, stride, kernel, group
        )
