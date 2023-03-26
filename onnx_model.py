import sys, os
import torch

sys.path.append(os.path.expanduser("model_source/shufflenet/ShuffleNetV2+"))
from network import ShuffleNetV2_Plus
from model_source.regnet.regnet import regnetx_002


def convert_model(model_name):
    if model_name == "shufflenet":
        architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        model = ShuffleNetV2_Plus(architecture=architecture, model_size="Small")
    elif model_name == "regnet":
        model = regnetx_002()
    else:
        TypeError("Model not supported")

    # ImageNet Tensor size is 3x224x224, recall that PyTorch expects NCHW
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model.eval()
    out = model(x)

    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        f"onnx_models/{model_name}.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    convert_model("shufflenet")
    convert_model("regnet")
