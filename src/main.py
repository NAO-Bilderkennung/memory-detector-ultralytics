import torch.cuda
import wx

from src.gui.mdframe import MDFrame

model_types = ["n", "s", "m", "l", "x"]

models = [
    [f"yolov8{model_type}" for model_type in model_types],
    [f"yolov5{model_type}" for model_type in model_types],
    [f"yolov3{model_type}" for model_type in ["", "-tiny", "-spp"]]
]

# Flatten list (https://stackoverflow.com/questions/10126983/flatten-a-list-in-python)
models = [model for model_list in models for model in model_list]
default_model_type = "yolov8x"

cuda_devices = ["cpu"]
cuda_device_names = ["CPU"]
default_cuda_device = "cpu"

if torch.cuda.is_available():
    cuda_device_count = torch.cuda.device_count()

    cuda_devices.extend([i for i in range(cuda_device_count)])
    cuda_device_names.extend([torch.cuda.get_device_name(i) for i in range(cuda_device_count)])
    default_cuda_device = 0

if torch.backends.mps.is_available():
    cuda_devices.append("mps")
    cuda_device_names.append("Metal Performance Shaders")
    default_cuda_device = "mps"

if __name__ == "__main__":
    app = wx.App()
    frame = MDFrame(models, models.index(default_model_type), cuda_devices, cuda_device_names,
                    cuda_devices.index(default_cuda_device))
    app.MainLoop()
