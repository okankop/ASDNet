import torch.nn as nn
from thop import profile
from backbones_video import resnet_2d_v, resnet_3d, resnext, mobilenet, mobilenetv2, shufflenet, shufflenetv2
from backbones_audio import resnet_2d_a, sincdsnet

# %%%%%%%%--------------------- SELECT THE MODEL BELOW ---------------------%%%%%%%%
# model = resnet_2d_v.get_model(layers=[2, 2, 2, 2], rgb_stack_size=8)
# model = resnet_3d.generate_model(model_depth=18)
# model = resnext.resnext101()
# model = mobilenet.get_model(num_classes=2, sample_size=160, width_mult=2.)
# model = mobilenetv2.get_model(num_classes=2, sample_size=160, width_mult=1.)
# model = shufflenet.get_model(groups=3, num_classes=2, width_mult=2.)
# model = shufflenetv2.get_model(num_classes=2, sample_size=160, width_mult=2.)

# model = resnet_2d_a.get_model()
model = sincdsnet.get_model(num_classes=2)
print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable parameters: ", pytorch_total_params)

# flops, prms = profile(model, input_size=(1, 3, 32, 160, 160))
flops, prms = profile(model, input_size=(1, 1, 4740))
print("Total number of FLOPs: ", flops)

