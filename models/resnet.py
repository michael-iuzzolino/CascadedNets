"""ResNet handler.

  Adapted from
  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

  Two primary changes from original ResNet code:
  1) Tapped delay line op is added to the output of every residual computation
    - See project.models.layers & project.models.tdl
  2) The timestep is set on the TDL in the forward pass
"""
import functools
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.utils import load_state_dict_from_url
from models import custom_ops
from models import layers as res_layers
from models import model_utils
from models.internal_classifiers import InternalClassifier

_MODEL_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


class ResNet(nn.Module):
  """Resnet base class."""

  def __init__(self, name, block, layers, num_classes, **kwargs):
    """Initialize resnet."""
    super(ResNet, self).__init__()
    self.name = name
    self._layers_arch = layers
    self._num_classes = num_classes
    self._train_mode = kwargs.get("train_mode", "baseline")
    self._sdn_train_mode = self._train_mode in ["sdn", "ic_only"]
    self._cascaded = kwargs.get("cascaded", False)
    self._cascaded_scheme = kwargs.get("cascaded_scheme", "parallel")
    
    # Set multiple FCs flag
    self._multiple_fcs = kwargs.get("multiple_fcs", False)
    self._multiple_fcs = not self._sdn_train_mode and self._multiple_fcs
      
    if self._train_mode == "baseline":
      self._time_bn = False
    else:
      self._time_bn = kwargs["bn_opts"]["temporal_stats"]
    
    # Set up batch norm operation
    self._norm_layer_op = self._setup_bn_op(**kwargs)

    # Head layer
    self.res_layer_count = 0
    self.inplanes = 64
    self.layer0 = res_layers.HeadLayer(
      self.res_layer_count, 
      self.inplanes,
      self._norm_layer_op,
      time_bn=self._time_bn,
      IC_active=self._sdn_train_mode,
      num_classes=self._num_classes,
      **kwargs,
    )
    self.res_layer_count += 1

    # Residual Layers
    self.layer1 = self._make_layer(block, 64, layers[0], **kwargs)
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2, 
                                   final_layer=True, **kwargs)
    self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
    
    if self._multiple_fcs:
      fcs = []
      for i in range(self.timesteps):
        fc_i = InternalClassifier(
          n_channels=512, 
          num_classes=num_classes,
          block_expansion=block.expansion,
        )
        fcs.append(fc_i)
      self.fcs = nn.ModuleList(fcs)
    else:
      self.fc = InternalClassifier(
        n_channels=512, 
        num_classes=num_classes,
        block_expansion=block.expansion,
      )
    
    # Weight initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, (self._norm_layer, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _setup_bn_op(self, **kwargs):
    if self._cascaded and self._time_bn:
      self._norm_layer = custom_ops.BatchNorm2d

      # Setup batchnorm opts
      self.bn_opts = kwargs["bn_opts"]
      self.bn_opts["n_timesteps"] = self.timesteps
      norm_layer_op = functools.partial(self._norm_layer, self.bn_opts)
    else:
      self._norm_layer = nn.BatchNorm2d
      norm_layer_op = self._norm_layer

    return norm_layer_op

  def _make_layer(self, block, planes, blocks, 
                  stride=1, final_layer=False, **kwargs):
    tdl_mode = kwargs.get("tdl_mode", "OSD")
    tdl_alpha = kwargs.get("tdl_alpha", 0.0)
    noise_var = kwargs.get("noise_var", 0.0)

    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          custom_ops.conv1x1(self.inplanes, planes * block.expansion, stride),
      )
    layers = []
    layers.append(
        block(
          self.res_layer_count,
          self.inplanes,
          planes,
          stride,
          downsample,
          self._norm_layer_op,
          tdl_alpha=tdl_alpha,
          tdl_mode=tdl_mode,
          noise_var=noise_var,
          cascaded=self._cascaded,
          cascaded_scheme=self._cascaded_scheme,
          time_bn=self._time_bn,
          num_classes=self._num_classes
        )
    )
    self.res_layer_count += 1
    
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
          block(
            self.res_layer_count,
            self.inplanes,
            planes,
            norm_layer=self._norm_layer_op,
            tdl_alpha=tdl_alpha,
            tdl_mode=tdl_mode,
            noise_var=noise_var,
            cascaded=self._cascaded,
            cascaded_scheme=self._cascaded_scheme,
            time_bn=self._time_bn,
            num_classes=self._num_classes
          )
      )
      self.res_layer_count += 1
    return nn.Sequential(*layers)

  @property
  def timesteps(self):
    if self._cascaded:
      n_timesteps = np.sum(self._layers_arch) + 1
    else:
      n_timesteps = 1
    return n_timesteps

  def _set_time(self, t):
    self.layer0.set_time(t)
    for layer in self.layers:
      for block in layer:
        block.set_time(t)
  
  def set_target_inference_costs(
      self, 
      normed_flops, 
      target_inference_costs, 
      use_all=False
    ):
    if use_all:
      print("Using all ICs!")
      selected_ICs = list(range(len(normed_flops)-1))
      IC_costs = normed_flops
    else:
      selected_ICs = []
      IC_costs = []
      for target_cost in target_inference_costs:
        diffs = []
        for normed_flop in normed_flops:
          abs_diff = np.abs(target_cost - normed_flop)
          diffs.append(abs_diff)
        min_idx = np.argmin(diffs)
        IC_cost = normed_flops[min_idx]
        IC_costs.append(IC_cost)
        selected_ICs.append(min_idx)
    self.selected_ICs = np.array(selected_ICs)
    self.IC_costs = np.array(IC_costs)

  def turn_off_IC(self):
    for k, params in self.named_parameters():
      if "IC" in k and "final" not in k:
        params.requires_grad = False
        
  def freeze_backbone(self, verbose=False):
    print("Freezing backbone param...")
    self.frozen_params = []
    self.unfrozen_params = []
    for k, params in self.named_parameters():
      if "IC" not in k:
        self.frozen_params.append(k)
        if verbose:
          print(f"\t{k} [frozen]")
        params.requires_grad = False
      else:
        self.unfrozen_params.append(k)
        
  def _forward(self, x, t=0):
    # Set time on all blocks
    if self._cascaded:
      self._set_time(t)

    # Head layer
    out = self.layer0(x)
    
    # Res Layers
    for layer in self.layers:
      out = layer(out)
      
    # Final layer
    if self._multiple_fcs:
      out = self.fcs[t](out)
    else:
      out = self.fc(out)

    return out
  
  def forward(self, x, t=0):
    return self._forward(x, t)
    
    
def make_resnet(arch, block, layers, pretrained, **kwargs):
  if kwargs.get("imagenet_pretrained", False):
    assert arch in _MODEL_URLS, f"{arch} not found in _MODEL_URLS"
    
    # Save specified num_classes and switch to imagenet # classes
    num_classes = kwargs["num_classes"]
    kwargs["num_classes"] = 1000
    
    # Load model
    model = ResNet(arch, block, layers, **kwargs)
    
    # Load imagenet state dict
    state_dict = load_state_dict_from_url(_MODEL_URLS[arch])
    
    # Adjust names from loaded state_dict to match our model
    new_dict = OrderedDict()
    for k, v in state_dict.items():
      if ".0.downsample.1" in k:
        continue
        
      # Prepend layer0 to head layer to match our code
      if k.startswith("conv1") or k.startswith("bn1"):
        k = f"layer0.{k}"
      
      # Fix fc.fc missing weight
      if k == "fc.weight":
        k = f"fc.{k}"
      if k == "fc.bias":
        k = f"fc.{k}"
      
      # Inflate batch norm along time dimension if cascaded model
      if kwargs["cascaded"] and "running_" in k:
        v = v.unsqueeze(dim=0).repeat(model.timesteps, 1)
      new_dict[k] = v
    
    # Load imagenet state dict into our model
    model.load_state_dict(new_dict)
    print("Success: Loaded pretrained state dict!")

    # Replace final layer to correct # class mapping
    num_ftrs = model.fc.in_features
    model.fc = InternalClassifier(num_ftrs, num_classes)  # nn.Linear(num_ftrs, num_classes)
  else: 
    model = ResNet(arch, block, layers, **kwargs)
    if pretrained:
      model = model_utils.load_model(model, kwargs)
  
  return model


def resnet18(pretrained=False, **kwargs):
  return make_resnet(
    "resnet18", 
    res_layers.BasicBlock, [2, 2, 2, 2],
    pretrained, 
    **kwargs,
  )


def resnet34(pretrained=False, **kwargs):
  return make_resnet(
    "resnet34", 
    res_layers.BasicBlock, 
    [3, 4, 6, 3],
    pretrained, 
    **kwargs,
  )


def resnet50(pretrained=False, **kwargs):
  return make_resnet(
    "resnet50", 
    res_layers.Bottleneck, 
    [3, 4, 6, 3],
    pretrained, 
    **kwargs,
  )


def resnet101(pretrained=False, **kwargs):
  return make_resnet(
    "resnet101", 
    res_layers.Bottleneck, 
    [3, 4, 23, 3],
    pretrained, 
    **kwargs,
  )


def resnet152(pretrained=False, **kwargs):
  return make_resnet(
    "resnet152", 
    res_layers.Bottleneck, 
    [3, 8, 36, 3],
    pretrained, 
    **kwargs,
  )
