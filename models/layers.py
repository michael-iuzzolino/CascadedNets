"""Resnet block components."""
import torch.nn as nn
from models import custom_ops
from models import tdl
from models.internal_classifiers import InternalClassifier

class HeadLayer(nn.Module):
  """Head layer of ResNet."""

  def __init__(self, layer_i, planes, norm_layer, **kwargs):
    """Initialize head layer."""
    super(HeadLayer, self).__init__()
    
    self.layer_i = layer_i
    self.cascaded = kwargs["cascaded"]
    self.time_bn = kwargs.get("time_bn", kwargs["cascaded"])
    self.num_classes = kwargs.get("num_classes", -1)

    # Set number of input channels
    inplanes = kwargs.get("n_channels", 3)

    if kwargs.get("imagenet", False):
      self.conv1 = nn.Conv2d(
        inplanes,
        planes,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
      )
    else:
      self.conv1 = nn.Conv2d(
        inplanes,
        planes,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
      )

    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    if self.cascaded:
      tdl_mode = kwargs.get("tdl_mode", "OSD")
      self.tdline = tdl.setup_tdl_kernel(tdl_mode, kwargs)

  def set_time(self, t):
    self.t = t
    if t == 0:
      self.tdline.reset()
    
    self.res_active = self.t >= self.layer_i
      
  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out, self.t) if self.time_bn else self.bn1(out)
    out = self.relu(out)
    out = self.maxpool(out)
      
    if self.cascaded:
      # Add delay line
      out = self.tdline(out)

    return out


class BasicBlock(nn.Module):
  """Basic resnet block."""
  expansion = 1

  def __init__(
      self,
      layer_i,
      inplanes,
      planes,
      stride=1,
      downsample=None,
      norm_layer=None,
      **kwargs,
    ):
    """Initialize basic block."""
    super(BasicBlock, self).__init__()
    
    self.layer_i = layer_i
    self.cascaded = kwargs["cascaded"]
    self.cascaded_scheme = kwargs.get("cascaded_scheme", "parallel")
    self.time_bn = kwargs.get("time_bn", kwargs["cascaded"])
    self.downsample = downsample
    self.stride = stride
    self.num_classes = kwargs.get("num_classes", None)
    
    # Setup ops
    self.conv1 = custom_ops.conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = custom_ops.conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)

    # TDL
    if self.cascaded:
      tdl_mode = kwargs.get("tdl_mode", "OSD")
      self.tdline = tdl.setup_tdl_kernel(tdl_mode, kwargs)

  def set_time(self, t):
    self.t = t
    if t == 0:
      self.tdline.reset()
    
    if self.cascaded_scheme == "serial":
      self.res_active = self.t >= self.layer_i
    else:
      self.res_active = True

  def _residual_block(self, x):
    # Conv1
    out = self.conv1(x)
    out = self.bn1(out, self.t) if self.time_bn else self.bn1(out)
    out = self.relu(out)

    # Conv2
    out = self.conv2(out)
    out = self.bn2(out, self.t) if self.time_bn else self.bn2(out)

    return out

  def forward(self, x):
    # Identity
    identity = x
    if self.downsample is not None:
      identity = self.downsample(x)

    # Residual
    residual = self._residual_block(x)

    # TDL if cascaded
    mask = 1.0
    if self.cascaded:
      residual = self.tdline(residual)
    
      if not self.res_active:
        mask = 0.0

    # Identity + Residual
    out = (residual * mask) + identity

    # Nonlinear activation
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  """Bottleneck Block."""
  expansion = 4

  def __init__(
      self,
      layer_i,
      inplanes,
      planes,
      stride=1,
      downsample=None,
      norm_layer=None,
      **kwargs,
    ):
    """Initialize bottleneck block."""
    super(Bottleneck, self).__init__()
    base_width = 64
    width = int(planes * (base_width / 64.))
    
    self.layer_i = layer_i
    self.downsample = downsample
    self.stride = stride
    self.cascaded = kwargs["cascaded"]
    self.cascaded_scheme = kwargs.get("cascaded_scheme", "scheme_2")
    self.time_bn = kwargs.get("time_bn", kwargs["cascaded"])
    self.num_classes = kwargs.get("num_classes", None)
  
    self.conv1 = custom_ops.conv1x1(inplanes, width)
    self.bn1 = norm_layer(width)
    self.conv2 = custom_ops.conv3x3(width, width, stride)
    self.bn2 = norm_layer(width)
    self.conv3 = custom_ops.conv1x1(width, planes * self.expansion)
    self.bn3 = norm_layer(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)

    if self.cascaded:
      tdl_mode = kwargs.get("tdl_mode", "OSD")
      self.tdline = tdl.setup_tdl_kernel(tdl_mode, kwargs)

  def set_time(self, t):
    self.t = t
    if t == 0:
      self.tdline.reset()
    
    if self.cascaded_scheme == "serial":
      self.res_active = self.t >= self.layer_i
    else:
      self.res_active = True

  def _residual_block(self, x):
    # Conv 1
    out = self.conv1(x)
    out = self.bn1(out, self.t) if self.time_bn else self.bn1(out)
    out = self.relu(out)

    # Conv 2
    out = self.conv2(out)
    out = self.bn2(out, self.t) if self.time_bn else self.bn2(out)
    out = self.relu(out)

    # Conv 3
    out = self.conv3(out)
    out = self.bn3(out, self.t) if self.time_bn else self.bn3(out)

    return out

  def forward(self, x):
    # Identity
    identity = x
    if self.downsample is not None:
      identity = self.downsample(x)

    # Residual
    residual = self._residual_block(x)

    # TDL if cascaded
    mask = 1.0
    if self.cascaded:
      residual = self.tdline(residual)
    
      if not self.res_active:
        mask = 0.0

    # Identity + Residual
    out = (residual * mask) + identity

    # Nonlinear activation
    out = self.relu(out)

    return out
