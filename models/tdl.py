"""Tapped Delay Line handler."""
import torch
import torch.nn as nn


class OneStepDelayKernel(nn.Module):
  """Single slot queue OSD kernel."""

  def __init__(self, *args, **kwargs):
    """Initialize OSD kernel."""
    super().__init__()
    self.reset()

  def reset(self):
    self.state = None

  def forward(self, current_state):
    if self.state is not None:
      prev_state = self.state
    else:
      prev_state = torch.zeros_like(current_state)
      prev_state.requires_grad = True

    self.state = current_state.clone()

    return prev_state


class ExponentiallyWeightedSmoothingKernel(nn.Module):
  """Exponentially Weighted Smoothing Kernel.

  alpha=0.0
  --> state(t) = current_state
  Functionally equivalent to sequential ResNet

  alpha=1.0
  --> state(t) = prev_state
  Functionally equivalent to tapped delay line for 1 timestep delay

  0.0 < alpha < 1.0
  Continuous interpolation between discrete 1 timestep TDL and sequential ResNet
  """

  def __init__(self, alpha=0.0):
    """Initialize EWS kernel."""
    super().__init__()
    self._alpha = alpha
    self.reset()

  def reset(self):
    self.state = None

  def forward(self, current_state):
    if self.state is not None:
      prev_state = self.state
    else:
      prev_state = torch.zeros_like(current_state)
      prev_state.requires_grad = True

    self.state = self._alpha*prev_state + (1-self._alpha)*current_state.clone()

    return self.state


def setup_tdl_kernel(tdl_mode, kwargs):
  """Temporal kernel interface."""
  if tdl_mode == "OSD":
    tdline = OneStepDelayKernel()
  elif tdl_mode == "EWS":
    tdline = ExponentiallyWeightedSmoothingKernel(kwargs["tdl_alpha"])
  return tdline
