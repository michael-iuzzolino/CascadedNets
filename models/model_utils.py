"""Model utils."""
import torch
from collections import OrderedDict


def apply_weight_decay(net, weight_decay):
  """Apply weight decay."""
  if weight_decay == 0:
    return
  for _, param in net.named_parameters():
    if param.grad is None:
      continue
    param.grad = param.grad.add(param, alpha=weight_decay)


def load_model(net, kwargs):
  """Load pretrained model."""
  pretrained_path = kwargs.get("pretrained_path", False)
  assert pretrained_path, "Could not find pretrained_path!"
  # print(f"Loading model from {pretrained_path}")
  state_dict = torch.load(pretrained_path)["model"]
  if kwargs.get("train_mode", None) == "cascaded_seq":
    fixed_dict = OrderedDict()
    for k, v in state_dict.items():
      if "running_" in k and len(v.size()):
        continue
      fixed_dict[k] = v
    net.load_state_dict(fixed_dict, strict=False)
  else:
    try:
      net.load_state_dict(state_dict)
    except:
      # Fix dictionary
      fixed_dict = OrderedDict()
      for k, v in state_dict.items():
        if k == "fc.weight":
          fixed_dict["fc.fc.weight"] = v
        elif k == "fc.bias":
          fixed_dict["fc.fc.bias"] = v
        else:
          fixed_dict[k] = v
      net.load_state_dict(fixed_dict)
        
  return net
