"""General utils.

  See GCS api for upload / download functionality:
  https://github.com/googleapis/python-storage/blob/master/google/cloud/storage/blob.py # pylint: disable=line-too-long
"""
import json
import numpy as np
import os
import random
import torch


def make_reproducible(random_seed):
  """Make experiments reproducible."""
  print(f"Making reproducible on seed {random_seed}")
  random.seed(random_seed)
  np.random.seed(random_seed)
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  os.environ["PYTHONHASHSEED"] = str(random_seed)


def save_metrics(metrics, output_dir, debug=False):
  """Save metrics and upload it to GCS."""
  if debug:
    return
  save_path = os.path.join(output_dir, "metrics.pt")
  torch.save(metrics, save_path)


def save_model(model, optimizer, output_dir, epoch_i, debug=False):
  """Save model and upload it to GCS."""
  if debug:
    return

  save_dict = {
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
  }

  ckpt_dir = os.path.join(output_dir, "ckpts")
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  save_path = os.path.join(ckpt_dir, f"ckpt__epoch_{epoch_i:04d}.pt")
  print(f"\nSaving ckpt to {save_path}")
  torch.save(save_dict, save_path)


def save_model_config(model_config, output_dir, debug=False):
  """Save model and upload it to GCS."""
  if debug:
    return
  save_path = os.path.join(output_dir, "model_config.pt")
  torch.save(model_config, save_path)

  
def save_args(args, exp_dir):
  if args.debug:
    return
  args_savepath = os.path.join(exp_dir, "args.json")
  with open(args_savepath, "w") as outfile:
    json.dump(vars(args), outfile, sort_keys=True, indent=4)
  print(f"Saved to {args_savepath}")