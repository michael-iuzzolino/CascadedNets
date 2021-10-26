"""Main eval script for Cascaded Nets."""
import argparse
import collections
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import torch
import torch.nn as nn
from collections import Counter, defaultdict, OrderedDict
from datasets.dataset_handler import DataHandler
from models import densenet
from models import resnet
from models import sdn_utils
from modules import eval_handler
from modules import losses
from modules import train_handler
from modules import utils
from torch import optim
from torchvision.utils import make_grid


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--random_seed", type=int, default=42,
                      help="random seed")
  
  # Paths
  parser.add_argument("--experiment_root", type=str, 
                      default="experiments",
                      help="Local output dir")
  parser.add_argument("--experiment_name", type=str, 
                      required=True,
                      help="Experiment name")
  
  # Dataset
  parser.add_argument("--dataset_root", type=str, required=True,
                      help="Dataset root")
  parser.add_argument("--dataset_name", type=str, required=True,
                      help="Dataset name: CIFAR10, CIFAR100, TinyImageNet")
  parser.add_argument("--dataset_key", type=str, required=True,
                      help="Dataset to eval: train, test, val")
  parser.add_argument("--split_idxs_root", type=str, default="split_idxs",
                      help="Split idxs root")
  parser.add_argument("--val_split", type=float, default=0.1,
                      help="Validation set split: 0.1 default")
  parser.add_argument("--augmentation_noise_type", type=str, 
                      default="occlusion",
                      help="Augmentation noise type: occlusion")
  parser.add_argument("--batch_size", type=int, default=128,
                      help="batch_size")
  parser.add_argument("--num_workers", type=int, default=2,
                      help="num_workers")
  parser.add_argument("--drop_last", action="store_true", default=False,
                      help="Drop last batch remainder")
  
  # Model
  parser.add_argument("--model_key", type=str, default="resnet18",
                      help="Model: resnet18, resnet34, ..., densenet_cifar")
  parser.add_argument("--train_mode", type=str, 
                      default="baseline",
                      help="Train mode: baseline, ic_only, sdn")
  parser.add_argument("--bn_time_affine", action="store_true", default=False,
                      help="Use temporal affine transforms in BatchNorm")
  parser.add_argument("--bn_time_stats", action="store_true", default=False,
                      help="Use temporal stats in BatchNorm")
  parser.add_argument("--tdl_mode", type=str, 
                      default="OSD",
                      help="TDL mode: OSD, EWS, noise")
  parser.add_argument("--tdl_alpha", type=float, default=0.0,
                      help="TDL alpha for EWS temporal kernel")
  parser.add_argument("--noise_var", type=float, default=0.0,
                      help="Noise variance on noise temporal kernel")
  parser.add_argument("--n_timesteps", type=int, default=0,
                      help="Number of timesteps")
  parser.add_argument("--lambda_val", type=float, default=1.0,
                      help="TD lambda value")
  parser.add_argument("--cascaded", action="store_true", default=False,
                      help="Cascaded net")
  parser.add_argument("--cascaded_scheme", type=str, default="scheme_2",
                      help="cascaded_scheme: scheme_1, scheme_2")
  parser.add_argument("--init_tau", type=float, default=0.01,
                      help="Initial tau valu")
  parser.add_argument("--target_IC_inference_costs", nargs="+", type=float, 
                      default=[0.15, 0.30, 0.45, 0.60, 0.75, 0.90],
                      help="target_IC_inference_costs")
  
  
  # Optimizer
  parser.add_argument("--learning_rate", type=float, default=0.1,
                      help="learning rate")
  parser.add_argument("--momentum", type=float, default=0.9,
                      help="momentum")
  parser.add_argument("--weight_decay", type=float, default=0.0005,
                      help="weight_decay")
  parser.add_argument("--nesterov", action="store_true", default=False,
                      help="Nesterov for SGD")
  parser.add_argument("--normalize_loss", action="store_true", default=False,
                      help="Normalize temporal loss")
  
  # LR scheduler
  parser.add_argument("--lr_milestones", nargs="+", type=float, 
                      default=[60, 120, 150],
                      help="lr_milestones")
  parser.add_argument("--lr_schedule_gamma", type=float, default=0.2,
                      help="lr_schedule_gamma")
  
  # Other
  parser.add_argument("--use_cpu", action="store_true", default=False,
                      help="Use cpu")
  parser.add_argument("--device", type=int, default=0,
                      help="GPU device num")
  parser.add_argument("--n_epochs", type=int, default=150,
                      help="Number of epochs to train")
  parser.add_argument("--eval_freq", type=int, default=10,
                      help="eval_freq")
  parser.add_argument("--save_freq", type=int, default=5,
                      help="save_freq")
  parser.add_argument("--keep_logits", action="store_true", default=False,
                      help="Keep logits")
  parser.add_argument("--keep_embeddings", action="store_true", default=False,
                      help="Keep embeddings")
  parser.add_argument("--debug", action="store_true", default=False,
                      help="Debug mode")
  parser.add_argument("--force_overwrite", action="store_true", default=False,
                      help="Force overwrite")
  
  args = parser.parse_args()
  
  if args.tdl_mode == "OSD":
    args.n_timesteps = None
    
  return args


def plot_training_curves(train_metrics, figs_root, show=False):
  fig, axes = plt.subplots(1,2,figsize=(12,4))
  for dataset_key, dataset_vals in train_metrics.items():
    if dataset_key == "test":
      continue
    for i, (metric_key, metric_vals) in enumerate(dataset_vals.items()):
      epochs = [ele[0] for ele in metric_vals]
      if os.path.basename(os.path.dirname(figs_root)).startswith("std"):
        vals = [np.mean(ele[1]) for ele in metric_vals]
      else:
        try:
          vals = [np.mean(ele[1], axis=0)[-1] for ele in metric_vals]
        except:
          vals = np.array([np.array(ele[1]) for ele in metric_vals])
          vals = np.mean(vals, axis=1)
        
      axes[i].plot(epochs, vals, label=dataset_key)
      axes[i].set_title(metric_key)
      axes[i].set_xlabel("Epochs")
      axes[i].set_ylabel(metric_key)
      axes[i].legend()
  fig_savepath = os.path.join(figs_root, "training_curves.png")
  plt.tight_layout()
  print(f"Saving figure to {fig_savepath}")
  plt.savefig(fig_savepath)
  
  if not show:
    plt.close()
  else:
    plt.show()
    plt.clf()

    
def compute_output_representations(logits, y):
  if len(logits.shape) == 2:
    logits = logits.unsqueeze(dim=0)
  softmax_vals = nn.Softmax(dim=2)(logits)
  preds = softmax_vals.argmax(dim=2)
  corrects = (preds==y).float()

  target_confs = []
  pred_confs = []
  for pred, softmax_val in zip(preds, softmax_vals):
    target_conf = softmax_val[torch.arange(len(softmax_val)), y]
    pred_conf = softmax_val[torch.arange(len(softmax_val)), pred]

    target_confs.append(target_conf)
    pred_confs.append(pred_conf)

  target_confs = torch.stack(target_confs)
  pred_confs = torch.stack(pred_confs)
  
  output_reps = {
      "logits": logits,
      "softmax": softmax_vals,
      "predictions": preds,
      "correct": corrects,
      "target": y,
      "target_confidence": target_confs,
      "prediction_confidence": pred_confs,
  }
    
  for k, v in output_reps.items():
    output_reps[k] = v.cpu()
  return output_reps


def main(args):
  # Make reproducible
  utils.make_reproducible(args.random_seed)

  # Set experiment root
  exp_root = os.path.join(args.experiment_root,
                          args.experiment_name,
                          "experiments")

  # Find exp paths
  exp_paths = glob.glob(f"{exp_root}/*")
  
  if args.train_mode == "baseline":
    train_mode_lookup = "std"
  elif args.train_mode == "cascaded":
    train_mode_lookup = "td("
  elif args.train_mode == "cascaded_seq":
    train_mode_lookup = "std"
  else:
    train_mode_lookup = args.train_mode
  
  exp_paths = [path for path in exp_paths
               if os.path.basename(path).startswith(train_mode_lookup)]
  exp_paths = list(np.sort(exp_paths))
  
  for exp_path in exp_paths:
    print(f"exp_path: {exp_path}")
    if "cascaded_seq" in exp_path:
      continue
      
    if args.tdl_mode != "OSD" and "multiple_fcs" in exp_path:
      continue
      
    load_exp_path = exp_path
    if args.train_mode == "cascaded_seq":
      exp_path = exp_path.replace("std", f"cascaded_seq__{args.cascaded_scheme}")
      if not os.path.exists(exp_path):
        os.makedirs(exp_path)
      
    output_rep_root = os.path.join(exp_path, "outputs")
    if not os.path.exists(output_rep_root):
      os.makedirs(output_rep_root)
    
    rep_basename = f"output_representations__{args.dataset_key}__{args.tdl_mode}.pt"
    output_rep_path = os.path.join(output_rep_root, rep_basename)
    
    embed_basename = f"emeddings__{args.dataset_key}__{args.tdl_mode}.npy"
    output_embeddings_path = os.path.join(output_rep_root, embed_basename)

    if os.path.exists(output_rep_path) and not args.force_overwrite:
      continue

    if os.path.exists(output_embeddings_path) and not args.force_overwrite:
      continue
    
    exp_args_path = os.path.join(load_exp_path, "args.json")
    with open(exp_args_path, "r") as infile:
      loaded_args = argparse.Namespace(**json.load(infile))

    # Data Handler
    data_dict = {
        "dataset_name": loaded_args.dataset_name,
        "data_root": args.dataset_root,
        "val_split": loaded_args.val_split,
        "split_idxs_root": args.split_idxs_root,
        "noise_type": loaded_args.augmentation_noise_type,
        "load_previous_splits": True,
        "verbose": False,
        "imagenet_params": {
          "target_classes": ["terrier"],
          "max_classes": 10,
        }
    }
    data_handler = DataHandler(**data_dict)

    # Set Loaders
    loader = data_handler.build_loader(args.dataset_key, loaded_args)
    print("Data handler loaded.")

    figs_root = os.path.join(exp_path, "figs")
    if not os.path.exists(figs_root):
      os.makedirs(figs_root)

    model_config_path = os.path.join(load_exp_path, "model_config.pt")
    model_dict = torch.load(model_config_path)

    # Load model ckpts
    ckpt_dir = os.path.join(load_exp_path, "ckpts")
    ckpts = np.sort(glob.glob(f"{ckpt_dir}/ckpt__*.pt"))
    try:
      selected_ckpt = ckpts[-1]
      print(f"Loading from ckpt {selected_ckpt}")
    except Exception as e:
      print(f"**Exception: {e}")
      continue
  
    model_dict["pretrained"] = True
    model_dict["pretrained_path"] = selected_ckpt
    
    if args.train_mode == "cascaded_seq":
      model_dict["cascaded"] = True
      model_dict["bn_opts"]["temporal_stats"] = False
      model_dict["cascaded_scheme"] = args.cascaded_scheme
    
    if args.tdl_mode != "OSD":
      model_dict["tdl_mode"] = args.tdl_mode
      model_dict["tdl_alpha"] = args.tdl_alpha
      model_dict["noise_var"] = args.noise_var
      
    # Model init op
    if args.model_key.startswith("resnet"):
      model_init_op = resnet
    elif args.model_key.startswith("densenet"):
      model_init_op = densenet

    # Initialize net
    print("Instantiating model...")
    net = model_init_op.__dict__[args.model_key](**model_dict).to(args.device)
    
    if args.train_mode in ["ic_only", "sdn", "cascaded"]:
      all_flops, normed_flops = sdn_utils.compute_inference_costs(
          data_handler, model_dict, args)
      net.set_target_inference_costs(normed_flops, 
                                     args.target_IC_inference_costs)
      # Set IC costs path and save to exp dir
      IC_costs_savepath = os.path.join(exp_path, "ic_costs.pt")
      torch.save({"flops": all_flops, "normed": normed_flops}, 
                 IC_costs_savepath)

    metrics_path = os.path.join(exp_path, f"metrics.pt")
    try:
      print(f"Loading {metrics_path}")
      train_metrics = torch.load(metrics_path)
      plot_training_curves(train_metrics, figs_root)
    except:
      print(f"Could not plot training curves. Issue with file {metrics_path}")
    try:
      tau_epoch_asymptote = 1 if args.train_mode == "ic_only" else 100
      tau_scheduling_active = args.train_mode != "ic_only"
      tau_handler = sdn_utils.IC_tau_handler(
          init_tau=loaded_args.init_tau,
          tau_targets=loaded_args.target_IC_inference_costs, 
          epoch_asymptote=tau_epoch_asymptote,
          active=tau_scheduling_active)
    except:
      tau_handler = None
    
    if args.n_timesteps is not None:
      n_timesteps = args.n_timesteps
    else:
      n_timesteps = net.timesteps
    
    eval_fxn = eval_handler.get_eval_loop(n_timesteps,
                                          data_handler.num_classes,
                                          cascaded=args.cascaded,
                                          flags=args,
                                          keep_logits=args.keep_logits,
                                          keep_embeddings=args.keep_embeddings,
                                          tau_handler=tau_handler)

    criterion = losses.categorical_cross_entropy

    test_loss, test_acc, logged_data = eval_fxn(net, 
                                                loader, 
                                                criterion, 
                                                0, 
                                                args.device)

    if args.train_mode == "baseline":
      final_mean_test_acc = np.mean(test_acc)
    else:
      final_mean_test_acc = test_acc[-1]
    print(f"Test Acc: {final_mean_test_acc*100:0.2f}%")

    if args.keep_logits:
      logits = logged_data["logits"]
      y = logged_data["y"].to(logits.device)
      output_reps = compute_output_representations(logits, y)

      print(f"Saving output representations to {output_rep_path}")
      torch.save(output_reps, output_rep_path)
    
    if args.keep_embeddings:
      embeddings = logged_data["embeddings"].cpu().detach().numpy()
      embeddings = embeddings.astype(np.float32)
      print(f"Saving embeddings to {output_embeddings_path}")
      np.save(output_embeddings_path, embeddings)

    
if __name__ == "__main__":
  args = setup_args()
  main(args)