import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn


def train_val_split(Xs, val_split):
  print(f"Random train-val splitting. Val split: {val_split}")
  n_val = int(Xs.shape[0] * val_split)
  rand_idxs = np.array(range(Xs.shape[0]))
  np.random.shuffle(rand_idxs)
  val_idxs = rand_idxs[:n_val]
  train_idxs = rand_idxs[n_val:]
  return train_idxs, val_idxs


def train_val_split_balanced(Xs, ys, val_split):
  print(f"Balanced train-val splitting. Val split: {val_split}")
  pos_y = ys[ys==1]
  neg_y = ys[ys==0]

  n_pos_y = len(pos_y)
  n_neg_y = len(neg_y)

  pos_idxs = list(range(n_pos_y))
  neg_idxs = list(range(n_neg_y))
  np.random.shuffle(pos_idxs)
  np.random.shuffle(neg_idxs)

  n_pos_val = int(len(pos_idxs) * val_split)
  n_neg_val = int(len(neg_idxs) * val_split)

  pos_val_idxs = pos_idxs[:n_pos_val]
  pos_train_idxs = pos_idxs[n_pos_val:]
  neg_val_idxs = neg_idxs[:n_neg_val]
  neg_train_idxs = neg_idxs[n_neg_val:]

  val_idxs = pos_val_idxs + neg_val_idxs
  train_idxs = pos_train_idxs + neg_train_idxs
  
  return train_idxs, val_idxs


def init_new_network(net, n_hidden, ckpt_init_dir, force_new_state_dict=True):
  print("Initializing new network...")
  # Generate / Load state dict
  state_dict_path = os.path.join(ckpt_init_dir, f'state_dict__hidden_{n_hidden}.pt')
  if not os.path.exists(state_dict_path) or force_new_state_dict:
    print("Generating new weight initializations...")
    print(f"Saving to {state_dict_path}")
    torch.save(net.state_dict(), state_dict_path)
  else:
    print(f"Loading weight initializations from {state_dict_path}...")
    state_dict = torch.load(state_dict_path)
    net.load_state_dict(state_dict)
    

def setup_network(input_dim, ckpt_init_dir, model, n_hidden,
                  pretrained_path=None, gpu_index=0, force_new_state_dict=False):
  net = model(input_dim=input_dim, hidden_dim=n_hidden)
  successful_load = False
  if pretrained_path is not None:
    print(f"Loading pretrained model from {pretrained_path}")
    try:
      state_dict = torch.load(pretrained_path)
      net.load_state_dict(state_dict)
      successful_load = True
    except:
      print(f"**ERROR: Could not find weights at {pretrained_path}!")
      init_new_network(net, n_hidden, ckpt_init_dir, force_new_state_dict)
  else:
    init_new_network(net, n_hidden, ckpt_init_dir, force_new_state_dict)
  return successful_load, net.to(gpu_index)


def compute_AUC(time_arr, correct_arr, n_timesteps, add_final=False):
  A = 0
  
  for i in range(len(time_arr)-1):
    ti = time_arr[i]
    tj = time_arr[i+1]
    ci = correct_arr[i]
    cj = correct_arr[i+1]

    dx = tj - ti
    dy = cj - ci
    A1 = dx * ci
    A2 = 0.5 * dx * dy
    A += (A1 + A2)
  
  if add_final:
    A += (n_timesteps-tj)*cj
  
  A /= n_timesteps
  return A
