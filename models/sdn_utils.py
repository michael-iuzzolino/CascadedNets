from models import densenet
from models import resnet
import modules.flops_benchmark as flops_benchmark
import numpy as np


class IC_tau_handler:
  def __init__(
      self, 
      init_tau, 
      tau_targets, 
      epoch_asymptote=50, 
      active=True
    ):
    self._init_tau = init_tau
    self._tau_targets = tau_targets
    self._epoch_asymptote = epoch_asymptote
    self._active = active
    self._slopes = (np.array(tau_targets) - init_tau) / epoch_asymptote
    
  def _compute_tau(self, tau_i, epoch_i):
    target_tau = self._tau_targets[tau_i]
    tau_i = self._slopes[tau_i] * epoch_i + self._init_tau
    tau_i = min(tau_i, target_tau)
    return tau_i
  
  def __call__(self, tau_i, epoch_i):
    if self._active:
      tau_val = self._compute_tau(tau_i, epoch_i)
    else:
      tau_val = self._tau_targets[tau_i]
    return tau_val
  
  
def compute_inference_costs(data_handler, model_dict, args, verbose=False):
  # Model init op
  if args.model_key.startswith("resnet"):
    model_init_op = resnet
  elif args.model_key.startswith("densenet"):
    model_init_op = densenet
    
  # Setup sample
  dataset_key = list(data_handler.datasets.keys())[0]
  loader = data_handler.build_loader(dataset_key, args)
  X_sample = next(iter(loader))[0]

  if X_sample.shape[0] != 1:
    X_sample = X_sample[:1]
  
  # Compute max flops
  net = model_init_op.__dict__[args.model_key](**model_dict)
  flops_benchmark.init(net, mode=args.train_mode)
  _ = net(X_sample)
  max_flops = net.compute_total_flops()
  if verbose:
    print(f"max_flops: {max_flops}")
  
  # Compute flops at each IC
  all_flops = []
  n_ICs = range(1, net.timesteps+1)
  for IC_i in n_ICs:
    net = model_init_op.__dict__[args.model_key](**model_dict)
    flops_benchmark.init(net, mode=args.train_mode, limit=IC_i)
    _ = net(X_sample)
    n_flops = net.compute_total_flops()
    perc_flops = n_flops / max_flops * 100
    if verbose:
      print((f"IC {IC_i:2}: # flops: {n_flops:0.4f} GFLOPS -- "
             f"% inference cost: {perc_flops:0.2f}%"))
    all_flops.append(n_flops)
  
  # Normalize flops
  normed_flops = [ele / max_flops for ele in all_flops]
  net.cleanup_flop_hooks()
  return all_flops, normed_flops