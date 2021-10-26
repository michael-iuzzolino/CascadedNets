"""Evaluation function handler for sequential or cascaded."""
import numpy as np
import sys
import torch
import torch.nn.functional as F


class SequentialEvalLoop:
  """Evaluation loop for sequential model."""

  def __init__(self, num_classes, keep_logits=False, keep_embeddings=False, verbose=False):
    self.num_classes = num_classes
    self.keep_logits = keep_logits
    self.keep_embeddings = keep_embeddings
    self.verbose = verbose

  def __call__(self, net, loader, criterion, epoch_i, device):
    net.eval()

    batch_losses = []
    batch_correct = []
    batch_logits = []
    batch_embeddings = []
    ys = []
    global embedding
    sample_count = 0

    # Embedding hook
    def embedding_hook_fn(module, x, output):  # pylint: disable=unused-argument
      global embedding  # pylint: disable=global-variable-undefined
      embedding = x[0]
    _ = net.fc.register_forward_hook(embedding_hook_fn)

    for batch_i, (data, targets) in enumerate(loader):
      if self.verbose:
        sys.stdout.write(f"\rBatch {batch_i+1}/{len(loader)}")
        sys.stdout.flush()
        
      # One-hot-ify targets
      y = torch.eye(self.num_classes)[targets]
      sample_count += y.shape[0]

      # Determine device placement
      data = data.to(device, non_blocking=True)

      # Forward pass
      with torch.no_grad():
        logits = net(data, t=0).cpu()

      if self.keep_logits:
        batch_logits.append(logits)
        ys.append(targets.cpu())
          
      if self.keep_embeddings:
        batch_embeddings.append(embedding.cpu())

      # Compute loss
      loss = criterion(logits, y)
      batch_losses.append(loss.item())

      # Predictions
      softmax = F.softmax(logits, dim=1)
      y_pred = torch.argmax(softmax, dim=1)

      # Updates running statistics
      n_correct = torch.eq(targets, y_pred).sum().item()
      batch_correct.append(n_correct)
    batch_accs = np.sum(batch_correct) / float(sample_count)
    
    logged_data = {}
    if self.keep_logits:
      logged_data["logits"] = torch.cat(batch_logits)
      logged_data["y"] = torch.cat(ys)
          
    if self.keep_embeddings:
      logged_data["embeddings"] = torch.cat(batch_embeddings)
      
    return batch_losses, batch_accs, logged_data
    
    
class CascadedEvalLoop(object):
  """Evaluation loop for cascaded model."""

  def __init__(self, n_timesteps, num_classes, 
               keep_logits=False, keep_embeddings=False, verbose=False):
    self.n_timesteps = n_timesteps
    self.num_classes = num_classes
    self.keep_logits = keep_logits
    self.keep_embeddings = keep_embeddings
    self.verbose = verbose

  def __call__(self, net, loader, criterion, epoch_i, device):
    net.eval()

    batch_logits = []
    batch_embeddings = []
    ys = []
    global embedding
    
    # Embedding hook
    def embedding_hook_fn(module, x, output):  # pylint: disable=unused-argument
      global embedding  # pylint: disable=global-variable-undefined
      embedding = x[0]
      
    if net._multiple_fcs:
      for i, fc in enumerate(net.fcs):
        fc.register_forward_hook(embedding_hook_fn)
    else:
      net.fc.register_forward_hook(embedding_hook_fn)
    
    batch_losses = []
    batch_correct = []
    sample_count = 0
    for batch_i, (x, targets) in enumerate(loader):
      if self.verbose:
        sys.stdout.write(f"\rBatch {batch_i+1}/{len(loader)}")
        sys.stdout.flush()
      # One-hot-ify targets
      y = torch.eye(self.num_classes)[targets]
      sample_count += y.shape[0]

      if self.keep_logits:
        ys.append(targets)

      # Determine device placement
      x = x.to(device, non_blocking=True)

      timestep_correct = []
      timestep_losses = torch.zeros(self.n_timesteps)
      timestep_logits = []
      timestep_embeddings = []
      for t in range(self.n_timesteps):
        # Forward pass
        with torch.no_grad():
          logits_t = net(x, t).cpu()
        
        if self.keep_logits:
          timestep_logits.append(logits_t)
          
        if self.keep_embeddings:
          global embedding
          timestep_embeddings.append(embedding)

        # Compute loss
        loss_i = criterion(logits_t, y)

        # Log loss
        timestep_losses[t] = loss_i.item()

        # Predictions
        softmax_t = F.softmax(logits_t, dim=1)
        y_pred = torch.argmax(softmax_t, dim=1)

        # Updates running accuracy statistics
        n_correct = torch.eq(targets, y_pred).sum()
        timestep_correct.append(n_correct)

      # Update batch loss and compute average
      batch_losses.append(timestep_losses)
      batch_correct.append(torch.stack(timestep_correct))

      if self.keep_logits:
        # stack into shape=(time, batch, n_classes)
        timestep_logits = torch.stack(timestep_logits)
        batch_logits.append(timestep_logits)
      
      if self.keep_embeddings:
        timestep_embeddings = torch.stack(timestep_embeddings)
        batch_embeddings.append(timestep_embeddings)

    # Average over the batches per timestep
    batch_losses = torch.stack(batch_losses).detach().numpy()
    batch_correct = torch.stack(batch_correct).sum(dim=0)
    batch_accs = batch_correct.cpu().detach().numpy() / float(sample_count)

    # Compute loss and accuracy
    logged_data = {}
    if self.keep_logits:
      # concat over batch dim into shape=(time, batch, n_classes)
      batch_logits = torch.cat(batch_logits, dim=1)
      ys = torch.cat(ys)
      logged_data["logits"] = batch_logits
      logged_data["y"] = ys
    
    if self.keep_embeddings:
      # concat over batch dim into shape=(time, batch, n_features, spatial_dim)
      batch_embeddings = torch.cat(batch_embeddings, dim=1)
      logged_data["embeddings"] = batch_embeddings
      
    return batch_losses, batch_accs, logged_data
    

def get_eval_loop(n_timesteps, num_classes, cascaded, flags,
                  keep_logits=False, keep_embeddings=False, 
                  verbose=False, tau_handler=None):
  """Retrieve sequential or cascaded eval function."""
  if flags.train_mode == "baseline":
    eval_fxn = SequentialEvalLoop(num_classes, 
                                  keep_logits, 
                                  keep_embeddings, 
                                  verbose)
  elif flags.train_mode == "cascaded":
    eval_fxn = CascadedEvalLoop(n_timesteps, 
                                num_classes, 
                                keep_logits, 
                                keep_embeddings, 
                                verbose)
  elif flags.train_mode == "cascaded_seq":
    eval_fxn = CascadedEvalLoop(n_timesteps, 
                                num_classes, 
                                keep_logits, 
                                keep_embeddings, 
                                verbose)
  return eval_fxn
