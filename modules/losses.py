"""Custom losses."""
import torch.nn as nn


def categorical_cross_entropy(pred_logits, y_true_softmax):
  """Categorical cross entropy."""
  log_softmax_pred = nn.LogSoftmax(dim=1)(pred_logits)
  soft_targets = y_true_softmax.detach().clone()  # Stop gradient
  cce_loss = -(soft_targets * log_softmax_pred).sum(dim=1).mean()
  return cce_loss
