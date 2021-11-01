import torch
import torch.nn as nn


class RNN(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super().__init__()
    self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, 1)

  def forward(self, x):
    # x = [timesteps, batch size, logits]
    output, hidden = self.rnn(x)
    # output = [sent len, batch size, hid dim]
    # hidden = [1, batch size, hid dim]
    
    outputs = []
    for o_t in output.permute(1,0,2):
      out_t = torch.sigmoid(self.fc(o_t).squeeze())
      outputs.append(out_t)
    return outputs