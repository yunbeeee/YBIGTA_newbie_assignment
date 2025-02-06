import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        # 구현하세요!
        self.hidden_size = hidden_size
        
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
        

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # 구현하세요!
        combined = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        h_tilde = torch.tanh(self.W_h(torch.cat([x, r * h], dim=1)))
        h_next = (1 - z) * h + z * h_tilde
        return h_next



class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        # 구현하세요!
        batch_size, seq_len, _ = inputs.shape
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        
        for t in range(seq_len):
            h = self.cell(inputs[:, t, :], h)
        
        return h  # 최종 hidden state만 반환