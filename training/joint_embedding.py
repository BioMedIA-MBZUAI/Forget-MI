import torch
from torch import nn

class Outer(nn.Module):
    def __init__(self,
                 inp1_size: int = 128,
                 inp2_size: int = 128,
                 n_neurons: int = 128):
        super(Outer, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear((inp1_size + 1) * (inp2_size + 1), n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
        )

    def forward(self, inp1, inp2):
        batch_size = inp1.size(0)
        append = torch.ones((batch_size, 1), device=inp1.device, dtype=inp1.dtype)
        inp1 = torch.cat([inp1, append], dim=-1)
        inp2 = torch.cat([inp2, append], dim=-1)

        # Compute the batched outer product
        fusion = torch.einsum('bi,bj->bij', inp1, inp2)  # Shape: [batch_size, inp1_size+1, inp2_size+1]
        fusion = fusion.flatten(1)  # Shape: [batch_size, (inp1_size+1) * (inp2_size+1)]
        return self.feedforward(fusion)


class Attention(nn.Module):
    def __init__(self,
                 embed_dim: int = 768,
                 dropout: float = 0.1):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=embed_dim // 64, dropout=dropout)

    def forward(self, inp1, inp2):
        # inp1, inp2 = inp1.unsqueeze(1), inp2.unsqueeze(1)
        inp1, inp2 = inp1.unsqueeze(0), inp2.unsqueeze(0)
                # Compute attention
        attn_output, _ = self.attention(inp1, inp2, inp2)  # Shape: [1, batch_size, embed_dim] #THIS METHOD PUTS MORE WEIGHT TO INPUT 2

        # Remove the sequence length dimension
        return attn_output.squeeze(0)

class Gate(nn.Module):
    def __init__(self, inp1_size, inp2_size):
        super(Gate, self).__init__()

        self.fc1 = nn.Linear(inp1_size + inp2_size, 1)
        self.fc2 = nn.Linear(inp1_size + inp2_size, inp1_size)
        self.beta = nn.Parameter(torch.randn((1,)))
        self.norm = nn.LayerNorm(inp1_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inp1, inp2):

        assert inp1.shape[0] == inp2.shape[0], "Batch sizes must match!"
        assert inp1.shape[1] + inp2.shape[1] == self.fc1.in_features, "Feature sizes must match initialization!"

        w2 = torch.sigmoid(self.fc1(torch.cat([inp1, inp2], -1)))
        # Adjust using both inp1 and w2 * inp2
        adjust = self.fc2(torch.cat([inp1, w2 * inp2], -1))
        
        # Scalar 'one' is created on the same device
        one = torch.tensor(1.0, device=inp1.device).type_as(adjust)
        alpha = torch.min(torch.norm(inp1) / torch.norm(adjust) * self.beta, one)
        output = inp1 + alpha * adjust #THIS METHOD PUTS MORE WEIGHT TO INPUT 1 
        output = self.dropout(self.norm(output))
        return output