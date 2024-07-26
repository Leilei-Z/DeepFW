import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, device):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_weights, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

class StatusUpdate(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StatusUpdate, self).__init__()
        self.linear1 = nn.Linear(input_dim * 2, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.state = torch.zeros(32, input_dim)  # 初始化状态

    def forward(self, conv_outputs, attention_outputs, attention_weights):
        combined_input = torch.cat((conv_outputs, attention_outputs), dim=-1)
        A = self.linear1(combined_input)
        batch_size = conv_outputs.size(0)
        new_state = torch.zeros(batch_size, conv_outputs.size(1), device=conv_outputs.device)
        for i in range(attention_weights.size(1)):
            state_part = new_state
            weight_part = attention_weights[:, i, :]
            B = self.linear2(state_part * weight_part)
            new_state = A + B
        self.state = new_state
        return self.state

class FFANModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(FFANModel, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
            for kernel_size in [3, 5, 7]
        ])
        self.self_attention = SelfAttention(hidden_dim * 3, device)
        self.state_space = StatusUpdate(hidden_dim * 3, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = F.relu(conv_layer(x))
            pooled_output = F.adaptive_max_pool1d(conv_output, 1).squeeze(2)
            conv_outputs.append(pooled_output)
        conv_outputs = torch.cat(conv_outputs, dim=1)
        attention_output, attention_weights = self.self_attention(conv_outputs.unsqueeze(1))
        attention_output = attention_output.squeeze(1)
        state = self.state_space(conv_outputs, attention_output, attention_weights)
        return state