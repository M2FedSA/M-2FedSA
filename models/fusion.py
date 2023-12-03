import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadA(nn.Module):
    def __init__(self, n_heads, input_dim, hid_dim, class_num,args):
        super(MultiHeadA, self).__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = class_num

        self.fc_q = nn.Linear(input_dim, n_heads * hid_dim)
        self.fc_k = nn.Linear(input_dim, n_heads * hid_dim)
        self.fc_v = nn.Linear(input_dim, n_heads * hid_dim)

        self.fc1 = nn.Linear(n_heads * hid_dim, 512, bias=True)
        self.fc2 = nn.Linear(512, self.output_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn = args.if_bn
        self.dropout1 = nn.Dropout(0.3)



    def forward(self, x):
        batch_size, dim = x.size()

        Q = self.fc_q(x.unsqueeze(1))  # (batch_size, 1, n_heads * output_dim)
        K = self.fc_k(x.unsqueeze(1))  # (batch_size, 1, n_heads * output_dim)
        V = self.fc_v(x.unsqueeze(1))  # (batch_size, 1, n_heads * output_dim)

        Q = Q.view(batch_size, self.n_heads, self.hid_dim).transpose(0, 1)  # (n_heads, batch_size, output_dim)
        K = K.view(batch_size, self.n_heads, self.hid_dim).transpose(0, 1)  # (n_heads, batch_size, output_dim)
        V = V.view(batch_size, self.n_heads, self.hid_dim).transpose(0, 1)  # (n_heads, batch_size, output_dim)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1))  # (n_heads, batch_size, batch_size)

        attn_weights = attn_weights / (self.hid_dim ** 0.5)  # Scale the attention weights

        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, V)  # (n_heads, batch_size, output_dim)

        attn_output = attn_output.transpose(0, 1).contiguous()  # (batch_size, n_heads, output_dim)
        attn_output = attn_output.view(batch_size, self.n_heads * self.hid_dim)  # (batch_size, n_heads * output_dim)


        if self.bn:
            output = F.gelu(self.bn1(self.fc1(attn_output)))
        else:
            output = F.gelu(self.fc1(attn_output))
        output = self.dropout1(output)
        fuse_fea = output
        output = self.fc2(output)

        return fuse_fea, output


