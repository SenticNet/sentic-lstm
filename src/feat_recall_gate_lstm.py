import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
def weight_variable(shape):
    
    initial = np.random.uniform(-0.01, 0.01,shape)
    
    initial = torch.from_numpy(initial)
    
    return initial.float()

class LSTMCell(nn.Module):

    def __init__(self, num_inputs, feat_dim,num_hidden, forget_gate_bias=-1):
        super(LSTMCell, self).__init__()
        
        self.forget_gate_bias = forget_gate_bias
        self.num_hidden = num_hidden
        self.fc_i2h = nn.Linear(num_inputs, 4 * num_hidden)
        self.fc_h2h = nn.Linear(num_hidden, 4 * num_hidden)

        self.input_dim = num_inputs
        self.feat_dim = feat_dim
        self.i2c = nn.Linear(self.input_dim,num_hidden)
        self.h2c = nn.Linear(2*self.num_hidden, num_hidden)
        self.c2c = nn.Linear(self.feat_dim, num_hidden)
        self.c2h = nn.Linear(self.feat_dim,num_hidden)
    def forward(self, inputs, feats, state):
        hx, cx = state
        i2h = self.fc_i2h(inputs)
        h2h = self.fc_h2h(hx)
        
        recall_gate = F.sigmoid(self.i2c(inputs) + self.h2c(torch.cat(state,-1)) + self.c2c(feats))
        
        x = i2h + h2h 
        gates = x.split(self.num_hidden, 1)
        
        in_gate = F.sigmoid(gates[0])
        forget_gate = F.sigmoid(gates[1] + self.forget_gate_bias)
        out_gate = F.sigmoid(gates[2])
        in_transform = F.tanh(gates[3])
        
        c_transform = F.tanh(self.c2h(feats))

        cx = forget_gate * cx + in_gate * in_transform + c_transform*recall_gate 
        hx = out_gate * F.tanh(cx)
        return hx, cx
