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
        self.fc_i2h = nn.Linear(num_inputs, 5 * num_hidden)
        self.fc_h2h = nn.Linear(num_hidden, 5 * num_hidden)
        
        self.input_dim = num_inputs
        self.feat_dim = feat_dim
        self.fc_c2h = nn.Linear(self.feat_dim,num_hidden*5)

    def forward(self, inputs, feats, state):
        
        hx, cx = state

        i2h = self.fc_i2h(inputs)
        h2h = self.fc_h2h(hx)
        c2h = self.fc_c2h(feats)
        x = i2h + h2h 

        gates = x.split(self.num_hidden, 1)
        c_gates = c2h.split(self.num_hidden,1)
        in_gate = F.sigmoid(gates[0] + c_gates[0])
        forget_gate = F.sigmoid(gates[1] + c_gates[1]+self.forget_gate_bias)
        out_gate = F.sigmoid(gates[2] + c_gates[2])
        in_transform = F.tanh(gates[3])
        concept_in_gate = F.sigmoid(gates[4] + c_gates[3])
        cx = forget_gate * cx + in_gate * in_transform #+ concept_in_gate*F.tanh(c_gates[3])
        hx = out_gate * F.tanh(cx) + concept_in_gate*F.tanh(c_gates[4])
        
        return hx, cx
