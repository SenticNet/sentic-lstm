import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from utils import *
from att_lstm_model import AttLSTM_Model
from utils import _lengths_to_masks,_multi_bilayer_attention



class BiAttLSTM_Model(AttLSTM_Model):

    def __init__(self,num_classes,max_length,num_tokens,embd,emb_dim = 300,hidden_dim=100,num_ways =3,lr=0.001):
        
        super(BiAttLSTM_Model,self).__init__(num_classes,max_length,num_tokens,embd,emb_dim=emb_dim,hidden_dim=hidden_dim,num_ways =num_ways,lr=lr)
        
        self.linear = nn.Linear(self.hidden_dim*2, self.num_ways)

        self.global_linear_att_l1 = nn.Linear(self.hidden_dim*4,self.att_dim)
        
        self.global_linear_att_l2 = nn.Linear(self.att_dim,1)
        
        self.global_multi_linear_att_l2 = nn.Linear(self.att_dim,self.num_att,bias=False)
        
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        
    
    def global_attention_forward(self, lstm_outputs,targets,lengths):
                        
        max_length = lstm_outputs.size()[1]
        
        batch_size = lstm_outputs.size()[0]
                
        mask =  _lengths_to_masks(lengths,max_length)
                
        targets = self.target_attention(lstm_outputs,targets)
        
        target_outputs  = (targets.unsqueeze(2).expand_as(lstm_outputs) * lstm_outputs).sum(1)
        
        target_outputs =  target_outputs.squeeze(1)
        
        target_outputs = target_outputs/targets.sum(1).expand_as(target_outputs)
                        
        global_att = _multi_bilayer_attention(lstm_outputs,target_outputs,mask,\
                                                           self.global_linear_att_l1,self.global_multi_linear_att_l2,self.tanh,num_att=self.num_att)
        
        norm = global_att.sum(1)
        
        global_att = global_att / norm.expand_as(global_att)
        
        global_outputs = lstm_outputs.unsqueeze(2).expand(batch_size,lstm_outputs.size()[1],self.num_att,lstm_outputs.size()[2])

        global_outputs  = (global_att.unsqueeze(3).expand_as(global_outputs) * global_outputs).sum(1)
        
        global_outputs =  global_outputs.squeeze(1)
                                                        
        output_ = self.linear(global_outputs.view(batch_size * self.num_att,-1)).view(batch_size,-1,self.num_ways)
                        
        output = self.dropout(output_)
        
        output = self.softmax(output.view(-1,self.num_ways))
        
        return output,output_.view(-1,self.num_ways) 
    
    
    
    
    
    def train_(self,x,y,targets,lengths,concepts=None,concept_lengths=None):
        
        self.zero_grad()
        
        self.train()
        
        lstm_outputs = self.lstm.forward(x,lengths)

        output,output_ = self.global_attention_forward(lstm_outputs,targets,lengths) 
            
        y = y.view(-1)
        
        loss = self.loss_fn(output_,y)
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss
    
    
    def test(self,x,targets,lengths):
        
        self.eval()
        
        lstm_outputs = self.lstm.forward(x,lengths)
            
        output,output_ = self.global_attention_forward(lstm_outputs,targets,lengths) 
    
        return output.view(-1,self.num_classes,self.num_ways).data.numpy()