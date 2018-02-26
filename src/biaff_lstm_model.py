import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from utils import *
from biatt_lstm_model import BiAttLSTM_Model
from utils import _lengths_to_masks,_multi_bilayer_attention
from feat_bilstm_model import FeatBiLSTM_Model


class BiAffLSTM_Model(BiAttLSTM_Model):

    def __init__(self,num_classes,max_length,num_tokens,embd,emb_dim = 300,hidden_dim=100,concept_vector=None,num_ways =3,lr=0.001,cell='recall'):
        
        super(BiAffLSTM_Model,self).__init__(num_classes,max_length,num_tokens,embd,emb_dim=emb_dim,hidden_dim=hidden_dim,num_ways =num_ways,lr=lr)
        
        self.concept_dim = concept_vector.size()[1]
        
        self.lstm = FeatBiLSTM_Model(max_length,num_tokens,embd,self.concept_dim,emb_dim,hidden_dim,concept_vector=concept_vector,cell=cell)

        self.concept_linear =  nn.Linear(self.concept_dim,self.hidden_dim*2)
  
        self.linear = nn.Linear(self.hidden_dim*2, self.num_ways)
        
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
    
    

    
    def train_(self,x,y,targets,lengths,concepts=None,concept_lengths=None):
        
        
        self.zero_grad()
        
        self.train()
        
        lstm_outputs = self.lstm.forward(x,lengths,concepts,concept_lengths)
                
        output,output_ = self.global_attention_forward(lstm_outputs,targets,lengths) 
        
        y = y.view(-1)
        
        loss = self.loss_fn(output_,y)
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss
    
    
    def test(self,x,targets,lengths,concepts=None,concept_lengths=None):
        
        self.eval()
        
        lstm_outputs = self.lstm.forward(x,lengths,concepts,concept_lengths)
        
        output,output_ = self.global_attention_forward(lstm_outputs,targets,lengths)
                   
        
        return output.view(-1,self.num_classes,self.num_ways).data.numpy()#,c_att,global_att