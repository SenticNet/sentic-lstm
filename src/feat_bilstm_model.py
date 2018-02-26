import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from utils import _lengths_to_masks
from feat_recall_gate_lstm import LSTMCell as Recall_cell
from feat_corr_lstm import LSTMCell as Sentic_cell


class FeatBiLSTM_Model(nn.Module):

    def __init__(self,max_length,num_tokens,embd,concept_dim,emb_dim = 300,hidden_dim=100,concept_vector=None,cell='recall'):
        
        super(FeatBiLSTM_Model,self).__init__()
    
        self.emb_dim = emb_dim
                
        self.hidden_dim = hidden_dim 
        
        self.max_length = max_length
        
        self.concept_dim = concept_dim
        
        self.embedding = nn.Embedding(num_tokens,emb_dim)
        
        self.embedding.weight = nn.Parameter(torch.from_numpy(embd),requires_grad=False)
        
        self.concept_embedding = nn.Embedding(concept_vector.size()[0],self.concept_dim)

        self.concept_embedding.weight = nn.Parameter(concept_vector,requires_grad=False)

        if cell == 'recall':
            
            print "using recall cell"
            
            self.lstm_fw = Recall_cell(self.emb_dim,self.concept_dim ,self.hidden_dim )

            self.lstm_bw = Recall_cell(self.emb_dim,self.concept_dim ,self.hidden_dim )
        else:
            
            print "using sentic cell"
            
            self.lstm_fw = Sentic_cell(self.emb_dim,self.concept_dim ,self.hidden_dim )

            self.lstm_bw = Sentic_cell(self.emb_dim,self.concept_dim ,self.hidden_dim )

        self.loss_fn = nn.functional.cross_entropy
        
        self.softmax = nn.Softmax()
        
        self.sigmoid = nn.Sigmoid()
        
        self.err = 1e-24
        
        self.tanh = nn.Tanh()
        
        self.dropout = nn.Dropout(0.1)
        
    def concept_avg(self,c,c_mask):
        
        max_length = c.size()[1]
        
        concept_outputs = []
        
        batch_size = c.size()[0]
        
        for i in range(max_length):
            
            c_att = c_mask[:,i] 
            
            norm = c_att.sum(1) + self.err
            
            c_att = c_att / norm.expand_as(c_att)
            
            c_vec = c[:,i]
            c_  = (c_att.unsqueeze(2).expand_as(c_vec) * c_vec).sum(1)
                    
            concept_outputs.append(c_)
            
        res = torch.cat(concept_outputs,1)
            
        return res           
    
         
    def init_hidden(self,batch_size):
        
        return (autograd.Variable(torch.zeros(batch_size, self.hidden_dim)),autograd.Variable(torch.zeros(batch_size, self.hidden_dim)))
    
    def forward(self,x,lengths,concepts,concept_lengths):
        
        max_length = x.size()[1]
                
        max_concept_length = concepts.size(2)
        batch_size = x.size()[0]

        mask =  _lengths_to_masks(lengths,max_length)
        
        x_embd =  self.dropout(self.embedding(x).transpose(0,1))
        hidden_fw = self.init_hidden(len(x))
        
        hidden_bw = self.init_hidden(len(x))
        
        lstm_fw_outputs = []
        
        lstm_bw_outputs = []
        
        c_mask = _lengths_to_masks(concept_lengths.view(-1),max_concept_length).view(-1,max_length,max_concept_length)
                
        c = self.concept_embedding(concepts.view(-1)).view(batch_size,max_length,-1,self.concept_dim)
        
        concept_outputs = self.concept_avg(c,c_mask)
        
        # concept_outputs = (concept_outputs * mask.unsqueeze(2).expand_as(concept_outputs)).sum(1).expand_as(concept_outputs)
                
        for i in range(self.max_length): 
        
            hidden_fw = self.lstm_fw(x_embd[i],concept_outputs[:,i],hidden_fw)
        
            hidden_fw = [fw * mask[:,i].unsqueeze(1).expand_as(fw) for fw in hidden_fw]
            
            
            hidden_bw = self.lstm_bw(x_embd[-i-1],concept_outputs[:,-i-1],hidden_bw)
            
            hidden_bw = [bw * mask[:,-i-1].unsqueeze(1).expand_as(bw) for bw in hidden_bw]
            
            
            lstm_fw_outputs.append(hidden_fw[0][:,:self.hidden_dim])
            
            lstm_bw_outputs.append(hidden_bw[0][:,:self.hidden_dim])
            
        
        lstm_bw_outputs = lstm_bw_outputs[::-1]
        
        
        lstm_outputs = torch.cat([torch.cat([fw,bw],1).unsqueeze(1) for fw,bw in zip(lstm_fw_outputs,lstm_bw_outputs)],1)
        

        return self.dropout(lstm_outputs)
    
    
   