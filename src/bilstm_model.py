import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from utils import _lengths_to_masks


class BiLSTM_Model(nn.Module):

    def __init__(self,max_length,num_tokens,embd,emb_dim = 300,hidden_dim=100):
        
        super(BiLSTM_Model,self).__init__()
    
        self.emb_dim = emb_dim
                
        self.hidden_dim = hidden_dim 
        
        self.max_length = max_length
            
        self.embedding = nn.Embedding(num_tokens,emb_dim)
        
        self.embedding.weight = nn.Parameter(torch.from_numpy(embd),requires_grad=True)
        
        self.lstm_fw = nn.LSTMCell(self.emb_dim ,self.hidden_dim )
        
        self.lstm_bw = nn.LSTMCell(self.emb_dim ,self.hidden_dim)

        self.loss_fn = nn.functional.cross_entropy
        
        self.softmax = nn.Softmax()
        
        self.sigmoid = nn.Sigmoid()
        
        self.tanh = nn.Tanh()
        
        self.dropout = nn.Dropout(0.1)
                
            
    def init_hidden(self,batch_size):
        
        return (autograd.Variable(torch.zeros(batch_size, self.hidden_dim)),autograd.Variable(torch.zeros(batch_size, self.hidden_dim)))
    
    def forward(self,x,lengths):
        
        max_length = x.size()[1]
                
        mask =  _lengths_to_masks(lengths,max_length)
        
        x_embd =  self.dropout(self.embedding(x).transpose(0,1))
        
        hidden_fw = self.init_hidden(len(x))
        
        hidden_bw = self.init_hidden(len(x))
        
        lstm_fw_outputs = []
        
        lstm_bw_outputs = []
        
                
        for i in range(self.max_length): 
        
            hidden_fw = self.lstm_fw(x_embd[i],hidden_fw)
        
            hidden_fw = [fw * mask[:,i].unsqueeze(1).expand_as(fw) for fw in hidden_fw]
            
            
            hidden_bw = self.lstm_bw(x_embd[-i-1],hidden_bw)
            
            hidden_bw = [bw * mask[:,-i-1].unsqueeze(1).expand_as(bw) for bw in hidden_bw]
            
            lstm_fw_outputs.append(hidden_fw[0][:,:self.hidden_dim])
            
            lstm_bw_outputs.append(hidden_bw[0][:,:self.hidden_dim])
            
        lstm_bw_outputs = lstm_bw_outputs[::-1]
        
        lstm_outputs = torch.cat([torch.cat([fw,bw],1).unsqueeze(1) for fw,bw in zip(lstm_fw_outputs,lstm_bw_outputs)],1)
        
        return self.dropout(lstm_outputs)
    
    
   