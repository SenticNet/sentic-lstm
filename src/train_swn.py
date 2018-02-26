import sys
import argparse
from utils import *
import time
import numpy as np
import logging
import math
from batcher import Batcher
from hook import acc_hook
from sklearn.externals import joblib
import torch
import torch.autograd as autograd

parser = argparse.ArgumentParser()

parser.add_argument("data",help="path to dataset")

parser.add_argument("mode",help="global-attention,target-attention,avg,sentic")

parser.add_argument("data_fn",help="semeval or sentihood")

args = parser.parse_args()

if args.mode[:6] == 'sentic':
    
    use_concepts = True
    
else:
    
    use_concepts = False
    
cell = 'sentic'
    
data_name = args.data_fn
    
input_dir =  args.data

if data_name == 'sentihood':
    num_epoch = 15
    data = joblib.load(input_dir+'/data_cpts_swn_all.150.pkl')
    
    num_ways = 3

else:
    num_epoch = 15
    data = joblib.load(input_dir+'/data_semeval_swn_all.150.pkl')

    num_ways = 4


train = data['train']

if data_name == 'sentihood':

    dev = data['dev']

test = data['test']

concepts_dict = dict([[v,k] for k,v in data['concepts_dict'].items()])

print "train size:",len(train[0])

if data_name == 'sentihood':

    print "dev size:",len(dev[0])

print"test size",len(test[0])

dicts = data['dicts']

id2token = dicts['id2token']

embd = data['embd']

num_classes = train[3].shape[1]

max_length = train[0].shape[1]

new_concept_embd = data['concepts_vecs']

concept_embd = torch.Tensor(new_concept_embd)


other_id = len(dicts['label2id'])

aspects = [dicts['id2label'][k] for k in dicts['id2label']]

if use_concepts:
    
    train_batcher = Batcher(train,16,data['concepts_train'])
    
    if data_name == 'sentihood':

        dev_batcher = Batcher(dev,len(dev[0]),data['concepts_dev'])

    test_batcher = Batcher(test,len(test[0]),data['concepts_test'])
    
else:
    
    train_batcher = Batcher(train,16)
    
    if data_name == 'sentihood':

        dev_batcher = Batcher(dev,len(dev[0]))

    test_batcher = Batcher(test,len(test[0]))
    
input_dim = 150

hidden_dim = 50
    
print args.mode

if args.mode == 'sentic-bi':
    
    from biaff_lstm_model import BiAffLSTM_Model
    
    model = BiAffLSTM_Model(num_classes,max_length,len(dicts['token2id']),embd\
              ,emb_dim = input_dim,hidden_dim=hidden_dim,concept_vector=concept_embd,lr=0.001,num_ways=num_ways,cell=cell)

elif args.mode == 'sentic':
    
    from aff_lstm_model import AffLSTM_Model
    
    model = AffLSTM_Model(num_classes,max_length,len(dicts['token2id']),embd\
              ,emb_dim = input_dim,hidden_dim=hidden_dim,concept_vector=concept_embd,lr=0.001,num_ways=num_ways,cell=cell)
else:
    
    print "using non-concept model"
    
    if args.mode == 'distance':
        
        from distance_lstm_model import DistAttLSTM_Model
        
        model = DistAttLSTM_Model(num_classes,max_length,len(dicts['token2id']),embd\
              ,emb_dim = input_dim,hidden_dim=hidden_dim,lr=0.001,num_ways=num_ways)
    elif args.mode == 'distance-deep':
        
        from distance_dmn_model import DistDMN_Model
        
        model = DistDMN_Model(num_classes,max_length,len(dicts['token2id']),embd\
              ,emb_dim = input_dim,hidden_dim=hidden_dim,lr=0.001,num_ways=num_ways)
    if args.mode == 'global-attention':
        
        from biatt_lstm_model import BiAttLSTM_Model
        
        model = BiAttLSTM_Model(num_classes,max_length,len(dicts['token2id']),embd\
              ,emb_dim = input_dim,hidden_dim=hidden_dim,lr=0.001,num_ways=num_ways)
    
    elif args.mode == 'target-attention':
        
        from att_lstm_model import AttLSTM_Model
        
        model = AttLSTM_Model(num_classes,max_length,len(dicts['token2id']),embd\
              ,emb_dim = input_dim,hidden_dim=hidden_dim,lr=0.001,num_ways=num_ways)

num_back=0

step_per_epoch = train_batcher.max_batch_num

best_acc = 0 

best_rep = ""

best_dev_rep =  ""

train_batcher.shuffle()

for epoch in range(num_epoch):
    
    loss = 0.0
    
    print "Epoch %d" % epoch
        
    for i in range(step_per_epoch):
        
        random_ = np.random.random_sample()
        
        if not use_concepts:
            
            input_x, y, targets, lengths = train_batcher.next()
        
        else:
            
            input_x,y,targets,lengths,cpts,cpts_len= train_batcher.next()
                        
            cpts = autograd.Variable(torch.from_numpy(cpts).long())
            
            cpts_len = autograd.Variable(torch.from_numpy(cpts_len))

                
        input_x = autograd.Variable(torch.from_numpy(input_x).long())
        
        y = autograd.Variable(torch.from_numpy(y)).long()
        
        targets = autograd.Variable(torch.from_numpy(targets).float())
        
        lengths = autograd.Variable(torch.from_numpy(lengths))
    
        if use_concepts:

            loss += model.train_(input_x,y,targets,lengths,cpts,cpts_len).data[0]

        else:

            loss += model.train_(input_x,y,targets,lengths).data[0]
    
    if data_name == 'sentihood':
    
        if not use_concepts:

            input_x, y, targets, lengths = dev_batcher.next()

        else:

            input_x,y,targets,lengths,cpts,cpts_len= dev_batcher.next() 

            cpts = autograd.Variable(torch.from_numpy(cpts).long())

            cpts_len = autograd.Variable(torch.from_numpy(cpts_len))

        input_x = autograd.Variable(torch.from_numpy(input_x).long())

        targets = autograd.Variable(torch.from_numpy(targets).float())

        lengths = autograd.Variable(torch.from_numpy(lengths))

        if use_concepts:

            logits = model.test(input_x,targets,lengths,cpts,cpts_len)

        else:

            logits = model.test(input_x,targets,lengths)

        dev_acc,dev_rep,_ = acc_hook(logits,y,other_id)

#         print "dev set"

#         print "\n".join(dev_rep)
    
    if not use_concepts:

        input_x, y, targets, lengths = test_batcher.next()

    else:

        input_x,y,targets,lengths,cpts,cpts_len= test_batcher.next()
        
        cpts = autograd.Variable(torch.from_numpy(cpts).long())
            
        cpts_len = autograd.Variable(torch.from_numpy(cpts_len))
        
    input_x = autograd.Variable(torch.from_numpy(input_x).long())
    
    targets = autograd.Variable(torch.from_numpy(targets).float())
    
    lengths = autograd.Variable(torch.from_numpy(lengths))
    
    if use_concepts:
            
        logits = model.test(input_x,targets,lengths,cpts,cpts_len)
        
    else:
    
        logits = model.test(input_x,targets,lengths)
    
    test_acc,rep,_ = acc_hook(logits,y,other_id)
    
#     print "test set"
    
#     print "\n".join(rep)
    
    if data_name == 'sentihood':
        
        if dev_acc > best_acc:
        
            best_rep = rep

            best_acc = dev_acc

            best_dev_rep = dev_rep

if data_name == 'sentihood':

    print "best dev" 
    print "\n".join(best_dev_rep)
    print "best test"
    print "\n".join(best_rep)
else:
    print "test set"
    
    print "\n".join(rep)

