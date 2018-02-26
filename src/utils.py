import json
import numpy as np
from sklearn.externals import joblib
#import spacy
import torch
import torch.autograd as autograd


def _bilayer_attention(inputs,query,mask,linear1,linear2,nonlinear_func,err=0.0):
    batch_size = len(inputs)
        
    dim = inputs.size()[-1] + query.size()[-1]
    
    inputs_ = torch.cat([inputs,query.unsqueeze(1).expand(batch_size,inputs.size()[1],query.size()[-1])],2)
    
    inputs_ = inputs_.view(-1,dim)

    att_vecs = linear2(nonlinear_func(linear1(inputs_))).squeeze().view(batch_size,-1).exp()

    att_vecs = att_vecs * mask

    att_vecs = att_vecs / (att_vecs.sum(-1).expand_as(att_vecs) + err)

    outputs  = (att_vecs.unsqueeze(2).expand_as(inputs) * inputs).sum(1)
        
    outputs =  outputs.squeeze(1)
    
    return att_vecs,outputs

def deep_bilayer_attention(inputs,mask,linear1s,linear2s,nonlinear_func,num_classes=1, err=0.0):
    
    batch_size = len(inputs)
    
    depth = len(linear1s)
    
    input_dim = inputs.size()[-1] 
    
    prev = None
    
    
    max_len = inputs.size()[1]
        
    for i in range(depth):
        
        if prev is not None:
            
            query_ = prev
            
            dim = input_dim + query_.size()[-1]

            inputs_ = torch.cat([inputs,query_.unsqueeze(1).expand(batch_size,max_len,num_classes,query_.size()[-1])],-1)
            
            inputs_ = inputs_.view(-1,dim)
            
            layer1_outputs = linear1s[i](inputs_)
            
            layer1_outputs  = nonlinear_func(layer1_outputs.view(batch_size*max_len,num_classes,-1))

        else:
            
            dim = input_dim
            
            inputs_ = inputs.view(-1,dim)
            
            layer1_outputs = linear1s[i](inputs_)
            
            att_dim = layer1_outputs.size()[-1]

            layer1_outputs = layer1_outputs.unsqueeze(1).expand(batch_size*max_len,num_classes,att_dim)
    
            layer1_outputs  = nonlinear_func(layer1_outputs)
        
        att_vecs = []
        
        for j in range(num_classes):
            
            att_vecs.append(linear2s[i][j](layer1_outputs[:,i]).view(batch_size,max_len,1))
        
        att_vecs = torch.cat(att_vecs,-1).exp()
        
        
        att_vecs = att_vecs * mask.unsqueeze(2).expand_as(att_vecs)
        
        att_vecs = att_vecs / (att_vecs.sum(1,keepdim=True).expand_as(att_vecs) + err)
        
        if prev is None:
            
            inputs = inputs.unsqueeze(2).expand(batch_size,max_len,num_classes,input_dim)
        
        prev  = (att_vecs.unsqueeze(3).expand_as(inputs) * inputs).sum(1)
        
        prev =  prev.squeeze(1)

    return att_vecs, prev

def multi_bilayer_attention(inputs,query,mask,linear1,linear2,nonlinear_func,num_att=12,err=0.0):
    
    batch_size = len(inputs)
        
    dim = inputs.size()[-1] + query.size()[-1]
    
    inputs_ = torch.cat([inputs,query.unsqueeze(1).expand(batch_size,inputs.size()[1],query.size()[-1])],2)
    
    inputs_ = inputs_.view(-1,dim)

    att_vecs = linear2(nonlinear_func(linear1(inputs_))).squeeze().view(batch_size,-1,num_att).exp()
    
    mask = mask.unsqueeze(2).expand_as(att_vecs)

    att_vecs = att_vecs * mask

    att_vecs = att_vecs / (att_vecs.sum(1).expand_as(att_vecs) + err)
        
    inputs = inputs.unsqueeze(2).expand(batch_size,inputs.size()[1],num_att,inputs.size()[2])

    outputs  = (att_vecs.unsqueeze(3).expand_as(inputs) * inputs).sum(1)
        
    outputs =  outputs.squeeze(1)
        
    return att_vecs,outputs


def _multi_bilayer_attention(inputs,query,mask,linear1,linear2,nonlinear_func,num_att=12,err=0.0):
    
    batch_size = len(inputs)
    
        
    dim = inputs.size()[-1] + query.size()[-1]
    
    inputs_ = torch.cat([inputs,query.unsqueeze(1).expand(batch_size,inputs.size()[1],query.size()[-1])],2)
    
    inputs_ = inputs_.view(-1,dim)
    
    att_vecs = linear2(nonlinear_func(linear1(inputs_))).view(batch_size,-1,num_att).exp()
    
    mask = mask.unsqueeze(2).expand_as(att_vecs)

    att_vecs = att_vecs * mask
        
        
    return att_vecs

def _lengths_to_masks(lengths, max_length):
    
    tiled_ranges = autograd.Variable(torch.arange(0,float(max_length)).unsqueeze(0).expand([len(lengths),max_length]))
    
    lengths = lengths.float().unsqueeze(1).expand_as(tiled_ranges)
    
    mask = tiled_ranges.lt(lengths).float()

    return mask

def weight_variable(shape):
    
    initial = np.random.uniform(-0.01, 0.01,shape)
    
    initial = torch.from_numpy(initial)
    
    return initial.float()


def add_item(label2id,id2label,tag):
        if tag not in label2id:
                label2id[tag] = len(label2id)
                id2label[len(label2id)-1] = tag
        return label2id[tag]
    
def target2distance(targets,t_max=100.0):
    start =0
    distances = []
    for target in targets:
        target_ =  [i for i in range(len(target)) if target[i] != 0]
        distance = [np.max([0,1- np.min([abs(i-t) for t in target_])/t_max]) for i in range(len(target))]
        distances.append(distance)
    return autograd.Variable(torch.FloatTensor(distances))
def construct_word_target_sequence(words,w2id,id2w):
	word_seq = []
	targets = []
	targets_ = [] 
	for i,w in enumerate(words):
		if 'LOCATION1' in w:
			targets.append(i)
		if 'LOCATION2' in w:
			targets_.append(i)
		if 'LOCATION' in w:
			word_seq.append(0)
		else:
			word_seq.append(add_item(w2id,id2w,w))
	return word_seq,targets,targets_



def align_vec(Xs,max_length):
     Xs_new = np.zeros([len(Xs),max_length],dtype=np.int32)
     lengths= np.zeros(len(Xs),dtype=np.int32)	
     for i in range(len(Xs)):
         lengths[i] = len(Xs[i])

         for j in range(len(Xs[i])):
             Xs_new[i][j]= Xs[i][j]

     return Xs_new,lengths

def sparse2vec(Ys,num_labels):
	Ys_new = np.zeros([len(Ys),num_labels],dtype=np.int32)
	for i in range(len(Ys)):
		Ys_new[i][Ys[i]] = 1.0
	return Ys_new

def sparse2vec_3D(Ys,num_labels):
        Ys_new = np.zeros([len(Ys),2,num_labels],dtype=np.int32)
        for i in range(len(Ys)):
                Ys_new[i][0][Ys[i][0]] = 1.0
                Ys_new[i][1][Ys[i][1]] = 1.0

        return Ys_new

def label2vec(Ys,num_labels,dim,num_ways=3):
        Ys_new = np.zeros([len(Ys),dim,num_labels,num_ways],dtype=np.float32)
	Ys_new[:,:,:,0] = 1.0
        for i in range(len(Ys)):
                for j in range(dim):
                        if Ys[i][j] != []:
                                for y in Ys[i][j]:
                                        Ys_new[i][j][y[0]][y[1]] = 1.0
					Ys_new[i][j][y[0]][0] = 0.0


        return Ys_new


def read_json_data(fn,dicts,nlp):
	data_array = []
	if dicts is None:
                token2id = {'LOCATION':0}
                label2id = {}#,2:_EOS,3:_UNK}
                label2id_ = {}
                id2token = {0:'LOCATION'}
                id2label = {}
                id2label_ = {}
                feat2id = {}
		id2feat = {}
		cfeat2id = {}
		id2cfeat = {}
                dicts = {'token2id':token2id,'label2id':label2id,'id2token':id2token,'id2label':id2label,"label2id_":label2id_,"id2label_":id2label_,"feat2id":feat2id,'id2feat':id2feat,'cfeat2id':cfeat2id,'id2cfeat':id2cfeat}
        else:
                token2id = dicts['token2id']
                label2id = dicts['label2id']
                id2token = dicts['id2token']
                id2label = dicts['id2label']
                id2label_ = dicts['id2label_']
                label2id_ = dicts['label2id_']
                feat2id = dicts['feat2id']
		id2feat = dicts['id2feat']
		cfeat2id = dicts['cfeat2id']
                id2cfeat = dicts['id2cfeat']

	max_word_length = 0 
	polarity = {'Positive':1,'Negative':2}
	with open(fn,'r') as ipt:
		data = json.loads("".join(ipt.readlines()))
		w_seqs = []
		target_seqs = []
		classes = []
		tags = []
#		target_other_seqs = []
		cnt_multi = 0
		couple_targets = []
		couple_classes = []
		couple_xs = [] 
		couple_tags = []
		for d in data:
                        d['text'] = d['text'].strip() 
			d_ = nlp(d['text'])
			tokens  = [w.text for w in d_ if w.text != "" and w.text != " "]
			pos_seqs =  [add_item(feat2id,id2feat,w.tag_) for w in d_ if w.text != "" and w.text!=" "]

			tid_seqs,targets,targets_ = construct_word_target_sequence(tokens,token2id,id2token)
			max_word_length = len(tid_seqs) if len(tid_seqs) > max_word_length else max_word_length
#			if len(d['opinions'])>1 and len(targets_) !=0:
#				print d['text']
#				print [(o['aspect'],o['target_entity'],o['sentiment'])  for o in d['opinions']]
			opinions = [o for o in d['opinions'] if o['target_entity'] == 'LOCATION1']
			opinions_ = [o for o in d['opinions'] if o['target_entity'] == 'LOCATION2']
			couple_features = []
			if len(targets_) != 0:


				features =  _feature(targets,targets_,tokens)
				if opinions != []:
                                        cls = [(add_item(label2id,id2label,l['aspect']),polarity[l['sentiment']]) for l in opinions]
                                else:
                                        cls = []

#				w_seqs.append(tid_seqs)
#	                        target_seqs.append(targets_)
				if opinions_ != []:
	                                cls_ = [(add_item(label2id,id2label,l['aspect']),polarity[l['sentiment']]) for l in opinions_]
        	                else:
                	                cls_ = []
				couple_targets.append([targets,targets_])
				couple_classes.append([cls,cls_])
				couple_xs.append(tid_seqs)	
				couple_tags.append(pos_seqs)
				couple_features.append(add_item(cfeat2id,id2cfeat,features[0]))
				#tags.append(pos_seqs)
				#raw.append((opinions_,d['text']))
			else:
				w_seqs.append(tid_seqs)
	                        target_seqs.append(targets)

        	                if opinions != []:
                	                classes.append([[(add_item(label2id,id2label,l['aspect']),polarity[l['sentiment']]) for l in opinions]])
                       		else:
                                	classes.append([[]])
        	                tags.append(pos_seqs)

	return w_seqs,target_seqs,classes,dicts,max_word_length,tags,couple_targets,couple_classes,couple_xs,couple_tags,np.asarray(couple_features)


			
		
def custom_pipeline(nlp):
    return [nlp.tagger,]#, nlp.parser, nlp.entity)


def create_dataset(input_dir,output_dir):
	
	nlp = spacy.load('en',create_pipeline=custom_pipeline) 

	train_set_fn = input_dir + '/sentihood-train.json'
	dev_set_fn = input_dir + '/sentihood-dev.json'
	test_set_fn = input_dir + '/sentihood-test.json'

	words_train,targets_train,classes_train,dicts,max_length_train,tags_train,couple_targets_train,couple_classes_train,couple_xs_train,couple_tags_train,couple_feature_train = read_json_data(train_set_fn,None,nlp)
	words_dev,targets_dev,classes_dev,dicts,max_length_dev,tags_dev,couple_targets_dev,couple_classes_dev,couple_xs_dev, couple_tags_dev,couple_feature_dev= read_json_data(dev_set_fn,dicts,nlp)
	words_test,targets_test,classes_test,dicts,max_length_test,tags_test,couple_targets_test,couple_classes_test,couple_xs_test,couple_tags_test,couple_feature_test = read_json_data(test_set_fn,dicts,nlp)

	max_length = max([max_length_train,max_length_dev,max_length_test])

	words_train,length_train = align_vec(words_train,max_length)
	words_dev,length_dev = align_vec(words_dev,max_length)
	words_test,length_test = align_vec(words_test,max_length)

	couple_xs_train,couple_length_train = align_vec(couple_xs_train,max_length)
        couple_xs_dev,couple_length_dev = align_vec(couple_xs_dev,max_length)
        couple_xs_test,couple_length_test = align_vec(couple_xs_test,max_length)
	
	tags_train,_ = align_vec(tags_train,max_length)
        tags_dev,_ = align_vec(tags_dev,max_length)
        tags_test,_ = align_vec(tags_test,max_length)

	couple_tags_train,_ = align_vec(couple_tags_train,max_length)
        couple_tags_dev,_ = align_vec(couple_tags_dev,max_length)
        couple_tags_test,_ = align_vec(couple_tags_test,max_length)

	num_classes = len(dicts['label2id'])
	classes_train = label2vec(classes_train,num_classes,1)[:,0,:]
	classes_dev = label2vec(classes_dev,num_classes,1)[:,0,:]
	classes_test = label2vec(classes_test,num_classes,1)[:,0,:]

        couple_classes_train = label2vec(couple_classes_train,num_classes,2)
        couple_classes_dev = label2vec(couple_classes_dev,num_classes,2)
        couple_classes_test = label2vec(couple_classes_test,num_classes,2)



	targets_train = sparse2vec(targets_train,max_length)
	targets_dev = sparse2vec(targets_dev,max_length)
	targets_test = sparse2vec(targets_test,max_length)

        couple_targets_train = sparse2vec_3D(couple_targets_train,max_length)
        couple_targets_dev = sparse2vec_3D(couple_targets_dev,max_length)
        couple_targets_test = sparse2vec_3D(couple_targets_test,max_length)

	
	w2v,dim = load_word2vec('./resources/all.bin')
	embd_table = create_id2vec(dicts['token2id'], w2v,dim)
	data = {'train': (words_train,length_train,targets_train,classes_train,tags_train),'dev':(words_dev,length_dev,targets_dev,classes_dev,tags_dev),'test':(words_test,length_test,targets_test,classes_test,tags_test),'dicts':dicts,'embd':embd_table,"couple_train":(couple_xs_train,couple_length_train,couple_targets_train,couple_classes_train,couple_tags_train,couple_feature_train),"couple_dev":(couple_xs_dev,couple_length_dev,couple_targets_dev,couple_classes_dev,couple_tags_dev,couple_feature_dev),"couple_test":(couple_xs_test,couple_length_test,couple_targets_test,couple_classes_test,couple_tags_test,couple_feature_test)}
	
	joblib.dump(data,output_dir)	


# def load_word2vec(file_path):
#     word2vec = {}
#     with open(file_path) as lines:
#         for line in lines:
#             split = line.split()
#             word = split[0]
#             vector_strings = split[1:]
#             vector = [float(num) for num in vector_strings]
#             word2vec[word] = np.array(vector)
#     return word2vec

def create_id2vec(word2id,word2vec,dim_of_vector):
    unk_vec = np.random.uniform(-0.01, 0.01,dim_of_vector)
    loc_vec = np.random.uniform(-0.01, 0.01,dim_of_vector)
    dim_of_vector = len(unk_vec)
    num_of_tokens = len(word2id)
    id2vec = np.zeros((num_of_tokens+1,dim_of_vector),dtype=np.float32)
    for word,t_id in word2id.items():
	if word == 'LOCATION':
		id2vec[t_id,:] = loc_vec
        elif word.lower() in word2vec:
                id2vec[t_id,:] = word2vec[word.lower()]
        else:
                id2vec[t_id,:] =  unk_vec
    return id2vec


def load_word2vec(file_path):
    word2vec = {}
    dim = 0
    with open(file_path) as lines:
        for line in lines:
            split = line.split()
            word = split[0]
            vector_strings = split[1:]
            vector = [float(num) for num in vector_strings]
            word2vec[word] = np.array(vector)
            dim  = len(vector)
    return word2vec,dim
def print_results(preds,raw,label_alphabet):

	for pred,item in zip(preds,raw):
		print "------"
		print item
		print [ label_alphabet[p] for p in pred]



