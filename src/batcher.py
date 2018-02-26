import numpy as np
from sklearn.externals import joblib
import random


class Batcher:
    def __init__(self,data,batch_size,concepts=None):
        
        self.data = data
        self.num_of_samples = len(data[0])
        self.max_length = data[0].shape[1]
        self.dim = 300 #len(id2vec[0])
        self.num_of_labels = data[3].shape[1] 
        self.batch_size = batch_size
        self.batch_num = 0
        self.max_batch_num = int(self.num_of_samples / self.batch_size) 
        self.indexes = np.arange(len(data[0]))
        
        if concepts is not None:
            self.use_concepts = True
            self.concepts = concepts[1]
            self.concept_lengths = concepts[0]
            self.max_concepts_length = np.max([l for ls in self.concept_lengths for l in ls])
            print 'max concepts length', self.max_concepts_length
        else:
            self.use_concepts = False
    def next(self):
        X = np.zeros((self.batch_size,self.max_length),dtype=np.int32)
        Y = np.zeros((self.batch_size,self.num_of_labels),dtype=np.int32)
        targets = np.zeros((self.batch_size,self.max_length))
        lengths = np.zeros((self.batch_size),dtype=np.int32)
        # tags = np.zeros((self.batch_size,self.max_length),dtype=np.int32)
        if self.use_concepts:
            cpts = np.zeros((self.batch_size,self.max_length,self.max_concepts_length),dtype=np.int32)
            cpt_lengths = np.zeros((self.batch_size,self.max_length),dtype=np.int32)
        for i in range(self.batch_size):
            index = self.indexes[self.batch_num * self.batch_size + i]
            X[i,:] = self.data[0][index,:]
            lengths[i] = self.data[1][index]
            targets[i,:] = self.data[2][index,:]
            # tags[i,:] = self.data[4][index,:]
            for k in range(self.num_of_labels):
                Y[i,k] = self.data[3][index,k].nonzero()[0][0]
            if self.use_concepts:
                # print len(self.concept_lengths),index
                cpt_lengths[i,:lengths[i]] = self.concept_lengths[index]
                for j in range(lengths[i]):
                    cpts[i,j,:cpt_lengths[i,j]] = self.concepts[index][j]
        self.batch_num = (self.batch_num + 1) % self.max_batch_num

        res = [X,Y,targets,lengths]
 
        if self.use_concepts:
            res.extend([cpts,cpt_lengths])
        return res
    def shuffle(self):
        np.random.shuffle(self.indexes)

