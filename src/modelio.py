
# coding: utf-8

# In[12]:


import torch
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


# In[ ]:


class prepare(object):
    def __init__(self, usingGPU):
        self.usingGPU = usingGPU 
        
    def prepare_sentence(self, conll):
        sentence, pos, dp, lu, frame = [],[],[], False, False
        for token in conll:
            w,p,d,l,f = token[1],token[5],token[10],token[12],token[13]
            sentence.append(w)
            pos.append(p)
            dp.append(d)
            if token[12] != '_':
                lu, frame = token[12], token[13]
        return sentence, pos, dp, lu, frame
    
    def prepare_sequence(self, seq, to_ix):
        vocab = list(to_ix.keys())
        idxs = []
        for w in seq:
            if w in vocab:
                idxs.append(to_ix[w])
            else:
                idxs.append(0)            
        if self.usingGPU:
            return torch.tensor(idxs).type(torch.cuda.LongTensor)
        else:
            return torch.tensor(idxs, dtype=torch.long)
        
    def prepare_ix(self, item, to_ix):
        idxs = [0]
        try:
            idxs = [ to_ix[item] ]
        except:
            pass
        if self.usingGPU:
            return torch.tensor(idxs).type(torch.cuda.LongTensor)
        else:
            return torch.tensor(idxs, dtype=torch.long)  
    


# In[13]:


class tensor2tag(object):
    
    def __init__(self, frame_to_ix, fe_to_ix, usingGPU):
        self.usingGPU = usingGPU 
        self.frame_to_ix = frame_to_ix
        self.fe_to_ix = fe_to_ix
        
    def get_frame_by_tensor(self, t):
        value, indices = t.max(1)
        score = pow(10, value)

        pred = None
        for frame, idx in self.frame_to_ix.items():
            if idx == indices:
                pred = frame
                break
        return score, pred
    
    def get_fe_by_tensor(self, t):
        value, indices = t.max(1)
        score = pow(10, value)

        pred = None
        for fe, idx in self.fe_to_ix.items():
            if idx == indices:
                pred = fe
                break
        return score, pred

