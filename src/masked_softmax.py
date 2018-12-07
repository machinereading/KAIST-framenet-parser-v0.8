
# coding: utf-8

# In[1]:


import torch
import torch
import numpy as np


# In[ ]:


class softmax(object):
    
    def __init__(self, lufrmap, frargmap, frame_to_ix, fe_to_ix, usingGPU):
        self.usingGPU = usingGPU
        self.lufrmap = lufrmap
        self.frargmap = frargmap
        self.frame_to_ix = frame_to_ix
        self.fe_to_ix = fe_to_ix
        
    def gen_mask_lu(self, lu):
        mask_list = []
        frame_candis = self.lufrmap[lu]
        for fr in self.frame_to_ix:
    #         print(frame_to_ix[fr])
            if fr in frame_candis:
                mask_list.append(1)
            else:
                mask_list.append(0)
        mask_numpy = np.array(mask_list)
        mask = torch.from_numpy(mask_numpy)
        if self.usingGPU:
            return mask.type(torch.cuda.LongTensor)
        else:
            return mask
        
    def gen_mask_frame(self, frame):
        mask_list = []
        fe_candis = self.frargmap[frame]  
        for fe in self.fe_to_ix:
    #         print(frame_to_ix[fr])
            if fe in fe_candis:
                mask_list.append(1)
            else:
                mask_list.append(0)
        mask_numpy = np.array(mask_list)
        mask = torch.from_numpy(mask_numpy)
        if self.usingGPU:
            return mask.type(torch.cuda.LongTensor)
        else:
            return mask
        
    def masked_softmax(self, vec, mask, dim=-1):
        mask = mask.float()
        vec_masked = vec * mask + (1 / mask - 1)
        vec_min = vec_masked.min(1)[0]
        vec_exp = (vec - vec_min.unsqueeze(-1)).exp()
        vec_exp = vec_exp * mask.float()
        result = vec_exp / vec_exp.sum(1).unsqueeze(-1)
        result = torch.clamp(result,1e-10,1.0)
        return result
    
    def masked_log_softmax(self, vec, mask, dim=-1):
        mask = mask.float()
        vec_masked = vec * mask + (1 / mask - 1)
        vec_min = vec_masked.min(1)[0]
        vec_exp = (vec - vec_min.unsqueeze(-1)).exp()
        vec_exp = vec_exp * mask.float()
        result = vec_exp / vec_exp.sum(1).unsqueeze(-1)
        result = torch.clamp(result,1e-10,1.0)
        result = torch.log(result)
        return result

