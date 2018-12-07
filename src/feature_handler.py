
# coding: utf-8

# In[2]:


import json
import torch
import torch
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


# In[3]:


class extractor(object):
    def __init__(self, language, josa_onlyPOS, usingGPU):
        self.language = language
        self.usingGPU = usingGPU 
        self.josa_onlyPOS = josa_onlyPOS
    
    def get_targetpositions(self, tokens):
        positions = []
        lu = False
        for i in tokens:
            if i[12] != '_':
                positions.append(int(i[0]))
        positions = np.asarray(positions)
        positions = torch.from_numpy(positions)
        if self.usingGPU:
            return positions.type(torch.cuda.LongTensor)
        else:
            return positions
        
    def get_target_span(self, sentence, targetposition):
        start, end = int(targetposition[0]), int(targetposition[-1])
        span = {}
        if start == 0: span['start'] = 0
        else: span['start'] = start -1
        if end == len(sentence): span['end'] = end+1
        else: span['end'] = end+2
        return span
    
    def get_argpositions(self, tokens):
        positions = []
        args = []
        for i in range(len(tokens)):

            token_id = int(tokens[i][0])
            bio_fe = tokens[i][14]
            args.append(bio_fe)
            begin = -1
            if bio_fe.startswith('O'):
                if i == 0:
                    begin = token_id

                if i > 0:
                    before = tokens[i-1][14]
                    if before == 'O':
                        pass
                    else:
                        begin = token_id

                if begin >= 0:
                    n = 1
                    while i+n < len(tokens):
                        nextfe = tokens[i+n][14]
                        if i+n == len(tokens)-1:
                            end = i+n
                            span = (begin, end, 'O')
                            positions.append(span)
                            break
                        elif nextfe != 'O':
                            end = token_id+n
                            if i+n == len(tokens)-1:
                                end = end+n
                            span = (begin, end, 'O')
                            positions.append(span)
                            break
                        else:
                            pass
                        n += 1

        for i in range(len(tokens)):
            token_id = int(tokens[i][0])
            bio_fe = tokens[i][14]        
            if bio_fe != 'O':
                fe = bio_fe.split('-')[1]        

                if bio_fe.startswith('B'):

                    begin = token_id
                    end = begin+1

                    n = 1
                    while i+n < len(tokens):
                        next_fe = tokens[i+n][14]
                        if next_fe != 'O':
                            next_fe = next_fe.split('-')[1]                    
                            if i+n == len(tokens)-1:
                                end = i+n
                                break
                            elif next_fe == fe:
                                end = int(tokens[i+n][0]) + 1
                                n = n+1
                            else:
                                break
                        else:
                            end = i+n
                            break
                    span = (begin, end, fe)
                    positions.append(span)
                elif bio_fe.startswith('S'):
                    begin = token_id
                    end = token_id +1
                    span = (begin, end, fe)
                    positions.append(span)
                else:
                    pass

        return positions
    
    def get_josa(self, tokens, position):
        tid = position[1] -1
        josa = 'null'
        dp_head = 'null'
        if self.language == 'ko':
            if tid >= len(tokens):
                tid = -1
            morphemes = tokens[tid][2].split('+')
            for m in morphemes:
                pos = m.split('/')[-1]
                if pos.startswith('J'):
                    if self.josa_onlyPOS:
                        josa = pos
                    else:
                        josa = m
                        
        return josa
    
    def get_last_dp(self, tokens, position):
        tid = position[1] -1
        last_dp = 'null'
        if self.language == 'ko':
            if tid >= len(tokens):
                tid = -1
            last_dp = tokens[tid][10]

        return last_dp
    
    
    def get_position_feature(self, target_position, arg_span):
        tgt_begin = target_position[0]
        tgt_end = int(target_position[-1])
        arg_begin = arg_span['begin']
        arg_end = arg_span['end']

        arg_len = arg_end - arg_begin
        # arg_len, before, after, overlapping, within
        position = torch.zeros(1,5)
        position[0][0] = arg_len
        if self.usingGPU:
            position =  position.type(torch.cuda.LongTensor)
        else:
            position = position

        if arg_end <= tgt_begin:
            position[0][1] = 1
        elif arg_begin >= tgt_end:
            position[0][2] = 1

        if arg_begin <= tgt_begin <= arg_end:
            position[0][3] = 1
        if tgt_begin <= arg_begin and arg_end <= tgt_end:
            position[0][4] = 1

        if self.usingGPU:
            return position.type(torch.cuda.FloatTensor)
        else:
            return position

