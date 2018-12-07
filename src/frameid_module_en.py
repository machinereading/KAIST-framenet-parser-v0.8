
# coding: utf-8

# In[19]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from optparse import OptionParser
import torch.autograd as autograd
import os
import sys
import pprint
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
sys.path.insert(0,'../')
sys.path.insert(0,'./src')
import preprocessor
import dataio
import feature_handler
import modelio
import masked_softmax
import evaluator

import time
import datetime
import json
from sklearn.metrics import precision_recall_fscore_support as f1score
start_time = time.time()
torch.manual_seed(1)


# In[33]:


try:
    dir_path = os.path.dirname( os.path.abspath( __file__ ) )
except:
    dir_path = './'
frameid_model = dir_path+'/../model/frameid-lstm-ko.pt'

language = 'en'


# In[21]:


if language == 'ko':
    PRETRAINED_DIM = 300
else:
    PRETRAINED_DIM = 100

configuration = {'token_dim': 60,
                 'hidden_dim': 64,
                 'pos_dim': 4,
                 'lu_dim': 64,
                 'lu_pos_dim': 5,
                 'lstm_input_dim': 64,
                 'lstm_dim': 64,
                 'lstm_depth': 2,
                 'hidden_dim': 64,
                 'num_epochs': 100,
                 'learning_rate': 0.001,
                 'dropout_rate': 0.01,
                 'using_GPU': True,
                 'using_pretrained_embedding': True,
                 'using_exemplar': False,
                 'pretrained_embedding_dim': PRETRAINED_DIM,
                 'language': language,
                 'batch_size': 64}
# print('\n### CONFIGURATION ###\n')
# pprint.pprint(configuration)

#Hyper-parameters
usingGPU = configuration['using_GPU']
TOKDIM= configuration['token_dim']
POSDIM = configuration['pos_dim']
LUDIM = configuration['lu_dim']
LPDIM = configuration['lu_pos_dim']
INPDIM = TOKDIM + POSDIM
LSTMINPDIM = configuration['lstm_input_dim']
LSTMDIM = configuration['lstm_dim']
LSTMDEPTH = configuration['lstm_depth']
HIDDENDIM = configuration['hidden_dim']
NUM_EPOCHS = configuration['num_epochs']
learning_rate = configuration['learning_rate']
DROPOUT_RATE = configuration['dropout_rate']
batch_size = configuration['batch_size']
USE_WV = configuration['using_pretrained_embedding']
USE_EXEM = configuration['using_exemplar']
PRETRAINED_DIM = configuration['pretrained_embedding_dim']


# In[22]:


training_data, test_data, dev_data, exemplar_data = preprocessor.load_data(language, USE_EXEM)
preprocessor.data_stat(language, USE_EXEM)

lufrmap, frargmap = preprocessor.read_map(language)
token2wv_dir = dir_path+'/../data/'+language+'.token2wv.json'
with open(token2wv_dir,'r') as f:
    token2wv = json.load(f)


# In[23]:


def prepare_index():
    word_to_ix = {}
    pos_to_ix = {}
    frame_to_ix = {}
    lu_to_ix = {}
    word_to_ix['UNSEEN'] = 0
    word_vocab, pos_vocab, frame_vocab, lu_vocab = [],[],[],[]
    all_data = training_data + test_data + dev_data
    for tokens in all_data:
        for t in tokens:
            word = t[1]
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
                word_vocab.append(word)
            pos = t[5]
            if pos not in pos_to_ix:
                pos_to_ix[pos] = len(pos_to_ix)
                pos_vocab.append(pos)
            frame = t[13]
            if frame != '_':
                if frame not in frame_to_ix:
                    frame_to_ix[frame] = len(frame_to_ix)
                    frame_vocab.append(frame)
            lu = t[12]
            if lu != '_':
                if lu not in lu_to_ix:
                    lu_to_ix[lu] = len(lu_to_ix)
                    lu_vocab.append(lu)
    return word_to_ix, pos_to_ix, frame_to_ix, lu_to_ix
word_to_ix, pos_to_ix, frame_to_ix, lu_to_ix = prepare_index()
WORD_VOCAB_SIZE, POS_VOCAB_SIZE, LU_VOCAB_SIZE, FRAME_VOCAB_SIZE = len(word_to_ix), len(pos_to_ix), len(lu_to_ix), len(frame_to_ix)


# In[24]:


def prepare_vocab():
    word_vocab, pos_vocab, frame_vocab, lu_vocab = [],[],[],[]
    for tokens in training_data:
        for t in tokens:
            lu = t[12]
            if lu != '_':
                lu_vocab.append(lu)
    lu_vocab = list(set(lu_vocab))
    return lu_vocab
lu_vocab = prepare_vocab()

def prepare_sentence(tokens):
    sentence, pos, lu, frame = [],[], False, False
    for token in tokens:
        w,p,l,f = token[1],token[5],token[12],token[13]
        sentence.append(w)
        pos.append(p)
        if token[12] != '_':
            lu, frame = token[12], token[13]
    return sentence, pos, lu, frame

def prepare_sequence(seq, to_ix):
    vocab = list(to_ix.keys())
    idxs = []
    for w in seq:
        if w in vocab:
            idxs.append(to_ix[w])
        else:
            idxs.append(0)            
    if usingGPU:
        return torch.tensor(idxs).type(torch.cuda.LongTensor)
    else:
        return torch.tensor(idxs, dtype=torch.long)
    
def prepare_lu_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq if w != '_']
    idxs = list(set(idxs))
    if usingGPU:
        return torch.tensor(idxs).type(torch.cuda.LongTensor)
    else:
        return torch.tensor(idxs, dtype=torch.long)
        
def prepare_ix(item, to_ix):
    idxs = [ to_ix[item] ]
    if usingGPU:
        return torch.tensor(idxs).type(torch.cuda.LongTensor)
    else:
        return torch.tensor(idxs, dtype=torch.long)    

def prepare_arg_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq if w != '_']
    idxs = list(set(idxs))
    if usingGPU:
        return torch.tensor(idxs).type(torch.cuda.LongTensor)
    else:
        return torch.tensor(idxs, dtype=torch.long) 

def prepare_frame_vector(seq, to_ix):
    if usingGPU:
        frame_vector =  torch.zeros(len(to_ix)).type(torch.cuda.LongTensor)
    else:
        frame_vector =  torch.zeros(len(to_ix), dtype=torch.long)
    for f in seq:
        if f != '_':
            fid = frame_to_ix[f]
            frame_vector[fid] = 1
    return frame_vector

def get_targetpositions(tokens):
    positions = []
    lu = False
    for i in tokens:
        if i[12] != '_':
            positions.append(int(i[0]))
    positions = np.asarray(positions)
    positions = torch.from_numpy(positions)
    if usingGPU:
        return positions.type(torch.cuda.LongTensor)
    else:
        return positions

def get_target_span(sentence, targetpositions):
    start, end = int(targetpositions[0]), int(targetpositions[-1])
    span = {}
    if start == 0: span['start'] = 0
    else: span['start'] = start -1
    if end == len(sentence): span['end'] = end+1
    else: span['end'] = end+2
    return span


# # Masked Softmax
def gen_mask(lu):
    mask_list = []
    frame_candis = lufrmap[lu]
    for fr in frame_to_ix:
#         print(frame_to_ix[fr])
        if fr in frame_candis:
            mask_list.append(1)
        else:
            mask_list.append(0)
    mask_numpy = np.array(mask_list)
    mask = torch.from_numpy(mask_numpy)
    if usingGPU:
        return mask.type(torch.cuda.LongTensor)
    else:
        return mask

def masked_softmax(vec, mask, dim=-1):
    mask = mask.float()
    vec_masked = vec * mask + (1 / mask - 1)
    vec_min = vec_masked.min(1)[0]
    vec_exp = (vec - vec_min.unsqueeze(-1)).exp()
    vec_exp = vec_exp * mask.float()
    result = vec_exp / vec_exp.sum(1).unsqueeze(-1)
    return result


# In[25]:


class LSTMTagger(nn.Module):
    
    def __init__(self, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        
        # define look-up embeddis for token, pos, and lu
        self.token_embeddings = nn.Embedding(WORD_VOCAB_SIZE, TOKDIM)
        self.pos_embeddings = nn.Embedding(POS_VOCAB_SIZE, POSDIM)
        self.lu_embeddings = nn.Embedding(LU_VOCAB_SIZE, LUDIM)
        self.word_embeddings = nn.Embedding(WORD_VOCAB_SIZE, PRETRAINED_DIM)
               
        # 1st LSTM network (bi-LSTM)
        self.lstm_1 = nn.LSTM(INPDIM, HIDDENDIM//2, bidirectional=True, num_layers=LSTMDEPTH, dropout=DROPOUT_RATE)
        self.hidden_lstm_1 = self.init_hidden_lstm_1()
        
        # 2nd LSTM network (LSTM)
        self.hidden_lstm_2 = self.init_hidden_lstm_2()
        self.lstm_2 = nn.LSTM(HIDDENDIM, HIDDENDIM, num_layers=LSTMDEPTH, dropout=DROPOUT_RATE)
        
        # Linear 
        self.target2lstminput = nn.Linear(INPDIM, LSTMINPDIM)
        self.addwv2lstminput = nn.Linear(PRETRAINED_DIM+INPDIM, LSTMINPDIM)
        self.target2hidden = nn.Linear(LSTMINPDIM+LUDIM, HIDDENDIM)
        self.hidden2tag = nn.Linear(HIDDENDIM, tagset_size) 
    
    def init_hidden_lstm_1(self):
        if usingGPU:
            return (torch.zeros(4, 1, HIDDENDIM//2).cuda(),
                torch.zeros(4, 1, HIDDENDIM//2).cuda())
        else:
            return (torch.zeros(4, 1, HIDDENDIM//2),
                torch.zeros(4, 1, HIDDENDIM//2))
        
    def init_hidden_lstm_2(self):
        if usingGPU:
            return (torch.zeros(2, 1, HIDDENDIM).cuda(),
                torch.zeros(2, 1, HIDDENDIM).cuda())
        else:
            return (torch.zeros(2, 1, HIDDENDIM),
                torch.zeros(2, 1, HIDDENDIM))
    
        
    def forward(self, sentence, pos, targetpositions, lu, tokens):
#         if USE_WV:
#             word_embs = self.token_embeddings(sentence)
#         else:
#             word_embs = self.token_embeddings(sentence)
        tok_embs = self.token_embeddings(sentence)
        pos_embs = self.pos_embeddings(pos)
        
        lu_ix = prepare_ix(lu, lu_to_ix)
        lu_embs = self.lu_embeddings(lu_ix)

        # 1) input vector
        if not USE_WV: #concat token and pos enbeddings 
            target_embeds = torch.cat((tok_embs, pos_embs), 1)
            lstm_embeds = self.target2lstminput(target_embeds)
        else: #concat token embedding and pretrained word embedding and pos embedding
            word_embs = self.word_embeddings(sentence)
            for i in range(len(tokens)):
                if tokens[i] in token2wv:
                    pretrained_wv = token2wv[tokens[i]].split(' ')
                    pretrained_wv = np.array([float(x) for x in pretrained_wv])
                    pretrained_wv = torch.from_numpy(pretrained_wv)
                    word_embs[i] = pretrained_wv       
            target_embeds = torch.cat((tok_embs, word_embs, pos_embs), 1)
            lstm_embeds = self.addwv2lstminput(target_embeds)
        
        embeds = lstm_embeds.view(len(sentence), 1, -1)
        embeds = F.relu(embeds)

        # 2) first Bi-LSTM for token sequence
        lstm_out_1, self.hidden_lstm_1 = self.lstm_1(
            embeds, self.hidden_lstm_1)
        span = get_target_span(sentence, targetpositions)
        target_lstm = lstm_out_1[span['start']:span['end']]
        
        # 3) second LSTM for last hidden state for the context of LU (output: target vector)
        lstm_out_2, self.hidden = self.lstm_2(
            target_lstm, self.hidden_lstm_2)
        target_vec = lstm_out_2[-1]

        # 4) concate target vector with lu embedding
        lu_vec = torch.cat( (target_vec, lu_embs) ,1)

        # 5) linear
        tag_space = self.target2hidden(lu_vec)
        tag_space = F.relu(tag_space)
        tag_space = self.hidden2tag(tag_space)

        # 6) masked softmax
        mask = gen_mask(lu)
        tag_scores = masked_softmax(tag_space, mask)

        return tag_scores


# In[26]:


def get_frame_by_tensor(t):
    value, indices = t.max(1)
    score = pow(10, value)
    
    pred = None
    for frame, idx in frame_to_ix.items():
        if idx == indices:
            pred = frame
            break
    return score, pred


# In[32]:


class frame_identifier():
    
    def __init__(self):
        pass
    
    def identifier(self, conll, model):
        
        score = 0
        pred = '_'

        sent = conll
        for t in sent:
            if t[13] != '_':
                answer_lu = t[12]
        if answer_lu in lu_vocab:
            sentence, pos, lu, frame = prepare_sentence(sent)            
            targetpositions = get_targetpositions(sent)
            sentence_in = prepare_sequence(sentence, word_to_ix)
            pos_in = prepare_sequence(pos, pos_to_ix)    
            tag_scores = model(sentence_in,pos_in, targetpositions, lu, sentence)
            score, pred = get_frame_by_tensor(tag_scores)

        else:
            pass
        
        return pred, score
        
        

    

# frame_identifier()

