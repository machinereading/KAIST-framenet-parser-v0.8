
# coding: utf-8

# In[1]:


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
# import preprocessor
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


# In[3]:


language = 'ko'

if language == 'ko':
    PRETRAINED_DIM = 300
else:
    PRETRAINED_DIM = 100

configuration = {'token_dim': 60,
                 'hidden_dim': 64,
                 'pos_dim': 4,
                 'lu_dim': 64,
                 'lu_pos_dim': 5,
                 'dp_label_dim': 10,
                 'josa_dim': 20,
                 'last_dp_dim': 4,
                 'frame_dim': 100,
                 'fe_dim': 50,
                 'lstm_input_dim': 64,
                 'lstm_dim': 64,
                 'lstm_depth': 2,
                 'hidden_dim': 64,
                 'position_feature_dim': 5,
                 'num_epochs': 50,
                 'learning_rate': 0.001,
                 'dropout_rate': 0.01,
                 'using_GPU': True,
                 'using_pretrained_embedding': True,
                 'using_exemplar': False,
                 'using_dependency_label': True,
                 'using_last_dependency_label': False,
                 'using_josa_pos': False,
                 'using_josa': False,
                 'using_full_context': True,
                 'pretrained_embedding_dim': PRETRAINED_DIM,
                 'language': language,
                 'batch_size': 64}
#Hyper-parameters
usingGPU = configuration['using_GPU']
TOKDIM= configuration['token_dim']
POSDIM = configuration['pos_dim']
LUDIM = configuration['lu_dim']
LPDIM = configuration['lu_pos_dim']
FRAMEDIM = configuration['frame_dim']
FEDIM = configuration['fe_dim']
DPLABELDIM = configuration['dp_label_dim']
LASTDPDIM = configuration['last_dp_dim']
JOSADIM = configuration['josa_dim']
LSTMINPDIM = configuration['lstm_input_dim']
LSTMDIM = configuration['lstm_dim']
POSITIONDIM = configuration['position_feature_dim']
LSTMDEPTH = configuration['lstm_depth']
HIDDENDIM = configuration['hidden_dim']
NUM_EPOCHS = configuration['num_epochs']
learning_rate = configuration['learning_rate']
DROPOUT_RATE = configuration['dropout_rate']
batch_size = configuration['batch_size']
USE_WV = configuration['using_pretrained_embedding']
USE_EXEM = configuration['using_exemplar']
USE_DP_LABEL = configuration['using_dependency_label']
USE_JOSA = configuration['using_josa']
USE_JOSA_POS = configuration['using_josa']
USE_LAST_DP = configuration['using_last_dependency_label']
USE_FULL_CONTEXT = configuration['using_full_context']
PRETRAINED_DIM = configuration['pretrained_embedding_dim']

if language =='en':
    USE_JOSA = False
    USE_JOSA_POS = False


# In[4]:


training_data, test_data, dev_data, exemplar_data = dataio.read_data(language, USE_EXEM)

lufrmap, frargmap = dataio.read_map(language)
token2wv = dataio.read_token2wv(language)

if USE_JOSA_POS: josa_onlyPOS = True
else: josa_onlyPOS = False
word_to_ix, pos_to_ix, dp_to_ix, josa_to_ix, frame_to_ix, lu_to_ix, fe_to_ix = dataio.prepare_idx(training_data + test_data + dev_data, language, josa_onlyPOS)
WORD_VOCAB_SIZE, POS_VOCAB_SIZE, DP_VOCAB_SIZE, JOSA_VOCAB_SIZE, LU_VOCAB_SIZE, FRAME_VOCAB_SIZE, FE_VOCAB_SIZE = len(word_to_ix), len(pos_to_ix), len(dp_to_ix), len(josa_to_ix), len(lu_to_ix), len(frame_to_ix), len(fe_to_ix)


# In[5]:


INPDIM = TOKDIM + POSDIM

if USE_DP_LABEL:
    INPDIM = INPDIM + DPLABELDIM

ARGDIM = LSTMINPDIM+LUDIM+FRAMEDIM+FEDIM+POSITIONDIM+HIDDENDIM
    
if USE_JOSA or USE_JOSA_POS:
    ARGDIM = ARGDIM + JOSADIM
if USE_LAST_DP:
    ARGDIM = ARGDIM + LASTDPDIM
    


class LSTMTagger(nn.Module):
    
    def __init__(self, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        
        # define look-up embeddis for token, pos, and lu
        self.token_embeddings = nn.Embedding(WORD_VOCAB_SIZE, TOKDIM)
        self.pos_embeddings = nn.Embedding(POS_VOCAB_SIZE, POSDIM)
        self.lu_embeddings = nn.Embedding(LU_VOCAB_SIZE, LUDIM)
        self.word_embeddings = nn.Embedding(WORD_VOCAB_SIZE, PRETRAINED_DIM)
        self.frame_embeddings = nn.Embedding(FRAME_VOCAB_SIZE, FRAMEDIM)
        self.fe_embeddings = nn.Embedding(FE_VOCAB_SIZE, FEDIM)
        self.dp_label_embeddings = nn.Embedding(DP_VOCAB_SIZE, DPLABELDIM)
        self.josa_embeddings = nn.Embedding(JOSA_VOCAB_SIZE, JOSADIM)
#         self.last_dp_embeddings = nn.Embedding(DP_VOCAB_SIZE, LASTDPDIM)
               
        # TOKEN LSTM network (bi-LSTM)
        self.lstm_tok = nn.LSTM(LSTMINPDIM, HIDDENDIM//2, bidirectional=True, num_layers=LSTMDEPTH, dropout=DROPOUT_RATE)
        self.hidden_lstm_tok = self.init_hidden_lstm_tok()
        
        # TARGET LSTM network (LSTM)
        self.hidden_lstm_tgt = self.init_hidden_lstm_tgt()
        self.lstm_tgt = nn.LSTM(HIDDENDIM, HIDDENDIM, num_layers=LSTMDEPTH, dropout=DROPOUT_RATE)
        
        # ARG LSTM network (bi-LSTM)
        self.hidden_lstm_arg = self.init_hidden_lstm_arg()
        self.lstm_arg = nn.LSTM(HIDDENDIM, HIDDENDIM//2, bidirectional=True, num_layers=LSTMDEPTH, dropout=DROPOUT_RATE)
        
        # Linear 
        self.target2lstminput = nn.Linear(INPDIM, LSTMINPDIM)
        self.addwv2lstminput = nn.Linear(PRETRAINED_DIM+INPDIM, LSTMINPDIM)
        self.arg2hidden = nn.Linear(ARGDIM, HIDDENDIM)
        self.hidden2tag = nn.Linear(HIDDENDIM, tagset_size) 
    
    def init_hidden_lstm_tok(self):
        if usingGPU:
            return (torch.zeros(4, 1, HIDDENDIM//2).cuda(),
                torch.zeros(4, 1, HIDDENDIM//2).cuda())
        else:
            return (torch.zeros(4, 1, HIDDENDIM//2),
                torch.zeros(4, 1, HIDDENDIM//2))
        
    def init_hidden_lstm_tgt(self):
        if usingGPU:
            return (torch.zeros(2, 1, HIDDENDIM).cuda(),
                torch.zeros(2, 1, HIDDENDIM).cuda())
        else:
            return (torch.zeros(2, 1, HIDDENDIM),
                torch.zeros(2, 1, HIDDENDIM))
        
    def init_hidden_lstm_arg(self):
        if usingGPU:
            return (torch.zeros(4, 1, HIDDENDIM//2).cuda(),
                torch.zeros(4, 1, HIDDENDIM//2).cuda())
        else:
            return (torch.zeros(4, 1, HIDDENDIM//2),
                torch.zeros(4, 1, HIDDENDIM//2))
    
        
    def forward(self, sentence, pos, dp, josa, last_dp, arg, target_position, arg_span, lu, frame, tokens):    
    
        tok_embs = self.token_embeddings(sentence)
        pos_embs = self.pos_embeddings(pos)
        dp_embs = self.dp_label_embeddings(dp)
        josa_embs = self.josa_embeddings(josa)
#         last_dp_embs = self.last_dp_embeddings(last_dp)
            
        lu_ix = prepare.prepare_ix(lu, lu_to_ix)
        lu_embs = self.lu_embeddings(lu_ix)
        
        frame_ix = lu_ix = prepare.prepare_ix(frame, frame_to_ix)
        frame_embs = self.frame_embeddings(frame_ix)
        
        fe_embs = self.fe_embeddings(arg)
        
        
        position_feature = feature_extractor.get_position_feature(target_position, arg_span)

        # 1) input vector
        if not USE_WV: #concat token and pos enbeddings 
            if USE_DP_LABEL:
                target_embeds = torch.cat((tok_embs, pos_embs, dp_embs), 1)
            else:
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
            
            if USE_DP_LABEL:
                target_embeds = torch.cat((tok_embs, word_embs, pos_embs, dp_embs), 1)
            else:
                target_embeds = torch.cat((tok_embs, word_embs, pos_embs), 1)
                
            lstm_embeds = self.addwv2lstminput(target_embeds)
        
        embeds = lstm_embeds.view(len(sentence), 1, -1)
        embeds = F.relu(embeds)

        # 2) first Bi-LSTM for token sequence
        lstm_out_tok, self.hidden_lstm_tok = self.lstm_tok(
            embeds, self.hidden_lstm_tok)
        
        # 3) GET frame-lu-target VECTOR
        target_span = feature_extractor.get_target_span(sentence, target_position)
        target_lstm = lstm_out_tok[target_span['start']:target_span['end']]
        lstm_out_tgt, self.hidden = self.lstm_tgt(
            target_lstm, self.hidden_lstm_tgt)
        target_vec = lstm_out_tgt[-1]
        lu_vec = torch.cat( (frame_embs, lu_embs, target_vec) ,1)        
        
        # 4) GET ARG VECTOR       
        if USE_FULL_CONTEXT == False:
            if arg_span['end'] > len(sentence):
                arg_span['begin'] = len(sentence) - 1
            else:
                arg_span['begin'] = arg_span['end'] -1
        
        arg_lstm = lstm_out_tok[arg_span['begin']:arg_span['end']]
        lstm_out_arg, self.hidden = self.lstm_arg(
            arg_lstm, self.hidden_lstm_arg)

        span_vec = lstm_out_arg[-1]
        
        # 5) argvec with FEvec and spanvec        
        arg_vec = torch.cat( (span_vec, fe_embs, position_feature), 1)
        
         # 6) CONCAT all vectors
        if USE_JOSA or USE_JOSA_POS:
            final_vec = torch.cat( (arg_vec, lu_vec, josa_embs), 1)
        else:
            final_vec = torch.cat( (arg_vec, lu_vec), 1)
            
        if USE_LAST_DP:
            final_vec = torch.cat( (final_vec, last_dp_embs), 1)
            

        # 7) linear
        tag_space = self.arg2hidden(final_vec)
        tag_space = F.relu(tag_space) 
        tag_space = self.hidden2tag(tag_space)
        
        # 8) masked softmax
        mask = m_softmax.gen_mask_frame(frame)
        tag_scores = m_softmax.masked_softmax(tag_space, mask)

        return tag_scores


# In[6]:


feature_extractor = feature_handler.extractor(language, josa_onlyPOS, usingGPU)
prepare = modelio.prepare(usingGPU)
m_softmax = masked_softmax.softmax(lufrmap, frargmap, frame_to_ix, fe_to_ix, usingGPU)


# In[10]:


def prepare_training_data_vocab(training_data):
    word_vocab_in_train, pos_vocab_in_train, frame_vocab_in_train, lu_vocab_in_train, fe_vocab_in_train = [],[],[],[], []
    for tokens in training_data:
        for t in tokens:
            lu = t[12]
            if lu != '_':
                lu_vocab_in_train.append(lu)
            frame = t[13]
            if frame != '_':
                frame_vocab_in_train.append(frame)
            bio_fe = t[14]
            if bio_fe != 'O':
                fe = bio_fe.split('-')[1]
    
    lu_vocab_in_train = list(set(lu_vocab_in_train))
    frame_vocab_in_train = list(set(frame_vocab_in_train))
    fe_vocab_in_train = list(set(fe_vocab_in_train))
    return lu_vocab_in_train, frame_vocab_in_train, fe_vocab_in_train
lu_vocab_in_train, frame_vocab_in_train, fe_vocab_in_train = prepare_training_data_vocab(training_data)


# In[12]:


prepare = modelio.prepare(usingGPU)
tensor2tag = modelio.tensor2tag(frame_to_ix, fe_to_ix, usingGPU)


class arg_identifier():
    def __init__(self):
        pass
    def identifier(self, conll, model):
        
        result = []
        
        tokens = conll
        sentence, pos, dp, lu, frame = prepare.prepare_sentence(tokens)
        if lu in lu_vocab_in_train:
            if frame in frame_vocab_in_train:
                sentence, pos, dp, lu, frame = prepare.prepare_sentence(tokens)
                target_position = feature_extractor.get_targetpositions(tokens)
                sentence_in = prepare.prepare_sequence(sentence, word_to_ix)
                pos_in = prepare.prepare_sequence(pos, pos_to_ix)
                dp_in = prepare.prepare_sequence(dp, dp_to_ix)
                positions = feature_extractor.get_argpositions(tokens)


                gold_spans = []
                for arg_position in positions:
                    arg = arg_position[2]
#                     arg_in = prepare.prepare_ix(arg, fe_to_ix)
                    arg_in = torch.tensor([0]).type(torch.cuda.LongTensor)
                    josa = feature_extractor.get_josa(tokens, arg_position)
                    last_dp = feature_extractor.get_last_dp(tokens, arg_position)

                    josa_in = prepare.prepare_ix(josa, josa_to_ix)
                    last_dp_in = prepare.prepare_ix(last_dp, dp_to_ix)

                    if arg_position[2] != 'O':
                        gold_span = {}
                        arg_span = {}
                        arg_span['begin'] = arg_position[0]
                        arg_span['end'] = arg_position[1]
                        gold_span['arg'] = arg
                        gold_span['span'] = arg_span
                        gold_span['arg_in'] = arg_in
                        gold_span['josa_in'] = josa_in
                        gold_span['last_dp_in'] = last_dp_in
                        gold_spans.append(gold_span)

                for gold_arg in gold_spans:                
                    model.zero_grad()
                    model.hidden_lstm_tok = model.init_hidden_lstm_tok()
                    model.hidden_lstm_tgt = model.init_hidden_lstm_tgt()
                    model.hidden_lstm_arg = model.init_hidden_lstm_arg()
                    arg_span = gold_arg['span']
                    arg_in = gold_arg['arg_in']
                    josa_in = gold_arg['josa_in']
                    last_dp_in = gold_arg['last_dp_in']
                    tag_scores = model(sentence_in, pos_in, dp_in, josa_in, last_dp_in, arg_in, target_position, arg_span, lu, frame, sentence)

                    
#                     print(tag_scores)

                    gold = gold_arg['arg']
                    score, pred = tensor2tag.get_fe_by_tensor(tag_scores)

                    tup = (arg_span, pred, score)
                    result.append(tup)
        return result

