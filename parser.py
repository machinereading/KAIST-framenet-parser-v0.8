
# coding: utf-8

# In[1]:


from optparse import OptionParser
from src import etri
from src import targetid
from copy import deepcopy
from src import frameid_module_ko
from src import frameid_module_en
from src import argid_module_ko
import torch
import os
import json


# In[2]:


fid_model_ko_dir = './model/frameid-lstm-ko.pt'
fid_model_en_dir = './model/frameid-lstm-en.pt'
argid_model_ko_dir = './model/argid-lstm-ko.pt'


# In[3]:


optpr = OptionParser()
optpr.add_option("--mode", dest='mode', type='choice', choices=['train', 'test', 'parsing'], default='parsing')
optpr.add_option("--lang", dest='lang', type='choice', choices=['en', 'ko'], default='ko')
optpr.add_option("--text", dest='text')

# (options, args) = optpr.parse_args()
# parser_mode = options.mode
# language = options.lang

parser_mode = 'parsing'
language = 'ko'
print('\nMODE:', parser_mode)
print('LANGUAGE:', language)


# In[4]:


def read_nlp(nlp):
    conll= etri.getETRI_CoNLL2009(nlp)
    pa = etri.phrase_parser(conll, nlp)
    
    return conll, pa


# In[5]:


with open('./data/en.lufrmap.json') as f:
        enlu = json.load(f)

def target_identification(conll, language):
    
    result = []
    if language == 'ko':
        targets = targetid.target_identification_ko(conll)

        for target in targets:        

            target_id = target['token_id']
            target_lemma = target['lu']
            
            old_conll = deepcopy(conll)
            new_conll = []
            for token_id in range(len(old_conll)):
                token = old_conll[token_id]
                if target_id == token_id:
                    token.append(target_lemma)
                    token.append('TARGET')
                else:
                    token.append('_')
                    token.append('_')
                
            new_conll = deepcopy(old_conll)
            result.append(new_conll)
            new_conll = []
            
    else:
        old_conll = deepcopy(conll)

        targets = []
        for token_ix in range(len(old_conll)):
            token = old_conll[token_ix]
            pos = token[5]
            if pos.startswith('N'):
                pos = 'n'
            elif pos.startswith('V'):
                pos = 'v'
            elif pos.startswith('RB'):
                pos = 'a'        

            lem = token[3]+'.'+pos

            for lu in enlu:
                if lem == lu:
                    tu = (token_ix, lu)
                    targets.append(tu)
                    break
        for i in targets:
            target_ix, lu = i[0],i[1]

            new_conll = deepcopy(old_conll)

            for token_ix in range(len(new_conll)):

                token = new_conll[token_ix]
                if token_ix == target_ix:
                    token.append(lu)
                    token.append('TARGET')
                else:
                    token.append('_')
                    token.append('_')
            result.append(new_conll)
            
            
    return result


# In[6]:


from frameid_module_ko import LSTMTagger
fid_model = torch.load(fid_model_ko_dir)
ko_frame_ider = frameid_module_ko.frame_identifier()

def frame_identification(conll_with_target, language):
    if language == 'ko':
        result = []
        scores = []
        for conll in conll_with_target:
            
            frame, score = ko_frame_ider.identifier(conll, fid_model)
            
            if frame != '_':
                new_conll = deepcopy(conll)
                score = score.item()*0.1
                for token in new_conll:
                    if token[12] != '_':
                        token[13] = frame
                    else:
                        pass
                result.append(new_conll)
                scores.append(score)
    return result, scores      


# In[7]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from argid_module_ko import LSTMTagger
argid_model = torch.load(argid_model_ko_dir)
ko_arg_ider = argid_module_ko.arg_identifier()


def arg_identification(conll_with_frame, language):
    result = []
    if language == 'ko':
        new_conll_with_frame = deepcopy(conll_with_frame)

        for p in pa:
            target_id = p['predicate']['id']

            for sent in new_conll_with_frame:
                for token in sent:
                    if token[0] == target_id:
                        pass

        for sent in new_conll_with_frame:
            for token in sent:
                if token[13] != '_':
                    target_id = token[0]
                token.append('O')

            if target_id:
                for p in pa:
                    pred_id = p['predicate']['id']
                    if target_id == pred_id:
                        n = 0
                        for arg in p['arguments']:
                            arg_ids = arg['tokens']
                            for arg_ix in range(len(arg_ids)):
                                arg_id = arg_ids[arg_ix]

                                if arg_ix == 0:
                                    if len(arg_ids) == 1:
                                        arg_label = 'S-arg'+str(n)
                                    else:
                                        arg_label = 'B-arg'+str(n)
                                else:
                                    arg_label = 'I-arg'+str(n)


                                sent[arg_id][-1] = arg_label

                            n += 1
        for i in new_conll_with_frame:
            arg_id_result = ko_arg_ider.identifier(i, argid_model)
            result.append(arg_id_result)
    return result


# In[8]:


def triplization(text, conll_with_frame, frame_scores, arg_id_result):
    triples = []
    for sent_ix in range(len(conll_with_frame)):
        sent = conll_with_frame[sent_ix]
        frame = None
        for token in sent:
            if token[13] != '_':
                lu = token[12]
                frame = token[13]
        if frame != None:
            triple = (frame, 'frdf:lu', lu, frame_scores[sent_ix])
            triples.append(triple)
            
            args = arg_id_result[sent_ix]
            for arg in args:
                arg_span, arg_label, arg_score = arg[0],arg[1],arg[2]

                arg_score = arg_score.item()*0.1

                tokens = text.split(' ')
                arg_text = ' '.join(tokens[arg_span['begin']:arg_span['end']])
                triple = (frame, 'fe:'+arg_label, arg_text, arg_score)
                
                triples.append(triple)
                
    for t in triples:
        print(t)


# In[20]:


def korean_frame_parsing(nlp):
    conll_with_target = target_identification(conll, language)
    conll_with_frame, frame_scores = frame_identification(conll_with_target, language)
    arg_id_result = arg_identification(conll_with_frame, language)
    
#     print(arg_id_result)
    
    text = nlp[0]['text']
    triplization(text, conll_with_frame, frame_scores, arg_id_result)
    
    
    return arg_id_result

# text = '헤밍웨이는 고등학교를 마친 이후 이탈리아의 전방 군대에 입대하여 구급차 운전사가 되기 전에 캔자스시티스타에서 몇 달 동안 기사를 썼다.'
# nlp = etri.getETRI(text)
# conll, pa = read_nlp(nlp)
# result = korean_frame_parsing(nlp)


# In[30]:


from frameid_module_en import LSTMTagger
en_fid_model = torch.load(fid_model_en_dir)
en_frame_ider = frameid_module_en.frame_identifier()

def en_frame_identification():
    with open('./test.conll') as f:
        lines = f.readlines()
    
    conll_orig = []
    sent = []
    for line in lines:
        line = line.rstrip('\n')
        if line.startswith('#'):          
            pass
        else:
            if line != '':
                token = line.split('\t')
                sent.append(token)
            else:
                conll_orig.append(sent)
                sent = []
 
    conll = []
    for sent in conll_orig:
        
        new_sent = []
        token_ix = 1
        for token in sent:
            new_token = []
            new_token.append(str(token_ix))
            new_token.append(token[3])
            new_token.append('_')
            new_token.append(token[6])
            new_token.append(token[4])
            new_token.append(token[4])
            new_token.append('0')
            new_token.append('_')
            new_token.append('_')
            new_token.append('_')
            new_token.append('_')
            new_token.append('_')
            new_sent.append(new_token)
            
            token_ix += 1
            
        conll.append(new_sent)
    
    conll_with_target = []
    for sent in conll:

        d = target_identification(sent, 'en')
        conll_with_target.append(d[0])

        

    result = []
    scores = []
    for conll in conll_with_target:


        frame, score = en_frame_ider.identifier(conll, en_fid_model)
        if frame != '_':
            score = score.item()*0.1
            new_conll = deepcopy(conll)
            for token in new_conll:
                if token[12] != '_':
                    token[13] = frame
                else:
                    pass
            result.append(new_conll)
            scores.append(score)
    return result, scores     
        
            
# result, scores = en_frame_identification()
        
    

