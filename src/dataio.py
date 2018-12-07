
# coding: utf-8

# In[14]:


import json
import os
from pathlib import Path


# In[15]:


data_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)+'/data/'


# In[16]:


def read_map(language):
    with open(data_path+language+'.lufrmap.json','r') as f:
        lufrmap = json.load(f)
    with open(data_path+language+'.frargmap.json','r') as f:
        frargmap = json.load(f)
    return lufrmap, frargmap


# In[17]:


def read_token2wv(language):
    with open(data_path+language+'.token2wv.json', 'r') as f:
        token2wv = json.load(f)
    return token2wv


# In[ ]:


def read_tsv(lines):
    result = []
    sent = []
    sent_ids = []
    for line in lines:
        line = line.rstrip('\n')
        if line.startswith('#'):
            if line[1] == 's':
                sent_id = line.split(':')[1]
                sent_ids.append(sent_id)                
            pass
        else:
            if line != '':
                token = line.split('\t')
                sent.append(token)
            else:
                result.append(sent)
                sent = []
    sent_num = len(list(set(sent_ids)))
    return result, sent_num

def data_stat(language, USE_EXAM=False):
    
    if language == 'ko':
        path = data_path+'kofn/'
        training_dir = path+'training.tsv'
        test_dir = path+'test.tsv'
        dev_dir = path+'dev.tsv'
        exemplar_dir = path+'examplar.tsv'
    elif language == 'en':
        path = data_path+'fn1.5/'
        training_dir = path+'fn1.5.fulltext.train.syntaxnet.conll'
        test_dir = path+'fn1.5.test.syntaxnet.conll'
        dev_dir = path+'fn1.5.dev.syntaxnet.conll'
        exemplar_dir = path+'fn1.5.exemplar.train.syntaxnet.conll'
    else:
        path = data_path+'kofn/'
        training_dir = path+'training.tsv'
        test_dir = path+'test.tsv'
        dev_dir = path+'dev.tsv'
        exemplar_dir = path+'examplar.tsv'
    
    with open(training_dir,'r') as f:
        d = f.readlines()
        training, n_training = read_tsv(d)
        if language == 'en':
            with open(path+'fn1.5.fulltext.train.syntaxnet.conll.sents','r') as f:
                d = f.readlines()
            n_training = len(d)
    with open(test_dir,'r') as f:
        d = f.readlines()
        test, n_test = read_tsv(d)
        if language == 'en':
            with open(path+'fn1.5.test.syntaxnet.conll.sents','r') as f:
                d = f.readlines()
            n_test = len(d)
    with open(dev_dir,'r') as f:
        d = f.readlines()
        dev, n_dev = read_tsv(d)
        if language == 'en':
            with open(path+'fn1.5.dev.syntaxnet.conll.sents','r') as f:
                d = f.readlines()
            n_dev = len(d)
    
    if USE_EXAM:
        with open(exemplar_dir,'r') as f:
            d = f.readlines()
            exemplar, n_exemplar = read_tsv(d)
            if language == 'en':
                with open(path+'fn1.5.exemplar.train.syntaxnet.conll.sents','r') as f:
                    d = f.readlines()
                n_exemplar = len(d)
    else:
        exemplar = []
        n_exemplar = 0
        
    print('# training_data')
    print(' - number of sentences:', n_training)
    print(' - number of annotations:', len(training), '\n')
    
    print('# test_data')
    print(' - number of sentences:', n_test)
    print(' - number of annotations:', len(test), '\n')
    
    print('# dev_data')
    print(' - number of sentences:', n_dev)
    print(' - number of annotations:', len(dev), '\n')
    
    if USE_EXAM:
        print('# exemplar data')
        print(' - number of sentences:', n_exemplar)
        print(' - number of annotations:', len(exemplar), '\n')
    else:
        pass

def read_data(language, USE_EXAM=False):

    if language == 'ko':
        path = data_path+'kofn/'
        training_dir = path+'training.tsv'
        test_dir = path+'test.tsv'
        dev_dir = path+'dev.tsv'
        exemplar_dir = path+'examplar.tsv'
    elif language == 'en':
        path = data_path+'fn1.5/'
        training_dir = path+'fn1.5.fulltext.train.syntaxnet.conll'
        test_dir = path+'fn1.5.test.syntaxnet.conll'
        dev_dir = path+'fn1.5.dev.syntaxnet.conll'
        exemplar_dir = path+'fn1.5.exemplar.train.syntaxnet.conll'
    else:
        path = data_path+'kofn/'
        training_dir = path+'training.tsv'
        test_dir = path+'test.tsv'
        dev_dir = path+'dev.tsv'
        exemplar_dir = path+'examplar.tsv'
    
    print('### loading data now...')
    with open(training_dir,'r') as f:
        d = f.readlines()
        training, n_training = read_tsv(d)

    with open(test_dir,'r') as f:
        d = f.readlines()
        test, n_test = read_tsv(d)

    with open(dev_dir,'r') as f:
        d = f.readlines()
        dev, n_dev = read_tsv(d)

    if USE_EXAM:
        with open(exemplar_dir,'r') as f:
            d = f.readlines()
            exemplar, n_exemplar = read_tsv(d)
    else:
        exemplar = []
        n_exemplar = 0
        
    return training, test, dev, exemplar


# In[ ]:


def prepare_idx(data, language, josa_onlyPOS=False):
    word_to_ix, pos_to_ix, frame_to_ix, lu_to_ix, fe_to_ix, dp_to_ix, josa_to_ix = {}, {}, {}, {}, {}, {}, {}
    
    word_to_ix['UNSEEN'] = 0
    fe_to_ix['O'] = 0
    josa_to_ix['null'] = 0
    dp_to_ix['null'] = 0
    for tokens in data:
        for t in tokens:
            if language == 'en':
                dp = t[5]
            elif language == 'ko':
                dp = t[10]
            if dp not in dp_to_ix:
                dp_to_ix[dp] = len(dp_to_ix)
            
            if language == 'ko':
                morphems = t[2].split('+')
                for m in morphems:
                    pos = m.split('/')[-1]
                    josa = ''
                    if pos.startswith('J'):
                        if josa_onlyPOS:
                            josa = pos
                        else:
                            josa = m
                        if len(josa) > 0:
                            if josa not in josa_to_ix:
                                josa_to_ix[josa] = len(josa_to_ix)
        for t in tokens:
            word = t[1]
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for t in tokens:
            pos = t[5]
            if pos not in pos_to_ix:
                pos_to_ix[pos] = len(pos_to_ix)
        for t in tokens:
            frame = t[13]
            if frame != '_':
                target_frame = frame
                if frame not in frame_to_ix:
                    frame_to_ix[frame] = len(frame_to_ix)
        for t in tokens:
            lu = t[12]
            if lu != '_':
                if lu not in lu_to_ix:
                    lu_to_ix[lu] = len(lu_to_ix)
        for t in tokens:
            fe_bio = t[14]
            if fe_bio != 'O':
                fe = fe_bio.split('-')[1]
                if fe not in fe_to_ix:
                    fe_to_ix[fe] = len(fe_to_ix)
    
    WORD_VOCAB_SIZE, POS_VOCAB_SIZE, DP_VOCAB_SIZE, JOSA_VOCAB_SIZE, LU_VOCAB_SIZE, FRAME_VOCAB_SIZE, FE_VOCAB_SIZE = len(word_to_ix), len(pos_to_ix), len(dp_to_ix), len(josa_to_ix), len(lu_to_ix), len(frame_to_ix), len(fe_to_ix)
    print('\n### VOCAB SIZE')
    print('# WORD_VOCAB_SIZE:', WORD_VOCAB_SIZE)
    print('# POS_VOCAB_SIZE:', POS_VOCAB_SIZE)
    print('# DP_VOCAB_SIZE:', DP_VOCAB_SIZE)
    if language == 'ko':
        print('# JOSA_VOCAB_SIZE:', JOSA_VOCAB_SIZE)
    print('# LU_VOCAB_SIZE:', LU_VOCAB_SIZE)    
    print('# FRAME_VOCAB_SIZE:', FRAME_VOCAB_SIZE)
    print('# FE_VOCAB_SIZE:', FE_VOCAB_SIZE)
    print('')

    return word_to_ix, pos_to_ix, dp_to_ix, josa_to_ix, frame_to_ix, lu_to_ix, fe_to_ix
    

