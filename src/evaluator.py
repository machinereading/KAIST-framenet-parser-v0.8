
# coding: utf-8

# In[2]:


import torch
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import modelio
import feature_handler
import masked_softmax


# In[ ]:


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


# In[ ]:


class live(object):
    
    def __init__(self, language, word_to_ix, pos_to_ix, dp_to_ix, josa_to_ix, frame_to_ix, lu_to_ix, fe_to_ix, josa_onlyPOS, usingGPU):
        self.usingGPU = usingGPU 
        self.josa_onlyPOS = josa_onlyPOS
        self.language = language
        
        self.word_to_ix = word_to_ix
        self.pos_to_ix = pos_to_ix
        self.dp_to_ix = dp_to_ix
        self.josa_to_ix = josa_to_ix
        self.frame_to_ix = frame_to_ix
        self.lu_to_ix = lu_to_ix
        self.fe_to_ix = fe_to_ix
        
    def frame_evaluation():
        pass
    
    def argument_evaluation(self, model, training_data, test_data):
        
        feature_extractor = feature_handler.extractor(self.language, self.josa_onlyPOS, self.usingGPU)
        prepare = modelio.prepare(self.usingGPU) 
        tensor2tag = modelio.tensor2tag(self.frame_to_ix, self.fe_to_ix, self.usingGPU)
        
        lu_vocab_in_train, frame_vocab_in_train, fe_vocab_in_train = prepare_training_data_vocab(training_data)
        
        acc, total = 0,0
        tp,fn,tn,fp, found = 0,0,0,0,0
        for tokens in test_data:
            sentence, pos, dp, lu, frame = prepare.prepare_sentence(tokens)
            if lu in lu_vocab_in_train:
                if frame in frame_vocab_in_train:
                    sentence, pos, dp, lu, frame = prepare.prepare_sentence(tokens)
                    target_position = feature_extractor.get_targetpositions(tokens)
                    sentence_in = prepare.prepare_sequence(sentence, self.word_to_ix)
                    pos_in = prepare.prepare_sequence(pos, self.pos_to_ix)
                    dp_in = prepare.prepare_sequence(dp, self.dp_to_ix)
#                     frame_in = prepare.prepare_ix(frame, self.frame_to_ix)
#                     lu_in = prepare.prepare_ix(lu, self.lu_to_ix)
                    positions = feature_extractor.get_argpositions(tokens)


                    gold_spans = []
                    for arg_position in positions:
                        arg = arg_position[2]
                        arg_in = prepare.prepare_ix(arg, self.fe_to_ix)
                        josa = feature_extractor.get_josa(tokens, arg_position)
                        last_dp = feature_extractor.get_last_dp(tokens, arg_position)

                        josa_in = prepare.prepare_ix(josa, self.josa_to_ix)
                        last_dp_in = prepare.prepare_ix(last_dp, self.dp_to_ix)

#                         if arg_position[2] != 'O':
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


                        gold = gold_arg['arg']
                        score, pred = tensor2tag.get_fe_by_tensor(tag_scores)

    #                     print(gold, pred)
                        if pred != 'O':
                            if pred == gold:
                                tp += 1
                            else:
                                fp += 1
                        if gold != 'O':
                            if pred == 'O':
                                fn += 1
                            else: 
                                found += 1
                        if pred == gold:
                            acc += 1
                            total += 1
                        else:
                            total += 1
    #             break
        if tp == 0:
            precision = 0
        else:
            precision = tp / (tp+fp)
        if found == 0:
            recall = 0
        else:
            recall = found / (found+fn)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision*recall) / (precision+recall)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        if acc != 0:
            accuracy = acc / total
        else:
            accuracy = 0
            
        return precision, recall, f1, accuracy

