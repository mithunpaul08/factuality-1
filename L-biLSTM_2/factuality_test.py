import numpy as np
import string, random
from collections import Counter
import getopt, sys
import pickle 
import dynet_config
dynet_config.set(
    mem=16384,
    autobatch=True,      # utilize autobatching
    random_seed=1978     # simply for reproducibility here
)
import dynet as dy   
import datetime
import math


def import_Treebank():

    ### with open('en-ud-dev.conllu', 'r') as test_f:
    with open('en-ud-test.conllu', 'r') as test_f:
        test_data = test_f.read()
        test_sentences = test_data.split('\n\n')

        test_words = []
        test_tags = []

        for test_l, test_sentence in enumerate(test_sentences):    
            test_lines = test_sentence.split('\n')
            test_sent_words = []
            test_sent_pos = []
            for test_line in test_lines:
                test_ine = test_line.strip('\n') 

                if(len(test_line.split('\t')) >= 3):   # avoid empty line
                    test_word = test_line.split('\t')[1].lower()
                    test_pos = test_line.split('\t')[3]
                    test_sent_words.append(test_word)
                    test_sent_pos.append(test_pos)

            test_words.append(test_sent_words)
            test_tags.append(test_sent_pos)

    return test_words, test_tags


def import_it_happened():

    test_sents, _ = import_Treebank()
    with open('it-happened_eng_ud1.2_07092017.tsv', 'r') as f:
        data_dict = {}
        test_factuality = []
        test_sentence = []
        test_pred_token = []

        for line in f:      
            fields = line.strip().split('\t')

            ### if (fields[0] == 'dev' and fields[10] == 'True' and fields[9] != 'na' ):
            if (fields[0] == 'test' and fields[10] == 'True' and fields[9] != 'na' ):
                entry_key = fields[0] + '\t' + fields[4] + '\t' + fields[3]
                if (fields[8] == 'false'):
                    factuality = -1 * 3 / 4 * float(fields[9])
                elif(fields[8] == 'true'):
                    factuality = 3 / 4 * float(fields[9])
                else:
                   print ("error!!!")

                data_dict.setdefault(entry_key,[]).append(factuality)

        for k, v in zip(data_dict.keys(), data_dict.values()):
            avg_factuality = sum(v) / len(v)
            data_dict[k] = avg_factuality
            if(k.split('\t')[0] == 'test'):
            ### if(k.split('\t')[0] == 'dev'):
                test_factuality.append(avg_factuality)
                test_pred_token.append(k.split('\t')[1])
                test_sentence_id = int(k.split('\t')[2].split(' ')[1]) - 1 
                test_sentence.append(test_sents[test_sentence_id])
            else:
                print ("error in data split!!!")

        return test_sentence, test_factuality, test_pred_token 

def words2indexes(seq_of_words, w2i_lookup):
    """
    This function converts our sentence into a sequence of indexes that correspond to the rows in our embedding matrix
    :param seq_of_words: the document as a <list> of words
    :param w2i_lookup: the lookup table of {word:index} that we built earlier
    """
    seq_of_idxs = []
    for w in seq_of_words:
        i = w2i_lookup.get(w, 0) # we use the .get() method to allow for default return value if the word is not found
                                 # we've reserved the 0th row of embedding matrix for out-of-vocabulary words
        seq_of_idxs.append(i)
    return seq_of_idxs

def bidirect_pass(x, p):
    """
    This function will wrap all the steps needed to feed one sentence through the biLSTM
    :param x: a <list> of indices
    """
    # convert sequence of ints to sequence of embeddings
    #input_seq = [embedding_parameters[i] for i in x]   # embedding_parameters can be used like <dict>
    input_seq = [dy.lookup(embedding_parameters, i, update=False) for i in x]   # embedding_parameters can be used like <dict>
    
    # convert Parameters to Expressions
    v1 = dy.parameter(pv1)
    b1 = dy.parameter(pb1)
    v2 = dy.parameter(pv2)
    b2 = dy.parameter(pb2)

    # initialize the RNN unit
    fw_rnn_seq = fw_RNN_unit.initial_state()
    bw_rnn_seq = bw_RNN_unit.initial_state()

    # run each timestep(word) through the RNN
    fw_rnn_hidden_outs = fw_rnn_seq.transduce(input_seq)
    bw_rnn_hidden_outs = bw_rnn_seq.transduce(reversed(input_seq))

    second_input_seq = [dy.concatenate([f,b]) for f,b in zip(fw_rnn_hidden_outs, reversed(bw_rnn_hidden_outs))]

    second_fw_rnn_seq = second_fw_RNN_unit.initial_state()
    second_bw_rnn_seq = second_bw_RNN_unit.initial_state()

    fw_rnn_second_hidden_outs = second_fw_rnn_seq.transduce(second_input_seq)
    bw_rnn_second_hidden_outs = second_bw_rnn_seq.transduce(reversed(second_input_seq))


    # biLSTM states
    bi = [dy.concatenate([f,b]) for f,b in zip(fw_rnn_second_hidden_outs, reversed(bw_rnn_second_hidden_outs))]
    # hidden_state at the position of predicate
    bi_pred = bi[p]

    # a two-layer regression model
    outputs = dy.dot_product(v2, dy.tanh(v1 * bi_pred  + b1)) + b2

    return outputs



def test():

    all_predictions = []

    for j in range(num_batches_testing):
        # begin a clean computational graph
        dy.renew_cg()

        batch_tokens = test_tokens[j*batch_size:(j+1)*batch_size]
        batch_facts = test_facts[j*batch_size:(j+1)*batch_size]
        batch_pred_tokens = test_pred_tokens[j*batch_size:(j+1)*batch_size]

        # iterate through the batch
        for k in range(len(batch_tokens)):
            # prepare input: words to indexes
            seq_of_idxs = words2indexes(batch_tokens[k], w2i)
            preds = bidirect_pass(seq_of_idxs, int(batch_pred_tokens[k])-1)
            all_predictions.append(preds.npvalue())
    return all_predictions


def evaluate(preds_fac, true_fac):

    mae = 0
    true_sum = 0
    pred_sum = 0
    n = len(true_fac)
    for i in range(len(true_fac)):
        mae = mae + abs(preds_fac[i] - float(true_fac[i]))
        true_sum = true_sum + float(true_fac[i])
        pred_sum = pred_sum + preds_fac[i]

    mae = mae / n
    true_mean = true_sum / n
    pred_mean = pred_sum / n


    sum_prod = 0
    true_square_sum = 0
    pred_square_sum = 0
    for i in range(len(true_fac)):
        sum_prod = sum_prod + (float(true_fac[i]) - true_mean) * (preds_fac[i] - pred_mean)
        true_square_sum = true_square_sum + (float(true_fac[i]) - true_mean) * (float(true_fac[i]) - true_mean) 
        pred_square_sum = pred_square_sum + (preds_fac[i] - pred_mean) * (preds_fac[i] - pred_mean)

    r = sum_prod / math.sqrt(true_square_sum) / math.sqrt(pred_square_sum) 

    return mae,r


test_tokens, test_facts, test_pred_tokens = import_it_happened()

#load tables
with open("tables.txt", 'rb') as model:     
    content = pickle.load(model)
    emb_matrix_pretrained = content[0]
    flat_train_tokens = content[1]
    w2i = content[2]

#initialize empty model
RNN_model = dy.ParameterCollection()    

################
# HYPERPARAMETER
################
hidden_size = 300
# number of layers in `lstm`
num_layers = 1

regression_hidden_size = 300

#pretrained embeddings
embedding_dim = emb_matrix_pretrained.shape[1]
embedding_parameters = RNN_model.lookup_parameters_from_numpy(emb_matrix_pretrained)

#add RNN unit
fw_RNN_unit = dy.LSTMBuilder(num_layers, embedding_dim, hidden_size, RNN_model)
bw_RNN_unit = dy.LSTMBuilder(num_layers, embedding_dim, hidden_size, RNN_model)

second_fw_RNN_unit = dy.LSTMBuilder(num_layers, 2*hidden_size, hidden_size, RNN_model)
second_bw_RNN_unit = dy.LSTMBuilder(num_layers, 2*hidden_size, hidden_size, RNN_model)
 
pv1 = RNN_model.add_parameters(
        (regression_hidden_size, 2*hidden_size))
dy.parameter(pv1).npvalue().shape

 
pb1 = RNN_model.add_parameters(
        (regression_hidden_size)        
)
dy.parameter(pb1).npvalue().shape



pv2 = RNN_model.add_parameters(
        (regression_hidden_size))
dy.parameter(pv2).npvalue().shape

 
pb2 = RNN_model.add_parameters(
        (1)        
)
dy.parameter(pb2).npvalue().shape



RNN_model.populate("trained.model")

batch_size = 1
num_batches_testing = int(np.ceil(len(test_tokens) / batch_size))
predictions = test()

mae, r = evaluate(predictions, test_facts)

print("Mean absolute error (MAE): %.2f\n" % (mae))
print("Pearson correlation coefficient (r): %.2f\n" % (r))

with open('test_predictions.txt', 'w') as m: 
    m.write("Test:   prediction_factuality\tTrue_factuality\n")
    for p,t in zip(predictions, test_facts):
        m.write("%s\t%s\n" % (str(p), str(t)))