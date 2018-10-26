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

def load_pretrained_embeddings(path_to_file, take):

    embedding_size = 300
    embedding_matrix = None
    lookup = {"<unk>": 0}
    c = 0
    with open(path_to_file, "r") as f:
        delimiter = " "
        for line in f:        
            if (take and c <= take) or not take:
                # split line
                line_split = line.rstrip().split(delimiter)
                # extract word and vector
                word = line_split[0]
                vector = np.array([float(i) for i in line_split[1:]])
                # get dimension of vector
                embedding_size = vector.shape[0]
                # add to lookup
                lookup[word] = c
                # add to embedding matrix
                if np.any(embedding_matrix):
                    embedding_matrix = np.vstack((embedding_matrix, vector))
                else:
                    embedding_matrix = np.zeros((2, embedding_size))
                    embedding_matrix[0] = np.random.uniform(-1, 1, embedding_size)  # for unk
                    embedding_matrix[1] = vector
                c += 1
    return embedding_matrix, lookup


def import_Treebank():

    with open('en-ud-train.conllu', 'r') as train_f:
        train_data = train_f.read()
        train_sentences = train_data.split('\n\n')

        train_words = []
        train_tags = []

        for train_l, train_sentence in enumerate(train_sentences):    
            train_lines = train_sentence.split('\n')
            train_sent_words = []
            train_sent_pos = []
            for train_line in train_lines:
                train_line = train_line.strip('\n') 

                if(len(train_line.split('\t')) >= 3):   # avoid empty line
                    train_word = train_line.split('\t')[1].lower()
                    train_pos = train_line.split('\t')[3]
                    train_sent_words.append(train_word)
                    train_sent_pos.append(train_pos)

            train_words.append(train_sent_words)
            train_tags.append(train_sent_pos)

    with open('en-ud-dev.conllu', 'r') as dev_f:
        dev_data = dev_f.read()
        dev_sentences = dev_data.split('\n\n')

        dev_words = []
        dev_tags = []

        for dev_l, dev_sentence in enumerate(dev_sentences):    
            dev_lines = dev_sentence.split('\n')
            dev_sent_words = []
            dev_sent_pos = []
            for dev_line in dev_lines:
                dev_line = dev_line.strip('\n') 

                if(len(dev_line.split('\t')) >= 3):   # avoid empty line
                    dev_word = dev_line.split('\t')[1].lower()
                    dev_pos = dev_line.split('\t')[3]
                    dev_sent_words.append(dev_word)
                    dev_sent_pos.append(dev_pos)

            dev_words.append(dev_sent_words)
            dev_tags.append(dev_sent_pos)


    return train_words, train_tags, dev_words, dev_tags



def import_it_happened():

    train_sents, _, dev_sents, _ = import_Treebank()
    with open('it-happened_eng_ud1.2_07092017.tsv', 'r') as f:
        data_dict = {}
        train_factuality = []
        train_sentence = []
        train_pred_token = []
        dev_factuality = []
        dev_sentence = []
        dev_pred_token = []


        for line in f:      
            fields = line.strip().split('\t')

            if (fields[0] != 'test' and fields[10] == 'True' and fields[9] != 'na' ):
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
            if(k.split('\t')[0] == 'train'):
                train_factuality.append(avg_factuality)
                train_pred_token.append(k.split('\t')[1])
                train_sentence_id = int(k.split('\t')[2].split(' ')[1]) - 1 
                train_sentence.append(train_sents[train_sentence_id])
            elif(k.split('\t')[0] == 'dev'):
                dev_factuality.append(avg_factuality)
                dev_pred_token.append(k.split('\t')[1])
                dev_sentence_id = int(k.split('\t')[2].split(' ')[1]) - 1 
                dev_sentence.append(dev_sents[dev_sentence_id])
            else:
                print ("error in data split!!!")

        return train_sentence, train_factuality, train_pred_token, dev_sentence, dev_factuality, dev_pred_token


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


def train():

    # i = epoch index
    # j = batch index
    # k = sentence index (inside batch j)
    # l = token index (inside sentence k)

    print("Epoch #\tTrain_MAE\tTrain_r\tDev_MAE\tDev_r\n") # Mean absolute error (MAE), Pearson correlation coefficient (r)

    for i in range(num_epochs):
        random.seed(i+100)
        random.shuffle(train_tokens) 
        random.seed(i+100)
        random.shuffle(train_facts) 
        random.seed(i+100)
        random.shuffle(train_pred_tokens) 

        train_predictions = []
        for j in range(num_batches_training):
            # begin a clean computational graph
            dy.renew_cg()
            # build the batch
            batch_tokens = train_tokens[j*batch_size:(j+1)*batch_size]
            batch_facts = train_facts[j*batch_size:(j+1)*batch_size]
            batch_pred_tokens = train_pred_tokens[j*batch_size:(j+1)*batch_size]

            # iterate through the batch
            for k in range(len(batch_tokens)):
                # prepare input: words to indexes
                seq_of_idxs = words2indexes(batch_tokens[k], w2i)
                preds = bidirect_pass(seq_of_idxs, int(batch_pred_tokens[k])-1)
                train_predictions.append(preds.npvalue())

                # calculate loss for each token in each example
                loss = dy.huber_distance(preds, dy.scalarInput(batch_facts[k]), c=1.0)

                # backpropogate the loss for the sentence
                loss.backward()
                trainer.update()


        dev_predictions = test()
        train_mae, train_r = evaluate(train_predictions, train_facts)
        dev_mae, dev_r = evaluate(dev_predictions, dev_facts)
        print("%d\t %.2f\t %.2f\t %.2f\t %.2f\n" % (i+1, train_mae, train_r, dev_mae, dev_r))
        model_name = 'trained_' + str(i) + '.model'
        RNN_model.save(model_name)

def test():

    all_predictions = []

    for j in range(num_batches_testing):
        # begin a clean computational graph
        dy.renew_cg()

        batch_tokens = dev_tokens[j*batch_size:(j+1)*batch_size]
        batch_facts = dev_facts[j*batch_size:(j+1)*batch_size]
        batch_pred_tokens = dev_pred_tokens[j*batch_size:(j+1)*batch_size]

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



train_tokens, train_facts, train_pred_tokens, dev_tokens, dev_facts, dev_pred_tokens = import_it_happened()

flat_train_tokens = []
for token in train_tokens:
    flat_train_tokens.extend(token)


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
emb_matrix_pretrained, w2i = load_pretrained_embeddings("glove.42B.300d.txt", take=10000)
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


################
# HYPERPARAMETER
################
trainer = dy.AdamTrainer(
    m=RNN_model
)
trainer.learning_rate = 0.001
batch_size = 1   # tune
num_epochs = 20
num_batches_testing = int(np.ceil(len(dev_tokens) / batch_size))
num_batches_training = int(np.ceil(len(train_tokens) / batch_size))
train()

# RNN_model.save("trained.model")
#save tables
tables = []
tables.append(emb_matrix_pretrained) 
tables.append(flat_train_tokens)
tables.append(w2i)

with open('tables.txt', "wb") as f:
    pickle.dump(tables, f, pickle.HIGHEST_PROTOCOL)
