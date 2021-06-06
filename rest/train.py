import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
import random
import argparse
import numpy as np
from data import *
from model import Encoder,Decoder, BiRNN

USE_CUDA = torch.cuda.is_available()

# since in the code we suppose the hidden layer's shape is [1, B, D], 
# if we set num_layers > 1 for LSTM whose hidden layer's shape will be [>1, B, D] so that some continued operation could not be executed successfully.

def train(config):
    
    train_data, word2index, tag2index, intent2index = preprocessing(config.file_path,config.max_length)
    
    if train_data==None:
        print("Please check your data or its path")
        return
    
    #encoder = Encoder(len(word2index),config.embedding_size,config.hidden_size)
    #decoder = Decoder(len(tag2index),len(intent2index),len(tag2index)//3,config.hidden_size*2)
    biRNN = BiRNN(len(tag2index), len(intent2index), len(word2index), config.embedding_size, config.hidden_size)
    # slot_size, intent_size, input_size, embedding_size, hidden_size, batch_size=16 ,n_layers=1, dropout_p =0.1, attention_bool = True):

    if USE_CUDA:
        #encoder = encoder.cuda()
        #decoder = decoder.cuda()
        biRNN = biRNN.cuda()

    #encoder.init_weights()
    #decoder.init_weights()
    biRNN.init_weights()

    loss_function_1 = nn.CrossEntropyLoss(ignore_index=0) # Specifies a target value that is ignored and does not contribute to the input gradient
    loss_function_2 = nn.CrossEntropyLoss()
    #enc_optim= optim.Adam(encoder.parameters(), lr=config.learning_rate)
    #dec_optim = optim.Adam(decoder.parameters(),lr=config.learning_rate)
    biRNN_optim = optim.Adam(biRNN.parameters(),lr=config.learning_rate)
    
    for step in range(config.step_size):
        losses=[]
        for i, batch in enumerate(getBatch(config.batch_size,train_data)):
            x,y_1,y_2 = zip(*batch)
            x = torch.cat(x).cuda() if USE_CUDA else torch.cat(x)
            tag_target = torch.cat(y_1).cuda() if USE_CUDA else torch.cat(y_1)
            intent_target = torch.cat(y_2).cuda() if USE_CUDA else torch.cat(y_2)

            # True or False, depending on whether one should ignore the index (index == 0)
            x_mask = torch.cat([torch.tensor(tuple(map(lambda s: s == 0, t.data)), dtype=torch.bool) for t in x]) 
            x_mask = x_mask.view(config.batch_size,-1)

            #encoder.zero_grad()
            #decoder.zero_grad()
            biRNN.zero_grad()

            #output, hidden_c = encoder(x, x_mask)
            #start_decode = Variable(torch.LongTensor([[word2index['<SOS>']]*config.batch_size])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[word2index['<SOS>']]*config.batch_size])).transpose(1,0)
            # Example [[2], [2], [2], [2], [2]; provides the model with the right embedding for starting to classify (SOS = Start Of Sentence)

            #tag_score, intent_score = decoder(start_decode,hidden_c,output,x_mask)
            tag_score, intent_score = biRNN(x, x_mask)

            loss_1 = loss_function_1(tag_score,tag_target.view(-1))
            loss_2 = loss_function_2(intent_score,intent_target)

            loss = loss_1+loss_2
            losses.append(loss.data.cpu().numpy().item() if USE_CUDA else loss.data.numpy().item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(biRNN.parameters(), 5.0)
            #torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            #torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)

            #enc_optim.step()
            #dec_optim.step()
            biRNN_optim.step()

            if i % 100==0:
                print("Step",step," epoch",i," : ",np.mean(losses))
                losses=[]
    
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    
    #torch.save(decoder.state_dict(),os.path.join(config.model_dir,'jointnlu-decoder.pkl'))
    #torch.save(encoder.state_dict(),os.path.join(config.model_dir, 'jointnlu-encoder.pkl'))
    print("Train Complete!")
    
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/atis-2.train.w-intent.iob' ,
                        help='path of train data')
    parser.add_argument('--model_dir', type=str, default='./models/' ,
                        help='path for saving trained models')

    # Model parameters
    parser.add_argument('--max_length', type=int , default=60 ,
                        help='max sequence length')
    parser.add_argument('--embedding_size', type=int , default=64 ,
                        help='dimension of word embedding vectors') 
    parser.add_argument('--hidden_size', type=int , default=64 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    config = parser.parse_args()
    train(config)