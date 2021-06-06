import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchcrf import CRF

USE_CUDA = torch.cuda.is_available()

class Encoder(nn.Module):
    def __init__(self, input_size,embedding_size, hidden_size,batch_size=16 ,n_layers=1):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size=batch_size
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True,bidirectional=True)
        # Set forget gate bias of LSTM to 1
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        #self.lstm.weight.data.
    
    def init_hidden(self,input):
        hidden = Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size)).cuda() if USE_CUDA else Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size))
        context = Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size)).cuda() if USE_CUDA else Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size))
        return (hidden,context)
     
    def forward(self, input,input_masking):
        """
        input : B,T (LongTensor)
        input_masking : B,T (PAD 마스킹한 ByteTensor)
        
        <PAD> 제외한 리얼 Context를 다시 만들어서 아웃풋으로
        """
        
        self.hidden = self.init_hidden(input)
        
        embedded = self.embedding(input)
        output, self.hidden = self.lstm(embedded, self.hidden)
        
        real_context=[]
        
        for i,o in enumerate(output): # B,T,D
            real_length = input_masking[i].data.tolist().count(0) # 실제 길이
            real_context.append(o[real_length-1])
            
        return output, torch.cat(real_context).view(input.size(0),-1).unsqueeze(1)

class Decoder(nn.Module):
    
    def __init__(self,slot_size,intent_size,embedding_size,hidden_size,batch_size=16,n_layers=1,dropout_p=0.5):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, self.embedding_size) #TODO encoder와 공유하도록 하고 학습되지 않게..

        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.embedding_size+self.hidden_size*2, self.hidden_size, self.n_layers, batch_first=True)
        self.attn = nn.Linear(self.hidden_size,self.hidden_size) # Attention
        self.slot_out = nn.Linear(self.hidden_size*2, self.slot_size)
        self.intent_out = nn.Linear(self.hidden_size*2,self.intent_size)
        # Set forget gate bias of LSTM to 1
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        #self.out.bias.data.fill_(0)
        #self.out.weight.data.uniform_(-0.1, 0.1)
        #self.lstm.weight.data.
    
    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        """
        
        hidden = hidden.squeeze(0).unsqueeze(2)  # 히든 : (1,배치,차원) -> (배치,차원,1)
        encoder_maskings = encoder_maskings.cuda() if USE_CUDA else encoder_maskings
        
        batch_size = encoder_outputs.size(0) # B
        max_len = encoder_outputs.size(1) # T
        energies = self.attn(encoder_outputs.contiguous().view(batch_size*max_len,-1)) # B*T,D -> B*T,D
        energies = energies.view(batch_size,max_len,-1) # B,T,D (배치,타임,차원)
        attn_energies = energies.bmm(hidden).transpose(1,2) # B,T,D * B,D,1 --> B,1,T
        attn_energies = attn_energies.squeeze(1).masked_fill(encoder_maskings,-1e12) # PAD masking
        
        alpha = F.softmax(attn_energies, dim=1) # B,T
        alpha = alpha.unsqueeze(1) # B,1,T
        context = alpha.bmm(encoder_outputs) # B,1,T * B,T,D => B,1,D
        
        return context # B,1,D

    def init_hidden(self,input):
        hidden = Variable(torch.zeros(self.n_layers*1, input.size(0), self.hidden_size)).cuda() if USE_CUDA else Variable(torch.zeros(self.n_layers*1,input.size(0), self.hidden_size))
        context = Variable(torch.zeros(self.n_layers*1, input.size(0), self.hidden_size)).cuda() if USE_CUDA else Variable(torch.zeros(self.n_layers*1, input.size(0), self.hidden_size))
        return (hidden,context)
    
    def forward(self, input,context,encoder_outputs,encoder_maskings,training=True):
        """
        input : B,L(length)
        enc_context : B,1,D
        """
        # Get the embedding of the current input word
        embedded = self.embedding(input)
        hidden = self.init_hidden(input)
        decode=[]
        aligns = encoder_outputs.transpose(0,1)
        length = encoder_outputs.size(1)
        for i in range(length): # Input_sequence와 Output_sequence의 길이가 같기 때문..
            aligned = aligns[i].unsqueeze(1)# B,1,D
            _, hidden = self.lstm(torch.cat((embedded,context,aligned),2), hidden) # input, context, aligned encoder hidden, hidden
            
            # for Intent Detection
            if i==0: 
                intent_hidden = hidden[0].clone() 
                intent_context = self.Attention(intent_hidden, encoder_outputs,encoder_maskings) 
                concated = torch.cat((intent_hidden,intent_context.transpose(0,1)),2) # 1,B,D
                intent_score = self.intent_out(self.dropout(concated.squeeze(0))) # B,D

            concated = torch.cat((hidden[0],context.transpose(0,1)),2)
            score = self.slot_out(self.dropout(concated.squeeze(0)))
            softmaxed = F.log_softmax(score, dim=1)
            decode.append(softmaxed)
            _,input = torch.max(softmaxed,1)
            embedded = self.embedding(input.unsqueeze(1))
            
            # 그 다음 Context Vector를 Attention으로 계산
            context = self.Attention(hidden[0], encoder_outputs,encoder_maskings) 
        # 요고 주의! time-step을 column-wise concat한 후, reshape!!
        slot_scores = torch.cat(decode,1)
        return slot_scores.view(input.size(0)*length,-1), intent_score


class CRFDecoder(nn.Module):
    def __init__(self, slot_size, intent_size, hidden_size, batch_size=16, n_layers=1, dropout_p=0.5):
        super(CRFDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.n_layers = n_layers
        self.batch_size = batch_size

        # Define the layers
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(self.hidden_size*2, self.hidden_size, self.n_layers, batch_first=True)
        self.attn = nn.Linear(self.hidden_size,self.hidden_size) # Attention
        self.slot_out = nn.Linear(self.hidden_size*2, self.slot_size)
        self.intent_out = nn.Linear(self.hidden_size*2,self.intent_size)
        self.crf = CRF(self.slot_size)

        # Set forget gate bias of LSTM to 1
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        """

        hidden = hidden.squeeze(0).unsqueeze(2)  # 히든 : (1,배치,차원) -> (배치,차원,1)
        encoder_maskings = encoder_maskings.cuda() if USE_CUDA else encoder_maskings

        batch_size = encoder_outputs.size(0)  # B
        max_len = encoder_outputs.size(1)  # T
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1))  # B*T,D -> B*T,D
        energies = energies.view(batch_size, max_len, -1)  # B,T,D (배치,타임,차원)
        attn_energies = energies.bmm(hidden).transpose(1, 2)  # B,T,D * B,D,1 --> B,1,T
        attn_energies = attn_energies.squeeze(1).masked_fill(encoder_maskings, -1e12)  # PAD masking

        alpha = F.softmax(attn_energies, dim=1)  # B,T
        alpha = alpha.unsqueeze(1)  # B,1,T
        context = alpha.bmm(encoder_outputs)  # B,1,T * B,T,D => B,1,D

        return context  # B,1,D

    def calculate_state(self, context, encoder_outputs, encoder_maskings):
        hidden = None
        aligns = encoder_outputs.transpose(0, 1)
        length = encoder_outputs.size(1)
        emissions = []
        for i in range(length):  # Input_sequence와 Output_sequence의 길이가 같기 때문..
            aligned = aligns[i].unsqueeze(1)  # B,1,D
            _, hidden = self.lstm(torch.cat((context, aligned), 2),
                                  hidden)  # input, context, aligned encoder hidden, hidden

            # for Intent Detection
            if i == 0:
                intent_hidden = hidden[0].clone()
                intent_context = self.Attention(intent_hidden, encoder_outputs, encoder_maskings)
                concated = torch.cat((intent_hidden, intent_context.transpose(0, 1)), 2)  # 1,B,D
                intent_score = self.intent_out(self.dropout(concated.squeeze(0))) # B,D

            concated = torch.cat((hidden[0], context.transpose(0, 1)), 2)
            score = self.slot_out(self.dropout(concated.squeeze(0)))
            emissions.append(score)

            # 그 다음 Context Vector를 Attention으로 계산
            context = self.Attention(hidden[0], encoder_outputs, encoder_maskings)
        slot_emissions = torch.stack(emissions)
        return slot_emissions, intent_score

    def forward(self, context, encoder_outputs, encoder_maskings, labels):
        slot_emissions, intent_score = self.calculate_state(context, encoder_outputs, encoder_maskings)
        return -self.crf(slot_emissions, labels, reduction='mean'), intent_score

    def predict(self, context, encoder_outputs, encoder_maskings):
        slot_emissions, intent_score = self.calculate_state(context, encoder_outputs, encoder_maskings)
        return self.crf.decode(slot_emissions), intent_score
