
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        #self.embedding = nn.Embedding(input_size,embed_size)
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout,batch_first=True)

    def forward(self, input_seqs, hidden=None):
        '''
        :param input_seqs: 
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        #embedded = self.embedding(input_seqs)
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(input_seqs, hidden)
        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size]  # Sum bidirectional outputs
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (B,layers*directions,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (B,T,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)
        hidden = hidden.repeat(max_len,1,1)
        #H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(hidden,encoder_outputs) # compute attention score
                
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        #print(hidden.size(),encoder_outputs.size())
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + output_size, hidden_size, n_layers, dropout=dropout_p,batch_first=True)
        #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.bn = torch.nn.BatchNorm1d(1)

    def forward(self, word_input, last_hidden, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)
        word_input = word_input.view(word_input.size(0),1, -1) # (1,B,V)
        #word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        
        attn_weights = self.attn(last_hidden[-1], encoder_outputs).transpose(0,2)
        #print(attn_weights.size(),encoder_outputs.size())

        encoder_outputs = encoder_outputs
        context = attn_weights.bmm(encoder_outputs)  # (B,1,V)
        #print(context.size(),encoder_outputs.size(),word_input.size())

        # Combine embedded input word and attended context, run through RNN
        #print(context.size(),word_input.size())

        rnn_input = torch.cat((word_input, context), 2)
        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        #print(rnn_input.size(),last_hidden.size())
        output, hidden = self.gru(rnn_input, last_hidden[-1].unsqueeze(0))
        
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        output = self.bn(output)
        output = self.out(output).squeeze(1)
        #print(output.size())
        # Return final output, hidden state
        return output, hidden
