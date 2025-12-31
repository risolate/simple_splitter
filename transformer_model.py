import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiheadSelfAttention(nn.Module):
    def __init__(self,n_head,d_k,d_v,d_model,mask=None):
        super(MultiheadSelfAttention,self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.q = nn.Linear(d_model, d_k)
        self.k = nn.Linear(d_model, d_k)
        self.v = nn.Linear(d_model, d_v)
        self.e = nn.Linear(d_model, d_model)
        self.mask = mask

    def forward(self,query,key,value):  # B * C * T * d_model
        batch_size = query.shape[0]
        channel_size = query.shape[1]
        query_length, key_length, value_length = (
            query.shape[2],
            key.shape[2],
            value.shape[2],
        )
        query = self.q(query)
        query = query.view(batch_size,channel_size,query_length,self.n_head,-1).transpose(-3,-2)          # B * C * n_head * T * d_k/8
        key = self.k(key)
        key = key.view(batch_size,channel_size,key_length,self.n_head,-1).transpose(-3,-2)              # B * C * n_head * T * d_k/8
        value = self.v(value)
        value = value.view(batch_size,channel_size,value_length,self.n_head,-1).transpose(-3,-2)
        
        attention_score = query.matmul(key.transpose(-2,-1)) / math.sqrt(self.d_k)  # B * C * n_head * T * T
        if self.mask is not None:
            attention_score = attention_score.masked_fill_(self.mask == 0, -1 * 1e20)
        attention = F.softmax(attention_score,dim = -1)

        output = self.e(attention.matmul(value).transpose(-3,-2).contiguous().view(batch_size,channel_size,value_length,-1))
        return output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.selfattention = MultiheadSelfAttention(n_head, d_k, d_v, d_model)
        self.feedforward = nn.Sequential(nn.Linear(d_model, d_inner), nn.GELU(), nn.Linear(d_inner, d_model))

    def forward(self, input):
        x = self.selfattention(input, input, input)
        input = self.layerNorm1(x + input)
        input = self.dropout(input)
        x = self.feedforward(input)
        output = self.layerNorm2(x + input)
        output = self.dropout(output)
        return output


class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.layerNorm3 = nn.LayerNorm(d_model)
        self.selfattention = MultiheadSelfAttention(n_head, d_k, d_v, d_model)
        self.edattention = MultiheadSelfAttention(n_head, d_k, d_v, d_model)
        self.feedforward = nn.Sequential(nn.Linear(d_model, d_inner), nn.GELU(), nn.Linear(d_inner, d_model))

    def forward(self, input, key, value):
        x = self.selfattention(input, input, input)
        input = self.dropout(self.layerNorm1(x + input))

        x = self.edattention(input, key, value)
        input = self.dropout(self.layerNorm2(x + input))

        x = self.feedforward(input)
        output = self.dropout(self.layerNorm3(x + input))

        return output

class Decoder(nn.Module):
    def __init__(self, n_block, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super(Decoder, self).__init__()
        self.pos_embedding = 0
        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(n_head, d_k, d_v, d_model, d_inner, dropout) for _ in range(n_block)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, key, value):
        x = self.dropout(input)
        for layer in self.layers:
            x = layer(x, key, value)
        output = x
        return output


class Encoder(nn.Module):
    def __init__(self, n_block, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(n_head, d_k, d_v, d_model, d_inner, dropout) for _ in range(n_block)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # B * N * d_model
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        output = x
        return output

class PreNet(nn.Module):
    def __init__(self, d_model, n_fft = 1024, dropout = 0.1, n_channel = 2):
        super(PreNet, self).__init__()
        self.linear1 = nn.Linear((n_fft//2) +1, d_model)
        self.linear2 = nn.Linear(d_model,d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x):    # B * T * fq
        x = self.dropout1(F.relu(self.linear1(x)))
        x = self.dropout2(F.relu(self.linear2(x)))
        output = x
        return output

class PostNet(nn.Module):
    def __init__(self, d_model, n_fft = 1024, dropout = 0.1):
        super(PostNet, self).__init__()
        self.linear1 = nn.Linear(d_model,d_model)
        self.linear2 = nn.Linear(d_model,(n_fft//2) +1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x):    # B * T * fq
        x = self.dropout1(F.relu(self.linear1(x)))
        x = self.dropout2(F.relu(self.linear2(x)))
        output = x
        return output



class Transformer_E(nn.Module):
    def __init__(
        self,
        d_model=512,
        d_inner=2048,
        n_block=6,
        n_head=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
        n_fft = 1024,
    ):
        super(Transformer_E, self).__init__()

        self.preNet = PreNet(d_model,n_fft = n_fft)
        self.encoder = Encoder(n_block, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.postNet = PostNet(d_model,n_fft = n_fft)
        self.n_fft = n_fft

    def sinusoidal_positional_embedding(self, seq_len, dim):

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe    # len * d_model


    def forward(self, input_ids, attention_mask = None, labels = None): # B * C * fq * T

        x = self.preNet(input_ids.transpose(-1,-2))  # B * C * T * d_model
        pe = self.sinusoidal_positional_embedding(x.shape[-2], x.shape[-1])
        x = x + pe.to(x.device)
        x = self.encoder(x)
        x = self.postNet(x)
        x = x.transpose(-1,-2) # B * C * fq * T
        loss = None
        if labels is not None:
            batch_size = x.shape[0]
            loss_fc = nn.MSELoss()
            spec_label = torch.stft(labels,
                                    n_fft = self.n_fft,
                                    window=torch.hann_window(self.n_fft).to(labels),
                                    return_complex = True
                                    )
            spec_label = torch.view_as_real(spec_label).permute(0,3,1,2)
            loss = loss_fc(x.contiguous().view(batch_size,-1) ,spec_label.contiguous().view(batch_size,-1))
            return (loss, x)
        else:
            return (x,)

#bass #drums #other #vocals

class multi_transformer_E(nn.Module):
    def __init__(
        self,
        d_model=512,
        d_inner=2048,
        n_block=6,
        n_head=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
        n_fft = 1024,
    ):
        super(Transformer_E, self).__init__()

        self.preNet = PreNet(d_model, n_fft = n_fft)
        self.encoder = Encoder(n_block, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.vocalpostNet = PostNet(d_model, n_fft = n_fft)
        self.basspostNet = PostNet(d_model, n_fft = n_fft)
        self.drumpostNet = PostNet(d_model, n_fft = n_fft)
        self.otherpostNet = PostNet(d_model, n_fft = n_fft)

    def sinusoidal_positional_embedding(self, seq_len, dim):

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe    # len * d_model


    def forward(self, input_ids, attention_mask = None, labels = None): # B * mel * len

        x = self.preNet(input_ids.transpose(-1,-2))  # B * len * d_model
        pe = self.sinusoidal_positional_embedding(x.shape[-2], x.shape[-1])
        x = x + pe.to(x.device)
        x = self.encoder(x)
        x = self.postNet(x)
        x = x.transpose(-1,-2)
        loss = None
        if labels is not None:
            batch_size = x.shape[0]
            loss_fc = nn.MSELoss()
            mel_label = self.transform(labels)
            loss = loss_fc(x.contiguous().view(batch_size,-1) ,mel_label.contiguous().view(batch_size,-1))
            return (loss, x)
        else:
            return (x,)