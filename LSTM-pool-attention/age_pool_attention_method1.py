import pandas as pd
import numpy as np
#import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import spacy
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as vocab
import torch.optim as optim
import sys
import csv


csv.field_size_limit(20000001)
base_dir = "/content/drive/My Drive/Colab Notebooks/"
LABEL = data.LabelField()
#LENGTH = data.Field(use_vocab=False,dtype=torch.long)
creative_id_TEXT = data.Field(sequential=True,lower=True,include_lengths=True,fix_length=100)
advertiser_id_TEXT = data.Field(sequential=True,lower=True,include_lengths=True,fix_length=100)
ad_id_TEXT = data.Field(sequential=True,lower=True,include_lengths=True,fix_length=100)

all_data = data.TabularDataset(path=base_dir + "embedding/all_3features.csv",format='csv',skip_header=True,fields=[('creative_id', creative_id_TEXT),('ad_id', ad_id_TEXT),('advertiser_id', advertiser_id_TEXT),('gender', None),('age', LABEL)])


creative_id_custom_embeddings = vocab.Vectors(name = base_dir + 'embedding/creative_id_128.csv.vec',cache = 'creative_id_custom_embeddings',unk_init = torch.Tensor.normal_)
creative_id_size = 1530017
creative_id_TEXT.build_vocab(all_data,max_size = creative_id_size,vectors = creative_id_custom_embeddings)

advertiser_id_custom_embeddings = vocab.Vectors(name = base_dir + 'embedding/advertiser_id_128.csv.vec',cache = 'advertiser_id_custom_embeddings',unk_init = torch.Tensor.normal_)
advertiser_id_size = 46407
advertiser_id_TEXT.build_vocab(all_data,max_size = advertiser_id_size,vectors = advertiser_id_custom_embeddings)

ad_id_custom_embeddings = vocab.Vectors(name = base_dir + 'embedding/ad_id_128.csv.vec',cache = 'ad_id_custom_embeddings',unk_init = torch.Tensor.normal_)
ad_id_size = 1483551
ad_id_TEXT.build_vocab(all_data,max_size = ad_id_size,vectors = ad_id_custom_embeddings)

LABEL.build_vocab(all_data)
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

del all_data
del creative_id_custom_embeddings,advertiser_id_custom_embeddings,ad_id_custom_embeddings
import gc
gc.collect()

train_data,val_data = data.TabularDataset.splits(path=base_dir + "embedding", train='train_3features_reverse_rd.csv',validation='val_3features.csv', format='csv',skip_header=True,fields=[('creative_id', creative_id_TEXT),('ad_id', ad_id_TEXT),('advertiser_id', advertiser_id_TEXT),('gender', None),('age', LABEL)])

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, val_data),
    batch_size = BATCH_SIZE,
    sort=False,
    device = device)
del train_data,val_data
gc.collect()

dim_config = {
    'creative_id': len(creative_id_TEXT.vocab),
    'advertiser_id': len(advertiser_id_TEXT.vocab),
    'ad_id': len(ad_id_TEXT.vocab)
}

pad_idx_dict = {
    'creative_id': creative_id_TEXT.vocab.stoi[creative_id_TEXT.pad_token],
    'advertiser_id': advertiser_id_TEXT.vocab.stoi[advertiser_id_TEXT.pad_token],
    'ad_id': ad_id_TEXT.vocab.stoi[ad_id_TEXT.pad_token]
}
class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
        # apply attention layer
        #input = [batch size,sent len,  hid dim * num directions]
        #hid dim * num directions = hidden_size
        #input = [batch size,sent len,  hidden_size]
        #[batch size,sent len,  hidden_size] * [batch_size, hidden_size, 1]=[batch size,sent len,1]
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )
        #[batch size,sent len]
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)
        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        # apply mask and renormalize attention scores (weights)
        #masked [batch size,sent len]
        #_sums [batch size,1]
        #attentions [batch size,sent len]
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        attentions = masked.div(_sums)
        # apply attention weights
        #[batch size,sent len,  hidden_size] * [batch size,sent len，1]
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()
        return representations, attentions


class RNN(nn.Module):
    def __init__(self,dim_dict, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx_dict):
        super().__init__()

        item_name = 'creative_id'
        self.creative_id_embedding = nn.Embedding(dim_dict[item_name], embedding_dim, padding_idx=pad_idx_dict[item_name])
        self.creative_id_rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.creative_id_fc = nn.Linear(hidden_dim * 2*4, output_dim)
        self.creative_id_atten = Attention(hidden_dim * 2, batch_first=True)

        item_name = 'advertiser_id'
        self.advertiser_id_embedding = nn.Embedding(dim_dict[item_name], embedding_dim,padding_idx=pad_idx_dict[item_name])
        self.advertiser_id_rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,dropout=dropout)
        self.advertiser_id_fc = nn.Linear(hidden_dim * 2*4, output_dim)
        self.advertiser_id_atten = Attention(hidden_dim * 2, batch_first=True)

        item_name = 'ad_id'
        self.ad_id_embedding = nn.Embedding(dim_dict[item_name], embedding_dim,padding_idx=pad_idx_dict[item_name])
        self.ad_id_rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,dropout=dropout)
        self.ad_id_fc = nn.Linear(hidden_dim * 2*4, output_dim)
        self.ad_id_atten = Attention(hidden_dim * 2, batch_first=True)

        self.output_fc = nn.Linear(output_dim * 3, output_dim)
        self.dropout = nn.Dropout(dropout)

    def attention_net(self, lstm_output, hidden):
        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        #hidden = final_state.view(-1, hidden_dim * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self,creative_id_text,creative_id_text_length,advertiser_id_text, advertiser_id_text_length,ad_id_text, ad_id_text_length):
        creative_id_embedded = self.creative_id_embedding(creative_id_text)
        creative_id_packed_embedded = nn.utils.rnn.pack_padded_sequence(creative_id_embedded, creative_id_text_length, enforce_sorted=False)
        creative_id_packed_output, (creative_id_hidden, creative_id_cell) = self.creative_id_rnn(creative_id_packed_embedded)
        creative_id_output, creative_id_output_lengths = nn.utils.rnn.pad_packed_sequence(creative_id_packed_output)
        creative_id_avg_pool = F.adaptive_avg_pool1d(creative_id_output.permute(1,2,0),1).squeeze()
        creative_id_max_pool = F.adaptive_max_pool1d(creative_id_output.permute(1,2,0),1).squeeze()
        creative_id_hidden = torch.cat((creative_id_hidden[-2, :, :], creative_id_hidden[-1, :, :]), dim=1)
        #creative_id_attn_output, creative_id_attention = self.attention_net(creative_id_output.permute(1, 0, 2), creative_id_hidden)
        creative_id_attn_output, creative_id_attention = self.creative_id_atten(creative_id_output.permute(1, 0, 2), creative_id_output_lengths)
        creative_id_cat_fea = torch.cat([creative_id_hidden,creative_id_avg_pool,creative_id_max_pool,creative_id_attn_output],dim=1)
        creative_id_fc = self.creative_id_fc(creative_id_cat_fea)
        #creative_id_fc = F.relu(creative_id_fc)

        advertiser_id_embedded = self.advertiser_id_embedding(advertiser_id_text)
        advertiser_id_packed_embedded = nn.utils.rnn.pack_padded_sequence(advertiser_id_embedded, advertiser_id_text_length,enforce_sorted=False)
        advertiser_id_packed_output, (advertiser_id_hidden, advertiser_id_cell) = self.advertiser_id_rnn(advertiser_id_packed_embedded)
        advertiser_id_output, advertiser_id_output_lengths = nn.utils.rnn.pad_packed_sequence(advertiser_id_packed_output)
        advertiser_id_avg_pool = F.adaptive_avg_pool1d(advertiser_id_output.permute(1,2,0),1).squeeze()
        advertiser_id_max_pool = F.adaptive_max_pool1d(advertiser_id_output.permute(1,2,0),1).squeeze()
        advertiser_id_hidden = torch.cat((advertiser_id_hidden[-2, :, :], advertiser_id_hidden[-1, :, :]), dim=1)
        #advertiser_id_attn_output, advertiser_id_attention = self.attention_net(advertiser_id_output.permute(1, 0, 2), advertiser_id_hidden)
        advertiser_id_attn_output, advertiser_id_attention = self.advertiser_id_atten(advertiser_id_output.permute(1, 0, 2), advertiser_id_output_lengths)
        advertiser_id_cat_fea = torch.cat([advertiser_id_hidden,advertiser_id_avg_pool,advertiser_id_max_pool,advertiser_id_attn_output],dim=1)
        advertiser_id_fc = self.advertiser_id_fc(advertiser_id_cat_fea)
        #advertiser_id_fc = F.relu(advertiser_id_fc)


        ad_id_embedded = self.ad_id_embedding(ad_id_text)
        ad_id_packed_embedded = nn.utils.rnn.pack_padded_sequence(ad_id_embedded, ad_id_text_length,enforce_sorted=False)
        ad_id_packed_output, (ad_id_hidden, ad_id_cell) = self.ad_id_rnn(ad_id_packed_embedded)
        ad_id_output, ad_id_output_lengths = nn.utils.rnn.pad_packed_sequence(ad_id_packed_output)
        ad_id_avg_pool = F.adaptive_avg_pool1d(ad_id_output.permute(1,2,0),1).squeeze()
        ad_id_max_pool = F.adaptive_max_pool1d(ad_id_output.permute(1,2,0),1).squeeze()
        ad_id_hidden = torch.cat((ad_id_hidden[-2, :, :], ad_id_hidden[-1, :, :]), dim=1)
        #ad_id_attn_output, ad_id_attention = self.attention_net(ad_id_output.permute(1, 0, 2), ad_id_hidden)
        ad_id_attn_output, ad_id_attention = self.ad_id_atten(ad_id_output.permute(1, 0, 2), ad_id_output_lengths)
        ad_id_cat_fea = torch.cat([ad_id_hidden,ad_id_avg_pool,ad_id_max_pool,ad_id_attn_output],dim=1)
        ad_id_fc = self.ad_id_fc(ad_id_cat_fea)
        #ad_id_fc = F.tanh(ad_id_fc)

        output_hidden = torch.cat((creative_id_fc,advertiser_id_fc,ad_id_fc),dim=1)

        return self.output_fc(output_hidden)


#INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 10
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5


#model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
model = RNN(dim_config,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            pad_idx_dict)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


model.creative_id_embedding.weight.data.copy_(creative_id_TEXT.vocab.vectors)
model.creative_id_embedding.weight.data[creative_id_TEXT.vocab.stoi[creative_id_TEXT.unk_token]] = torch.zeros(EMBEDDING_DIM)
model.creative_id_embedding.weight.data[creative_id_TEXT.vocab.stoi[creative_id_TEXT.pad_token]] = torch.zeros(EMBEDDING_DIM)
model.advertiser_id_embedding.weight.data.copy_(advertiser_id_TEXT.vocab.vectors)
model.advertiser_id_embedding.weight.data[advertiser_id_TEXT.vocab.stoi[advertiser_id_TEXT.unk_token]] = torch.zeros(EMBEDDING_DIM)
model.advertiser_id_embedding.weight.data[advertiser_id_TEXT.vocab.stoi[advertiser_id_TEXT.pad_token]] = torch.zeros(EMBEDDING_DIM)
model.ad_id_embedding.weight.data.copy_(ad_id_TEXT.vocab.vectors)
model.ad_id_embedding.weight.data[ad_id_TEXT.vocab.stoi[ad_id_TEXT.unk_token]] = torch.zeros(EMBEDDING_DIM)
model.ad_id_embedding.weight.data[ad_id_TEXT.vocab.stoi[ad_id_TEXT.pad_token]] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)
def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        creative_id_text,creative_id_text_length = batch.creative_id
        advertiser_id_text, advertiser_id_text_length = batch.advertiser_id
        ad_id_text, ad_id_text_length = batch.ad_id
        predictions = model(creative_id_text,creative_id_text_length,advertiser_id_text, advertiser_id_text_length,ad_id_text, ad_id_text_length)
        loss = criterion(predictions, batch.age)
        acc = categorical_accuracy(predictions, batch.age)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            creative_id_text, creative_id_text_length = batch.creative_id
            advertiser_id_text, advertiser_id_text_length = batch.advertiser_id
            ad_id_text, ad_id_text_length = batch.ad_id
            predictions = model(creative_id_text, creative_id_text_length, advertiser_id_text, advertiser_id_text_length,ad_id_text, ad_id_text_length)
            loss = criterion(predictions, batch.age)
            acc = categorical_accuracy(predictions, batch.age)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
FREEZE_FOR = 5
best_valid_loss = float('inf')
# freeze embeddings
#model.embedding.weight.requires_grad = unfrozen = False
model.creative_id_embedding.weight.requires_grad = unfrozen = False
model.advertiser_id_embedding.weight.requires_grad = unfrozen = False
model.ad_id_embedding.weight.requires_grad = unfrozen = False
for epoch in range(N_EPOCHS):
    print(epoch)
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | Frozen? {not unfrozen}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'age-3features-pool-attention-0621-model.pt')
    if (epoch + 1) >= FREEZE_FOR:
        # unfreeze embeddings
        #model.embedding.weight.requires_grad = unfrozen = True
        model.creative_id_embedding.weight.requires_grad = unfrozen = True
        model.advertiser_id_embedding.weight.requires_grad = unfrozen = True
        model.ad_id_embedding.weight.requires_grad = unfrozen = True


