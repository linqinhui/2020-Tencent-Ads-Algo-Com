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
import gc

csv.field_size_limit(20000001)
base_dir = "./"
LABEL = data.LabelField()
#LENGTH = data.Field(use_vocab=False,dtype=torch.long)
creative_id_TEXT = data.Field(sequential=True,lower=True,include_lengths=True,fix_length=100)
advertiser_id_TEXT = data.Field(sequential=True,lower=True,include_lengths=True,fix_length=100)
ad_id_TEXT = data.Field(sequential=True,lower=True,include_lengths=True,fix_length=100)
product_id_TEXT = data.Field(sequential=True,lower=True,include_lengths=True,fix_length=100)

all_data = data.TabularDataset(path=base_dir + "embedding/all_4features.csv",format='csv',skip_header=True,fields=[('creative_id', creative_id_TEXT),('ad_id', ad_id_TEXT),('advertiser_id', advertiser_id_TEXT),('product_id', product_id_TEXT),('gender', None),('age', LABEL)])
#train_data,val_data = data.TabularDataset.splits(path=base_dir + "embedding", train='train_3features.csv',validation='val_3features.csv', format='csv',skip_header=True,fields=[('creative_id', creative_id_TEXT),('ad_id', ad_id_TEXT),('advertiser_id', advertiser_id_TEXT),('product_id', product_id_TEXT),('gender', None),('age', LABEL)])
VAL_USER_ID = data.Field()
train_data = data.TabularDataset(path=base_dir + "embedding/train_4features_rd_swap_3.csv",format='csv',skip_header=True,fields=[('creative_id', creative_id_TEXT),('ad_id', ad_id_TEXT),('advertiser_id', advertiser_id_TEXT),('product_id', product_id_TEXT),('gender', None),('age', LABEL)])
val_data = data.TabularDataset(path=base_dir + "embedding/val_3features.csv",format='csv',skip_header=True,fields=[('user_id', VAL_USER_ID),('creative_id', creative_id_TEXT),('ad_id', ad_id_TEXT),('advertiser_id', advertiser_id_TEXT),('product_id', product_id_TEXT),('gender', None),('age', LABEL)])
VAL_USER_ID.build_vocab(val_data)

creative_id_custom_embeddings = vocab.Vectors(name = base_dir + 'embedding/creative_id_128.csv.vec',cache = 'creative_id_custom_embeddings',unk_init = torch.Tensor.normal_)
creative_id_size = 1530017
creative_id_TEXT.build_vocab(all_data,max_size = creative_id_size,vectors = creative_id_custom_embeddings)

advertiser_id_custom_embeddings = vocab.Vectors(name = base_dir + 'embedding/advertiser_id_128.csv.vec',cache = 'advertiser_id_custom_embeddings',unk_init = torch.Tensor.normal_)
advertiser_id_size = 46407
advertiser_id_TEXT.build_vocab(all_data,max_size = advertiser_id_size,vectors = advertiser_id_custom_embeddings)

ad_id_custom_embeddings = vocab.Vectors(name = base_dir + 'embedding/ad_id_128.csv.vec',cache = 'ad_id_custom_embeddings',unk_init = torch.Tensor.normal_)
ad_id_size = 1483551
ad_id_TEXT.build_vocab(all_data,max_size = ad_id_size,vectors = ad_id_custom_embeddings)

product_id_custom_embeddings = vocab.Vectors(name = base_dir + 'embedding/product_id_128.csv.vec',cache = 'product_id_custom_embeddings',unk_init = torch.Tensor.normal_)
product_id_size = 27749
product_id_TEXT.build_vocab(all_data,max_size = product_id_size,vectors = product_id_custom_embeddings)

LABEL.build_vocab(all_data)
BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

del all_data
gc.collect()

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, val_data),
    batch_size = BATCH_SIZE,
    sort=False,
    device = device)

del train_data,val_data,creative_id_custom_embeddings,advertiser_id_custom_embeddings,ad_id_custom_embeddings,product_id_custom_embeddings
gc.collect()

dim_config = {
    'creative_id': len(creative_id_TEXT.vocab),
    'advertiser_id': len(advertiser_id_TEXT.vocab),
    'ad_id': len(ad_id_TEXT.vocab),
    'product_id':len(product_id_TEXT.vocab)
}

pad_idx_dict = {
    'creative_id': creative_id_TEXT.vocab.stoi[creative_id_TEXT.pad_token],
    'advertiser_id': advertiser_id_TEXT.vocab.stoi[advertiser_id_TEXT.pad_token],
    'ad_id': ad_id_TEXT.vocab.stoi[ad_id_TEXT.pad_token],
    'product_id': product_id_TEXT.vocab.stoi[product_id_TEXT.pad_token]
}


class RNN(nn.Module):
    def __init__(self, dim_dict, embedding_dim, hidden_dim, kernel_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx_dict, kernels=(2, 3, 4)):
        super(RNN, self).__init__()
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.kernels = kernels
        self.dropout = nn.Dropout(dropout)
        flag = 1
        if bidirectional:
            flag = 2
        item_name = 'creative_id'
        self.creative_id_embedding = nn.Embedding(dim_dict[item_name], embedding_dim,
                                                  padding_idx=pad_idx_dict[item_name])
        self.creative_id_rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                                       dropout=dropout)
        self.creative_id_convs1 = nn.ModuleList(nn.Conv1d(flag * hidden_dim, kernel_dim, k) for k in kernels)
        self.creative_id_convs2 = nn.ModuleList(nn.Conv1d(flag * hidden_dim, kernel_dim, k) for k in kernels)
        self.creative_id_fc = nn.Linear(3 * kernel_dim, output_dim)
        item_name = 'advertiser_id'
        self.advertiser_id_embedding = nn.Embedding(dim_dict[item_name], embedding_dim,
                                                    padding_idx=pad_idx_dict[item_name])
        self.advertiser_id_rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                                         dropout=dropout)
        self.advertiser_id_convs1 = nn.ModuleList(nn.Conv1d(flag * hidden_dim, kernel_dim, k) for k in kernels)
        self.advertiser_id_convs2 = nn.ModuleList(nn.Conv1d(flag * hidden_dim, kernel_dim, k) for k in kernels)
        self.advertiser_id_fc = nn.Linear(3 * kernel_dim, output_dim)
        item_name = 'ad_id'
        self.ad_id_embedding = nn.Embedding(dim_dict[item_name], embedding_dim, padding_idx=pad_idx_dict[item_name])
        self.ad_id_rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                                 dropout=dropout)
        self.ad_id_convs1 = nn.ModuleList(nn.Conv1d(flag * hidden_dim, kernel_dim, k) for k in kernels)
        self.ad_id_convs2 = nn.ModuleList(nn.Conv1d(flag * hidden_dim, kernel_dim, k) for k in kernels)
        self.ad_id_fc = nn.Linear(3 * kernel_dim, output_dim)
        item_name = 'product_id'
        self.product_id_embedding = nn.Embedding(dim_dict[item_name], embedding_dim, padding_idx=pad_idx_dict[item_name])
        self.product_id_rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                                 dropout=dropout)
        self.product_id_convs1 = nn.ModuleList(nn.Conv1d(flag * hidden_dim, kernel_dim, k) for k in kernels)
        self.product_id_convs2 = nn.ModuleList(nn.Conv1d(flag * hidden_dim, kernel_dim, k) for k in kernels)
        self.product_id_fc = nn.Linear(3 * kernel_dim, output_dim)
        self.output_fc = nn.Linear(4 * output_dim, output_dim)
    #
    def pool(self, x):
        return F.max_pool1d(x, x.shape[2])
    #
    def forward(self, creative_id_text, creative_id_text_length, advertiser_id_text, advertiser_id_text_length,ad_id_text, ad_id_text_length, product_id_text, product_id_text_length):
        creative_id_embedded = self.creative_id_embedding(creative_id_text)
        creative_id_packed_embedded = nn.utils.rnn.pack_padded_sequence(creative_id_embedded, creative_id_text_length,enforce_sorted=False)
        creative_id_packed_output, creative_id_hidden = self.creative_id_rnn( creative_id_packed_embedded)
        creative_id_hidden, _ = nn.utils.rnn.pad_packed_sequence(creative_id_packed_output)
        creative_id_hidden = creative_id_hidden.permute(1, 2, 0)
        out1 = [self.relu(conv(creative_id_hidden)) for conv in self.creative_id_convs1]
        out2 = [self.tanh(conv(creative_id_hidden)) for conv in self.creative_id_convs2]
        out = torch.cat([self.pool(out1[i] * out2[i]).squeeze(2) for i in range(len(self.kernels))], dim=1)
        creative_id_fc = self.creative_id_fc(out)
        advertiser_id_embedded = self.advertiser_id_embedding(advertiser_id_text)
        advertiser_id_packed_embedded = nn.utils.rnn.pack_padded_sequence(advertiser_id_embedded,advertiser_id_text_length,enforce_sorted=False)
        advertiser_id_packed_output, advertiser_id_hidden = self.advertiser_id_rnn(advertiser_id_packed_embedded)
        advertiser_id_hidden, _ = nn.utils.rnn.pad_packed_sequence(advertiser_id_packed_output)
        advertiser_id_hidden = advertiser_id_hidden.permute(1, 2, 0)
        out1 = [self.relu(conv(advertiser_id_hidden)) for conv in self.advertiser_id_convs1]
        out2 = [self.tanh(conv(advertiser_id_hidden)) for conv in self.advertiser_id_convs2]
        out = torch.cat([self.pool(out1[i] * out2[i]).squeeze(2) for i in range(len(self.kernels))], dim=1)
        advertiser_id_fc = self.advertiser_id_fc(out)
        ad_id_embedded = self.ad_id_embedding(ad_id_text)
        ad_id_packed_embedded = nn.utils.rnn.pack_padded_sequence(ad_id_embedded, ad_id_text_length,enforce_sorted=False)
        ad_id_packed_output, ad_id_hidden = self.ad_id_rnn(ad_id_packed_embedded)
        ad_id_hidden, _ = nn.utils.rnn.pad_packed_sequence(ad_id_packed_output)
        ad_id_hidden = ad_id_hidden.permute(1, 2, 0)
        out1 = [self.relu(conv(ad_id_hidden)) for conv in self.ad_id_convs1]
        out2 = [self.tanh(conv(ad_id_hidden)) for conv in self.ad_id_convs2]
        out = torch.cat([self.pool(out1[i] * out2[i]).squeeze(2) for i in range(len(self.kernels))], dim=1)
        ad_id_fc = self.ad_id_fc(out)
        product_id_embedded = self.product_id_embedding(product_id_text)
        product_id_packed_embedded = nn.utils.rnn.pack_padded_sequence(product_id_embedded, product_id_text_length,enforce_sorted=False)
        product_id_packed_output, product_id_hidden = self.product_id_rnn(product_id_packed_embedded)
        product_id_hidden, _ = nn.utils.rnn.pad_packed_sequence(product_id_packed_output)
        product_id_hidden = product_id_hidden.permute(1, 2, 0)
        out1 = [self.relu(conv(product_id_hidden)) for conv in self.product_id_convs1]
        out2 = [self.tanh(conv(product_id_hidden)) for conv in self.product_id_convs2]
        out = torch.cat([self.pool(out1[i] * out2[i]).squeeze(2) for i in range(len(self.kernels))], dim=1)
        product_id_fc = self.product_id_fc(out)
        output_hidden = torch.cat((creative_id_fc, advertiser_id_fc, ad_id_fc,product_id_fc), dim=1)
        #output_hidden = torch.cat((creative_id_fc, advertiser_id_fc, ad_id_fc), dim=1)
        return self.output_fc(output_hidden)

EMBEDDING_DIM = 128
HIDDEN_DIM = 512
KERNEL_DIM=256
OUTPUT_DIM = 10
N_LAYERS = 3
BIDIRECTIONAL = True
DROPOUT = 0.5


#model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
model = RNN(dim_config,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            KERNEL_DIM,
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


model.product_id_embedding.weight.data.copy_(product_id_TEXT.vocab.vectors)
model.product_id_embedding.weight.data[product_id_TEXT.vocab.stoi[product_id_TEXT.unk_token]] = torch.zeros(EMBEDDING_DIM)
model.product_id_embedding.weight.data[product_id_TEXT.vocab.stoi[product_id_TEXT.pad_token]] = torch.zeros(EMBEDDING_DIM)

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
        product_id_text, product_id_text_length = batch.product_id
        predictions = model(creative_id_text,creative_id_text_length,advertiser_id_text, advertiser_id_text_length,ad_id_text, ad_id_text_length,product_id_text, product_id_text_length)
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
            product_id_text, product_id_text_length = batch.product_id
            predictions = model(creative_id_text, creative_id_text_length, advertiser_id_text, advertiser_id_text_length,ad_id_text, ad_id_text_length,product_id_text, product_id_text_length)
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

"""
N_EPOCHS = 10
FREEZE_FOR = 7
best_valid_loss = float('inf')
# freeze embeddings
#model.embedding.weight.requires_grad = unfrozen = False
model.creative_id_embedding.weight.requires_grad = unfrozen = False
model.advertiser_id_embedding.weight.requires_grad = unfrozen = False
model.ad_id_embedding.weight.requires_grad = unfrozen = False
model.product_id_embedding.weight.requires_grad = unfrozen = False
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
        torch.save(model.state_dict(), 'age-3features-model.pt')
    if (epoch + 1) >= FREEZE_FOR:
        # unfreeze embeddings
        #model.embedding.weight.requires_grad = unfrozen = True
        model.creative_id_embedding.weight.requires_grad = unfrozen = True
        model.advertiser_id_embedding.weight.requires_grad = unfrozen = True
        model.ad_id_embedding.weight.requires_grad = unfrozen = True
        model.product_id_embedding.weight.requires_grad = unfrozen = True

del model
gc.collect()
"""
model = RNN(dim_config,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            KERNEL_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            pad_idx_dict)
model.load_state_dict(torch.load('age-3features-model.pt'))
model = model.to(device)

USER_ID = data.Field()
test_data = data.TabularDataset(path=base_dir + "embedding/test_3features.csv", format='csv',skip_header=True,fields=[('user_id', USER_ID),('creative_id', creative_id_TEXT),('ad_id', ad_id_TEXT),('advertiser_id', advertiser_id_TEXT),('product_id', product_id_TEXT)])
USER_ID.build_vocab(test_data)
test_iterator = data.Iterator(
    test_data,
    batch_size = BATCH_SIZE,
    sort=False,
    sort_within_batch = False,
    device = device)

result_list = []
user_id_list = []
total_predictions = None
model.eval()
with torch.no_grad():
    for i, batch in enumerate(valid_iterator):
        print(i)
        creative_id_text, creative_id_text_length = batch.creative_id
        advertiser_id_text, advertiser_id_text_length = batch.advertiser_id
        ad_id_text, ad_id_text_length = batch.ad_id
        product_id_text, product_id_text_length = batch.product_id
        predictions = model(creative_id_text, creative_id_text_length, advertiser_id_text, advertiser_id_text_length,ad_id_text,
                            ad_id_text_length,product_id_text,product_id_text_length)
        predictions = predictions.to('cpu').numpy()
        if total_predictions is None:
            total_predictions = predictions
        else:
            total_predictions = np.vstack((total_predictions,predictions))
        user_id_list.extend(batch.user_id.squeeze(0).to('cpu').numpy())
result_df = pd.DataFrame(np.array([user_id_list]).reshape(-1,1),columns=["user_id_index"])
predict_df =pd.DataFrame(total_predictions)
result_df = pd.concat([result_df,predict_df],axis=1)
result_df["user_id"] = result_df["user_id_index"].apply(lambda x:VAL_USER_ID.vocab.itos[x])
result_df.to_csv(base_dir + "result/val_3features_age.csv",index=False)
print(LABEL.vocab.stoi)

result_list = []
user_id_list = []
total_predictions = None
model.eval()
with torch.no_grad():
    for i, batch in enumerate(test_iterator):
        print(i)
        creative_id_text, creative_id_text_length = batch.creative_id
        advertiser_id_text, advertiser_id_text_length = batch.advertiser_id
        ad_id_text, ad_id_text_length = batch.ad_id
        product_id_text, product_id_text_length = batch.product_id
        predictions = model(creative_id_text, creative_id_text_length, advertiser_id_text, advertiser_id_text_length,ad_id_text, ad_id_text_length,product_id_text,product_id_text_length)
        predictions = predictions.to('cpu').numpy()
        if total_predictions is None:
            total_predictions = predictions
        else:
            total_predictions = np.vstack((total_predictions,predictions))
        user_id_list.extend(batch.user_id.squeeze(0).to('cpu').numpy())

result_df = pd.DataFrame(np.array([user_id_list]).reshape(-1,1),columns=["user_id_index"])
predict_df =pd.DataFrame(total_predictions)
result_df = pd.concat([result_df,predict_df],axis=1)
result_df["user_id"] = result_df["user_id_index"].apply(lambda x:USER_ID.vocab.itos[x])
result_df.to_csv(base_dir + "result/test_3features_age.csv",index=False)


"""
gender_df = pd.read_csv(base_dir + "result/test_3features_gender.csv")[["user_id","predicted_gender"]]
gender_df["user_id"] = gender_df["user_id"].astype("str")
result_df = result_df.merge(gender_df,on="user_id",how="inner")
result_df[["user_id","predicted_gender","predicted_age"]].to_csv(base_dir + "result/test_3feature_age_gender_v2.csv",index=False)
"""

"""
gender_df = pd.read_csv(base_dir + "result/test_creative_id_gender_result_alldata.csv")
gender_df["user_id"] = gender_df["user_id"].astype("str")
result_df = result_df.merge(gender_df,on="user_id",how="inner")
result_df[["user_id","predicted_gender","predicted_age"]].to_csv(base_dir + "result/test_creative_id_result_alldata.csv",index=False)
"""
