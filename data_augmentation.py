import pandas as pd
import numpy as np
import fasttext
import os
#import lightgbm as lgb
import json
from sklearn.metrics import accuracy_score,confusion_matrix
import random
from random import shuffle
random.seed(1)

base_dir = "./"
click_log_df = pd.read_csv(base_dir + "train_preliminary/click_log.csv")
ad_df = pd.read_csv(base_dir + "train_preliminary/ad.csv")
click_log_df = click_log_df.merge(ad_df,on="creative_id",how="left")

#test_click_log_df = pd.read_csv(base_dir + "test/click_log.csv")
#test_ad_df = pd.read_csv(base_dir + "test/ad.csv")
#test_click_log_df = test_click_log_df.merge(test_ad_df,on="creative_id",how="left")
#all_data_df = pd.concat([click_log_df,test_click_log_df],axis=0)
train_df = click_log_df[click_log_df["user_id"]<=675242]
user_df = pd.read_csv(base_dir + "train_preliminary/user.csv")

creative_id_fasttext_model = fasttext.load_model(base_dir + "embedding/{}_{}.csv.bin".format('creative_id',128))
ad_id_fasttext_model = fasttext.load_model(base_dir + "embedding/{}_{}.csv.bin".format('ad_id',128))
advertiser_id_fasttext_model = fasttext.load_model(base_dir + "embedding/{}_{}.csv.bin".format('advertiser_id',128))


creative_id_df = train_df["creative_id"].value_counts().reset_index()
creative_id_array = creative_id_df[creative_id_df["creative_id"]>=200]["index"].values

creative_id_synonyms_dict = {}
for i,id in enumerate(creative_id_array):
    if i % 100 == 0:
        print(i)
    synonyms_word_list = creative_id_fasttext_model.get_nearest_neighbors(str(id))
    for j,(word_score,word) in enumerate(synonyms_word_list):
        if float(word_score) >= 0.85:
            if j == 0:
                creative_id_synonyms_dict[str(id)] = [word]
            else:
                creative_id_synonyms_dict[str(id)].append(word)
        else:
            break

f=open("creative_id_dict","w")
json.dump(creative_id_synonyms_dict,f)
f.close()

ad_id_df = train_df["ad_id"].value_counts().reset_index()
ad_id_array = ad_id_df[ad_id_df["ad_id"]>=200]["index"].values
ad_id_synonyms_dict = {}
for i,id in enumerate(ad_id_array):
    if i % 100 == 0:
        print(i)
    synonyms_word_list = ad_id_fasttext_model.get_nearest_neighbors(str(id))
    for j,(word_score,word) in enumerate(synonyms_word_list):
        if float(word_score) >= 0.85:
            if j == 0:
                ad_id_synonyms_dict[str(id)] = [word]
            else:
                ad_id_synonyms_dict[str(id)].append(word)
        else:
            break

f=open("ad_id_dict","w")
json.dump(ad_id_synonyms_dict,f)
f.close()

advertiser_id_df = train_df["advertiser_id"].unique()
advertiser_id_synonyms_dict = {}
for i,id in enumerate(advertiser_id_df):
    if i % 100 == 0:
        print(i)
    synonyms_word_list = advertiser_id_fasttext_model.get_nearest_neighbors(str(id))
    for j,(word_score,word) in enumerate(synonyms_word_list):
        if float(word_score) >= 0.9:
            if j == 0:
                advertiser_id_synonyms_dict[str(id)] = [word]
            else:
                advertiser_id_synonyms_dict[str(id)].append(word)
        else:
            break

f=open("advertiser_id_dict","w")
json.dump(advertiser_id_synonyms_dict,f)
f.close()


f=open("creative_id_dict","r")
creative_id_synonyms_dict = json.load(f)
f.close()
f=open("ad_id_dict","r")
ad_id_synonyms_dict = json.load(f)
f.close()
f=open("advertiser_id_dict","r")
advertiser_id_synonyms_dict = json.load(f)
f.close()


def generate_sequence(df,id_name):
    df[id_name] = df[id_name].astype("str")
    user_time_group_df = df.groupby(["user_id", "time"])[id_name].agg(" ".join).reset_index()
    user_group_df = user_time_group_df.groupby("user_id")[id_name].agg(" ".join).reset_index()
    return user_group_df

target_id_list =[ 'creative_id', 'ad_id','advertiser_id','product_id']
total_train_user_embedding_df = None
for target_id in target_id_list:
    print("start {} embedding!".format(target_id))
    train_user_embedding_df = generate_sequence(train_df, target_id)
    if total_train_user_embedding_df is None:
        total_train_user_embedding_df = train_user_embedding_df
    else:
        total_train_user_embedding_df = total_train_user_embedding_df.merge(train_user_embedding_df,on="user_id",how="left")

user_id_df = total_train_user_embedding_df[["user_id"]]

#reverse
def reverse_seq(input_seq):
    return " ".join(input_seq.split()[::-1])

reverse_df = total_train_user_embedding_df[["user_id"]]
target_id_list =[ 'creative_id', 'ad_id','advertiser_id','product_id']
for target_id in target_id_list:
    print(target_id)
    reverse_df['creative_id'] = total_train_user_embedding_df['creative_id'].apply(lambda row:reverse_seq(row))
    reverse_df['ad_id'] = total_train_user_embedding_df['ad_id'].apply(lambda row: reverse_seq(row))
    reverse_df['advertiser_id'] = total_train_user_embedding_df['advertiser_id'].apply(lambda row: reverse_seq(row))
    reverse_df['product_id'] = total_train_user_embedding_df['product_id'].apply(lambda row: reverse_seq(row))

#random delete
random_deletion_df = total_train_user_embedding_df[["user_id"]]
def random_deletion(words_str, p=0.2):
    words = words_str.split()
    if len(words) == 1:
        return words
    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]
    return " ".join(new_words)

target_id_list =[ 'creative_id', 'ad_id','advertiser_id','product_id']
for target_id in target_id_list:
    print(target_id)
    random_deletion_df['creative_id'] = total_train_user_embedding_df['creative_id'].apply(lambda row:random_deletion(row))
    random_deletion_df['ad_id'] = total_train_user_embedding_df['ad_id'].apply(lambda row: random_deletion(row))
    random_deletion_df['advertiser_id'] = total_train_user_embedding_df['advertiser_id'].apply(lambda row: random_deletion(row))
    random_deletion_df['product_id'] = total_train_user_embedding_df['product_id'].apply(lambda row: random_deletion(row))

#random swap
def random_swap(words_str, n=3):
    words = words_str.split()
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

random_swap_df = total_train_user_embedding_df[["user_id"]]
target_id_list =[ 'creative_id', 'ad_id','advertiser_id']
for target_id in target_id_list:
    print(target_id)
    random_swap_df['creative_id'] = total_train_user_embedding_df['creative_id'].apply(lambda row:random_swap(row))
    random_swap_df['ad_id'] = total_train_user_embedding_df['ad_id'].apply(lambda row: random_swap(row))
    random_swap_df['advertiser_id'] = total_train_user_embedding_df['advertiser_id'].apply(lambda row: random_swap(row))


#random synonym replace
def synonym_replacement(words_str, model,n=3):
    words = words_str.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word,model)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break
    return  ' '.join(new_words)

"""
def get_synonyms(word,model):
    synonyms = set()
    if model == "creative_id":
        synonyms_word_list = creative_id_fasttext_model.get_nearest_neighbors(word)
    elif model == "ad_id":
        synonyms_word_list = ad_id_fasttext_model.get_nearest_neighbors(word)
    elif model == "advertiser_id":
        synonyms_word_list = advertiser_id_fasttext_model.get_nearest_neighbors(word)
    else:
        return []
    for (word_score,word) in synonyms_word_list:
        if float(word_score) >= 0.85:
            synonyms.add(word)
        else:
            break
    return list(synonyms)
"""

def get_synonyms(word,model):
    synonyms_word_list = None
    if model == "creative_id":
        synonyms_word_list = creative_id_synonyms_dict.get(str(word))
        return synonyms_word_list if synonyms_word_list is not None else []
    elif model == "ad_id":
        synonyms_word_list = ad_id_synonyms_dict.get(str(word))
        return synonyms_word_list if synonyms_word_list is not None else []
    elif model == "advertiser_id":
        synonyms_word_list = advertiser_id_synonyms_dict.get(str(word))
        return synonyms_word_list if synonyms_word_list is not None else []
    else:
        return []

random_replace_df = total_train_user_embedding_df[["user_id"]]
target_id_list =[ 'creative_id', 'ad_id','advertiser_id']
for target_id in target_id_list:
    print(target_id)
    random_replace_df['creative_id'] = total_train_user_embedding_df['creative_id'].apply(lambda row:synonym_replacement(row,'creative_id'))
    random_replace_df['ad_id'] = total_train_user_embedding_df['ad_id'].apply(lambda row: synonym_replacement(row,'ad_id'))
    random_replace_df['advertiser_id'] = total_train_user_embedding_df['advertiser_id'].apply(lambda row: synonym_replacement(row,'advertiser_id'))

#random synonym insert
def random_insertion(words_str,model, n=3):
    words = words_str.split()
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words,model)
    return new_words

def add_word(new_words,model):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word,model)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

random_insert_df = total_train_user_embedding_df[["user_id"]]
target_id_list =[ 'creative_id', 'ad_id','advertiser_id']
for target_id in target_id_list:
    print(target_id)
    random_insert_df['creative_id'] = total_train_user_embedding_df['creative_id'].apply(lambda row:random_insertion(row,'creative_id'))
    random_insert_df['ad_id'] = total_train_user_embedding_df['ad_id'].apply(lambda row: random_insertion(row,'ad_id'))
    random_insert_df['advertiser_id'] = total_train_user_embedding_df['advertiser_id'].apply(lambda row: random_insertion(row,'advertiser_id'))

new_df = pd.concat([total_train_user_embedding_df,reverse_df,random_deletion_df],axis=0)
train_join_df = user_df.merge(new_df,on="user_id",how="inner")
train_join_df[['creative_id', 'ad_id','advertiser_id','product_id','gender','age']].to_csv(base_dir+"embedding/train_4features_rd_swap_3.csv", index=False)


new_df = pd.concat([total_train_user_embedding_df,random_swap_df],axis=0)
train_join_df = user_df.merge(new_df,on="user_id",how="inner")
train_join_df[['creative_id', 'ad_id','advertiser_id','gender','age']].to_csv(base_dir+"embedding/train_3features_rd_swap_3.csv", index=False)

new_df = pd.concat([total_train_user_embedding_df,random_replace_df],axis=0)
train_join_df = user_df.merge(new_df,on="user_id",how="inner")
train_join_df[['creative_id', 'ad_id','advertiser_id','gender','age']].to_csv(base_dir+"embedding/train_3features_rp_3.csv", index=False)

new_df = pd.concat([total_train_user_embedding_df,random_insert_df],axis=0)
train_join_df = user_df.merge(new_df,on="user_id",how="inner")
train_join_df[['creative_id', 'ad_id','advertiser_id','gender','age']].to_csv(base_dir+"embedding/train_3features_ri_3.csv", index=False)

