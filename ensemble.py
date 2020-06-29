import pandas as pd
import numpy as np
#import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
#import spacy
import torch
#from torchtext import data, datasets
#from torchtext.vocab import Vectors
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchtext.vocab as vocab
import torch.optim as optim
import sys
import csv
from sklearn.metrics import accuracy_score,confusion_matrix

base_dir = "./"
age_df_1 = pd.read_csv(base_dir + "ensemble/lin_val_3features_age.csv")
age_df_1.sort_values(by="user_id",axis=0,inplace=True)
age_df_1 = age_df_1.reset_index(drop=True)
age_df_1["user_id"] = age_df_1["user_id"].astype("str")

age_df_2 = pd.read_csv(base_dir + "ensemble/val_3features_age_attention_prob.csv")
age_df_2.sort_values(by="user_id",axis=0,inplace=True)
age_df_2 = age_df_2.reset_index(drop=True)
age_df_2["user_id"] = age_df_2["user_id"].astype("str")

user_df = pd.read_csv(base_dir + "train_preliminary/user.csv")
user_df["user_id"] = user_df["user_id"].astype("str")

age_df_3 = pd.read_csv(base_dir + "ensemble/val_3features_age_attention_method3_prob_0621.csv")
age_df_3.sort_values(by="user_id",axis=0,inplace=True)
age_df_3 = age_df_3.reset_index(drop=True)
age_df_3["user_id"] = age_df_3["user_id"].astype("str")

age_df_4 = pd.read_csv(base_dir + "ensemble/lin_val_3features_age_0622.csv")
age_df_4.sort_values(by="user_id",axis=0,inplace=True)
age_df_4 = age_df_4.reset_index(drop=True)
age_df_4["user_id"] = age_df_4["user_id"].astype("str")

stoi_dict_1 = {'3': 0, '4': 1, '2': 2, '5': 3, '6': 4, '7': 5, '1': 6, '8': 7, '9': 8, '10': 9}
itos_dict_1 = dict((str(value),key) for key,value in stoi_dict_1.items())
age_df_1["max_predict_index"] = age_df_1[["0","1","2","3","4","5","6","7","8","9"]].idxmax(axis=1)
age_df_1["max_predict"]=age_df_1["max_predict_index"].apply(lambda x:int(itos_dict_1[str(x)]))
age_df_1 = age_df_1.merge(user_df, on="user_id", how="inner")
print(accuracy_score(age_df_1["max_predict"], age_df_1["age"]))

stoi_dict_2 = {'1': 6,'10': 9,'2': 2,'3': 0, '4': 1,'5': 3,'6': 4,'7': 5,'8': 7,'9': 8}
itos_dict_2 = ['3', '4', '2', '5', '6', '7', '1', '8', '9', '10']
age_df_2["max_predict_index"] = age_df_2[["0","1","2","3","4","5","6","7","8","9"]].idxmax(axis=1)
age_df_2["max_predict"]=age_df_2["max_predict_index"].apply(lambda x:int(itos_dict_2[int(x)]))
age_df_2 = age_df_2.merge(user_df, on="user_id", how="inner")
print(accuracy_score(age_df_2["max_predict"], age_df_2["age"]))

stoi_dict_3 = {'1': 6,'10': 9,'2': 2,'3': 0, '4': 1,'5': 3,'6': 4,'7': 5,'8': 7,'9': 8}
itos_dict_3 = ['3', '4', '2', '5', '6', '7', '1', '8', '9', '10']
age_df_3["max_predict_index"] = age_df_3[["0","1","2","3","4","5","6","7","8","9"]].idxmax(axis=1)
age_df_3["max_predict"]=age_df_3["max_predict_index"].apply(lambda x:int(itos_dict_3[int(x)]))
age_df_3 = age_df_3.merge(user_df, on="user_id", how="inner")
print(accuracy_score(age_df_3["max_predict"], age_df_3["age"]))

age_df_4["max_predict_index"] = age_df_4[["0","1","2","3","4","5","6","7","8","9"]].idxmax(axis=1)
age_df_4["max_predict"]=age_df_4["max_predict_index"].apply(lambda x:int(itos_dict_3[int(x)]))
age_df_4 = age_df_4.merge(user_df, on="user_id", how="inner")
print(accuracy_score(age_df_4["max_predict"], age_df_3["age"]))

X_1 = age_df_1[["0","1","2","3","4","5","6","7","8","9"]].values
X_1 = np.exp(X_1)
X_sum_1 = np.sum(X_1,axis=1)
age_df_1[["0","1","2","3","4","5","6","7","8","9"]] = pd.DataFrame((X_1 / X_sum_1.reshape(-1,1)))


X_2 = age_df_2[["0","1","2","3","4","5","6","7","8","9"]].values
X_2 = np.exp(X_2)
X_sum_2 = np.sum(X_2,axis=1)
age_df_2[["0","1","2","3","4","5","6","7","8","9"]] = pd.DataFrame((X_2 / X_sum_2.reshape(-1,1)))

X_3 = age_df_3[["0","1","2","3","4","5","6","7","8","9"]].values
X_3 = np.exp(X_3)
X_sum_3 = np.sum(X_3,axis=1)
age_df_3[["0","1","2","3","4","5","6","7","8","9"]] = pd.DataFrame((X_3 / X_sum_3.reshape(-1,1)))

X_4 = age_df_4[["0","1","2","3","4","5","6","7","8","9"]].values
X_4 = np.exp(X_4)
X_sum_4 = np.sum(X_4,axis=1)
age_df_4[["0","1","2","3","4","5","6","7","8","9"]] = pd.DataFrame((X_4 / X_sum_4.reshape(-1,1)))

max_score = -1
best_weight_1,best_weight_2,best_weight_3,best_weight_4 = 0.0,0.0,0.0,0.0
for i in range(0,11):
    for j in range(11-i):
        for m in range(11-i-j):
            weight_1 = i * 0.1
            weight_2 = j * 0.1
            weight_3 = m * 0.1
            weight_4 = 1 - weight_1 - weight_2 - weight_3
            print(weight_1, weight_2,weight_3,weight_4)
            merge_df = age_df_2[['user_id']].reset_index(drop=True)
            merge_df[["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]] = pd.DataFrame(
                age_df_1[["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]].values * weight_1 + age_df_2[
                    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]].values * weight_2+age_df_3[
                    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]].values * weight_3+age_df_4[
                    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]].values * weight_4)
            merge_df["max_predict_index"] = merge_df[["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]].idxmax(axis=1)
            # merge_df = pd.concat([age_df_2[['user_id']].sort_values(by="user_id",axis=0),merge_df],axis=1)
            merge_df["max_predict"] = merge_df["max_predict_index"].apply(lambda x: int(itos_dict_2[int(x)]))
            merge_df = merge_df.merge(user_df, on="user_id", how="inner")
            score = accuracy_score(merge_df["max_predict"], merge_df["age"])
            print(score)
            if max_score <= score:
                max_score = score
                best_weight_1, best_weight_2, best_weight_3,best_weight_4 = weight_1, weight_2,weight_3,weight_4



base_dir = "./"
age_df_1 = pd.read_csv(base_dir + "ensemble/lin_test_3features_age.csv")
age_df_1.sort_values(by="user_id",axis=0,inplace=True)
age_df_1 = age_df_1.reset_index(drop=True)
age_df_1["user_id"] = age_df_1["user_id"].astype("str")
age_df_2 = pd.read_csv(base_dir + "ensemble/test_3features_age_attention_prob.csv")
age_df_2.sort_values(by="user_id",axis=0,inplace=True)
age_df_2 = age_df_2.reset_index(drop=True)
age_df_2["user_id"] = age_df_2["user_id"].astype("str")
age_df_3 = pd.read_csv(base_dir + "ensemble/test_3features_age_attention_method3_prob_0621.csv")
age_df_3.sort_values(by="user_id",axis=0,inplace=True)
age_df_3 = age_df_3.reset_index(drop=True)
age_df_3["user_id"] = age_df_3["user_id"].astype("str")
age_df_4 = pd.read_csv(base_dir + "ensemble/lin_test_3features_age_0622.csv")
age_df_4.sort_values(by="user_id",axis=0,inplace=True)
age_df_4 = age_df_4.reset_index(drop=True)
age_df_4["user_id"] = age_df_4["user_id"].astype("str")

stoi_dict_1 = {'3': 0, '4': 1, '2': 2, '5': 3, '6': 4, '7': 5, '1': 6, '8': 7, '9': 8, '10': 9}
itos_dict_1 = dict((str(value),key) for key,value in stoi_dict_1.items())
age_df_1["max_predict_index"] = age_df_1[["0","1","2","3","4","5","6","7","8","9"]].idxmax(axis=1)
age_df_1["max_predict"]=age_df_1["max_predict_index"].apply(lambda x:int(itos_dict_1[str(x)]))

stoi_dict_2 = {'1': 6,'10': 9,'2': 2,'3': 0, '4': 1,'5': 3,'6': 4,'7': 5,'8': 7,'9': 8}
itos_dict_2 = ['3', '4', '2', '5', '6', '7', '1', '8', '9', '10']
age_df_2["max_predict_index"] = age_df_2[["0","1","2","3","4","5","6","7","8","9"]].idxmax(axis=1)
age_df_2["max_predict"]=age_df_2["max_predict_index"].apply(lambda x:int(itos_dict_2[int(x)]))

stoi_dict_3 = {'1': 6,'10': 9,'2': 2,'3': 0, '4': 1,'5': 3,'6': 4,'7': 5,'8': 7,'9': 8}
itos_dict_3 = ['3', '4', '2', '5', '6', '7', '1', '8', '9', '10']
age_df_3["max_predict_index"] = age_df_3[["0","1","2","3","4","5","6","7","8","9"]].idxmax(axis=1)
age_df_3["max_predict"]=age_df_3["max_predict_index"].apply(lambda x:int(itos_dict_3[int(x)]))

age_df_4["max_predict_index"] = age_df_4[["0","1","2","3","4","5","6","7","8","9"]].idxmax(axis=1)
age_df_4["max_predict"]=age_df_4["max_predict_index"].apply(lambda x:int(itos_dict_3[int(x)]))


X_1 = age_df_1[["0","1","2","3","4","5","6","7","8","9"]].values
X_1 = np.exp(X_1)
X_sum_1 = np.sum(X_1,axis=1)
age_df_1[["0","1","2","3","4","5","6","7","8","9"]] = pd.DataFrame((X_1 / X_sum_1.reshape(-1,1)))


X_2 = age_df_2[["0","1","2","3","4","5","6","7","8","9"]].values
X_2 = np.exp(X_2)
X_sum_2 = np.sum(X_2,axis=1)
age_df_2[["0","1","2","3","4","5","6","7","8","9"]] = pd.DataFrame((X_2 / X_sum_2.reshape(-1,1)))

X_3 = age_df_3[["0","1","2","3","4","5","6","7","8","9"]].values
X_3 = np.exp(X_3)
X_sum_3 = np.sum(X_3,axis=1)
age_df_3[["0","1","2","3","4","5","6","7","8","9"]] = pd.DataFrame((X_3 / X_sum_3.reshape(-1,1)))

X_4 = age_df_4[["0","1","2","3","4","5","6","7","8","9"]].values
X_4 = np.exp(X_4)
X_sum_4 = np.sum(X_4,axis=1)
age_df_4[["0","1","2","3","4","5","6","7","8","9"]] = pd.DataFrame((X_4 / X_sum_4.reshape(-1,1)))

weight_1, weight_2, weight_3,weight_4 = 0.2,0.3,0.3,0.2
merge_df = age_df_2[['user_id']].reset_index(drop=True)
merge_df[["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]] = pd.DataFrame(
                age_df_1[["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]].values * weight_1 + age_df_2[
                    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]].values * weight_2+age_df_3[
                    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]].values * weight_3+age_df_4[
                    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]].values * weight_4)
merge_df["max_predict_index"] = merge_df[["0","1","2","3","4","5","6","7","8","9"]].idxmax(axis=1)
#merge_df = pd.concat([age_df_2[['user_id']].sort_values(by="user_id",axis=0),merge_df],axis=1)
merge_df["max_predict"]=merge_df["max_predict_index"].apply(lambda x:int(itos_dict_2[int(x)]))
#accuracy_score(merge_df["max_predict"],merge_df["age"])



result_df=  pd.read_csv(base_dir + "result/test_3feature_age_gender_v2.csv")
result_df['user_id'] = result_df['user_id'].astype('str')
result_df = result_df.rename(columns={"predicted_gender":"old_predicted_gender","predicted_age":"old_predicted_age"})
#result_df.sort_values(by="user_id",axis=0,inplace=True)
#new_result_df = pd.concat([result_df.reindex(),merge_df.reindex()],axis=1)
new_result_df = result_df.merge(merge_df,on="user_id",how="inner")
accuracy_score(new_result_df["old_predicted_age"],new_result_df["max_predict"])
new_gender_df = pd.read_csv(base_dir+"result/test_3features_gender_pool_attention_method3_0621.csv")
new_gender_df['user_id'] = new_gender_df['user_id'].astype('str')
new_result_df = new_result_df.merge(new_gender_df,on="user_id",how="inner")
accuracy_score(new_result_df["old_predicted_gender"],new_result_df["predicted_gender"])
#new_gender_prob_df = pd.read_csv(base_dir+"ensemble/test_3features_gender_pool_attention_method3_0621_prob.csv")
#new_gender_prob_df['user_id'] = new_gender_prob_df['user_id'].astype('str')
#new_gender_prob_df["predicted_gender"] = new_gender_prob_df["0"].apply(lambda row:1 if row <= 0.52 else 2)
#new_gender_prob_df.pop('0')
new_result_df = new_result_df.merge(new_gender_prob_df,on="user_id",how="inner")
accuracy_score(new_result_df["old_predicted_gender"],new_result_df["predicted_gender"])

new_result_df = new_result_df.rename(columns={"max_predict":"predicted_age"})
new_result_df[["user_id","predicted_gender","predicted_age"]].to_csv(base_dir + "result/ensemble_0622_v2.csv",index=False)

new_gender_prob_df = pd.read_csv(base_dir+"ensemble/val_3features_gender_pool_attention_method3_0621_prob.csv")
new_gender_prob_df["predicted_gender"] = new_gender_prob_df["0"].apply(lambda row:1 if row <= 0.50 else 2)
new_gender_prob_df['user_id'] = new_gender_prob_df['user_id'].astype('str')
val_ = new_gender_prob_df.merge(user_df,on="user_id",how="inner")
accuracy_score(val_["gender"],val_["predicted_gender"])