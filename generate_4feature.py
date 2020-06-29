import pandas as pd
import numpy as np
import fasttext
import os
import lightgbm as lgb
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import KFold,StratifiedKFold

base_dir = ""
click_log_df = pd.read_csv(base_dir + "train_preliminary/click_log.csv")
ad_df = pd.read_csv(base_dir + "train_preliminary/ad.csv")
click_log_df = click_log_df.merge(ad_df,on="creative_id",how="left")

test_click_log_df = pd.read_csv(base_dir + "test/click_log.csv")
test_ad_df = pd.read_csv(base_dir + "test/ad.csv")
test_click_log_df = test_click_log_df.merge(test_ad_df,on="creative_id",how="left")
all_data_df = pd.concat([click_log_df,test_click_log_df],axis=0)
train_df = click_log_df[click_log_df["user_id"]<=675242]
val_df = click_log_df[click_log_df["user_id"]>675242]
"""
#kf=KFold(n_splits=5,random_state=2020,shuffle=True)
for train_index, val_index in kf.split(click_log_df):
    train_df,val_df=click_log_df.iloc[train_index,:],click_log_df.iloc[val_index,:]
"""

user_df = pd.read_csv(base_dir + "train_preliminary/user.csv")


def generate_sequence(df,id_name):
    df[id_name] =  df[id_name].astype("str")
    user_time_group_df = df.groupby(["user_id", "time"])[id_name].agg(" ".join).reset_index()
    user_group_df = user_time_group_df.groupby("user_id")['creative_id',].agg(" ".join).reset_index()
    return user_group_df

target_id_list =[ 'creative_id', 'ad_id','advertiser_id','product_id']
total_train_user_embedding_df = None
total_val_user_embedding_df = None
total_test_user_embedding_df = None
for target_id in target_id_list:
    print("start {} embedding!".format(target_id))
    all_user_embedding_df = generate_sequence(all_data_df, target_id)
    train_user_embedding_df = generate_sequence(train_df, target_id)
    val_user_embedding_df = generate_sequence(val_df, target_id)
    test_user_embedding_df = generate_sequence(test_click_log_df, target_id)
    if total_train_user_embedding_df is None:
        total_all_user_embedding_df = all_user_embedding_df
        total_train_user_embedding_df = train_user_embedding_df
        total_val_user_embedding_df = val_user_embedding_df
        total_test_user_embedding_df = test_user_embedding_df
    else:
        total_all_user_embedding_df = total_all_user_embedding_df.merge(all_user_embedding_df, on="user_id",how="left")
        total_train_user_embedding_df = total_train_user_embedding_df.merge(train_user_embedding_df,on="user_id",how="left")
        total_val_user_embedding_df = total_val_user_embedding_df.merge(val_user_embedding_df, on="user_id",how="left")
        total_test_user_embedding_df = total_test_user_embedding_df.merge(test_user_embedding_df, on="user_id", how="left")

all_join_df =  user_df.merge(total_all_user_embedding_df,on="user_id",how="right")
all_join_df = all_join_df.fillna({"age":3,"gender":1})
all_join_df[['creative_id', 'ad_id','advertiser_id','product_id','gender','age']].to_csv(base_dir+"embedding/all_3features.csv", index=False)

train_join_df = user_df.merge(total_train_user_embedding_df,on="user_id",how="inner")
train_join_df[['creative_id', 'ad_id','advertiser_id','product_id','gender','age']].to_csv(base_dir+"embedding/train_3features.csv", index=False)

val_join_df = user_df.merge(total_val_user_embedding_df,on="user_id",how="inner")
val_join_df[['user_id','creative_id', 'ad_id','advertiser_id','product_id','gender','age']].to_csv(base_dir+"embedding/val_3features.csv", index=False)
"""
target_id_list =[ 'creative_id', 'ad_id','advertiser_id','product_id']
total_test_user_embedding_df = None
for target_id in target_id_list:
    print("start {} embedding!".format(target_id))
    test_user_embedding_df = generate_sequence(test_click_log_df, target_id)
    if total_test_user_embedding_df is None:
        total_test_user_embedding_df = test_user_embedding_df
    else:
        total_test_user_embedding_df = total_test_user_embedding_df.merge(test_user_embedding_df, on="user_id", how="left")


total_test_user_embedding_df[['user_id','creative_id', 'ad_id','advertiser_id','product_id']].to_csv(base_dir+"embedding/test_3features.csv", index=False)


