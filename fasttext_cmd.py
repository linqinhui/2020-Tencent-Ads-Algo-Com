import pandas as pd
import numpy as np
import fasttext
import os
import lightgbm as lgb
from sklearn.metrics import accuracy_score,confusion_matrix

base_dir = "./"
click_log_df = pd.read_csv(base_dir + "train_preliminary/click_log.csv")
ad_df = pd.read_csv(base_dir + "train_preliminary/ad.csv")
click_log_df = click_log_df.merge(ad_df,on="creative_id",how="left")

test_click_log_df = pd.read_csv(base_dir + "test/click_log.csv")
test_ad_df = pd.read_csv(base_dir + "test/ad.csv")
test_click_log_df = test_click_log_df.merge(test_ad_df,on="creative_id",how="left")

all_data_df = pd.concat([click_log_df,test_click_log_df],axis=0)
all_data_df.nunique()
"""
time                     91
user_id             1900000
creative_id         3412772
click_times              94
ad_id               3027360
product_id            39057
product_category         18
advertiser_id         57870
industry                332
"""

#100 100 50 50 50 50         400
#128 128 64 32 64 32        448
target_id_list =[ 'creative_id', 'ad_id', 'advertiser_id','product_id']
id_vector_dict = {'creative_id':128, 'ad_id':128, 'product_id':128, 'product_category':32, 'advertiser_id':128, 'industry':32}
for id_name in target_id_list:
    print(id_name)
    all_data_df[id_name] = all_data_df[id_name].astype("str")
    user_time_group_df = all_data_df.groupby(["user_id", "time"])[id_name].agg(" ".join).reset_index()
    user_group_df = user_time_group_df.groupby("user_id")[id_name].agg(" ".join).reset_index()
    id_csv = base_dir + "embedding/{}.csv".format(id_name)
    user_group_df[id_name].to_csv(id_csv, index=False, header=False)
    dim = id_vector_dict[id_name]
    output_name = base_dir + "embedding/{}_{}.csv".format(id_name,dim)
    cmd = "/public/home/zhouquan/lqh/fastText-0.9.2/fasttext skipgram -input {} -output {} -minn 0 -maxn 0 -ws 100 -minCount 3 -dim {}".format(id_csv,output_name,dim)
    print("excute {}".format(cmd))
    os.system(cmd)
