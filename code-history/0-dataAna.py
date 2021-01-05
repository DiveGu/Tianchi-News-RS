import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import collections

data_path = 'F:/data/tianchi-news-rs/' # 读取根path
save_path = './save/'  # 保存根path



# 看train中有多少test
def get_all_click_sample(data_path):

    train_click = pd.read_csv(data_path + 'train_click_log.csv')
    train_user_ids = train_click['user_id'].unique()

    test_click = pd.read_csv(data_path + 'testA_click_log.csv')
    test_user_ids = test_click['user_id'].unique()

    double_click = train_click[train_click['user_id'].isin(test_user_ids)]
    double_user_ids = double_click['user_id'].unique()

    print('train_uid_num:{}'.format(len(train_user_ids)))
    print('tst_uid_num:{}'.format(len(test_user_ids)))
    print('double_uid_num:{}'.format(len(double_user_ids)))
    print(train_click.shape)
    print(test_click.shape)

    double_click = train_click[train_click['click_article_id'].isin(test_click['click_article_id'].unique())]
    train_iid_num=len(train_click['click_article_id'].unique())
    test_iid_num=len(test_click['click_article_id'].unique())
    double_iid_num=len(double_click['click_article_id'].unique())
    print('train_iid_num:{}'.format(train_iid_num))
    print('tst_iid_num:{}'.format(test_iid_num))
    print('double_iid_num:{}'.format(double_iid_num))

#get_all_click_sample(data_path)

# 查看item的嵌入
def get_item_emb(data_path):
    item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')
    print(item_emb_df.head())
    #item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    #item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
    ## 进行归一化
    #item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    #item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    #pickle.dump(item_emb_dict, open(save_path + 'item_content_emb.pkl', 'wb'))

#get_item_emb(data_path)

# 看train中用户点击数的分布
def get_data_count():
    train_click = pd.read_csv(data_path + 'train_click_log.csv')
    group=train_click['user_id'].value_counts()

    group_df=pd.DataFrame()
    group_df['user_id']=group.index
    group_df['click_count']=group.values

    print(group_df)
    one_click_users=group_df[group_df['click_count']==1]
    print(one_click_users)

    #group=group_df['click_count'].value_counts()
    #print(group)
    #df=pd.DataFrame()
    #df['user_id']=group.index
    #df['click_count']=group.values

    rank=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    result=group_df['click_count'].quantile(rank)
    ##plt.bar(temp.index,temp.values)
    ##group_df['click_count'].value_counts().plot(kind='bar')
    ##print(group_df['click_count'].value_counts())
    ##plt.show()
    print(result)

get_data_count()
