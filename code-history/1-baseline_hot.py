import time, math, os
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
import collections


data_path = 'F:/data/tianchi-news-rs/' # 读取根path
save_path = './save/'  # 保存根path



# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path='./data_raw/', offline=True):
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        all_click = trn_click.append(tst_click)


    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click,tst_click

# 全量训练集 测试集
all_click_df,tst_click_df = get_all_click_df(data_path, offline=False)


# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


# 从点击最多的文章中选出5个 给每个用户 
def get_user_recall(uid_list,top_click,target_num=5):
    result_df=pd.DataFrame()
    result_df['user_id']=uid_list
    user_result_col=[]
    for uid in uid_list:
        user_result_col.append(random.sample(top_click, target_num))
    result_df['recall_list']=user_result_col
    return result_df

# 获取最终的推荐
def get_final_result(all_click_df,tst_click_df,k):
    uid_list=tst_click_df['user_id'].unique()
    top_click=list(get_item_topk_click(all_click_df,k))
    result_df=get_user_recall(uid_list,top_click)
    return result_df

# 生成提交文件
def submit(recall_df, model_name=None):
    # 把recall_df现在的两列item 转化成五列item
    recall_item_list=np.array(list(recall_df['recall_list'])).T
    #print(recall_item_list)
    submit=pd.DataFrame(list(recall_df['user_id']),columns=['user_id'])
    for i in range(0,5):
        print(recall_item_list[i])
        submit['article_{}'.format(i+1)]=recall_item_list[i]
    
    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)



## 生成提交文件
result_df=get_final_result(all_click_df,tst_click_df,5)
print(result_df)
submit(result_df, model_name='hot_baseline')
print('save')






