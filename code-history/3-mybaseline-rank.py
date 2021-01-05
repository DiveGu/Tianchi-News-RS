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


# debug模式：从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click['user_id'].unique()

    test_click = pd.read_csv(data_path + 'testA_click_log.csv')
    test_user_ids = all_click['user_id'].unique()

    #sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False) 

    #all_click = all_click[all_click['user_id'].isin(sample_user_ids)]
    all_click = all_click[all_click['user_id'].isin(test_user_ids)]

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp'])) #数据去重
    return all_click

# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path='./data_raw/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        all_click = trn_click.append(tst_click)
    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

# 自定义返回df
def get_self_click_df(data_path):
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    return tst_click


# 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time(click_df):
    
    click_df = click_df.sort_values('click_timestamp') # 升序
    
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))
    
    # 先用上面的pair函数构造item_time_list对，出现新的一列，默认名字是'0'，rename成'item_time_list'
    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(lambda x: make_item_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'item_time_list'})

    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    
    return user_item_time_dict

# 划分训练集和验证集
def set_trian_evaluate(user_item_time_dict):
    # df 分组求max 的index

    evaluate=[]
    #把每位用户最新的点击文章取出来作为验证集
    for uid,items in user_item_time_dict.items():
        last=items.pop()
        evaluate.append([uid,last[0]])

    evaluate=pd.DataFrame(evaluate,columns=['user_id','last_click_id'])
    return evaluate


# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)
    
    for k in range(10, topk+1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1
        
        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)



# 获取数据集
all_click_df = get_all_click_df(data_path, offline=True)
# 获取用户点击文章字典
user_item_time_dict=get_user_item_time(all_click_df[:10000])
# 获取验证集
#evaluate=set_trian_evaluate(user_item_time_dict) #用户最后点击的文章
# 此时的user_item_time_dict已经没有evaluate了





