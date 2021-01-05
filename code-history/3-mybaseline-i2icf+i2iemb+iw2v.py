import pandas as pd  
import numpy as np
from tqdm import tqdm  
from collections import defaultdict  
import os, math, warnings, math, pickle
from tqdm import tqdm
#import faiss
import collections
import random
from datetime import datetime


data_path = 'F:/data/tianchi-news-rs/' # 读取根path
save_path = './save/'  # 保存根path



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


# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df

# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


def itemcf_sim(df):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
        思路: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
    """
    
    user_item_time_dict = get_user_item_time(df)
    
    # 计算物品相似度
    i2i_sim = {} # 是字典类型 i2i_sim[i]也是字典
    item_cnt = defaultdict(int) # 计算item的出现次数 #字典 默认value=0 key不存在的情况下 value=0
    for user, item_time_list in tqdm(user_item_time_dict.items(),total=1000):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素 先不管

        # 把之前的写法n^2 改成n^2/2
        if(len(item_time_list)<2):
            continue
        # 循环ind1 ind2=ind1+1—len()
        for ind1 in range(0,len(item_time_list)-1):
            i,i_click_time=item_time_list[ind1]
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for ind2 in range(ind1+1,len(item_time_list)):
                j,j_click_time=item_time_list[ind2]
                item_cnt[j] += 1
                i2i_sim.setdefault(j, {})
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[j].setdefault(i, 0)
                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)
                i2i_sim[j][i] += 1 / math.log(len(item_time_list) + 1)

        #for i, i_click_time in item_time_list:
        #    item_cnt[i] += 1
        #    i2i_sim.setdefault(i, {})
        #    for j, j_click_time in item_time_list:
        #        if(i == j):
        #            continue
        #        i2i_sim[i].setdefault(j, 0)
                
        #        i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)
                
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    
    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
    
    return i2i_sim_

#i2i_sim = itemcf_sim(all_click_df)

# 对i2i_sim进行排序
def i2i_sim_sort(i2i_sim,key_list):
    for i in key_list:
        if(i in i2i_sim.keys()):
            i2i_sim[i]=sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)
    return i2i_sim

# 基于商品的召回i2i 给一个用户
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全        
        return: 召回的文章列表 {item1:score1, item2: score2...}
        注意: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
    """
    
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {user_id for user_id, _ in user_hist_items} # 这返回的应该是物品id集合
    
    item_rank = {} # 单个用户召回的rank结果 item_id—item_score
    for loc, (i, click_time) in enumerate(user_hist_items):
        # i2i_sim[i]也是dict 按照i2i_sim[i]的values降序排列 取前topk个
        if(i not in i2i_sim.keys()):
            continue
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue
                
            item_rank.setdefault(j, 0)
            item_rank[j] +=  wij
    
    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items(): # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100 # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break
    
    # item_id:item_score
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num] # 得用item_rank.items()
        
    return item_rank


# 向量检索相似度计算
def search_sim_topk(cur_item,emb_matrix,topk):
    sim=[] # 相似度值
    idx=[] # 相似的id的List
    # 1*维度 维度*N 结果取前topk个
    result=np.matmul(cur_item,emb_matrix.T) # 1 * N 一维数组
    idx=result.argsort()[::-1][:topk]
    result.sort()
    sim=result[-topk:][::-1]
    return sim,idx


# topk指的是每个item, faiss搜索后返回最相似的topk个item
def embdding_sim(query_item_id, item_emb_df, save_path, topk):
    """
        基于内容的文章embedding相似性矩阵计算
        :param click_df: 数据表
        :param item_emb_df: 文章的embedding
        :param save_path: 保存路径
        :patam topk: 找最相似的topk篇
        return 文章相似性矩阵
        
        思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """
    #item_emb_df[['article_id']] = item_emb_df[['article_id']].astype(int)
    # 文章索引与文章id的字典映射
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))
    # 文章id与文章索引的字典映射
    item_rawid_2_idx_dict=dict(zip(item_emb_df['article_id'],item_emb_df.index))

    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32) # None * 250
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)# np.array类型 None * 250

    # itemid:{sim_id:sim_value}
    item_sim_dict = collections.defaultdict(dict)

    # query是查询物品的id
    for query_id in tqdm(query_item_id):
        cur_item=item_emb_np[item_rawid_2_idx_dict[query_id]]
        # 相似度查询，给每个query的物品id向量返回topk个item以及相似度
        sim, idx = search_sim_topk(cur_item,item_emb_np,topk)
        #for id in idx:
        #    print(np.matmul(cur_item,item_emb_np[id].T))
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for sim_value,sim_idx in zip(sim[1:],idx[1:]):
            sim_id=item_idx_2_rawid_dict[sim_idx]
            item_sim_dict[query_id][sim_id]=item_sim_dict.get(query_id, {}).get(sim_id, 0) + sim_value
    
    print(item_sim_dict)
    # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open(save_path + 'emb_i2i_sim.pkl', 'wb'))   
    
    return item_sim_dict



# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=50):
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


# 制作文章嵌入相似词典可惜只保存了top9个 应该多保存一些
def make_item_emb_sim_dict():
    # 制作item_emb相似召回
    all_clcik_df=get_self_click_df(data_path)
    # 制作所有的出现的文章id的集合作为参数 算相似度
    query_item_id=list(all_clcik_df['click_article_id'].unique())
    print(len(query_item_id))

    #根据给出的嵌入找出每个物品相似的k个物品
    item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')

    emb_i2i_sim = embdding_sim(query_item_id,item_emb_df, save_path, topk=10) # topk可以自行设置

# make_item_emb_sim_dict()

# 制作item_emb的召回结果
def make_item_emb_recall_result():
    user_recall_items_dict = collections.defaultdict(dict)

    all_click_df=get_all_click_df(data_path, offline=False)

    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    tst_users=tst_click['user_id'].unique()

    emb_i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))

    # 获取 用户 - 文章 - 点击时间的字典
    user_item_time_dict = get_user_item_time(tst_click)

    # 相似文章的数量
    sim_item_topk = 10 #可惜只保存了9个 不然这里可以多找一些

    # 召回文章数量
    recall_item_num = 10

    # 用户热度补全
    item_topk_click = get_item_topk_click(all_click_df, k=50)

    for user in tqdm(tst_users,total=100):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, emb_i2i_sim, 
                                                            sim_item_topk, recall_item_num, item_topk_click)

    pickle.dump(user_recall_items_dict, open(save_path + 'item_emb_recall_dict.pkl', 'wb'))

#make_item_emb_recall_result()

# 制作itemcf的召回结果
def make_itemcf_recall_result():
    # 定义
    user_recall_items_dict = collections.defaultdict(dict)
    all_click_df=get_all_click_df(data_path, offline=False)

    trn_hist_click_df, trn_last_click_df=get_hist_and_last_click(all_click_df)

    # 获取 用户 - 文章 - 点击时间的字典
    user_item_time_dict = get_user_item_time(trn_hist_click_df)

    i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))

    # 相似文章的数量
    sim_item_topk = 20

    # 召回文章数量
    recall_item_num = 10

    # 用户热度补全
    item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

    for user in tqdm(trn_hist_click_df['user_id'].unique(),total=100,desc='total_num:{}'.format(len(trn_hist_click_df['user_id'].unique()))):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, 
                                                            sim_item_topk, recall_item_num, item_topk_click)
    pickle.dump(user_recall_items_dict, open(save_path + 'itemcf_i2i_recall_dict.pkl', 'wb'))

make_itemcf_recall_result()

#合并多个召回结果
def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=20):
    final_recall_items_dict = {}
    
    # 对每一种召回结果按照用户进行归一化，方便后面多种召回结果，相同用户的物品之间权重相加
    def norm_user_recall_items_sim(sorted_item_list):
        # 如果冷启动中没有文章或者只有一篇文章，直接返回，出现这种情况的原因可能是冷启动召回的文章数量太少了，
        # 基于规则筛选之后就没有文章了, 这里还可以做一些其他的策略性的筛选
        if len(sorted_item_list) < 2:
            return sorted_item_list
        
        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]
        
        norm_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))
            
        return norm_sorted_item_list
    
    print('多路召回合并...')
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + '...')
        # 在计算最终召回结果的时候，也可以为每一种召回结果设置一个权重
        if weight_dict == None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict[method]
        
        for user_id, sorted_item_list in user_recall_items.items(): # 进行归一化
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)
        
        for user_id, sorted_item_list in user_recall_items.items():
            # print('user_id')
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score  
    
    final_recall_items_dict_rank = {}
    # 多路召回时也可以控制最终的召回数量
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

    # 将多路召回后的最终结果字典保存到本地
    pickle.dump(final_recall_items_dict_rank, open(os.path.join(save_path, 'final_recall_items_dict.pkl'),'wb'))

    return final_recall_items_dict_rank


# 生成提交文件
def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
    
    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    print(tmp)
    assert tmp.min() >= topk
    
    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()
    
    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2', 
                                                  3: 'article_3', 4: 'article_4', 5: 'article_5'})
    
    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)

## 读取所有的召回结果 合并成一个字典
#user_multi_recall_dict={}
#user_multi_recall_dict['itemcf_sim_itemcf_recall']=pickle.load(open(save_path + 'itemcf_i2i_recall_dict.pkl', 'rb'))
#user_multi_recall_dict['embedding_sim_item_recall']=pickle.load(open(save_path + 'item_emb_recall_dict.pkl', 'rb'))
##print(user_multi_recall_dict['itemcf_sim_itemcf_recall'][249999])
##print(user_multi_recall_dict['embedding_sim_item_recall'][249999])
## 合并多路召回
## 这里直接对多路召回的权重给了一个相同的值，其实可以根据前面召回的情况来调整参数的值
#weight_dict = {'itemcf_sim_itemcf_recall': 1.0,
#               'embedding_sim_item_recall': 0.2,
#               #'youtubednn_recall': 1.0,
#               #'youtubednn_usercf_recall': 1.0, 
#               #'cold_start_recall': 1.0,
#               }

## 最终合并之后每个用户召回150个商品进行排序
#final_recall_items_dict_rank = combine_recall_results(user_multi_recall_dict, weight_dict, topk=20)

## 将字典的形式转换成df
#user_item_score_list = []

#for user, items in tqdm(final_recall_items_dict_rank.items(),total=10):
#    for item, score in items:
#        user_item_score_list.append([user, item, score])

#recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])

#submit(recall_df, topk=5, model_name='itemcf+item_emb')