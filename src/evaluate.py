#!/usr/bin/env python
# coding: utf-8

import numpy as np
from Session import Session
import pickle
import pandas as pd
from tqdm.contrib import tzip


def mrr( rank, label ):
    for i, item in enumerate(rank[:20]):
        if item == label:
            return 1.0/(i + 1)
    return 0

def recall( rank, label ):
    return int(label in rank[:20])

print("loading data")

with open('../output/STAN_result.csv', 'r') as f:
    STAN_df = pd.read_csv(f)
with open('../output/SKNN_result.csv', 'r') as f:
    SKNN_df = pd.read_csv(f)
with open('../output/VSKNN_result.csv', 'r') as f:
    VSKNN_df = pd.read_csv(f)
with open( '../output/itemCF_result.csv', 'r') as f:
    itemCF_df = pd.read_csv(f)
with open('../output/NARM_100epochs.csv', 'r') as f:
    NARM_df = pd.read_csv(f)
with open('../data/yoochoose1_64_4col/test.txt', 'rb') as f:
    session_ids, _, _, labels = pickle.load(f)

print("preprocessing")

session_set = set(STAN_df.session_id) & set(SKNN_df.session_id) & set(VSKNN_df.session_id) & set(itemCF_df.session_id) & set(NARM_df.session_id)
x_test = []
y_test = []
for i in session_set:
    x_test.append( session_ids[i] )
    y_test.append( labels[i] )

# for res_df, name in zip([STAN_df, SKNN_df, VSKNN_df, itemCF_df, NARM_df], ['STAN', 'SKNN', 'VSKNN', 'itemCF', 'NARM']):
#     global_mrr = 0
#     global_recall = 0
#     for x, y in zip(x_test, y_test):
#         rank = res_df.loc[res_df.session_id == x, 'item_id'].values
#         local_mrr = mrr( rank, y )
#         local_recall = recall( rank, y )
#
#         global_mrr += local_mrr
#         global_recall += local_recall
#     
#     global_mrr /= len(x_test)
#     global_recall /= len(x_test)
#     print(name, global_mrr, global_recall)
candidates = set(y_test)

for res_df in [STAN_df, itemCF_df, NARM_df]:
    res_df = res_df[res_df['item_id'].isin(candidates)]

print("evaluating")

# ensemble
global_mrr = 0
global_recall = 0
for x, y in tzip(x_test, y_test):
    score = {}
    for res_df, weight in zip( [STAN_df, itemCF_df, NARM_df], [1, 1, 1]):
        for i, item in enumerate(res_df.loc[res_df.session_id == x, 'item_id'].values.tolist()):
            if item not in score:
                score[item] = 0
            # if item in set(y_test):
            score[item] += 40-i 

    rank = [ k for k, _ in sorted(score.items(), key=lambda item: item[1], reverse = True)[:20] ]

    local_mrr = mrr( rank, y )
    local_recall = recall( rank, y )

    global_mrr += local_mrr
    global_recall += local_recall
global_mrr /= len(x_test)
global_recall /= len(x_test)
print("ensemble", global_mrr, global_recall)



