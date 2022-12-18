import math
import random
import pandas as pd
import pickle
from collections import defaultdict
from operator import itemgetter
import argparse

def LoadData(filepath):
    trainData = dict()
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    for i in range(0, len(data)):
        trainData[i] = data[i]
        i = i + 1
    return trainData

def Load_testData(filepath):
    SID = []
    X = []
    y = []
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    for i in range(0, len(data[0])):
        SID.append(data[0][i])
        i = i + 1
    for i in range(0, len(data[1])):
        X.append(data[1][i])
        i = i + 1
    for i in range(0, len(data[3])):
        y.append(data[3][i])
        i = i + 1
    return SID, X, y

def CreateCandidate(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        candidate = set()
        for i in range(1, len(lines)):
            line = lines[i]
            candidate.add(int(line))

        return candidate

def PreProcessData(originData):
    """
    create Session-Item dictionary as:
        {"Session1": [ ItemID1, ItemID2, ItemID3,... ]
         "Session2": [ ItemID5, ItemID6, ItemID7,... ]
        }
    """
    trainData = dict()
    for i in range(1, len(originData)):
        line = originData[i].split('\t')
        session_id = line[0]
        item_id = line[1]
        if( session_id == originData[i-1].split('\t')[0]):
            session_id = int(session_id)
            item_id = int(item_id)
            trainData[session_id].append(item_id)
        else:
            session_id = int(session_id)
            item_id = int(item_id)
            trainData.setdefault(session_id, [])
            trainData[session_id].append(item_id)
    return trainData


class ItemCF(object):
    """ Item based Collaborative Filtering Algorithm Implementation"""
    def __init__(self, trainData, similarity="cosine", norm=True):
        self._trainData = trainData
        self._similarity = similarity
        self._isNorm = norm
        self._itemSimMatrix = dict() # item similarity matrix
        """
        self._itemSimMatrix
        { itemID1: { relatedItemID1 : similarity, relatedItemID2 : similarity, ...}
          itemID2: { relatedItemID1 : similarity, relatedItemID2 : similarity, ...}
          ...
        }
        """
    def similarity(self):
        N = defaultdict(int) # count item popularity
        for user, items in self._trainData.items():
            itemset = set(items)
            for i in itemset:
                N[i] += 1
            
            for i in items:
                self._itemSimMatrix.setdefault(i, dict())
                
                for j in items:
                    if i == j:
                        continue
                    self._itemSimMatrix[i].setdefault(j, 0)
                    if self._similarity == "cosine":
                        self._itemSimMatrix[i][j] += 1
                    elif self._similarity == "iuf":
                        self._itemSimMatrix[i][j] += 1. / math.log1p(len(items) * 1.)
        for i, related_items in self._itemSimMatrix.items():
            for j, cij in related_items.items():
                self._itemSimMatrix[i][j] = cij / math.sqrt(N[i]*N[j])

        # save similarity matrix to file
        with open("itemSimMatrix.pickle", 'wb') as f:
            pickle.dump(self._itemSimMatrix, f)

    def recommend(self, items, N, K):
        """
        items: a list containing known prefered items
        N: number of item to recommend
        K: number of similar items to search for
        return value: recommend item list of user, sort by similarity
        """
        recommends = dict()
        
        point = 1
        for item in items:
            # find K most similar item from the similarity matrix
            if(item in self._itemSimMatrix):
                for i, sim in sorted(self._itemSimMatrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
                    if i in items:
                        continue  # skip if already in user preference item list
                    recommends.setdefault(i, 0.)
                    recommends[i] += sim * point
                point += 1.5
        
        # return recommend item list of user, sort by similarity
        return sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N]
    
    def train(self):
        self.similarity()

    def load_itemSimMatrix(self, path):
        """
        path: Path to pretrained item similarity matrix
        """
        with open(path, 'rb') as f:
            self._itemSimMatrix = pickle.load(f)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', default=None, type=str)
    return parser

def evaluate(model, SID, test_X, test_y, k):
    """
    model: pretrained model itemCF
    test_X: list of sessions
    test_y: list of the labels of the sessions
    k: length of recommendation list
    """
    mrr = 0
    recall = 0
    data_len = len(test_X)
    fp = open(r'../../output/itemCF_test_out_64_'+str(k)+'.txt', 'w')
    fp.write('session_id, item_id, score\n')

    for i in range(0, len(test_X)):
        session = test_X[i]
        ground_truth = test_y[i]
        ans = model.recommend(session, k, 1000)
        rank = 0
        for j,m in ans:
            fp.write(str(SID[i])+','+str(j)+','+str(m)+'\n')
            rank = rank + 1
            if(j ==  ground_truth):
                mrr = mrr + float(1/rank)
                recall = recall + 1            
    fp.close()
    mrr = float(mrr/data_len)
    recall = float(recall/data_len)

    return mrr, recall

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    test_path = args.test

    myItemCF = ItemCF([], similarity='iuf', norm=False)
    cand_score_dict = {}

    if(test_path!=None):
        print("Loading test list...")
        SID, test_X, test_y = Load_testData(test_path)
        
        print("Loading itemSimMatrix...")
        myItemCF.load_itemSimMatrix("itemSimMatrix.pickle")

        print("Evaluate on %s ..." % test_path.split('/')[-1])
        k = 40
        mrr, recall = evaluate(myItemCF, SID, test_X, test_y, k)
        print("MRR@%d: %.4f" % (k, mrr))
        print("Recall@%d: %.4f" % (k, recall))

    else:
        print("Loading train data...")
        train = LoadData(r"../../data/all_train_seq.txt")
        myItemCF = ItemCF(train, similarity='iuf', norm=False)
        print("Training...")
        myItemCF.train()