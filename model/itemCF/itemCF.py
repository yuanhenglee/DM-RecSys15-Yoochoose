import math
import random
import pandas as pd
import pickle
from collections import defaultdict
from operator import itemgetter
import argparse

def LoadData(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        return PreProcessData(lines)

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

    def recommend(self, user, N, K):
        """
        user: target user (to provide recommendation)
        N: number of item to recommend
        K: number of similar items to search for
        return value: recommend item list of user, sort by similarity
        """
        recommends = dict()
        
        # get the items that target user prefer
        items = self._trainData[user]
        
        point = 1
        for item in items:
            # find K most similar item from the similarity matrix
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
    parser.add_argument('-s', '--sid', default=1, type=int)
    parser.add_argument('-c', '--cand', default=None, type=str)
    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    target_user = args.sid
    candidate_path = args.cand

    print("Loading train data...")
    train = LoadData(r"../../data/rsc15_train_tr.txt")

    ItemCF = ItemCF(train, similarity='iuf', norm=False)
    cand_score_dict = {}

    if(candidate_path!=None):
        print("Loading candidate item list...")
        candidate = CreateCandidate(candidate_path)
        
        print("Loading itemSimMatrix...")
        ItemCF.load_itemSimMatrix("itemSimMatrix.pickle")

        print("Creating recommendation...")
        fp = open(r"../../output/itemCF_recommend_"+str(target_user)+".txt", "w")
        fp.write("item_id\tscore\n")

        ans = ItemCF.recommend(target_user, 1000, 2000)
        count = 1
        for j,k in ans:
            if(count > 100):
                break
            if(j in candidate):
                fp.write(str(j)+'\t'+str(k)+'\n')
                cand_score_dict[j] = k
                count = count + 1
    else:
        print("Training...")
        ItemCF.train()