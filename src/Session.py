
import pickle
import numpy as np

CANDIDATE_SIZE = 3216

class Session:
    def __init__(self, items, label = -1, lambda1 = 1.04):
        self.items = items
        self.label = label
        self.lambda1 = lambda1
    def session_embedding( self, candidates, method = 'binary', normalized = False ):
        emb = np.zeros( len( candidates ))
        for i, item in enumerate(self.items):
            if item in candidates:
                if method == 'binary':
                    emb[i] = 1
                elif method == 'count':
                    emb[i] += 1
                elif method == 'acc':
                    emb[i] = np.exp(i-len(self.items)/self.lambda1)
        
        if normalized:
            emb /= len(item_list)

        return emb


if __name__ == '__main__':
    with open('../data/yoochoose1_64/train.txt', 'rb') as f:
        train_itemsets, train_labels = pickle.load(f)

    with open('../data/yoochoose1_64/test.txt', 'rb') as f:
        test_itemsets, test_labels = pickle.load(f)

    candidates = [test_labels[0]]
    for i in range(1, len(test_labels)):
        if test_itemsets[i][0] != test_itemsets[i-1][0]:
            candidates.append(test_labels[i])
    candidates = sorted(list(set(candidates)))
    # print(len(candidates))
    assert len(candidates) == CANDIDATE_SIZE

    # try to leave only training row that have something appear in candidates
    train_sessions = []
    for i in range( len(train_labels) ):
        if train_labels[i] in candidates or len(set(train_itemsets[i]) & set(candidates)) == 1:
            train_sessions.append( [Session(train_itemsets[i], train_labels[i]), train_labels[i]] )

    test_sessions = [ [Session(items, label), label] for items, label in zip(test_itemsets, test_labels) ]

    with open( 'candidate_itemset.pkl', 'wb' ) as f:
        pickle.dump( candidates, f)
    with open( 'train_sessions.pkl', 'wb') as f:
        pickle.dump( train_sessions, f)
    with open( 'test_sessions.pkl', 'wb') as f:
        pickle.dump( test_sessions, f)
