from Session import Session
import argparse
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
import joblib

N_NEIGHBOR = 500

print( "load data...")

with open( 'train_sessions.pkl', 'rb' ) as f:
    train_sessions = pickle.load(f)
with open( 'test_sessions.pkl', 'rb' ) as f:
    test_sessions = pickle.load(f)
with open( 'candidate_itemset.pkl', 'rb') as f:
    candidates = pickle.load(f)

# print(train_sessions[0].session_embedding( candidates, method = 'acc' ))

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train mode')
    parser.add_argument('--test', action='store_true', help='test mode')
    args = parser.parse_args()

    if args.train:
        print( "train mode" )

        x_train = [s[0].session_embedding( candidates, method = 'acc') for s in train_sessions]
        y_train = [s[1] for s in train_sessions]


        print( "model training")
        clf = KNeighborsClassifier( n_neighbors=N_NEIGHBOR, metric='cosine', n_jobs=8 )
        clf.fit( x_train, y_train )

        print( "dump result")

        joblib.dump(clf, '../model/Classifier/knn.model')

    if args.test:
        print( "test mode" )

        clf = joblib.load('../model/Classifier/knn.model')

        x_test = [s[0].session_embedding( candidates, method = 'acc') for s in test_sessions]
        y_test = [s[1] for s in test_sessions]

        y_pred = clf.predict_proba(x_test)

        result = []
        for pred in y_pred:
            top20_items = []
            top_indices = np.argsort(pred)[::-1]
            for i in top_indices:
                if pred[i] in candidates:
                    top20_items.append( clf.classes_[i] )
                    if len(top20_items) == 20:
                        break
            if len(top20_items) < 20:
                top20_items.extend([0]*(20-len(top20_items)))
            result.append( top20_items )

        np.save( '../output/knn_pred', np.array(result) )