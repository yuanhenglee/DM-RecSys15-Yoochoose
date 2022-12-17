from Session import Session

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


x_train = np.array( [s[0].session_embedding( candidates, method = 'acc') for s in train_sessions[:1000]] )
y_train = np.array( [s[1] for s in train_sessions[:1000]] )


print( "model training...")
clf = KNeighborsClassifier( n_neighbors=5 )
clf.fit( x_train, y_train )

print( "dump result...")

joblib.dump(clf, '../model/Classifier/knn.model')

x_test = np.array( [s[0].session_embedding( candidates, method = 'acc') for s in test_sessions] )
y_test = np.array( [s[1] for s in test_sessions] )

y_pred = clf.predict(x_test)

np.save( '../output/knn_pred', y_pred )

