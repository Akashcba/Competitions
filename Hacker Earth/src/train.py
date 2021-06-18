import os
import time
import pandas as pd
from sklearn import preprocessing

from sklearn import ensemble
from sklearn import metrics

from . import dispatcher
import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

Fold_Mapping = {
    0 : [1,2,3,4,5,6],
    1 : [0,2,3,4,5,6],
    2 : [0,1,3,4,5,6],
    3 : [0,1,2,4,5,6],
    4 : [0,1,2,3,5,6],
    5 : [0,1,2,3,4,6],
    6 : [0,1,2,3,4,5]
}

if __name__ == "__main__":
    print("\nExecuting the Train Module")
    time.sleep(1)
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(Fold_Mapping.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    print("\nTraining Fold : ", FOLD, "\n")

## Add Same column sequence in both
    valid_df = valid_df[train_df.columns]

    '''
    label_encoders ={}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
        df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")
        lbl.fit(train_df[c].values.tolist() + 
                valid_df[c].values.tolist() +
                df_test[c].values.tolist())

        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl
    '''
    # Data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    #print(preds)
    print("Accuracy : ", metrics.roc_auc_score(yvalid, preds))

    ## Storing the data for predict.py
    
    joblib.dump(clf, f"/Users/my_mac/Documents/Machine Learning/ML/models/{MODEL}_{FOLD}.pkl")
