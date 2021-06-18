import pandas as pd
import os
import joblib
import time
from sklearn import preprocessing

from colorama import Fore, Style, init
init()

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TESTING_DATA = os.environ.get("TEST_DATA")
TYPE = os.environ.get("TYPE")
NA = os.environ.get("NA")
PATH = os.environ.get("MODEL_PATH")

class Warning(Exception):
    def __init__(self, message):
        super().__init__( Fore.RED + message)



class CategoricalFeatures:
    def __init__(self, dataframe, categorical_features, encoding_type, handle_na=False):
        """
        dataframe: pandas dataframe
        categorical_features: list of column names, e.g. ["ord_1", "nom_0"......]
        encoding_type: label, binary, ohe
        handle_na: True/False
        """
        self.dataframe = dataframe
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na or self.dataframe[self.cat_feats].isnull().values.any():
            print(f"\n{Fore.YELLOW}There are Null Values in the DataSet or handle_na is set as True.{Style.RESET_ALL}")
            print("\nNull Values are as follows: \n",self.dataframe[self.cat_feats].isnull().sum())
            print(f"{Fore.YELLOW}\nPlease handle na values separately or PRESS - Y or y to continue with standard procedure. {Style.RESET_ALL}")
            i=input("> ")
            if i not in ('Y','y'):
                raise Warning("\nNan Values Found.\nExiting ...............")
        
        if self.handle_na:
            for c in self.cat_feats:
                self.dataframe.loc[:, c] = self.dataframe.loc[:, c].astype(str).fillna('-1')
        ## Ouput DataFrame
        self.output_dataframe = self.dataframe.copy(deep=True)
    
    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.dataframe[c].values)
            self.output_dataframe.loc[:, c] = lbl.transform(self.dataframe[c].values)
            self.label_encoders[c] = lbl

        joblib.dump(self.label_encoders,f"{PATH}label_encoder.pkl")
        #TRAINING_DATA="/Users/my_mac/Documents/Machine Learning/ML/input/train_folds.csv"
        # for c,lbl in self.label_encoder.items():
        return self.output_dataframe
    
    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.dataframe[c].values)
            val = lbl.transform(self.dataframe[c].values)
            self.output_dataframe = self.output_dataframe.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_dataframe[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        joblib.dump(self.binary_encoders,f"{PATH}binary_encoder.pkl")
        
        return self.output_dataframe

    def _one_hot(self):
        self.ohe = preprocessing.OneHotEncoder()
        self.ohe.fit(self.dataframe[self.cat_feats].values)
        self.output_dataframe = self.ohe.transform(self.dataframe[self.cat_feats].values)
        joblib.dump(self.ohe,f"{PATH}ohe_.pkl")
        return self.output_dataframe

    def fit_transform(self):
        print()
        if self.enc_type == "label":
            print("Running label_encoding .......")
            return self._label_encoding()
        elif self.enc_type == "binary":
            print("Running Binarization .......")
            return self._label_binarization()
        elif self.enc_type == "ohe":
            print("Running Ohe ........")
            return self._one_hot()
        else:
            raise Warning("Encoding type not understood")
    


if __name__ == "__main__":
    print("\nExecuting The Encoding Module .........")
    time.sleep(7)
    train = pd.read_csv(TRAINING_DATA)
    test = pd.read_csv(TESTING_DATA)
    print("Shape of Train input : ", train.shape)
    print("Shape of Test : ",test.shape)
    print("\nTrain Columns : \n", train.columns)
    print("\nTest Columns : \n", test.columns)
    train_len = len(train)
    df = pd.concat([train, test], axis=0,ignore_index=True)

    #cols = [c for c in df.columns if c not in ["id", "target"]]
    ## Select categorical columns
    #cols = df.select_dtypes(include=['object']).columns.tolist()
    
    cols = [c for c in df.columns if c not in ["id", "target"]]
    '''
    try:
        cols.remove('id')
    except:
        print("\nColumn id not Found in object Data Type")
    print(f"{Fore.GREEN}\nProcessing .........{Style.RESET_ALL}\n")
    try:
        cols.remove('target')
    except:
        print("Column target not Found in object Data Type")
    '''
    cat_feats = CategoricalFeatures(dataframe=df,
                                    categorical_features=cols,
                                    encoding_type=TYPE,
                                    handle_na=NA)
    df = cat_feats.fit_transform()
    
    train = df.iloc[ : train_len ,]
    test = df.iloc[train_len :, ]
    ## Storing the Files .....
    print("\nShape of Train : ",train.shape)
    print("\nShape of Test : ",test.shape)
    print("Train Columns : \n", train.columns)
    print("Test Columns : \n", test.columns)
    train.to_csv(f"{TRAINING_DATA[:-4]}.csv", index=False)
    test.to_csv(f"{TESTING_DATA[:-4]}.csv", index=False)
    joblib.dump(train.columns, f"{PATH}columns.pkl")
    print(f"{Fore.GREEN}File Successfully Modified ...{Style.RESET_ALL}")

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