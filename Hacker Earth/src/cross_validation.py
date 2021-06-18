import os
import time
import pandas as pd
from sklearn import model_selection


TRAINING_DATA = os.environ.get("TRAINING_DATA")
PROBLEM_TYPE = os.environ.get("PROBLEM_TYPE")
LABEL_DELIMETER = os.environ.get("LABEL_DELIMETER")
TARGET_COLS = list((os.environ.get("TARGET_COLS").split(" ") ))
NUM_FOLDS = int(os.environ.get("NUM_FOLDS"))

class CrossValidation:
    def __init__(
            self,
            df, 
            target_cols,
            shuffle=True, 
            problem_type="binary_classification",
            multilabel_delimiter=",",
            num_folds=5,
            random_state=42
        ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.multilabel_delimiter = multilabel_delimiter
        
        self.dataframe["kfold"] = -1
    
    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                     shuffle= self.shuffle, random_state=self.random_state)
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            ### Changes ...
            ## Implementation of Sorted Stratified Kfold for regression
            ## https://github.com/biocore/calour/blob/master/calour/training.py#L144
            ## Better implementation above
            self.dataframe.sort_values(self.target_cols,ascending=True, inplace=True)
            for i in range(0,num_samples,self.kfolds):
                k_counter = 0
                for j in range(i,min(i+self.kfolds,num_samples)):
                    self.df.loc[j,'kfolds'] = k_counter
                    k_counter +=1
            # ......... Updated For Regression
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle=self.shuffle, random_state=self.random_state)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold
        
        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1

        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle=self.shuffle, random_state=self.random_state)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        else:
            raise Exception("Problem type not understood!")

        return self.dataframe

if __name__ == "__main__":
    print("\nExecuting the Cross_Val Module.")
    time.sleep(7)
    df = pd.read_csv(TRAINING_DATA)
#    print(TARGET_COLS, len(TARGET_COLS))
#    print(PROBLEM_TYPE)
    cv = CrossValidation(df, shuffle=True, num_folds=NUM_FOLDS, target_cols=TARGET_COLS, 
                         problem_type=PROBLEM_TYPE, multilabel_delimiter=LABEL_DELIMETER )
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
    df_split.to_csv(f"{TRAINING_DATA[:-4]}.csv", index=False)
    print("File Successfully Modified ...")