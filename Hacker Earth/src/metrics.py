from sklearn import metrics as skmetrics
'''
Binary Classification Metrics
-> Accuracy
-> Precision
-> Recall
-> F1-Score
-> AUC (Area Under ROC curve), Receiver Operating Characteristics
-> logloss
'''
# Theory behind Loss Calculation (Fo binary case)
'''
target = [0,1]; 0 = Negative, 1 = Positive

True Positive (TP) = Target is + and we predicted +
True Negative (TN) = Target is - and we predicted -
False Positive (FP) = Target is - but we predicted +
False Negative (FN) = Target is + but we predicted -

Accuracy = Correct_Predictions / Total_No_Of_Samples
         = (TP + TN)/(TP+FP+FN+TN)
Precision = How precise the Model is = Out of all Positives PREDICTED how many were truly Positive.
          = TP/(TP + FP) = Precision High => False Positive Rate is Low -> That is what we want (Predict less False +)
Recall = TP/(TP+FN) = Out of all REAL + Values how many we predicted Correctly.
F1 = (weighted Average of Precision and Recall)
   = 2*Precision*Recall/(Precision+Recall)
   = 2.TP/(2.TP + FP + FN)

TPR = True Positive Rate
    = TP/(TP + FN) = Recall
FPR = False Positive Rate = Out of all Negative samples how many were incorrectly predicted as +
    = FP/(TN + FP)

'''
# [0,   0,    1,   0,   1,   1]   => True Labels
# [0.6, 0.4, 0.7, 0.3, 0.5, 0.9]  => Predicted Labels
'''
AUC is the Graph b/w TPR and FPR for different values of THRESHOLD
The area under this curve is the AUC
AUC = 1 -> great model
AUC = 0.5 -> Random Model
Given any + sample from the dataset and any - sample in the dataset
what is the prob that the + sample will rank higher than - sample
this values is the AUC
'''
# LogLoss => Lower logloss is better
'''
logloss = -(y.log(P) + (1-y).log(1-P)); P = prediction
Penalizes quite high for wrong prediction.
For all samples is just average of individual sample loglosses
'''

class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy": self._accuracy,
            "f1": self._f1,
            "precision": self._precision,
            "recall": self._recall,
            "auc": self._auc,
            "logloss": self._logloss
        }
    
    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception("Metric not implemented")
        if metric == "auc":
            if y_proba is not None:
                return self._auc(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None for AUC")
        elif metric == "logloss":
            if y_proba is not None:
                return self._logloss(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None for logloss")
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)