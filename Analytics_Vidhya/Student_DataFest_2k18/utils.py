## Utils functions for EDA.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

## Calculating the number of records and columns in the test and train dataset.
def plot_num_records(kwargs):
    '''
    kwargs : dict (
    {"Train" : train_df ,
    "Test" : test_df }
    '''

    n_rows = []
    n_cols = []
    names = []

    for name ,df in kwargs.items():
        n_rows.append(df.shape[0])
        n_cols.append(df.shape[1])
        names.append(name)
    
    if len(names) != 0:
        ## Plotting the values
        ''' Number of rows in each file '''
        fig, axs =plt.subplots(ncols=2, figsize=(8,6))
        fig.suptitle("Number Tuples in Files")
        ax_rows = sns.barplot(names, n_rows, ax=axs[0])
        ax_rows.set(xlabel="Files", ylabel='Number Of Rows')
        for n, da in enumerate(zip(names, n_rows)):
            if da[1]!=0:
                ax_rows.text(n, da[1], da[1], ha='center')

        ''' Number of Columns in each file'''
        ax_cols = sns.barplot(names, n_cols, ax=axs[1])
        ax_cols.set(xlabel='Files', ylabel='Number Of Columns')
        for n, data in enumerate(zip(names, n_cols)):
            if data[1]!=0:
                ax_cols.text(n, data[1], data[1], va = 'center')

    else :
        print("Invalid Input Format")
