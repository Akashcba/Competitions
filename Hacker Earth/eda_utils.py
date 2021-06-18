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
                ax_rows.text(n, da[1], da[1], ha='center', fontsize=12)

        ''' Number of Columns in each file'''
        ax_cols = sns.barplot(names, n_cols, ax=axs[1])
        ax_cols.set(xlabel='Files', ylabel='Number Of Columns')
        for n, data in enumerate(zip(names, n_cols)):
            if data[1]!=0:
                ax_cols.text(n, data[1], data[1], va = 'center', fontsize=12)

    else :
        print("Invalid Input Format")
    plt.show()

####### Calculating the missing values count
def plot_nan(data, title, figsize = (10,10)):
    n_rows = data.shape[0]
    cols = data.isnull().sum().index.values
    nans = data.isnull().sum().values.astype('int')

    ### Plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax=sns.barplot(nans, cols, ax=ax)
    ax.set(xlabel="Number Of Nans", ylabel="Columns")
    ax.set_title(title)
    for n, da in enumerate(zip(cols, nans)):
        if da[1]!=0:
            ax.text(da[1], n, str(da[1]) + ",    " + str(round(100*(da[1]/n_rows), 2)) + " %", va='center', fontsize=12)
    plt.show()

### Checking test and train data for data values
def data_check(train_df, test_df, cols=None, use_all_cols=True):
    if cols == None:
        if use_all_cols:
            train_cols = set(train_df.columns)
            test_cols = set(test_df.columns)
            cols = train_cols.intersection(test_cols)
        else:
            train_cols = set(train_df.select_dtypes(['object', 'category']).columns)
            test_cols = set(test_df.select_dtypes(['object', 'category']).columns)
            cols = train_cols.intersection(test_cols)
        
    for i, col in enumerate(cols):
        #display(HTML('<h3><font id="'+ col + '-ttdc' + '" color="blue">' + str(i+1) + ') ' + col + '</font></h3>'))
        print("\nDatatype : " + str(train_df[col].dtype) )
        print(str(train_df[col].dropna().nunique()) + " unique " + col  + " in Train dataset")
        print(str(test_df[col].dropna().nunique()) + " unique " + col  + " in Test dataset")
        ## Subtracting from test set ....
        extra = len(set(test_df[col].dropna().unique()) - set(train_df[col].dropna().unique()))
        print(str(extra) + " extra " + col + " in Test dataset")
        if extra == 0:
            print('\nAll values present in Test dataset also present in Train dataset for column ' + col)
        else:
            print( '\n'+str(extra) + ' ' +  col + ' are not present in Train dataset which are in Test dataset')

### Barplots for categorical data
def barplot(data_se, title, figsize= (10,10), sort_by_counts=False):
    info = data_se.value_counts()
    info_norm = data_se.value_counts(normalize=True)
    categories = info.index.values
    counts = info.values
    counts_norm = info_norm.values
    fig, ax = plt.subplots(figsize=figsize)
    if data_se.dtype in ['object']:
        if sort_by_counts == False:
            inds = categories.argsort()
            counts = counts[inds]
            counts_norm = counts_norm[inds]
            categories = categories[inds]
        ax = sns.barplot(counts, categories, orient = "h", ax=ax)
        ax.set(xlabel="count", ylabel=data_se.name)
        ax.set_title("Distribution of " + title)
        for n, da in enumerate(counts):
            ax.text(da, n, str(da)+ ",  " + str(round(counts_norm[n]*100,2)) + " %", fontsize=10, va='center')
    else:
        inds = categories.argsort()
        counts_sorted = counts[inds]
        counts_norm_sorted = counts_norm[inds]
        ax = sns.barplot(categories, counts, orient = "v", ax=ax)
        ax.set(xlabel=data_se.name, ylabel='count')
        ax.set_title("Distribution of " + title)
        for n, da in enumerate(counts_sorted):
            ax.text(n, da, str(da)+ ",  " + str(round(counts_norm_sorted[n]*100,2)) + " %", fontsize=10, ha='center')
    plt.show()

### 
def barplot_hue(data_se, target, title, figsize=(10,10), sort_by_counts=False):
    hue_se = target
    if sort_by_counts == False:
        order = data_se.unique()
        order.sort()
    else:
        order = data_se.value_counts().index.values
    off_hue = hue_se.nunique()
    off = len(order)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.countplot(y=data_se, hue=hue_se, order=order, ax=ax)
    ax.set_title(title)
    patches = ax.patches
    for i, p in enumerate(ax.patches):
        x=p.get_bbox().get_points()[1,0]
        y=p.get_bbox().get_points()[:,1]
        total = x
        p = i
        q = i
        while(q < (off_hue*off)):
            p = p - off
            if p >= 0:
                total = total + (patches[p].get_bbox().get_points()[1,0] if not np.isnan(patches[p].get_bbox().get_points()[1,0]) else 0)
            else:
                q = q + off
                if q < (off*off_hue):
                    total = total + (patches[q].get_bbox().get_points()[1,0] if not np.isnan(patches[q].get_bbox().get_points()[1,0]) else 0)
       
        perc = str(round(100*(x/total), 2)) + " %"
        
        if not np.isnan(x):
            ax.text(x, y.mean(), str(int(x)) + ",  " + perc, va='center')
    plt.show()