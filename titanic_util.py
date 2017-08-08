import numpy as np
import pandas as pd
import os
from collections import OrderedDict

from pandas import Series, DataFrame

pi=np.pi; e=np.e

#TODO split into multiple files

#TODO document everything more thoroughly.

#TODO implement NB, NN, and SVM to make predictions

def load(opt='train'):

    '''Simplifies the loading of various csv files'''

    cwd = os.getcwd()
    
    if opt=='train':
        address = os.path.join(cwd,'train.csv')
    elif opt=='test':
        address = os.path.join(cwd,'test.csv')
    else:
        raise NameError('"{}" is not a valid loading input.'.format(opt))
    return pd.read_csv(address)


def sep_x_y(data,key):

    '''Separates data into features and outputs'''

    y = data[key]; X = data[data.keys().drop(key)]

    return X,y
    

def mung(data, attaching_keys, base_truth=None):

    '''Performs preprocessing on the data at hand and converts data to relevant types'''

    #TODO implement workflow as a pipeline; or implement transforms as a decorator/parent class
    
    #TODO account for titles; analyze ethnicity? Maybe perform clustering on letter frequency and two-letter pairs; correlation between other family members living or dying, including considerations for age

    target = DataFrame(index=data.index)

    for i in attaching_keys:
        target = attach(target,data,i)

    target = feature_scaling(fill_nans(target))

    if base_truth is not None:
        true_value = Series(data[base_truth])
        return target, true_value
    else:
        return target
    

def attach(target, base_data, data_key):
    
    '''Accepts a two datasets and a key as args; 
    transforms subset of second dataset then adds it to first;
    returns appended dataset'''

    #TODO fix the awkward passing of multiple variables
    
    if Series(data_key).isin(base_data.keys()).all():
        new_data = transform(base_data, data_key)
        if type(new_data) == Series:
            target = target.join(new_data, how='left')
        elif type(new_data) == DataFrame:
            target = pd.concat([target, new_data],axis=1)
        else:
            print('{} is not a valid type for attaching to target data. Requires pandas Series or pandas DataFrame'.format(type(new_data)))

    else:
        print('Tried to add {}, which is not a key in {}.'.format(data_key, base_data))
        return
    
    return target

def transform(data, key):

    '''Accepts data and key as args; returns the proper
    transformation applied to specified subset of data as 
    a pandas Series. If no transformation, returns just 
    the subset as as pandas Series.'''

    if not (Series(key).isin(data.keys())).all():
        print('{} key is invalid. Try a different one.'.format(key))
        return

    if type(key)==str:
        if key == 'Sex':
            return sex_transform(data)
        elif key == 'Embarked':
            return embark_transform(data)
        elif key == 'Pclass':
            return pclass_transform(data)
        elif key == 'Cabin':
            return cabin_transform(data)
        else:
            return Series(data[key])

    if type(key)==np.ndarray:
        if (key == np.array(['Parch','SibSp'])).all():
            return Series(family_size(data))
        if (key == np.array(['Sex','SibSp','Age'])).all():
            return married(data)
        else:
            print('The combination {} does not have a transformation for derived data.'.format(key))


def get_misclass(data, predictions, base_truth):

    '''takes in data and tells you where prediction failed, so you can figure out new feature to add or variables to account for'''

    return (predictions != base_truth).nonzero()


def analyze_misclass(data, predictions, base_truth, display_an=True, return_an=False):

    indices = get_misclass(data, predictions, base_truth)
    
    data_copy = DataFrame(data, copy=True); data_copy.columns = np.arange(data.shape[1])
    data_misclass = DataFrame(data.iloc[indices]); data_misclass.columns = np.arange(data.shape[1])
    
    mu_overall = data_copy.mean(axis=0); mu_misclass = data_misclass.mean(axis=0)
    sig_overall = data_copy.std(axis=0); sig_misclass = data_misclass.std(axis=0)

    if display_an == True or return_an == True:
        mean_overall = mu_overall.to_frame(name='mean_overall')
        mean_misclass = mu_misclass.to_frame(name='mean_misclass')
        std_overall = sig_overall.to_frame(name='std_overall')
        std_misclass = sig_misclass.to_frame(name='std_misclass')
        analysis = pd.concat([mean_overall, mean_misclass, std_overall, std_misclass],
                             join='outer', axis=1)
        analysis.index = data.columns
    
    if display_an == True:
        display(analysis)

    if return_an == True:
        return analysis

            
def test_significance(data, key):

    '''For determining how relevant certain features are'''

    bias = data.groupby(data[key])

    if len(bias.unique()) <= 10:
        plt.bar(bias.unique())
        
    return bias.mean()['Survived']


def sex_transform(data):

    '''Takes the string variable for sex and converts it to boolean, 
    where 1 is female and 0 is male.'''

    return Series(data['Sex'].replace({'female':1, 'male':0}))
    

def embark_transform(data):

    '''Converts the location of embarkation to an int for use in logistic regression'''

    return one_hot_encoding(Series(data['Embarked']))
    

def pclass_transform(data):

    return one_hot_encoding(Series(data['Pclass']))


def cabin_transform(data):

    '''Selects the cabin area and one-hot encodes it'''
    
    return one_hot_encoding(Series(data['Cabin']).str.extract('(\D)', expand=False))


# def title(data):

# one-hot encode people's titles(e.g. Mr., Mrs., Rev., Dr., Lord, HRH)

#     return


# def married(data):

# select for anyone above 18 who has 1 sibsp; split into m and f

#     return


def family_size(data):

    '''Takes the sibsp and parch data and gets total family size'''
    
    return Series(data['SibSp'] + data['Parch']).rename('FamSize')


def f1_score(predictions, base_truth):
    
    true_pos = (predictions * base_truth).sum()
    all_pos = predictions.astype(bool).sum()
    pos_truth = base_truth.astype(bool).sum()
    
    prec = true_pos / all_pos
    rec = true_pos / pos_truth
    return 2*prec*rec / (prec + rec)


def sigmoid(z):

    '''Applies the sigmoid function. Needed for logistic regression.'''

    return 1/(1 + np.exp(-z))


def heaviside(X, threshold=0.5):

    '''Applies the heaviside step function, using the half-maximum convention'''

    step = lambda n : 0 if n < threshold else 1
    
    return X.map(step)
           

def feature_scaling(X):

    mu = np.mean(X,axis=0); sig = np.std(X,axis=0)
    X = (X-mu)/sig
    
    return X


def one_hot_encoding(cat_data):

    return pd.get_dummies(cat_data) 


def fill_nans(data):

    #TODO find better way to fill missing values; try multiple imputation or maximum, a la http://www.bu.edu/sph/files/2014/05/Marina-tech-report.pdf sections 4.2.1 and 4.2.2
    
    incomplete = np.array(data)
    
    for i in range(incomplete.shape[1]):
        empty = np.isnan(incomplete[:,i]).astype(int)
        if empty.any():
            filled = np.delete(incomplete[:,i], np.nonzero(empty))
            mu = np.mean(filled)
            incomplete[np.nonzero(empty),i] = mu * empty[np.nonzero(empty)]
    
    return DataFrame(incomplete, columns=data.columns)


def split_train_cv(data, base_truth, portion_train=0.8):

    #TODO split into train, cv, and test

    total_examples = data.shape[0]

    train_number = int(portion_train * total_examples)
    train_index = np.sort(np.random.choice(data.index, train_number, replace=False))

    train_set = data.loc[train_index]
    cv_set = data.iloc[~data.index.isin(train_index)]

    train_truth = base_truth.loc[train_index]
    cv_truth = base_truth.iloc[~base_truth.index.isin(train_index)]
    
    return train_set, train_truth, cv_set, cv_truth


def learning_curves(X, y, classifier, min_val=10, max_val=100,
                    step=10, metric='f1_score', avg_type=None,
                    runs_per_step=3, print_stats=False):
    
    max_val = X.shape[0]
    iterator = np.arange(min_val, max_val, step)
        
    train_error = OrderedDict({}); cv_error = OrderedDict({})
    
    for i in iterator:

        i_train_err = []
        i_cv_err = []

        for j in range(runs_per_step):

            subset_ix = np.random.choice(np.arange(max_val), i, replace=False)
            X_subset = X.iloc[subset_ix]
            y_subset = y.iloc[subset_ix]

            X_train, y_train, X_val, y_val = split_train_cv(X_subset, y_subset)
        
            params = classifier.fit(X_train, y_train, print_stats=print_stats)

            if avg_type is not None:
                train_pred = classifier.predict(X_train, params, avg_type=avg_type)
                cv_pred = classifier.predict(X_val, params, avg_type=avg_type)

            train_pred = classifier.predict(X_train, params)
            cv_pred = classifier.predict(X_val, params)

            i_train_err.append(get_error(train_pred, y_train, metric=metric))
            i_cv_err.append(get_error(cv_pred, y_val, metric=metric))
        
        train_error[i] = np.mean(i_train_err)
        cv_error[i] = np.mean(i_cv_err)

        if print_stats is True:
            print('{} is {} on training.'.format(metric, 1 - train_error[i]))
            print('{} is {} on cv.'.format(metric, 1 - cv_error[i]))
            print('Run with set of size {} successful.'.format(i))

    return params, train_error, cv_error


def get_error(pred, y, metric='f1_score', show_differences=False):

    #TODO make proper call to analyze_misclass that includes all kwargs

    if metric == 'accuracy':
        err = (1 - np.mean(pred == y))*100
    elif metric == 'f1_score':
        err = 1 - f1_score(pred, y)
    else:
        NotImplementedError('That metric has not yet been implemented.') 
    
    if show_differences == True:
        analyze_misclass(X, pred, y)
        
    return err


    #TODO automate cross-validation search for hyperparams
