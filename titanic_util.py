import numpy as np
import pandas as pd
import os
from collections import OrderedDict

from pandas import Series, DataFrame

pi=np.pi; e=np.e

#TODO document everything more thoroughly.

#TODO implement NB, NN, and SVM to make predictions

#TODO consolidate; don't need an extra method for each individual transform

def load(opt='train'):

    '''Easier interface for loading the datafiles at the expense
    of assuming they are in the same directory as the current file.

    Parameters
    ----------

    opt :string
        indicates which csv file (train or test) to be loaded

    Returns
    -------

    dataset : pandas DataFrame
        desired dataset
    
    '''

    cwd = os.getcwd()
    
    if opt=='train':
        address = os.path.join(cwd,'train.csv')
    elif opt=='test':
        address = os.path.join(cwd,'test.csv')
    else:
        raise NameError('"{}" is not a valid loading input.'.format(opt))
    return pd.read_csv(address)


def sep_x_y(data,key):

    '''Separates dataset into relevant features and base truths

    Parameters
    ----------

    data : pandas DataFrame
    	data that is being analyzed

    Returns
    -------

    X : pandas DataFrame
    	features from each instance which are being examined
    y : pandas DataFrame
    	base truths of the data, which will be the target of learning
    '''

    y = data[key]; X = data[data.keys().drop(key)]

    return X,y
    

def mung(data, attaching_keys, base_truth=None):

    '''
    Transforms input data according to a pre-defined set of transformations, 
    converting the data to a more usable form and returns a new DataFrame
    with the transformed data.

    Parameters
    ----------

    data : pandas DataFrame
    	This is the dataset to which the transformations will be applied
    attaching_keys : array-like
    	This specifies which features will be in the resulting DataFrame
    base_truth : string
    	This indicates which feature is to be learned
    
    Returns
    -------

    transformed_data : pandas DataFrame
    	DataFrame containing the specified features from "data" argument
    	after passing through pre-defined transformations
    desired_feature : pandas series
    	Series containing the feature of "data" which is to be learned later

    '''
    
    #TODO implement workflow as a pipeline; or implement transforms as a decorator/parent class
    
    transformed_data = DataFrame(index=data.index)

    for i in attaching_keys:
        transformed_data = attach(transformed_data, data, i)

    transformed_data = feature_scaling(fill_nans(transformed_data))

    if base_truth is not None:
        desired_feature = Series(data[base_truth])

        return transformed_data, desired_feature

    else:
        return transformed_data
    

def attach(target, base_data, data_key):
    
    '''
    Takes data from pre-existing pandas DataFrame and adds it to another
    DataFrame after transforming it

    Parameters
    ----------

    target : pandas DataFrame
    	DataFrame onto which new data is attached
    base_data : pandas DataFrame
    	source of data that will be attached to target
    data_key : str or array of str
    	describes which feature from base_data will be added to target;
    	for derived features, all source features are passed as an array

    Returns
    -------

    target : pandas DataFrame
    	result of adding the transformed data from base_data to target

    '''

    #TODO fix the awkward passing of multiple variables
    
    if Series(data_key).isin(base_data.keys()).all():
        new_data = transform(base_data, data_key)
        if type(new_data) == Series:
            target = target.join(new_data, how='left')
        elif type(new_data) == DataFrame:
            target = pd.concat([target, new_data],axis=1)
        else:
            print('{} is not a valid type for attaching to target data. Requires pandas Series or pandas DataFrame'.format(type(new_data)))
        return target

    else:
        print('Tried to add {}, which is not a key in {}.'.format(data_key, base_data))
        return

def transform(data, key):
    
    '''
    Accepts a DataFrame and the name of a feature and returns the given
    feature of the data after applying the predefined transform for it

    Parameters
    ----------

    data : pandas DataFrame
    	DataFrame which will be used as the source for data to be returned
    key : str or array of strings
   	determines which feature or features will be used to calculate the output

    Returns
    -------

    result : pandas DataFrame or Series
    	the result of passing the specified data through the predefined transformation
    '''
    
    if not (Series(key).isin(data.keys())).all():
        print('{} key is invalid. Try a different one.'.format(key))
        return

    if type(key)==str:
        if key == 'Sex':
            result = sex_transform(data)
        elif key == 'Embarked':
            result = embark_transform(data)
        elif key == 'Pclass':
            result = pclass_transform(data)
        elif key == 'Cabin':
            result = cabin_transform(data)
        else:
            result = Series(data[key])

    if type(key)==np.ndarray:
        if (key == np.array(['Parch','SibSp'])).all():
            result = Series(family_size(data))
        # if (key == np.array(['Sex','SibSp','Age'])).all():
        #     result = married(data)
        else:
            print('The combination {} does not have a transformation for derived data.'.format(key))

    return result


def get_misclass(predictions, base_truth):

    '''
    Determines which indices correspond to false predictions from classifier

    Parameters
    ----------

    predictions : pandas Series
    	contains predictions output by classifier
    base_truth : pandas Series
    	contains actual values of instances passsed to classifier

    Returns
    -------

    ind : pandas Series
    	contains  the indices of instances in which predictions differed 
    	from actual values

    '''
    
    ind = (predictions != base_truth).nonzero()

    return ind


def analyze_misclass(data, predictions, base_truth):

    '''
    Used for determining trends in instances that were misclassified

    Parameters
    ----------

    data : pandas DataFrame
        source of data to be analyzed
    predictions : pandas Series
    	predictions given by the classifier
    base_truth : pandas Series
    	actual values for instances given to classifier 

    Returns
    -------

    analysis : pandas DataFrame
    	contains means and standard deviations of the data as a whole, 
    	as well as the data that was misclassified

    '''

    indices = get_misclass(data, predictions, base_truth)
    
    data_copy = DataFrame(data, copy=True); data_copy.columns = np.arange(data.shape[1])
    data_misclass = DataFrame(data.iloc[indices]); data_misclass.columns = np.arange(data.shape[1])
    
    mu_overall = data_copy.mean(axis=0); mu_misclass = data_misclass.mean(axis=0)
    sig_overall = data_copy.std(axis=0); sig_misclass = data_misclass.std(axis=0)

    mean_overall = mu_overall.to_frame(name='mean_overall')
    mean_misclass = mu_misclass.to_frame(name='mean_misclass')
    std_overall = sig_overall.to_frame(name='std_overall')
    std_misclass = sig_misclass.to_frame(name='std_misclass')
    analysis = pd.concat([mean_overall, mean_misclass, std_overall, std_misclass],
                             join='outer', axis=1)
    analysis.index = data.columns

    return analysis

            
def test_significance(data, key):

    bias = data.groupby(data[key])

    if len(bias.unique()) <= 10:
        plt.bar(bias.unique())
        
    return bias.mean()['Survived']


def sex_transform(data):

    return Series(data['Sex'].replace({'female':1, 'male':0}))
    

def embark_transform(data):

    return one_hot_encoding(Series(data['Embarked']))
    

def pclass_transform(data):

    return one_hot_encoding(Series(data['Pclass']))


def cabin_transform(data):
    
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

    '''
    Gives the F1 score on predictions output by the classifier.

    Parameters
    ----------

    predictions : pandas Series
    	contains predictions output by classifier
    base_truth : pandas Series
    	contains actual values of instances passsed to classifier

    Returns
    -------

    score : float
    	F1 score of predictions on base_truth

    '''
    
    true_pos = (predictions * base_truth).sum()
    all_pos = predictions.astype(bool).sum()
    pos_truth = base_truth.astype(bool).sum()
    
    prec = true_pos / all_pos
    rec = true_pos / pos_truth
    return 2*prec*rec / (prec + rec)


def sigmoid(z):

    '''
    Applies the logistic sigmoid function to data passed in

    Parameters
    ----------

    z : int, float, or numpy or pandas container

    Returns
    -------
    
    sigmoid : type identical to z
    	if z is an int or float, returns the sigmoid function applied to that value;
    	if z is a container, returns that container with the sigmoid function 
    	applied to each element

    '''

    return 1/(1 + np.exp(-z))


def heaviside(X, threshold=0.5):

    '''
    Applies the heaviside step function to argument; returns 1 if value is
    equal to threshold
    '''
    
    step = lambda n : 0 if n < threshold else 1
    
    return X.map(step)
           

def feature_scaling(X):

    '''Normalizes each feature in X to have mean=0 and standard deviation=1'''
    
    mu = np.mean(X,axis=0); sig = np.std(X,axis=0)
    X = (X-mu)/sig
    
    return X


def one_hot_encoding(cat_data):

    '''
    One-hot encodes categorical data passed in argument

    Parameters
    ----------

    cat_data : pandas object
    	contains data that will be one-hot encoded

    Returns
    -------

    encoded : pandas DataFrame
    	DataFrame with one column for each unique value in cat_data, according 
    	to a one-hot encoding

    '''

    return pd.get_dummies(cat_data) 


def fill_nans(data):

    '''
    detects missing values and fills them with the mean of their column

    Parameters
    ----------

    data : pandas DataFrame
    	source of data with missing values

    Returns
    -------

    filled_frame : pandas DataFrame
    	original DataFrame with empty values filled
    '''
    
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

    '''
    Splits data into training set and cross-validation set, including 
    X and y separation

    Parameters
    ----------

    data : pandas DataFrame
    	data to be split up
    base_truth : str
    	key for column in data; this will be the column returned as the target data

    Returns
    -------

    train_X : pandas DataFrame
    	features that the classifier will be trained on

    train_y : pandas Series
    	target data that will be used for training

    cv_X : pandas DataFrame
    	features that the classifier will use for cross-validation predictions

    cv_y : pandas Series
    	target data that will be used for cross-validation

    '''
    
    #TODO split into train, cv, and test

    total_examples = data.shape[0]

    train_number = int(portion_train * total_examples)
    train_index = np.sort(np.random.choice(data.index, train_number, replace=False))

    train_X = data.loc[train_index]
    cv_X = data.iloc[~data.index.isin(train_index)]

    train_y = base_truth.loc[train_index]
    cv_y = base_truth.iloc[~base_truth.index.isin(train_index)]

    return train_X, train_y, cv_X, cv_y


def learning_curves(X, y, classifier, min_val=10, max_val=100,
                    step=10, metric='f1_score', runs_per_step=3,
                    print_stats=False):

    '''
    Trains classifier on a range of different training set sizes
    to demonstrate how its predictions change over time

    Parameters
    ----------

    X : pandas DataFrame
    	data which will be used for training

    y : pandas Series
    	data which will be used as target values for training
    
    classifier : object
    	contains an instance of a classifier to be trained on data    

    min_val : int
    	smallest dataset size to be used for training classifier

    max_val : int
    	maximum dataset size to be used for training classifier

    step : int
    	step length for range of dataset sizes

    metric : str
    	determines what performance measure will be used to determine 
    	training and cv error

    runs_per_step : int
    	number of runs per dataset size; performances are averaged before 
    	being recorded

    print_stats : bool
    	if true, the error and number of iterations needed to converge is 
	displayed before being returned and the num

    Returns
    ------

    train_error : OrderedDict
    	contains training errors indexed by the size of the dataset used to produce them

    cv_error : float
	contains cv errors indexed by the size of the dataset used to produce them

    '''
    
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

            # if avg_type is not None:
            #     train_pred = classifier.predict(X_train, params, avg_type=avg_type)
            #     cv_pred = classifier.predict(X_val, params, avg_type=avg_type)

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

    return train_error, cv_error


def get_error(pred, y, metric='f1_score', show_differences=False):

    '''
    gets prediction error for a given set of data

    Parameters
    ----------

    pred : pandas Series
    	predictions made by classifier

    y : pandas Series
    	target values classifier was trying to predict

    metric : str
    	metric used for claculating error (accuracy or f1_score)

    show_differences : bool
    	if true, displays statistics on means and standard deviations of misclassified data

    Returns
    -------

    err : float
    	prediction error on given data
    '''
    
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
