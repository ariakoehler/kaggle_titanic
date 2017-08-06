import numpy as np
import pandas as pd
import os
from collections import OrderedDict

import matplotlib.pyplot as plt

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
    
    #age in context of family; do children in large families have better chance of suvival than children in small families etc.
      
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


def heaviside(x, threshold=0):

    '''Applies the heaviside step function, using the half-maximum convention'''

    ret_vec = np.zeros(len(x))

    for i in range(len(x)):
        if x.iloc[i]<threshold:
            ret_vec[i]=0
        elif x.iloc[i]>=threshold:
            ret_vec[i]=1

    return Series(ret_vec)
            

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

    
def grad_desc(X, y, theta_init, reg, cost, grad, tol=10**(-4), max_it=100, alpha=0.001):
    
    '''Implements gradient descent given variables from a particular optimization problem.'''

    theta = theta_init
    cost_log = np.zeros(max_it)
    finish=max_it
    
    for i in range(max_it):
        cost_log[i] = cost(X,y,theta,reg)
        gradient = grad(X,y,theta,reg)
        theta = (theta - alpha * gradient)
        if abs(cost_log[i] - cost_log[i-1]) < tol:
            break

    print('converged after {} iterations.'.format(np.count_nonzero(cost_log)))
    
    return theta, cost_log


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


    
class LogisticRegressor():

    #TODO implement preprocessing as part of regressor or in a pipeline
    
    #TODO also use existing ml libraries to do predictions, instead of building my own, shitty implementations
    
    def __init__(self, lambd=1, tol=10**(-4), alpha=0.01, max_it=100, metric='accuracy'):
        self.metric=metric
        self.max_it=max_it
        self.lambd=lambd
        self.tol=tol
        self.alpha=alpha

    def theta_init(self,length):

        '''Initializes the values of theta before perfroming gradient descent'''
        
        theta = Series(np.zeros(length))
        
        return theta
        
    def cost_function(self, X, y, theta, lambd):
        
        theta_reg = theta.copy(); theta_reg[0] = 0; m = len(y)

        hypothesis = lambda X : sigmoid(np.dot(X,theta))
        
        return (1/(2*m) * sum( -y*np.log(hypothesis(X)) - (1 - y) * np.log(1 - hypothesis(X))) ) + (lambd/(2*m)) * sum( theta_reg**2 )


    def learning_curves(self, X, y, portion_train=0.8, run_count=1,
                        min_val=10, max_val=100, step=10):

        max_val=X.shape[0]
        
        train_error = OrderedDict({}); cv_error = OrderedDict({})
        
        for i in np.arange(min_val, max_val, step):
            subset_ix = np.random.choice(np.arange(max_val), i, replace=False)
            X_subset = X.iloc[subset_ix]
            y_subset = y.iloc[subset_ix]
            weights, train_error[i], cv_error[i] = self.fit(X_subset, y_subset,
                                                   portion_train=portion_train,
                                                   run_count=run_count)
            print('Run with set of size {} successful.'.format(i))

        return weights, train_error, cv_error

    
    def gradient(self, X, y, theta, lambd):

        theta_reg = theta.copy(); theta_reg[0] = 0; m = len(y)

        hypothesis = lambda X : sigmoid(np.dot(X,theta))
        
        return (1/m) * Series((np.matmul(hypothesis(X) - y, X))  +  (lambd/m) * theta_reg)


    def fit(self, X, y, portion_train=0.8, run_count=0,
            display_cost_log=False, return_differences=False, return_errors=False):

        # TODO Average over multiple runs to get more stable results; will require changing theta from being member data 
        
        X_train, y_train, X_val, y_val = split_train_cv(X, y, portion_train=portion_train)

        weights, cost_log = self.fit_grad_desc(X_train, y_train)

        if display_cost_log is not False:
            display(cost_log)

        if return_differences is True and return_errors is False:
            raise NotImplementedError('Cannot return differences and not return errors.')

        #TODO clean up this API.
        
        if return_errors is True:
            if return_differences is True:
                return weights, self.get_error(X_train, y_train, weights, show_differences=True), self.get_error(X_val, y_val, weights, show_differences=True) 
            else:
                return weights, self.get_error(X_train, y_train, weights), self.get_error(X_val, y_val, weights)

        else:
            return weights

        
    def fit_grad_desc(self, X, y):

        #TODO implement stochastic and mini-batch
        
        theta_len = X.shape[1]; theta_init = self.theta_init(theta_len)
       
        theta_result, cost_log = grad_desc(X, y, theta_init, self.lambd,
                                 self.cost_function, self.gradient, tol=self.tol,
                                 max_it=self.max_it, alpha=self.alpha)

        return theta_result, cost_log
   

    def get_error(self, X, y, theta, metric='accuracy', show_differences=False):

        #TODO make proper call to analyze_misclass that includes all kwargs

        pred = self.predict(X, theta)
        metric = self.metric

        if metric == 'accuracy':
            err = (1 - np.mean(pred == y))*100
        elif metric == 'f1_score':
            err = 1 - f1_score(pred, y)
        else:
            NotImplementedError('That metric has not yet been implemented.')

            
        if show_differences == True:
            analyze_misclass(X, pred, y)
        
        return err
        
        
    def predict(self, X, theta):

        predictions = Series(heaviside(sigmoid(Series(np.matmul(X,theta))), threshold=0.5))
        predictions.index = X.index
            
        return predictions

    def predict_proba(self, X, theta):

        predictions = Series(sigmoid(Series(np.matmul(X,theta))))

        predictions.index = X.index
        
        return predictions


    #TODO automate cross-validation search for hyperparams

class PCA():

    #TODO find a way to initialize pc_count at the same place every time

    #TODO detect number of PC's to use based on threshhold of variance
    
    def __init__(self, task_type='visualize',
                 pc_count='detect'):

        #TODO be more specific with exception messages

        if task_type != 'visualize' and task_type != 'reduce_dim':
            raise ValueError('That is not a valid option for task_type.')
        else:
            self.task_type = task_type

        if pc_count != 'detect' and type(pc_count) != int:
            raise ValueError('That is not a valid option for pc_count.')
        if pc_count != 'detect':
            self.pc_count = pc_count
        
        
    def decompose(self, X):

        m = X.shape[0]
        Sigma = (1/m) * np.matmul(X.T,X)
        U,S,V = np.linalg.svd(Sigma, compute_uv=True)
        
        self.enc_mat = U
        self.eig_val_mat = S
        self.dec_mat = V


    def project(self, X, print_weights=True):

        if self.pc_count is 'detect':
            self.pc_count = detect_principal_components(self, X, tol=0.05)

        principal_components = self.enc_mat[:,:self.pc_count]
            
        Z =  np.matmul(X, principal_components)

        if print_weights is True:
            print(principal_components)

        return Z
    
    def fit(self, X):

        self.decompose(X)
        if self.task_type == 'reduce_dim':
            return self.project(X)
        elif self.task_type == 'visualize':
            self.project(X)


class VotingEnsemble():

    def __init__(self, classifier, number_runs=3):
        self.classifier = classifier
        self.number_runs = number_runs

    def fit(self, X, y, portion_train=0.8, avg_type='geometric',
            weighted=False):

        results = np.zeros((self.number_runs, X.shape[1]))
        errors = np.zeros(self.number_runs)

        for i in range(self.number_runs):
            # print(type(self.classifier))
            results[i,:] = self.classifier.fit(X, y, portion_train=portion_train,
                                               return_errors=False)
            errors[i] = self.classifier.get_error(X, y, results[i,:])

        if weighted is True:
            weights = 1 - errors
            return results, weights
        else:
            return results

    def predict(self, X, params, weights=None, avg_type='geometric',
                return_errors=True):

        #TODO implement different averaging strategies
        
        predictions = np.zeros((X.shape[0], self.number_runs))
        
        for i in range(self.number_runs):
            predictions[:,i] = self.classifier.predict_proba(X, params[i])

        if weights is not None:
            if avg_type == 'geometric':
                mean = (np.prod(predictions ** weights, axis=1) ** (1/np.sum(weights)))
                pred = heaviside(Series(mean), threshold=0.5)

        return pred

    def get_error(self, X, y, predictions, metric='accuracy'):

        if metric == 'accuracy':
            err = (1 - np.mean(predictions == y))*100
        elif metric == 'f1_score':
            err = 1 - f1_score(predictions, y)
        else:
            NotImplementedError('That metric has not yet been implemented.')

        return err
    
