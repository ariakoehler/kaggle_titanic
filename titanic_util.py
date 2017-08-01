import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pandas import Series, DataFrame

pi=np.pi; e=np.e

#TODO document everything more thoroughly.

def load(opt='train'):

    '''Simplifies the loading of various csv files'''
    
    if opt=='train':
        address = '~/Dropbox/Education/Coding/Kaggle/titanic/train.csv'
    elif opt=='test':
        address = '~/Dropbox/Education/Coding/Kaggle/titanic/test.csv'
    else:
        raise NameError('"{}" is not a valid loading input.'.format(opt))
    return pd.read_csv(address)


def sep_x_y(data,key):

    '''Separates data into features and outputs'''

    y = data[key]; X = data[data.keys().drop(key)]

    return X,y
    

def mung(data, attaching_keys, base_truth=None):

    '''Performs preprocessing on the data at hand and converts data to relevant types'''

    #TODO implement workflow as a pipeline
    
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
        else:
            print('The combination {} does not have a transformation for derived data.'.format(key))


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
    
    # init = '\D.*'; trans = lambda s : re.search('\D', s).group()

    return one_hot_encoding(Series(data['Cabin']).str.extract('(\D)', expand=False))# {s : re.search('\D', s).group()}) 


def family_size(data):

    '''Takes the sibsp and parch data and gets total family size'''
    
    return Series(data['SibSp'] + data['Parch']).rename('FamSize')


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
    cost_log = np.zeros((X.shape[0],max_it))

    for i in range(max_it):
        cost_log[:,i] = cost(X,y,theta,reg)
        gradient = grad(X,y,theta,reg)
        theta = (theta - alpha * gradient)
        
    return theta, cost_log


def split_train_cv(data, base_truth, portion_train=0.8):

    total_examples = data.shape[0]

    train_number = int(portion_train * total_examples)
    train_index = np.sort(np.random.choice(data.index, train_number, replace=False))
    train_set = data.iloc[train_index]; cv_set = data.iloc[~data.index.isin(train_index)]
    train_truth = base_truth.iloc[train_index]; cv_truth = base_truth.iloc[~base_truth.index.isin(train_index)]
    
    return train_set, train_truth, cv_set, cv_truth


    
class LogisticRegressor():

    def __init__(self, lambd=1, tol=10**(-4), alpha=0.01, max_it=100):
        self.max_it=max_it
        self.lambd=lambd
        self.tol=tol
        self.alpha=alpha

    def theta_init(self,length):

        '''Initializes the values of theta before perfroming gradient descent'''
        
        theta = Series(np.zeros(length))
        
        return theta
        
    def cost_function(self, X, y, theta, lambd):
        
        theta_reg = theta.copy(); theta_reg[0] = 0
        m = len(y)
        
        def hypothesis(X):
            return (sigmoid(np.dot(X,theta)))
        
        return (1/(2*m) * sum( -y*np.log(hypothesis(X)) - (1 - y) * np.log(1 - hypothesis(X))) ) + (lambd/(2*m)) * sum( theta_reg**2 )

            
    def gradient(self, X, y, theta, lambd):

        theta_reg = theta.copy(); theta_reg[0] = 0; m = len(y)

        def hypothesis(X):
            return sigmoid(np.matmul(X,theta))

        return (1/m) * Series((np.matmul(hypothesis(X) - y, X))  +  (lambd/m) * theta_reg)


    def fit(self, X, y, portion_train=0.8, display_cost_log=False):

        X_train, y_train, X_val, y_val = split_train_cv(X, y, portion_train=portion_train)

        self.weights, cost_log = self.fit_grad_desc(X_train, y_train)
        if display_cost_log is not False:
            display(cost_log)

        self.train_error = self.get_error(X_train, y_train)
        self.cv_error = self.get_error(X_val, y_val)

        
    def fit_grad_desc(self, X, y):
        
        theta_len = X.shape[1]; theta_init = self.theta_init(theta_len)
       
        theta_result, cost_log = grad_desc(X, y, theta_init, self.lambd,
                                 self.cost_function, self.gradient, tol=self.tol,
                                 max_it=self.max_it, alpha=self.alpha)

        return theta_result, cost_log
        

    def get_error(self, X, y, metric='accuracy'):

        pred = self.predict(X)
        err = (1 - np.mean(pred == y))*100
        return err
        
        
    def predict(self, X):

        predictions = Series(heaviside(sigmoid(Series(np.matmul(X,self.weights))), threshold=0.5))
        predictions.index = X.index

        return predictions



# class CVGridSearch():

#     def __init__():


class PCA():

    #TODO find a way to initialize pc_count at the same place every time

    #TODO be more specific with exception messages
    
    def __init__(self, task_type='visualize',
                 pc_count='detect'):

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


    def project(self, X):

        if self.pc_count is 'detect':
            self.pc_count = detect_principal_components(self, X, tol=0.05)
        
        Z =  np.matmul(X, self.enc_mat[:,:self.pc_count])

        if self.task_type is 'visualize':
            plt.scatter(Z[:,0], Z[:,1])
            plt.show()
        elif self.task_type is 'reduce_dim':
            return Z
    
    def fit(self, X):

        self.decompose(X)
        if self.task_type == 'reduce_dim':
            return self.project(X)
        elif self.task_type == 'visualize':
            self.project(X)
        
    # def detect_principal_components(self, X, tol=0.05):

        
