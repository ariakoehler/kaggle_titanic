import numpy as np
import pandas as pd

from pandas import Series, DataFrame
from titanic_util import * 

pi=np.pi; e=np.e


def grad_desc(X, y, theta_init, reg, cost, grad, tol=10**(-4), max_it=100, alpha=0.001,
              display_cost=False, print_stats=False):
    
    '''Implements gradient descent given variables from a particular optimization problem.'''

    theta = theta_init
    cost_log = Series(0, index=np.arange(max_it))
    
    for i in range(max_it):
        cost_log.iloc[i] = cost(X,y,theta,reg)
        gradient = grad(X,y,theta,reg)
        theta = (theta - alpha * gradient)
        if abs(cost_log.iloc[i] - cost_log.iloc[i-1]) < tol: break

    if display_cost is True:
        display(cost_log)

    if print_stats is True:
        print('converged after {} iterations.'.format(np.count_nonzero(cost_log)))
    
    return theta


class LogisticRegressor():

    #TODO implement preprocessing as part of regressor or in a pipeline
    
    #TODO also use existing ml libraries to do predictions, instead of building my own, shitty implementations
    
    def __init__(self, lambd=1, tol=10**(-4), alpha=0.01, max_it=100):
        self.max_it=max_it
        self.lambd=lambd
        self.tol=tol
        self.alpha=alpha

    def theta_init(self,X):

        '''Initializes the values of theta before perfroming gradient descent'''
        
        theta = Series(np.zeros(X.shape[1]), index=X.columns)
        theta.index = X.columns
        theta.loc['Bias'] = 0
        
        return theta
        
    def cost_function(self, X, y, theta, lambd):
        
        theta_reg = theta.copy()
        theta_reg.loc['Bias'] = 0
        m = len(y)

        hypothesis = lambda X : sigmoid(X.dot(theta))
        
        return (1/(2*m) * sum( -y*np.log(hypothesis(X)) - (1 - y) * np.log(1 - hypothesis(X))) ) + (lambd/(2*m)) * sum( theta_reg**2 )

    
    def gradient(self, X, y, theta, lambd):

        theta_reg = theta.copy()
        theta_reg.loc['Bias'] = 0
        m = len(y)
        
        return (1/m) * (sigmoid(X.dot(theta)) - y).dot(X) + (lambd/m) * theta_reg


    def fit(self, X, y, run_count=0, return_differences=False, method='batch',
            print_stats=False):

        X_bias = Series(1, index=X.index, name='Bias')
        X = pd.concat([X_bias, X], axis=1)        

        if method == 'batch':
            params = self.fit_grad_desc(X, y, print_stats=print_stats)
        else:
            raise NotImplementedError('That fitting method either has not been implemented yet or will not be implemented.')

        return params
        
    def fit_grad_desc(self, X, y, print_stats=False):

        #TODO implement stochastic with variable batch size
        
        theta_len = X.shape[1]
        theta_init = self.theta_init(X)
      
        theta_result = grad_desc(X, y, theta_init, self.lambd,
                                 self.cost_function, self.gradient, tol=self.tol,
                                 max_it=self.max_it, alpha=self.alpha, print_stats=print_stats)

        return theta_result
   
        
    def predict(self, X, theta):

        predictions = heaviside(self.predict_proba(X, theta))
        
        return predictions

    
    def predict_proba(self, X, theta):

        X_bias = Series(1, index=X.index, name='Bias')
        X = pd.concat([X_bias, X], axis=1)
        
        predictions = sigmoid(X.dot(theta))

        return predictions

