import numpy as np
import pandas as pd

from pandas import Series, DataFrame
from titanic_util import * 

pi=np.pi; e=np.e


def grad_desc(X, y, theta_init, reg, cost, grad, tol=10**(-4), max_it=100, learning_rate=0.001,
              display_cost=False, print_stats=False):
    
    '''
    Implements gradient descent given information from a particular optimization problem.

    Parameters
    ----------

    X : pandas DataFrame
    	training data to be used for GD
    y : pandas Series
    	base truths used for training in GD
    reg : float
    	defines the L2 regularization parameter
    cost : function
    	defines the cost function that GD sets out to optimizes
    grad : function
    	gradient of cost function for calculating gradient steps
    tol : float
    	tolerance for determining convergence; optimization stops once difference in cost between steps is less than tol
    max_it : int
    	maximum number of iterations in case convergence doesn't happen or happens very slowly
    learning_rate : float
    	GD's learning rate; determines size of gradient steps in each iteration
    display_cost : bool
    	if true, will display history of costs in each iteration after running GD
    print_stats : bool
    	if true, displays how many iterations were required to reach convergence

    Returns
    -------

    theta : pandas Series
    	optimal weights determined for each feature by GD
    
    '''

    theta = theta_init
    cost_log = Series(0, index=np.arange(max_it))
    
    for i in range(max_it):
        cost_log.iloc[i] = cost(X,y,theta,reg)
        gradient = grad(X,y,theta,reg)
        theta = (theta - learning_rate * gradient)
        if abs(cost_log.iloc[i] - cost_log.iloc[i-1]) < tol: break

    if display_cost is True:
        display(cost_log)

    if print_stats is True:
        print('converged after {} iterations.'.format(np.count_nonzero(cost_log)))
    
    return theta


class LogisticRegressor():

    '''
    Supports the process of gradient descent through logistic regression
    '''
    
    #TODO also use existing ml libraries to do predictions, instead of building my own, shitty implementations
    
    def __init__(self, l2_reg=1, tol=10**(-4), learning_rate=0.01, max_it=100):

        '''
        Parameters
        ----------

        l2_reg : float
            L2 regularization parameter for GD
        tol : float
            tolerance for determining convergence; optimization stops once difference in cost between steps is less than tol
        max_it : int
            maximum number of iterations in case convergence doesn't happen or happens very slowly
        learning_rate : float
            GD's learning rate; determines size of gradient steps in each iteration
        '''
        
        self.max_it=max_it
        self.l2_reg=l2_reg
        self.tol=tol
        self.learning_rate=learning_rate

    def theta_init(self,X):

        '''Initializes the values of theta to the zero vector before perfroming gradient descent'''
        
        theta = Series(np.zeros(X.shape[1]), index=X.columns)
        theta.index = X.columns
        theta.loc['Bias'] = 0
        
        return theta
        
    def cost_function(self, X, y, theta, l2_reg):

        '''
        Implements the cost function for logistic regression, which is to be optimized

        Parameters
        ----------

        X : pandas DataFrame
            relevant features from the dataset for calculating cost of theta
        y : pandas Series
            target values for calculating cost
        theta : pandas Series
            current weights used to determine cost
        l2_reg : float
            L2 regularization for theta

        Returns
        -------

        cost : float
            logistic regression cost of current theta
        '''
        
        theta_reg = theta.copy()
        theta_reg.loc['Bias'] = 0
        m = len(y)

        hypothesis = lambda X : sigmoid(X.dot(theta))
        
        return (1/(2*m) * sum( -y*np.log(hypothesis(X)) - (1 - y) * np.log(1 - hypothesis(X))) ) + (l2_reg/(2*m)) * sum( theta_reg**2 )

    
    def gradient(self, X, y, theta, l2_reg):
        
        '''
        Implements the gradient function for logistic regression, which is to be optimized

        Parameters
        ----------

        X : pandas DataFrame
            relevant features from the dataset for calculating gradient at theta
        y : pandas Series
            target values for calculating gradient at theta
        theta : pandas Series
            current weights used to determine gradient
        l2_reg : float
            L2 regularization for theta

        Returns
        -------

        gradient : float
            gradient of logistic regression cost function at current theta
        '''
        
        theta_reg = theta.copy()
        theta_reg.loc['Bias'] = 0
        m = len(y)
        
        return (1/m) * (sigmoid(X.dot(theta)) - y).dot(X) + (l2_reg/m) * theta_reg


    def fit(self, X, y, method='batch', print_stats=False):

        '''
        Fits optimal weights to provided training data

        Parameters
        ----------

        X : pandas DataFrame
            training 
        y : pandas Series
            training base truths  
        method : string
            determines how optimization will occur (batch, stochastic, etc.)
        print_stats : bool
            if true, shows number of iterations needed to reach convergence in GD

        Returns
        -------

        params : pandas Series
            optimal weights determined by selected method of optimization

        '''
        
        X_bias = Series(1, index=X.index, name='Bias')
        X = pd.concat([X_bias, X], axis=1)        

        if method == 'batch':
            params = self.fit_grad_desc(X, y, print_stats=print_stats)
        else:
            raise NotImplementedError('That fitting method either has not been implemented yet or will not be implemented.')

        return params
        
    def fit_grad_desc(self, X, y, print_stats=False):
        
        theta_len = X.shape[1]
        theta_init = self.theta_init(X)
      
        theta_result = grad_desc(X, y, theta_init, self.l2_reg,
                                 self.cost_function, self.gradient, tol=self.tol,
                                 max_it=self.max_it, learning_rate=self.learning_rate, print_stats=print_stats)

        return theta_result
   
        
    def predict(self, X, theta):

        '''
        Makes predictions on a given dataset with a given set of weights

        Parameters
        ----------

        X : pandas DataFrame
            dataset for making predictions
        theta : pandas Series
            weights for determining what to predict on X

        Returns
        -------

        predictions : pandas Series
            predictions made by classifier on X, given theta
        
        '''

        predictions = heaviside(self.predict_proba(X, theta))
        
        return predictions

    
    def predict_proba(self, X, theta):

        X_bias = Series(1, index=X.index, name='Bias')
        X = pd.concat([X_bias, X], axis=1)
        
        predictions = sigmoid(X.dot(theta))

        return predictions

