import numpy as np
import pandas as pd

from pandas import Series, DataFrame

pi=np.pi; e=np.e


class PCA():

    '''
    Implements the process of Principal Component Analysis
    '''
    
    #TODO find a way to initialize pc_count at the same place every time

    #TODO detect number of PC's to use based on threshhold of variance
    
    def __init__(self, pc_count='detect'):

        '''
        Parameters
        ----------

        pc_count : object
        	number of principal components to use; if 'detect,' will be chosen
        	later to retains a certain quantity of variance;       
        '''
        
        #TODO be more specific with exception messages

        if pc_count != 'detect' and type(pc_count) != int:
            raise ValueError('That is not a valid option for pc_count.')
        else:
            self.pc_count = pc_count

            
        
    def decompose(self, X):

        '''
        applies singular value decomposition covariance matrix of data used for PCA

        Parameters
        ----------

        X : pandas DataFrame
        	data which will be reduced in dimension
        '''
        
        m = X.shape[0]
        Sigma = (1/m) * np.matmul(X.T,X)
        U,S,V = np.linalg.svd(Sigma, compute_uv=True)
        
        self.enc_mat = U
        self.eig_val_mat = S
        self.dec_mat = V


    def project(self, X, print_weights=False):

        '''
        projects data onto principal components of the data

        Parameters
        ----------

        X : pandas DataFrame
            data which will be projected onto principal components
        print_weights : bool
            if true, will print the principal components to show feature importance for each one

        Returns
        -------

        Z : pandas DataFrame
            contains projected data

        '''
        
        if self.pc_count is 'detect':
            self.pc_count = detect_principal_components(self, X, tol=0.05)

        principal_components = self.enc_mat[:,:self.pc_count]
            
        Z = np.matmul(X, principal_components)

        if print_weights is True:
            print(principal_components)

        return Z
    
    def fit(self, X):

        '''Wrapper function for whole process of PCA and projection'''
        
        self.decompose(X)

        return self.project(X)
        
