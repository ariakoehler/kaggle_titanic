import numpy as np
import pandas as pd

from pandas import Series, DataFrame

pi=np.pi; e=np.e


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


    def project(self, X, print_weights=False):

        if self.pc_count is 'detect':
            self.pc_count = detect_principal_components(self, X, tol=0.05)

        principal_components = self.enc_mat[:,:self.pc_count]
            
        Z = np.matmul(X, principal_components)

        if print_weights is True:
            print(principal_components)

        return Z
    
    def fit(self, X):

        self.decompose(X)
        if self.task_type == 'reduce_dim':
            return self.project(X)
        elif self.task_type == 'visualize':
            self.project(X)

