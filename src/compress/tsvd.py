import scipy as scp
import numpy as np

class TSVD:
    def __init__(self, D, rank, return_approx=False):
        self.D  = D
        self.rank = rank
        self.return_approx = return_approx

    def fit(self):    
        U, s, Vs = scp.sparse.linalg.svds(self.D, k=self.rank, which='LM')
        U = U @ np.diag(s)
        V = Vs
        return U, V