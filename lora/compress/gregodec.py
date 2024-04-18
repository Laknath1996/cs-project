import numpy as np
import scipy as scp

class GreGoDec:
    def __init__(self, D, rank, tau, tol, power, k) -> None:
        self.D = D
        self.rank = rank
        self.tau = tau
        self.tol = tol
        self.power = power
        self.k = k

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')
    
    @staticmethod
    def norm(M):
        return np.linalg.norm(M, ord=2)
    
    @staticmethod
    def wthresh(x, thresh):
        return np.sign(x) * np.maximum(np.abs(x)-thresh, 0.)
    
    def fit(self):
        # matrix size
        D = self.D
        m, n = D.shape
        if m < n:
            D = D.T
        normD = self.norm(self.D.ravel())

        # initialization of L and S
        rankk = np.round(self.rank/self.k).astype('int')
        error = np.zeros((rankk*self.power, 1))
        X, s, Y = scp.sparse.linalg.svds(self.D, k=self.k, which='LM')
        s = np.diag(s)
        X = X @ s
        L = X @ Y
        S = self.wthresh(D - L, self.tau)
        T = D - L - S
        error[0] = self.norm(T.ravel())/normD
        iii = 0
        stop = False
        alf = 0

        for r in range(1, rankk+1, 1):
            rrank = self.rank
            est_rank = 1
            rank_min = 1
            rk_jump = 10
            alf = 0
            increment = 1
            itr_rank = 0
            minitr_reduce_rank = 5
            maxitr_reduce_rank = 50
            if iii == self.power * (r-2)+1:
                iii += self.power
            for iter in range(1, self.power+1):
                # update of X
                X = L @ Y.T
                if est_rank == 1:
                    X, R, E = scp.linalg.qr(X, pivoting=True)
                else:
                    X, R = scp.linalg.qr(X, pivoting=False)

                # update of Y
                Y = X.T @ L
                L = X @ Y

                # update of S
                T = D - L
                S = self.wthresh(T, self.tau)

                # error, stopping criteria
                T = T - S
                ii = iii + iter
                error[ii] = self.norm(T.ravel())/normD
                if error[ii] < self.tol:
                    stop = True
                    break
                
                # adjust est_rank
                if est_rank >= 1:
                    if est_rank == 1:
                        dR = abs(np.diag(R))
                        drops = dR[:-1] / dR[1:]
                        dmx = np.max(drops)
                        imx = np.argmax(drops)
                        rel_drp = (self.rank - 1) * dmx / (drops.sum() - dmx)
                        if (rel_drp > rk_jump and itr_rank > minitr_reduce_rank) or (itr_rank > maxitr_reduce_rank):
                            rrank = np.max([imx, np.floor(0.1 * self.rank), rank_min])
                            error[ii] = self.norm(res)/normz
                            est_rank = 0
                            itr_rank = 0

                if rrank != self.rank:
                    rank = rrank
                    if est_rank == 0:
                        alf = 0
                        continue
                ratio = error[ii]/error[ii-1]
                if ratio >= 1.1:
                    increment = np.maximum(0.1 * alf, 0.1 * increment)
                    X = X1
                    Y = Y1
                    L = L1
                    S = S1
                    T = T1
                    error[ii] = error[ii-1]
                    alf = 0
                elif ratio > 0.7:
                    increment = max(increment, 0.25 * alf)
                    alf = alf + increment

                # update of L
                X1 = X
                Y1 = Y
                L1 = L
                S1 = S
                T1 = T
                L = L + (1 + alf) * T

                # add coreset
                if iter > 8:
                    if np.mean(error[ii-7:ii])/error[ii-8] > 0.92:
                        iii = ii
                        sf = X.shape[1]
                        if Y.shape[0] - sf >= self.k:
                            Y = Y[:sf, :]
                            break
            if stop:
                break

            if r < rankk:
                v = np.random.randn(self.k, m) @ L
                Y = np.concatenate((Y, v), axis=0)
        
        L = X @ Y
        if m < n:
            L = L.T
            S = S.T
        
        return X, Y, S, error

def main():
    from lora.models.roberta import get_model   
    from omegaconf import OmegaConf
    import numpy as np
    from scipy.special import softmax
    import matplotlib.pyplot as plt

    args = OmegaConf.load('/cis/home/adesilva/ashwin/research/cs-project/config.yaml')
    model = get_model(args)

    Wq = model.encoder.encoder.layer[-1].attention.self.query.weight.numpy()
    Wk = model.encoder.encoder.layer[-1].attention.self.key.weight.numpy()
    B = Wq @ Wk.T

    U, V, S, error = GreGoDec(
        D = B,
        rank = 100,
        tau = 0.5,
        tol = 1e-2, 
        power = 10,
        k = 5
    ).fit()

    np.random.seed(1996)
    X = np.random.randn(128, B.shape[0])

    Y = softmax(X @ B @ X.T)
    Yhat = softmax(X @ (U @ V + S) @ X.T)

if __name__ == "__main__":
    main()