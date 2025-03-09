import numpy as np

from scipy.sparse import csr_matrix, hstack   ## for converting dense matrix to sparse matrix

from Sigmoid import sigmoid

class logistic_regression():
    """
    X should be a sparse matrix of dimensions nxd
    y should be an numpy array of dimensions nx1
    """
    np.random.seed(23)   #setting random seed for reproducibility

    def del_fx(self, X, y, w):
        """
        Returns the value of the derivative of the loss function with respect to w
        """
        return -1 / y.shape[0] * X.T @ (y - sigmoid(X @ w))

    def __init__(self, threshold = 0.5, max_iter = 10**3, step_size_multiplier = 1, intermediate_w = False):
        self.iter_ = max_iter
        self.ssm_ = step_size_multiplier
        self.intermediate_w_ = intermediate_w
        self.fit_ = False
        self.threshold = threshold
        
    def add_bias_(self):   ## adds bias column
        X_ = hstack([csr_matrix(np.ones((self.n, 1))), self.X])
        return X_
    
    def fit(self, X, y):
        self.n = X.shape[0]
        self.X = X
        self.X = self.add_bias_()
        self.y = y
        self.w = np.random.randn(self.X.shape[1], 1)
        self.grads = [np.linalg.norm(self.del_fx(self.X, y, self.w))]

        ## implementing gradient descent
        iters = 0
        while (iters < self.iter_) and self.grads[-1] > 10**-6:
            self.w = self.w - self.ssm_ * 1/(iters*(10**-3) + 1) * self.del_fx(self.X, y, self.w)
            iters += 1
            self.grads.append(np.linalg.norm(self.del_fx(self.X, y, self.w)))
        self.fit_ = True
        return self.grads
    
    def predict(self, X_test, weights = None):
        if self.fit_:
            X_test = hstack([csr_matrix(np.ones((X_test.shape[0], 1))), X_test])
            return np.where(sigmoid(X_test @ self.w) < self.threshold, 0, 1)
        else:
            print("Model not fitted yet")