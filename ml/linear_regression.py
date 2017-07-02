class LinearRegression(object):
    def __init__(self):
        # Number of training examples
        self.N = None
        
        # Number of features
        self.p = None
        
        self.X = None
        self.y = None
        
        self.beta = None
        
        self.df = None
        
        # If fit intercept
        self.fit_intercept = None
        
        # T scores for each coefficient
        self.t = None
        
        # Standard errors for each coefficient
        self.std_error = None
        
        # Estimated variance
        self.pop_var = None
        
        # Prediction
        self.y_hat = None
        
        # Residual sum of squares
        self.rss = None

    def train(self, X, y, features=None, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.y = y
        self.N = y.shape[0]
        # Ignore column vector
        if self.fit_intercept:
            self.features = ['bias'] + features
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
            self.p = X.shape[1] - (1 if self.fit_intercept else 0)
        else:
            self.features = features
            self.p = X.shape[1]
            
        self.X = X
        XT = X.T
        std_error_matrix = inv(XT.dot(X))
        self.beta = std_error_matrix.dot(XT).dot(y)
        
        # Prediction
        self.y_hat = X.dot(self.beta)
    
        # Residual sum of squares
        self.rss = np.sum((y - self.y_hat)**2)
        
        # Estimated variance
        self.df = (self.N - self.p - 1)
        self.pop_var = self.rss / self.df
        
        # Standard error
        self.std_error = np.sqrt(std_error_matrix.diagonal() * self.pop_var)
    
        # t scores
        self.t = self.beta / self.std_error
        
    def predict(self, X):
        return X.dot(beta)
    
    def plot_f_distrib_for_many_coefficients(self, features):
        from scipy.stats import f
                
        # Remove a particular subset of features
        X = np.delete(self.X, [self.features.index(_) for _ in features], 1)
        
        # Prediction from reduced model
        XT = X.T
        std_error_matrix = inv(XT.dot(X))
        beta = std_error_matrix.dot(XT).dot(self.y)
        y_hat = X.dot(beta)
        rss_reduced_model = np.sum((self.y - y_hat)**2)
                        
        dfn = len(features)
        dfd = self.df
        
        # This should be distributed as chi squared
        # with degrees of freedom equal to number
        # of dropped features
        rss_diff = (rss_reduced_model - self.rss)
        
        chi_1 = rss_diff / dfn
        chi_2 = self.pop_var
        
        f_score = chi_1 / chi_2
        
        # 5% and 95% percentile
        f_05, f_95 = f.ppf([0.05, 0.95], dfn, dfd)
                
        x = np.linspace(0.001, 5.0)
        
        plt.axvline(x=f_05)
        plt.axvline(x=f_95)
        
        plt.scatter(f_score, f.pdf(f_score, dfn, dfd), marker='o', color='red')
        plt.plot(x, f.pdf(x, dfn, dfd), color='gray', lw=5, alpha=0.6)
        plt.title('f-distribtion for dropping features: {0}'.format(features))
        plt.show()
    
    def print_results(self):
        print('**** Results ****')
        print('N: {0}'.format(self.N))
        print('p: {0}'.format(self.p))
        print('df RSS: {0}'.format(self.df))
        print('RSS: {0}'.format(lr.rss))
        print('Coefficients: {0}'.format(self.features))
        print('beta: {0}'.format(self.beta))
        print('std errors: {0}'.format(self.std_error))
        print('t-scores: {0}'.format(self.t))
    
    def plot_coefficient(self, feature):
        from scipy.stats import t
        import matplotlib.pyplot as plt
        
        x = np.linspace(-5.0, 5.0)
        t_score = self.t[self.features.index(feature)]
        
        # At what point is the 5% percentile?
        t_05 = t.ppf(0.05, self.df)
        
        # At what point is the 95% percentile?
        t_95 = t.ppf(0.95, self.df)
                
        plt.axvline(x=t_05)
        plt.axvline(x=t_95)
        
        plt.scatter(t_score, t.pdf(t_score, self.df), marker='o', color='red')
        plt.plot(x, t.pdf(x, self.df), color='gray', lw=5, alpha=0.6)
        plt.title('t-distribtion for {0} coefficient'.format(feature))
        plt.show()