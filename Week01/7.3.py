"""
Fit T Regression using test7_3.csv and testout_7.3.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.optimize import minimize

# negative log likelihood
def neg_log_likelihood(params, X, y):
    alpha, b1, b2, b3, sigma, nu = params
    
    # bad params
    if sigma <= 0 or nu <= 2:
        return np.inf
    
    mu = alpha + b1 * X[:, 0] + b2 * X[:, 1] + b3 * X[:, 2]
    resid = y - mu
    ll = t.logpdf(resid / sigma, df=nu) - np.log(sigma)
    
    return -np.sum(ll)
    
data = pd.read_csv("testfiles/data/test7_3.csv")
x = data[["x1", "x2", "x3"]].to_numpy()
y = data["y"].to_numpy()
n = len(y)

# OLS for intercept and betas
X_ols = np.column_stack([np.ones(n), x])
beta_ols = np.linalg.lstsq(X_ols, y, rcond=None)[0]
alpha_ols = beta_ols[0]
b1_ols, b2_ols, b3_ols = beta_ols[1], beta_ols[2], beta_ols[3]

# residuals and t-fit
fitted_mean = alpha_ols + b1_ols * x[:, 0] + b2_ols * x[:, 1] + b3_ols * x[:, 2]
resid = y - fitted_mean

# fit Student-t to residuals
nu, est_loc, sigma = t.fit(resid)

print("mu,sigma,nu,Alpha,B1,B2,B3")
print(f"0.0,{sigma},{nu},{alpha_ols},{b1_ols},{b2_ols},{b3_ols}")
# 0.0,0.05079942753011614,5.597957314863795,0.041566864325963924,1.0284306249796855,2.1846966538624457,3.1728912932406867
