"""
Fit Normal Distribution using test7_1.csv and testout_7.1.csv
"""

import numpy as np
from scipy.stats import norm

# fit normal dist
def fit_normal_dist(data):
    mu, sigma = norm.fit(data)
    return mu, sigma

# read input data
data = np.loadtxt('testfiles/data/test7_1.csv', skiprows=1)
est_mu, est_sigma = fit_normal_dist(data)

print(f"mu,sigma")
print(f"{est_mu},{est_sigma}")
# 0.04602573645286829,0.04654545217470701