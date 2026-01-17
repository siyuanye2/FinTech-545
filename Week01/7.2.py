"""
Fit T Distribution using test7_2.csv and testout_7.2.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import t

# fit T dist
def fit_t_dist(data):
    data = np.asarray(data).ravel()
    nu, mu, sigma = t.fit(data)
    return nu, mu, sigma

# read input data
data = pd.read_csv('testfiles/data/test7_2.csv')
data = data.iloc[:, 0].to_numpy()
est_nu, est_mu, est_sigma = fit_t_dist(data)

print(f"mu,sigma,nu")
print(f"{est_mu},{est_sigma},{est_nu}")
# 0.04594038004735414,0.04544287220830122,6.336866997308613 
