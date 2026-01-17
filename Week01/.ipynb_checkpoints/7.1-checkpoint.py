import numpy as np
from scipy.stats import norm

def fit_normal_dist(data):
    mu, sigma = norm.fit(data)
    return mu, sigma

np.random.seed(1)
true_mu = 5
true_sigma = 2
data = np.random.normal(true_mu, true_sigma, 10_000)
est_mu, est_sigma = fit_normal_dist(data)
print(f"Estimated mu: {est_mu}, True mu: {true_mu}")
print(f"Estimated sigma: {est_sigma}, True sigma: {true_sigma}")