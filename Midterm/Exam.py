#!/usr/bin/env python
# coding: utf-8

# # Exam Helper Functions

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t, norm, multivariate_normal, skew, kurtosis, spearmanr
import statsmodels.miscmodels.tmodel as tmodel
from statsmodels.stats.correlation_tools import cov_nearest, corr_nearest


# In[38]:


# load data
# data = np.loadtxt('testfiles/data/test7_1.csv', skiprows=1)
# data = pd.DataFrame(data)

# data = pd.read_csv('testfiles/data/test7_1.csv')   # index_col = 'Date'


# In[33]:


# Mean, SD, variance, skewness, kurtosis
# data in np array format
# mean = np.mean(data)
# std = np.std(data, ddof=1)
# var = np.var(data)
# skewness = skew(data)
# kurt = kurtosis(data)


# In[39]:


# PDF, CDF, Quantile functions
# data in np array format
# pdf_values = norm.pdf(data, loc=mean, scale=std)
# cdf_value = norm.cdf(1.96)
# custom_cdf_value = norm.cdf(1.96, loc=mean, scale=std)
# median_quantile = np.quantile(data, 0.5)


# In[35]:


# AIC, BIC, AICc
# data in np array format
def model_selection_metrics(data):
    n = len(data)

    # Normal
    mu_n = np.mean(data)
    sigma_n = np.std(data, ddof=1)
    loglik_n = np.sum(norm.logpdf(data, mu_n, sigma_n))
    k_n = 2

    AIC_n = 2*k_n - 2*loglik_n
    BIC_n = k_n*np.log(n) - 2*loglik_n
    AICc_n = AIC_n + (2*k_n*(k_n+1)) / (n - k_n - 1)

    # T 
    nu_t, mu_t, sigma_t = t.fit(data)
    loglik_t = np.sum(t.logpdf(data, nu_t, mu_t, sigma_t))
    k_t = 3

    AIC_t = 2*k_t - 2*loglik_t
    BIC_t = k_t*np.log(n) - 2*loglik_t
    AICc_t = AIC_t + (2*k_t*(k_t+1)) / (n - k_t - 1)

    return AIC_n, BIC_n, AICc_n, AIC_t, BIC_t, AICc_t


# In[ ]:


# R2 and adj. R2
# X = sm.add_constant(X)  # add intercept
# model = sm.OLS(y, X).fit()

# print(model.rsquared)
# print(model.rsquared_adj)


# In[10]:


# Fit unbiased normal dist (7.1)
# data in np array format
# est_mu = np.mean(data)
# est_sigma = np.std(data, ddof=1)  # sample


# In[190]:


# Fit T dist (7.2)
# data in np array format
# est_nu, est_mu, est_sigma = t.fit(data)


# In[191]:


# Fit T regression (7.3)
def t_reg(data):
    y = data["y"].to_numpy()
    x = data.drop(columns=["y"]).to_numpy()
    n = len(y)
    
    X_ols = np.column_stack([np.ones(n), x])

    model = tmodel.TLinearModel(y, X_ols)
    result = model.fit()
    
    params = result.params
    alpha_est, b1_est, b2_est, b3_est = result.params[:4]
    sigma_est = result.params[5]
    nu_est = result.params[4]

    return alpha_est, b1_est, b2_est, b3_est, sigma_est, nu_est


# In[192]:


# cov & corr skipping missing rows, matrix input (1.1, 1.2)
def skip_na(data):
    data_clean = data.dropna()
    cov = data_clean.cov()
    corr = data_clean.corr()
    # spearman_corr = data_clean.corr(method='spearman')
    return cov, corr


# In[193]:


# Pairwise cov & corr, matrix input (1.3, 1.4)
# cov = data.cov()
# corr = data.corr()
# # spearman_corr = data.corr(method='spearman')


# In[194]:


# EW covariance & variance, matrix input (2.1)
def ew_cov_var(data, lambda_):
    alpha = 1 - lambda_

    # cov
    cov = data.ewm(alpha=alpha, adjust=True).cov()
    ew_cov_unbiased = cov.loc[data.index[-1]]

    # var
    ew_var = data.ewm(alpha=alpha).var()
    var_last = ew_var.loc[data.index[-1]]    

    n = len(data)
    w = np.array([alpha * lambda_**i for i in range(n)])
    w /= w.sum()
    correction_factor = 1 - np.sum(w**2)

    ew_cov_biased = ew_cov_unbiased * correction_factor
    var_biased = var_last * correction_factor

    return ew_cov_biased, var_biased


# In[195]:


# EW correlation, matrix input (2.2)
def ew_corr(data, lambda_):
    alpha = 1 - lambda_

    corr = data.ewm(alpha=alpha).corr()

    ew_corr_last = data.ewm(alpha=alpha).corr()
    ew_corr_last = corr.loc[data.index[-1]]

    return ew_corr_last


# In[196]:


# Covariance with EW Variance and EW Correlation, matrix input (2.3)
def ew_variance(data, var_lambda, corr_lambda):
    alpha = 1 - var_lambda

    ew_var = data.ewm(alpha=alpha).var()
    var_last = ew_var.loc[data.index[-1]]

    n = len(data)
    w = np.array([alpha * var_lambda**i for i in range(n)])
    w /= w.sum()
    correction_factor = 1 - np.sum(w**2)
    
    var_biased = var_last * correction_factor
    
    ew_sd = np.sqrt(var_biased)
    D = np.diag(ew_sd)

    ew_corr_last = ew_corr(data, corr_lambda)
    cov = D @ ew_corr_last @ D

    return cov


# In[197]:


# Near PSD cov & corr, matrix input (3.1, 3.2)
def near_psd(cov):
    D = np.sqrt(np.diag(cov))
    corr = cov / np.outer(D, D)
    
    # eigen-decomposition
    eigval, eigvec = np.linalg.eigh(corr)

    # clip eigenvalues
    eigval_clipped = np.maximum(eigval, 1e-8)

    # reconstruct corr matrix
    corr_psd = eigvec @ np.diag(eigval_clipped) @ eigvec.T

    # normalize
    corr_psd = corr_psd / np.outer(np.sqrt(np.diag(corr_psd)), np.sqrt(np.diag(corr_psd)))

    # corr to cov
    cov_psd = np.outer(D, D) * corr_psd
    
    return cov_psd, corr_psd


# In[199]:


# Higham cov & corr (3.3, 3.4)
def higham(data):
    higham_corr = corr_nearest(corr)
    
    std = np.sqrt(np.diag(data))
    corr = data / np.outer(std, std)
    D = np.diag(std)
    higham_cov = D @ higham_corr @ D
    
    return higham_cov, higham_corr


# In[202]:


# PSD Cholesky, matrix input (4.1)
# L = np.linalg.cholesky(data)


# In[203]:


# Arithmetic returns (6.1)
def arith_return(data):
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")
    data = data.set_index("Date")
    
    returns = data.pct_change()
    returns = returns.dropna()
    return returns


# In[204]:


# log returns (6.2)
def log_returns(data):
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")
    data = data.set_index("Date")
    
    returns = np.log(data).diff()
    returns = returns.dropna()
    return returns


# In[2]:


# Geometric returns
def geo_returns(data):
    n = len(data)

    # Calculate the product of (1 + return) values
    product_of_factors = 1
    for r in data:
        product_of_factors *= (1 + r)
    
    # Calculate the nth root and subtract 1
    returns = (product_of_factors ** (1/n)) - 1
    return returns


# In[41]:


# De-mean return series to have mean of 0
# demean = data - data.mean()


# In[40]:


# check eigenvalue to confirm matrix is PD / PSD / non definite, matrix input (5.1-5.4)
def eigenvalue(data):
    eigvals = np.linalg.eigvalsh(data)
    
    tol = 1e-10
    if np.all(eigvals > tol):
        print("Positive Definite")
    elif np.all(eigvals >= -tol):
        print("Positive Semi-Definite")
    else:
        print("Indefinite")


# In[205]:


# Normal simulation, PD / PSD matrix input (5.1, 5.2)
def normal_sim_pd_psd(data, mu=0):
    # mean vector
    n = len(data)
    mu = np.zeros(n)
    
    np.random.seed(42)
    n_sims = 100_000

    # if Cholesky doesn't fail, also confirm PD / PSD
    L = np.linalg.cholesky(data)
    
    # draw random num from normal dist, 100,000 x 5
    z = np.random.randn(n_sims, n)
    
    # correlated simulations
    x = z @ L.T + mu

    sim_cov = np.cov(x, rowvar=False, ddof=0)

    return sim_cov


# In[3]:


# Normal simulation, PSD matrix input (5.2)
def normal_sim_svd(data):
    n = data.shape[0]
    n_sims = 100_000
    np.random.seed(42)

    U, s, Vt = np.linalg.svd(data)

    # clip small negatives (numerical issue)
    s[s < 0] = 0

    # SVD
    L = U @ np.diag(np.sqrt(s))

    z = np.random.randn(n_sims, n)
    x = z @ L.T

    sim_cov = np.cov(x, rowvar=False, ddof=0)

    return sim_cov


# In[206]:


# Normal simulation, non PSD matrix input, near PSD fix (5.3)
# cov_psd, corr_psd = near_psd(data)
# sim_cov = normal_sim_pd_psd(cov_psd)


# In[207]:


# Normal simulation, non PSD matrix input, Higham fix (5.4)
# higham_cov = cov_nearest(data, method="higham")
# sim_cov = normal_sim_pd_psd(higham_cov)


# In[208]:


# PCA simulation (5.5)
def pca_sim(data, threshold=0.99):
    # eigen decompose
    eigen_val, eigen_vec = np.linalg.eigh(data)
    
    # sort descending
    idx = np.argsort(eigen_val[::-1])
    eigen_val = eigen_val[idx]
    eigen_vec = eigen_vec[:, idx]
    
    # variance explained
    total_var = eigen_val.sum()
    pct_explained = eigen_val / total_var
    cum_explained = np.cumsum(pct_explained)
    
    # choose k to reach 99% explained
    k = int(np.searchsorted(cum_explained, threshold) + 1)
    
    # top k eigenvectors and eigenvalues
    k_vec = eigen_vec[:, :k]
    k_val = eigen_val[:k]
    
    # PCA simulation
    np.random.seed(42)
    n_sims = 100_000
        
    # draw random num from normal dist
    z = np.random.randn(n_sims, k)
    
    S = z * np.sqrt(k_val[None, :])
        
    # correlated simulations
    x = S @ k_vec.T
    
    sim_cov = np.cov(x, rowvar=False, ddof=0)
    
    return sim_cov


# In[209]:


# VaR & ES from normal dist (8.1, 8.4)
# data in np array format
def var_es_normal(data, alpha):
    est_mu = np.mean(data)
    est_sigma = np.std(data, ddof=1)
    
    z_score = norm.ppf(alpha)
    phi_z = norm.pdf(z_score)

    # VaR
    var_percentile = est_mu + z_score * est_sigma
    var_abs = abs(var_percentile)  # as distance from 0
    var_diff = est_mu - var_percentile

    # ES
    es_abs = -est_mu + est_sigma * phi_z / alpha  # as distance from 0
    es_diff = est_sigma * phi_z / alpha
    
    return var_abs, var_diff, es_abs, es_diff


# In[210]:


# VaR & ES from T dist (8.2, 8.5)
# data in np array format
def var_es_t(data, alpha):
    est_nu, est_mu, est_sigma = t.fit(data)
    
    t_stat = t.ppf(alpha, df=est_nu)
    phi_t = t.pdf(t_stat, df=est_nu)

    # VaR
    var_percentile = est_mu + t_stat * est_sigma
    var_abs = abs(var_percentile)
    var_diff = est_mu - var_percentile

    # ES
    es_abs = -est_mu + est_sigma * ((est_nu + t_stat**2) / (est_nu - 1)) * (phi_t / alpha)
    es_diff = est_sigma * ((est_nu + t_stat**2) / (est_nu - 1)) * (phi_t / alpha)
    
    return var_abs, var_diff, es_abs, es_diff


# In[211]:


# VaR & ES from T simulation (8.3, 8.6)
# data in np array format
def var_es_t_sim(data):
    np.random.seed(42)
    n_sims = 100_000
    alpha = 0.05

    est_nu, est_mu, est_sigma = t.fit(data)
    t_sim = t.rvs(est_nu, est_mu, est_sigma, size=n_sims)

    # VaR
    q_alpha = np.quantile(t_sim, alpha)
    var_abs = abs(q_alpha)
    var_diff = est_mu - np.quantile(t_sim, alpha)

    # ES
    es_abs = abs(t_sim[t_sim <= q_alpha].mean())
    es_diff = est_mu - t_sim[t_sim <= q_alpha].mean()
    
    return var_abs, var_diff, es_abs, es_diff


# In[ ]:


# # Delta normal VaR
# # inputs
# w = np.array([0.4, 0.3, 0.3])
# Sigma = np.array([[0.04, 0.01, 0.02],
#                   [0.01, 0.09, 0.03],
#                   [0.02, 0.03, 0.16]])

# V0 = 1_000_000
# alpha = 0.05

# # portfolio volatility
# sigma_p = np.sqrt(w.T @ Sigma @ w)

# # z score
# z = norm.ppf(alpha)

# VaR = -V0 * z * sigma_p


# In[ ]:


# # historical VaR
# # single asset
# alpha = 0.05
# V0 = 1_000_000
# q_alpha = np.quantile(returns, alpha)
# VaR = -V0 * q_alpha

# # portfolio
# portfolio_returns = returns @ weights
# VaR = -V0 * np.quantile(portfolio_returns, alpha)


# In[4]:


# Portfolio VaR & ES on normal & T dist, from Copula (9.1)
def copula(portfolio, returns):
    # A: normal dist
    A_return = returns['A'].values
    A_mu = np.mean(A_return)
    A_sigma = np.std(A_return, ddof=1)

    # B: T dist
    B_return = returns['B'].values
    B_nu, B_mu, B_sigma = t.fit(B_return)

    # transform to uniforms
    U_A = norm.cdf(A_return, A_mu, A_sigma)
    U_B = t.cdf(B_return, df=B_nu, loc=B_mu, scale=B_sigma)
    U = np.column_stack([U_A, U_B])

    # Gaussian Copula corr
    Z = norm.ppf(np.column_stack([U_A, U_B]))
    # copula_corr = np.corrcoef(Z.T)  # Pearson
    rho_s, _ = spearmanr(A_return, B_return)  # Spearman
    copula_rho = 2 * np.sin(np.pi * rho_s / 6)
    copula_corr = np.array([[1, copula_rho],
                            [copula_rho, 1]])

    # simulate joint uniforms
    np.random.seed(42)
    n_sims = 100_000
    sim = multivariate_normal.rvs(np.zeros(2), copula_corr, n_sims)
    U_sim = norm.cdf(sim)

    # transform back to normal / T dist
    A_sim = norm.ppf(U_sim[:, 0], A_mu, A_sigma)
    B_sim = t.ppf(U_sim[:, 1], df=B_nu, loc=B_mu, scale=B_sigma)
    
    R_sim = np.column_stack([A_sim, B_sim])

    # convert return to P&L
    A_pos_value = portfolio.loc[0, "Holding"] * portfolio.loc[0, "Starting Price"]  # latest price
    B_pos_value = portfolio.loc[1, "Holding"] * portfolio.loc[1, "Starting Price"]  # latest price
    V0 = A_pos_value + B_pos_value
    
    # weighted portfolio return
    A_weight = A_pos_value / V0
    B_weight = B_pos_value / V0
    
    weights = np.array([A_weight, B_weight])
    
    A_loss = -A_sim * A_pos_value
    B_loss = -B_sim * B_pos_value
    
    R_port = A_weight * A_sim + B_weight * B_sim

    return A_mu, A_sigma, A_pos_value, B_sim, B_pos_value, B_nu, B_mu, B_sigma, V0, R_port


# In[5]:


# Portfolio VaR & ES output
def copula_output(portfolio, returns):
    A_mu, A_sigma, A_pos_value, B_sim, B_pos_value, B_nu, B_mu, B_sigma, V0, R_port = copula(portfolio, returns)
    alpha = 0.05

    # total VaR & ES
    q_alpha = np.quantile(R_port, alpha)
    var_pct = -q_alpha
    es_pct = -R_port[R_port <= q_alpha].mean()
    var = V0 * var_pct
    es = V0 * es_pct

    # A VaR & ES
    z_score = norm.ppf(alpha)
    A_var_pct = abs(A_mu + z_score * A_sigma)
    A_var = A_var_pct * A_pos_value
    
    phi_z = norm.pdf(z_score)
    A_es_pct = -A_mu + A_sigma * phi_z / alpha
    A_es = A_es_pct * A_pos_value

    # B VaR & ES
    B_var_pct = abs(np.quantile(B_sim, alpha))
    B_var = B_var_pct * B_pos_value
    
    t_stat = t.ppf(alpha, df=B_nu)
    phi_t = t.pdf(t_stat, df=B_nu)
    B_es_pct = -B_mu + B_sigma * ((B_nu + t_stat**2) / (B_nu - 1)) * (phi_t / alpha)
    B_es = B_es_pct * B_pos_value
    
    return A_var, A_es, A_var_pct, A_es_pct, B_var, B_es, B_var_pct, B_es_pct, var, es, var_pct, es_pct


# In[ ]:




