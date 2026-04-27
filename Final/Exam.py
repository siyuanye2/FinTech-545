#!/usr/bin/env python
# coding: utf-8

# ## Exam Helper Functions
# https://prodduke-my.sharepoint.com/:w:/r/personal/sy348_duke_edu/Documents/Fintech%20545%20Final%20Review.docx?d=weba335371acf4b73a95357e5c9bfdad0&csf=1&web=1&e=5nWS1D

# In[1]:


import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import t, norm, multivariate_normal, skew, kurtosis, gaussian_kde, spearmanr, rankdata
import statsmodels.miscmodels.tmodel as tmodel
from statsmodels.stats.correlation_tools import cov_nearest, corr_nearest
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# #### Midterm

# In[2]:


# load data
# data = np.loadtxt('testfiles/data/test7_1.csv', skiprows=1)
# data = pd.DataFrame(data)

# data = pd.read_csv('testfiles/data/test5_2.csv')   # index_col = 'Date'


# In[3]:


# # Mean, SD, variance, skewness, kurtosis
# # data in np array format
# mean = np.mean(data)
# std = np.std(data, ddof=1)
# var = np.var(data)
# skewness = skew(data)
# kurt = kurtosis(data)


# In[4]:


# # PDF, CDF, Quantile functions
# # data in np array format
# pdf_values = norm.pdf(data, loc=mean, scale=std)
# cdf_value = norm.cdf(1.96)
# custom_cdf_value = norm.cdf(1.96, loc=mean, scale=std)
# median_quantile = np.quantile(data, 0.5)


# In[5]:


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


# In[6]:


# # R2 and adj. R2
# X = sm.add_constant(X)  # add intercept
# model = sm.OLS(y, X).fit()

# print(model.rsquared)
# print(model.rsquared_adj)


# In[7]:


# # Fit unbiased normal dist (7.1)
# # data in np array format
# est_mu = np.mean(data)
# est_sigma = np.std(data, ddof=1)  # sample


# In[8]:


# # Fit T dist (7.2)
# # data in np array format
# est_nu, est_mu, est_sigma = t.fit(data)


# In[9]:


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


# In[10]:


# cov & corr skipping missing rows, matrix input (1.1, 1.2)
def skip_na(data):
    data_clean = data.dropna()
    cov = data_clean.cov()
    corr = data_clean.corr()
    # spearman_corr = data_clean.corr(method='spearman')
    return cov, corr


# In[11]:


# # Pairwise cov & corr, matrix input (1.3, 1.4)
# cov = data.cov()
# corr = data.corr()
# # spearman_corr = data.corr(method='spearman')


# In[12]:


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


# In[13]:


# EW correlation, matrix input (2.2)
def ew_corr(data, lambda_):
    alpha = 1 - lambda_

    corr = data.ewm(alpha=alpha).corr()

    ew_corr_last = data.ewm(alpha=alpha).corr()
    ew_corr_last = corr.loc[data.index[-1]]

    return ew_corr_last


# In[14]:


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


# In[15]:


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


# In[16]:


# Higham cov & corr, matrix input (3.3, 3.4)
def higham(data):
    higham_corr = corr_nearest(data)
    
    std = np.sqrt(np.diag(data))
    corr = data / np.outer(std, std)
    D = np.diag(std)
    higham_cov = D @ higham_corr @ D
    
    return higham_cov, higham_corr


# In[17]:


# # PSD Cholesky, matrix input (4.1)
# L = np.linalg.cholesky(data)


# In[18]:


# # De-mean return series to have mean of 0
# demean = data - data.mean()


# In[19]:


# Arithmetic and geometric total return
# data in np array format
def total_returns(returns):
    R = np.prod(1 + returns) - 1
    GR = np.log(1 + R)
    return R, GR


# In[20]:


# Arithmetic returns (6.1)
def arith_return(data):
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")
    data = data.set_index("Date")
    
    returns = data.pct_change().dropna()
    return returns


# In[21]:


# log returns (6.2)
def log_returns(data):
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")
    data = data.set_index("Date")
    
    returns = np.log(data).diff().dropna()
    return returns


# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[25]:


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


# In[26]:


# # Normal simulation, non PSD matrix input, near PSD fix (5.3)
# cov_psd, corr_psd = near_psd(data)
# sim_cov = normal_sim_pd_psd(cov_psd)


# In[27]:


# # Normal simulation, non PSD matrix input, Higham fix (5.4)
# higham_cov = cov_nearest(data, method="higham")
# sim_cov = normal_sim_pd_psd(higham_cov)


# In[28]:


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


# In[29]:


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


# In[30]:


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


# In[31]:


# VaR & ES from T simulation (8.3, 8.6)
# data in np array format
def var_es_t_sim(data, alpha):
    np.random.seed(42)
    n_sims = 100_000

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


# In[32]:


# VaR & ES from historical simulation (MidTerm)
# data in np array format
def var_es_hist_sim(data, alpha):
    q_alpha = np.quantile(data, alpha)
    var_abs = -q_alpha

    # ES
    es_abs = -data[data <= q_alpha].mean()
    
    return var_abs, es_abs


# In[33]:


# # historical VaR
# # single asset
# alpha = 0.05
# V0 = 1_000_000
# q_alpha = np.quantile(returns, alpha)
# VaR = -V0 * q_alpha

# # KDE
# kde = gaussian_kde(returns)
# x_grid = np.linspace(returns.min(), returns.max(), 10000)
# pdf_vals = kde(x_grid)
# cdf_vals = np.cumsum(pdf_vals)
# cdf_vals /= cdf_vals[-1]   # normalize to 1
# VaR_return = np.interp(alpha, cdf_vals, x_grid)
# VaR = -V0 * VaR_return

# # portfolio
# portfolio_returns = returns @ weights
# VaR = -V0 * np.quantile(portfolio_returns, alpha)


# In[34]:


# Delta normal VaR & ES, matrix input (linear approx)
# data in np array format
def delta_normal(
    cov,   # Covariance matrix of asset arithmetic returns
    V0,
    alpha,
    exposures=None,   # Optional, normalized asset exposures directly, shape (n_assets,). ex: weights \ delta-equivalent exposures divided by V0
    asset_deltas=None   # Raw asset delta exposures in dollar/share units before normalization
):
    cov = np.asarray(cov, dtype=float)

    if exposures is None:
        if asset_deltas is None:
            raise ValueError("Provide either exposures or asset_deltas.")
        exposures = np.asarray(asset_deltas, dtype=float) / V0
    else:
        exposures = np.asarray(exposures, dtype=float)

    sigma_p = np.sqrt(exposures @ cov @ exposures)

    z = norm.ppf(alpha)
    var = -V0 * z * sigma_p
    es = V0 * norm.pdf(z) / alpha * sigma_p

    return {
        "VaR": var,
        "ES": es
    }


# In[35]:


# # Delta normal VaR & ES of portfolio

# # +100 shares A
# # +100 puts on A, each option controls 1 share unless your class uses x100
# # +50 shares B
# # -50 calls on B

# # convert to value-normalized exposures
# V0 = (
#     100 * S
#     + 100 * A_put_price
#     + 50 * S
#     - 50 * B_call_price
# )

# # aggregate asset exposures exactly like the key
# A_exposure = (100 * 1.0 + 100 * A_delta) * 100
# B_exposure = (50 * 1.0 - 50 * B_delta) * 100

# deltas = np.array([A_exposure, B_exposure]) / V0

# port_std = np.sqrt(deltas @ cov @ deltas)

# z_5 = norm.ppf(0.05)
# var_dn = -V0 * z_5 * port_std
# es_dn = V0 * norm.pdf(z_5) / 0.05 * port_std

# print("Portfolio value =", V0)
# print("Delta-normal VaR =", var_dn)
# print("Delta-normal ES  =", es_dn)


# In[36]:


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
    Z = norm.ppf(U)
    
    # Pearson
    # copula_corr = np.corrcoef(Z.T)
    
    # Spearman
    rho_s, _ = spearmanr(data)
    copula_corr = 2 * np.sin(np.pi * rho_s / 6)
    # copula_corr = np.array([[1, copula_rho],   # 2 assets only
    #                     [copula_rho, 1]])

    # simulate joint uniforms
    np.random.seed(42)
    n_sims = 100_000
    n_assets = U.shape[1]
    sim = multivariate_normal.rvs(np.zeros(n_assets), copula_corr, n_sims)
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


# In[37]:


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


# In[38]:


# Portfolio VaR & ES from historical simulation (MidTerm)
def hist_sim(returns, alpha=0.01):
    weights = np.repeat(1 / returns.shape[1], data.shape[1])  # equal weights
    R_port = returns.values @ weights

    q_alpha = np.quantile(R_port, alpha)
    var_pct = -q_alpha
    es_pct = -R_port[R_port <= q_alpha].mean()

    var = V0 * var_pct
    es = V0 * es_pct

    return var, es


# In[39]:


# Each stock's VaR & ES from historical simulation (MidTerm)
def var_es_sim(series, position_value, alpha=0.01):
    q_alpha = np.quantile(series, alpha)
    var_pct = -q_alpha
    es_pct = -series[series <= q_alpha].mean()

    var = position_value * var_pct
    es = position_value * es_pct

    return var, es


# #### Final

# In[40]:


# GBSM: European options (12.1)
def GBSM_euro_option(data):
    output = pd.DataFrame()
    output['ID'] = data['ID']
    
    for i in range(data.shape[0]):
        S = data.loc[i, 'Underlying']
        K = data.loc[i, 'Strike']
        T = data.loc[i, 'DaysToMaturity'] / data.loc[i, 'DayPerYear']
        r = data.loc[i, 'RiskFreeRate']
        q = data.loc[i, 'DividendRate']
        sigma = data.loc[i, 'ImpliedVol']
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    
        helper = np.exp(-q * T) * norm.cdf(d1)
    
        output.loc[i, 'Gamma'] = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        output.loc[i, 'Vega'] = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        
        if data.loc[i, 'Option Type'] == "Call":
            output.loc[i, 'Value'] = S * helper - K * np.exp(-r * T) * norm.cdf(d2)
            output.loc[i, 'Delta'] = helper
            output.loc[i, 'Rho'] = K * T * np.exp(-r * T) * norm.cdf(d2)
            output.loc[i, 'Theta'] = (
                -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * helper
            )
        else:
            output.loc[i, 'Value'] = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            output.loc[i, 'Delta'] = np.exp(-q * T) * (norm.cdf(d1) - 1)
            output.loc[i, 'Rho'] = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            output.loc[i, 'Theta'] = (
                -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
                - q * S * np.exp(-q * T) * norm.cdf(-d1)
            )

    new_order = ['ID', 'Value', 'Delta', 'Gamma', 'Vega', 'Rho', 'Theta']
    return output[new_order]


# In[41]:


# Monte Carlo: European options
# data in np array format
def mc_euro_option(returns, S0, K, r, T, trading_days, option_type):
    returns = returns.flatten()
    TTM = T / trading_days
    discount = np.exp(-r * TTM)   # convert annual risk free rate by TTM
    ST = S0 * (1 + returns)

    if option_type == "Call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    price = discount * np.mean(payoff)
    
    return price


# In[42]:


# European implied volatility from Monte Carlo and GBSM
def bs_price(S, K, r, q, T, sigma, op_type):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if op_type == "Call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def implied_volatility_bsm(price, S, K, r, q, T, op_type, sigma_low=1e-6, sigma_high=5.0):
    objective = lambda sigma: bs_price(S, K, r, q, T, sigma, op_type) - price
    return brentq(objective, sigma_low, sigma_high)

def implied_volatility_plot(strike_grid, implied_vols):
    plt.figure(figsize=(8, 5))
    plt.plot(strike_grid, implied_vols, marker="o")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatility v. Strike")
    plt.grid(True)
    plt.show()


# In[43]:


# Binomial tree: American options with continuous dividends (12.2)
def bt_am_continuous_div_value(call, underlying, strike, ttm, rf, b, ivol, N):
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1

    # price tree at maturity
    prices = np.array([underlying * (u ** (N - i)) * (d ** i) for i in range(N + 1)])

    # Initialize option values at maturity
    values = np.maximum(z * (prices - strike), 0)

    # Backward induction
    for step in range(N - 1, -1, -1):
        for i in range(step + 1):
            price = underlying * (u ** (step - i)) * (d ** i)
            # Continuation value
            values[i] = df * (pu * values[i] + pd * values[i + 1])
            # Am. option only: Early exercise value
            exercise = max(0, z * (price - strike))
            values[i] = max(values[i], exercise)

    return values[0]


# In[44]:


def am_option_continuous_div_greeks(S, K, T, r, q, sigma, option_type):
    call = (option_type == 'Call')
    b = r - q
    N = 500

    price = bt_am_continuous_div_value(call, S, K, T, r, b, sigma, N)

    dS = 1.0
    price_up = bt_am_continuous_div_value(call, S + dS, K, T, r, b, sigma, N)
    price_down = bt_am_continuous_div_value(call, S - dS, K, T, r, b, sigma, N)
    delta = (price_up - price_down) / (2 * dS)

    gamma = (price_up - 2 * price + price_down) / (dS ** 2)

    dv = 0.01
    price_vol_up = bt_am_continuous_div_value(call, S, K, T, r, b, sigma + dv, N)
    vega = (price_vol_up - price) / dv

    dr = 0.001
    price_r_up = bt_am_continuous_div_value(call, S, K, T, r + dr, b + dr, sigma, N)
    rho = (price_r_up - price) / dr

    dt = 1 / 365
    price_t_up = bt_am_continuous_div_value(call, S, K, T + dt, r, b, sigma, N)
    theta = (price_t_up - price) / dt

    return price, delta, gamma, vega, rho, theta


# In[45]:


def am_option_continuous_div_output(data):
    results = []

    for i in range(data.shape[0]):
        S = data.loc[i, 'Underlying']
        K = data.loc[i, 'Strike']
        T = data.loc[i, 'DaysToMaturity'] / data.loc[i, 'DayPerYear']
        r = data.loc[i, 'RiskFreeRate']
        q = data.loc[i, 'DividendRate']
        sigma = data.loc[i, 'ImpliedVol']
        option_type = data.loc[i, 'Option Type']
    
        value, delta, gamma, vega, rho, theta = am_option_continuous_div_greeks(S, K, T, r, q, sigma, option_type)
        results.append([value, delta, gamma, vega, rho, theta])
    
    # output = pd.DataFrame()
    # output['ID'] = data['ID']
    # output[['Value', 'Delta', 'Gamma', 'Vega', 'Rho', 'Theta']] = results
    
    return results


# In[46]:


# Binomial tree: American options value, discreet dividends (12.3)
def bt_am_discreet_div_value(S, K, T, r, sigma, option_type, div_dates, div_amts, steps):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    div_step_map = {}
    for t, amt in zip(div_dates, div_amts):
        if amt == 0:
            continue
        if 0 < t <= T:
            k = int(round(t / T * steps))
            k = min(max(k, 1), steps)
            div_step_map[k] = div_step_map.get(k, 0.0) + float(amt)

    stock_tree = np.zeros((steps + 1, steps + 1), dtype=float)
    stock_tree[0, 0] = S

    for step in range(1, steps + 1):
        cash_div = div_step_map.get(step, 0.0)

        # first node: all up moves from prior first node
        stock_tree[step, 0] = max(stock_tree[step - 1, 0] * u - cash_div, 0.0)

        # interior and bottom nodes
        for i in range(1, step + 1):
            parent = stock_tree[step - 1, i - 1]
            stock_tree[step, i] = max(parent * d - cash_div, 0.0)

    option_values = np.zeros(steps + 1, dtype=float)

    terminal_prices = stock_tree[steps, :steps + 1]
    if option_type == "call":
        option_values[:] = np.maximum(terminal_prices - K, 0.0)
    else:
        option_values[:] = np.maximum(K - terminal_prices, 0.0)

    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            hold = discount * (p * option_values[i] + (1.0 - p) * option_values[i + 1])
            
            # Am. option only
            S_node = stock_tree[step, i]
            if option_type == "call":
                exercise = max(S_node - K, 0.0)
            else:
                exercise = max(K - S_node, 0.0)

            option_values[i] = max(hold, exercise)
            # option_value[i] = hold   # Euro. option

    return option_values[0]


# In[47]:


# Binomial tree: American options value & greeks, discreet dividends
def am_option_discreet_div_greeks(S, K, T, r, sigma, option_type, div_dates, div_amts):
    steps = max(100, int(T * 365))

    price = bt_am_discreet_div_value(S, K, T, r, sigma, option_type, div_dates, div_amts, steps)

    dS = S * 0.01
    price_up = bt_am_discreet_div_value(S + dS, K, T, r, sigma, option_type, div_dates, div_amts, steps)
    price_down = bt_am_discreet_div_value(S - dS, K, T, r, sigma, option_type, div_dates, div_amts, steps)
    delta = (price_up - price_down) / (2 * dS)
    gamma = (price_up - 2 * price + price_down) / (dS ** 2)

    dsigma = 0.01
    price_sigma_up = bt_am_discreet_div_value(S, K, T, r, sigma + dsigma, option_type, div_dates, div_amts, steps)
    vega = (price_sigma_up - price) / dsigma

    dr = 0.01
    price_r_up = bt_am_discreet_div_value(S, K, T, r + dr, sigma, option_type, div_dates, div_amts, steps)
    rho = (price_r_up - price) / dr

    dT = 1/365
    if T > dT:
        # Adjust dividend dates for time shift
        div_dates_shifted = [max(0, d - dT) for d in div_dates]
        price_T_down = bt_am_discreet_div_value(S, K, T - dT, r, sigma, option_type, div_dates_shifted, div_amts, steps)
        theta = (price_T_down - price) / dT
    else:
        theta = 0

    return price, delta, gamma, vega, rho, theta


# In[48]:


def am_option_discreet_div_output(data):
    results = []

    for _, row in data.iterrows():
        
        S = row['Underlying']
        K = row['Strike']
        T = row['DaysToMaturity'] / row['DayPerYear']
        r = row['RiskFreeRate']
        sigma = row['ImpliedVol']
        option_type = row['Option Type']
    
        # Parse dividend dates and amounts
        div_dates_str = str(row['DividendDates']).split(',')
        div_amts_str = str(row['DividendAmts']).split(',')
    
        div_dates = [float(d.strip()) / row['DayPerYear'] for d in div_dates_str]  # [] list
        div_amts = [float(a.strip()) for a in div_amts_str]
    
        value, delta, gamma, vega, rho, theta = am_option_discreet_div_greeks(S, K, T, r, sigma, option_type, div_dates, div_amts)
    
        results.append([value, delta, gamma, vega, rho, theta])
    
    return results

# output = pd.DataFrame()
# output['ID'] = data['ID']
# output[['Value', 'Delta', 'Gamma', 'Vega', 'Rho', 'Theta']] = results


# In[49]:


# # Delta hedging
def bs_euro_delta(S, K, r, q, T, sigma, op_type):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if op_type == "Call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1)
        
# S0 = 100.50
# K = 100.0
# r = 0.0525
# q = 0.0
# C0 = 2.50   # call value

# T0 = 15 / 255       # current time to maturity
# T1 = 14 / 255       # after 1 day
# dt = 1 / 255

# n_options = 100
# contract_size = 100
# position_size = n_options * contract_size

# # 1. Solve implied vol and delta
# sigma = implied_volatility_bsm(C0, S0, K, r, q, T0, "Call")
# delta = bs_euro_delta(S0, K, r, q, T0, sigma, "Call")

# # 2. Simulate 1-day stock prices under GBSM
# np.random.seed(0)
# n_sims = 100000

# z = np.random.normal(size=n_sims)
# S1 = S0 * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

# # 3. Reprice call after 1 day
# C1 = bs_price(S1, K, r, q, T1, sigma, "Call")

# # 4. Unhedged short-call PnL
# # Short 100 calls, each for 100 shares
# pnl_unhedged = position_size * (C0 - C1)

# var_unhedged, es_unhedged = var_es_hist_sim(pnl_unhedged, alpha=0.05)

# print("Unhedged 1-day VaR =", var_unhedged)
# print("Unhedged 1-day ES  =", es_unhedged)

# # 5. Delta hedge: buy stock equal to negative position delta
# hedge_shares = position_size * delta

# # Hedged PnL = short-call pnl + stock pnl
# pnl_stock = hedge_shares * (S1 - S0)
# pnl_hedged = pnl_unhedged + pnl_stock

# var_hedged, es_hedged = var_es_hist_sim(pnl_hedged, alpha=0.05)

# print("Hedged 1-day VaR =", var_hedged)
# print("Hedged 1-day ES  =", es_hedged)


# In[50]:


# Ex-post (Realized) return attribution & Risk attribution (11.1)
# data in np array format, weights squeezed
def attribution(returns, weights):
    row, n_stock = returns.shape
    
    w_dynamic = np.zeros((row, n_stock))
    portfolio_ret = np.zeros(row)
    w_curr = weights.copy()
    
    # return attribution
    for i in range(row):
        w_dynamic[i] = w_curr
        portfolio_ret[i] = w_curr @ returns[i]
        
        w_after = w_curr * (1 + returns[i])
        w_curr = w_after / w_after.sum()
    
    cumulative_ret = np.prod(1 + returns, axis=0) - 1  # factor return, ind. of portfolio construction
    p_cumulative_ret = np.prod(1 + portfolio_ret) - 1  # portfolio: total return = return attribution
    
    k = np.log1p(p_cumulative_ret) / (p_cumulative_ret)
    carino = np.where(
        np.isclose(portfolio_ret, 0),
        1 / k,
        np.log1p(portfolio_ret) / (k * portfolio_ret)
    )
    
    return_attrib = np.sum(returns * w_dynamic * carino[:, None], axis=0)
    
    # risk attribution
    weighted_ret = returns * w_dynamic
    
    X = np.column_stack([np.ones(row), portfolio_ret])  # regression of each asset's contribution on total portfolio return
    beta = np.linalg.lstsq(X, weighted_ret, rcond=None)[0][1]
    
    portfolio_sd = np.std(portfolio_ret, ddof=1)
    vol_contrib = beta * portfolio_sd
    
    return cumulative_ret, p_cumulative_ret, return_attrib, vol_contrib, portfolio_sd

# col_name = ['Value', 'x1', 'x2', 'x3', 'Portfolio']
# output = pd.DataFrame([
#     ['TotalReturn', *cumulative_ret, p_cumulative_ret],
#     ['Return Attribution', *return_attrib, p_cumulative_ret],
#     ['Vol Attribution', *vol_contrib, portfolio_sd]
# ], columns=col_name)


# In[51]:


# Ex-ante (Expected) return attribution & Risk attribution
# data in np array format, weights squeezed
def exp_attribution(weights, mu, cov):
    # Expected return attribution
    ret_attr = weights * mu
    port_ret = ret_attr.sum()
    ret_attr_pct = ret_attr / port_ret if not np.isclose(port_ret, 0) else np.full_like(ret_attr, np.nan)

    # Expected volatility attribution
    port_vol = np.sqrt(weights @ cov @ weights)
    mcr_vol = (cov @ weights) / port_vol
    vol_attr = weights * mcr_vol
    vol_attr_pct = vol_attr / vol_attr.sum() if not np.isclose(vol_attr.sum(), 0) else np.full_like(vol_attr, np.nan)

    return ret_attr, ret_attr_pct, vol_attr, vol_attr_pct

# ret_df = pd.DataFrame({
#     "Asset": returns.columns,
#     "Expected_Return_Attribution": ret_attr,
#     "Expected_Return_Attrib_Pct": ret_attr_pct
# })

# vol_df = pd.DataFrame({
#     "Asset": returns.columns,
#     "Expected_Vol_Attribution": vol_attr,
#     "Expected_Vol_Attrib_Pct": vol_attr_pct
#     })


# In[52]:


# Ex-post (Realized) return & risk attribution to factors (11.2)
def attribution_to_factors(factor_ret, stock_ret, beta, weights):
    # stock return = beta1 * F1 + beta2 * F2 + ... + alpha
    row, n_stock = stock_ret.shape
    n_factor = factor_ret.shape[1]
    
    w_dynamic = np.zeros((row, n_stock))
    w_curr = weights.values.squeeze()
    
    portfolio_ret = np.zeros(row)
    factor_contrib = np.zeros((row, n_factor))
    
    for i in range(row):
        w_dynamic[i] = w_curr
        portfolio_ret[i] = w_curr @ stock_ret.values[i]
        
        w_after = w_curr * (1 + stock_ret.values[i])
        w_curr = w_after / w_after.sum()
    
    for i in range(row):
        for j, f in enumerate(factor_ret.columns):
            factor_contrib[i, j] = np.sum(   # how much factor j contributed to portfolio return in period i
                w_dynamic[i] * beta[f].values * factor_ret.iloc[i, j]
            )
    
    alpha_ret = portfolio_ret - factor_contrib.sum(axis=1)
    
    total_factor_ret = np.prod(1 + factor_ret.values, axis=0) - 1
    total_alpha_ret = np.prod(1 + alpha_ret) - 1   # residual total, portfolio alpha
    p_cumulative_ret = np.prod(1 + portfolio_ret) - 1  # portfolio: total return = return attribution
    
    k = np.log1p(p_cumulative_ret) / (p_cumulative_ret)
    carino = np.where(
        np.isclose(portfolio_ret, 0),
        1 / k,
        np.log1p(portfolio_ret) / (k * portfolio_ret)
    )
    
    return_attrib_factor = np.sum(factor_contrib * carino[:, None], axis=0)
    return_attrib_alpha = np.sum(alpha_ret * carino)
    
    components = np.column_stack([factor_contrib, alpha_ret])
    X = np.column_stack([np.ones(row), portfolio_ret])  # regression of components on total portfolio return
    beta = np.linalg.lstsq(X, components, rcond=None)[0][1]
    
    portfolio_sd = np.std(portfolio_ret, ddof=1)
    vol_contrib = beta * portfolio_sd
    
    return total_factor_ret, total_alpha_ret, p_cumulative_ret, return_attrib_factor, return_attrib_alpha, vol_contrib, portfolio_sd

# col_name = ['Value', 'F1', 'F2', 'F3', 'Alpha', 'Portfolio']
# output = pd.DataFrame([
#     ['TotalReturn', *total_factor_ret, total_alpha_ret, p_cumulative_ret],
#     ['Return Attribution', *return_attrib_factor, return_attrib_alpha, p_cumulative_ret],
#     ['Vol Attribution', *vol_contrib, portfolio_sd]
# ], columns=col_name)


# In[53]:


# Ex-post (Realized) return & risk attribution to factors
# data in np array format
def exp_attribution_to_factor(
    weights,
    factor_exposure
    # specific_var=None   # Idiosyncratic variances for each asset. If provided, residual risk attribution is included
):
    w = np.asarray(weights, dtype=float).reshape(-1)
    B = np.asarray(factor_exposure, dtype=float)
    F = np.asarray(factor_cov, dtype=float)

    n_assets, n_factors = factor_exposure.shape

    # Portfolio factor exposures
    b_p = weights @ factor_exposure   # shape (n_factors,)

    # Expected return attribution to factors
    ret_df = None
    if factor_exp_returns is not None:
        lam = np.asarray(factor_exp_returns, dtype=float).reshape(-1)
        factor_ret_attr = b_p * lam
        port_ret = factor_ret_attr.sum()
        factor_ret_attr_pct = (
            factor_ret_attr / port_ret if not np.isclose(port_ret, 0) else np.full_like(factor_ret_attr, np.nan)
        )

        ret_df = pd.DataFrame({
            "Factor": factor_names,
            "Portfolio_Exposure": b_p,
            "Expected_Factor_Return": lam,
            "Expected_Return_Attribution": factor_ret_attr,
            "Expected_Return_Attrib_Pct": factor_ret_attr_pct
        })

    # Risk attribution to factors
    # Portfolio factor variance = b_p @ F @ b_p
    factor_marginal = (factor_cov @ b_p)   # shape (n_factors,)
    factor_var_attr = b_p * factor_marginal

    residual_var_attr = None
    if specific_var is not None:
        D = np.diag(specific_var.reshape(-1))
        residual_var_attr = weights @ D @ weights

    total_var = factor_var_attr.sum()
    if residual_var_attr is not None:
        total_var += residual_var_attr

    port_vol = np.sqrt(total_var)

    factor_vol_attr = factor_var_attr / port_vol
    factor_vol_attr_pct = factor_vol_attr / port_vol if not np.isclose(port_vol, 0) else np.full_like(factor_vol_attr, np.nan)

    vol_rows = pd.DataFrame({
        "Factor": factor_names,
        "Portfolio_Exposure": b_p,
        "Expected_Vol_Attribution": factor_vol_attr,
        "Expected_Vol_Attrib_Pct": factor_vol_attr_pct
    })

    if residual_var_attr is not None:
        residual_row = pd.DataFrame([{
            "Factor": "Residual",
            "Portfolio_Exposure": np.nan,
            "Expected_Vol_Attribution": residual_var_attr / port_vol,
            "Expected_Vol_Attrib_Pct": (residual_var_attr / port_vol) / port_vol if not np.isclose(port_vol, 0) else np.nan
        }])
        vol_df = pd.concat([vol_rows, residual_row], ignore_index=True)
    else:
        vol_df = vol_rows

    return ret_df, vol_df


# In[54]:


# risk parity helper
def make_psd_cov(cov):
    eigvals = np.linalg.eigvalsh(cov)
    tol = 1e-10
    if np.all(eigvals >= -tol):
        return cov
    cov_psd, _ = near_psd(cov)   # near PSD
    # cov_psd, _ = higham(cov)   # Higham
    return cov_psd

def component_sd(w, cov):
    cov = make_psd_cov(cov)
    port_vol = np.sqrt(w @ cov @ w)
    return w * (cov @ w) / port_vol

def component_es(w, sim_returns, alpha, eps=1e-6):
    _, base_es = var_es_hist_sim(sim_returns @ w, alpha)
    ces = np.zeros(len(w))
    for i in range(len(w)):
        w_up = w.copy()
        w_up[i] += eps
        _, bumped_es = var_es_hist_sim(sim_returns @ w_up, alpha)
        ces[i] = w[i] * (bumped_es - base_es) / eps
    return ces

def solve_risk_parity(
    component_fun,
    n_assets,
    args=(),
    budget_vec=None,
    w0=None,
    bounds=None,
    ftol=1e-9,
    maxiter=1000,
    scale=1e5,
):
    if w0 is None:
        w0 = np.ones(n_assets) / n_assets

    if budget_vec is None:
        budget_vec = np.ones(n_assets)
    budget_vec = np.asarray(budget_vec, dtype=float)

    if bounds is None:
        bounds = [(0, None)] * n_assets

    def objective(w):
        rc = np.asarray(component_fun(w, *args), dtype=float)
        normalized_rc = rc / budget_vec
        avg_rc = np.mean(normalized_rc)
        return scale * np.sum((normalized_rc - avg_rc) ** 2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': ftol, 'maxiter': maxiter}
    )

    w_opt = result.x / np.sum(result.x)
    rc_opt = np.asarray(component_fun(w_opt, *args), dtype=float)

    return w_opt, rc_opt, result


# In[55]:


# Normal risk parity with equal weights, matrix input (10.1)
# data in np array format
def normal_risk_parity_equal_w(cov):
    w, rc, res = solve_risk_parity(   # weights, risk contribution, optimization object
        component_fun=component_sd,
        n_assets=cov.shape[0],
        args=(cov,),
        budget_vec=np.ones(cov.shape[0]),
        ftol=1e-9,
        maxiter=1000,
        scale=1e5
    )

    return w


# In[56]:


# Normal risk parity with unequal weights, matrix input (10.2)
# data in np array format
def normal_risk_parity_unequal_w(cov):
    budget = np.array([1.0, 1.0, 1.0, 1.0, 0.5])  # ex: 1/2 weight on X5

    w, rc, res = solve_risk_parity(   # weights, risk contribution, optimization object
        component_fun=component_sd,
        n_assets=cov.shape[0],
        args=(cov,),
        budget_vec=budget,
        ftol=1e-9,
        maxiter=1000,
        scale=1e5
    )
    
    return w


# In[57]:


# ES Risk parity helper
def fit_t_marginals(returns):
    params = []
    u = np.zeros_like(returns, dtype=float)

    for i in range(returns.shape[1]):
        x = returns[:, i]

        # fit Student-t marginal
        df, loc, scale = t.fit(x)
        params.append((df, loc, scale))
        u[:, i] = t.cdf(x, df=df, loc=loc, scale=scale)

    return params, u

def gaussian_copula_simulation(u, marginal_params, n_sim=5000, seed=123):
    rng = np.random.default_rng(seed)

    # convert uniforms to Gaussian scores
    z = np.zeros_like(u)
    for i in range(u.shape[1]):
        ui = rankdata(u[:, i], method="average") / (len(u[:, i]) + 1)
        z[:, i] = norm.ppf(ui)

    # estimate copula correlation
    corr = np.corrcoef(z, rowvar=False)

    # simulate correlated normals
    z_sim = rng.multivariate_normal(
        mean=np.zeros(u.shape[1]),
        cov=corr,
        size=n_sim
    )

    # map to uniforms
    u_sim = norm.cdf(z_sim)

    # map uniforms to simulated returns
    sim_returns = np.zeros_like(u_sim)
    for i, (df, loc, scale) in enumerate(marginal_params):
        sim_returns[:, i] = t.ppf(u_sim[:, i], df=df, loc=loc, scale=scale)

    return sim_returns, corr


# In[58]:


# ES Risk parity with equal weights
# data in np array format
def es_risk_parity_equal_w(returns, alpha):
    # returns_centered = returns - returns.mean(axis=0, keepdims=True)   # Assume 0 mean return by asset
    params, u = fit_t_marginals(returns)
    sim_returns, corr = gaussian_copula_simulation(u, params, n_sim=5000)
    
    n_assets = sim_returns.shape[1]
    
    w, rc, res = solve_risk_parity(   # weights, risk contribution, optimization object
        component_fun=component_es,
        n_assets=n_assets,
        args=(sim_returns, alpha, 1e-6),
        budget_vec=np.ones(n_assets),
        ftol=1e-12,
        maxiter=2000,
        scale=1.0
    )
    
    fitted_df = np.array([param[0] for param in params])   # dof
    
    return w, fitted_df


# In[59]:


# ES Risk parity with unequal weights
# data in np array format
def es_risk_parity_unequal_w(returns, alpha):
    # returns_centered = returns - returns.mean(axis=0, keepdims=True)   # Assume 0 mean return by asset
    params, u = fit_t_marginals(returns)
    sim_returns, corr = gaussian_copula_simulation(u, params, n_sim=5000)
    
    budget = np.array([1.0, 1.0, 1.0, 1.0, 0.5])  # ex: 1/2 weight on X5
    n_assets = sim_returns.shape[1]
    
    w, rc, res = solve_risk_parity(   # weights, risk contribution, optimization object
        component_fun=component_es,
        n_assets=n_assets,
        args=(sim_returns, alpha, 1e-6),
        budget_vec=budget,
        ftol=1e-12,
        maxiter=2000,
        scale=1.0
    )

    fitted_df = np.array([param[0] for param in params])   # dof
    
    return w, fitted_df


# In[60]:


# Max Sharpe ratio (return-to-risk) helper
def portfolio_volatility(w, cov):
    cov = make_psd_cov(cov)
    return np.sqrt(w @ cov @ w)

def portfolio_expected_shortfall(w, sim_returns, alpha=0.05):
    _, port_es = var_es_hist_sim(sim_returns @ w, alpha)
    return port_es

def solve_max_return_to_risk(
    exp_returns,
    risk_fun,
    n_assets,
    risk_args=(),
    rf=0.0,
    w0=None,
    bounds=None,
    ftol=1e-9,
    maxiter=1000,
):
    if w0 is None:
        w0 = np.ones(n_assets) / n_assets

    if bounds is None:
        bounds = [(0, 1)] * n_assets

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    def objective(w):
        port_mean = w @ exp_returns
        port_risk = risk_fun(w, *risk_args)
        return -(port_mean - rf) / port_risk

    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': ftol, 'maxiter': maxiter}
    )

    w_opt = result.x
    port_mean = w_opt @ exp_returns   # portfolio expected returns (mean)
    port_risk = risk_fun(w_opt, *risk_args)   # portfolio vol
    ratio = (port_mean - rf) / port_risk   # Sharpe ratio

    return w_opt


# In[61]:


# Normal Max Sharpe ratio, matrix and means inputs (10.3, 10.4)
# data in np array format
def normal_max_sharpe(data, mu, rf, weights_lb, weights_ub):
    cov = make_psd_cov(data)
    n_assets = len(mu)
    bounds = [(weights_lb, weights_ub)] * n_assets   # ex: (0, 1) or (0.1, 0.5)

    return solve_max_return_to_risk(
        exp_returns=mu,
        risk_fun=portfolio_volatility,
        n_assets=n_assets,
        risk_args=(cov,),
        rf=rf,
        bounds=bounds,
        ftol=1e-9,
        maxiter=1000,
    )


# In[62]:


# Max Sharpe ratio optimization using ES
# data in np array format
def es_max_sharpe(returns, rf, alpha, weights_lb, weights_ub):
    # returns_centered = returns - returns.mean(axis=0, keepdims=True)   # Assume 0 mean return by asset
    params, u = fit_t_marginals(returns)
    sim_returns, corr = gaussian_copula_simulation(u, params, n_sim=5000)
    mu = returns.mean(axis=0) * 255   # annualize mean returns if rf is annual
    
    n_assets = sim_returns.shape[1]
    bounds = [(weights_lb, weights_ub)] * n_assets   # (0, 1) by default

    result = solve_max_return_to_risk(
        exp_returns=mu,
        risk_fun=portfolio_expected_shortfall,
        n_assets=n_assets,
        risk_args=(sim_returns, alpha),
        rf=rf,
        bounds=bounds,
        ftol=1e-12,
        maxiter=2000,
    )

    return result


# In[63]:


# efficient frontier
# data in np array format
def efficient_frontier_with_cml(mu, cov, rf=0.0, graph_points=50, allow_short=False):
    n = len(mu)

    def port_return(w):
        return w @ mu

    def port_variance(w):
        return w @ cov @ w

    bounds = None if allow_short else [(0, 1)] * n
    w0 = np.ones(n) / n

    target_returns = np.linspace(mu.min(), mu.max(), graph_points)
    frontier_risk = []
    frontier_ret = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: port_return(w) - t},
        ]

        res = minimize(
            port_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if res.success:
            risk = np.sqrt(port_variance(res.x))
            ret = port_return(res.x)
            frontier_risk.append(risk)
            frontier_ret.append(ret)

    # Tangency portfolio
    tangency_w = solve_max_return_to_risk(
        exp_returns=mu,
        risk_fun=portfolio_volatility,
        n_assets=n,
        risk_args=(cov,),
        rf=rf,
        bounds=bounds,
    )

    tangency_ret = tangency_w @ mu
    tangency_risk = portfolio_volatility(tangency_w, cov)

    # CML
    cml_x = np.linspace(0, max(frontier_risk) * 1.2, 100)
    slope = (tangency_ret - rf) / tangency_risk
    cml_y = rf + slope * cml_x

    # graph efficient frontier and CML
    plt.figure(figsize=(8, 5))
    plt.plot(frontier_risk, frontier_ret, label="Efficient Frontier")
    plt.plot(cml_x, cml_y, "--", label="Capital Market Line")
    plt.scatter(tangency_risk, tangency_ret, color="red", label="Tangency Portfolio")
    plt.scatter(0, rf, color="black", label="Risk-Free Rate")

    plt.xlabel("Volatility")
    plt.ylabel("Portfolio Expected Return")
    plt.title("Efficient Frontier and Capital Market Line")
    plt.legend()
    plt.grid(True)
    plt.show()

# corr = np.array([
#     [1.0, 0.5, 0.0],
#     [0.5, 1.0, 0.5],
#     [0.0, 0.5, 1.0]
# ])

# std = np.array([0.2, 0.1, 0.05])
# mu = np.array([0.05, 0.04, 0.03])
# rf = 0.02

# cov = np.outer(std, std) * corr

# efficient_frontier_with_cml(mu, cov, rf=rf, allow_short=False)


# In[ ]:




