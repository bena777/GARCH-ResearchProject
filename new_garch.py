import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf
import warnings
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("always")
returns_train = yf.download("SPY", start="2010-01-01", end="2019-12-31")["Close"].pct_change().values[1::] * 100
returns_test = yf.download("SPY", start="2020-01-01", end="2024-12-31")["Close"].pct_change().values[1::] * 100

def mean_squared_error(estimate,actual): # estimate and actual need to be the same length
    err = 0
    for i in range(len(estimate)):
        err+=(estimate[i]-actual[i])**2
    return err/len(estimate)

def plot_prediction(estimate,actual,p,q,ticker,dates):
    plt.plot(dates[1::],actual,label="Actual Volatility")
    plt.xticks(dates[::120],rotation=25)
    plt.plot(dates[1::],estimate,label="Predicted Volatility")
    lower = [x*(1-1.96) for x in estimate]
    upper = [x*(1+1.96) for x in estimate]
    plt.fill_between(dates[1::],[-x*1.96 for x in estimate], [x*1.96 for x in estimate], color="gray", alpha=0.5,label="95% Confidence Interval")
    plt.title(f"GARCH ({p},{q}) model of {ticker}")
    plt.legend()
    plt.ylabel("Volatility")
    plt.xlabel("Date")
    plt.show()

def garch_1_1(ticker):
    returns_train = yf.download(ticker, start="2010-01-01", end="2019-12-31")["Close"].pct_change().values[1::] * 100
    returns_test = yf.download(ticker, start="2020-01-01", end="2024-01-01")["Close"].pct_change().values[1::] * 100
    rolling_predictions = []
    model = arch_model(returns_train, p=1, q=1)
    fit = model.fit(disp="off")
    cond_variance = fit.conditional_volatility[-1]**2
    omega = fit.params.values[1]
    alpha_1 = fit.params.values[2]
    beta_1 = fit.params.values[3]
    mu = sum(returns_train)/len(returns_train)
    for i in range(1,len(returns_test)):
        residual = returns_test[i-1]-mu # return of t-1 day
        cond_variance = omega + (alpha_1 * residual**2) + (beta_1*cond_variance) # conditional (bayesian) variance of t-1 da1y
        rolling_predictions.append(cond_variance**0.5)
    plot_prediction(rolling_predictions,returns_test,1,1,ticker)
    return mean_squared_error(rolling_predictions,returns_test)
def garch_2_2(ticker):
    returns_train = yf.download(ticker, start="2010-01-01", end="2019-12-31")["Close"].pct_change().values[1::] * 100
    returns_test = yf.download(ticker, start="2020-01-01", end="2024-01-01")["Close"].pct_change().values[1::] * 100
    rolling_predictions = []
    model = arch_model(returns_train, p=2, q=2)
    fit = model.fit(disp="off")
    cond_variance_1 = fit.conditional_volatility[-1]**2
    cond_variance_2 = fit.conditional_volatility[-2]**2
    omega = fit.params.values[1]
    alpha_1 = fit.params.values[2]
    alpha_2 = fit.params.values[3]
    beta_1 = fit.params.values[4]
    beta_2 = fit.params.values[5]
    for i in range(2,len(returns_test)):
        residual_1 = returns_test[i-1] # return of t-1 day
        residual_2 = returns_test[i-2]
        cond_variance = omega + (alpha_1 * residual_1**2) + (alpha_2 * residual_2**2) + (beta_1*cond_variance_1) + (beta_2*cond_variance_2) # conditional (bayesian) variance of t-1 da1y
        rolling_predictions.append(cond_variance**0.5)
    plot_prediction(rolling_predictions,returns_test,2,2,ticker)
    return mean_squared_error(rolling_predictions,returns_test)
def garch_3_3(ticker):
    returns_train = yf.download(ticker, start="2010-01-01", end="2019-12-31")["Close"].pct_change().values[1::] * 100
    returns_test = yf.download(ticker, start="2020-01-01", end="2024-11-01")["Close"].pct_change().values[1::] * 100
    rolling_predictions = []
    model = arch_model(returns_train, p=3, q=3)
    fit = model.fit(disp="off")
    cond_variance_1 = fit.conditional_volatility[-1]**2
    cond_variance_2 = fit.conditional_volatility[-2]**2
    cond_variance_3 = fit.conditional_volatility[-3]**2
    omega = fit.params.values[1]
    alpha_1 = fit.params.values[2]
    alpha_2 = fit.params.values[3]
    alpha_3 = fit.params[4]
    beta_1 = fit.params.values[5]
    beta_2 = fit.params.values[6]
    beta_3 = fit.params.values[7]
    for i in range(3,len(returns_test)):
        residual_1 = returns_test[i-1] # return of t-1 day
        residual_2 = returns_test[i-2]
        residual_3 = returns_test[i-3]
        cond_variance = omega + (alpha_1 * residual_1**2) + (alpha_2 * residual_2**2) + (alpha_3*residual_3**2) + (beta_1*cond_variance_1) + (beta_2*cond_variance_2) + (beta_3*cond_variance_3) # conditional (bayesian) variance of t-1 da1y
        rolling_predictions.append(cond_variance**0.5)
    plot_prediction(rolling_predictions,returns_test,3,3,ticker)
    return mean_squared_error(rolling_predictions,returns_test)


def garch_generalized(ticker,p,q):
    returns_train = yf.download(ticker, start="2010-01-01", end="2019-12-31")["Close"].pct_change().values[1::] * 100
    returns_test = yf.download(ticker, start="2020-01-01", end="2024-11-01")["Close"].pct_change().values[1::] * 100
    dates = yf.download(ticker,start="2020-01-01",end="2024-11-01").index
    model = arch_model(returns_train,p=p,q=q)
    fit = model.fit(disp="off")
    cond_variances = []
    alphas = []
    betas = []
    omega = fit.params.values[1]
    mu = sum(returns_train)/len(returns_train)
    for i in range(1,p+1):
        alphas.append(fit.params[1+i])
    for i in range(1,q+1):
        betas.append(fit.params[1+p+i])
        cond_variances.append(fit.conditional_volatility[-i]**2)
    for x in range(max(p,q),len(returns_test)):
        cond_variance = omega
        for i in range(0,p):
            resid = returns_test[x-i-1]-mu
            cond_variance+=alphas[i]*resid**2
        for j in range(0,q):
            cond_variance+=betas[j]*cond_variances[-(j+1)]
        cond_variances.append(cond_variance)
    predicted_volatilities = [x ** 0.5 for x in cond_variances]
    plot_prediction(predicted_volatilities,returns_test,p,q,ticker,dates)
    return mean_squared_error(predicted_volatilities,returns_test)

def ema(ticker):
    returns_test = pd.Series(yf.download(ticker, start="2020-01-01", end="2024-11-01")["Close"].pct_change().values[1::] * 100)
    return returns_test.ewm()


# # model = arch_model(np.array([3,4,5,6,2,1]),p=2,q=2)
# # fit = model.fit()
# # print(fit.conditional_volatility)
ticker = "SPY"
one_one = garch_generalized(ticker,1,2)
three_three = garch_generalized(ticker,3,3)
print(one_one,three_three)
