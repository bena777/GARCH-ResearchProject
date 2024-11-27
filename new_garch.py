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


#engine function of the project, models a GARCH prediction for any GARCH(p,q)
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

#function to calculated exponential weighted moving average for predicting volatility
def ewma(ticker,lamb=0.95):
    returns_test = pd.Series(yf.download(ticker, start="2020-01-01", end="2024-11-01")["Close"].pct_change().values[1::] * 100)
    vars = [returns_test[0]**2]
    for i in range(1,len(returns_test)):
        new_var = lamb * vars[-1] + (1 - lamb) * (returns_test[i - 1] ** 2)
        vars.append(new_var)
    return mean_squared_error(vars,returns_test)


ticker = "SPY"
one_one = garch_generalized(ticker,1,1)
three_three = garch_generalized(ticker,3,3)
print(one_one,three_three)
