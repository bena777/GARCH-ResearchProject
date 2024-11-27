import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf
import warnings

warnings.filterwarnings("always")

# example of a GARCH(1,1) model
# This ended up being my foundation for the creation of the generalized_garch function in garch.py

returns_train = yf.download("SPY",start="2010-01-01",end="2019-12-31")["Close"].pct_change().values[1::]*100
returns_test= yf.download("SPY",start="2020-01-01",end="2024-01-01")["Close"].pct_change().values[1::]*100
rolling_predictions = []

model = arch_model(returns_train, p=1, q=1)
fit = model.fit(disp="off")

omega = fit.params.values[1]
alpha_1 = fit.params.values[2]
beta_1 = fit.params.values[3]

cond_variance = fit.conditional_volatility[-1]**2
for i in range(1,len(returns_test)):
    residual = returns_test[i-1]
    cond_variance = omega + (alpha_1 * residual**2) + (beta_1*cond_variance)
    rolling_predictions.append(cond_variance**0.5)
print(rolling_predictions)
plt.plot(abs(returns_test))
plt.plot(rolling_predictions)
plt.show()
