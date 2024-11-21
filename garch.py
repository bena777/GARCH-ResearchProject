import yfinance as yf
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = (yf.download("SPY",start="2000-01-01",end="2020-06-01")['Close'].pct_change().values*100)[1::]
print(df)


model = arch_model(df,p=2,q=2)
model_fit = model.fit()
print(model_fit.summary())


def predict_garch(returns, size, rolling_prediction=None): # this function is creating a rolling volatility predicted by garch
    if rolling_prediction is None:
        rolling_prediction = []
    for i in range(size):
        train = returns[:-(size-i)]
        model = arch_model(train,p=2, q=2)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_prediction.append(np.sqrt(pred.variance.values[-1,:][0]))
    return rolling_prediction

rolling = predict_garch(df,1500)
plt.plot(df[-1500:])
plt.plot(rolling)
plt.ylabel("Volatility")
plt.xlabel("Date")
plt.show()
