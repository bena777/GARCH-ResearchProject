import yfinance as yf
from arch import arch_model
import random
import numpy as np
import matplotlib.pyplot as plt
df = [100]

for i in range(0,1000):
    high_low = random.randint(1,3)
    for j in range(0,20):
        up_down = random.randint(1,2)
        if up_down == 1:
            if high_low == 1:
                df.append(df[-1]*(random.randrange(1050,1100))/1000)
            elif high_low == 2:
                df.append(df[-1] * (random.randrange(1030, 1050)) / 1000)
            else:
                df.append(df[-1] * (random.randrange(1001, 1002)) / 1000)
        else:
            if high_low == 1:
                df.append(df[-1] * (random.randrange(925, 950)) / 1000)
            elif high_low == 2:
                df.append(df[-1] * (random.randrange(950, 970)) / 1000)
            else:
                df.append(df[-1] * (random.randrange(998, 999)) / 1000)

df_pct = []
for i in range(1,len(df)):
    df_pct.append((df[i-1]-df[i])/df[i-1])

model = arch_model(np.array(df_pct),p=3,q=3)
fit = model.fit()
print(fit.summary())


plt.plot(df)
plt.ylabel("Value")
plt.xlabel("Days")
plt.show()

