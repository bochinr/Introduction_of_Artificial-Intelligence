import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('attention.csv')
print(data)

beta = data['beta'].to_numpy()
theta = data['theta'].to_numpy()
attention = theta/beta

length = len(attention)
x = np.linspace(0, length, length)
plt.plot(x, attention)
