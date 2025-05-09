import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
def get_label(stress_value):
    if stress_value <= 50:
        return 0
    else:
        return 1

MARKERS = ['x', '+', 'P', '^', 'v', 's', '*', 'o']
hrv_file = 'hrv_data.csv'
hrv_data = pd.read_csv(hrv_file)[['sdnn', 'lf_hf', 'hr', 'stress']].to_numpy()
stress = hrv_data[:, -1]
label = [get_label(x) for x in stress]

std = StandardScaler()
data = std.fit_transform(hrv_data[:, :-1])

clf = svm.SVC()
clf.fit(data, label)
print("准确度", clf.score(data, label))

tsne = TSNE(n_components=2, init='pca', random_state=42)
X_tsne = tsne.fit_transform(data)
fig = plt.figure()
ax = fig.add_subplot(111)
for c in range(2):
    c_data = X_tsne[np.array(label) == c]
    ax.scatter(c_data[:, 0], c_data[:, 1], label = 'class' + str(c) + ' ' + str(len(c_data)), marker=MARKERS[c])
ax.legend(labels=('无压力', '有压力'))
plt.show()