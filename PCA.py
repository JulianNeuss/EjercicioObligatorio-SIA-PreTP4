# Libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

df = pd.read_csv('europe.csv')

columns_names = df.columns.tolist()
columns_names.pop(0)
print(columns_names)
# print("Columns names:")
# print(columns_names)
# Output
# Columns names:
# ['Country', 'Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']

# print(df.shape)
# Output
# (28, 8)

# First 5 elements
# print(df.head())
#           Country    Area    GDP  ...  Military  Pop.growth  Unemployment
# 0         Austria   83871  41600  ...      0.80        0.03           4.2
# 1         Belgium   30528  37800  ...      1.30        0.06           7.2
# 2        Bulgaria  110879  13800  ...      2.60       -0.80           9.6
# 3         Croatia   56594  18000  ...      2.39       -0.09          17.7
# 4  Czech Republic   78867  27100  ...      1.15       -0.13           8.5

# Correlation heatmap
# correlation = df.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
# plt.title('Correlation between different fearures')
# plt.show()

X = df.iloc[:, 1:8].values
y = df.iloc[:, 0].values
# print(X)
# [[ 8.38710e+04  4.16000e+04  3.50000e+00  7.99100e+01  8.00000e-01
#    3.00000e-02  4.20000e+00]
# .......
# [ 2.43610e+05  3.65000e+04  4.50000e+00  8.01700e+01  2.70000e+00
#   5.50000e-01  8.10000e+00]]

# print(y)
# ['Austria' 'Belgium' 'Bulgaria' 'Croatia' 'Czech Republic' 'Denmark'
#  'Estonia' 'Finland' 'Germany' 'Greece' 'Hungary' 'Iceland' 'Ireland'
#  'Italy' 'Latvia' 'Lithuania' 'Luxembourg' 'Netherlands' 'Norway' 'Poland'
#  'Portugal' 'Slovakia' 'Slovenia' 'Spain' 'Sweden' 'Switzerland' 'Ukraine'
#  'United Kingdom']

# print(np.shape(X))
# (28, 7)
# print(np.shape(y))
# (28,)
X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)

# print("Mean:")
# print(mean_vec)
# Mean:
# [ 4.16333634e-17 -3.96508223e-17 -7.93016446e-17 -5.49560397e-15
#   -3.17206578e-17 -1.98254112e-17 -3.09276414e-16]

# print("Cov Matrix:")
# print(cov_mat)
# Cov Matrix:
# [[ 1.03703704 -0.14364715  0.33194482 -0.02247327  0.10545766 -0.09190295
#    0.02639174]
#  [-0.14364715  1.03703704 -0.51100531  0.72693995 -0.2949983   0.78858747
#  -0.5473025 ]
# [ 0.33194482 -0.51100531  1.03703704 -0.7043479   0.05007077 -0.49647685
# 0.20613618]
# [-0.02247327  0.72693995 -0.7043479   1.03703704 -0.06559162  0.80020784
#  -0.25508553]
# [ 0.10545766 -0.2949983   0.05007077 -0.06559162  1.03703704 -0.2928048
# 0.30310386]
# [-0.09190295  0.78858747 -0.49647685  0.80020784 -0.2928048   1.03703704
#  -0.18124219]
# [ 0.02639174 -0.5473025   0.20613618 -0.25508553  0.30310386 -0.18124219
# 1.03703704]]


eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)
# Eigenvectors
# [[ 1.24873902e-01 -1.72872202e-01  8.98296740e-01 -3.24016926e-01
#    -6.66428246e-02  1.90118083e-01  4.48503976e-02]
#  [-5.00505858e-01 -1.30139553e-01  8.39557607e-02  3.90632444e-01
#  3.97408435e-01  6.38657073e-01 -8.42554739e-02]
# [ 4.06518155e-01 -3.69657243e-01  1.98194675e-01  6.89500539e-01
# 2.26700295e-01 -3.23867263e-01  1.64685649e-01]
# [-4.82873325e-01  2.65247797e-01  2.46082460e-01 -1.01786561e-01
#  5.07031305e-01 -6.06434187e-01  2.67714373e-02]
# [ 1.88111616e-01  6.58266888e-01  2.43679433e-01  3.68147581e-01
#                                                   -1.37309597e-01  3.55960680e-02 -5.62374796e-01]
# [-4.75703554e-01  8.26219831e-02  1.63697207e-01  3.47867772e-01
#  -6.71146682e-01 -1.20855625e-01  3.92462767e-01]
# [ 2.71655820e-01  5.53203705e-01  5.00135736e-04  1.01587422e-02
# 2.44662434e-01  2.59704965e-01  7.01967912e-01]]
#
# Eigenvalues
# [3.34669033 1.23109094 1.10256796 0.47480597 0.13029529 0.17492107
#  0.79888768]


pca = PCA().fit(X_std)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlim(0,7,1)
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance')
# plt.show()


sklearn_pca = PCA(n_components=5)
Y_sklearn = sklearn_pca.fit_transform(X_std)


# print(Y_sklearn)

def myplot(score, coeff, labels=None, variables=None):
    plt.figure(figsize=(16,10))
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c=y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')
    for i in range(df.shape[0]):
        plt.text(xs[i] * scalex * 1.05, ys[i] * scaley * 1.05, s=df.values[i, 0],
                 fontdict=dict(color='white', size=10),
                 bbox=dict(facecolor='black',alpha=0.5))

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()


# Call the function. Use only the 2 PCs.
temp = y
# Fix y data into numbers
counter = 0
pais = []
for i in y:
    pais.append(counter)
    counter += 1

y = pais
myplot(Y_sklearn[:, 0:2], np.transpose(sklearn_pca.components_[0:2, :]), columns_names, temp)
plt.show()
