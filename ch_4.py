import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

X = 2 * np.random.rand(100, 1)
y = 4 + 3*X + np.random.rand(100,1)

#plt.plot(X,y, 'bo', markersize=1)
#plt.show()


#
#
#
X_b = np.c_[np.ones((100,1)), X]
theta_best =  np.linalg.inv( X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)

X_new = np.array( [[0],[2]])
X_new_b = np.c_[ np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)
print("Predict")
print(y_predict)

lin_reg = LinearRegression()
lin_reg.fit( X, y )
print( lin_reg.intercept_ )
print( lin_reg.coef_ )

lin_reg.predict(X_new)


#
# 4.3 다항회귀
#
# make data
m = 100
X = 6 * np.random.rand( m, 1 ) - 3
y = 0.5 * X**2 + X + 2 + np.random.rand(m,1)
#plt.plot(X,y, 'bo', markersize=1)
#plt.show()

# estimate
poly_features = PolynomialFeatures( degree=2, include_bias=False )
X_poly = poly_features.fit_transform(X)

lin_reg.fit( X_poly, y)
print( lin_reg.intercept_ )
print( lin_reg.coef_ )


#
# 4.6.3 결정 경계
#
from sklearn import datasets
iris = datasets.load_iris()
list( iris.keys() )
X = iris["data"][:,3:]
y = (iris["target"]==2).astype(np.int)

# train
log_reg = LogisticRegression()
log_reg.fit( X, y )

# get info
X_new = np.linspace(0, 3, 1000).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)
#plt.plot( X_new, y_proba[:,1], 'g-')
#plt.plot( X_new, y_proba[:,0], 'b--')
#plt.show()




#
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression(solver='liblinear', C=10**10, random_state=42)
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)

left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Iris-Virginica 아님", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
plt.xlabel("꽃잎의 길이", fontsize=14)
plt.ylabel("꽃잎의 폭", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])

#plt.show()




#
#
#
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]
softmax_reg = LogisticRegression( multi_class="multinomial", solver="lbfgs", C=10 )
softmax_reg.fit(X, y)
softmax_reg.predict([[5,2]])
softmax_reg.predict_proba([[5,2]])

x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]


y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris-Setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("꽃잎의 길이", fontsize=14)
plt.ylabel("꽃잎의 폭", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
#save_fig("softmax_regression_contour_plot")
plt.show()