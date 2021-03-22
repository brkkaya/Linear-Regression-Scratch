#%%

from itertools import tee
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
import matplotlib.pyplot as plt
import seaborn as sns

#part1

#%%

def generate_data(m=10):
    X = np.random.rand(m,1)*2
    y= np.sin(2*math.pi*X) +np.random.randn(m,1)
    plt.figure(figsize=(12,8))
    plt.scatter(X,y)
    plt.show()
    return X,y
#%%
X,y =generate_data()
d_order = [0,1,3,9]
# %%
def lin_reg(X,y):
    for order in d_order:
        poly_feat = PolynomialFeatures(order)
        X_poly =poly_feat.fit_transform(X)
        reg = LinearRegression()
        reg.fit(X_poly,y)
        y_poly = reg.predict(X_poly)
        plt.figure()
        sns.scatterplot(X.reshape(-1),y.reshape(-1))
        sns.lineplot(X.reshape(-1),y_poly.reshape(-1),color='red')

# %%
lin_reg(X,y)
# %%
X,y = generate_data(m=100)
lin_reg(X,y)
# %%
#We do not overfit to data in when we have more data 


# %%
# part2
m=100
X=np.random.rand(m,1)
y=100+3*X+np.random.randn(m,1)
sns.scatterplot(X.reshape(-1),y.reshape(-1))
# %%
def linear_regression(X,y,iterNo=1000,eta=0.1):
    b = 0 
    m = X.shape[0]
    W = np.zeros(X.shape[1])
    for _ in range(iterNo):
        
        
        # print(theta.T.shape,X.shape)
        y_pred = (X * W)+b 

        dW = 2/m*np.sum(np.subtract(y_pred,y)*X)
        db = 2/m*np.sum(np.subtract(y_pred,y))
        
        b = b - eta*db
        W = W - eta*dW
    return b,W
    
# %%
t0,t1 = linear_regression(X,y)

# %%

t1
# %%
t0
# %%
y_line = t0 + t1[0]*X
plt.plot(X,y,'r.')
plt.plot(X,y_line,'b-')

# %%

# %%
