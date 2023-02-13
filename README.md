# Support-Vector-Machine

Support Vector Machine implementation on IRIS Dataset 

## Dependencies

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
```

## Create an Instance of SVM and Fit out the data.
Data is not scaled so as to be able to plot the support vectors

```
svc = svm.SVC(kernel ='linear', C = 1).fit(X, y)
```

## create a mesh to plot

```
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
		np.arange(y_min, y_max, h))
```


##  Plot the data for Proper Visual Representation

```
plt.figure(figsize=(16,9))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap = plt.cm.Paired, alpha = 0.8)

plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
```

###  Output the Plot

```
plt.show()
```

![image](https://user-images.githubusercontent.com/78251168/218433633-b9539f2f-73f5-4035-88b4-9cd2840ebd6d.png)
