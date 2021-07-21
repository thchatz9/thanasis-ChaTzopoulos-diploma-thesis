import pandas as pd
import numpy as np


#data input
data=pd.read_csv("matrix_k_7.csv")
print(data)
X=data.iloc[:,1:]
y=data.iloc[:,0]
print(y)
print(X)

#sinartisi axiologisis selected features
def evaluation(X) :
    i=0
    k=0
    j=0
    variance = X.var()
    print(variance)
    for i in X:
        if X[i].var() == 0:
            k = k + 1
            print("Column %s is one unique value column \n" % i)

        if X[i].var() > 0.1:
            j=j+1
            print("Column %s has large variance \n" % i)
    print("IPARXOUN %s STILES ME IDIA STOIXEIA" %k)
    print("iparnoun %s stiles me variance :"%j)
evaluation(X)

X_new=0

#lets start with a variance threshold
from sklearn.feature_selection import VarianceThreshold

selector=VarianceThreshold(.8*(1-.8))
sel_var=selector.fit_transform(X)
print (sel_var)
X_new=X[X.columns[selector.get_support(indices=True)]]
print(X[X.columns[selector.get_support(indices=True)]])
print("X_new=",X_new)
evaluation(X_new)

#next a pearson correlation coefficient


#correlation matrix
cor = X_new.corr().abs()
print(cor)
cor_target = abs(cor["GGTGTTA"])
selected = cor_target[cor_target>0.75]
print("selected features is : = ",selected)
X_new=X_new[selected.index]
print(X_new)
evaluation(X_new)



#next step





