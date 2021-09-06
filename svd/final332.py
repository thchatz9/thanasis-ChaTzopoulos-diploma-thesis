import pandas as pd
import numpy as np


#data input
data=pd.read_csv("matrix_k_7.csv")
print(data)
X=data.iloc[:,1:]
print("we must check if we talking about regression or classification problem"
      "1:REGRESSION - y table not important we only need the 3 algorithm sequence"
      "2:CLASSIFICATION - Y table must be initialized and uses on step 3 to RFE")
j = input("Enter 1 for regression - 2 for classification:")
if  j ==1 :
    print(" WE have regression problem")
if j ==2:
    print(" we have classification problem ")
if j!=1 or j!=2 :
    print("give a value between 1-2")


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
X_new.shape
evaluation(X_new)

if j==2:
 ##RFE
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeClassifier
    rfe=RFE(estimator=DecisionTreeClassifier(), n_features_to_select=10)
    rfe.fit(X_new, y)
    # summarize all features
    for i in range(X.shape[1]):
         print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
    model=DecisionTreeClassifier()
    X_new= model.fit(rfe.transform(X_new), y)
evaluation(X_new)
     
from sklearn.decomposition import PCA

pca=PCA(n_components=4)
df_pca=pca.fit_transform(X_new)
df_pca=pd.DataFrame(df_pca)
print(df_pca.shape)
df_pca.round(2).head()
print(pca.explained_variance_ratio_.round(2)[:10])
df_orig = pca.inverse_transform(df_pca)
pd.DataFrame(df_orig).round().head()
X_new = df_pca
X_orig=pca.inverse_transform(df_pca)
print(X_new.shape)
print(pca.components_)
print(X_new)
print(X_orig)
