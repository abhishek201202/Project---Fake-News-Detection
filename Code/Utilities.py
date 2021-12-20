from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def PCAPercentage(train, test, percentage=0.95):
    pca = PCA()
    pca.fit_transform(train)
    total_variance = sum(pca.explained_variance_)
    k = 0    ## number of features to take
    curr_variance = 0
    while curr_variance/total_variance < percentage:
        curr_variance += pca.explained_variance_[k]
        k += 1
    print("Optimal Number of Features: ", k)
    pca = PCA(n_components=k)    
    train_pca = pca.fit_transform(train)
    test_pca = pca.transform(test)
    return train_pca, test_pca
    
def FeatureScaling(train, test):
    sc = StandardScaler()
    train_std = sc.fit_transform(train)
    test_std = sc.transform(test)
    return train_std, test_std