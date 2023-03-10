def findVariantFeaturesGWO(features, classes, method) :
    from sklearn.ensemble import ExtraTreesClassifier as GWO2
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import LinearSVC as GWO
    
    out_features = features
    
    if(method == 1) :
        lsvc = GWO(C=0.01, penalty="l1", dual=False).fit(out_features, classes)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(out_features)
    else :
        clf = GWO2(n_estimators=50)
        clf = clf.fit(out_features, classes)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(out_features)
        
    return X_new