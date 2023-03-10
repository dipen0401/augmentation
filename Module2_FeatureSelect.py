import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time
import featureClassification as fc
import FeatureSelect as fs
from sklearn.model_selection import train_test_split
import easygui as gui
import statistics
import csv

t1 = time.time()

print('Reading...')
df_train = pd.read_csv("feature_sets.csv", header=None).fillna(0)
df_train = df_train.sample(frac=1)

class_idx = int(df_train.size/len(df_train))
class_idx = class_idx - 1
Y = np.array(df_train[class_idx].values)
X = np.array(df_train[list(range(class_idx))].values)[..., np.newaxis]
X = X.reshape((X.shape[0], X.shape[1]))

print('Processing...')
X_new = fs.findVariantFeaturesGWO(X, Y, 2)
print('Done!')
new_array=np.vstack([X_new.T, Y])
new_array = new_array.T
f = open('out_features.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerows(new_array)
f.close()