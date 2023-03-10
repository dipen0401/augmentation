import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
from scipy import stats
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import modelDL as mdl
import AugmentImages as imgAugment

warnings.filterwarnings("ignore")

from tensorflow import keras
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import time

def findPRFC(predicted, actual, display=True) :
    f1 = f1_score(predicted, actual, average="macro")
    pre = precision_score(predicted, actual, average="macro")
    acc = accuracy_score(predicted, actual)
    rec = recall_score(predicted, actual, average="macro")
    conf_matrix = confusion_matrix(predicted, actual);
    diff_arr = np.array(predicted) - np.array(actual)
    idx = np.where(diff_arr == 0)[0]
    
    if(display) :
        print("Test f1 score : %s "% f1)
        print("Test Precision score : %s "% pre)
        print("Test accuracy score : %s "% acc)
        print("Test Recall score : %s "% rec)
        print('Confusion matrix')
        print(conf_matrix)
    return idx
    
def classifyDLAndFindCorrect(features, classes, test, test_classes, method) :
    t1 = time.time()
    
    Y = classes.astype(np.int8)
    X = np.asarray(features)
    X = X.reshape((X.shape[0], X.shape[1]))
    
    Y_test = test_classes.astype(np.int8)
    X_test = np.asarray(test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

    scaler = StandardScaler()  
    scaler.fit(X)
    X_train = scaler.transform(X)  
    X_Test = scaler.transform(X_test)
    
    if(method == 1) :
        neigh1 = KNeighborsClassifier(n_neighbors=1)
        neigh1.fit(X_train, Y)
        y_pred1 = neigh1.predict(X_Test)  
        return findPRFC(test_classes, y_pred1)
    elif(method == 2) :
        neigh2 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
        neigh2.fit(X_train, Y)
        y_pred2 = neigh2.predict(X_Test)  
        return findPRFC(test_classes, y_pred2)
    elif(method == 3) :
        neigh3 = LinearSVC(random_state=0, tol=1e-5)
        neigh3.fit(X_train, Y)
        y_pred3 = neigh3.predict(X_Test)  
        return findPRFC(test_classes, y_pred3)
    elif(method == 4) :
        logisticRegressionClassifier = LogisticRegression(random_state=0,multi_class='auto',solver='lbfgs',max_iter=1000)
        logisticRegressionClassifier.fit(X_train,Y)
        y_pred_lrc = logisticRegressionClassifier.predict(X_test)
        return findPRFC(test_classes, y_pred_lrc)
    elif(method == 5) :
        randomForestClassifier = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
        randomForestClassifier.fit(X_train,Y)
        y_pred_rfc = randomForestClassifier.predict(X_test)
        return findPRFC(test_classes, y_pred_rfc)
    
def cnn1DClassify(features, classes, test, test_classes, files) :
    t1 = time.time()
    
    #classes = classes - 1
    #test_classes = test_classes - 1
    
    Y = classes
    X = features
    X = X.reshape((X.shape[0], X.shape[1]))
    
    Y_test = test_classes
    X_test = test
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
    
    scaler = StandardScaler()  
    scaler.fit(X)
    X_train = scaler.transform(X)  
    X_Test = scaler.transform(X_test)  
    
    nclass = len(np.unique(classes))
    inp = Input(shape=(len(features[0]), 1))
    img_1 = Convolution1D(4, kernel_size=3, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(4, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(8, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(8, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(8, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(8, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(16, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(16, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    
    file_path = "mitbih_model.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]  # early
    
    model.fit(X, Y, epochs=2, verbose=1, callbacks=callbacks_list, validation_split=0.1)
    model.load_weights(file_path)
    
    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)
    counter=0
    pred_final = pred_test
    t2 = time.time()
    
    delay = t2 - t1
    print('CPU Time needed %0.0f s' % (delay))
    f1 = f1_score(Y_test, pred_test, average="macro")
    print("Test f1 score : %s "% f1)
    pre = precision_score(Y_test, pred_test, average="macro")
    print("Test Precision score : %s "% pre)
    acc = accuracy_score(Y_test, pred_final)
    print("Test accuracy score : %s "% acc)
    rec = recall_score(Y_test, pred_test, average="macro")
    print("Test Recall score : %s "% rec)
    conf_matrix = confusion_matrix(Y_test, pred_test);
    print('Confusion matrix')
    print(conf_matrix)
    
    y_data = stats.norm.pdf(Y_test, 0, 1)
    plt.plot(Y_test, y_data);
    plt.show()
    
    outFolder = 'augmented/'
    for count in range(0, len(files)) :
        imgAugment.augmentImage(files[count], outFolder)
        

dataset = 'out_features.csv'
t1 = time.time()

df_train = pd.read_csv(dataset, header=None)
df_train = df_train.sample(frac=1)

class_idx = int(df_train.size/len(df_train)-1)
feat_idx = class_idx-1
Y = np.array(df_train[class_idx].values)
files = Y
Y = [0] * len(Y)
for count in range(0, len(Y)) :
    Y[count] = count%3
Y = np.array(Y)
X = np.array(df_train[list(range(feat_idx))].values)[..., np.newaxis]
X = X.reshape((X.shape[0], X.shape[1]))

mode = 'test'

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.4, random_state=0)
if(mode == 'train') :
    X_test = X_train
    y_test = y_train
    
print('Classify 1...')
arr1 = classifyDLAndFindCorrect(X_train, y_train, X_test, y_test, 1)
print('Classify 2...')
arr2 = classifyDLAndFindCorrect(X_train, y_train, X_test, y_test, 2)
print('Classify 3...')
arr3 = classifyDLAndFindCorrect(X_train, y_train, X_test, y_test, 3)
print('Classify 4...')
arr4 = classifyDLAndFindCorrect(X_train, y_train, X_test, y_test, 4)
print('Classify 5...')
arr5 = classifyDLAndFindCorrect(X_train, y_train, X_test, y_test, 5)
cnn1DClassify(X_train, y_train, X_test, y_test, files)
