#Online Learning demo
import numpy as np
from online_pa import *
from sklearn.datasets import load_svmlight_files
from sklearn.datasets import load_svmlight_file
from sklearn import svm
from sklearn import preprocessing
import timeit
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
#%%
'''
Load the dataset and normalize it using min-max method
'''

X_train, Y_train, X_test, Y_test = load_svmlight_files(("splice_train.txt", "splice_test.txt"))
X_train = X_train.toarray()
X_test = X_test.toarray()
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

iterations = 5000

#%%

'''
Set the C values for first and second update strategy for Online-PA algorithm.
The pa_binaryClf function takes five arguments: 
train_data, 
labels (-1, 1) since we are in binary classification context, 
iterations:number of iterations of PA algorithm,
C: C values for the update strategies
update_option (string): 'classic', 'first', 'second'

We measure the accuracy and the execution time.
'''

C = np.array(([0.001, 0.01, 0.1, 1, 5]))
pa_acc = np.zeros((len(C)))

elapsed_pa = np.zeros((len(C)))
elapsed_svm = np.zeros((len(C)))

pa_acc = np.zeros(len(C))
for i in range(len(C)):
                
    start_time = timeit.default_timer()
    w = pa_binaryClf(X_train, Y_train, iterations, C[i], 'second')
    elapsed_pa[i] = timeit.default_timer() - start_time
    pred = np.sign(np.dot(w.T, X_test.T))
    c = np.count_nonzero(pred - Y_test)
    pa_acc[i] = (1 - float(c) / X_test.shape[0])
    
    print('PA accuracy: {}'.format(1 - float(c) / X_test.shape[0]))
#%%
'''
Study the behavior of PA algorithm using the three different update strategies.
For this task, we consider different sizes of training sets and we compare the accuracies
For convenience we fix C = 0.01, as it gives the best accuracy for each update rule.
'''
C = 0.01

#Process them sequentialy
ranges = np.array(([5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]))
elapsed_pa_b = np.zeros((len(ranges)))
pa_acc_classic = np.zeros(len(ranges))
pa_acc_first = np.zeros(len(ranges))
pa_acc_second = np.zeros(len(ranges))
for s in range(len(ranges)):
    
    
    #idx = np.random.randint(1000, size = ranges[s])
    X = X_train[0:ranges[s], :]
    Y = Y_train[0:ranges[s]]
    start_time = timeit.default_timer()
    w_classic = pa_binaryClf(X, Y, iterations, C, 'classic')
    w_first = pa_binaryClf(X, Y, iterations, C, 'first')
    w_second = pa_binaryClf(X, Y, iterations, C, 'second')
    #elapsed_pa_b[s] = timeit.default_timer() - start_time
    pred_classic = np.sign(np.dot(w_classic.T, X_test.T))
    pred_first = np.sign(np.dot(w_first.T, X_test.T))
    pred_second = np.sign(np.dot(w_second.T, X_test.T))
    
    c_classic = np.count_nonzero(pred_classic - Y_test)
    c_first = np.count_nonzero(pred_first - Y_test)
    c_second = np.count_nonzero(pred_second - Y_test)
    pa_acc_classic[s] = (1 - float(c_classic) / X_test.shape[0])
    pa_acc_first[s] = (1 - float(c_first) / X_test.shape[0])
    pa_acc_second[s] = (1 - float(c_second) / X_test.shape[0])
    
    #print('PA accuracy: {}'.format(1 - float(c) / X_test.shape[0]))

plt.figure(figsize=(15,10))
plt.plot(ranges, pa_acc_classic, label="$PA - Classic$")
plt.plot(ranges, pa_acc_first, label="$PA - First Relaxation$")
plt.plot(ranges, pa_acc_second, label="$PA - Second Relaxation$")

plt.legend()
plt.xlabel("Number of observations in training set")
plt.ylabel("Accuracy")
plt.show()

#%%
'''
To compare with libSVM, first we conduct a k-fold cross validation on our training data, to tune
the C hyper-parameter, and after we apply the best C value to train and test our model.
'''
k_fold = 10
rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=65555)
C = np.array(([2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5]))
best_acc = 0
svm_acc = np.zeros(k_fold)
#svm_acc = np.empty((k_fold, 0)).tolist()
tmp_acc = np.zeros(k_fold)
avg_acc = np.zeros(len(C))

for C_values in range(len(C)):
    k = 0
    count = 0
    for train_index, test_index in rkf.split(X_train):
        train_data, train_labels = X_train[train_index], Y_train[train_index]
        test_data, test_labels = X_test[test_index], Y_test[test_index]
        model = svm.libsvm.fit(train_data, train_labels, kernel='linear', C=C[C_values])
        y_pred = svm.libsvm.predict(test_data, *model)
        y_pred[y_pred==0] = -1
        c1 = np.count_nonzero(y_pred - test_labels)
        svm_acc[count] = (1 - float(c1) / test_data.shape[0])
        count += 1

    
    avg_acc[C_values] = np.sum(svm_acc) / len(svm_acc)
    k += 1
    if (avg_acc[C_values] >= best_acc):
        best_acc = avg_acc[C_values]
        best_C = C[C_values]
        
#%%
'''
SVM with linear kernel
'''
start_time = timeit.default_timer()

svm_model = svm.libsvm.fit(X_train, Y_train, kernel='linear', C=best_C)
elapsed_svm = timeit.default_timer() - start_time
predictions = svm.libsvm.predict(X_test, *svm_model)
predictions[predictions==0] = -1
c2 = np.count_nonzero(predictions - Y_test)
svm_acc = (1 - float(c2) / X_test.shape[0])
print('SVM accuracy: {}'.format(1 - float(c2) / X_test.shape[0]))
    
#%%
'''
Introduce some noise by flipping randomly some training labels
'''
q = np.array(([0.05*len(Y_train), 0.1*len(Y_train), 0.2*len(Y_train), 0.3*len(Y_train), 0.4*len(Y_train), 0.5*len(Y_train)]))
C1 = 0.01
#mask = np.copy(Y_train)
#mask[0:10] = mask[0:10]*-1
#mask[100:110] = mask[100:110]*-1
#mask[200:210] = mask[200:210]*-1
#mask[500:510] = mask[500:510]*-1
#mask[900:920] = mask[900:920]*-1
paClassic_acc = np.zeros(len(q))
paFirst_acc = np.zeros(len(q))
paSecond_acc = np.zeros(len(q))
for s in range(len(q)):
    idx = np.random.randint(len(Y_train), size = np.int(q[s]))
    
    #start_time = timeit.default_timer()
    mask = np.copy(Y_train)
    mask[idx] = mask[idx]*(-1)
    
    w1 = pa_binaryClf(X_train, mask, iterations, C1, 'classic')
    w2 = pa_binaryClf(X_train, mask, iterations, C1, 'first')
    w3 = pa_binaryClf(X_train, mask, iterations, C1, 'second')
    #elapsed_pa[s] = timeit.default_timer() - start_time
    pred1 = np.sign(np.dot(w1.T, X_test.T))
    pred2 = np.sign(np.dot(w2.T, X_test.T))
    pred3 = np.sign(np.dot(w3.T, X_test.T))
    
    d1 = np.count_nonzero(pred1 - Y_test)
    d2 = np.count_nonzero(pred2 - Y_test)
    d3 = np.count_nonzero(pred3 - Y_test)
    paClassic_acc[s] = (1 - float(d1) / X_test.shape[0])
    paFirst_acc[s] = (1 - float(d2) / X_test.shape[0])
    paSecond_acc[s] = (1 - float(d3) / X_test.shape[0])
    
    #print('PA accuracy: {}'.format(1 - float(c) / X_test.shape[0]))
    
plt.figure(figsize=(15,10))
plt.plot(q, paClassic_acc, label="$PA - Classic$")
plt.plot(q, paFirst_acc, label="$PA - First Relaxation$")
plt.plot(q, paSecond_acc, label="$PA - Second Relaxation$")

plt.legend()
plt.xlabel("Number of random flipped labels")
plt.ylabel("Accuracy")
plt.show()

