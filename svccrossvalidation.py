import threading
from copy import deepcopy
import pandas as pd
from sklearn.utils import shuffle
import numpy as np


def svc_cross_validation(svm, x, y):
    confidence = 0

    for i in range(10):
        SVM_aux = deepcopy(svm)
        x, y = shuffle(x, y)  # Shuffle before splitting

        start = i * 5000
        end = start + 5000

        X_1 = x.iloc[start:end]
        X_2 = pd.concat([x.iloc[:start], x.iloc[end:]])

        y_1 = y.iloc[start:end]
        y_2 = pd.concat([y.iloc[:start], y.iloc[end:]])

        SVM_aux.fit(X_2, y_2)
        current_confidence = SVM_aux.score(X_1, y_1)
        print(current_confidence)
        confidence += current_confidence

    return confidence / 10



# -------------------------
# MULTITHREADING EXPERIMENT
# -------------------------

results = []

def cross_validate(svm, x_1, x_2, y_1, y_2, i):
    svm.fit(x_2, y_2)
    score = svm.score(x_1, y_1)
    print(f"Job {i} completed")
    results.append(score)

def svc_multithreaded_cross_validation(svm, x, y):
    threads = []

    for i in range(10):
        x, y = shuffle(x, y)

        start = i * 5000
        end = start + 5000

        x_1 = x.iloc[start:end]
        x_2 = pd.concat([x.iloc[:start], x.iloc[end:]])

        y_1 = y.iloc[start:end]
        y_2 = pd.concat([y.iloc[:start], y.iloc[end:]])

        thread = threading.Thread(target=cross_validate, args=(deepcopy(svm), x_1, x_2, y_1, y_2, i))
        threads.append(thread)
        print(f"Job {i} started")
        thread.start()

    for t in threads:
        t.join()

    return np.mean(results)

