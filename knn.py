import time
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter
from utilityfunctions import retrieve_mnist


def knn_experiment():
    X_train, X_test, y_train, y_test = retrieve_mnist()

    classifier = KNN(k=3)
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    print(f"Classifier accuracy for k = 3 {accuracy:.6f}")



class KNN:
    def __init__(self, k=1):
        self.k = k
        self.X_train = None
        self.y_train = None



    def fit(self, X, y):
        self.X_train = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        self.y_train = np.array(y)
        return self



    def predict(self, X):
        start_time = time.time()
        X_test = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        distances = cdist(X_test, self.X_train)

        k_indices = distances.argsort(axis=1)[:, :self.k]

        predictions = []
        for indices in k_indices:
            k_nearest_labels = self.y_train[indices]

            counter = Counter(k_nearest_labels)
            predictions.append(counter.most_common(1)[0][0])

        print("prediction took %s" % (time.time() - start_time))
        return np.array(predictions)



    def score(self, X, y):
        y_test = np.array(y)
        predictions = self.predict(X)

        return np.mean(predictions == y_test)

