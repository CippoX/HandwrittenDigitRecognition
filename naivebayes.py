import scipy
import numpy as np
import sys
import math
import time
from utilityfunctions import retrieve_mnist, create_image_from_array


def naive_bayes_experiment():
    X_train, X_test, y_train, y_test = retrieve_mnist()
    classifier = NaiveBayes()
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test))



class NaiveBayes:
    def __init__(self, k=1):
        self.alpha_beta_matrix = []
        self.unique_labels = []



    def fit(self, X, y):
        start_time = time.time()
        print("fitting")

        X_array = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        y_array = np.array(y)
        self.unique_labels = np.unique(y_array)
        self.alpha_beta_matrix = []

        for label in self.unique_labels:
            label_mask = y_array == label
            filtered_data = X_array[label_mask]
            alpha_beta_matrix_aux = [filtered_data[:, col].tolist() for col in range(X_array.shape[1])]
            self.alpha_beta_matrix.append(alpha_beta_matrix_aux)

        for i, row in enumerate(self.alpha_beta_matrix):
            for j, col in enumerate(row):
                mean = np.mean(col)
                var = np.var(col) + sys.float_info.min
                k = (mean * (1 - mean)) / var

                a = k * mean
                b = k * (1 - mean)

                self.alpha_beta_matrix[i][j] = {"alpha": a, "beta": b}

        print("Fitting took %s" % (time.time() - start_time))

        for i in range(len(self.unique_labels)):
            create_image_from_array(np.array([d["alpha"] for d in self.alpha_beta_matrix[i]]), output_path=f"output_image{i}.png")



    def predict(self, X):
        start_time = time.time()
        X_array = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        predictions = []

        for i, row in enumerate(X_array):
            predictions.append(self.__score_row(row))

        print("Prediction took %s" % (time.time() - start_time))
        return np.array(predictions)



    def score(self, X, y):
        y_test = np.array(y)
        predictions = self.predict(X)

        return np.mean(predictions == y_test)



    def __score_row(self, X):
        likelihood = []

        for i in range(len(self.unique_labels)):
            aux = math.log(0.1)
            for j, n in enumerate(X):
                li = self.__likelihood(n, self.alpha_beta_matrix[i][j]["alpha"], self.alpha_beta_matrix[i][j]["beta"])
                if li != 0:
                    aux = aux + math.log(li)

            likelihood.append(aux)

        return max(range(len(likelihood)), key=likelihood.__getitem__)



    def __likelihood(self, x, a, b):
        if x <= 0 or x >= 1:
            return 0

        denominator = scipy.special.beta(a, b)
        if denominator == 0 or np.isnan(denominator):
            return 0

        try:
            numerator = x ** (a - 1) * (1 - x) ** (b - 1)
        except ZeroDivisionError:
            return 0

        likelihood = numerator / denominator
        return likelihood
