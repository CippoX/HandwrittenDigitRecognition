from knn import knn_experiment
from naivebayes import naive_bayes_experiment
from random_forests import random_forest_experiment
from svm import svm_experiment

if __name__ == '__main__':
    svm_experiment()
    random_forest_experiment()
    naive_bayes_experiment()
    knn_experiment()