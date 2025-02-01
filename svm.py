from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pandas as pd
from utilityfunctions import retrieve_mnist, save_model



def svm_experiment():
    X_train, X_test, y_train, y_test = retrieve_mnist()
    svm_linear_kernel(X_train, X_test, y_train, y_test)
    svm_polynomial_kernel(X_train, X_test, y_train, y_test)
    svm_rbf_kernel(X_train, X_test, y_train, y_test)
    svm_polynomial_kernel_test(X_train, X_test, y_train, y_test)



def svm_linear_kernel(X_train, X_test, y_train, y_test):
    svcl = svm.SVC(kernel='linear')

    params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }

    gs = GridSearchCV(svcl, params, n_jobs=-1, cv=10, scoring='accuracy', verbose=5)

    gs.fit(X_train, y_train)

    results = pd.DataFrame(gs.cv_results_)
    display_columns = ['param_C', 'mean_test_score', 'std_test_score', 'rank_test_score']
    results = results[display_columns]
    results = results.sort_values(by='rank_test_score')
    print(results)

    best_C = gs.best_params_['C']
    print(f"Best value of C: {best_C}")

    svcl = svm.SVC(kernel='linear', C=best_C)
    print(svcl)
    svcl.fit(X_train, y_train)
    print(f"Trained model accuracy: {svcl.score(X_test, y_test)}")

    save_model(svcl, "svm", "svc_linear.pkl")



def svm_polynomial_kernel(X_train, X_test, y_train, y_test):
    svcp = svm.SVC(kernel='poly', degree=2)

    params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'coef0': [0.01, 0.5, 1],
        'gamma': ["auto", "scale", 1]
    }

    gs = GridSearchCV(svcp, params, n_jobs=-1, cv=10, scoring='accuracy', verbose=5)

    gs.fit(X_train, y_train)

    results = pd.DataFrame(gs.cv_results_)
    display_columns = ['param_C', 'mean_test_score', 'std_test_score', 'rank_test_score']
    results = results[display_columns]
    results = results.sort_values(by='rank_test_score')
    print(results)

    best_C = gs.best_params_['C']
    best_coef0 = gs.best_params_['coef0']
    best_gamma = gs.best_params_['gamma']
    print(f"Best value of C: {best_C}")
    print(f"Best value of coef0: {best_coef0}")
    print(f"Best value of gamma: {best_gamma}")

    svcp = svm.SVC(kernel='poly', degree=2, C=best_C, coef0=best_coef0, gamma=best_gamma)
    print(svcp)
    svcp.fit(X_train, y_train)
    print(f"Trained model accuracy: {svcp.score(X_test, y_test)}")

    save_model(svcp, "svm", "svc_polynomial.pkl")



def svm_rbf_kernel(X_train, X_test, y_train, y_test):
    svcr = svm.SVC(kernel='rbf')

    params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'gamma': ["auto", "scale"]
    }

    gs = GridSearchCV(svcr, params, n_jobs=-1, cv=10, scoring='accuracy', verbose=5)

    gs.fit(X_train, y_train)

    results = pd.DataFrame(gs.cv_results_)
    display_columns = ['param_C', 'param_gamma', 'mean_test_score', 'rank_test_score']
    results = results[display_columns]
    results = results.sort_values(by='rank_test_score')
    print(results)

    best_C = gs.best_params_['C']
    best_gamma = gs.best_params_['gamma']
    print(f"Best value of C: {best_C}")
    print(f"Best value of gamma: {best_gamma}")

    svcr = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)
    print(svcr)
    svcr.fit(X_train, y_train)
    print(f"Trained model accuracy: {svcr.score(X_test, y_test)}")

    save_model(svcr, "svm", "svc_rbf.pkl")



def svm_polynomial_kernel_test(X_train, X_test, y_train, y_test):
    svcp = svm.SVC(kernel='poly', degree=2, C=10, coef0=0.01, gamma="scale")
    print(svcp)
    svcp.fit(X_train, y_train)
    print(f"Trained model accuracy: {svcp.score(X_test, y_test)}")
    save_model(svcp, "svm", "svc_polynomial1.pkl")
    print("\n")

    svcp = svm.SVC(kernel='poly', degree=2, C=1000, coef0=0.01, gamma="auto")
    print(svcp)
    svcp.fit(X_train, y_train)
    print(f"Trained model accuracy: {svcp.score(X_test, y_test)}")
    save_model(svcp, "svm", "svc_polynomial2.pkl")
    print("\n")

    svcp = svm.SVC(kernel='poly', degree=2, C=0.001, coef0=0.5, gamma=1)
    print(svcp)
    svcp.fit(X_train, y_train)
    print(f"Trained model accuracy: {svcp.score(X_test, y_test)}")
    save_model(svcp, "svm", "svc_polynomial3.pkl")
    print("\n")

    svcp = svm.SVC(kernel='poly', degree=2, C=0.001, coef0=1, gamma=1)
    print(svcp)
    svcp.fit(X_train, y_train)
    print(f"Trained model accuracy: {svcp.score(X_test, y_test)}")
    save_model(svcp, "svm", "svc_polynomial4.pkl")
    print("\n")