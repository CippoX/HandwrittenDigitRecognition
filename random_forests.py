import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from utilityfunctions import retrieve_mnist, save_model


def random_forest_experiment():
    X_train, X_test, y_train, y_test = retrieve_mnist()
    rndfor = RandomForestClassifier(criterion='entropy')

    params = {
        'n_estimators': [750],
        'max_depth': [40]
    }

    gs = GridSearchCV(rndfor, params, n_jobs=-1, cv=10, scoring='accuracy', verbose=5)
    gs.fit(X_train, y_train)

    results = pd.DataFrame(gs.cv_results_)
    display_columns = ['param_n_estimators', 'param_max_depth', 'mean_test_score', 'rank_test_score']
    results = results[display_columns]
    results = results.sort_values(by='rank_test_score')
    print(results)

    best_n_estimators = gs.best_params_['n_estimators']
    print(f"Best value of n_estimators: {best_n_estimators}")

    best_max_depth = gs.best_params_['max_depth']
    print(f"Best value of max_depth: {best_max_depth}")

    rndfor = RandomForestClassifier(criterion='entropy', max_depth=best_max_depth, n_estimators=best_n_estimators)
    rndfor.fit(X_train, y_train)

    print(f"Trained model accuracy: {rndfor.score(X_test, y_test)}")
    save_model(rndfor, "random_forests", "random_forest.pky")
