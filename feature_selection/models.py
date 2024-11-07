from copy import deepcopy
import numpy as np

classification_scoring = ['accuracy',
                          'f1_weighted',
                          'f1',
                          'roc_auc',
                          'precision',
                          'recall']

seeds = [102772449, 21744986, 57857517, 62360057, 62450291, 116237632, 65948098, 33356060, 12040344, 60578026, 54547281,
         21329461, 47880249, 64116921, 56016230, 78385625, 92247720, 9003199, 69653188, 122180222, 29340303, 24999848,
         31737090, 100239444, 59938718, 122955617, 120009515, 93139258, 57288187, 19566776, 51681088, 87123957, 4012812,
         2290168, 4822929, 51976174, 87871136, 82816674, 70466145, 26615348, 33972790, 71810036, 41001483, 70823297,
         5361943, 33646744, 91585880, 48456857, 39370803, 81447646]


def models_training(models, x, y, cv):
    """
    Train models on x and y with supervised learning using cv cross validation
    :param models: array of models that will be trained
    :param x: samples' training set
    :param y: labels' training set
    :param cv: cross validation technique (scikit-learn one)
    :return: mean accuracies over folds and deviations
    """
    accuracies = []
    deviations = []
    for model in models:
        acc = []
        # m = grid_search_the_model(model, param, cv, x, y)
        for train_index, test_index in cv.split(x, y):
            mdl = deepcopy(model)
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            mdl.fit(X_train, y_train)
            accuracy = mdl.score(X_test, y_test)
            acc.append(accuracy)
        accuracies.append(np.array(acc).mean())
        deviations.append(np.array(acc).std())
        model.fit(x, y)  # train the model on the full dataset
    return accuracies, deviations






