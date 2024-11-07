import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import get_scorer, roc_curve
from .models import classification_scoring
from sklearn.model_selection import train_test_split
import sys
from sklearn.preprocessing import StandardScaler

def get_abs_rel_psd(eeg, relative=True, biosignal="EEG"):
    if biosignal != "EEG":
        return eeg.columns
    psd = "relative" if relative else "absolute"
    cols = []
    for col in eeg.columns:
        if psd in col:
            cols.append(col)
    return cols


def train_models(feature_selection_model, subjects, biosignal,
                 proneness="MSProne_IDX",
                 bio_index="BioVRSea_Effect_IDX",
                 proneness_level=1,
                 test_percentage=0.2,
                 seed=12347):

    X, y = create_dataset(subjects, biosignal, proneness, bio_index, proneness_level)
    X_train, X_test, y_train, y_test, _ = test_split(X, y, seed, test_percentage, StandardScaler)

    feature_selection_model.fit(X_train, y_train)

    selected_features = biosignal[biosignal.columns[feature_selection_model.get_support()]]
    X, y = create_dataset(subjects,
                          selected_features,
                          proneness, bio_index, proneness_level)
    X_train, X_test, y_train, y_test, _ = test_split(X, y, seed, test_percentage, StandardScaler)

    return selected_features, (X_train, X_test, y_train, y_test)

def read_signals(eeg_file, emg_file, cop_file, lifestyle=False):
    eeg = pd.read_excel(eeg_file, index_col=0).dropna()
    eeg = eeg[get_abs_rel_psd(eeg)]
    emg = pd.read_excel(emg_file, index_col=0).dropna()
    biosignal = eeg.merge(emg, how='inner', indicator=False, left_index=True,
                              right_index=True)
    cop = pd.read_excel(cop_file, index_col=0).dropna()
    biosignal = biosignal.merge(cop, how='inner', indicator=False, left_index=True,
                              right_index=True)
    return biosignal

def get_biosignal_with_lifestyle(biosignal, subjects, proneness, proneness_level):
        return biosignal.merge(get_demographic_attributes(subjects[subjects[proneness] == proneness_level]),
                                    how='inner',
                                    indicator=False, left_index=True,
                                    right_index=True)

def read_patients(subjects_file, sheet=1):
    """
    This function allows to have a preprocessed subject dataframe for machine learning purposes (drop na etc...)
    :param subjects_file: (string) name of the Excel file with patients and labels
    :param sheet: (any kind accepted by pandas sheet_name) number of the sheet in which the data is stored, 1 by default
    :return: pandas dataframe with updated Index. It is made of patient IDs.
    """
    subjects = pd.read_excel(subjects_file, sheet_name=sheet)
    # subjects = subjects.drop(["Timestamp"], axis=1)
    subjects = subjects.drop([column for column in subjects.columns if column.startswith("Unnamed: ")], axis=1, errors='ignore')
    subjects.set_index('ID', drop=True, inplace=True)
    subjects.fillna({"Gender":"NA"}, inplace=True)
    return subjects


def get_na_subjects(subjects):
    """
    This function returns a dataframe containing only subjects with one or more NA value(s)
    :param subjects: pandas dataframe with subjects
    :return: pandas dataframe with subjects containing one or more NA value(s)
    """
    sub = subjects.copy()
    sub.dropna(axis=0, how='any', inplace=True)  # without na
    # full join on indexes with a column indicating if the row belongs to both or only one of them
    na_subject = subjects.merge(sub, how='outer', indicator=True, left_index=True, right_index=True)
    # get only element that are in subjects and not in sub (meaning only the ones with na values)
    na_subject = na_subject[na_subject['_merge'] == 'left_only']
    return na_subject


def variance_thresholding(dataframe, threshold=0.0):
    """
    This function allows to do a variance thresholding on the features of dataframe with maximum threshold value
    :param dataframe: dataframe containing the samples on the rows and features on the columns
    :param threshold: maximum standard deviation threshold for dropping a feature
    :return: data transformed and a list of selected indices
    """
    selector = VarianceThreshold(threshold=threshold)
    # Fit the selector to your dataset and transform it
    data_transformed = selector.fit_transform(dataframe)
    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)
    return data_transformed, selected_indices


def create_dataset(subjects, biosignal, target, label, index):
    """
    Function used to create the dataset from subjects and biosignal
    :param subjects: pandas dataframe with subjects
    :param biosignal: pandas dataframe with biosignal's features
    :param target: subject column to be selected
    :param label: subject column that will be used as label in training
    :param index: value of target column to be selected
    :return: dataset in form of x and y as numpy array
    """
    dataset = biosignal.merge(subjects[subjects[target] == index][label], how='inner', indicator=False, left_index=True,
                              right_index=True)
    x = dataset.drop([label], axis=1).to_numpy()
    y = dataset[label].to_numpy()
    return x, y

def test_split(X, y, seed, test_percentage, scalerClass):
    X_test = np.array([]).reshape(1, -1)
    y_test = np.array([]).reshape(1, -1)
    if test_percentage == 0:
        X_train, y_train = X, y
        scaler = scalerClass(copy=True, with_mean=True, with_std=True)
        X_train = scaler.fit_transform(X_train)
        
    else:
        # Define the number of folds
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                                test_size=test_percentage, shuffle=True,
                                                                random_state=seed)
        scaler = scalerClass(copy=True, with_mean=True, with_std=True)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def get_demographic_attributes(subjects):
    subjects = subjects.copy()
    subjects["Gender"] = subjects["Gender"].replace({"Prefer not to specify": "NA"})
    subjects["Gender"] = subjects["Gender"].replace({"Male": 0, "Female": 1, "NA": 2})
    return subjects[["Gender"]] #subjects[["Gender", "Lifestyle_2_Idx"]] #subjects[["Gender", "Age", "Lifestyle_2_Idx"]]  #


def grid_search_the_model(clf, parameters, kf, x_train, y_train):
    """
    Perform grid search on clf model using parameters and kf cross validation
    :param clf: model to be trained
    :param parameters: parameters on which search
    :param kf: cross validation techique (scikit-learn)
    :param x_train: training samples
    :param y_train: training labels
    :return: best estimator
    """
    clf_feature = GridSearchCV(estimator=clf, param_grid=parameters, cv=kf, n_jobs=-1, scoring=None, verbose=3)
    clf_feature.fit(x_train, y_train)
    best_fs = clf_feature.best_estimator_
    return best_fs


"----------------------------------------------------------------------"
"------------------------SCORING FUNCTIONS-----------------------------"
"----------------------------------------------------------------------"


def get_test_scores_cv(model, x, y, cv, metric):
    """
    Evaluate metric on test set given the dataset x,y and the cross validation technique
    :param model: trained model
    :param x: samples
    :param y: labels
    :param cv: scikit-learn cross validation technique
    :param metric: score metric
    :return: folds' scores list
    """
    scores = []
    for train_index, test_index in cv.split(x, y):
        _, X_test = x[train_index], x[test_index]
        _, y_test = y[train_index], y[test_index]
        scores.append(get_test_scores(model, X_test, y_test, metric=metric))
    return scores


def get_test_scores(model, X_test, y_test, cv=None, metric='accuracy'):
    """

    :param model:
    :param X_test:
    :param y_test:
    :param cv:
    :param metric:
    :return:
    """
    scorer = get_scorer(metric)
    return [scorer(model, X_test, y_test)]


def get_model_scores(model, x, y, cv=None):
    # initialization of variables
    scores = []
    dev = []

    # definition of score function
    score_function = get_test_scores_cv  # by default, it uses the one with cross validation
    if cv is None:
        # if cross validation is not defined it switches to the one without cross validation
        score_function = get_test_scores

    for metric in classification_scoring:
        scr = np.array(list(score_function(model, x, y, cv, metric)))
        scores.append(scr.mean())
        dev.append(scr.std())
    return scores, dev


def get_detailed_model_scores(model, x, y, cv=None):
    scores = {}
    score_function = get_test_scores_cv  # by default, it uses the one with cross validation
    if cv is None:
        # if cross validation is not defined it switches to the one without cross validation
        score_function = get_test_scores

    for metric in classification_scoring:
        scr = np.array(list(score_function(model, x, y, cv, metric)))
        scores[metric] = scr
    return scores


"----------------------------------------------------------------------"
"----------------------HYPOTESIS FUNCTIONS-----------------------------"
"----------------------------------------------------------------------"


def zero_hp_seeded_models(model_class, x, y, cv, seeds):
    print(model_class.__name__, end="")
    all_scores = {}
    for e in classification_scoring:
        all_scores[e] = []
    for seed in seeds:
        print(".", end="")
        sys.stdout.flush()
        for train_index, test_index in cv.split(x, y):
            mdl = model_class(random_state=seed)
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            mdl.fit(X_train, y_train)
            scores = get_detailed_model_scores(mdl, X_test, y_test, cv=None)
            for key, value in scores.items():
                all_scores[key].append(value)  # it is a numpy array of one e
    print()
    return all_scores


def test_all(mdl, x, y, cv=None):
    all_scores = {}
    for e in classification_scoring:
        all_scores[e] = []
    scores = get_detailed_model_scores(mdl, x, y, cv)
    for key, value in scores.items():
        all_scores[key].append(value)  # it is a numpy array of one e
    print()
    return all_scores

def get_importances(model, support_indices, features):
    importance_df = pd.DataFrame(
        {'Feature': features[support_indices], 'Importance': model.feature_importances_[support_indices]})

    # Sort the DataFrame by feature importances in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    return importance_df

