import pandas as pd
from sklearn.preprocessing import StandardScaler
import feature_selection
from feature_selection import *
from sklearn.model_selection import StratifiedKFold
from opt import opt
import numpy as np
import joblib
from sklearn.feature_selection import SelectFromModel
# IMPORT MODELS
from sklearn.ensemble import GradientBoostingClassifier as classifier
#from sklearn.ensemble import RandomForestClassifier as classifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
import time
import sys

biosignal_file = opt.input_file
subjects_file = opt.subjects
n_splits = opt.nfolds
seed = opt.seed
test_percentage = opt.test
biosignal_name = opt.biosignal
out_dir = opt.out
proneness = opt.proneness
bio_index = opt.bioindex
proneness_level = opt.pronidx
subjects = read_patients(subjects_file)
na_sub = get_na_subjects(subjects)

np.random.seed(seed=seed)

def pipeline_feature_selection_dataframes(subjects, biosignal,
                               proneness, bio_index, proneness_level, test_percentage,
                               cv, seed, max_features=None):
    print("*" * 40)
    print(" " * 12, "SEED", seed)
    print("*" * 40)
    X, y = create_dataset(subjects, biosignal, proneness, bio_index, proneness_level)

    print("*" * 40)
    print(" " * 12, "BASE PERFORMANCES")
    print("*" * 40)

    X_train, X_test, y_train, y_test, _ = test_split(X, y, seed, test_percentage, StandardScaler)

    estimator = classifier(random_state=seed)
    estimator.fit(X_train, y_train)
    scores = get_detailed_model_scores(estimator, X_test, y_test)
    print(scores)

    selector = SelectFromModel(estimator=estimator, prefit=True, max_features=max_features)
    f1 = biosignal.columns[selector.get_support()]
    print(f1)
    print("number of features: ", len(biosignal.columns[selector.get_support()]))

    print("*" * 40)
    print(" " * 14, "WEIGHTS")
    print("*" * 40)
    weights = get_importances(selector.estimator, selector.get_support(indices=True), biosignal.columns)
    print(weights)
    sys.stdout.flush()


    new_biosignal = biosignal[biosignal.columns[selector.get_support()]]
    estimator = classifier(random_state=seed)
    X, y = create_dataset(subjects, new_biosignal, proneness, bio_index, proneness_level)
    X_train, X_test, y_train, y_test, _ = test_split(X, y, seed, test_percentage, StandardScaler)
    estimator.fit(X_train, y_train)
    scores = get_detailed_model_scores(estimator, X_test, y_test)
    print(scores)

    print("*" * 40)
    print(" " * 3, "SFM+B-SFS PERFORMANCES")
    print("*" * 40)
    start_time = time.time()

    sfs_estimator = classifier(random_state=seed, verbose=0)
    sfs = SequentialFeatureSelector(sfs_estimator,
                                    n_features_to_select='auto',
                                    tol=-0.1, #negative because the direction is backward
                                    direction='backward',
                                    scoring='f1',
                                    cv=cv,
                                    n_jobs=None)

    selected_features, (X_train_new, X_test_new, y_train_new, y_test_new) = train_models(sfs, subjects.loc[subjects.index],
                                                                                         biosignal=new_biosignal.loc[biosignal.index],
                                                                                         proneness=proneness,
                                                                                         bio_index=bio_index,
                                                                                         proneness_level=proneness_level,
                                                                                         test_percentage=test_percentage,
                                                                                         # cv = 5 by default
                                                                                         seed=seed)

    print(selected_features.columns)

    estimator = classifier(random_state=seed)
    estimator.fit(X_train_new, y_train_new)
    scores = get_detailed_model_scores(estimator, X_test_new, y_test_new)
    print(scores)
    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)

    '''
    print("*" * 40)
    print(" " * 14, "WEIGHTS")
    print("*" * 40)
    print(get_importances(sfs.estimator_, selector.get_support(indices=True), biosignal.columns))
    '''

    return biosignal.columns[selector.get_support()], selected_features, weights




def pipeline_feature_selection(subjects, biosignal,
                               proneness, bio_index, proneness_level, test_percentage,
                               cv, seed, max_features=None):

    a, b, _ = pipeline_feature_selection_dataframes(subjects, biosignal,
                                          proneness, bio_index, proneness_level, test_percentage,
                                          cv, seed, max_features)

    return a,b


if __name__ == "__main__":
    print("*" * 40)
    print(" " * 12, "SUBJECTS")
    print("*" * 40)
    print("Total subjects: ", len(subjects), "with", len(na_sub), "na patients")

    biosignal = pd.read_excel(biosignal_file, index_col=0)
    print("*" * 40)
    print(" " * 16, biosignal_name)
    print("*" * 40)

    print('correct imported samples:', len(biosignal.dropna()), '\nover a total of:', len(biosignal))
    biosignal = biosignal.dropna()
    if biosignal_name == "EEG":
        biosignal = biosignal[get_abs_rel_psd(biosignal)]
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    pipeline_feature_selection(subjects, biosignal,
                               proneness, bio_index, proneness_level, test_percentage,
                               stratified_kfold, seed)

    sys.stdout.flush()

    print("*" * 40)
    print(" " * 6, "B-SFS PERFORMANCES")
    print("*" * 40)
    start_time = time.time()
    estimator = classifier(random_state=seed, verbose=0)
    sfs = SequentialFeatureSelector(estimator,
                                    n_features_to_select='auto',
                                    tol=-0.1,
                                    direction='backward',
                                    scoring=None,
                                    cv=stratified_kfold,
                                    n_jobs=None)

    selected_features, (X_train_new, X_test_new, y_train_new, y_test_new) = train_models(sfs, subjects,
                                                                                         biosignal=biosignal,
                                                                                         proneness=proneness,
                                                                                         bio_index=bio_index,
                                                                                         proneness_level=proneness_level,
                                                                                         test_percentage=test_percentage,
                                                                                         seed=seed)
    print(selected_features.columns)

    estimator = classifier(random_state=seed)
    estimator.fit(X_train_new, y_train_new)
    scores = get_detailed_model_scores(estimator, X_test_new, y_test_new)
    print(scores)

    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)
