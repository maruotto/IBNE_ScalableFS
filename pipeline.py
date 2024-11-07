import pandas as pd
import numpy as np
from feature_selection import *
from opt import opt
from os.path import join
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from io import StringIO
# IMPORT MODELS
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import time
import sys
# IMPORT PLOTS
import seaborn as sns
from matplotlib import pyplot as plt

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
max_features = opt.maxFeats
perc = opt.maxFeatsPerc
np.random.seed(seed=seed)

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
print("IGNORING UNDEFINED METRIC WARNING")

def extract_f6(all_features):
    final_set = set()
    for feature_set in all_features:
        final_set.update(feature_set)
    final_set = list(final_set)
    for feat in final_set:
        if feat not in all_features[1] or feat not in all_features[2]:
            del final_set[final_set.index(feat)]
    return final_set


def get_features(features, all=True, quartile=75, column_name="Occurencies"):
    if all:
        return features.index

    quart = np.percentile(features[column_name].to_list(), q=quartile)
    a = features[features[column_name] >= quart]
    return a.index


def update_dictionary(di, lis):
    for element in lis:
        if element in di.keys():
            di[element] = di[element] + 1
        else:
            di[element] = 1


print("*" * 40)
print(" " * 12, "SUBJECTS")
print("*" * 40)
print("Total subjects: ", len(subjects), "with", len(na_sub), "na patients")
subjects=subjects.dropna()

if biosignal_name.startswith("ALL"):
    eeg_file = "eeg.xlsx"
    emg_file = "emg.xlsx"
    cop_file = "cop.xlsx"
    biosignal = read_signals(eeg_file, emg_file, cop_file)
else:
    biosignal = pd.read_excel(biosignal_file, index_col=0)

if biosignal_name.endswith("LIFE"):
    get_biosignal_with_lifestyle(biosignal, subjects, proneness, proneness_level)

print("*" * 40)
print(" " * 16, biosignal_name)
print("*" * 40)

print('correct imported samples:', len(biosignal.dropna()), '\nover a total of:', len(biosignal))
biosignal = biosignal.dropna()
if biosignal_name == "EEG":
    biosignal = biosignal[get_abs_rel_psd(biosignal)]

X, y = create_dataset(subjects, biosignal, proneness, bio_index, proneness_level)

print("*" * 40)
print(" " * 16, "DATASET")
print("*" * 40)

print("Total samples useful to purpose:", len(X))
print('Proneness index analyzed', proneness_level)
print(bio_index, 0, (y == 0).sum())
print(bio_index, 1, (y == 1).sum())
print("sum check: ", (y == 0).sum() + (y == 1).sum(), len(y))

if perc:
    max_features = round(max_features * X.shape[1] / 100)
print("Max features allowed: ", max_features, " over a total of ", X.shape[1])
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

X_train, X_test, y_train, y_test, scaler = test_split(X, y, seed, test_percentage, StandardScaler)


print("Total samples useful to train purpose:", len(X))
print('Proneness index analyzed', proneness_level)
print(bio_index, 0, (y_train == 0).sum())
print(bio_index, 1, (y_train == 1).sum())
print("sum check: ", (y_train == 0).sum() + (y_train == 1).sum(), len(y))

"----------------------------------------------------------------------"
"---------------------------HYPOTESIS TEST-----------------------------"
"----------------------------------------------------------------------"

models_seeded = [RandomForestClassifier, SVC]

all_scores = np.zeros((len(models_seeded), len(classification_scoring), 2))
print("*" * 40)
print(" " * 15, "TRAINING")
print("*" * 40)

for i in range(len(models_seeded)):
    # print(models_seeded[i].__name__)
    start_time = time.time()
    model_class = models_seeded[i]
    scores = zero_hp_seeded_models(model_class, X_train, y_train, stratified_kfold, seeds)
    print("Time for execution: ", round(time.time() - start_time))
    print("Scoring", ["Mean", "Std"])
    for j in range(len(scores.keys())):
        value = np.array(scores[list(scores.keys())[j]])
        print(list(scores.keys())[j], [round(value.mean(), 2), round(value.std(), 4)])

"----------------------------------------------------------------------"
"------------------------FEATURE SELECTION-----------------------------"
"----------------------------------------------------------------------"
# original_stdout = sys.stdout
# Redirect stdout to a StringIO object
# sys.stdout = StringIO()

final_weights = pd.DataFrame({"Feature": []})
sfs_final_selection = {}

for seed in seeds:  # TODO: replace seeds
    sys.stdout.flush()
    np.random.seed(seed=seed)
    sfs_selection, _, weights = pipeline_feature_selection_dataframes(subjects, biosignal,
                                                                      proneness, bio_index, proneness_level,
                                                                      test_percentage,
                                                                      stratified_kfold,
                                                                      seed, max_features)
    final_weights = final_weights.merge(weights, on="Feature", how="outer")
    final_weights.rename(columns={"Importance": "Importance_" + str(seed)}, inplace=True)
    final_weights.set_index("Feature", inplace=True)
    update_dictionary(sfs_final_selection, sfs_selection)

"----------------------------------------------------------------------"
"---------------------------FEATURE SAVING-----------------------------"
"----------------------------------------------------------------------"

final_weights = final_weights.merge(pd.DataFrame(final_weights.sum(axis=1), columns=["Weigths"]), on="Feature")
final_weights = final_weights.merge(pd.DataFrame(final_weights.count(axis=1), columns=["Repetition"]), on="Feature")
final_weights.to_excel(join(out_dir, "weights_" + biosignal_name.lower() + ".xlsx"))

data_list = [(key, value) for key, value in sfs_final_selection.items()]

# Create a DataFrame from the list of tuples
f2 = pd.DataFrame(data_list, columns=['Feature', 'Repetition'])
f2.to_excel(join(out_dir, "SFS_" + biosignal_name.lower() + ".xlsx"))

# FINAL FEATURE SET EXTRACTION
f1 = list(get_features(final_weights, all=False, column_name="Repetition"))
f2 = list(f2["Feature"])
f3 = list(get_features(final_weights, all=False, column_name="Weigths"))

f6 = extract_f6([f1, f2, f3])

with open(join(out_dir, biosignal_name.lower() + ".txt"), "w") as f:
    # Write F1
    f.write("F1\n")
    f.write(str(f1))
    f.write("\n")

    # Write F2
    f.write("F2\n")
    f.write(str(f2))
    f.write("\n")

    # Write F3
    f.write("F3\n")
    f.write(str(f3))
    f.write("\n")

    # Write F6
    f.write("F6\n")
    f.write(str(f6))
    f.write("\n")

"----------------------------------------------------------------------"
"-----------------------------FINAL TEST-------------------------------"
"----------------------------------------------------------------------"

# sys.stdout = original_stdout

X, y = create_dataset(subjects, biosignal[f6], proneness, bio_index, proneness_level)
X_train, X_test, y_train, y_test, scaler = test_split(X, y, seed, test_percentage, StandardScaler)

models_seeded = [RandomForestClassifier, SVC]

all_scores = np.zeros((len(models_seeded), len(classification_scoring), 2))
print("*" * 40)
print(" " * 16, "TESTING")
print("*" * 40)

for i in range(len(models_seeded)):
    # print(models_seeded[i].__name__)
    start_time = time.time()
    model_class = models_seeded[i]
    scores = zero_hp_seeded_models(model_class, X_train, y_train, stratified_kfold, seeds)
    print("Time for execution: ", round(time.time() - start_time))
    print("Scoring", ["Mean", "Std"])
    for j in range(len(scores.keys())):
        value = np.array(scores[list(scores.keys())[j]])
        print(list(scores.keys())[j], [round(value.mean(), 2), round(value.std(), 4)])
    model = models_seeded[i]()
    model.fit(X_train, y_train)
    if test_percentage != 0:
        print("test scores: ", get_model_scores(model, X_test, y_test, cv=None)[0])
