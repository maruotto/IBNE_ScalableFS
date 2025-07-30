import numpy as np

def extract_f6(all_features):
    final_set = set()
    for feature_set in all_features:
        final_set.update(feature_set)
    
    # Keep only features that are in both all_features[1] and all_features[2]
    filtered = [feat for feat in final_set if feat in all_features[1] and feat in all_features[2]]
    
    return filtered



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