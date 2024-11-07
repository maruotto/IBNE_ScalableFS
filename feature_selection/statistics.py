import scipy.stats as stats
from sklearn.model_selection import KFold
import numpy as np
import math
from .general import get_test_scores_cv


def chi_square_test(observed, expected):
    """
    Function that uses the scikit-learn built-in chi method
    :param observed: observed values
    :param expected: expected values
    :return: p_value of the two distributions
    """
    chi_statistic, p_value = stats.chisquare(observed, expected)
    return p_value


def t_test(group1, group2):
    """
    Function that uses the scikit-learn built-in t test method
    :param group1: observed value
    :param group2: expected value
    :return: p_value of the two distributions
    """
    t_statistic, p_value = stats.ttest_ind(group1, group2)
    return p_value


def hp_test(mean1, std1, mean2, std2, n):
    """
    Hypotesis test given two distributions and the same number of samples
    ref. https://www.medcalc.org/calc/comparison_of_means.php
    :param mean1: mean of the first distribution
    :param std1: standard deviation of the first distribution
    :param mean2: mean of the second distribution
    :param std2: standard deviation of the second distribution
    :param n: number of samples over which calculate the p_value
    :return: two tailed p_value
    """
    num = (n - 1) * ((std1 ** 2) + (std2 ** 2))
    pooled_std = math.sqrt((num / (n + n - 2)))
    standard_error = pooled_std * math.sqrt((2 / n))
    t_statistic = (mean1 - mean2) / standard_error
    # print(t_statistic)
    p_value = stats.t.sf(abs(t_statistic), df=n - 1) * 2
    return p_value


def p_value_tables(all_scores, n_models, number_of_samples ):
    """
    Returns a numpy array with cross p-values between models
    :param all_scores: dictionary with score format (es. 'accuracy': [acc1, acc2, acc3])
    :param n_models: number of models in all_scores
    :param number_of_samples: number of samples evaluated
    :return:
    """
    table = np.zeros((n_models, n_models))

    for i in range(n_models):
        mean1 = all_scores[i, 2, 0]
        std1 = all_scores[i, 2, 1]
        for j in range(n_models):
            mean2 = all_scores[j, 2, 0]
            std2 = all_scores[j, 2, 1]
            # print(mean1, std1, mean2, std2)
            table[i, j] = hp_test(mean1, std1, mean2, std2, number_of_samples)
            # print(table[i, j])
    return table
