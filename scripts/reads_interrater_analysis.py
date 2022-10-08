'''
Assess interrater agreement of CQ500 dataset

Fleiss' Kappa implementation from:
https://github.com/Shamya/FleissKappa/blob/master/fleiss.py
@author: skarumbaiah
'''

import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from tabulate import tabulate


#%%
def checkInput(rate, n):
    """
    Check correctness of the input matrix
    @param rate - ratings matrix
    @return n - number of raters
    @throws AssertionError
    """
    N = len(rate)
    k = len(rate[0])
    assert all(len(rate[i]) == k for i in range(k)), "Row length != #categories)"
    assert all(isinstance(rate[i][j], int) for i in range(N) for j in range(k)), "Element not integer"
    assert all(sum(row) == n for row in rate), "Sum of ratings != #raters)"


def fleissKappa(rate, n):
    """
    Computes the Kappa value
    @param rate - ratings matrix containing number of ratings for each subject per category
    [size - N X k where N = #subjects and k = #categories]
    @param n - number of raters
    @return fleiss' kappa
    """

    N = len(rate)
    k = len(rate[0])
    print("#raters = ", n, ", #subjects = ", N, ", #categories = ", k)
    checkInput(rate, n)

    # mean of the extent to which raters agree for the ith subject
    PA = sum([(sum([i ** 2 for i in row]) - n) / (n * (n - 1)) for row in rate]) / N
    print("PA = ", PA)

    # mean of squares of proportion of all assignments which were to jth category
    PE = sum([j ** 2 for j in [sum([rows[i] for rows in rate]) / (N * n) for i in range(k)]])
    print("PE =", PE)

    kappa = -float("inf")
    try:
        kappa = (PA - PE) / (1 - PE)
        kappa = float("{:.3f}".format(kappa))
    except ZeroDivisionError:
        print("Expected agreement = 1")

    print("Fleiss' Kappa =", kappa)

    return kappa


#%%

GT_basepath = '.../CQ500-dataset/'
GT = pd.read_csv(os.path.join(GT_basepath, 'reads.csv'))
raters = ['R1', 'R2', 'R3']
labels = ['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH']

n_agree_dict = {}
cohen_dict = {}
fleiss_dict = {}

for label in labels:
    readsABC = GT[[f'{raters[0]}:{label}', f'{raters[1]}:{label}', f'{raters[2]}:{label}']]
    readA = GT[f'{raters[0]}:{label}']
    readB = GT[f'{raters[1]}:{label}']
    readC = GT[f'{raters[2]}:{label}']

    n_agree_dict[f'{label}'] = [(GT[f'R1:{label}'] == GT[f'R2:{label}']).sum(),
                                (GT[f'R1:{label}'] == GT[f'R3:{label}']).sum(),
                                (GT[f'R2:{label}'] == GT[f'R3:{label}']).sum()]

    cohen_dict[f'{label}'] = cohen_kappa_score(readA, readB), cohen_kappa_score(readA, readC), cohen_kappa_score(readB, readC)

    readsABC_counted = pd.DataFrame(columns=['0', '1'])
    readsABC_counted['1'] = readA + readB + readC
    readsABC_counted['0'] = 3 - readsABC_counted['1']
    readsABC_counted = np.asarray(readsABC_counted).tolist()

    fleiss_dict[f'{label}'] = fleissKappa(readsABC_counted, 3)


orig_stdout = sys.stdout
f = open(os.path.join(GT_basepath, 'reads_interrater_analysis.txt'), 'w', encoding='utf-8')
sys.stdout = f

print('############### Agreement, n ###############')
n_agree_df = pd.DataFrame.from_dict(n_agree_dict)
n_agree_df = n_agree_df.transpose()
print(tabulate(n_agree_df, headers=['R1 vs R2', 'R1 vs R3', 'R2 vs R3'], floatfmt='.0f', tablefmt='fancy_grid'))

print('############### Agreement, % ###############')
percentagree_df = n_agree_df / len(GT) * 100
print(tabulate(percentagree_df, headers=['R1 vs R2', 'R1 vs R3', 'R2 vs R3'], floatfmt='.0f', tablefmt='fancy_grid'))

print('############### Cohen\'s Kappa ###############')
cohen_df = pd.DataFrame.from_dict(cohen_dict)
cohen_df = cohen_df.transpose()
print(tabulate(cohen_df, headers=['R1 vs R2', 'R1 vs R3', 'R2 vs R3'], floatfmt='.2f', tablefmt='fancy_grid'))

print('############### Fleiss\' Kappa ###############')
fleiss_df = pd.DataFrame(fleiss_dict, index=[0])
fleiss_df = fleiss_df.transpose()
print(tabulate(fleiss_df, headers=['', 'Fleiss\' Kappa'], floatfmt='.2f', tablefmt='fancy_grid'))

sys.stdout = orig_stdout
f.close()
