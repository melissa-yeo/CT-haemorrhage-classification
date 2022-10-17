import sys
import os
import numpy as np
import pandas as pd
from numpy import interp
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
import scipy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from itertools import cycle
from tabulate import tabulate

from metrics_compare_auc_delong import *

label_cols = ['EDH', 'ICH', 'IVH', 'SAH', 'SDH', 'Intracranial haemorrhage']

# # Set font sizes for large graph (for results images in research poster)
# S_SIZE = 18
# SM_SIZE = 24
# M_SIZE = 30
# L_SIZE = 40
# plt.rcParams['font.family'] = 'sans-serif'  # set to always use sans-serif family fonts
# plt.rcParams['font.sans-serif'] = 'Lucida Grande'   # set the default sans-setif to Lucida Grande
# plt.rc('font', family='arial', size=S_SIZE)  # controls default text sizes
# plt.rc('axes', titlesize=M_SIZE)  # fontsize of the axes title
# plt.rc('axes', labelsize=M_SIZE)  # fontsize of the x and y labels
# plt.rc('xtick', labelsize=S_SIZE)  # fontsize of the tick labels
# plt.rc('ytick', labelsize=S_SIZE)  # fontsize of the tick labels
# plt.rc('legend', fontsize=SM_SIZE)  # legend fontsize
# plt.rc('figure', titlesize=L_SIZE)  # fontsize of the figure title


def clopper_pearson(k,n,alpha=0.05):
    '''
    Calculate confidence intervals basecd on exact Clopper-Pearson method based on β distribution.
    :param k: stat to be evaluated
    :param n: n observations
    :param alpha: significance level (alpha=0.05 for 95% CI)
    :return:
    '''
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi

def make_confusion_matrix(y_true, y_pred):
    return multilabel_confusion_matrix(y_true, y_pred)

def calc_acc_sens_spec_PPV_NPV_precision_recall(mcm, metrics='all'):
    """

    :param mcm: multilabel confusion matrix
    :param metrics: list of strings specifying which metric to calculate. default='all' will output all metrics.
    :return: dict containing metrics for each class
    """
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]

    all_metrics = dict()
    conf_intervals = dict()

    all_metrics['acc'] = (tp + tn) / (tp + tn + fp + fn)
    all_metrics['sens'] = tp / (tp + fn)
    all_metrics['spec'] = tn / (tn + fp)
    all_metrics['PPV'] = tp / (tp + fp)
    all_metrics['NPV'] = tn / (tn + fn)
    all_metrics['precision'] = tp / (tp + fp)
    all_metrics['recall'] = all_metrics['sens']

    conf_intervals['acc'] = np.array(clopper_pearson((tp + tn),(tp + tn + fp + fn))).T.tolist()
    conf_intervals['sens'] = np.array(clopper_pearson(tp, (tp + fn))).T.tolist()
    conf_intervals['spec'] = np.array(clopper_pearson(tn, (tn + fp))).T.tolist()
    conf_intervals['PPV'] = np.array(clopper_pearson(tp, (tp + fp))).T.tolist()
    conf_intervals['NPV'] = np.array(clopper_pearson(tn, (tn + fn))).T.tolist()
    conf_intervals['precision'] = np.array(clopper_pearson(tp, (tp + fp))).T.tolist()
    conf_intervals['recall'] = conf_intervals['sens']

    if metrics == 'all':
        return all_metrics, conf_intervals
    else:
        output = {}
        assert isinstance(metrics, list), f'{metrics} must be a list of strings!'
        for metric in metrics:
            output[metric] = all_metrics[metric]
        return output, _

def calc_avg_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

def calc_CI_by_bootstrapping(y_true, y_score, metric='ROC'):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_score), len(y_score))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        if metric == 'ROC':
            score = roc_auc_score(y_true[indices], y_score[indices])
        if metric == 'PR':
            score = average_precision_score(y_true[indices], y_score[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} area: {:0.3f}".format(i + 1, score))

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # Change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval.
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper))

    return confidence_lower, confidence_upper


def plot_roc(y_score, y_true, filename=None, dirpath=None):
    ##################################################
    #              Calculate ROC curves              #
    ##################################################
    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc, thresholds, lower_CI, upper_CI = dict(), dict(), dict(), dict(), dict(), dict()
    n_classes = y_true.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        lower_CI[i], upper_CI[i] = calc_CI_by_bootstrapping(y_true[:, i], y_score[:, i], metric='ROC')

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    for i in range(n_classes):
        df = pd.DataFrame(columns=['TPR', 'FPR', '1-FPR', 'thresholds'])
        df['TPR'] = tpr[i]
        df['FPR'] = fpr[i]
        df['1-FPR'] = 1 - fpr[i]
        df['thresholds'] = thresholds[i]
        df.to_csv(os.path.join(dirpath, f'{filename}_ROCvals_{label_cols[i]}.csv'), index=False)

    ##################################################
    #               Plot average ROCs                #
    ##################################################

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    lw = 2
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc="lower right")
    if filename:
        plt.savefig(os.path.join(dirpath, f'{filename}_avg_ROC.png'), bbox_inches='tight')
    plt.show()

    ##################################################
    #           Plot ROC for each subtype            #
    ##################################################

    # Plot all ROC curves
    lw = 2
    plt.figure()

    colors = cycle(['black', 'teal', 'cornflowerblue', 'tomato', 'sandybrown', 'gray'])
    # colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'red', 'teal'])
    for i, color in zip(range(n_classes - 1), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'{label_cols[i]} (AUC: {roc_auc[i]:.3f} ({lower_CI[i]:.3f} - {upper_CI[i]:.3f}))')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    # plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    if filename:
        plt.savefig(os.path.join(dirpath, f'{filename}_ROC.png'), bbox_inches='tight')
    plt.show()

    ##################################################
    #           Plot ROC for single subtype          #
    ##################################################

    # Plot single ROC curve of (specify i to change which class is plotted)
    plt.figure()
    lw = 3
    color = 'darkorange'
    i = 5
    # plt.plot(fpr[i], tpr[i], color=color, lw=lw)
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label=f'{label_cols[i]} (AUC: {roc_auc[i]:.3f} ({lower_CI[i]:.3f}-{upper_CI[i]:.3f}))')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc="lower right")
    if filename:
        plt.savefig(os.path.join(dirpath, f'{filename}_ROCsng.png'), bbox_inches='tight')
    plt.show()

    return


def plot_prcurve_with_isoF1(y_score, y_true, filename=None, dirpath=None):
    # For each class
    precision, recall, average_precision = dict(), dict(), dict()
    n_classes = y_true.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.4}'
          .format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.4f}'
            .format(average_precision["micro"]))
    if filename:
        plt.savefig(os.path.join(dirpath, f'{filename}_avg_PR_curve.png'), bbox_inches='tight')
    plt.show()

    # Plot PR curve for each class and iso-f1 curves
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'red', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.4f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.4f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    if filename:
        plt.savefig(os.path.join(dirpath, f'{filename}_indiv_PR_curves.png'), bbox_inches='tight')
    plt.show()
    return

def plot_prcurve_without_isoF1(y_score, y_true, filename=None, dirpath=None):
    ##################################################
    #             Plot average PR curve              #
    ##################################################
    precision, recall, average_precision, lower_CI, upper_CI = dict(), dict(), dict(), dict(), dict()
    n_classes = y_true.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])
        lower_CI[i], upper_CI[i] = calc_CI_by_bootstrapping(y_true[:, i], y_score[:, i], metric='PR')

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.4f}'
          .format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.4f}'
            .format(average_precision["micro"]))
    if filename:
        plt.savefig(os.path.join(dirpath, f'{filename}_avg_PR_curve.png'), bbox_inches='tight')
    plt.show()


    ##################################################
    # Plot PR curve for each class w/o iso-f1 curves #
    ##################################################
    # setup plot details
    colors = cycle(['black', 'teal', 'cornflowerblue', 'tomato', 'sandybrown', 'gray'])

    fig = plt.figure()
    # fig = plt.figure(figsize=(6, 7), dpi=300)
    gs = GridSpec(1, 1, width_ratios=[7], height_ratios=[6])
    ax1 = fig.add_subplot(gs[0])

    lines = []
    labels = []

    for i, color in zip(range(n_classes - 1), colors):
        l, = ax1.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append(f'{label_cols[i]} (AUC: {average_precision[i]:.3f} ({lower_CI[i]:.3f} - {upper_CI[i]:.3f}))')
        # labels.append('{0} (AUC = {1:0.3f})'
        #               ''.format(label_cols[i], average_precision[i]))
    # ax1.grid(which='both', axis='both', linestyle='--')
    # ax1.xaxis.set_major_locator(MultipleLocator(0.2))
    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    # ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    # ax1.set_title('Precision-Recall curves')
    ax1.legend(lines, labels, loc='lower left')

    if filename:
        plt.savefig(os.path.join(dirpath, f'{filename}_PR.png'), bbox_inches='tight')
    plt.show()


    #################################################
    #    Plot PR curve for all haemorrhage class    #
    #################################################
    plt.figure()
    lw = 3
    color = 'darkorange'
    i = 5
    # plt.plot(fpr[i], tpr[i], color=color, lw=lw)
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label=f'{label_cols[i]} (AUC: {average_precision[i]:.3f} ({lower_CI[i]:.3f}-{upper_CI[i]:.3f}))')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.00])
    plt.ylim([0.0, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    if filename:
        plt.savefig(os.path.join(dirpath, f'{filename}_PRsng.png'), bbox_inches='tight')
    plt.show()


    return

def create_report(y_true, y_score, y_pred, filename, dirpath):
    filename_shortened = str(filename.split('_')[0]) + '__' + str(filename.split('_')[2])

    orig_stdout = sys.stdout
    # f = open(f'{filename}.txt', 'w', encoding='utf-8')
    f = open(os.path.join(dirpath, f'{filename}.txt'), 'w', encoding='utf-8')
    sys.stdout = f

    cm = make_confusion_matrix(y_true, y_pred)
    metric_dict, CI_dict = calc_acc_sens_spec_PPV_NPV_precision_recall(cm)
    metric_df = pd.DataFrame.from_dict(metric_dict)
    metric_df = metric_df.transpose()
    print(tabulate(metric_df, headers=label_cols, floatfmt='.4f', tablefmt='fancy_grid'))

    print('95% confidence intervals shown below:')

    CI_df = pd.DataFrame.from_dict(CI_dict)
    CI_df = CI_df.transpose()
    print(tabulate(CI_df, headers=label_cols, floatfmt='.4f', tablefmt='fancy_grid'))

    # # Print same table but swapped rows and columns
    # cm = make_confusion_matrix(y_true2, y_pred)
    # metric_dict = calc_acc_sens_spec_PPV_NPV_precision_recall(cm)
    # metric_df = pd.DataFrame.from_dict(metric_dict)
    # metric_df.insert(0, column='Subtype', value=labels)
    # print(tabulate(metric_df, headers=['accuracy', 'sensitivity', 'specificity', 'PPV', 'NPV', 'precision', 'recall'],
    #                floatfmt='.4f', tablefmt='fancy_grid'))
    print('\n' + '*' * 80)
    print('*' * 80)
    print(f'The micro-averaged F1-score is {calc_avg_f1_score(y_true, y_pred)}.')

    print('\n' + '*' * 80)
    print('*' * 80)
    print('Confusion matrices shown below:')
    for i, matrix in enumerate(cm):
        print(label_cols[i])
        matrixdf = pd.DataFrame(data=matrix, columns=['Predicted neg', 'Predicted pos'])
        matrixdf.insert(0, '', ['Actual neg', 'Actual pos'])
        print(tabulate(matrixdf, headers=['Predicted neg', 'Predicted pos'], floatfmt='.1f', tablefmt='fancy_grid', showindex='never'))

        # print(tabulate(matrix, headers=['Labelled neg', 'Labelled pos'], floatfmt='.1f', tablefmt='fancy_grid'))

    sys.stdout = orig_stdout
    f.close()

    plot_roc(y_score, y_true, filename_shortened, dirpath)
    # plot_roc_single(y_score, y_true, filename_shortened, dirpath)
    # plot_prcurve_with_isoF1(y_score, y_true, filename_shortened, dirpath)
    plot_prcurve_without_isoF1(y_score, y_true, filename_shortened, dirpath)
    return
