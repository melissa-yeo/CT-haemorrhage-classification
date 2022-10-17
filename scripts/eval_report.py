"""
What this does:
1) From confidence intervals for each label for each slice, get a binary output for each label (present or absent)
2) Combine all slices of a volume, if any haemorrhage subtype is present then it is present for the volume

Match rows in qure.csv file to the CT image name (add a CT image name column) - use metadata file
Create [n_samples, n_classes] y_pred labels based on a selected threshold
Analyse compared to ground truth of slices
Analyse compared to ground truth of volume

"""

#%%
from metrics import *
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd

path_data = 'data'
path_preds = 'preds'
path_eval = 'eval'
testmetadata_name = 'CQ500_test_metadata.csv'
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

# following are the config options
model_name = 'resnext101v03_cat2'
# CNN only without concat/windowing = 'resnext101v03_noRNN_nowindow'
# CNN only with concat images = 'resnext101v03_noRNN_cat'
# CNN only with windowed images = 'resnext101v03_noRNN'
# CNN only with windowed + concat images = 'resnext101v03_noRNN+cat'
# CNN-RNN without concat/windowing = 'resnext101v03_nowindow'
# CNN-RNN with concat images = 'resnext101v03_cat2'
# CNN-RNN with windowed images = 'resnext101v03_2'
# CNN-RNN with windowed + concat images = 'resnext101v03+cat2'

preds_name = f'CQ500submission_{model_name}.csv'

# thresholds based on validation set
threshold = [0.017, 0.061, 0.033, 0.028, 0.035, 0.110]    # high sensitivity point
# threshold = [0.017, 0.24, 0.3, 0.07, 0.16, 0.45]    # balanced point (based on Youden's index)



%%

#############################################################################
# Get the ground truth slice labels for each CT slice ID.
# Columns: 'CT-idx_sliceidx', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'
#############################################################################

tstmdf = pd.read_csv(os.path.join(path_data, testmetadata_name))
rawgtdf = pd.read_csv(os.path.join(path_data, 'CQ500_slice_groundtruth.csv'))

# using metadata, identify CT-id_sliceidx from series, study, SOP UIDs in rawgt file
rawgtdf['SliceID'] = rawgtdf[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']].apply(lambda x: '{}__{}__{}'.format(*x.tolist()), 1)
tstmdf['SliceID'] = tstmdf[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']].apply(lambda x: '{}__{}__{}'.format(*x.tolist()), 1)
rawgtdf = rawgtdf[['SliceID', 'labelName']]

# for each row, create 1 column for each subclass with 0/1 labels
for column in label_cols:
    if column == 'any':
        # if any label is present, there is some haemorrhage
        rawgtdf[column] = 1.0
        continue
    rawgtdf[column] = rawgtdf['labelName'].apply(lambda x: int(column == x.lower()))
rawgtdf = rawgtdf.sort_values(by='SliceID')

# merge dfs, only keys from left df are kept; missing data from right df is replaced by NaN
gtdf = tstmdf.merge(rawgtdf, on='SliceID', how='left')
keepcols = ['CT-idx_sliceidx', 'SliceID', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
gtdf = gtdf[keepcols]

# deal with duplicates (atm, each row only shows one label present so some images have multiple rows)
grouped_gtdf = gtdf.groupby(by='CT-idx_sliceidx')
gtdf = grouped_gtdf.agg('max')
gtdf = gtdf.drop('SliceID', axis=1).reset_index()
gtdf.to_csv('CQ500_slice_groundtruth_proc.csv')


#############################################################################
# Get the ground truth subject labels for each CT-ID.
# Columns: 'CT-idx', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'
#############################################################################

# GT labels from slice by slice labelling
y_true = pd.read_csv(os.path.join(path_data, 'CQ500_slice_groundtruth_proc.csv'))
y_true['CT-idx'] = y_true['CT-idx_sliceidx'].apply(lambda x: x.split('_')[0])
y_true = y_true.sort_values(by='CT-idx').fillna(0)
grouped_y_true = y_true.groupby(by='CT-idx')
y_true = grouped_y_true.agg('max')
y_true = y_true.drop(labels=['CT-idx_sliceidx','Unnamed: 0'], axis=1)
y_true.to_csv('CQ500_sub_GT_based_on_slices.csv')

# GT labels from original three-radiologist subject labelled dataset
y_true2 = pd.read_csv(os.path.join(path_data, 'CQ500_sub_groundtruth_proc.csv'))
y_true2['CT-idx'] = y_true2['name'].apply(lambda x: x.split('-')[1:])
y_true2['CT-idx'] = y_true2['CT-idx'].apply(lambda x: '-'.join(x))
y_true2 = y_true2.sort_values(by='CT-idx')
y_true2 = y_true2.rename(columns={'ICH': 'any', 'IPH': 'intraparenchymal', 'IVH': 'intraventricular',
                                  'SDH': 'subdural', 'EDH': 'epidural', 'SAH': 'subarachnoid'})
keepcols = ['CT-idx', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
y_true2 = y_true2[keepcols]
y_true2.to_csv('CQ500_sub_GT_based_on_subject.csv')


#%%
#############################################################################
# Binarise sigmoid output for each class, based on threshold
#############################################################################

y_scores_raw = pd.read_csv(os.path.join(path_preds, preds_name))

# # plot dist of outputs to see what it looks like
# plt.hist(y_scores_raw['Label'], bins=50, histtype='step')
# plt.show()

# create y_score dataframe with unique IDs and respective labels
y_scores = pd.DataFrame(columns=['CT-idx_sliceidx', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'])
y_scores['CT-idx_sliceidx'] = y_scores_raw['ID'][::6].apply(lambda x: x.split('_')[:2])
y_scores['CT-idx_sliceidx'] = y_scores['CT-idx_sliceidx'].apply(lambda x: '_'.join(x))
for idx, col in enumerate(label_cols):
    y_scores[col] = y_scores_raw['Label'][idx::6].values
y_scores.to_csv(os.path.join(path_preds, f'CQ500_scores_{model_name}.csv'))

# for any score that is >threshold, label it as 1 (label present)
y_preds = y_scores
for i, col in enumerate(label_cols):
    y_preds[col] = y_preds[col].apply(lambda x: 1 if x > threshold[i] else 0)
y_preds.to_csv(os.path.join(path_preds, f'CQ500_preds_{model_name}__threshold={threshold}.csv'))





#%%
#############################################################################
# Per slice analysis
# NOTE: not all ground truth labels for each slice were available
#############################################################################

# y_true = pd.read_csv(os.path.join(path_data, 'CQ500_slice_groundtruth_proc.csv'))
# y_true = y_true.sort_values(by='CT-idx_sliceidx').fillna(0).drop(labels=['CT-idx_sliceidx','Unnamed: 0'], axis=1)
# y_true = y_true.to_numpy().astype(np.float)
# print(y_true.shape)
#
# y_pred = pd.read_csv(os.path.join(path_preds, f'CQ500_preds_{model_name}__threshold={threshold}.csv'))
# y_pred = y_pred.sort_values(by='CT-idx_sliceidx').drop(labels=['CT-idx_sliceidx','Unnamed: 0'], axis=1)
# y_pred = y_pred.to_numpy().astype(np.float)
# print(y_pred.shape)
#
# y_score = pd.read_csv(os.path.join(path_preds, f'CQ500_scores_{model_name}.csv'))
# y_score = y_score.sort_values(by='CT-idx_sliceidx').drop(labels=['CT-idx_sliceidx','Unnamed: 0'], axis=1)
# y_score = y_score.to_numpy().astype(np.float)
# print(y_score.shape)
#
# create_report(y_true, y_score, y_pred, filename=f'evalreport__slc_threshold={threshold}', dirpath=path_eval)





#%%
#############################################################################
# Per volume analysis
#############################################################################

# The model's predictions
# if any slice has a haemorrhage, classify the whole volume as having that subtype label
y_pred = pd.read_csv(os.path.join(path_preds, f'CQ500_preds_{model_name}__threshold={threshold}.csv'))
y_pred['CT-idx'] = y_pred['CT-idx_sliceidx'].apply(lambda x: x.split('_')[0])
grouped_y_pred = y_pred.groupby(by=['CT-idx'], as_index=False)
y_pred = grouped_y_pred.agg('max')
y_pred = y_pred.drop(labels=['CT-idx_sliceidx', 'Unnamed: 0'], axis=1)

# The model's confidence indices
# take the highest confidence indices of any slice in the volume
y_score = pd.read_csv(os.path.join(path_preds, f'CQ500_scores_{model_name}.csv'))
y_score['CT-idx'] = y_score['CT-idx_sliceidx'].apply(lambda x: x.split('_')[0])
y_score = y_score.groupby(by=['CT-idx'], as_index=False).agg('max')
y_score = y_score.drop(labels=['CT-idx_sliceidx', 'Unnamed: 0'], axis=1)

y_true1 = pd.read_csv(os.path.join(path_data, 'CQ500_sub_GT_based_on_slices.csv'))
y_true1 = y_true1.sort_values(by='CT-idx')
y_true2 = pd.read_csv(os.path.join(path_data, 'CQ500_sub_GT_based_on_subject.csv'))
y_true2 = y_true2.sort_values(by='CT-idx')

print(y_pred.shape, y_score.shape, y_true1.shape, y_true2.shape)

# y_true2 is not the same length as y_pred, but we want it to have the same indices as y_pred
y_true2 = y_true2.merge(y_pred['CT-idx'].to_frame(), on='CT-idx', how='right')
print(y_pred.shape, y_true1.shape, y_true2.shape)

# convert to np arrays
y_pred = y_pred.sort_values(by='CT-idx').drop(labels='CT-idx', axis=1)
y_pred = y_pred.to_numpy().astype(np.float)
y_score = y_score.sort_values(by='CT-idx').drop(labels='CT-idx', axis=1)
y_score = y_score.to_numpy().astype(np.float)
y_true1 = y_true1.sort_values(by='CT-idx').drop(labels='CT-idx', axis=1)
y_true1 = y_true1.to_numpy().astype(np.float)
y_true2 = y_true2.sort_values(by='CT-idx').drop(labels=['CT-idx', 'Unnamed: 0'], axis=1)
y_true2 = y_true2.to_numpy().astype(np.float)
print(y_pred.shape, y_true1.shape, y_true2.shape)

# analyse metrics using ground truth labels from slice labels
# create_report(y_true1, y_score, y_pred, filename=f'evalreport__sub_slicelab_threshold={threshold}', dirpath=path_eval)

# finally: analyse metrics using ground truth labels from subject labels
create_report(y_true1, y_score, y_pred, filename=f'evalreport__sub_threshold={threshold}', dirpath=path_eval)
# create_report(y_true2, y_score, y_pred, filename=f'evalreport__sub_threshold={threshold}', dirpath=path_eval)


#%%
#############################################################################
# Helper functions
#############################################################################

def get_preds(model_name, threshold):
    # Get the model's predictions from the CSV file
    # if any slice has a haemorrhage, classify the whole volume as having that subtype label
    y_pred = pd.read_csv(os.path.join(path_preds, f'CQ500_preds_{model_name}__threshold={threshold}.csv'))
    y_pred['CT-idx'] = y_pred['CT-idx_sliceidx'].apply(lambda x: x.split('_')[0])
    grouped_y_pred = y_pred.groupby(by=['CT-idx'], as_index=False)
    y_pred = grouped_y_pred.agg('max')
    y_pred = y_pred.drop(labels=['CT-idx_sliceidx', 'Unnamed: 0'], axis=1)
    y_pred = y_pred.sort_values(by='CT-idx').drop(labels='CT-idx', axis=1)
    y_pred = y_pred.to_numpy().astype(np.float)
    return y_pred

def get_scores(model_name):
    # The model's confidence indices - take the highest confidence indices of any slice in the volume
    y_score = pd.read_csv(os.path.join(path_preds, f'CQ500_scores_{model_name}.csv'))
    y_score['CT-idx'] = y_score['CT-idx_sliceidx'].apply(lambda x: x.split('_')[0])
    y_score = y_score.groupby(by=['CT-idx'], as_index=False).agg('max')
    y_score = y_score.drop(labels=['CT-idx_sliceidx', 'Unnamed: 0'], axis=1)
    y_score = y_score.sort_values(by='CT-idx').drop(labels='CT-idx', axis=1)
    y_score = y_score.to_numpy().astype(np.float)
    return y_score

def get_groundtruths():
    # Get ground truths. shape(484, 6) where 6 = number of classes
    # ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
    y_true1 = pd.read_csv(os.path.join(path_data, 'CQ500_sub_GT_based_on_slices.csv'))
    y_true1 = y_true1.sort_values(by='CT-idx')
    y_true1 = y_true1.sort_values(by='CT-idx').drop(labels='CT-idx', axis=1)
    y_true1 = y_true1.to_numpy().astype(np.float)
    return y_true1




#%%
#############################################################################
# Delong test to compare AUCs between 2 models
#############################################################################

# Get y_scores for each model
model_a = 'resnext101v03_2'
model_b = 'resnext101v03+cat2'
# CNN only without concat/windowing = 'resnext101v03_noRNN_nowindow'
# CNN only with concat images = 'resnext101v03_noRNN_cat'
# CNN only with windowed images = 'resnext101v03_noRNN'
# CNN only with windowed + concat images = 'resnext101v03_noRNN+cat'
# CNN-RNN without concat/windowing = 'resnext101v03_nowindow'
# CNN-RNN with concat images = 'resnext101v03_cat2'
# CNN-RNN with windowed images = 'resnext101v03_2'
# CNN-RNN with windowed + concat images = 'resnext101v03+cat2'

y_scores_a = get_scores(model_a)
y_scores_b = get_scores(model_b)
y_true = get_groundtruths()

# # Delong test to compare AUCs
classes = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
for i in range(len(classes)):
    p_value = delong_roc_test(y_true[:,i], y_scores_a[:,i], y_scores_b[:,i])
    p_value = np.exp(np.log(10)*p_value)
    print(p_value)



#%%
#############################################################################
# McNemar's test to compare models
# - checks if the disagreements between two models match
# - null hypothesis: classifiers have a similar proportion of errors on the test set, 
#   i.e. no statistically significant difference in the disagreements between the two models.
#############################################################################

# Get predictions of two models
model_a = 'resnext101v03_2'
model_b = 'resnext101v03+cat2'
# CNN only without concat/windowing = 'resnext101v03_noRNN_nowindow'
# CNN only with concat images = 'resnext101v03_noRNN_cat'
# CNN only with windowed images = 'resnext101v03_noRNN'
# CNN only with windowed + concat images = 'resnext101v03_noRNN+cat'
# CNN-RNN without concat/windowing = 'resnext101v03_nowindow'
# CNN-RNN with concat images = 'resnext101v03_cat2'
# CNN-RNN with windowed images = 'resnext101v03_2'
# CNN-RNN with windowed + concat images = 'resnext101v03+cat2'

# Get predictions of both models and ground truths. shape (484, 6)
y_preds_a = get_preds(model_a, threshold)
y_preds_b = get_preds(model_b, threshold)
y_true = get_groundtruths()

# For each haemorrahge class: 
i = 0

# For each scan, check if each model made a correct or incorrect prediction
model_a_correct = np.equal(y_preds_a[:,i], y_true[:,i])     # for each scan, returns true if clf is correct
model_b_correct = np.equal(y_preds_b[:,i], y_true[:,i])


# Calculate contingency table of shape (2,2)
contingency = np.zeros(shape=(2,2))
for scan in range(len(model_a_correct)):
    # n_instances that both clfs are correct
    if model_a_correct[scan] == True and model_b_correct[scan] == True:
        contingency[0][0] += 1
    # n_instances that clf 1 is correct but clf 2 is incorrect
    if model_a_correct[scan] == True and model_b_correct[scan] == False:
        contingency[0][1] += 1
    # n_instances that clf 1 is incorrect but clf 2 is correct
    if model_a_correct[scan] == False and model_b_correct[scan] == True:
        contingency[1][0] += 1
    # n_instances that both clfs are incorrect
    if model_a_correct[scan] == False and model_b_correct[scan] == False:
        contingency[1][1] += 1

print(contingency)

# if any cell in the contingency table has a value < 25, need to use a different mcnemar calculation
if np.amin(contingency) < 25:
    mcnemar_pval = mcnemar(contingency, exact=True).pvalue
else:
    mcnemar_pval = mcnemar(contingency, exact=False, correction=True).pvalue
print(mcnemar_pval)








#%%
#############################################################################
# Get the volumes which the model predicted wrong (compare to GT obtained from subject labels)
#############################################################################

# The model's predictions
# if any slice has a haemorrhage, classify the whole volume as having that subtype label
y_pred = pd.read_csv(os.path.join(path_preds, f'CQ500_preds_{model_name}__threshold={threshold}.csv'))
y_pred['CT-idx'] = y_pred['CT-idx_sliceidx'].apply(lambda x: x.split('_')[0])
grouped_y_pred = y_pred.groupby(by=['CT-idx'], as_index=False)
y_pred = grouped_y_pred.agg('max')
y_pred = y_pred.drop(labels=['CT-idx_sliceidx', 'Unnamed: 0'], axis=1)

# The model's confidence indices
# take the highest confidence indices of any slice in the volume
y_score = y_score_orig = pd.read_csv(os.path.join(path_preds, f'CQ500_scores_{model_name}.csv'))
y_score['CT-idx'] = y_score['CT-idx_sliceidx'].apply(lambda x: x.split('_')[0])
y_score = y_score.groupby(by=['CT-idx'], as_index=False).agg('max')
y_score = y_score.drop(labels=['CT-idx_sliceidx', 'Unnamed: 0'], axis=1)

y_true2 = pd.read_csv(os.path.join(path_data, 'CQ500_sub_GT_based_on_subject.csv'))
y_true2 = y_true2.sort_values(by='CT-idx')
# y_true2 is not the same length as y_pred, but we want it to have the same indices as y_pred
y_true2 = y_true2.merge(y_pred['CT-idx'].to_frame(), on='CT-idx', how='right')
y_true2 = y_true2.drop(labels='Unnamed: 0', axis=1)



# Get the ones which the model predicted wrong and with high confidence (i.e. y_score value is v different from y_true2)
y_diff = y_true2.set_index('CT-idx').subtract(y_score.set_index('CT-idx'))
#########################################################
# In y_diff:                                            #
# Large (+) number = true positive but predicted false  #
# Large (-) number = true negative but predicted true   #
#########################################################

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
class_idx = 5   # pick which subtype {0: EDH, 1: IPH, 2: IVH, 3: SAH, 4: SDH, 5: any}
label = label_cols[class_idx]   # label name (spelled fully)
n = 5

y_diff_sorted = y_diff.sort_values(by=label, ascending=False)   # sort from biggest -> smallest
FN_idx = pd.DataFrame(columns=['CT-idx_sliceidx','epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'])
FP_idx = pd.DataFrame(columns=['CT-idx_sliceidx','epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'])
# find the slice within the volume that was wrong
top_n = y_diff_sorted[:n].index
for vol_idx in top_n:
    # get all slices within this vol_idx
    vol = y_score_orig.loc[y_score_orig['CT-idx_sliceidx'].str.contains(vol_idx)]
    FN_idx = FN_idx.append(vol, ignore_index=True)
bottom_n = y_diff_sorted[-n:].index
for vol_idx in bottom_n:
    # get all slices within this vol_idx
    vol = y_score_orig.loc[y_score_orig['CT-idx_sliceidx'].str.contains(vol_idx)]
    FP_idx = FP_idx.append(vol, ignore_index=True)
FN_idx = FN_idx.drop(labels=['CT-idx', 'Unnamed: 0'], axis=1)
FP_idx = FP_idx.drop(labels=['CT-idx', 'Unnamed: 0'], axis=1)

FN_idx.to_csv(os.path.join(path_preds, f'CQ500_scores_{model_name}_falseneg_{label}.csv'))
FP_idx.to_csv(os.path.join(path_preds, f'CQ500_scores_{model_name}_falsepos_{label}.csv'))



#%%
#############################################################################
# Get specific volumes for model to visualise haemorrhages
#############################################################################

idx_list = [137, 139, 159, 173, 121, 106, 12]
idx_list_str = [f'CT-{x}_' for x in idx_list]

# The model's confidence indices
# take the highest confidence indices of any slice in the volume
y_score_orig = pd.read_csv(os.path.join(path_preds, f'CQ500_scores_{model_name}.csv'))

all_vols = pd.DataFrame(columns=['CT-idx_sliceidx','epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'])

for vol_idx in idx_list_str:
    # get all slices within this vol_idx
    vol = y_score_orig.loc[y_score_orig['CT-idx_sliceidx'].str.contains(vol_idx)]
    all_vols = all_vols.append(vol, ignore_index=True)
all_vols = all_vols.drop(labels=['Unnamed: 0'], axis=1)

all_vols.to_csv(os.path.join(path_preds, f'CQ500_scores_{model_name}_sample.csv'))



#%%
#############################################################################
# Ground truth labels are different between slice and subject-based csv files. Find the CTs with mismatches.
#############################################################################

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
slc_GT = pd.read_csv(os.path.join(path_data, f'CQ500_sub_GT_based_on_slices.csv'))
sub_GT = pd.read_csv(os.path.join(path_data, f'CQ500_sub_GT_based_on_subject.csv'))


# merge dfs
merged_GT = slc_GT.merge(sub_GT, on='CT-idx', how='left')

# if label in sub_GT doesn't match slc_GT, indicate '1' in 'mismatch' column
conditions = (
    (merged_GT['epidural_x'] != merged_GT['epidural_y']) |
    (merged_GT['intraparenchymal_x'] != merged_GT['intraparenchymal_y']) |
    (merged_GT['intraventricular_x'] != merged_GT['intraventricular_y']) |
    (merged_GT['subarachnoid_x'] != merged_GT['subarachnoid_y']) |
    (merged_GT['subdural_x'] != merged_GT['subdural_y']) |
    (merged_GT['any_x'] != merged_GT['any_y'])
)
merged_GT['mismatch'] = np.where(conditions, 1, 0)

# delete the rows which are not mismatched
indexes_to_drop = merged_GT[merged_GT['mismatch'] == 0].index
merged_GT.drop(indexes_to_drop, inplace=True)
merged_GT = merged_GT.drop(labels=['mismatch', 'Unnamed: 0'], axis=1)

# tidy up the column names
name_dict = {}
for label in label_cols:
    for str, char in zip(['slc', 'sub'], ['x','y']):
        old_name = f'{label}_{char}'
        new_name = f'{str}_{label}'
        name_dict[old_name] = new_name
merged_GT = merged_GT.rename(name_dict, axis=1)
merged_GT.reset_index(drop=True, inplace=True)

merged_GT.to_csv(os.path.join(path_data, f'CQ500_groundtruth_mismatches.csv'))
