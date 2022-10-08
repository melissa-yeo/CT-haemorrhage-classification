"""
1. Put all CQ500 dataset images into a single folder where images are labelled in the following format:
    CQ500-CT-idx_sliceidx

2. Preprocess test data:
- Create test metadata
- Convert input 2D CT slices to .jpg. Each slice has the format: CT-idx-sliceidx
- Create test.csv.gz containing columns: 'Image'(ID) and 'Label'(placeholder 0.5)
    (.csv file ID should correspond to name of .jpg file)

"""

import os
import shutil
import pandas as pd
import numpy as np
import cv2
import ast
import pydicom
from tqdm import tqdm

RAW_DATA_DIR = '/'
NEW_DATA_DIR = ''

for dirpath, dirnames, files in os.walk(RAW_DATA_DIR):
    print(f'Found directory: {dirpath}')
    dirname = dirpath.split('/')[-1]
    for file_name in files:
        # Copy the .dcm slice and paste into NEW_DATA_DIR
        if file_name.endswith('.dcm'):
            new_file_name = dirname + '_' + file_name
        else:
            new_file_name = file_name
        src = os.path.join(RAW_DATA_DIR, dirname, file_name)
        dst = os.path.join(NEW_DATA_DIR, new_file_name)
        shutil.copy2(src, dst)



CLEAN_DATA_DIR = ''
TEST_DIR = RAW_DATA_DIR
PATHPROC = os.path.join(CLEAN_DATA_DIR, 'proc/')
PATHRAW = os.path.join(CLEAN_DATA_DIR, 'raw/')
PATHPROCBRAIN = os.path.join(CLEAN_DATA_DIR, 'proc_brain/')
PATHPROCCAT = os.path.join(CLEAN_DATA_DIR, 'proc_cat/')
PATHPROCCQ500 = os.path.join(CLEAN_DATA_DIR, 'proc_CQ500/')


#%%
def generate_df(base, files):

    train_di = {}
    train_di['CT-idx_sliceidx'] = []

    for filename in tqdm(files):
        if not filename.endswith('.dcm'):
            continue
        path = os.path.join(base, filename)
        dcm = pydicom.dcmread(path)
        all_keywords = dcm.dir()
        included = ['BitsAllocated', 'BitsStored', 'HighBit', 'ImageOrientationPatient', 'ImagePositionPatient',
                    'Modality', 'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
                    'RescaleIntercept', 'RescaleSlope', 'SOPInstanceUID', 'SamplesPerPixel', 'SeriesInstanceUID',
                    'StudyInstanceUID', 'WindowCenter', 'WindowWidth']

        for name in all_keywords:
            if name not in included:
                continue

            if name not in train_di:
                train_di[name] = []

            if dcm[name].value == '':
                print(name, filename)

            train_di[name].append(dcm[name].value)

        filename = filename[6:-4]
        train_di['CT-idx_sliceidx'].append(filename)

    df = pd.DataFrame(train_di)

    return df


def rescale_image(image, slope, intercept):
    image = image.astype(np.float64) * slope
    image = image.astype(np.int16) + intercept
    return image

  
def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image

  
def apply_window_policy(image):

    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 80, 200) # subdural/ blood
    image3 = apply_window(image, 40, 380) # soft tissue
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)

    return image


def apply_window_policy_brain(image):

    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 40, 80) # brain
    image3 = apply_window(image, 40, 80) # brain
    image1 = (image1 - 0) / 80
    image2 = (image2 - 0) / 80
    image3 = (image3 - 0) / 80
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)

    return image

def convert_dicom_to_jpg(name, path, window='all'):
    try:
        imgnm = (name.split('/')[-1][6:-4])
        dicom = pydicom.dcmread(name)
        image = dicom.pixel_array
        if window == 'all':
            image = rescale_image(image, rescaledict['RescaleSlope'][imgnm], rescaledict['RescaleIntercept'][imgnm])
            image = apply_window_policy(image)
            image -= image.min((0,1))
            image = (255*image).astype(np.uint8)
        if window == 'brain':
            image = rescale_image(image, rescaledict['RescaleSlope'][imgnm], rescaledict['RescaleIntercept'][imgnm])
            image = apply_window_policy_brain(image)
            image -= image.min((0,1))
            image = (255*image).astype(np.uint8)
        if window == 'none':
            # skip conversion to HU step
            image = image.astype(float)                             # convert to float to avoid overflow/underflow losses
            image = (np.maximum(image, 0) / image.max()) * 255.0    # rescale greyscale between 0-255
            image = image.astype(np.uint8)                          # convert to uint8 (unsigned from 0-255)
        cv2.imwrite(os.path.join(path, imgnm)+'.jpg', image)
    except:
        print(f'Failed: {name}')

        
def sort_slices(df):
    # Create a SliceID which has format: PatientID_SeriesInstanceUID_StudyInstanceUID
    df['SliceID'] = df[['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID']].apply(lambda x: '{}__{}__{}'.format(*x.tolist()), 1)
    # Each ImagePositionPatient is originally a list of 3 float numbers [a,b,c]
    # Change this to separate columns of: ImagePos1, ImagePos2, ImagePos3
    poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
    df[poscols] = pd.DataFrame(df['ImagePositionPatient'].apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
    # Sort by SliceID and ImagePos, then extract just the created SliceID, PatientID and SOPInstanceUID
    # note for CQ500 dataset, don't need the SOPInstanceUID column, bc we're using CT-idx_sliceidx to identify images
    df = df.sort_values(['SliceID'] + poscols)[['CT-idx_sliceidx', 'SliceID', 'SOPInstanceUID'] + poscols].reset_index(drop=True)
    # Group by SliceID (now in seq), then index it by the cumulative count +1 (i.e. first slice = 1, second slice = 2)
    df['seq'] = (df.groupby(['SliceID']).cumcount() + 1)
    # Entire dataframe now only has these 4 columns
    dfkeepcols = ['CT-idx_sliceidx', 'SliceID', 'SOPInstanceUID', 'seq']
    df = df[dfkeepcols]
    # Rename the columns
    df.columns = ['CT-idx_sliceidx', 'SliceID', 'Image', 'seq']
    return df

  
def combine_slices(df, pathin, pathout, imageid):
    slice_metadata = df.loc[df['CT-idx_sliceidx'] == imageid]
    seq = float(slice_metadata['seq'].values)
    studyid = slice_metadata['SliceID'].values
    sliceid = slice_metadata['CT-idx_sliceidx'].values[0]

    # match to slice with the same 'SliceID' and seq above, where possible; then get imageID
    try:
        sliceid_up = df.loc[(df['SliceID'].values == studyid) & (df['seq'].values == seq+1)]['CT-idx_sliceidx'].values[0]
    except:
        sliceid_up = sliceid
    try:
        sliceid_down = df.loc[(df['SliceID'].values == studyid) & (df['seq'].values == seq-1)]['CT-idx_sliceidx'].values[0]
    except:
        sliceid_down = sliceid

    # print(sliceid, sliceid_up, sliceid_down)

    slice = cv2.imread(os.path.join(pathin, sliceid + '.jpg'))
    slice_up = cv2.imread(os.path.join(pathin, sliceid_up + '.jpg'))
    slice_down = cv2.imread(os.path.join(pathin, sliceid_down + '.jpg'))
    # print(sliceid, sliceid_up, sliceid_down)
    slice = np.transpose(slice, (2,0,1))
    slice_up = np.transpose(slice_up, (2,0,1))
    slice_down = np.transpose(slice_down, (2,0,1))

    slice_cat = np.stack((slice_up[0], slice[1], slice_down[2]), axis=0)
    slice_cat = np.transpose(slice_cat, (1,2,0))

    cv2.imwrite(os.path.join(pathout, imageid) + '.jpg', slice_cat)
    return


  
#%% Generate CQ500_test_metadata.csv

print('Create test meta files')
test_files = os.listdir(TEST_DIR)
test_df = generate_df(TEST_DIR, test_files)
test_df.to_csv(os.path.join(CLEAN_DATA_DIR, 'CQ500_test_metadata.csv'))



#%% Preprocessing pipeline 1: windowed images (3-channel: brain, subdural, soft tissue)

print('Load test meta files')
tstmdf = pd.read_csv(os.path.join(CLEAN_DATA_DIR, 'CQ500_test_metadata.csv'))
print('Shape {} {}'.format(*tstmdf.shape))

mdf = tstmdf
rescaledict = mdf.set_index('CT-idx_sliceidx')[['RescaleSlope', 'RescaleIntercept']].to_dict()

print('Create raw images (.jpg)')
for dirpath, dirnames, files in os.walk(RAW_DATA_DIR):
    print('Currently in {}'.format(dirpath))
    for idx, file_name in enumerate(tqdm(files)):
        if file_name.endswith('.dcm'):
            convert_dicom_to_jpg(os.path.join(dirpath, file_name), PATHRAW, window='none')

print('Create brain-brain-brain windowed images (.jpg)')
for dirpath, dirnames, files in os.walk(RAW_DATA_DIR):
    print('Currently in {}'.format(dirpath))
    for idx, file_name in enumerate(tqdm(files)):
        if file_name.endswith('.dcm'):
            convert_dicom_to_jpg(os.path.join(dirpath, file_name), PATHPROCBRAIN, window='brain')

print('Create brain-subdural-bone windowed images (.jpg)')
for dirpath, dirnames, files in os.walk(RAW_DATA_DIR):
    print('Currently in {}'.format(dirpath))
    for idx, file_name in enumerate(tqdm(files)):
        if file_name.endswith('.dcm'):
            convert_dicom_to_jpg(os.path.join(dirpath, file_name), PATHPROC, window='all')



#%% Preprocessing pipeline 2: images with concatenated slices (3-channel: slice above, slice, slice below)

tstmdf = pd.read_csv(os.path.join(CLEAN_DATA_DIR, 'CQ500_test_metadata.csv'))
mdf = tstmdf

mdf = sort_slices(mdf)

# Create concatenated images (3-channel: slice above, slice, slice below)
print('Create slice concatenated images')
for dirpath, dirnames, files in os.walk(PATHPROCBRAIN):
    print('Currently in {}'.format(dirpath))
    for idx, file_name in enumerate(tqdm(files)):
        if file_name.endswith('.jpg') and file_name.startswith('CT'):
            imgnm = (file_name.split('/')[-1]).replace('.jpg', '')
            combine_slices(mdf, PATHPROCBRAIN, PATHPROCCAT, imgnm)
print('Slice concatenated images created!')


# #%% Testing if concatenate is working right by reverse concatenating

# img = cv2.imread(os.path.join(PATHPROCCAT, 'CT-0_CT000008' + '.jpg'))
# img = np.transpose(img, (2,0,1))
# slice_cat1 = np.stack((img[0], img[0], img[0]), axis=0)
# slice_cat2 = np.stack((img[1], img[1], img[1]), axis=0)
# slice_cat3 = np.stack((img[2], img[2], img[2]), axis=0)
# slice_cat1 = np.transpose(slice_cat1, (1,2,0))
# slice_cat2 = np.transpose(slice_cat2, (1,2,0))
# slice_cat3 = np.transpose(slice_cat3, (1,2,0))
# cv2.imwrite(os.path.join(TOY_OUT, 'slice_cat1.jpg'), slice_cat1)
# cv2.imwrite(os.path.join(TOY_OUT, 'slice_cat2.jpg'), slice_cat2)
# cv2.imwrite(os.path.join(TOY_OUT, 'slice_cat3.jpg'), slice_cat3)

# import matplotlib.pyplot as plt
# img = cv2.imread(os.path.join(PATHPROCCAT, 'CT-0_CT000008' + '.jpg'))
# plt.hist(img[0])
# plt.show()
# img = cv2.imread(os.path.join(PATHPROCCQ500, 'CT-0_CT000008' + '.jpg'))
# plt.hist(img[0])
# plt.show()


#%% Generate CQ500_test.csv.gz

tstmdf = pd.read_csv(os.path.join(CLEAN_DATA_DIR, 'CQ500_test_metadata.csv'))
print(tstmdf.shape)

data = {'Image': tstmdf['CT-idx_sliceidx'],
        'Label': 0.5}
tstdf = pd.DataFrame(data)

tstdf.to_csv(os.path.join(CLEAN_DATA_DIR, 'CQ500_test.csv.gz'), index=False, compression='gzip')


#%% Testing how dataloader will read in images

# idx=2
# test = pd.read_csv(os.path.join(CLEAN_DATA_DIR, 'CQ500_test.csv.gz'))
#
# img_name = os.path.join(PATHPROC, test.loc[idx, 'Image'] + '.jpg')
# img = cv2.imread(img_name)
# print(img_name)


#%% Count number of slices per study (range)

# tstmdf = pd.read_csv(os.path.join(CLEAN_DATA_DIR, 'CQ500_test_metadata.csv'))
# tstmdf['CT-idx'] = tstmdf['CT-idx_sliceidx'].apply(lambda x: x.split('_')[0])
# counts = tstmdf.groupby(['CT-idx']).count()
# print(counts['Unnamed: 0'].min())
# print(counts['Unnamed: 0'].max())
# print(counts['Unnamed: 0'].median())
