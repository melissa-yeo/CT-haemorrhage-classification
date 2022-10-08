import os
import datetime
import pandas as pd
import numpy as np
import cv2
import ast
import pydicom
from tqdm import tqdm
import sys
sys.path.insert(0, 'scripts')


RAW_DATA_DIR = ''
CLEAN_DATA_DIR = ''
TRAIN_DIR = os.path.join(RAW_DATA_DIR, 'train/')
TEST_DIR = os.path.join(RAW_DATA_DIR, 'test/')
PATHPROC = os.path.join(CLEAN_DATA_DIR, 'proc/')
PATHPROCBRAIN = os.path.join(CLEAN_DATA_DIR, 'procbrain/')
PATHPROCCAT = os.path.join(CLEAN_DATA_DIR, 'proccat/')


#%%
def get_dicom_value(x, cast=int):
    if type(x) in [pydicom.multival.MultiValue, tuple]:
        return cast(x[0])
    else:
        return cast(x)


def cast(value):
    if type(value) is pydicom.valuerep.MultiValue:
        return tuple(value)
    return value


def get_dicom_raw(dicom):
    return {attr:cast(getattr(dicom,attr)) for attr in dir(dicom) if attr[0].isupper() and attr not in ['PixelData']}


def rescale_image(image, slope, intercept):
    return image * slope + intercept


def get_dicom_meta(dicom):
    return {
        'PatientID': dicom.PatientID, # can be grouped (20-548)
        'StudyInstanceUID': dicom.StudyInstanceUID, # can be grouped (20-60)
        'SeriesInstanceUID': dicom.SeriesInstanceUID, # can be grouped (20-60)
        'WindowWidth': get_dicom_value(dicom.WindowWidth),
        'WindowCenter': get_dicom_value(dicom.WindowCenter),
        'RescaleIntercept': float(dicom.RescaleIntercept),
        'RescaleSlope': float(dicom.RescaleSlope), # all same (1.0)
    }


def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image


def apply_window_type(image, window='brain'):
    if window == 'brain':
        image = apply_window(image, 40, 80)
        image = (image - 0) / 80
    if window == 'subdural':
        image = apply_window(image, 80, 200)
        image = (image - (-20)) / 200
    if window == 'bone':
        image = apply_window(image, 40, 380)
        image = (image - (-150)) / 380
    return image
  
  
def apply_window_policy(image, win1, win2, win3):
    image1 = apply_window_type(image, win1)
    image2 = apply_window_type(image, win2)
    image3 = apply_window_type(image, win3)
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)
    return image

  
def convert_dicom_to_jpg(name, path, win1='brain', win2='subdural', win3='bone'):
    try:
        # data = f.read(name)
        # dirtype = 'train' if 'train' in name else 'test'
        imgnm = (name.split('/')[-1]).replace('.dcm', '')
        # dicom = pydicom.dcmread(DicomBytesIO(data))
        dicom = pydicom.dcmread(name)
        image = dicom.pixel_array
        image = rescale_image(image, rescaledict['RescaleSlope'][imgnm], rescaledict['RescaleIntercept'][imgnm])
        image = apply_window_policy(image, win1, win2, win3)
        image -= image.min((0,1))
        image = (255*image).astype(np.uint8)
        cv2.imwrite(os.path.join(path, imgnm)+'.jpg', image)
    except:
        logger.info(name)
        
        
def generate_df(base, files):
    train_di = {}

    for filename in tqdm(files):
        path = os.path.join( base ,  filename)
        dcm = pydicom.dcmread(path)
        all_keywords = dcm.dir()
        ignored = ['Rows', 'Columns', 'PixelData']

        for name in all_keywords:
            if name in ignored:
                continue

            if name not in train_di:
                train_di[name] = []

            train_di[name].append(dcm[name].value)

    df = pd.DataFrame(train_di)
    
    return df

  

def sort_slices(df):
    # Create a SliceID which has format: PatientID_SeriesInstanceUID_StudyInstanceUID
    df['SliceID'] = df[['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID']].apply(lambda x: '{}__{}__{}'.format(*x.tolist()), 1)
    # Each ImagePositionPatient is originally a list of 3 float numbers [a,b,c]
    # Change this to separate columns of: ImagePos1, ImagePos2, ImagePos3
    poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
    df[poscols] = pd.DataFrame(df['ImagePositionPatient'].apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
    # Sort by SliceID and ImagePos, then extract just the created SliceID, PatientID and SOPInstanceUID (same as .png ID)
    df = df.sort_values(['SliceID'] + poscols)[['PatientID', 'SliceID', 'SOPInstanceUID'] + poscols].reset_index(drop=True)
    # Group by SliceID (now in seq), then index it by the cumulative count +1 (i.e. first slice = 1, second slice = 2)
    df['seq'] = (df.groupby(['SliceID']).cumcount() + 1)
    # Entire dataframe now only has these 4 columns
    dfkeepcols = ['PatientID', 'SliceID', 'SOPInstanceUID', 'seq']
    df = df[dfkeepcols]
    # Rename the columns
    df.columns = ['PatientID', 'SliceID', 'Image', 'seq']
    return df
  

def combine_slices(df, pathin, pathout, imageid):
    try:
        slice_metadata = df.loc[df['Image'] == imageid]
        seq = float(slice_metadata['seq'].values)
        studyid = slice_metadata['SliceID'].values
        sliceid = slice_metadata['Image'].values[0]

        # match to slice with the same 'SliceID' and seq above, where possible; then get imageID
        try:
            sliceid_up = df.loc[(df['SliceID'].values == studyid) & (df['seq'].values == seq+1)]['Image'].values[0]
        except:
            sliceid_up = sliceid
        try:
            sliceid_down = df.loc[(df['SliceID'].values == studyid) & (df['seq'].values == seq-1)]['Image'].values[0]
        except:
            sliceid_down = sliceid

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
    except:
        print(f'Failed: {imageid}')
    return


  
#%%
# Create metadata files if not yet created
if not os.path.exists(os.path.join(CLEAN_DATA_DIR, 'test_metadata.csv')):
    logger.info('Create test meta files')
    test_files = os.listdir(TEST_DIR)
    test_df = generate_df(TEST_DIR, test_files)
    test_df.to_csv(os.path.join(CLEAN_DATA_DIR, 'test_metadata.csv'))
if not os.path.exists(os.path.join(CLEAN_DATA_DIR, 'train_metadata.csv')):
    logger.info('Create train meta files')
    train_files = os.listdir(TRAIN_DIR)
    train_df = generate_df(TRAIN_DIR, train_files)
    train_df.to_csv(os.path.join(CLEAN_DATA_DIR, 'train_metadata.csv'))


# Load metadata files
trnmdf = pd.read_csv(os.path.join(CLEAN_DATA_DIR, 'train_metadata.csv'))
tstmdf = pd.read_csv(os.path.join(CLEAN_DATA_DIR, 'test_metadata.csv'))

mdf = pd.concat([trnmdf, tstmdf], 0)
rescaledict = mdf.set_index('SOPInstanceUID')[['RescaleSlope', 'RescaleIntercept']].to_dict()


# Create windowed images (3-channel: brain, subdural, bone)
for dirpath, dirnames, files in os.walk(RAW_DATA_DIR):
    logger.info('Currently in {}'.format(dirpath))
    for idx, file_name in enumerate(tqdm(files)):
        if file_name.endswith('.dcm'):
            convert_dicom_to_jpg(os.path.join(dirpath, file_name), PATHPROC)

# Create windowed images (3-channel: brain, brain, brain)
for dirpath, dirnames, files in os.walk(RAW_DATA_DIR):
    logger.info('Currently in {}'.format(dirpath))
    for idx, file_name in enumerate(tqdm(files)):
        if file_name.endswith('.dcm'):
            convert_dicom_to_jpg(os.path.join(dirpath, file_name), PATHPROCBRAIN, win1='brain', win2='brain', win3='brain')


trnmdf = pd.read_csv(os.path.join(CLEAN_DATA_DIR, 'train_metadata.csv'))
tstmdf = pd.read_csv(os.path.join(CLEAN_DATA_DIR, 'test_metadata.csv'))
mdfs = [trnmdf, tstmdf]
mdf = pd.concat(mdfs)
mdf = sort_slices(mdf)

# Create concatenated images (3-channel: slice above, slice, slice below)
logger.info('Create slice concatenated images')
for dirpath, dirnames, files in os.walk(PATHPROCBRAIN):
    logger.info('Currently in {}'.format(dirpath))
    for idx, file_name in enumerate(tqdm(files)):
        if file_name.endswith('.jpg') and file_name.startswith('ID'):
            imgnm = (file_name.split('/')[-1]).replace('.jpg', '')
            if not os.path.exists(os.path.join(PATHPROCCAT, imgnm) + '.jpg'):
                combine_slices(mdf, PATHPROCBRAIN, PATHPROCCAT, imgnm)


                
#%% Count number of slices per study (range)

trnmdf = pd.read_csv(os.path.join(CLEAN_DATA_DIR, 'train_metadata.csv'))

# Create a SliceID which has format: PatientID_SeriesInstanceUID_StudyInstanceUID
trnmdf['SliceID'] = trnmdf[['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID']].apply(lambda x: '{}__{}__{}'.format(*x.tolist()), 1)

# Each ImagePositionPatient is originally a list of 3 float numbers [a,b,c]
# Change this to separate columns of: ImagePos1, ImagePos2, ImagePos3
poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())

# Sort by SliceID and ImagePos, then extract just the created SliceID, PatientID and SOPInstanceUID (same as .png ID)
trnmdf = trnmdf.sort_values(['SliceID']+poscols)\
                [['PatientID', 'SliceID', 'SOPInstanceUID']+poscols].reset_index(drop=True)

counts = trnmdf.groupby(['PatientID']).count()
print(counts['SliceID'].min())
print(counts['SliceID'].max())
print(counts['SliceID'].median())

difference = trnmdf['ImagePos3'].astype(int).diff()
slicethicknesses = difference.value_counts()


#%% Get subject labels from slice labels

CLEAN_DATA_DIR = ''
trndf = pd.read_csv(os.path.join(CLEAN_DATA_DIR, 'train_analysis.csv'))
grouped_trndf = trndf.groupby(by='SeriesInstanceUID', as_index=False)
trndf = grouped_trndf.agg('max')
trndf = trndf.drop(labels=['SOPInstanceUID'], axis=1)
trndf.to_csv(os.path.join(CLEAN_DATA_DIR, f'train_sub_analysis.csv'))

